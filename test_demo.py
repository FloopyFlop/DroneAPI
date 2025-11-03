# test_demo.py
import asyncio
import json
import numpy as np
from typing import List, Tuple

from .magpie import DroneAPI
from .interpolation import Cubic, TrapezoidalVelocity, Linear, Bezier, BarrelTowards, L1Guidance


# ---------- helpers ----------

def _with_yaw(points, closed: bool = False):
    out = []
    total = len(points)
    for i, (x, y, z) in enumerate(points):
        if i + 1 < total:
            nx, ny, nz = points[i + 1]
        elif closed and total > 1:
            nx, ny, nz = points[0]
        else:
            nx, ny, nz = x, y, z
        dx, dz = nx - x, nz - z
        yaw = 0.0 if (dx == 0 and dz == 0) else float(np.rad2deg(np.arctan2(dx, dz)))
        out.append((float(x), float(y), float(z), yaw))
    return out


def build_square(side: float = 4.0, alt: float = 3.0):
    h = side / 2.0
    pts = [(-h, alt, -h), (h, alt, -h), (h, alt, h), (-h, alt, h), (-h, alt, -h)]
    return _with_yaw(pts, closed=True)


def build_circle(radius: float = 2.5, alt: float = 3.0, samples: int = 36):
    thetas = np.linspace(0, 2 * np.pi, samples + 1, endpoint=True)
    pts = [(radius * np.sin(t), alt, radius * np.cos(t)) for t in thetas]
    return _with_yaw(pts, closed=True)


def build_dense_curve(length: float = 10.0, alt: float = 3.0, samples: int = 120, amp: float = 1.8):
    xs = np.linspace(-length / 2.0, length / 2.0, samples)
    zs = amp * np.sin(xs * np.pi / (length / 1.2)) * np.cos(xs * np.pi / (length / 2.0))
    pts = list(zip(xs, np.full_like(xs, alt), zs))
    return _with_yaw(pts, closed=False)


def _rot_xyz(p, rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return (R @ p.reshape(3, 1)).ravel()


def build_star3d_tilted(size: float = 3.0, alt: float = 3.0, tilt=(25.0, 15.0, 35.0)):
    angles_outer = np.deg2rad(np.linspace(-90, 270, 5, endpoint=False))
    angles_inner = angles_outer + np.deg2rad(36)
    r_outer = 1.0
    r_inner = 0.4
    verts = []
    for i in range(5):
        verts.append((r_outer * np.cos(angles_outer[i]), r_outer * np.sin(angles_outer[i])))
        verts.append((r_inner * np.cos(angles_inner[i]), r_inner * np.sin(angles_inner[i])))
    verts.append(verts[0])
    pts = []
    for x0, z0 in verts:
        p = np.array([size * x0, 0.0, size * z0])
        pr = _rot_xyz(p, *tilt)
        pts.append((pr[0], alt + pr[1], pr[2]))
    return _with_yaw(pts, closed=True)


async def _run_waypoints_with_timeout(
    drone: DroneAPI,
    waypoints: List[Tuple[float, float, float, float]],
    interp_cls,
    segment_timeout: float = 20.0,
    threshold: float = 0.20,
):
    """
    Enqueue and fly each waypoint as its own 'segment' with a timeout.
    If a segment doesn't finish in time, raise TimeoutError to fail the test run.
    """
    for (x, y, z, yaw) in waypoints:
        drone.enqueue_waypoint(x, y, z, yaw=yaw, interpolation=interp_cls, threshold=threshold)
        try:
            await asyncio.wait_for(drone.follow_waypoints(), timeout=segment_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Segment to ({x:.2f},{y:.2f},{z:.2f}) timed out for {interp_cls.__name__}")


# ---------- ROS2 publishing helpers ----------

class ConsoleTelemetryPublisher:
    """
    Optional stand-in publisher used when STREAM_TO_CONSOLE is enabled for debugging.
    """

    def __init__(self, topic: str, node_name: str, qos_depth: int):
        self.topic = topic
        self.node_name = node_name
        self.qos_depth = qos_depth
        print(f"[{self.topic}] ConsoleTelemetryPublisher started (node={self.node_name}, qos={self.qos_depth})")

    def publish(self, payload: str) -> None:
        snapshot = json.loads(payload)
        print(f"\n[{self.topic}] snapshot:")
        print(json.dumps(snapshot, indent=2, sort_keys=True))

    def close(self) -> None:
        print(f"[{self.topic}] ConsoleTelemetryPublisher closed")


def console_factory(topic: str, node_name: str, qos_depth: int) -> ConsoleTelemetryPublisher:
    return ConsoleTelemetryPublisher(topic, node_name, qos_depth)


# ---------- test configuration ----------

STREAM_RAW = True
STREAM_TO_CONSOLE = False  # flip to True when you want human-readable dumps without ROS2
STREAM_TOPIC = "/magpie/test_demo/raw"
STREAM_NODE = "magpie_test_demo"
STREAM_QOS_DEPTH = 10


# ---------- test runner ----------

async def _async_main():
    initial_altitude = 3.0

    scenarios = [
        ("square", build_square(side=4.0, alt=initial_altitude)),
        ("circle", build_circle(radius=2.5, alt=initial_altitude, samples=36)),
        ("dense_curve", build_dense_curve(length=10.0, alt=initial_altitude, samples=120, amp=1.8)),
        ("star3d_tilted", build_star3d_tilted(size=3.0, alt=initial_altitude, tilt=(25, 15, 35))),
    ]

    interps = [
        ("L1Guidance", L1Guidance),
        ("Linear", Linear),
        ("Bezier", Bezier),
        ("BarrelTowards", BarrelTowards),
        ("TrapezoidalVelocity", TrapezoidalVelocity),
        ("Cubic", Cubic),
    ]

    # Create a single drone instance for all tests to avoid connection reuse issues
    ros_kwargs = {}
    if STREAM_RAW:
        ros_kwargs = {
            "ros2_enabled": True,
            "ros2_topic": STREAM_TOPIC,
            "ros2_node_name": STREAM_NODE,
            "ros2_qos_depth": STREAM_QOS_DEPTH,
        }
        if STREAM_TO_CONSOLE:
            ros_kwargs["ros2_publisher_factory"] = console_factory

    drone = DroneAPI(
        system_address="udp://:14540",
        default_interpolation=Linear,
        control_rate_hz=20.0,  # Increased from 10 to handle faster telemetry
        max_speed_m_s=1.5,
        use_velocity_command=True,
        log_enabled=True,
        log_path=f"/media/sf_Testing/flight_log.png",
        telemetry_stream_mode="essential",
        telemetry_rate_hz=20.0,  # Set explicit rate to speed up telemetry
        telemetry_publish_interval=0.05,  # Publish 20 times per second (fast updates for dashboard)
        **ros_kwargs,
    )

    try:
        # Start mission once at the beginning
        await drone.begin_mission(initial_altitude=initial_altitude, yaw=0.0)

        for name_interp, interp_cls in interps:
            for name_scn, waypoints in scenarios:
                print(f"\n=== Starting demo: interp={name_interp} scenario={name_scn} ===")

                # Update log path and interpolation for this test
                drone.log_path = f"/media/sf_Testing/{name_interp}_{name_scn}.png"
                drone.default_interpolation_cls = interp_cls

                try:
                    await _run_waypoints_with_timeout(
                        drone, waypoints, interp_cls, segment_timeout=25.0, threshold=0.20
                    )
                except TimeoutError as e:
                    print(f"[FAIL] {e}")

                # Return to home position between tests
                try:
                    await _run_waypoints_with_timeout(
                        drone,
                        [(0.0, initial_altitude, 0.0, 0.0)],
                        Cubic,
                        segment_timeout=25.0,
                        threshold=0.15,
                    )
                except TimeoutError as e:
                    print(f"[FAIL] (return to home) {e}")

                # Save the plot for this test
                if drone.log_enabled and len(drone.telemetry_log) >= 2:
                    drone.save_flight_plot(drone.log_path)
                    # Clear logs for next test but keep logging enabled
                    drone.telemetry_log.clear()
                    drone.goal_history.clear()

                # Small pause between tests
                await asyncio.sleep(1.0)

        # Final return to ground
        try:
            await _run_waypoints_with_timeout(
                drone,
                [(0.0, 0.0, 0.0, 0.0)],
                Cubic,
                segment_timeout=25.0,
                threshold=0.15,
            )
        except TimeoutError as e:
            print(f"[FAIL] (final landing) {e}")

        await drone.end_mission()
    finally:
        await drone.shutdown()


def main() -> None:
    """
    ros2 entry points expect a synchronous callable.
    Wrap the async mission execution in asyncio.run so ros2 run modular1 service works.
    """
    asyncio.run(_async_main())


if __name__ == "__main__":
    print("Starting test_demo.py...")
    main()
