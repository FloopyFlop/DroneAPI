# test_demo.py
import asyncio
import numpy as np

from .magpie import DroneAPI
from .interpolation import Cubic, Precision, Linear, MinJerk


def _with_yaw(points, closed=False):
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
        out.append((x, y, z, yaw))
    return out


def build_square(side=4.0, alt=3.0):
    h = side / 2.0
    pts = [(-h, alt, -h), (h, alt, -h), (h, alt, h), (-h, alt, h), (-h, alt, -h)]
    return _with_yaw(pts, closed=True)


async def _async_main():
    initial_altitude = 3.0

    drone = DroneAPI(
        system_address="udp://:14540",
        default_interpolation=Precision,   # global default (GOTO)
        control_rate_hz=10.0,
        max_speed_m_s=1.5,
        use_velocity_command=True,
    )

    await drone.begin_mission(initial_altitude=initial_altitude, yaw=0.0)

    # Enqueue a loop with different interpolations per leg
    waypoints = build_square(side=4.0, alt=initial_altitude)
    interps = [Linear, Cubic, MinJerk, Precision, Cubic]

    for (x, y, z, yaw), interp in zip(waypoints, interps):
        drone.enqueue_waypoint(x, y, z, yaw=yaw, interpolation=interp, threshold=0.20)

    await drone.follow_waypoints()

    # Return home
    drone.enqueue_waypoint(0.0, initial_altitude, 0.0, yaw=0.0, interpolation=Cubic, threshold=0.15)
    drone.enqueue_waypoint(0.0, 0.0, 0.0, yaw=0.0, interpolation=MinJerk, threshold=0.15)
    await drone.follow_waypoints()

    await drone.end_mission()
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

