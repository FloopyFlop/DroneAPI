# test_demo.py
import asyncio
import numpy as np

from .magpie import DroneAPI
from .interpolation import (
    Cubic, Linear,
    TrapezoidalVelocity, PDVelocity, Bezier, LookAheadBlend, SineEase
)

# ---------- helpers ----------

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
        out.append((float(x), float(y), float(z), yaw))
    return out


def build_square(side=4.0, alt=3.0):
    h = side / 2.0
    pts = [(-h, alt, -h), (h, alt, -h), (h, alt, h), (-h, alt, h), (-h, alt, -h)]
    return _with_yaw(pts, closed=True)


def build_circle(radius=2.5, alt=3.0, samples=24):
    thetas = np.linspace(0, 2*np.pi, samples+1, endpoint=True)
    pts = [(radius*np.sin(t), alt, radius*np.cos(t)) for t in thetas]
    return _with_yaw(pts, closed=True)


def build_dense_curve(length=8.0, alt=3.0, samples=80, amp=1.5):
    xs = np.linspace(-length/2.0, length/2.0, samples)
    zs = amp * np.sin(xs * np.pi / (length/1.2)) * np.cos(xs * np.pi / (length/2.0))
    pts = list(zip(xs, np.full_like(xs, alt), zs))
    return _with_yaw(pts, closed=False)


def _rot_xyz(p, rx_deg=0.0, ry_deg=0.0, rz_deg=0.0):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    R = Rz @ Ry @ Rx
    return (R @ p.reshape(3,1)).ravel()


def build_star3d_tilted(size=3.0, alt=3.0, tilt=(25.0, 15.0, 30.0)):
    """
    Build a 5-point star in XZ plane, then tilt in 3D and offset altitude.
    """
    # 5-point star vertices (unit scale)
    angles_outer = np.deg2rad(np.linspace(-90, 270, 5, endpoint=False))
    angles_inner = angles_outer + np.deg2rad(36)  # 360/10
    r_outer = 1.0
    r_inner = 0.4
    verts = []
    for i in range(5):
        verts.append((r_outer*np.cos(angles_outer[i]), r_outer*np.sin(angles_outer[i])))
        verts.append((r_inner*np.cos(angles_inner[i]), r_inner*np.sin(angles_inner[i])))
    verts.append(verts[0])  # close

    # Scale and map to (x,z); y = 0 then tilt
    pts = []
    for x0, z0 in verts:
        p = np.array([size*x0, 0.0, size*z0])
        pr = _rot_xyz(p, *tilt)
        pts.append((pr[0], alt + pr[1], pr[2]))

    return _with_yaw(pts, closed=True)


# ---------- test runner ----------

async def _async_main():
    initial_altitude = 3.0

    # Scenarios to test
    scenarios = [
        ("square", build_square(side=4.0, alt=initial_altitude)),
        ("circle", build_circle(radius=2.5, alt=initial_altitude, samples=36)),
        ("dense_curve", build_dense_curve(length=10.0, alt=initial_altitude, samples=120, amp=1.8)),
        ("star3d_tilted", build_star3d_tilted(size=3.0, alt=initial_altitude, tilt=(25, 15, 35))),
    ]

    # Interpolations to test (Precision/MinJerk removed)
    interps = [
        ("Cubic", Cubic),
        ("Linear", Linear),
        ("TrapezoidalVelocity", TrapezoidalVelocity),
        ("PDVelocity", PDVelocity),
        ("Bezier", Bezier),
        ("LookAheadBlend", LookAheadBlend),
        ("SineEase", SineEase),
    ]

    # Run each interpolation on each scenario in its own mission (fresh logs)
    for name_interp, interp_cls in interps:
        for name_scn, waypoints in scenarios:
            print(f"\n=== Starting demo: interp={name_interp} scenario={name_scn} ===")

            drone = DroneAPI(
                system_address="udp://:14540",
                default_interpolation=interp_cls,   # global default for this run
                control_rate_hz=10.0,
                max_speed_m_s=1.5,
                use_velocity_command=True,
                log_enabled=True,
                log_path=f"/media/sf_Testing/{name_interp}_{name_scn}.png",
            )

            await drone.begin_mission(initial_altitude=initial_altitude, yaw=0.0)

            for (x, y, z, yaw) in waypoints:
                drone.enqueue_waypoint(x, y, z, yaw=yaw, interpolation=interp_cls, threshold=0.20)

            await drone.follow_waypoints()

            # Return home (use Cubic both legs since MinJerk/Precision removed)
            drone.enqueue_waypoint(0.0, initial_altitude, 0.0, yaw=0.0, interpolation=Cubic, threshold=0.15)
            drone.enqueue_waypoint(0.0, 0.0, 0.0, yaw=0.0, interpolation=Cubic, threshold=0.15)
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
