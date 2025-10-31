# DroneAPI Guide

`DroneAPI` (defined in `magpie.py`) is the high-level mission wrapper that MAGPIE uses to fly MAVSDK-compatible vehicles with queued waypoints, pluggable interpolation strategies, and optional telemetry logging. This guide walks through the lifecycle, explains the coordinate frame, and demonstrates idiomatic usage pulled directly from `magpie.py` and `test_demo.py`.

## Repository Tour
- `magpie.py` – exposes `DroneAPI`, the `MagpieEnv` helper, and logging utilities.
- `interpolation.py` – interpolation strategies (`Linear`, `Cubic`, `Bezier`, `BarrelTowards`, `TrapezoidalVelocity`, `L1Guidance`, …).
- `test_demo.py` – end-to-end async demos that exercise multiple interpolations and shapes.
- `SetupGuide.md` / `FromScratchSetup.md` – environment preparation and MAVSDK notes.

## Coordinate Frame & Assumptions
`MagpieEnv` keeps motion in a local XYZ frame:
- x → East, y → Up, z → North.
- Yaw is expressed in degrees around the up axis.
- `begin_mission()` samples the current position to build the offset, so all waypoints and telemetry are relative to the takeoff location.

## Mission Lifecycle At A Glance
1. Instantiate `DroneAPI`, choosing defaults (connection URL, interpolation, max speed, logging).
2. `await begin_mission(initial_altitude, yaw)` – connect, verify health, arm, take off, optionally start logging.
3. Populate the waypoint queue with `enqueue_waypoint(...)`.
4. `await follow_waypoints()` to fly the queue (repeatable; queue can be refilled on the fly).
5. `await end_mission()` to land and save logs.
6. `await shutdown()` for hard cleanup (ensures offboard is stopped and motors are disarmed).

All motion commands require `begin_mission()` first so the local frame and offboard session are ready.

## Quickstart Mission

```python
import asyncio

from DroneAPI.magpie import DroneAPI
from DroneAPI.interpolation import L1Guidance, Cubic


async def main():
    drone = DroneAPI(
        system_address="udp://:14540",
        default_interpolation=L1Guidance,   # pass-through profile
        max_speed_m_s=1.8,
        log_enabled=True,
        log_path="logs/figure8.png",
        segment_timeout_sec=25.0,
    )

    await drone.begin_mission(initial_altitude=3.0, yaw=0.0)

    # Enqueue a figure-eight pattern; override the third segment to stop at the goal
    drone.enqueue_waypoint(2.0, 3.0, 0.0, yaw=90.0)
    drone.enqueue_waypoint(-2.0, 3.0, 0.0, yaw=270.0)
    drone.enqueue_waypoint(0.0, 3.0, 2.0, yaw=0.0, interpolation=Cubic, threshold=0.12)

    await drone.follow_waypoints()

    # Return home and land with a stop-style interpolation
    drone.enqueue_waypoint(0.0, 3.0, 0.0, yaw=0.0, interpolation=Cubic)
    drone.enqueue_waypoint(0.0, 0.0, 0.0, yaw=0.0, interpolation=Cubic, threshold=0.1)
    await drone.follow_waypoints()

    await drone.end_mission()
    await drone.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

### What the Example Highlights
- `system_address` is the MAVSDK connection string; adjust for your transport.
- `default_interpolation` defines how pass-through segments behave; override per waypoint when needed.
- Each waypoint uses local-frame coordinates (meters relative to takeoff).
- The per-waypoint `threshold` controls the completion radius (defaults to `0.15` m).
- `segment_timeout_sec` guards against a stuck interpolation by forcing completion after the watchdog expires.

## Managing the Waypoint Queue
- `enqueue_waypoint(x, y, z, yaw=0.0, interpolation=None, threshold=None)`  
  Adds a segment to the queue. Provide an interpolation class from `interpolation.py` to override the default, and optionally tighten/loosen the completion tolerance.
- `await follow_waypoints(wait_for_new=False, idle_sleep=0.25)`  
  Consumes the queue until empty. Set `wait_for_new=True` to keep the worker alive; the method will idle until new waypoints arrive, which is useful when another task feeds the queue opportunistically.
- `clear_waypoints()`  
  Drops any queued-but-unflown segments (safe to call mid-mission before `follow_waypoints()` is awaited again).

Because `follow_waypoints` is awaited, you decide how to sequence the mission: either enqueue everything at once, or stream waypoints while the worker loops with `wait_for_new=True`.

### Guarding Individual Segments
`DroneAPI` exposes `segment_timeout_sec` (default `25` s). When the watchdog fires, the current interpolation is forced to finish and the next waypoint begins. For higher-level control, wrap `follow_waypoints()` in `asyncio.wait_for`, as shown in `test_demo.py`, to fail fast if a specific segment takes too long:

```python
await asyncio.wait_for(drone.follow_waypoints(), timeout=20.0)
```

## Choosing an Interpolation Profile
Import from `DroneAPI.interpolation`. Every class delivers positions (and optionally velocities) to MAVSDK’s offboard interface:

- `Linear` – straight-line pass-through, constant-speed velocity hints, minimal overshoot guard.
- `Bezier` – quadratic Bézier for smoother arcs between waypoints, still pass-through aware.
- `BarrelTowards` – constant-speed bias with cross-track damping; good for aggressive tracking.
- `L1Guidance` – pure-pursuit style controller tuned for reliable pass-through and endgame handling; a solid default for continuous paths.
- `Cubic` – ease-in/ease-out profile that comes to a full stop at the goal, ideal for precise holds.
- `TrapezoidalVelocity` – stop-at-goal with explicit accel/cruise/decel shaping.

Set the mission-wide default via the constructor (`default_interpolation`) and override per waypoint with the `interpolation` argument. All interpolations respect `max_speed_m_s` and the per-waypoint `threshold`.

## Logging & Flight Plots
Enable logging by passing `log_enabled=True`. Once enabled:
- `begin_mission()` automatically calls `start_logging()`; `end_mission()` saves and clears the log via `save_flight_plot(log_path)`.
- `save_flight_plot()` builds a PNG summarizing the XY path, altitude, velocities, accelerations, and distance-to-goal traces.
- Call `start_logging()` / `end_logging()` manually if you need custom recording windows while the mission runs.

The logging buffer only persists for the active mission. A successful save prints `-- Saved flight log plot to ...`.

## Cleanup Patterns
- Always follow `end_mission()` with `shutdown()` if your script exits immediately. `end_mission()` handles landing and log persistence; `shutdown()` guarantees offboard is stopped and the vehicle is disarmed even if an exception occurred.
- In error handlers where the mission never started, call `await drone.shutdown()` to leave the system safe.

## Putting It Together: Demos & Tests
`test_demo.py` demonstrates more advanced patterns:
- Builds reusable waypoint generators (square, circle, dense spline, 3D star).
- Iterates over interpolation classes, instantiating `DroneAPI` for each mission run.
- Wraps every `follow_waypoints()` call in `asyncio.wait_for` to enforce per-segment timeouts.
- Uses `Cubic` for the “return home and land” sequence so the drone stops before descending.

Use the demo as a template when you need to batch missions, compare interpolations, or integrate with ROS2 (`main()` calls `asyncio.run` to stay compatible with synchronous entry points).

## Troubleshooting Checklist
- Stuck before takeoff → confirm MAVSDK connection string and that hardening scripts (See `SetupGuide.md`) were followed.
- Vehicle drifts off path → try `L1Guidance` or `TrapezoidalVelocity`, or reduce `max_speed_m_s`.
- Logs not saved → ensure `log_enabled=True` and call `end_mission()`; the plot is emitted during the landing cleanup.
- Need to abort mid-flight → `await drone.shutdown()` stops offboard and disarms immediately.

With these patterns, `DroneAPI` becomes a predictable building block for structured missions and research experiments.
