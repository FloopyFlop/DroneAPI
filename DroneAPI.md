# DroneAPI Reference

`DroneAPI` wraps MAVSDK offboard control with a tiny movement queue, multiple interpolation profiles, and optional telemetry logging. Everything lives in `drone_env_modular.py`.

## Quick Example
```python
import asyncio
from drone_env_modular import DroneAPI

async def mission():
    drone = DroneAPI(system_address="udp://:14540", log_enabled=True)
    await drone.begin_mission(initial_altitude=3.0, yaw=0.0)

    drone.enqueue_waypoint(1.0, 3.0, 4.0, yaw=10.0)
    drone.enqueue_waypoint(0.0, 4.0, 2.0, yaw=0.0, interpolation="precision")
    await drone.follow_waypoints()

    await drone.goto(0.0, 3.0, 0.0)
    await drone.end_mission()
    # Logging automatically saves when end_mission finishes.

asyncio.run(mission())
```

## Lifecycle
- `await begin_mission(initial_altitude=2.0, yaw=0.0)`  
  Connects to the vehicle, waits for healthy telemetry, arms, climbs to the requested altitude, and (when `log_enabled=True`) starts a fresh logging session.
- `await end_mission()`  
  Commands a land, waits for touchdown, turns off offboard, and auto-saves the current flight plot if logging is active.
- `await shutdown()`  
  Emergency-safe cleanup: ensures offboard is stopped and the vehicle is disarmed.

## Movement & Queue
- `await goto(x, y, z, yaw=0.0, interpolation=None)`  
  Sends the drone to a single XYZ target (local frame) using the selected interpolation. Passing `interpolation=None` uses the `default_interpolation` from the constructor.
- `enqueue_waypoint(x, y, z, yaw=0.0, interpolation=None)` / `clear_waypoints()`  
  Push or reset the queued track. Each waypoint may override the interpolation mode.
- `await follow_waypoints(wait_for_new=False, idle_sleep=0.25)`  
  Consumes the queue in order. Set `wait_for_new=True` to keep the worker alive until `stop_waiting_for_waypoints()` is called.
- `stop_waiting_for_waypoints()`  
  Signals a waiting worker to finish when `wait_for_new=True`.
- `available_interpolations()`  
  Returns the list of supported interpolation keywords.

All movement commands require that `begin_mission()` has already been awaited so the local reference frame is known.

## Interpolation Modes
- `linear` – straight-line position samples with constant-speed velocity targets.
- `cubic` – clamped cubic splines (zero velocity at endpoints) for smooth arrivals.
- `minimum_jerk` – quintic easing curve for gradual acceleration and deceleration.
- `precision` – step toward the goal in a straight line, ignoring future waypoints. The drone keeps inching forward until it is within `precision_tolerance` meters of the target. Useful when you need accurate stops more than path smoothness.

Change the default with `default_interpolation` in the constructor or per-call via the `interpolation` argument.

### Precision Parameters
The constructor exposes two tuning knobs:
- `precision_tolerance` (meters, default `0.1`) – distance required before the next waypoint is released.
- `precision_timeout_sec` (default `20.0`) – safety cap on how long the precision controller will try before emitting a warning and moving on.

## Logging & Flight Plots
- Pass `log_enabled=True` (and optionally `log_path="flight_log.png"`) to record telemetry.
- Logging starts automatically inside `begin_mission()`; you can also call `start_logging()` manually for ad-hoc sessions.
- After a mission, call `save_flight_plot(path=None)` to emit a diagnostic PNG with:
  - top-down x–z path with goal markers,
  - x/z/y vs. time traces,
  - velocity components and total speed,
  - distance-to-active-goal vs. time.
  The routine clears the buffered log (`end_logging()`) so the next session begins clean.

## Configuration Helpers
- `set_max_speed(value)` – updates the maximum interpolation speed (affects all modes).
- `set_velocity_ramp(seconds)` – adjusts the soft-start ramp that scales the initial velocity commands.
- `set_velocity_command` (constructor flag) – disable velocity setpoints to issue position-only commands if your vehicle prefers it.

## Demo Scenarios
`python drone_env_modular.py` runs four back-to-back demo missions (linear, cubic, minimum_jerk, precision). Each:
1. Takes off to 3 m.
2. Generates a random track of waypoints with the chosen interpolation.
3. Commands a precise return to `(0, altitude, 0)` and then a gentle descent toward `(0, 0.5, 0)`.
4. Lands, disarms, and saves a log image (`demo_<mode>_flight.png`).

Use the demo output to compare interpolation styles and verify the logging pipeline end-to-end.
