# Drone API Overview

This module bundles point-cloud planning, MAVSDK bindings, and a thin facade for flying a vehicle with Velocity Field Histogram (VFH) avoidance. Use it when you want a minimal path-planning+execution loop without wiring up the planner and environment by hand.

## Quick Start
- Instantiate `DroneAPI(system_address="udp://:14540")`.
- `await begin_mission()` to arm and climb to a safe altitude.
- (Optional) `set_point_cloud(cloud)` with a numpy `(N, 3)` obstacle cloud to seed the planner.
- Drive the vehicle with `goto_xyz`, `follow_waypoints`, or `goto_with_pathfinding`.
- `await end_mission()` followed by `await shutdown()` to land and disarm safely.

```python
import asyncio
import numpy as np
from drone_env_modular import DroneAPI

async def fly_square():
    api = DroneAPI()
    await api.begin_mission(initial_altitude=3.0)
    api.set_point_cloud(np.random.uniform(-5, 5, size=(500, 3)))
    await api.goto_with_pathfinding(0, 5, 5)
    await api.end_mission()
    await api.shutdown()

asyncio.run(fly_square())
```

## Facade: `DroneAPI`
- **Lifecycle**: `begin_mission`, `end_mission`, and `shutdown` wrap MAVSDK connection, arming, takeoff, landing, and offboard management.
- **Planning settings**: `set_goal`, `set_point_cloud`, and `set_vfh_params` feed new goals or obstacle data to the underlying planner.
- **Motion commands**:
  - `goto_xyz(x, y, z, yaw=0)` issues a single offboard position setpoint.
  - `follow_waypoints([(x, y, z, yaw), ...])` reuses `goto_xyz` with dwell times.
  - `plan_path_to(x, y, z)` returns an `N x 12` state array (xyz populated) computed with VFH.
  - `goto_with_pathfinding(x, y, z, yaw=0)` plans first, then iterates through the path.
  - `repeat_path(waypoints, times=2)` loops over a set of waypoints.

All drone-facing methods assume `begin_mission()` has run: this call computes `state_offset`, starts offboard mode, and guarantees telemetry is healthy.

## Environment Layer: `MagpieEnv`
- Owns the MAVSDK `System` instance and handles async helpers (`arm`, `disarm`, `turn_on_offboard`, etc.).
- Maintains the current 12-element state vector, converts between the module's XYZ frame and MAVSDK NED via `xyz_to_NED`, and caches the latest planned path.
- `plan_path()` delegates to the planner, stores the result on `self.path`, and `path_is_safe()` re-checks candidate goals against VFH collision tests.

## Planner Stack
- **`pathPlanner`**: wraps concrete planners (currently only `VFH`). It tracks the active goal, interpolation style (`linear` or `spline`), and desired average speed. `compute_desired_path(state, point_cloud)` returns a numpy array of 12-element states ready for offboard dispatch.
- **`basePathPlanner.VFH`**: converts point clouds into a layered polar histogram (`polarHistogram3D`) and selects collision-free bins near the goal. Tunable parameters include `radius`, `layers`, `min_obstacle_distance`, and `angle_sections`.
- **`polarHistogram3D`** (internal): bins obstacle points, tracks per-bin Gaussian statistics, and runs safety checks (`check_point_safety`, `confirm_candidate_distance`) during planning iterations.

## Parameter Tweaks
- Increase `radius` and `layers` for look-ahead distance at the cost of compute time.
- `min_obstacle_distance` acts as the clearance radius; increasing it yields wider margins.
- Switch `interpolation="linear"` if you prefer straight segments over splined paths; adjust `spline_points` to control sampling density.

## Concurrency Notes
All public methods that touch the drone (`begin_mission`, `goto_xyz`, `goto_with_pathfinding`, etc.) are `async` and must run inside an asyncio event loop. Planner configuration (`set_goal`, `set_vfh_params`, `set_point_cloud`) is synchronous.
