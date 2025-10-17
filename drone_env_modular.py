import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw
from scipy.interpolate import CubicSpline


PRECISION_METHOD = "precision"


def _to_np(values: Union[List[float], np.ndarray]) -> np.ndarray:
    return values if isinstance(values, np.ndarray) else np.array(values, dtype=float)


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw: float = 0.0
    interpolation: Optional[str] = None


class PathInterpolator:
    """
    Generates smooth pose samples (position + yaw) between two points using different
    interpolation schemes. Results are expressed in the drone's local frame.
    """

    def __init__(self, dt: float, max_speed: float = 1.0):
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self._methods = {
            "linear": self._linear_path,
            "cubic": self._cubic_path,
            "minimum_jerk": self._minimum_jerk_path,
        }

    def available_methods(self) -> List[str]:
        return list(self._methods.keys())

    def generate(
        self,
        start: np.ndarray,
        target: np.ndarray,
        method: str,
    ) -> List[Dict[str, np.ndarray]]:
        method = method.lower()
        if method not in self._methods:
            raise ValueError(f"Unknown interpolation method '{method}'. Options: {self.available_methods()}.")

        samples = self._methods[method](start=start, target=target)
        velocities = self._compute_velocities(samples)

        return [
            {"position": sample.copy(), "velocity": vel.copy()}
            for sample, vel in zip(samples, velocities)
        ]

    def _compute_num_steps(self, start: np.ndarray, target: np.ndarray) -> int:
        distance = float(np.linalg.norm(target[:3] - start[:3]))
        duration = max(distance / max(self.max_speed, 1e-3), self.dt)
        steps = max(int(np.ceil(duration / self.dt)) + 1, 2)
        return steps

    def _interp_yaw(self, start_yaw: float, end_yaw: float, steps: int) -> np.ndarray:
        start_rad = np.deg2rad(start_yaw)
        end_rad = np.deg2rad(end_yaw)
        delta = np.arctan2(np.sin(end_rad - start_rad), np.cos(end_rad - start_rad))
        blend = np.linspace(0.0, 1.0, steps, dtype=float)
        yaw_rad = start_rad + blend * delta
        return np.rad2deg(yaw_rad)

    def _linear_path(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        steps = self._compute_num_steps(start, target)
        positions = np.linspace(start[:3], target[:3], steps, axis=0)
        yaw = self._interp_yaw(start[3], target[3], steps)
        return np.column_stack((positions, yaw))

    def _cubic_path(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        steps = self._compute_num_steps(start, target)
        t_samples = np.linspace(0.0, 1.0, steps, dtype=float)

        axis_curves = [
            CubicSpline(
                [0.0, 1.0],
                [float(start[i]), float(target[i])],
                bc_type=((1, 0.0), (1, 0.0)),
            )
            for i in range(3)
        ]
        positions = np.column_stack([curve(t_samples) for curve in axis_curves])
        yaw = self._interp_yaw(start[3], target[3], steps)
        return np.column_stack((positions, yaw))

    def _minimum_jerk_path(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        steps = self._compute_num_steps(start, target)
        blend = np.linspace(0.0, 1.0, steps, dtype=float)
        blend = 10 * np.power(blend, 3) - 15 * np.power(blend, 4) + 6 * np.power(blend, 5)
        positions = start[:3][None, :] + (target[:3] - start[:3])[None, :] * blend[:, None]
        yaw = self._interp_yaw(start[3], target[3], steps)
        return np.column_stack((positions, yaw))

    def _compute_velocities(self, samples: np.ndarray) -> np.ndarray:
        velocities = np.zeros_like(samples, dtype=float)
        velocities[:-1] = (samples[1:] - samples[:-1]) / self.dt
        velocities[-1] = velocities[-2]
        return velocities


class MagpieEnv:
    """
    Thin wrapper around MAVSDK System that keeps track of the drone's state in a local XYZ frame.
    """

    def __init__(self, control_rate_hz: float = 10.0):
        self.drone = System()
        self.control_rate_hz = float(control_rate_hz)
        self.control_dt = 1.0 / self.control_rate_hz

        self.offset_xyz = np.zeros(3, dtype=float)
        self.offset_ready = False

        self.position_xyz = np.zeros(3, dtype=float)
        self.velocity_xyz = np.zeros(3, dtype=float)

    @staticmethod
    def xyz_to_ned(xyz: Union[List[float], np.ndarray]) -> np.ndarray:
        xyz = _to_np(xyz)
        ned = np.zeros(3, dtype=float)
        ned[0] = xyz[2]          # north
        ned[1] = xyz[0]          # east
        ned[2] = -xyz[1]         # down
        return ned

    @staticmethod
    def ned_to_xyz(ned: Union[List[float], np.ndarray]) -> np.ndarray:
        ned = _to_np(ned)
        xyz = np.zeros(3, dtype=float)
        xyz[0] = ned[1]
        xyz[1] = -ned[2]
        xyz[2] = ned[0]
        return xyz

    @staticmethod
    def xyz_to_ned_velocity(vel_xyz_yaw: Union[List[float], np.ndarray]) -> np.ndarray:
        vel_xyz_yaw = _to_np(vel_xyz_yaw)
        ned = np.zeros(4, dtype=float)
        ned[0] = vel_xyz_yaw[2]      # north velocity from z
        ned[1] = vel_xyz_yaw[0]      # east velocity from x
        ned[2] = -vel_xyz_yaw[1]     # down velocity from y
        ned[3] = vel_xyz_yaw[3]      # yaw rate deg/s
        return ned

    @staticmethod
    def ned_to_xyz_velocity(vel_ned: Union[List[float], np.ndarray]) -> np.ndarray:
        vel_ned = _to_np(vel_ned)
        vel_xyz = np.zeros(3, dtype=float)
        vel_xyz[0] = vel_ned[1]          # x from east
        vel_xyz[1] = -vel_ned[2]         # y from -down
        vel_xyz[2] = vel_ned[0]          # z from north
        return vel_xyz

    async def get_position_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        async for message in self.drone.telemetry.position_velocity_ned():
            pos = np.array(
                [message.position.north_m, message.position.east_m, message.position.down_m],
                dtype=float,
            )
            vel = np.array(
                [message.velocity.north_m_s, message.velocity.east_m_s, message.velocity.down_m_s],
                dtype=float,
            )
            return pos, vel

    async def update_state(self) -> None:
        pos_ned, vel_ned = await self.get_position_velocity()
        self.position_xyz = self.ned_to_xyz(pos_ned)
        self.velocity_xyz = self.ned_to_xyz_velocity(vel_ned)

    async def compute_offset(self) -> None:
        await self.update_state()
        self.offset_xyz = self.position_xyz.copy()
        self.offset_ready = True

    async def command_position(
        self,
        xyz_world: Union[List[float], np.ndarray],
        yaw_deg: float,
        velocity_xyz_yaw: Optional[Union[List[float], np.ndarray]] = None,
    ) -> None:
        ned_position = self.xyz_to_ned(xyz_world)
        position_cmd = PositionNedYaw(
            float(ned_position[0]),
            float(ned_position[1]),
            float(ned_position[2]),
            float(yaw_deg),
        )

        if velocity_xyz_yaw is None:
            await self.drone.offboard.set_position_ned(position_cmd)
            return

        ned_velocity = self.xyz_to_ned_velocity(velocity_xyz_yaw)
        velocity_cmd = VelocityNedYaw(
            float(ned_velocity[0]),
            float(ned_velocity[1]),
            float(ned_velocity[2]),
            float(ned_velocity[3]),
        )
        await self.drone.offboard.set_position_velocity_ned(position_cmd, velocity_cmd)

    async def turn_on_offboard(self) -> None:
        try:
            await self.drone.offboard.start()
        except OffboardError as error:
            print(f"Starting offboard failed with: {error._result.result}")
            print("-- Disarming")
            await self.drone.action.disarm()

    async def turn_off_offboard(self) -> None:
        try:
            await self.drone.offboard.stop()
        except OffboardError as error:
            print(f"Stopping offboard failed with: {error._result.result}")

    async def arm(self) -> None:
        print("-- Arming")
        await self.drone.action.arm()

    async def disarm(self) -> None:
        print("-- Disarming")
        await self.drone.action.disarm()

    async def takeoff_to_altitude(self, altitude: float, yaw: float = 0.0) -> None:
        await self.compute_offset()
        target_world = self.offset_xyz + np.array([0.0, altitude, 0.0], dtype=float)

        print("-- Taking off")
        await self.command_position(target_world, yaw_deg=yaw)
        await asyncio.sleep(0.5)
        await self.turn_on_offboard()

        climb_profile = np.array([0.0, -0.2, 0.0, 0.0], dtype=float)
        await self.command_position(target_world, yaw_deg=yaw, velocity_xyz_yaw=climb_profile)
        await asyncio.sleep(3.0)
        await self.update_state()

    async def simple_land(self) -> None:
        print("-- Landing")
        await self.drone.action.land()


class DroneAPI:
    """
    Minimal drone control surface with only:
      * goto()  - fly to a single point using the selected interpolation
      * follow_waypoints() driven by an easy-to-use queue + dynamic interpolation
    Optional logging produces a diagnostic plot of position/velocity vs. goals.
    """

    def __init__(
        self,
        system_address: str = "udp://:14540",
        control_rate_hz: float = 10.0,
        default_interpolation: str = "minimum_jerk",
        max_speed_m_s: float = 1.5,
        use_velocity_command: bool = True,
        log_enabled: bool = False,
        log_path: str = "flight_log.png",
        takeoff_settle_sec: float = 1.5,
        velocity_ramp_sec: float = 1.0,
        precision_tolerance: float = 0.1,
        precision_timeout_sec: float = 20.0,
    ):
        self.system_address = system_address
        self.env = MagpieEnv(control_rate_hz=control_rate_hz)
        self.interpolator = PathInterpolator(dt=self.env.control_dt, max_speed=max_speed_m_s)
        self.default_interpolation = default_interpolation.lower()
        self.use_velocity_command = bool(use_velocity_command)
        self.takeoff_settle_sec = float(takeoff_settle_sec)
        self.velocity_ramp_sec = float(max(0.0, velocity_ramp_sec))
        self._update_velocity_ramp_steps()
        self.precision_method = PRECISION_METHOD
        self.precision_tolerance = float(max(precision_tolerance, 0.0))
        self.precision_timeout_sec = float(max(precision_timeout_sec, self.env.control_dt))

        self._waypoint_queue: Deque[Waypoint] = deque()
        self._queue_stop_event: Optional[asyncio.Event] = None
        self._queue_active = False

        self._current_yaw = 0.0
        self._mission_start = 0.0

        self.log_enabled = log_enabled
        self.log_path = log_path
        self.telemetry_log: List[Dict[str, np.ndarray]] = []
        self.goal_history: List[np.ndarray] = []

    def _update_velocity_ramp_steps(self) -> None:
        if self.velocity_ramp_sec <= 0.0:
            self._velocity_ramp_steps = 1
        else:
            self._velocity_ramp_steps = max(int(self.velocity_ramp_sec / self.env.control_dt), 1)

    def set_max_speed(self, max_speed_m_s: float) -> None:
        self.interpolator.max_speed = float(max_speed_m_s)

    def set_velocity_ramp(self, ramp_seconds: float) -> None:
        self.velocity_ramp_sec = float(max(0.0, ramp_seconds))
        self._update_velocity_ramp_steps()

    def available_interpolations(self) -> List[str]:
        methods = list(self.interpolator.available_methods())
        if self.precision_method not in methods:
            methods.append(self.precision_method)
        return methods

    # ----- Mission lifecycle -------------------------------------------------

    async def begin_mission(self, initial_altitude: float = 2.0, yaw: float = 0.0) -> None:
        await self.env.drone.connect(system_address=self.system_address)

        print("Waiting for connection...")
        async for state in self.env.drone.core.connection_state():
            if state.is_connected:
                print("-- Connection successful")
                break

        print("Waiting for global position / home position...")
        async for health in self.env.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("-- Global position estimate OK")
                break

        await self.env.turn_off_offboard()

        await self.env.arm()
        await self.env.takeoff_to_altitude(altitude=initial_altitude, yaw=yaw)
        self._current_yaw = yaw
        if self.log_enabled:
            self.start_logging()
        if self.takeoff_settle_sec > 0.0:
            await asyncio.sleep(self.takeoff_settle_sec)
            await self.env.update_state()

    async def end_mission(self) -> None:
        await self.env.simple_land()
        await asyncio.sleep(10.0)
        await self.env.turn_off_offboard()
        if self.log_enabled:
            self.save_flight_plot()

    async def shutdown(self) -> None:
        await self.env.turn_off_offboard()
        await self.env.disarm()

    # ----- Queue management --------------------------------------------------

    def enqueue_waypoint(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        interpolation: Optional[str] = None,
    ) -> None:
        self._waypoint_queue.append(Waypoint(x, y, z, yaw, interpolation))

    def clear_waypoints(self) -> None:
        self._waypoint_queue.clear()

    async def follow_waypoints(self, wait_for_new: bool = False, idle_sleep: float = 0.25) -> None:
        """
        Process the waypoint queue until empty. Optionally keep waiting for new waypoints.
        """
        if self._queue_active:
            raise RuntimeError("Waypoint queue is already being processed.")

        self._queue_active = True
        self._queue_stop_event = asyncio.Event()

        try:
            while True:
                if self._waypoint_queue:
                    waypoint = self._waypoint_queue.popleft()
                    await self.goto(
                        waypoint.x,
                        waypoint.y,
                        waypoint.z,
                        yaw=waypoint.yaw,
                        interpolation=waypoint.interpolation,
                        record_goal=True,
                    )
                elif wait_for_new and self._queue_stop_event is not None:
                    if self._queue_stop_event.is_set():
                        break
                    await asyncio.sleep(idle_sleep)
                else:
                    break
        finally:
            self._queue_active = False
            self._queue_stop_event = None

    def stop_waiting_for_waypoints(self) -> None:
        if self._queue_stop_event is not None:
            self._queue_stop_event.set()

    # ----- Movement ---------------------------------------------------------

    async def goto(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        interpolation: Optional[str] = None,
        record_goal: bool = True,
    ) -> None:
        if not self.env.offset_ready:
            raise RuntimeError("Call begin_mission() first to initialize the local frame.")

        await self.env.update_state()

        current_local = self.env.position_xyz - self.env.offset_xyz
        start = np.array([current_local[0], current_local[1], current_local[2], self._current_yaw], dtype=float)
        target = np.array([x, y, z, yaw], dtype=float)

        method = (interpolation or self.default_interpolation).lower()
        samples: Optional[List[Dict[str, np.ndarray]]] = None

        if method != self.precision_method:
            samples = self.interpolator.generate(start=start, target=target, method=method)

        if record_goal:
            self.goal_history.append(target[:3].copy())

        if self.log_enabled:
            self.telemetry_log.append(
                self._make_log_sample(target_local=target)
            )

        if method == self.precision_method:
            await self._fly_precise(target_local=target)
        else:
            assert samples is not None
            await self._fly_path(samples=samples, target_local=target)
        self._current_yaw = float(yaw)

    async def _fly_path(self, samples: List[Dict[str, np.ndarray]], target_local: np.ndarray) -> None:
        # Skip the first sample (current state)
        for step_idx, sample in enumerate(samples[1:], start=1):
            pos_local = sample["position"]
            vel_local = sample["velocity"]

            world_xyz = pos_local[:3] + self.env.offset_xyz
            yaw_deg = float(pos_local[3])

            velocity_xyz_yaw = None
            if self.use_velocity_command:
                ramp_scale = min(step_idx / float(self._velocity_ramp_steps), 1.0)
                velocity_xyz_yaw = np.array(
                    [
                        vel_local[0] * ramp_scale,
                        vel_local[1] * ramp_scale,
                        vel_local[2] * ramp_scale,
                        vel_local[3] * ramp_scale,
                    ],
                    dtype=float,
                )

            await self.env.command_position(world_xyz, yaw_deg=yaw_deg, velocity_xyz_yaw=velocity_xyz_yaw)
            await asyncio.sleep(self.env.control_dt)

            if self.log_enabled:
                await self.env.update_state()
                self.telemetry_log.append(self._make_log_sample(target_local=target_local))

    async def _fly_precise(self, target_local: np.ndarray) -> None:
        max_step_distance = max(self.interpolator.max_speed * self.env.control_dt, 1e-3)
        max_iterations = max(int(self.precision_timeout_sec / self.env.control_dt), 1)
        yaw_deg = float(target_local[3])

        for _ in range(max_iterations):
            await self.env.update_state()
            position_local = self.env.position_xyz - self.env.offset_xyz
            error_vec = target_local[:3] - position_local
            distance = float(np.linalg.norm(error_vec))

            if self.log_enabled:
                self.telemetry_log.append(self._make_log_sample(target_local=target_local))

            if distance <= self.precision_tolerance:
                break

            if distance > 1e-6:
                direction = error_vec / distance
            else:
                direction = np.zeros(3, dtype=float)

            step_distance = min(distance, max_step_distance)
            next_local = position_local + direction * step_distance
            world_xyz = next_local + self.env.offset_xyz

            velocity_xyz_yaw = None
            if self.use_velocity_command and distance > 1e-6:
                commanded_speed = min(self.interpolator.max_speed, distance / self.env.control_dt)
                velocity_xyz_yaw = np.array(
                    [
                        direction[0] * commanded_speed,
                        direction[1] * commanded_speed,
                        direction[2] * commanded_speed,
                        0.0,
                    ],
                    dtype=float,
                )

            await self.env.command_position(world_xyz, yaw_deg=yaw_deg, velocity_xyz_yaw=velocity_xyz_yaw)
            await asyncio.sleep(self.env.control_dt)
        else:
            print(
                f"-- Precision interpolation timed out before reaching tolerance "
                f"{self.precision_tolerance:.2f} m"
            )

        target_world = target_local[:3] + self.env.offset_xyz
        await self.env.command_position(target_world, yaw_deg=yaw_deg)
        await asyncio.sleep(self.env.control_dt)
        if self.log_enabled:
            await self.env.update_state()
            self.telemetry_log.append(self._make_log_sample(target_local=target_local))

    # ----- Logging ----------------------------------------------------------

    def _make_log_sample(self, target_local: np.ndarray) -> Dict[str, np.ndarray]:
        time_stamp = time.time() - self._mission_start
        position_local = (self.env.position_xyz - self.env.offset_xyz).copy()
        velocity_local = self.env.velocity_xyz.copy()
        distance_to_goal = np.linalg.norm(position_local - target_local[:3])
        return {
            "time": np.array(time_stamp, dtype=float),
            "position": position_local,
            "velocity": velocity_local,
            "goal": target_local[:3].copy(),
            "distance_to_goal": np.array(distance_to_goal, dtype=float),
        }

    def start_logging(self) -> None:
        self.telemetry_log.clear()
        self.goal_history.clear()
        self._mission_start = time.time()

    def end_logging(self) -> None:
        self.telemetry_log.clear()
        self.goal_history.clear()
        self._mission_start = 0.0

    def save_flight_plot(self, path: Optional[str] = None) -> Optional[str]:
        if not self.log_enabled or len(self.telemetry_log) < 2:
            self.end_logging()
            return None

        path = path or self.log_path

        times = np.array([sample["time"] for sample in self.telemetry_log], dtype=float)
        positions = np.stack([sample["position"] for sample in self.telemetry_log])
        goals = np.stack([sample["goal"] for sample in self.telemetry_log])
        velocities = np.stack([sample["velocity"] for sample in self.telemetry_log])
        distances = np.array([sample["distance_to_goal"] for sample in self.telemetry_log], dtype=float)

        speed = np.linalg.norm(velocities, axis=1)

        goal_change_indices = [0]
        for i in range(1, len(goals)):
            if not np.allclose(goals[i], goals[i - 1]):
                goal_change_indices.append(i)
        goal_change_times = times[goal_change_indices]

        fig = plt.figure(figsize=(12, 14))
        gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.0, 1.0, 0.9], hspace=0.35, wspace=0.25)

        ax_path = fig.add_subplot(gs[0, :])
        ax_path.plot(positions[:, 0], positions[:, 2], label="trajectory", color="C0")
        unique_goals = goals[goal_change_indices]
        ax_path.scatter(unique_goals[:, 0], unique_goals[:, 2], color="C1", marker="x", s=60, label="goals")
        ax_path.set_xlabel("x (m)")
        ax_path.set_ylabel("z (m)")
        ax_path.set_title("Top-down path (x vs z)")
        ax_path.grid(True, linestyle=":")
        ax_path.axis("equal")
        ax_path.legend(loc="best")

        ax_pos_x = fig.add_subplot(gs[1, 0])
        ax_pos_x.plot(times, positions[:, 0], label="x actual", color="C0")
        ax_pos_x.plot(times, goals[:, 0], "--", label="x target", color="C1")
        for t in goal_change_times:
            ax_pos_x.axvline(t, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax_pos_x.set_ylabel("x (m)")
        ax_pos_x.grid(True, linestyle=":")
        ax_pos_x.legend(loc="upper right")

        ax_pos_z = fig.add_subplot(gs[1, 1])
        ax_pos_z.plot(times, positions[:, 2], label="z actual", color="C0")
        ax_pos_z.plot(times, goals[:, 2], "--", label="z target", color="C1")
        for t in goal_change_times:
            ax_pos_z.axvline(t, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax_pos_z.set_ylabel("z (m)")
        ax_pos_z.grid(True, linestyle=":")
        ax_pos_z.legend(loc="upper right")

        ax_alt = fig.add_subplot(gs[2, 0])
        ax_alt.plot(times, positions[:, 1], label="y actual", color="C0")
        ax_alt.plot(times, goals[:, 1], "--", label="y target", color="C1")
        for t in goal_change_times:
            ax_alt.axvline(t, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax_alt.set_ylabel("y (m)")
        ax_alt.set_xlabel("time (s)")
        ax_alt.grid(True, linestyle=":")
        ax_alt.legend(loc="upper right")

        ax_vel = fig.add_subplot(gs[2, 1])
        ax_vel.plot(times, velocities[:, 0], label="vx", color="C0")
        ax_vel.plot(times, velocities[:, 1], label="vy", color="C1")
        ax_vel.plot(times, velocities[:, 2], label="vz", color="C2")
        ax_vel.plot(times, speed, label="speed", color="C3", linewidth=1.5)
        ax_vel.set_ylabel("velocity (m/s)")
        ax_vel.set_xlabel("time (s)")
        ax_vel.grid(True, linestyle=":")
        ax_vel.legend(loc="upper right")

        ax_dist = fig.add_subplot(gs[3, :])
        ax_dist.plot(times, distances, color="C4")
        for t in goal_change_times:
            ax_dist.axvline(t, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax_dist.set_ylabel("distance to goal (m)")
        ax_dist.set_xlabel("time (s)")
        ax_dist.grid(True, linestyle=":")
        ax_dist.set_title("Distance to active goal")

        fig.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close(fig)
        self.end_logging()

        print(f"-- Saved flight log plot to {path}")
        return path


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def _demo():
    tests = [
        ("linear", "demo_linear_flight.png"),
        ("cubic", "demo_cubic_flight.png"),
        ("minimum_jerk", "demo_minimum_jerk_flight.png"),
        (PRECISION_METHOD, "demo_precision_flight.png"),
    ]

    rng = np.random.default_rng(seed=42)
    initial_altitude = 3.0

    for method, output_file in tests:
        print(f"\n=== Running {method} interpolation test ===")
        drone = DroneAPI(
            system_address="udp://:14540",
            default_interpolation=method,
            log_enabled=True,
            log_path=output_file,
        )

        await drone.begin_mission(initial_altitude=initial_altitude, yaw=0.0)

        # Generate a random waypoint track (x, y, z) within a reasonable cube around the origin.
        num_waypoints = int(rng.integers(4, 7))
        random_track = rng.uniform(low=[-2.0, 2.0, -2.0], high=[2.0, 5.0, 2.0], size=(num_waypoints, 3))
        yaw_samples = rng.uniform(low=-35.0, high=35.0, size=num_waypoints)

        for idx in range(num_waypoints):
            x, y, z = random_track[idx]
            yaw = yaw_samples[idx]
            drone.enqueue_waypoint(float(x), float(y), float(z), yaw=float(yaw), interpolation=method)

        await drone.follow_waypoints()

        # Return to the origin before landing so we touch down at (0,0).
        await drone.goto(0.0, initial_altitude, 0.0, yaw=0.0, interpolation=method)
        await drone.goto(0.0, 0.0, 0.0, yaw=0.0, interpolation=method)

        await drone.end_mission()
        await drone.shutdown()


def main():
    asyncio.run(_demo())


if __name__ == "__main__":
    main()
