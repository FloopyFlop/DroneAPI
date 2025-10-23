# magpie.py
import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw

from .interpolation import (
    InterpContext,
    InterpOutput,
    BaseInterpolation,
    Linear,   # default fallback (GOTO)
)

# ------------------ utilities ------------------

def _to_np(values: Union[List[float], np.ndarray]) -> np.ndarray:
    return values if isinstance(values, np.ndarray) else np.array(values, dtype=float)


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw: float = 0.0
    interpolation: Optional[Type[BaseInterpolation]] = None
    threshold: float = 0.15  # per-waypoint tolerance override (meters)


# ------------------ MagpieEnv ------------------

class MagpieEnv:
    """
    Thin wrapper around MAVSDK System that keeps track of the drone's state in a local XYZ frame.
    Local XYZ frame mapping (consistent with your original):
      x -> East, y -> Up, z -> North
    """

    def __init__(self, control_rate_hz: float = 10.0):
        self.drone = System()
        self.control_rate_hz = float(control_rate_hz)
        self.control_dt = 1.0 / self.control_rate_hz

        self.offset_xyz = np.zeros(3, dtype=float)
        self.offset_ready = False

        self.position_xyz = np.zeros(3, dtype=float)
        self.velocity_xyz = np.zeros(3, dtype=float)

    # ---------- frame transforms ----------

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
        xyz[0] = ned[1]          # x from east
        xyz[1] = -ned[2]         # y from -down
        xyz[2] = ned[0]          # z from north
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

    # ---------- telemetry ----------

    async def _read_telemetry_once(self) -> Tuple[np.ndarray, np.ndarray]:
        async for message in self.drone.telemetry.position_velocity_ned():
            pos_ned = np.array(
                [message.position.north_m, message.position.east_m, message.position.down_m],
                dtype=float,
            )
            vel_ned = np.array(
                [message.velocity.north_m_s, message.velocity.east_m_s, message.velocity.down_m_s],
                dtype=float,
            )
            return pos_ned, vel_ned

    async def update_state(self) -> None:
        pos_ned, vel_ned = await self._read_telemetry_once()
        self.position_xyz = self.ned_to_xyz(pos_ned)
        self.velocity_xyz = self.ned_to_xyz_velocity(vel_ned)

    async def compute_offset(self) -> None:
        await self.update_state()
        self.offset_xyz = self.position_xyz.copy()
        self.offset_ready = True

    # ---------- command helpers ----------

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

    # ---------- lifecycle ----------

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


# ------------------ DroneAPI ------------------

class DroneAPI:
    """
    Control surface with pluggable interpolation strategies + optional logging/plotting.

    Use:
        from .interpolation import Cubic, Precision, Linear, MinJerk

        drone = DroneAPI(log_enabled=True, log_path="flight_log.png")
        drone.enqueue_waypoint(1, 2, 3, yaw=45, interpolation=Cubic, threshold=0.10)
        await drone.follow_waypoints()
    """

    def __init__(
        self,
        system_address: str = "udp://:14540",
        control_rate_hz: float = 10.0,
        default_interpolation: Type[BaseInterpolation] = Linear,
        max_speed_m_s: float = 1.5,
        use_velocity_command: bool = True,
        *,
        log_enabled: bool = False,
        log_path: Optional[str] = None,
    ):
        self.system_address = system_address
        self.env = MagpieEnv(control_rate_hz=control_rate_hz)
        self.control_dt = self.env.control_dt
        self.default_interpolation_cls = default_interpolation
        self.max_speed = float(max_speed_m_s)
        self.use_velocity_command = bool(use_velocity_command)

        self._waypoint_queue: Deque[Waypoint] = deque()
        self._current_yaw = 0.0
        self._mission_started = False

        # ---- logging state ----
        self.log_enabled = bool(log_enabled)
        self.log_path = log_path or "flight_log.png"
        self._mission_start = 0.0
        self.telemetry_log: List[Dict[str, np.ndarray]] = []
        self.goal_history: List[np.ndarray] = []

    # ---------- mission lifecycle ----------

    async def begin_mission(self, initial_altitude: float = 2.0, yaw: float = 0.0) -> None:
        if self._mission_started:
            return

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
        self._current_yaw = float(yaw)

        if self.log_enabled:
            self.start_logging()

        self._mission_started = True

    async def end_mission(self) -> None:
        await self.env.simple_land()
        await asyncio.sleep(8.0)
        await self.env.turn_off_offboard()
        if self.log_enabled:
            self.save_flight_plot(self.log_path)

    async def shutdown(self) -> None:
        await self.env.turn_off_offboard()
        await self.env.disarm()

    # ---------- queue management ----------

    def enqueue_waypoint(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        interpolation: Optional[Type[BaseInterpolation]] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self._waypoint_queue.append(
            Waypoint(
                x=float(x), y=float(y), z=float(z), yaw=float(y),
                interpolation=interpolation, threshold=float(threshold) if threshold is not None else 0.15
            )
        )

    def clear_waypoints(self) -> None:
        self._waypoint_queue.clear()

    # ---------- movement loop with interpolation ----------

    async def follow_waypoints(self, wait_for_new: bool = False, idle_sleep: float = 0.25) -> None:
        if not self.env.offset_ready:
            raise RuntimeError("Call begin_mission() first to initialize the local frame.")

        while True:
            if not self._waypoint_queue:
                if wait_for_new:
                    await asyncio.sleep(idle_sleep)
                    continue
                else:
                    break

            current_wp = self._waypoint_queue.popleft()
            future_wps = list(self._waypoint_queue)  # shallow copy
            await self._run_interpolation_to_waypoint(current_wp, future_wps)

    async def _run_interpolation_to_waypoint(self, wp: Waypoint, future_wps: List[Waypoint]) -> None:
        await self.env.update_state()

        # Local frame start pose
        current_local_pos = self.env.position_xyz - self.env.offset_xyz
        start_pos = np.array([current_local_pos[0], current_local_pos[1], current_local_pos[2]], dtype=float)
        start_yaw = float(self._current_yaw)
        start_vel = self.env.velocity_xyz.copy()

        # Local frame target pose
        target_local = np.array([wp.x, wp.y, wp.z], dtype=float)
        target_yaw = float(wp.yaw)
        if self.log_enabled:
            self.goal_history.append(target_local.copy())

        # Build context for interpolation
        ctx = InterpContext(
            dt=self.control_dt,
            max_speed=self.max_speed,
            threshold=max(0.0, float(wp.threshold)),
            now_state_pos=start_pos,
            now_state_vel=start_vel,
            now_state_yaw=start_yaw,
            target_pos=target_local,
            target_yaw=target_yaw,
            future_waypoints=[(w.x, w.y, w.z, w.yaw, w.threshold) for w in future_wps],
            use_velocity_command=self.use_velocity_command,
        )

        # Pick strategy
        interp_cls = wp.interpolation or self.default_interpolation_cls
        interp: BaseInterpolation = interp_cls()
        interp.start(ctx)  # allow strategy to initialize internal state

        # Prime logging (initial sample)
        if self.log_enabled:
            self.telemetry_log.append(self._make_log_sample(target_local=target_local))

        # control loop until strategy says done
        while True:
            await self.env.update_state()

            # refresh ctx with latest state
            current_local_pos = self.env.position_xyz - self.env.offset_xyz
            ctx.now_state_pos = current_local_pos.copy()
            ctx.now_state_vel = self.env.velocity_xyz.copy()
            ctx.now_state_yaw = float(self._current_yaw)

            out: InterpOutput = interp.step(ctx)

            # Prepare command
            desired_local_pos = out.position_local if out.position_local is not None else target_local
            world_xyz = desired_local_pos + self.env.offset_xyz
            yaw_cmd = float(out.yaw_deg if out.yaw_deg is not None else target_yaw)

            vel_cmd = out.velocity_xyz_yaw if (self.use_velocity_command and out.velocity_xyz_yaw is not None) else None

            await self.env.command_position(world_xyz, yaw_deg=yaw_cmd, velocity_xyz_yaw=vel_cmd)
            self._current_yaw = yaw_cmd  # track commanded yaw
            await asyncio.sleep(self.control_dt)

            # log a sample each tick
            if self.log_enabled:
                self.telemetry_log.append(self._make_log_sample(target_local=target_local))

            if out.done:
                break

    # ----- Logging (restored) ----------------------------------------------------------

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

        cumulative_path = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(positions[:, :3], axis=0), axis=1)))
        )
        diffs = np.diff(times) if len(times) > 1 else np.array([self.env.control_dt])
        dt_candidate = float(self.env.control_dt) if getattr(self.env, "control_dt", 0.0) > 0 else float(np.median(diffs))
        if not np.isfinite(dt_candidate) or dt_candidate <= 0.0:
            dt_candidate = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0

        accelerations = np.zeros_like(velocities)
        if len(times) >= 2:
            accelerations[1:] = np.diff(velocities, axis=0) / dt_candidate
            accelerations[0] = accelerations[1]
        accel_magnitude = np.linalg.norm(accelerations, axis=1)

        scale_factor = 3.0 ** 0.5
        fig = plt.figure(figsize=(12.8 * scale_factor, 4.32 * scale_factor), dpi=400)
        gs = fig.add_gridspec(
            2,
            4,
            width_ratios=[1.25, 1.0, 1.0, 1.0],
            height_ratios=[1.0, 1.0],
            hspace=0.32,
            wspace=0.28,
        )

        # Row 0
        ax_path = fig.add_subplot(gs[0, 0])
        path_points = np.column_stack((positions[:, 0], positions[:, 2]))
        path_segments = np.concatenate(
            [path_points[:-1, None, :], path_points[1:, None, :]],
            axis=1,
        )
        time_norm = plt.Normalize(times[0], times[-1])
        lc = LineCollection(
            path_segments,
            cmap="viridis",
            norm=time_norm,
            linewidth=1.6,
            alpha=0.95,
        )
        lc.set_array(times[:-1])
        ax_path.add_collection(lc)
        unique_goals = goals[goal_change_indices]
        ax_path.scatter(unique_goals[:, 0], unique_goals[:, 2], color="C1", marker="x", s=45, label="goals")
        ax_path.scatter(positions[0, 0], positions[0, 2], color="C2", s=55, marker="o", label="start")
        ax_path.scatter(positions[-1, 0], positions[-1, 2], color="C3", s=55, marker="s", label="end")
        ax_path.set_title("Top-down path (x vs z)", pad=8)
        ax_path.set_xlabel("x (m)")
        ax_path.set_ylabel("z (m)")
        x_span = positions[:, 0].ptp()
        z_span = positions[:, 2].ptp()
        max_span = max(x_span, z_span, 1.0)
        margin = 0.08 * max_span + 0.15
        ax_path.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
        ax_path.set_ylim(positions[:, 2].min() - margin, positions[:, 2].max() + margin)
        if z_span > 0 and x_span > 0:
            ax_path.set_box_aspect(z_span / x_span)
        else:
            ax_path.set_box_aspect(1.0)
        ax_path.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax_path.set_facecolor("#fbfbfb")
        ax_path.legend(loc="upper left", fontsize=7, frameon=False, handlelength=1.6)
        cbar = fig.colorbar(lc, ax=ax_path, orientation="vertical", fraction=0.055, pad=0.02)
        cbar.set_label("time (s)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax_pos_x = fig.add_subplot(gs[0, 1])
        ax_pos_x.plot(times, positions[:, 0], label="x", color="C0", linewidth=1.2)
        ax_pos_x.plot(times, goals[:, 0], linestyle="--", label="x goal", color="C1", linewidth=1.0)
        for t in goal_change_times:
            ax_pos_x.axvline(t, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
        ax_pos_x.set_title("x position vs target", pad=8)
        ax_pos_x.set_ylabel("x (m)")
        ax_pos_x.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax_pos_x.legend(loc="upper left", fontsize=7, frameon=False, handlelength=1.6)

        ax_pos_y = fig.add_subplot(gs[0, 2])
        ax_pos_y.plot(times, positions[:, 1], label="y", color="C0", linewidth=1.2)
        ax_pos_y.plot(times, goals[:, 1], linestyle="--", label="y goal", color="C1", linewidth=1.0)
        for t in goal_change_times:
            ax_pos_y.axvline(t, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
        ax_pos_y.set_title("y position vs target", pad=8)
        ax_pos_y.set_ylabel("y (m)")
        ax_pos_y.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax_pos_y.legend(loc="upper left", fontsize=7, frameon=False, handlelength=1.6)

        ax_pos_z = fig.add_subplot(gs[0, 3])
        ax_pos_z.plot(times, positions[:, 2], label="z", color="C0", linewidth=1.2)
        ax_pos_z.plot(times, goals[:, 2], linestyle="--", label="z goal", color="C1", linewidth=1.0)
        for t in goal_change_times:
            ax_pos_z.axvline(t, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
        ax_pos_z.set_title("z position vs target", pad=8)
        ax_pos_z.set_ylabel("z (m)")
        ax_pos_z.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax_pos_z.legend(loc="upper left", fontsize=7, frameon=False, handlelength=1.6)

        # Row 1
        ax_vel_components = fig.add_subplot(gs[1, 0])
        ax_vel_components.plot(times, velocities[:, 0], label="vx", color="C0", linewidth=1.1)
        ax_vel_components.plot(times, velocities[:, 1], label="vy", color="C1", linewidth=1.1)
        ax_vel_components.plot(times, velocities[:, 2], label="vz", color="C2", linewidth=1.1)
        for t in goal_change_times:
            ax_vel_components.axvline(t, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
        ax_vel_components.set_title("Velocity components", pad=8)
        ax_vel_components.set_ylabel("velocity (m/s)")
        ax_vel_components.set_xlabel("time (s)")
        ax_vel_components.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax_vel_components.legend(loc="upper left", fontsize=7, frameon=False, ncol=2, handlelength=1.6)

        ax_speed = fig.add_subplot(gs[1, 1])
        ax_speed.plot(times, speed, color="C3", linewidth=1.3)
        for t in goal_change_times:
            ax_speed.axvline(t, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
        ax_speed.set_title("Speed (||v||)", pad=8)
        ax_speed.set_ylabel("speed (m/s)")
        ax_speed.set_xlabel("time (s)")
        ax_speed.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)

        ax_accel = fig.add_subplot(gs[1, 2])
        ax_accel.plot(times, accelerations[:, 0], label="ax", color="C0", linewidth=1.1)
        ax_accel.plot(times, accelerations[:, 1], label="ay", color="C1", linewidth=1.1)
        ax_accel.plot(times, accelerations[:, 2], label="az", color="C2", linewidth=1.1)
        ax_accel.plot(times, accel_magnitude, label="|a|", color="C4", linewidth=1.3, linestyle="--")
        for t in goal_change_times:
            ax_accel.axvline(t, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
        ax_accel.set_title("Acceleration profile", pad=8)
        ax_accel.set_ylabel("acceleration (m/s²)")
        ax_accel.set_xlabel("time (s)")
        ax_accel.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax_accel.legend(loc="upper left", fontsize=7, frameon=False, ncol=2, handlelength=1.6)

        ax_distance = fig.add_subplot(gs[1, 3])
        ax_distance.plot(times, distances, color="C5", linewidth=1.3, label="distance to goal")
        ax_distance.plot(times, cumulative_path, color="C6", linewidth=1.0, linestyle="--", label="path length")
        for t in goal_change_times:
            ax_distance.axvline(t, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
        ax_distance.set_title("Goal distance & path length", pad=8)
        ax_distance.set_ylabel("meters")
        ax_distance.set_xlabel("time (s)")
        ax_distance.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax_distance.legend(loc="upper left", fontsize=7, frameon=False, handlelength=1.6)

        for ax in fig.axes:
            ax.tick_params(labelsize=8)

        fig.patch.set_facecolor("white")

        plt.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        self.end_logging()

        print(f"-- Saved flight log plot to {path}")
        return path
