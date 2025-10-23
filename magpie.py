# magpie.py
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple, Type, Union

import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw

from .interpolation import (
    InterpContext,
    InterpOutput,
    BaseInterpolation,
    Precision,   # default fallback (GOTO)
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
    Minimal control surface with pluggable interpolation strategies.

    Use:
        from interpolation import Cubic, Precision, Linear, MinJerk

        drone = DroneAPI()
        drone.enqueue_waypoint(1, 2, 3, yaw=45, interpolation=Cubic, threshold=0.10)
        await drone.follow_waypoints()
    """

    def __init__(
        self,
        system_address: str = "udp://:14540",
        control_rate_hz: float = 10.0,
        default_interpolation: Type[BaseInterpolation] = Precision,
        max_speed_m_s: float = 1.5,
        use_velocity_command: bool = True,
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
        self._mission_started = True

    async def end_mission(self) -> None:
        await self.env.simple_land()
        await asyncio.sleep(8.0)
        await self.env.turn_off_offboard()

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
                x=float(x), y=float(y), z=float(z), yaw=float(yaw),
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

            if out.done:
                break
