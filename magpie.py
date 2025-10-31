# magpie.py
import asyncio
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Type, Union

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


# ------------------ ROS2 publisher helpers ------------------

class _BaseStatePublisher:
    """Minimal interface so DroneAPI can swap publishers for ROS2 or tests."""

    def publish(self, payload: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class _Ros2StatePublisher(_BaseStatePublisher):
    """
    ROS 2 publisher that spins in a background thread.

    Designed so DroneAPI can publish JSON snapshots without tying up the main asyncio loop.
    """

    def __init__(self, topic: str, node_name: str, qos_depth: int):
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from std_msgs.msg import String
        except ImportError as exc:  # pragma: no cover - ROS2 optional dependency
            raise RuntimeError(
                "ROS 2 telemetry streaming requested, but rclpy/std_msgs are not installed."
            ) from exc

        self._rclpy = rclpy
        self._String = String
        self._executor = SingleThreadedExecutor()
        self._lock = threading.Lock()
        self._topic = topic
        self._running = threading.Event()
        self._running.set()
        self._owns_context = False

        if not self._rclpy.ok():
            self._rclpy.init(args=None)
            self._owns_context = True

        self._node = Node(node_name)
        self._publisher = self._node.create_publisher(String, topic, qos_depth)
        self._executor.add_node(self._node)

        self._spin_thread = threading.Thread(target=self._spin, name=f"{node_name}_spin", daemon=True)
        self._spin_thread.start()

    def _spin(self) -> None:
        while self._running.is_set() and self._rclpy.ok():  # pragma: no cover - background thread
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception:
                break

    def publish(self, payload: str) -> None:
        if not self._running.is_set():
            return
        msg = self._String()
        msg.data = payload
        with self._lock:
            self._publisher.publish(msg)

    def close(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        try:
            self._executor.call_soon_threadsafe(lambda: None)
        except Exception:
            pass

        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.5)

        with self._lock:
            try:
                self._executor.remove_node(self._node)
            except Exception:
                pass
            try:
                self._node.destroy_publisher(self._publisher)
            except Exception:
                pass
            try:
                self._node.destroy_node()
            except Exception:
                pass

        try:
            self._executor.shutdown()
        except Exception:
            pass

        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


# ------------------ Telemetry collector ------------------

class TelemetryCollector:
    """
    Subscribes to every available MAVSDK telemetry stream and forwards updates.
    """

    # (alias, attribute_name)
    CANDIDATE_STREAMS: Tuple[Tuple[str, str], ...] = (
        ("position", "position"),
        ("home", "home"),
        ("position_velocity_ned", "position_velocity_ned"),
        ("velocity_ned", "velocity_ned"),
        ("ground_speed_ned", "ground_speed_ned"),
        ("attitude_euler", "attitude_euler"),
        ("attitude_quaternion", "attitude_quaternion"),
        ("angular_velocity_body", "angular_velocity_body"),
        ("linear_acceleration_body", "linear_acceleration_body"),
        ("imu", "imu"),
        ("scaled_imu", "scaled_imu"),
        ("raw_imu", "raw_imu"),
        ("magnetometer", "magnetometer"),
        ("gps_info", "gps_info"),
        ("gps_raw", "raw_gps"),
        ("health", "health"),
        ("health_all_ok", "health_all_ok"),
        ("battery", "battery"),
        ("flight_mode", "flight_mode"),
        ("status_text", "status_text"),
        ("unix_epoch_time", "unix_epoch_time"),
        ("distance_sensor", "distance_sensor"),
        ("landed_state", "landed_state"),
        ("in_air", "in_air"),
        ("odometry", "odometry"),
        ("fixedwing_metrics", "fixedwing_metrics"),
        ("actuator_control_target", "actuator_control_target"),
        ("actuator_output_status", "actuator_output_status"),
        ("rc_status", "rc_status"),
        ("airspeed", "airspeed"),
        ("heading", "heading"),
        ("acceleration_ned", "acceleration_ned"),
        ("camera_attitude_euler", "camera_attitude_euler"),
        ("camera_attitude_quaternion", "camera_attitude_quaternion"),
    )

    def __init__(self, system: System, on_stream_update: Callable[[str, Dict[str, Any]], None]):
        self._system = system
        self._on_stream_update = on_stream_update
        self._tasks: List[asyncio.Task] = []
        self._generators: List[Any] = []
        self._running = False
        self._active_streams: List[str] = []

    @property
    def active_streams(self) -> List[str]:
        return list(self._active_streams)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks.clear()
        self._generators.clear()
        self._active_streams.clear()

        for alias, attr_name in self.CANDIDATE_STREAMS:
            attr = getattr(self._system.telemetry, attr_name, None)
            if attr is None or not callable(attr):
                continue
            try:
                generator = attr()
            except TypeError:
                continue
            except Exception as exc:
                print(f"[WARN] Telemetry stream '{attr_name}' failed to start: {exc}")
                continue

            task = asyncio.create_task(self._consume_stream(alias, generator))
            self._tasks.append(task)
            self._generators.append(generator)
            self._active_streams.append(alias)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for gen in self._generators:
            try:
                await gen.aclose()  # type: ignore[attr-defined]
            except AttributeError:
                continue
            except Exception:
                continue
        self._tasks.clear()
        self._generators.clear()
        self._active_streams.clear()

    async def _consume_stream(self, name: str, generator: Any) -> None:
        try:
            async for message in generator:
                normalized = self._message_to_dict(message)
                payload = {
                    "timestamp": time.time(),
                    "data": normalized,
                }
                self._on_stream_update(name, payload)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"[WARN] Telemetry stream '{name}' terminated: {exc}")

    @classmethod
    def _message_to_dict(cls, message: Any) -> Any:
        return cls._normalize(message)

    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple, set)):
            return [cls._normalize(v) for v in value]
        if isinstance(value, dict):
            return {str(k): cls._normalize(v) for k, v in value.items()}
        if hasattr(value, "_asdict"):
            return {k: cls._normalize(v) for k, v in value._asdict().items()}
        if hasattr(value, "__slots__"):
            result: Dict[str, Any] = {}
            for slot in value.__slots__:
                if slot.startswith("_"):
                    continue
                try:
                    slot_value = getattr(value, slot)
                except AttributeError:
                    continue
                result[slot] = cls._normalize(slot_value)
            if result:
                return result
        attrs: Dict[str, Any] = {}
        if hasattr(value, "__dict__"):
            for key, val in value.__dict__.items():
                if key.startswith("_"):
                    continue
                attrs[key] = cls._normalize(val)
            if attrs:
                return attrs
        # Fallback: inspect dir for simple attributes
        dynamic: Dict[str, Any] = {}
        for attr in dir(value):
            if attr.startswith("_"):
                continue
            try:
                attr_val = getattr(value, attr)
            except AttributeError:
                continue
            if callable(attr_val):
                continue
            dynamic[attr] = cls._normalize(attr_val)
        if dynamic:
            return dynamic
        return str(value)

    def inject_sample(self, stream_name: str, payload: Dict[str, Any]) -> None:
        """
        Helper for tests: bypass MAVSDK and push a telemetry sample directly.
        """
        self._on_stream_update(stream_name, {"timestamp": time.time(), "data": payload})


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
        self.on_state_update: Optional[Callable[[np.ndarray, np.ndarray], None]] = None

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
        if self.on_state_update is not None:
            self.on_state_update(self.position_xyz.copy(), self.velocity_xyz.copy())

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
        segment_timeout_sec: float = 25.0,
        ros2_enabled: bool = False,
        ros2_topic: str = "/magpie/raw_telemetry",
        ros2_node_name: str = "magpie_raw_stream",
        ros2_qos_depth: int = 10,
        ros2_publisher_factory: Optional[Callable[[str, str, int], _BaseStatePublisher]] = None,
    ):
        self.system_address = system_address
        self.env = MagpieEnv(control_rate_hz=control_rate_hz)
        self.control_dt = self.env.control_dt
        self.default_interpolation_cls = default_interpolation
        self.max_speed = float(max_speed_m_s)
        self.use_velocity_command = bool(use_velocity_command)
        self.env.on_state_update = self._on_env_state_update

        self._waypoint_queue: Deque[Waypoint] = deque()
        self._current_yaw = 0.0
        self._mission_started = False
        
        self._segment_timeout_sec = float(segment_timeout_sec)

        # ---- ROS2 + telemetry ----
        self.ros2_enabled = bool(ros2_enabled)
        self.ros2_topic = ros2_topic
        self._ros2_node_name = ros2_node_name
        self._ros2_qos_depth = int(ros2_qos_depth)
        self._ros2_factory = ros2_publisher_factory
        self._ros2_publisher: Optional[_BaseStatePublisher] = None
        self._last_snapshot: Dict[str, Any] = {}
        self._latest_telemetry: Dict[str, Dict[str, Any]] = {}
        self._telemetry_streams_started: List[str] = []
        self._telemetry_collector: Optional[TelemetryCollector] = None

        # ---- logging state ----
        self.log_enabled = bool(log_enabled)
        self.log_path = log_path or "flight_log.png"
        self._mission_start = 0.0
        self.telemetry_log: List[Dict[str, np.ndarray]] = []
        self.goal_history: List[np.ndarray] = []

        self._setup_ros2_publisher()
        self._publish_state_snapshot(trigger="init")
    
    @staticmethod
    def _movement_intersects_sphere(a: np.ndarray, b: np.ndarray, center: np.ndarray, radius: float) -> bool:
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            return float(np.linalg.norm(a - center)) <= radius
        t = float(np.clip(np.dot(center - a, ab) / ab2, 0.0, 1.0))
        closest = a + t * ab
        return float(np.linalg.norm(closest - center)) <= radius

    # ---------- telemetry + ROS helpers ----------

    def _setup_ros2_publisher(self) -> None:
        if not self.ros2_enabled or self._ros2_publisher is not None:
            return
        factory = self._ros2_factory or _Ros2StatePublisher
        try:
            self._ros2_publisher = factory(self.ros2_topic, self._ros2_node_name, self._ros2_qos_depth)
        except Exception as exc:
            print(f"[WARN] Unable to initialize ROS2 publisher: {exc}")
            self._ros2_publisher = None

    def _teardown_ros2_publisher(self) -> None:
        if self._ros2_publisher is None:
            return
        try:
            self._ros2_publisher.close()
        except Exception:
            pass
        finally:
            self._ros2_publisher = None

    async def _start_telemetry_collector(self) -> None:
        if self._telemetry_collector is None:
            self._telemetry_collector = TelemetryCollector(self.env.drone, self._handle_telemetry_update)
        await self._telemetry_collector.start()
        self._telemetry_streams_started = self._telemetry_collector.active_streams
        self._publish_state_snapshot(trigger="telemetry_streams_started", extra={"streams": self._telemetry_streams_started})

    async def _stop_telemetry_collector(self) -> None:
        if self._telemetry_collector is None:
            return
        await self._telemetry_collector.stop()
        self._telemetry_streams_started = []

    def _on_env_state_update(self, position_xyz: np.ndarray, velocity_xyz: np.ndarray) -> None:
        data = {
            "position_xyz": position_xyz.tolist(),
            "velocity_xyz": velocity_xyz.tolist(),
        }
        self._handle_telemetry_update("magpie_env_local_state", {"timestamp": time.time(), "data": data})

    def _handle_telemetry_update(self, stream_name: str, payload: Dict[str, Any]) -> None:
        self._latest_telemetry[stream_name] = payload
        self._publish_state_snapshot(
            trigger=f"telemetry_update:{stream_name}",
            extra={"stream": stream_name, "payload": payload["data"]},
        )

    @staticmethod
    def _snapshot_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return [DroneAPI._snapshot_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): DroneAPI._snapshot_value(v) for k, v in value.items()}
        if hasattr(value, "__slots__"):
            out: Dict[str, Any] = {}
            for slot in value.__slots__:
                if slot.startswith("_"):
                    continue
                try:
                    out[slot] = DroneAPI._snapshot_value(getattr(value, slot))
                except AttributeError:
                    continue
            if out:
                return out
        if hasattr(value, "__dict__"):
            return {k: DroneAPI._snapshot_value(v) for k, v in value.__dict__.items() if not k.startswith("_")}
        return str(value)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (Enum,)):
            return value.name
        return DroneAPI._snapshot_value(value)

    def _collect_state_snapshot(self, trigger: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = time.time()
        env = self.env
        queue_serialized = [self._snapshot_value(wp) for wp in self._waypoint_queue]

        telemetry_buffer = [
            {key: self._snapshot_value(val) for key, val in sample.items()}
            for sample in self.telemetry_log
        ]

        snapshot: Dict[str, Any] = {
            "timestamp": now,
            "trigger": trigger,
            "system_address": self.system_address,
            "mission": {
                "started": self._mission_started,
                "current_yaw_deg": self._current_yaw,
                "segment_timeout_sec": self._segment_timeout_sec,
                "max_speed_m_s": self.max_speed,
                "use_velocity_command": self.use_velocity_command,
                "default_interpolation": self.default_interpolation_cls.__name__,
                "waypoint_queue_length": len(self._waypoint_queue),
            },
            "environment": {
                "control_rate_hz": env.control_rate_hz,
                "control_dt": env.control_dt,
                "offset_ready": env.offset_ready,
                "offset_xyz": self._snapshot_value(env.offset_xyz),
                "position_xyz": self._snapshot_value(env.position_xyz),
                "velocity_xyz": self._snapshot_value(env.velocity_xyz),
            },
            "waypoints": queue_serialized,
            "goal_history": [self._snapshot_value(goal) for goal in self.goal_history],
            "logging": {
                "enabled": self.log_enabled,
                "log_path": self.log_path,
                "buffer_length": len(self.telemetry_log),
                "telemetry_samples": telemetry_buffer,
            },
            "telemetry_raw": {
                "streams_active": list(self._telemetry_streams_started),
                "streams": {
                    name: {
                        "timestamp": data.get("timestamp"),
                        "data": self._snapshot_value(data.get("data")),
                    }
                    for name, data in self._latest_telemetry.items()
                },
            },
        }
        if extra is not None:
            snapshot["extra"] = extra
        return snapshot

    def _publish_state_snapshot(self, *, trigger: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        snapshot = self._collect_state_snapshot(trigger, extra)
        self._last_snapshot = snapshot
        if self._ros2_publisher is not None:
            try:
                payload = json.dumps(snapshot, default=self._json_default, sort_keys=False)
                self._ros2_publisher.publish(payload)
            except Exception as exc:
                print(f"[WARN] Failed to publish ROS2 snapshot: {exc}")
        return snapshot

    def publish_state_snapshot(self, trigger: str = "manual", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Public helper to push the current snapshot (used by test/demo scripts)."""
        return self._publish_state_snapshot(trigger=trigger, extra=extra)

    def get_last_state_snapshot(self) -> Dict[str, Any]:
        return self._last_snapshot.copy()

    def ingest_telemetry_sample(self, stream_name: str, payload: Dict[str, Any]) -> None:
        """
        Testing hook: push a telemetry payload directly (bypasses MAVSDK collectors).
        """
        self._handle_telemetry_update(stream_name, {"timestamp": time.time(), "data": payload})

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

        await self._start_telemetry_collector()

        await self.env.turn_off_offboard()
        await self.env.arm()
        await self.env.takeoff_to_altitude(altitude=initial_altitude, yaw=yaw)
        self._current_yaw = float(yaw)

        if self.log_enabled:
            self.start_logging()

        self._mission_started = True
        self._publish_state_snapshot(trigger="begin_mission_complete")

    async def end_mission(self) -> None:
        await self.env.simple_land()
        await asyncio.sleep(8.0)
        await self.env.turn_off_offboard()
        if self.log_enabled:
            self.save_flight_plot(self.log_path)
        self._publish_state_snapshot(trigger="end_mission_complete")

    async def shutdown(self) -> None:
        await self.env.turn_off_offboard()
        await self.env.disarm()
        await self._stop_telemetry_collector()
        self._publish_state_snapshot(trigger="shutdown")
        self._teardown_ros2_publisher()

    # ---------- queue management ----------

    def enqueue_waypoint(
        self, x: float, y: float, z: float, yaw: float = 0.0,
        interpolation: Optional[Type[BaseInterpolation]] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self._waypoint_queue.append(
            Waypoint(
                x=float(x), y=float(y), z=float(z), yaw=float(yaw),   # <- fixed
                interpolation=interpolation,
                threshold=float(threshold) if threshold is not None else 0.15,
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

        prev_pos = start_pos.copy()
        t0 = time.time()
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
            self._current_yaw = yaw_cmd
            await asyncio.sleep(self.control_dt)

            # log a sample each tick
            if self.log_enabled:
                self.telemetry_log.append(self._make_log_sample(target_local=target_local))

            # --------- robust completion guards ----------
            # 1) if I'm simply inside the sphere, count it as done
            if not out.done:
                if float(np.linalg.norm(ctx.now_state_pos - target_local)) <= max(wp.threshold, 1e-6):
                    out.done = True

            # 2) movement segment intersected the sphere (line-through-sphere)
            if not out.done:
                dyn_tol = float(wp.threshold + 0.8 * float(np.linalg.norm(self.env.velocity_xyz)) * self.control_dt)
                if self._movement_intersects_sphere(prev_pos, ctx.now_state_pos, target_local, dyn_tol):
                    out.done = True

            # 3) watchdog
            if not out.done and (time.time() - t0) > self._segment_timeout_sec:
                print(f"-- Segment watchdog tripped at {self._segment_timeout_sec:.1f}s; forcing completion.")
                out.done = True

            prev_pos = ctx.now_state_pos.copy()
            # ---------------------------------------------

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
        self._publish_state_snapshot(trigger="start_logging")

    def end_logging(self) -> None:
        self.telemetry_log.clear()
        self.goal_history.clear()
        self._mission_start = 0.0
        self._publish_state_snapshot(trigger="end_logging")

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
        self._publish_state_snapshot(trigger="save_flight_plot", extra={"path": path})
        return path
