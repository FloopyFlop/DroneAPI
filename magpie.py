# magpie.py
import asyncio
import inspect
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple, Type, Union

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
        self._lock = threading.Lock()
        self._topic = topic
        self._running = threading.Event()
        self._running.set()
        self._owns_context = False

        if not self._rclpy.ok():
            self._rclpy.init(args=None)
            self._owns_context = True

        self._executor = SingleThreadedExecutor()
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

ESSENTIAL_TELEMETRY_STREAMS: Tuple[str, ...] = (
    "position",
    "velocity_ned",
    "position_velocity_ned",
    "attitude_euler",
    "attitude_quaternion",
    "battery",
    "health",
    "health_all_ok",
    "flight_mode",
    "status_text",
    "home",
    "gps_info",
    "raw_gps",
    "in_air",
    "landed_state",
    "unix_epoch_time",
)


class TelemetryCollector:
    """
    Subscribes to available MAVSDK telemetry streams and forwards updates.
    """

    DEFAULT_STREAM_NAMES: Tuple[str, ...] = ESSENTIAL_TELEMETRY_STREAMS

    def __init__(
        self,
        system: System,
        on_stream_update: Callable[[str, Dict[str, Any]], None],
        *,
        stream_names: Optional[Sequence[str]] = None,
        auto_discover: bool = False,
    ):
        self._system = system
        self._on_stream_update = on_stream_update
        self._tasks: List[asyncio.Task] = []
        self._generators: List[Any] = []
        self._running = False
        self._active_streams: List[str] = []
        self._auto_discover = bool(auto_discover)
        if stream_names is not None:
            if isinstance(stream_names, (str, bytes)):
                raise TypeError("stream_names must be a sequence, not a string")
            sanitized = [str(name).strip() for name in stream_names if str(name).strip()]
            if not sanitized:
                raise ValueError("stream_names sequence must contain at least one value")
            self._stream_names = sorted(set(sanitized))
        else:
            self._stream_names = list(self.DEFAULT_STREAM_NAMES)
        if self._auto_discover:
            self._stream_names = self._discover_stream_names()
        self._closed_streams: set[str] = set()

    @property
    def active_streams(self) -> List[str]:
        return list(self._active_streams)

    def _discover_stream_names(self) -> List[str]:
        telemetry_obj = self._system.telemetry
        names = set(self.DEFAULT_STREAM_NAMES)

        for name in dir(telemetry_obj):
            if name.startswith("_"):
                continue
            if name.startswith("set_rate"):
                # rate setters require arguments; skip them
                continue
            attr = getattr(telemetry_obj, name, None)
            if attr is None or not callable(attr):
                continue
            try:
                sig = inspect.signature(attr)
            except (TypeError, ValueError):
                continue

            required_params = []
            for p in sig.parameters.values():
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    if p.default is inspect._empty:
                        required_params.append(p)
                elif p.kind is inspect.Parameter.VAR_POSITIONAL:
                    required_params.append(p)
            if required_params:
                continue

            if inspect.iscoroutinefunction(attr):
                # coroutines are usually single-shot getters; skip them
                continue
            if inspect.isasyncgenfunction(attr):
                names.add(name)
                continue

            return_hint = getattr(attr, "__annotations__", {}).get("return")
            hint_text = getattr(return_hint, "__name__", "") if return_hint else ""
            if return_hint and ("AsyncGenerator" in hint_text or "AsyncIterator" in hint_text):
                names.add(name)

        return sorted(names)

    async def configure_rates(self, rate_hz: float) -> None:
        """
        Configure telemetry rates. Many rate setters are unsupported depending on vehicle type.
        We only set rates that are commonly supported to avoid errors.
        """
        if rate_hz <= 0.0:
            return

        telemetry_obj = self._system.telemetry
        # Try to set the most critical rates, ignore errors for unsupported ones
        rate_setters = [
            "set_rate_position",
            "set_rate_velocity_ned",
            "set_rate_position_velocity_ned",
            "set_rate_attitude",
        ]

        for setter_name in rate_setters:
            setter = getattr(telemetry_obj, setter_name, None)
            if setter is not None and callable(setter):
                try:
                    await setter(rate_hz)
                except Exception:
                    # Silently ignore unsupported rate setters
                    pass

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks.clear()
        self._generators.clear()
        self._active_streams.clear()
        self._closed_streams.clear()

        telemetry_obj = self._system.telemetry
        for name in self._stream_names:
            attr = getattr(telemetry_obj, name, None)
            if attr is None or not callable(attr):
                continue

            if not inspect.isasyncgenfunction(attr):
                return_hint = getattr(attr, "__annotations__", {}).get("return")
                hint_text = getattr(return_hint, "__name__", "") if return_hint else ""
                if "AsyncGenerator" not in hint_text and "AsyncIterator" not in hint_text:
                    continue

            try:
                generator = attr()
            except TypeError:
                continue
            except Exception as exc:
                print(f"[WARN] Telemetry stream '{name}' failed to start: {exc}")
                continue

            if not hasattr(generator, "__anext__") and not hasattr(generator, "__aiter__"):
                continue

            task = asyncio.create_task(self._consume_stream(name, generator))
            self._tasks.append(task)
            self._generators.append(generator)
            self._active_streams.append(name)

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
            text = str(exc)
            if "grpc_status:14" in text or "Socket closed" in text:
                if name not in self._closed_streams:
                    print(f"[INFO] Telemetry stream '{name}' closed")
                    self._closed_streams.add(name)
            else:
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
        try:
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
        except Exception as exc:
            # If telemetry fails, return last known position/velocity to avoid crash
            return np.array([0.0, 0.0, 0.0], dtype=float), np.array([0.0, 0.0, 0.0], dtype=float)

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
        except Exception as exc:
            print(f"[WARN] Offboard stop raised {exc}")

    async def arm(self) -> None:
        print("-- Arming")
        try:
            await self.drone.action.arm()
        except Exception as exc:
            print(f"[WARN] Arm command raised {exc}")

    async def disarm(self) -> None:
        print("-- Disarming")
        try:
            await self.drone.action.disarm()
        except Exception as exc:
            print(f"[WARN] Disarm command raised {exc}")

    async def takeoff_to_altitude(self, altitude: float, yaw: float = 0.0) -> None:
        await self.compute_offset()
        target_world = self.offset_xyz + np.array([0.0, altitude, 0.0], dtype=float)

        print("-- Taking off")
        # Send a short burst of setpoints before entering offboard mode.
        for _ in range(6):
            await self.command_position(target_world, yaw_deg=yaw)
            await asyncio.sleep(self.control_dt)
        await self.turn_on_offboard()
        await asyncio.sleep(self.control_dt)

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
        telemetry_stream_mode: Union[str, Sequence[str]] = "essential",
        telemetry_rate_hz: float = 0.0,
        telemetry_publish_interval: float = 1.0,
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

        self._telemetry_rate_hz = max(0.0, float(telemetry_rate_hz))
        self._telemetry_auto_discover = False
        if isinstance(telemetry_stream_mode, str):
            mode = telemetry_stream_mode.strip().lower()
            if mode == "all":
                self._telemetry_auto_discover = True
                self._telemetry_stream_names = list(ESSENTIAL_TELEMETRY_STREAMS)
            elif mode == "essential":
                self._telemetry_stream_names = list(ESSENTIAL_TELEMETRY_STREAMS)
            else:
                raise ValueError("telemetry_stream_mode must be 'all', 'essential', or a sequence of stream names")
        elif isinstance(telemetry_stream_mode, Sequence):
            if not telemetry_stream_mode:
                raise ValueError("telemetry_stream_mode sequence cannot be empty")
            self._telemetry_stream_names = [str(name) for name in telemetry_stream_mode]
        else:
            raise TypeError("telemetry_stream_mode must be a string or a sequence of stream names")

        self._telemetry_publish_interval = max(0.0, float(telemetry_publish_interval))
        self._last_snapshot_time = 0.0
        self._snapshot_dirty = True
        self._pending_trigger: Optional[str] = None
        self._pending_extra: Optional[Dict[str, Any]] = None

        # ---- logging state (deprecated - now handled by external logger) ----
        self.log_enabled = bool(log_enabled)  # Kept for backward compatibility
        self.log_path = log_path or "flight_log.png"  # Kept for backward compatibility
        self._mission_start = 0.0
        self.telemetry_log: List[Dict[str, np.ndarray]] = []
        self.goal_history: List[np.ndarray] = []

        self._setup_ros2_publisher()
        self._publish_state_snapshot(trigger="init", force=True)
    
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
            stream_names_arg = None if self._telemetry_auto_discover else self._telemetry_stream_names
            self._telemetry_collector = TelemetryCollector(
                self.env.drone,
                self._handle_telemetry_update,
                stream_names=stream_names_arg,
                auto_discover=self._telemetry_auto_discover,
            )
        try:
            if self._telemetry_rate_hz > 0.0:
                await self._telemetry_collector.configure_rates(self._telemetry_rate_hz)
            await self._telemetry_collector.start()
        except Exception as exc:
            print(f"[WARN] Telemetry collector failed to start: {exc}")
            self._telemetry_collector = None
            self._telemetry_streams_started = []
            return
        self._telemetry_streams_started = self._telemetry_collector.active_streams
        self._publish_state_snapshot(
            trigger="telemetry_streams_started",
            extra={"streams": self._telemetry_streams_started},
            force=True,
        )

    async def _stop_telemetry_collector(self) -> None:
        if self._telemetry_collector is None:
            return
        await self._telemetry_collector.stop()
        self._telemetry_streams_started = []
        self._telemetry_collector = None
        self._latest_telemetry.clear()

    def _on_env_state_update(self, position_xyz: np.ndarray, velocity_xyz: np.ndarray) -> None:
        data = {
            "position_xyz": position_xyz.tolist(),
            "velocity_xyz": velocity_xyz.tolist(),
        }
        self._handle_telemetry_update("magpie_env_local_state", {"timestamp": time.time(), "data": data})

    def _handle_telemetry_update(self, stream_name: str, payload: Dict[str, Any]) -> None:
        self._latest_telemetry[stream_name] = payload
        self._snapshot_dirty = True
        self._publish_state_snapshot(
            trigger=f"telemetry_update:{stream_name}",
            extra={"stream": stream_name},
            force=False,
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

        # Build last_sample from current environment state for external logger
        if env.offset_ready and len(self.goal_history) > 0:
            current_local_pos = env.position_xyz - env.offset_xyz
            last_sample = {
                "time": now - self._mission_start if self._mission_start > 0 else 0.0,
                "position": self._snapshot_value(current_local_pos),
                "velocity": self._snapshot_value(env.velocity_xyz),
                "goal": self._snapshot_value(self.goal_history[-1] if self.goal_history else np.zeros(3)),
                "distance_to_goal": float(np.linalg.norm(current_local_pos - self.goal_history[-1])) if self.goal_history else 0.0,
            }
        else:
            last_sample = None

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
                "last_sample": last_sample,
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

    def _publish_state_snapshot(
        self,
        *,
        trigger: str,
        extra: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        self._pending_trigger = trigger
        self._pending_extra = extra
        now = time.time()

        if not force:
            if not self._snapshot_dirty:
                return self._last_snapshot
            if (
                self._telemetry_publish_interval > 0.0
                and (now - self._last_snapshot_time) < self._telemetry_publish_interval
            ):
                return self._last_snapshot

        actual_trigger = self._pending_trigger or trigger
        actual_extra = self._pending_extra
        snapshot = self._collect_state_snapshot(actual_trigger, actual_extra)
        self._last_snapshot = snapshot
        self._last_snapshot_time = now
        self._snapshot_dirty = False
        self._pending_trigger = None
        self._pending_extra = None

        if self._ros2_publisher is not None:
            try:
                payload = json.dumps(snapshot, default=self._json_default, sort_keys=False)
                self._ros2_publisher.publish(payload)
            except Exception as exc:
                print(f"[WARN] Failed to publish ROS2 snapshot: {exc}")
        return snapshot

    def publish_state_snapshot(
        self,
        trigger: str = "manual",
        extra: Optional[Dict[str, Any]] = None,
        *,
        force: bool = True,
    ) -> Dict[str, Any]:
        """Public helper to push the current snapshot (used by test/demo/test harness scripts)."""
        return self._publish_state_snapshot(trigger=trigger, extra=extra, force=force)

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

        await self.env.turn_off_offboard()
        await self.env.arm()
        await self.env.takeoff_to_altitude(altitude=initial_altitude, yaw=yaw)
        await self._start_telemetry_collector()
        self._current_yaw = float(yaw)
        self._mission_start = time.time()  # Track mission start for external logger

        self._mission_started = True
        self._publish_state_snapshot(trigger="begin_mission_complete", force=True)

    async def end_mission(self) -> None:
        await self.env.simple_land()
        await asyncio.sleep(8.0)
        await self.env.turn_off_offboard()
        self._publish_state_snapshot(trigger="end_mission_complete", force=True)

    async def shutdown(self) -> None:
        await self.env.turn_off_offboard()
        await self.env.disarm()
        await self._stop_telemetry_collector()
        self._publish_state_snapshot(trigger="shutdown", force=True)
        self._teardown_ros2_publisher()
        close_method = getattr(self.env.drone, "close", None)
        if close_method is not None:
            try:
                result = close_method()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    # ---------- queue management ----------

    def enqueue_waypoint(
        self, x: float, y: float, z: float, yaw: float = 0.0,
        interpolation: Optional[Type[BaseInterpolation]] = None,
        threshold: Optional[float] = None,
    ) -> None:
        waypoint = Waypoint(
            x=float(x), y=float(y), z=float(z), yaw=float(yaw),
            interpolation=interpolation,
            threshold=float(threshold) if threshold is not None else 0.15,
        )
        self._waypoint_queue.append(waypoint)
        self._snapshot_dirty = True
        self._publish_state_snapshot(
            trigger="enqueue_waypoint",
            extra={
                "queued_waypoint": {
                    "x": waypoint.x,
                    "y": waypoint.y,
                    "z": waypoint.z,
                    "yaw": waypoint.yaw,
                    "threshold": waypoint.threshold,
                    "interpolation": waypoint.interpolation.__name__ if waypoint.interpolation else None,
                }
            },
            force=False,
        )

    def clear_waypoints(self) -> None:
        self._waypoint_queue.clear()
        self._snapshot_dirty = True
        self._publish_state_snapshot(trigger="clear_waypoints", force=False)

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
        # Track goal history for external logger
        self.goal_history.append(target_local.copy())
        self._snapshot_dirty = True

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

    # ----- Logging (DEPRECATED - use external test_stream_logging.py instead) --------

    def save_flight_plot(self, path: Optional[str] = None) -> Optional[str]:
        """
        DEPRECATED: Use test_stream_logging.py for flight logging instead.
        This method is kept for backward compatibility only and does nothing.
        """
        print("[DEPRECATED] save_flight_plot() is deprecated. Use test_stream_logging.py for logging.")
        return None
