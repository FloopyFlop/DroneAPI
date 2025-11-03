# test_stream_logging.py
"""
Live terminal dashboard for DroneAPI telemetry snapshots.

Run this script in parallel with `test_demo.py` (or any publisher using the same
topic). The dashboard refreshes in-place so you can monitor position, velocity,
mission state, and a rotating sample of raw telemetry streams without scrolling.
Press `q` to exit.
"""

import curses
import json
import threading
import time
from typing import Any, Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# Match the publishing settings in test_demo.py
STREAM_TOPIC = "/magpie/test_demo/raw"
NODE_NAME = "magpie_telemetry_listener"
QOS_DEPTH = 10


class TelemetryListener(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self._subscription = self.create_subscription(
            String,
            STREAM_TOPIC,
            self._handle_snapshot,
            QOS_DEPTH,
        )
        self._lock = threading.Lock()
        self._latest_snapshot: Optional[Dict[str, Any]] = None
        self._last_trigger: Optional[str] = None
        self._message_count = 0
        self._last_received = 0.0
        self.get_logger().info(
            f"Listening on topic '{STREAM_TOPIC}' (qos_depth={QOS_DEPTH}); dashboard hotkeys: 'q' to quit."
        )

    def _handle_snapshot(self, msg: String) -> None:
        try:
            snapshot = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received non-JSON payload; printing raw message.")
            print(msg.data)
            return

        with self._lock:
            self._message_count += 1
            self._last_trigger = snapshot.get("trigger", "unknown")
            self._latest_snapshot = snapshot
            self._last_received = time.time()

    def get_state(self) -> Tuple[int, float, Optional[str], Optional[Dict[str, Any]]]:
        with self._lock:
            return (
                self._message_count,
                self._last_received,
                self._last_trigger,
                self._latest_snapshot.copy() if self._latest_snapshot else None,
            )


def _format_list(values: Any, precision: int = 2) -> str:
    if not isinstance(values, list):
        return "-"
    formatted = []
    for value in values[:3]:
        if isinstance(value, (int, float)):
            formatted.append(f"{value:.{precision}f}")
        else:
            formatted.append(str(value))
    if len(values) > 3:
        formatted.append("…")
    return "[" + ", ".join(formatted) + "]"


def _format_value(value: Any, precision: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return _format_list(value, precision=precision)
    if isinstance(value, dict):
        items = ", ".join(f"{k}:{_format_value(v, precision)}" for k, v in list(value.items())[:3])
        if len(value) > 3:
            items += ", …"
        return "{" + items + "}"
    return str(value)


def _render_dashboard(stdscr, listener: TelemetryListener) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)

    while True:
        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            break

        count, last_received, trigger, snapshot = listener.get_state()
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        stdscr.addstr(0, 0, "MAGPIE Telemetry Dashboard — press 'q' to quit")
        stdscr.hline(1, 0, curses.ACS_HLINE, width)

        age = (time.time() - last_received) if last_received else float("inf")
        status_line = (
            f"Messages: {count}  |  Last trigger: {trigger or '-'}  |  "
            f"Age: {age:0.2f}s"
        )
        stdscr.addstr(2, 0, status_line[:width])

        if snapshot is None:
            stdscr.addstr(4, 0, "Waiting for telemetry snapshots...")
            stdscr.refresh()
            time.sleep(0.1)
            continue

        env = snapshot.get("environment", {})
        mission = snapshot.get("mission", {})
        telemetry = snapshot.get("telemetry_raw", {})
        logging_info = snapshot.get("logging", {})

        stdscr.addstr(4, 0, "Mission:")
        stdscr.addstr(
            5,
            2,
            (
                f"Started: {mission.get('started')}  "
                f"Yaw: {_format_value(mission.get('current_yaw_deg'))}°  "
                f"Queue: {mission.get('waypoint_queue_length')}  "
                f"Default interp: {mission.get('default_interpolation')}"
            )[:width - 2],
        )
        stdscr.addstr(
            6,
            2,
            (
                f"Max speed: {_format_value(mission.get('max_speed_m_s'))} m/s  "
                f"Seg timeout: {_format_value(mission.get('segment_timeout_sec'))} s  "
                f"Use velocity cmd: {mission.get('use_velocity_command')}"
            )[:width - 2],
        )

        stdscr.addstr(8, 0, "Environment:")
        stdscr.addstr(9, 2, f"Position XYZ [m]: {_format_list(env.get('position_xyz', []))}"[:width - 2])
        stdscr.addstr(10, 2, f"Velocity XYZ [m/s]: {_format_list(env.get('velocity_xyz', []))}"[:width - 2])
        stdscr.addstr(11, 2, f"Offset XYZ [m]: {_format_list(env.get('offset_xyz', []))}"[:width - 2])

        stdscr.addstr(13, 0, "Logging:")
        stdscr.addstr(
            14,
            2,
            f"Enabled: {logging_info.get('enabled')}  Samples buffered: {logging_info.get('buffer_length')}"[:width - 2],
        )

        streams = telemetry.get("streams", {})
        streams_active = telemetry.get("streams_active", [])

        stdscr.addstr(
            16,
            0,
            (
                f"Telemetry streams — active: {len(streams_active)} / total: {len(streams)} "
                f"(showing up to {max(0, height - 19)} streams)"
            )[:width],
        )
        stdscr.hline(17, 0, curses.ACS_HLINE, width)

        row = 18
        max_rows = height - row - 1
        for name, payload in list(sorted(streams.items(), key=lambda item: item[0]))[:max_rows]:
            data = payload.get("data", {})
            parts = []
            if isinstance(data, dict):
                for key, value in list(data.items())[:4]:
                    parts.append(f"{key}={_format_value(value)}")
            else:
                parts.append(_format_value(data))
            line = f"{name}: " + ", ".join(parts)
            stdscr.addstr(row, 2, line[: width - 4])
            row += 1

        stdscr.refresh()
        time.sleep(0.1)


def main() -> None:
    rclpy.init()
    listener = TelemetryListener()
    spin_thread = threading.Thread(target=rclpy.spin, args=(listener,), daemon=True)
    spin_thread.start()

    try:
        curses.wrapper(_render_dashboard, listener)
    except KeyboardInterrupt:
        listener.get_logger().info("Telemetry dashboard interrupted by user.")
    finally:
        listener.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
