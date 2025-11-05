# test_stream_logging.py
"""
Live terminal dashboard and flight logger for DroneAPI telemetry snapshots.

This script replaces the built-in logging in magpie.py, providing:
- Real-time terminal dashboard with mission status
- Automatic flight plot generation for each test scenario
- Data collection and visualization independent of drone control code

Run this script in parallel with `test_demo.py` (or any publisher using the same
topic). The dashboard refreshes in-place so you can monitor position, velocity,
mission state, and telemetry streams without scrolling.

Press 'q' to exit, 's' to manually save current plot.
"""

import curses
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# Match the publishing settings in test_demo.py
STREAM_TOPIC = "/magpie/test_demo/raw"
NODE_NAME = "magpie_telemetry_listener"
QOS_DEPTH = 10

# Logging configuration
LOG_OUTPUT_DIR = "/media/sf_Testing"
AUTO_SAVE_ON_SCENARIO_CHANGE = True
MIN_SAMPLES_FOR_PLOT = 10


@dataclass
class LogSample:
    """Single telemetry sample for plotting."""
    timestamp: float
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    goal: np.ndarray  # [gx, gy, gz]
    distance_to_goal: float


class FlightLogger:
    """
    Collects telemetry data and generates matplotlib plots similar to magpie.py's built-in logging.
    """

    def __init__(self, output_dir: str = LOG_OUTPUT_DIR):
        self.output_dir = output_dir
        self.samples: deque[LogSample] = deque(maxlen=100000)  # Large buffer for long flights
        self.mission_start_time: Optional[float] = None
        self.current_scenario: Optional[str] = None
        self.current_interpolation: Optional[str] = None
        self._lock = threading.Lock()

    def start_mission(self, scenario: str = "unknown", interpolation: str = "unknown") -> None:
        """Start a new mission logging session."""
        with self._lock:
            self.mission_start_time = time.time()
            self.current_scenario = scenario
            self.current_interpolation = interpolation
            self.samples.clear()

    def add_sample(self, position: List[float], velocity: List[float], goal: List[float]) -> None:
        """Add a telemetry sample."""
        if self.mission_start_time is None:
            return

        with self._lock:
            pos = np.array(position[:3], dtype=float)
            vel = np.array(velocity[:3], dtype=float)
            goal_arr = np.array(goal[:3], dtype=float)
            distance = float(np.linalg.norm(pos - goal_arr))

            sample = LogSample(
                timestamp=time.time() - self.mission_start_time,
                position=pos,
                velocity=vel,
                goal=goal_arr,
                distance_to_goal=distance,
            )
            self.samples.append(sample)

    def get_stats(self) -> Dict[str, Any]:
        """Get current logging statistics."""
        with self._lock:
            return {
                "active": self.mission_start_time is not None,
                "scenario": self.current_scenario,
                "interpolation": self.current_interpolation,
                "sample_count": len(self.samples),
                "duration": time.time() - self.mission_start_time if self.mission_start_time else 0.0,
            }

    def save_plot(self, path: Optional[str] = None) -> Optional[str]:
        """Generate and save a flight plot from collected samples."""
        with self._lock:
            if len(self.samples) < MIN_SAMPLES_FOR_PLOT:
                return None

            # Convert samples to numpy arrays
            times = np.array([s.timestamp for s in self.samples], dtype=float)
            positions = np.stack([s.position for s in self.samples])
            velocities = np.stack([s.velocity for s in self.samples])
            goals = np.stack([s.goal for s in self.samples])
            distances = np.array([s.distance_to_goal for s in self.samples], dtype=float)

            # Generate path if not provided
            if path is None:
                scenario = self.current_scenario or "unknown"
                interp = self.current_interpolation or "unknown"
                path = f"{self.output_dir}/{interp}_{scenario}.png"

        # Everything below doesn't need the lock (matplotlib operations)
        speed = np.linalg.norm(velocities, axis=1)

        # Find goal changes
        goal_change_indices = [0]
        for i in range(1, len(goals)):
            if not np.allclose(goals[i], goals[i - 1], atol=0.01):
                goal_change_indices.append(i)
        goal_change_times = times[goal_change_indices]

        # Cumulative path length
        cumulative_path = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        )

        # Compute accelerations
        diffs = np.diff(times) if len(times) > 1 else np.array([0.05])
        dt_candidate = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.05
        accelerations = np.zeros_like(velocities)
        if len(times) >= 2:
            accelerations[1:] = np.diff(velocities, axis=0) / dt_candidate
            accelerations[0] = accelerations[1]
        accel_magnitude = np.linalg.norm(accelerations, axis=1)

        # Create figure
        scale_factor = 3.0 ** 0.5
        fig = plt.figure(figsize=(12.8 * scale_factor, 4.32 * scale_factor), dpi=400)
        gs = fig.add_gridspec(
            2, 4,
            width_ratios=[1.25, 1.0, 1.0, 1.0],
            height_ratios=[1.0, 1.0],
            hspace=0.32,
            wspace=0.28,
        )

        # Row 0: Path plot
        ax_path = fig.add_subplot(gs[0, 0])
        path_points = np.column_stack((positions[:, 0], positions[:, 2]))
        path_segments = np.concatenate([path_points[:-1, None, :], path_points[1:, None, :]], axis=1)
        time_norm = plt.Normalize(times[0], times[-1])
        lc = LineCollection(path_segments, cmap="viridis", norm=time_norm, linewidth=1.6, alpha=0.95)
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

        # Position plots
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

        # Row 1: Velocity and dynamics
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

        print(f"[FlightLogger] Saved plot to {path}")
        return path

    def clear_samples(self) -> None:
        """Clear all samples (but keep mission active)."""
        with self._lock:
            self.samples.clear()


class TelemetryListener(Node):
    """ROS2 node that listens to telemetry snapshots and updates the logger."""

    def __init__(self, flight_logger: FlightLogger):
        super().__init__(NODE_NAME)
        self.flight_logger = flight_logger
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
        self._last_scenario: Optional[str] = None
        self._last_interpolation: Optional[str] = None
        self.get_logger().info(
            f"Listening on topic '{STREAM_TOPIC}' (qos_depth={QOS_DEPTH}); hotkeys: 'q' quit, 's' save plot"
        )

    def _handle_snapshot(self, msg: String) -> None:
        try:
            snapshot = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received non-JSON payload")
            return

        with self._lock:
            self._message_count += 1
            trigger = snapshot.get("trigger", "unknown")
            self._last_trigger = trigger
            self._latest_snapshot = snapshot
            self._last_received = time.time()

            # Extract position, velocity, goal for logging
            env = snapshot.get("environment", {})
            position = env.get("position_xyz", [0.0, 0.0, 0.0])
            velocity = env.get("velocity_xyz", [0.0, 0.0, 0.0])

            # Get current goal from logging section
            logging_info = snapshot.get("logging", {})
            last_sample = logging_info.get("last_sample", None)
            if last_sample is not None and isinstance(last_sample, dict):
                goal = last_sample.get("goal", [0.0, 0.0, 0.0])
            else:
                # Fallback: try to get goal from goal_history
                goal_history = snapshot.get("goal_history", [])
                goal = goal_history[-1] if goal_history else [0.0, 0.0, 0.0]

            # Get mission info for scenario tracking
            mission = snapshot.get("mission", {})
            current_interp = mission.get("default_interpolation", "unknown")

            # Detect scenario changes by watching interpolation method changes
            if current_interp != "unknown" and self._last_interpolation != current_interp:
                # Interpolation changed - save old plot and start new
                if AUTO_SAVE_ON_SCENARIO_CHANGE and self._last_interpolation is not None:
                    saved_path = self.flight_logger.save_plot()
                    if saved_path:
                        self.get_logger().info(f"Auto-saved plot: {saved_path}")
                    self.flight_logger.clear_samples()
                    self.flight_logger.current_interpolation = current_interp
                self._last_interpolation = current_interp

            # Start logging on mission start
            if trigger == "begin_mission_complete" and self.flight_logger.mission_start_time is None:
                self.flight_logger.start_mission(scenario="auto", interpolation=current_interp)
                self.get_logger().info(f"Started flight logging for {current_interp}")

            # Add sample to logger (only if we have valid data)
            if isinstance(position, list) and isinstance(velocity, list) and isinstance(goal, list):
                if len(position) >= 3 and len(velocity) >= 3 and len(goal) >= 3:
                    self.flight_logger.add_sample(position, velocity, goal)

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
    last_snapshot_timestamp = 0.0

    while True:
        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            break
        elif ch in (ord("s"), ord("S")):
            # Manual save
            listener.flight_logger.save_plot()

        count, last_received, trigger, snapshot = listener.get_state()

        # Only redraw if we have a new snapshot (reduces CPU usage)
        current_timestamp = snapshot.get("timestamp", 0.0) if snapshot else 0.0
        if current_timestamp == last_snapshot_timestamp and snapshot is not None:
            time.sleep(0.02)  # Fast polling for new data
            continue

        last_snapshot_timestamp = current_timestamp
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        stdscr.addstr(0, 0, "MAGPIE Telemetry Dashboard — 'q' quit, 's' save plot")
        stdscr.hline(1, 0, curses.ACS_HLINE, width)

        age = (time.time() - last_received) if last_received else float("inf")
        logger_stats = listener.flight_logger.get_stats()
        status_line = (
            f"Msgs: {count} | Trigger: {trigger or '-'} | Age: {age:0.3f}s | "
            f"Log samples: {logger_stats['sample_count']}"
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
            5, 2,
            (
                f"Started: {mission.get('started')}  "
                f"Yaw: {_format_value(mission.get('current_yaw_deg'))}°  "
                f"Queue: {mission.get('waypoint_queue_length')}  "
                f"Interp: {mission.get('default_interpolation')}"
            )[:width - 2],
        )
        stdscr.addstr(
            6, 2,
            (
                f"Max speed: {_format_value(mission.get('max_speed_m_s'))} m/s  "
                f"Seg timeout: {_format_value(mission.get('segment_timeout_sec'))} s  "
                f"Vel cmd: {mission.get('use_velocity_command')}"
            )[:width - 2],
        )

        stdscr.addstr(8, 0, "Environment:")
        stdscr.addstr(9, 2, f"Position XYZ [m]: {_format_list(env.get('position_xyz', []))}"[:width - 2])
        stdscr.addstr(10, 2, f"Velocity XYZ [m/s]: {_format_list(env.get('velocity_xyz', []))}"[:width - 2])
        stdscr.addstr(11, 2, f"Offset XYZ [m]: {_format_list(env.get('offset_xyz', []))}"[:width - 2])

        stdscr.addstr(13, 0, "Flight Logger:")
        stdscr.addstr(
            14, 2,
            (
                f"Active: {logger_stats['active']} | "
                f"Scenario: {logger_stats['scenario']} | "
                f"Interp: {logger_stats['interpolation']} | "
                f"Samples: {logger_stats['sample_count']} | "
                f"Duration: {logger_stats['duration']:.1f}s"
            )[:width - 2],
        )

        streams = telemetry.get("streams", {})
        streams_active = telemetry.get("streams_active", [])

        stdscr.addstr(
            16, 0,
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
        time.sleep(0.02)  # Fast refresh for smooth updates


def main() -> None:
    rclpy.init()
    flight_logger = FlightLogger(output_dir=LOG_OUTPUT_DIR)
    listener = TelemetryListener(flight_logger)

    spin_thread = threading.Thread(target=rclpy.spin, args=(listener,), daemon=True)
    spin_thread.start()

    try:
        curses.wrapper(_render_dashboard, listener)
    except KeyboardInterrupt:
        listener.get_logger().info("Dashboard interrupted by user.")
    finally:
        # Save final plot on exit
        listener.flight_logger.save_plot()
        listener.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
