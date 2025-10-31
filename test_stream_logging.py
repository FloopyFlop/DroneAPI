# test_stream_logging.py
"""
Live ROS 2 listener for DroneAPI telemetry snapshots.

Run this script in parallel with `test_demo.py`. While the demo mission executes,
this node subscribes to the JSON snapshots that DroneAPI publishes and prints
stream updates to the console so you can monitor the raw telemetry feed.
"""

import json
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# Match the publishing settings in test_demo.py
STREAM_TOPIC = "/magpie/test_demo/raw"
NODE_NAME = "magpie_telemetry_listener"
QOS_DEPTH = 10

# Tweak these to control console verbosity
PRINT_FULL_JSON = False      # set True to dump the complete snapshot payload
PRINT_STREAM_SUMMARY = True  # keep True for readable per-stream updates


class TelemetryListener(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self._subscription = self.create_subscription(
            String,
            STREAM_TOPIC,
            self._handle_snapshot,
            QOS_DEPTH,
        )
        self._last_trigger: Optional[str] = None
        self._message_count = 0
        self.get_logger().info(f"Listening on topic '{STREAM_TOPIC}' (qos_depth={QOS_DEPTH})")

    def _handle_snapshot(self, msg: String) -> None:
        self._message_count += 1
        try:
            snapshot = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received non-JSON payload; printing raw message.")
            print(msg.data)
            return

        trigger = snapshot.get("trigger", "unknown")
        telemetry = snapshot.get("telemetry_raw", {})
        extra = snapshot.get("extra", {})

        if PRINT_FULL_JSON:
            print(json.dumps(snapshot, indent=2, sort_keys=True))
            return

        summary = self._summarize_snapshot(trigger, telemetry, extra)
        print(summary)

    def _summarize_snapshot(self, trigger: str, telemetry: Dict[str, Any], extra: Dict[str, Any]) -> str:
        streams_active = telemetry.get("streams_active", [])
        total_streams = len(telemetry.get("streams", {}))

        if PRINT_STREAM_SUMMARY:
            stream_hint = extra.get("stream") if isinstance(extra, dict) else None
            if stream_hint is None and telemetry.get("streams"):
                stream_hint = next(iter(telemetry["streams"].keys()))

            summary = (
                f"[{self._message_count:06d}] trigger={trigger} "
                f"active_streams={len(streams_active)} total_streams={total_streams}"
            )
            if stream_hint:
                stream_payload = telemetry["streams"].get(stream_hint, {})
                data = stream_payload.get("data", {})
                summary += f" stream='{stream_hint}' keys={list(data)[:5]}"
            return summary

        return f"[{self._message_count:06d}] trigger={trigger} streams={total_streams}"


def main() -> None:
    rclpy.init()
    node = TelemetryListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Telemetry listener interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
