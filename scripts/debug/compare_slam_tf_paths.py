#!/usr/bin/env python3
"""Raw terminal monitor for the TF edges used by mobile-base pose candidates."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass


DEFAULT_EDGES = (
    ("slamware_map", "odom"),
    ("odom", "base_link"),
    ("slamware_map", "robot_pose"),
    ("robot_pose", "base_link"),
    ("slamware_map", "laser"),
)
@dataclass(frozen=True)
class RawTransform:
    parent: str
    child: str
    header_stamp_ns: int
    received_wall_time_ns: int
    received_monotonic_ns: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    yaw: float
    update_count: int


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class RawTfMonitor:
    def __init__(self, *, all_tf: bool):
        self.all_tf = bool(all_tf)
        self._targets = set(DEFAULT_EDGES)
        self._latest: dict[tuple[str, str], RawTransform] = {}
        self._counts: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def on_tf(self, message) -> None:
        received_wall_time_ns = int(time.time_ns())
        received_monotonic_ns = int(time.monotonic_ns())
        with self._lock:
            for transform in message.transforms:
                parent = str(transform.header.frame_id)
                child = str(transform.child_frame_id)
                edge = (parent, child)
                if not self.all_tf and edge not in self._targets:
                    continue
                self._counts[edge] = self._counts.get(edge, 0) + 1
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                qx = float(rotation.x)
                qy = float(rotation.y)
                qz = float(rotation.z)
                qw = float(rotation.w)
                self._latest[edge] = RawTransform(
                    parent=parent,
                    child=child,
                    header_stamp_ns=_stamp_ns(transform.header.stamp),
                    received_wall_time_ns=received_wall_time_ns,
                    received_monotonic_ns=received_monotonic_ns,
                    x=float(translation.x),
                    y=float(translation.y),
                    z=float(translation.z),
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    qw=qw,
                    yaw=_yaw_from_quaternion(qx, qy, qz, qw),
                    update_count=self._counts[edge],
                )

    def snapshot(self) -> tuple[dict[tuple[str, str], RawTransform], dict[tuple[str, str], int]]:
        with self._lock:
            return dict(self._latest), dict(self._counts)


def _format_transform(transform: RawTransform, now_wall_time_ns: int) -> list[str]:
    age_ms = (int(now_wall_time_ns) - transform.header_stamp_ns) / 1e6
    return [
        f"[{transform.parent} -> {transform.child}] updates={transform.update_count} header_age_ms={age_ms:.1f}",
        f"  header_stamp_ns       {transform.header_stamp_ns}",
        f"  received_wall_ns      {transform.received_wall_time_ns}",
        f"  received_monotonic_ns {transform.received_monotonic_ns}",
        f"  translation_xyz       ({transform.x:+.6f}, {transform.y:+.6f}, {transform.z:+.6f}) m",
        f"  quaternion_xyzw       ({transform.qx:+.6f}, {transform.qy:+.6f}, {transform.qz:+.6f}, {transform.qw:+.6f})",
        f"  yaw_rad               {transform.yaw:+.6f}",
    ]


def render_panel(monitor: RawTfMonitor, start_time: float) -> str:
    latest, counts = monitor.snapshot()
    now_wall_time_ns = int(time.time_ns())
    lines = [
        "RAW TF MONITOR",
        "No alignment. No composition. No filtering by timestamp.",
        f"elapsed_sec={time.monotonic() - start_time:.1f} tracked_edges={len(latest)}",
        "",
    ]
    edges = sorted(latest) if monitor.all_tf else DEFAULT_EDGES
    for edge in edges:
        transform = latest.get(edge)
        if transform is None:
            lines.append(f"[{edge[0]} -> {edge[1]}] NOT RECEIVED updates={counts.get(edge, 0)}")
            lines.append("")
            continue
        lines.extend(_format_transform(transform, now_wall_time_ns))
        lines.append("")
    lines.append("Ctrl-C exits. --all-tf displays every transform carried by /tf.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-hz", type=float, default=5.0)
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 means run until Ctrl-C.")
    parser.add_argument("--all-tf", action="store_true", help="Show all /tf transforms, not only the five candidate edges.")
    args = parser.parse_args()
    if args.refresh_hz <= 0.0:
        raise ValueError("--refresh-hz must be positive")
    if args.duration_sec < 0.0:
        raise ValueError("--duration-sec must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    import rclpy
    from rclpy.node import Node
    from tf2_msgs.msg import TFMessage

    stop_event = threading.Event()

    def request_stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    monitor = RawTfMonitor(all_tf=args.all_tf)
    rclpy.init()
    node = Node("raw_slam_tf_monitor")
    node.create_subscription(TFMessage, "/tf", monitor.on_tf, 1000)
    start_time = time.monotonic()
    refresh_period_sec = 1.0 / args.refresh_hz
    next_refresh_time = start_time

    while not stop_event.is_set():
        now = time.monotonic()
        if args.duration_sec > 0.0 and now - start_time >= args.duration_sec:
            break
        rclpy.spin_once(node, timeout_sec=0.02)
        if now < next_refresh_time:
            continue
        sys.stdout.write("\x1b[H\x1b[2J")
        sys.stdout.write(render_panel(monitor, start_time) + "\n")
        sys.stdout.flush()
        next_refresh_time = now + refresh_period_sec

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
