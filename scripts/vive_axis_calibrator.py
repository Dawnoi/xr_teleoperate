#!/usr/bin/env python3
"""Calibrate VIVE lighthouse axes for xr_teleoperate.

Records three VIVE Tracker points in lighthouse coordinates:
  o: robot origin/reference point
  x: point after moving along robot +X
  y: point after moving along robot +Y

Uses the same three-point rotation and origin-offset model as the AGX VIVE
calibrator, then writes a JSON file consumable by --vive-calibration-file.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, name: str) -> np.ndarray:
    n = _norm(v)
    if n <= 1e-9:
        raise ValueError(f"{name} is too short")
    return np.asarray(v, dtype=float) / n


def solve_axis_calibration(
    origin_vive,
    x_vive,
    y_vive,
    *,
    robot_origin_xyz=None,
    min_axis_distance_m=0.15,
    min_sin_angle=0.5,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(origin_vive, dtype=float).reshape(3)
    dx = np.asarray(x_vive, dtype=float).reshape(3) - origin
    dy = np.asarray(y_vive, dtype=float).reshape(3) - origin
    if _norm(dx) < min_axis_distance_m:
        raise ValueError(f"+X sample is too close to origin: {_norm(dx):.3f}m")
    if _norm(dy) < min_axis_distance_m:
        raise ValueError(f"+Y sample is too close to origin: {_norm(dy):.3f}m")
    ex = _unit(dx, "+X")
    ey_raw = _unit(dy, "+Y")
    sin_angle = _norm(np.cross(ex, ey_raw))
    if sin_angle < min_sin_angle:
        raise ValueError(f"+X/+Y samples are too parallel: sin(angle)={sin_angle:.3f}")
    ey = _unit(ey_raw - float(np.dot(ey_raw, ex)) * ex, "+Y orthogonalized")
    ez = _unit(np.cross(ex, ey), "+Z")
    r_vive_from_robot = np.column_stack((ex, ey, ez))
    r_robot_from_vive = r_vive_from_robot.T
    det = float(np.linalg.det(r_robot_from_vive))
    if abs(det - 1.0) > 1e-3:
        raise ValueError(f"invalid rotation det={det:.6f}")
    robot_origin = np.zeros(3, dtype=float) if robot_origin_xyz is None else np.asarray(robot_origin_xyz, dtype=float).reshape(3)
    offset_xyz = robot_origin - r_robot_from_vive @ origin
    return r_robot_from_vive, offset_xyz


class Calibrator(Node):
    def __init__(self, args):
        super().__init__("vive_axis_calibrator")
        self.args = args
        self.samples = deque(maxlen=500)
        self.points = {}
        self.create_subscription(PoseStamped, args.pose_topic, self._on_pose, 50)
        self.get_logger().info(
            "keys: o=origin, x=robot +X point, y=robot +Y point, p=print, s=save, q=quit"
        )
        self.get_logger().info(f"listening: {args.pose_topic}")

    def _on_pose(self, msg: PoseStamped):
        p = msg.pose.position
        self.samples.append((time.monotonic(), np.array([p.x, p.y, p.z], dtype=float)))

    def current_pos(self):
        now = time.monotonic()
        recent = [p for t, p in self.samples if now - t <= self.args.sample_window_sec]
        if not recent:
            return None
        return np.mean(np.vstack(recent), axis=0)

    def record(self, key: str):
        pos = self.current_pos()
        if pos is None:
            self.get_logger().warning("no recent tracker pose")
            return
        self.points[key] = pos
        self.get_logger().info(f"recorded {key}: {pos.tolist()}")

    def solve(self):
        missing = [k for k in ("o", "x", "y") if k not in self.points]
        if missing:
            self.get_logger().warning(f"missing samples: {missing}")
            return None
        try:
            return solve_axis_calibration(
                self.points["o"],
                self.points["x"],
                self.points["y"],
                robot_origin_xyz=self.args.robot_origin_xyz,
                min_axis_distance_m=self.args.min_axis_distance_m,
            )
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return None

    def print_solution(self):
        solution = self.solve()
        if solution is None:
            return
        rot, offset = solution
        flat = [float(f"{v:.10g}") for v in rot.reshape(-1)]
        offset_values = [float(f"{v:.10g}") for v in offset]
        rotation_args = " ".join(str(v) for v in flat)
        offset_args = " ".join(str(v) for v in offset_values)
        print(
            "\n--vive-rotation-robot-from-vive " + rotation_args
            + " --vive-offset-xyz " + offset_args + "\n",
            flush=True,
        )
        self.get_logger().info(f"rotation_robot_from_vive={flat}")
        self.get_logger().info(f"offset_xyz={offset_values}")

    def save(self):
        solution = self.solve()
        if solution is None:
            return
        rot, offset = solution
        flat = [float(f"{v:.10g}") for v in rot.reshape(-1)]
        offset_values = [float(f"{v:.10g}") for v in offset]
        identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        path = Path(self.args.output_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "rotation_robot_from_vive": flat,
                    "offset_xyz": offset_values,
                    "left_mount_rotation": identity,
                    "right_mount_rotation": identity,
                    "position_scale": self.args.position_scale,
                    "output_frame": self.args.output_frame,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.get_logger().info(f"saved: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-topic", default="/vive_pose_l")
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    default_output = config_home / "xr_teleoperate" / "vive_calibration.json"
    parser.add_argument("--output-file", default=str(default_output))
    parser.add_argument("--sample-window-sec", type=float, default=0.25)
    parser.add_argument("--min-axis-distance-m", type=float, default=0.15)
    parser.add_argument("--robot-origin-xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--output-frame", default="base_link")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rclpy.init(args=None)
    node = Calibrator(args)
    if not sys.stdin.isatty():
        node.get_logger().error("interactive terminal required")
        node.destroy_node()
        rclpy.shutdown()
        return 2
    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if not select.select([sys.stdin], [], [], 0.0)[0]:
                continue
            ch = sys.stdin.read(1).lower()
            if ch in ("o", "x", "y"):
                node.record(ch)
            elif ch == "p":
                node.print_solution()
            elif ch == "s":
                node.save()
            elif ch == "q":
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
