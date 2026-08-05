from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from core.input.base import (
    BaseTeleopInputProvider,
    MotionIntent,
    TeleopInputSample,
    _finite_pose_matrix,
    _make_identity_pose,
)
from core.input.xr_input_types import TeleData


def _quat_xyzw_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    quat = np.asarray([x, y, z, w], dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-8 or not np.all(np.isfinite(quat)):
        raise ValueError("invalid quaternion")
    x, y, z, w = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _pose_stamped_to_matrix(msg) -> np.ndarray:
    pose = msg.pose
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    matrix[:3, :3] = _quat_xyzw_to_matrix(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("pose contains NaN or Inf")
    return matrix


def _rotation_from_flat(values) -> np.ndarray:
    rot = np.asarray(values, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(rot)):
        raise ValueError("vive_to_robot_rotation contains NaN or Inf")
    return rot


def default_vive_calibration_file() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "xr_teleoperate" / "vive_calibration.json"


def vive_config_from_args(args):
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    config = {
        "rotation_robot_from_vive": getattr(args, "vive_rotation_robot_from_vive", identity),
        "offset_xyz": getattr(args, "vive_offset_xyz", [0.0, 0.0, 0.0]),
        "left_mount_rotation": getattr(args, "vive_left_mount_rotation", identity),
        "right_mount_rotation": getattr(args, "vive_right_mount_rotation", identity),
        "position_scale": getattr(args, "vive_position_scale", 1.0),
    }
    calibration_file = str(getattr(args, "vive_calibration_file", "") or "").strip()
    default_file = default_vive_calibration_file()
    if not calibration_file and default_file.is_file():
        calibration_file = str(default_file)
    if not calibration_file:
        return config
    data = json.loads(Path(calibration_file).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("VIVE calibration file must contain a JSON object")
    if "rotation_robot_from_vive" not in data and "vive_rotation_robot_from_vive" in data:
        data["rotation_robot_from_vive"] = data["vive_rotation_robot_from_vive"]
    for key in config:
        if key in data:
            config[key] = data[key]
    return config


class ViveTrackerInputProvider(BaseTeleopInputProvider):
    """VIVE Tracker PoseStamped topics -> robot-base wrist targets.

    VIVE poses are first mapped with the AGX calibration convention, then use
    the same anchored position/orientation delta semantics as the PICO input.
    """

    def __init__(
        self,
        *,
        left_topic: str = "/vive_pose_l",
        right_topic: str = "/vive_pose_r",
        timeout_sec: float = 0.25,
        position_scale: float = 1.0,
        orientation_mode: str = "relative",
        rotation_robot_from_vive=None,
        offset_xyz=None,
        left_mount_rotation=None,
        right_mount_rotation=None,
        enable_left_topic: str = "/vive/enable_left",
        enable_right_topic: str = "/vive/enable_right",
        enable_timeout_sec: float = 0.5,
    ):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Bool

        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self._node = rclpy.create_node("xr_teleoperate_vive_input")
        self._node.create_subscription(PoseStamped, left_topic, self._on_left_pose, 10)
        self._node.create_subscription(PoseStamped, right_topic, self._on_right_pose, 10)
        self._node.create_subscription(Bool, enable_left_topic, self._on_left_enable, 5)
        self._node.create_subscription(Bool, enable_right_topic, self._on_right_enable, 5)
        self._timeout_sec = float(timeout_sec)
        self._enable_timeout_sec = float(enable_timeout_sec)
        if self._enable_timeout_sec <= 0.0:
            raise ValueError("enable_timeout_sec must be positive")
        self._position_scale = float(position_scale)
        self._orientation_mode = str(orientation_mode)
        if self._orientation_mode not in {"absolute", "relative", "neutral"}:
            raise ValueError(f"unsupported vive orientation mode: {self._orientation_mode}")
        if rotation_robot_from_vive is None:
            rotation_robot_from_vive = np.eye(3, dtype=float)
        self._r_robot_vive = _rotation_from_flat(rotation_robot_from_vive)
        self._offset_xyz = np.asarray(offset_xyz if offset_xyz is not None else np.zeros(3), dtype=float).reshape(3)
        self._mount_rotation = {
            "left": _rotation_from_flat(left_mount_rotation if left_mount_rotation is not None else np.eye(3)),
            "right": _rotation_from_flat(right_mount_rotation if right_mount_rotation is not None else np.eye(3)),
        }
        if not np.all(np.isfinite(self._offset_xyz)):
            raise ValueError("vive offset_xyz contains NaN or Inf")

        self._left_pose = None
        self._right_pose = None
        self._left_recv_time = 0.0
        self._right_recv_time = 0.0
        self._left_tracker_anchor = None
        self._right_tracker_anchor = None
        self._left_robot_anchor = None
        self._right_robot_anchor = None
        self._left_enabled = False
        self._right_enabled = False
        self._left_enable_recv_time = 0.0
        self._right_enable_recv_time = 0.0
        self._frame_index = 0

    def _on_left_pose(self, msg) -> None:
        try:
            self._left_pose = _pose_stamped_to_matrix(msg)
            self._left_recv_time = time.monotonic()
        except ValueError:
            self._left_pose = None

    def _on_right_pose(self, msg) -> None:
        try:
            self._right_pose = _pose_stamped_to_matrix(msg)
            self._right_recv_time = time.monotonic()
        except ValueError:
            self._right_pose = None

    def _on_left_enable(self, msg) -> None:
        enabled = bool(msg.data)
        if enabled != self._left_enabled:
            self._clear_anchor("left")
        self._left_enabled = enabled
        self._left_enable_recv_time = time.monotonic()

    def _on_right_enable(self, msg) -> None:
        enabled = bool(msg.data)
        if enabled != self._right_enabled:
            self._clear_anchor("right")
        self._right_enabled = enabled
        self._right_enable_recv_time = time.monotonic()

    def _clear_anchor(self, side: str) -> None:
        if side == "left":
            self._left_tracker_anchor = None
            self._left_robot_anchor = None
        else:
            self._right_tracker_anchor = None
            self._right_robot_anchor = None

    def _fresh_pose(self, side: str, now: float) -> np.ndarray | None:
        pose = self._left_pose if side == "left" else self._right_pose
        recv_time = self._left_recv_time if side == "left" else self._right_recv_time
        if pose is None or now - recv_time > self._timeout_sec:
            self._clear_anchor(side)
            return None
        return pose.copy()

    def _enable_is_fresh(self, side: str, now: float) -> bool:
        enabled = self._left_enabled if side == "left" else self._right_enabled
        recv_time = self._left_enable_recv_time if side == "left" else self._right_enable_recv_time
        return enabled and now - recv_time <= self._enable_timeout_sec

    def _map_tracker_pose(self, side: str, tracker_pose: np.ndarray) -> np.ndarray:
        mapped = np.eye(4, dtype=float)
        mapped[:3, :3] = (
            self._r_robot_vive
            @ tracker_pose[:3, :3]
            @ self._r_robot_vive.T
            @ self._mount_rotation[side]
        )
        mapped[:3, 3] = self._position_scale * (self._r_robot_vive @ tracker_pose[:3, 3]) + self._offset_xyz
        return mapped

    def _target_pose(self, side: str, tracker_pose: np.ndarray, current_robot_pose: np.ndarray) -> np.ndarray:
        mapped_pose = self._map_tracker_pose(side, tracker_pose)
        if side == "left":
            if self._left_tracker_anchor is None:
                self._left_tracker_anchor = mapped_pose.copy()
                self._left_robot_anchor = current_robot_pose.copy()
            tracker_anchor = self._left_tracker_anchor
            robot_anchor = self._left_robot_anchor
        else:
            if self._right_tracker_anchor is None:
                self._right_tracker_anchor = mapped_pose.copy()
                self._right_robot_anchor = current_robot_pose.copy()
            tracker_anchor = self._right_tracker_anchor
            robot_anchor = self._right_robot_anchor

        target = robot_anchor.copy()
        target[:3, 3] = robot_anchor[:3, 3] + mapped_pose[:3, 3] - tracker_anchor[:3, 3]

        if self._orientation_mode == "absolute":
            target[:3, :3] = mapped_pose[:3, :3]
        elif self._orientation_mode == "relative":
            delta_rotation = mapped_pose[:3, :3] @ tracker_anchor[:3, :3].T
            target[:3, :3] = delta_rotation @ robot_anchor[:3, :3]
        return target

    def get_sample(self, *args, **kwargs) -> TeleopInputSample | None:
        del args
        self._rclpy.spin_once(self._node, timeout_sec=0.0)
        now = time.monotonic()

        current_left = _finite_pose_matrix(kwargs.get("current_left_robot_wrist_pose"), "current_left_robot_wrist_pose")
        current_right = _finite_pose_matrix(kwargs.get("current_right_robot_wrist_pose"), "current_right_robot_wrist_pose")
        left_tracker = self._fresh_pose("left", now)
        right_tracker = self._fresh_pose("right", now)
        if left_tracker is None and right_tracker is None:
            return None

        enabled_arms = []
        deadman_enabled_arms = [side for side in ("left", "right") if self._enable_is_fresh(side, now)]
        left_target = current_left.copy()
        right_target = current_right.copy()
        if left_tracker is not None and "left" in deadman_enabled_arms:
            left_target = self._target_pose("left", left_tracker, current_left)
            enabled_arms.append("left")
        else:
            self._clear_anchor("left")
        if right_tracker is not None and "right" in deadman_enabled_arms:
            right_target = self._target_pose("right", right_tracker, current_right)
            enabled_arms.append("right")
        else:
            self._clear_anchor("right")

        tele_data = TeleData(
            head_pose=_make_identity_pose(),
            left_wrist_pose=left_target,
            right_wrist_pose=right_target,
            left_ctrl_squeeze="left" in enabled_arms,
            left_ctrl_squeezeValue=1.0 if "left" in enabled_arms else 0.0,
            left_ctrl_triggerValue=10.0,
            left_ctrl_thumbstickValue=np.zeros(2, dtype=float),
            right_ctrl_squeeze="right" in enabled_arms,
            right_ctrl_squeezeValue=1.0 if "right" in enabled_arms else 0.0,
            right_ctrl_triggerValue=10.0,
            right_ctrl_thumbstickValue=np.zeros(2, dtype=float),
        )
        motion_intent = MotionIntent(
            kind="pose",
            left_wrist_pose=left_target,
            right_wrist_pose=right_target,
            timestamp=now,
            frame_index=self._frame_index,
            source="vive_tracker",
            metadata={
                "enabled_arms": enabled_arms,
                "deadman_enabled_arms": deadman_enabled_arms,
                "keyboard_enabled_arms": deadman_enabled_arms,
                "rotation_robot_from_vive": self._r_robot_vive.tolist(),
                "offset_xyz": self._offset_xyz.tolist(),
            },
        )
        self._frame_index += 1
        return TeleopInputSample(tele_data=tele_data, motion_intent=motion_intent, done=False)

    def has_live_pose_data(self) -> bool:
        now = time.monotonic()
        return self._fresh_pose("left", now) is not None or self._fresh_pose("right", now) is not None

    def sync_reference_to_current_live_pose(self, *args, **kwargs):
        del args, kwargs
        self._clear_anchor("left")
        self._clear_anchor("right")
        return None

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
            self._node = None
