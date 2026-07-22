from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from core.input.xr_input_types import TeleData


CHUNK_NAME = "chunk-000"
ACTION_SIZE = 16
ARM_SIZE = 14
POSE6_SIZE = 6


def finite_vector(values, expected_len: int, label: str = "vector") -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != expected_len:
        raise ValueError(f"{label} expected length {expected_len}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or Inf")
    return arr.copy()


def _finite_pose_matrix(values, label: str) -> np.ndarray:
    pose = np.asarray(values, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"{label} expected shape (4, 4), got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{label} contains NaN or Inf")
    return pose.copy()


def split_lerobot_action(action_values) -> tuple[np.ndarray, np.ndarray]:
    action = finite_vector(action_values, ACTION_SIZE, "action")
    arm_q = np.concatenate([action[0:7], action[8:15]])
    gripper_q = np.asarray([action[7], action[15]], dtype=float)
    return arm_q, gripper_q


def dex1_q_to_trigger_value(q) -> float:
    q_value = float(q)
    if not np.isfinite(q_value):
        raise ValueError("gripper q contains NaN or Inf")
    trigger = (q_value / 5.4) * 2.0 + 5.0
    return float(np.clip(trigger, 5.0, 7.0))


def dex1_width_m_to_trigger_value(width_m, max_width_m: float = 0.054) -> float:
    width_value = float(width_m)
    if not np.isfinite(width_value):
        raise ValueError("gripper width contains NaN or Inf")
    max_width_value = float(max_width_m)
    if max_width_value <= 0.0 or not np.isfinite(max_width_value):
        raise ValueError("max_width_m must be positive and finite")
    trigger = (np.clip(width_value, 0.0, max_width_value) / max_width_value) * 2.0 + 5.0
    return float(np.clip(trigger, 5.0, 7.0))


def _finite_gripper_width(value: Any, label: str) -> float:
    width = float(value)
    if not np.isfinite(width):
        raise ValueError(f"{label} contains NaN or Inf")
    return width


def pose6_to_matrix(pose6) -> np.ndarray:
    pose = finite_vector(pose6, POSE6_SIZE, "pose6")
    from scipy.spatial.transform import Rotation as R

    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = pose[:3]
    matrix[:3, :3] = R.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
    return matrix


def _make_identity_pose() -> np.ndarray:
    return np.eye(4, dtype=float)


def _build_offline_tele_data(
    left_wrist_pose: np.ndarray,
    right_wrist_pose: np.ndarray,
    gripper_q: np.ndarray | None,
) -> TeleData:
    left_pose = _finite_pose_matrix(left_wrist_pose, "left_wrist_pose")
    right_pose = _finite_pose_matrix(right_wrist_pose, "right_wrist_pose")
    trigger_values = [10.0, 10.0]
    trigger_pressed = False
    if gripper_q is not None:
        q = finite_vector(gripper_q, 2, "gripper_q")
        trigger_values = [dex1_q_to_trigger_value(q[0]), dex1_q_to_trigger_value(q[1])]
        trigger_pressed = True
    return TeleData(
        head_pose=_make_identity_pose(),
        left_wrist_pose=left_pose,
        right_wrist_pose=right_pose,
        left_ctrl_trigger=trigger_pressed,
        left_ctrl_triggerValue=float(trigger_values[0]),
        left_ctrl_squeeze=True,
        left_ctrl_squeezeValue=1.0,
        left_ctrl_thumbstickValue=np.zeros(2, dtype=float),
        right_ctrl_trigger=trigger_pressed,
        right_ctrl_triggerValue=float(trigger_values[1]),
        right_ctrl_squeeze=True,
        right_ctrl_squeezeValue=1.0,
        right_ctrl_thumbstickValue=np.zeros(2, dtype=float),
    )


@dataclass
class BaseCommandIntent:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    z: float = 0.0
    source: str = "none"
    frame_index: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = [self.vx, self.vy, self.wz, self.z]
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError(f"base command intent contains non-finite values: {values}")
        self.vx = float(self.vx)
        self.vy = float(self.vy)
        self.wz = float(self.wz)
        self.z = float(self.z)
        self.source = str(self.source)
        self.frame_index = int(self.frame_index)
        if self.metadata is None:
            self.metadata = {}
        elif isinstance(self.metadata, Mapping):
            self.metadata = dict(self.metadata)
        else:
            raise TypeError("base command metadata must be a mapping or None")


@dataclass
class MotionIntent:
    kind: str
    left_wrist_pose: np.ndarray | None = None
    right_wrist_pose: np.ndarray | None = None
    arm_q: np.ndarray | None = None
    arm_dq: np.ndarray | None = None
    gripper_q: np.ndarray | None = None
    timestamp: float = 0.0
    frame_index: int = 0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed_kinds = {"pose", "joint_position", "joint_velocity"}
        if self.kind not in allowed_kinds:
            raise ValueError(f"unsupported motion intent kind: {self.kind}")

        self.timestamp = float(self.timestamp)
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp contains NaN or Inf")
        self.frame_index = int(self.frame_index)
        self.source = str(self.source)

        if self.metadata is None:
            self.metadata = {}
        elif isinstance(self.metadata, Mapping):
            self.metadata = dict(self.metadata)
        else:
            raise TypeError("metadata must be a mapping or None")

        if self.gripper_q is not None:
            self.gripper_q = finite_vector(self.gripper_q, 2, "gripper_q")

        if self.left_wrist_pose is not None:
            self.left_wrist_pose = _finite_pose_matrix(self.left_wrist_pose, "left_wrist_pose")
        if self.right_wrist_pose is not None:
            self.right_wrist_pose = _finite_pose_matrix(self.right_wrist_pose, "right_wrist_pose")

        if self.kind == "pose":
            if self.left_wrist_pose is None or self.right_wrist_pose is None:
                raise ValueError("pose motion intent requires left_wrist_pose and right_wrist_pose")
        elif self.kind == "joint_position":
            self.arm_q = finite_vector(self.arm_q, ARM_SIZE, "arm_q")
        elif self.kind == "joint_velocity":
            self.arm_dq = finite_vector(self.arm_dq, ARM_SIZE, "arm_dq")


@dataclass
class TeleopInputSample:
    tele_data: TeleData
    motion_intent: MotionIntent
    done: bool = False
    base_intent: BaseCommandIntent | None = None

    def __post_init__(self) -> None:
        if self.base_intent is None or isinstance(self.base_intent, BaseCommandIntent):
            return
        if isinstance(self.base_intent, Mapping):
            self.base_intent = BaseCommandIntent(**dict(self.base_intent))
            return
        raise TypeError("base_intent must be BaseCommandIntent, mapping, or None")


class BaseTeleopInputProvider:
    def get_sample(self, *args, **kwargs) -> TeleopInputSample | None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def calibrate_head_reference(self, *args, **kwargs):
        return None

    def sync_reference_to_current_live_pose(self, *args, **kwargs):
        return None

    def has_live_pose_data(self) -> bool:
        return False

    @property
    def done(self) -> bool:
        return False
