from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from teleop.utils.xr_input_types import TeleData


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


def pose6_to_matrix(pose6) -> np.ndarray:
    pose = finite_vector(pose6, POSE6_SIZE, "pose6")
    from scipy.spatial.transform import Rotation as R

    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = pose[:3]
    matrix[:3, :3] = R.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
    return matrix


def episode_parquet_path(dataset_root: str | Path, episode_index: int) -> Path:
    return Path(dataset_root) / "data" / CHUNK_NAME / f"episode_{int(episode_index):06d}.parquet"


def required_lerobot_columns(arm_source: str) -> list[str]:
    source = str(arm_source)
    if source == "action":
        return ["timestamp", "action"]
    if source == "state":
        return ["timestamp", "observation.state"]
    if source == "fk_cmd_pose":
        return [
            "timestamp",
            "observation.fk.cmd.left.gripper_flange",
            "observation.fk.cmd.right.gripper_flange",
        ]
    raise ValueError(f"unsupported arm_source: {source}")


def validate_lerobot_offline_episode(
    dataset_root: str | Path,
    episode_index: int,
    arm_source: str,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet_path = episode_parquet_path(dataset_root, episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(f"episode parquet not found: {parquet_path}")

    required_columns = required_lerobot_columns(arm_source)
    schema_names = set(pq.read_schema(parquet_path).names)
    missing_columns = [column for column in required_columns if column not in schema_names]
    if missing_columns:
        raise KeyError(
            f"missing required column(s) for arm_source={arm_source}: "
            f"{', '.join(missing_columns)} in {parquet_path}"
        )

    table = pq.read_table(parquet_path, columns=required_columns)
    frame_count = table.num_rows
    if frame_count <= 0:
        raise RuntimeError(f"episode has no frames: {parquet_path}")

    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=float).reshape(-1)
    if timestamps.shape[0] != frame_count or not np.all(np.isfinite(timestamps)):
        raise ValueError(f"timestamp column contains invalid values in {parquet_path}")

    rows = table.to_pydict()
    if arm_source == "action":
        for row_index, value in enumerate(rows["action"]):
            finite_vector(value, ACTION_SIZE, f"action[{row_index}]")
    elif arm_source == "state":
        for row_index, value in enumerate(rows["observation.state"]):
            finite_vector(value, ACTION_SIZE, f"observation.state[{row_index}]")
    elif arm_source == "fk_cmd_pose":
        for row_index, value in enumerate(rows["observation.fk.cmd.left.gripper_flange"]):
            finite_vector(value, POSE6_SIZE, f"observation.fk.cmd.left.gripper_flange[{row_index}]")
        for row_index, value in enumerate(rows["observation.fk.cmd.right.gripper_flange"]):
            finite_vector(value, POSE6_SIZE, f"observation.fk.cmd.right.gripper_flange[{row_index}]")

    return {
        "parquet_path": parquet_path,
        "frame_count": frame_count,
        "timestamp_range": (float(timestamps[0]), float(timestamps[-1])),
        "required_columns": required_columns,
    }


def _load_xr_robotics_wrapper():
    from teleop.utils.xr_robotics_wrapper import XRRoboticsWrapper

    return XRRoboticsWrapper


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


def _wrist_poses_from_arm_q(arm_ik, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import pinocchio as pin

    q = finite_vector(arm_q, ARM_SIZE, "arm_q")
    model = arm_ik.reduced_robot.model
    data = arm_ik.reduced_robot.data
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    left_pose = np.eye(4, dtype=float)
    right_pose = np.eye(4, dtype=float)
    left_se3 = data.oMf[arm_ik.L_hand_id]
    right_se3 = data.oMf[arm_ik.R_hand_id]
    left_pose[:3, :3] = left_se3.rotation
    left_pose[:3, 3] = left_se3.translation
    right_pose[:3, :3] = right_se3.rotation
    right_pose[:3, 3] = right_se3.translation
    return left_pose, right_pose


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


class XRTeleopInputProvider(BaseTeleopInputProvider):
    def __init__(self, xr_wrapper):
        self._xr_wrapper = xr_wrapper
        self._frame_index = 0

    def get_sample(self, *args, **kwargs) -> TeleopInputSample | None:
        del args
        tele_data = self._xr_wrapper.get_tele_data(
            current_left_robot_wrist_pose=kwargs.get("current_left_robot_wrist_pose"),
            current_right_robot_wrist_pose=kwargs.get("current_right_robot_wrist_pose"),
        )
        if tele_data is None:
            return None
        sample = TeleopInputSample(
            tele_data=tele_data,
            motion_intent=MotionIntent(
                kind="pose",
                left_wrist_pose=tele_data.left_wrist_pose,
                right_wrist_pose=tele_data.right_wrist_pose,
                timestamp=time.time(),
                frame_index=self._frame_index,
                source="xr",
                metadata={},
            ),
            done=False,
        )
        self._frame_index += 1
        return sample

    def close(self) -> None:
        close_fn = getattr(self._xr_wrapper, "close", None)
        if callable(close_fn):
            close_fn()

    def calibrate_head_reference(self, *args, **kwargs):
        calibrate_fn = getattr(self._xr_wrapper, "calibrate_head_reference", None)
        if callable(calibrate_fn):
            return calibrate_fn(*args, **kwargs)
        return None

    def sync_reference_to_current_live_pose(self, *args, **kwargs):
        sync_fn = getattr(self._xr_wrapper, "sync_reference_to_current_live_pose", None)
        if callable(sync_fn):
            return sync_fn(*args, **kwargs)
        return None

    def has_live_pose_data(self) -> bool:
        has_live_pose_fn = getattr(self._xr_wrapper, "has_live_pose_data", None)
        if callable(has_live_pose_fn):
            return bool(has_live_pose_fn())
        return True

    @property
    def done(self) -> bool:
        return False


class LeRobotOfflineInputProvider(BaseTeleopInputProvider):
    def __init__(
        self,
        dataset_root: str | Path,
        episode_index: int,
        arm_source: str = "action",
        speed_scale: float = 1.0,
        arm_ik=None,
    ):
        self.dataset_root = Path(dataset_root)
        self.episode_index = int(episode_index)
        self.arm_source = str(arm_source)
        self.speed_scale = float(speed_scale)
        self.arm_ik = arm_ik

        allowed_sources = {"action", "state", "fk_cmd_pose"}
        if self.arm_source not in allowed_sources:
            raise ValueError(f"unsupported arm_source: {self.arm_source}")

        self._columns = self._load_episode_columns()
        self._frame_count = len(self._columns["timestamp"])
        self._cursor = 0
        self._done = self._frame_count == 0
        self._first_timestamp = float(self._columns["timestamp"][0]) if self._frame_count else 0.0
        self._start_wall_time: float | None = None

    def _load_episode_columns(self) -> dict[str, list[Any]]:
        import pyarrow.parquet as pq

        parquet_path = episode_parquet_path(self.dataset_root, self.episode_index)
        validate_lerobot_offline_episode(self.dataset_root, self.episode_index, self.arm_source)
        columns = pq.read_table(parquet_path).to_pydict()
        if "frame_index" not in columns:
            columns["frame_index"] = list(range(len(columns["timestamp"])))
        return columns

    def _column_value(self, column_name: str, row_index: int):
        if column_name not in self._columns:
            raise KeyError(f"missing required column: {column_name}")
        return self._columns[column_name][row_index]

    def _gripper_q_for_row(self, row_index: int) -> np.ndarray | None:
        if self.arm_source == "action":
            _, gripper_q = split_lerobot_action(self._column_value("action", row_index))
            return gripper_q
        if self.arm_source == "state":
            state = finite_vector(self._column_value("observation.state", row_index), ACTION_SIZE, "observation.state")
            return np.asarray([state[7], state[15]], dtype=float)
        if "action" in self._columns:
            _, gripper_q = split_lerobot_action(self._column_value("action", row_index))
            return gripper_q
        if "observation.state" in self._columns:
            state = finite_vector(self._column_value("observation.state", row_index), ACTION_SIZE, "observation.state")
            return np.asarray([state[7], state[15]], dtype=float)
        return None

    def _joint_position_motion_intent(self, values, label: str, timestamp: float, frame_index: int) -> MotionIntent:
        vector = finite_vector(values, ACTION_SIZE, label)
        arm_q = np.concatenate([vector[0:7], vector[8:15]])
        gripper_q = np.asarray([vector[7], vector[15]], dtype=float)
        return MotionIntent(
            kind="joint_position",
            arm_q=arm_q,
            gripper_q=gripper_q,
            timestamp=timestamp,
            frame_index=frame_index,
            source=f"lerobot_offline:{self.arm_source}",
            metadata={
                "arm_source": self.arm_source,
                "column": label,
                "episode_index": self.episode_index,
            },
        )

    def _pose_motion_intent(self, row_index: int, timestamp: float, frame_index: int) -> MotionIntent:
        left_pose = pose6_to_matrix(self._column_value("observation.fk.cmd.left.gripper_flange", row_index))
        right_pose = pose6_to_matrix(self._column_value("observation.fk.cmd.right.gripper_flange", row_index))
        return MotionIntent(
            kind="pose",
            left_wrist_pose=left_pose,
            right_wrist_pose=right_pose,
            gripper_q=self._gripper_q_for_row(row_index),
            timestamp=timestamp,
            frame_index=frame_index,
            source="lerobot_offline:fk_cmd_pose",
            metadata={
                "arm_source": self.arm_source,
                "episode_index": self.episode_index,
            },
        )

    def _build_motion_intent(self, row_index: int) -> MotionIntent:
        timestamp = float(self._column_value("timestamp", row_index))
        frame_index = int(self._column_value("frame_index", row_index))

        if self.arm_source == "action":
            return self._joint_position_motion_intent(
                self._column_value("action", row_index),
                "action",
                timestamp,
                frame_index,
            )
        if self.arm_source == "state":
            return self._joint_position_motion_intent(
                self._column_value("observation.state", row_index),
                "observation.state",
                timestamp,
                frame_index,
            )
        return self._pose_motion_intent(row_index, timestamp, frame_index)

    def _tele_data_poses(self, motion_intent: MotionIntent) -> tuple[np.ndarray, np.ndarray]:
        if motion_intent.kind == "pose":
            return motion_intent.left_wrist_pose.copy(), motion_intent.right_wrist_pose.copy()
        if self.arm_ik is not None and motion_intent.arm_q is not None:
            return _wrist_poses_from_arm_q(self.arm_ik, motion_intent.arm_q)
        return _make_identity_pose(), _make_identity_pose()

    def _sleep_until_current_frame(self) -> None:
        if self.speed_scale <= 0 or self._cursor >= self._frame_count:
            return
        if self._start_wall_time is None:
            self._start_wall_time = time.time()
        if self._cursor == 0:
            return
        frame_timestamp = float(self._column_value("timestamp", self._cursor))
        target_elapsed = (frame_timestamp - self._first_timestamp) / self.speed_scale
        sleep_s = self._start_wall_time + target_elapsed - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)

    def get_sample(self, *args, **kwargs) -> TeleopInputSample | None:
        del args, kwargs
        if self._done:
            return None

        self._sleep_until_current_frame()
        row_index = self._cursor
        motion_intent = self._build_motion_intent(row_index)
        left_pose, right_pose = self._tele_data_poses(motion_intent)
        tele_data = _build_offline_tele_data(left_pose, right_pose, motion_intent.gripper_q)

        is_last_sample = row_index == self._frame_count - 1
        self._cursor += 1
        self._done = self._cursor >= self._frame_count
        return TeleopInputSample(
            tele_data=tele_data,
            motion_intent=motion_intent,
            done=is_last_sample,
        )

    def has_live_pose_data(self) -> bool:
        return not self.done

    @property
    def done(self) -> bool:
        return self._done


def create_teleop_input_provider(args, arm_ik=None) -> BaseTeleopInputProvider:
    input_provider = getattr(args, "input_provider", "xr") or "xr"

    if input_provider == "xr":
        xr_wrapper_cls = _load_xr_robotics_wrapper()
        xr_wrapper = xr_wrapper_cls(
            use_hand_tracking=getattr(args, "input_mode", "controller") == "hand",
            head_reference_mode=getattr(args, "head_reference_mode", "calibrated"),
            controller_orientation_mode=getattr(args, "controller_orientation_mode", "neutral"),
            controller_mapping_mode=getattr(args, "controller_mapping_mode", "anchored_safe"),
        )
        return XRTeleopInputProvider(xr_wrapper)

    if input_provider == "lerobot_offline":
        dataset_root = getattr(args, "offline_replay_dataset_root", None)
        if not dataset_root:
            raise ValueError("offline_replay_dataset_root is required for lerobot_offline input provider")
        if not hasattr(args, "offline_replay_episode_index"):
            raise ValueError("offline_replay_episode_index is required for lerobot_offline input provider")
        return LeRobotOfflineInputProvider(
            dataset_root=dataset_root,
            episode_index=getattr(args, "offline_replay_episode_index"),
            arm_source=getattr(args, "offline_replay_arm_source", "action"),
            speed_scale=getattr(args, "offline_replay_speed_scale", 1.0),
            arm_ik=arm_ik,
        )

    raise ValueError(f"unsupported input_provider: {input_provider}")
