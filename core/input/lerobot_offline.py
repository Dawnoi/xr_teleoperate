from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import numpy as np

from core.input.base import (
    ACTION_SIZE,
    ARM_SIZE,
    CHUNK_NAME,
    POSE6_SIZE,
    BaseTeleopInputProvider,
    MotionIntent,
    TeleopInputSample,
    _build_offline_tele_data,
    _make_identity_pose,
    finite_vector,
    pose6_to_matrix,
    split_lerobot_action,
)


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
    from core.input.raw_offline import raw_episode_exists, validate_raw_episode

    if raw_episode_exists(dataset_root, episode_index):
        motion_repr = "pose" if str(arm_source) == "fk_cmd_pose" else "qpos"
        validation = validate_raw_episode(dataset_root, episode_index, arm_source, motion_repr=motion_repr)
        validation["dataset_kind"] = "raw_episode"
        validation["required_columns"] = []
        validation["timestamp_range"] = tuple(float(value) / 1e9 for value in validation["sample_monotonic_ns_range"])
        return validation

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
        "dataset_kind": "lerobot_parquet",
        "parquet_path": parquet_path,
        "frame_count": frame_count,
        "timestamp_range": (float(timestamps[0]), float(timestamps[-1])),
        "required_columns": required_columns,
    }


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
