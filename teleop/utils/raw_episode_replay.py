from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from teleop.utils.teleop_input_provider import (
    MotionIntent,
    TeleopInputSample,
    _build_offline_tele_data,
    _finite_pose_matrix,
    dex1_q_to_trigger_value,
    finite_vector,
    pose6_to_matrix,
)


ARM_SIZE = 14
GRIPPER_SIZE = 2


def raw_episode_dir_path(dataset_root: str | Path, episode_index: int) -> Path:
    root = Path(dataset_root)
    if root.is_dir() and root.name.startswith("episode_") and (root / "data.json").is_file():
        return root
    return root / f"episode_{int(episode_index):04d}"


def raw_episode_json_path(dataset_root: str | Path, episode_index: int) -> Path:
    return raw_episode_dir_path(dataset_root, episode_index) / "data.json"


def raw_episode_exists(dataset_root: str | Path, episode_index: int) -> bool:
    return raw_episode_json_path(dataset_root, episode_index).is_file()


def load_raw_episode_items(dataset_root: str | Path, episode_index: int) -> list[dict[str, Any]]:
    data_path = raw_episode_json_path(dataset_root, episode_index)
    if not data_path.is_file():
        raise FileNotFoundError(f"raw episode data.json not found: {data_path}")
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"raw episode data.json must contain top-level data list: {data_path}")
    items = payload["data"]
    if not isinstance(items, list) or not items:
        raise ValueError(f"raw episode data list must be non-empty: {data_path}")
    for row_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"raw episode frame {row_index} must be an object")
    return items


def _mapping_value(container: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"{label} missing {key}")
    return value


def _sample_monotonic_ns(item: Mapping[str, Any], frame_index: int) -> int:
    timestamps = _mapping_value(item, "timestamps", f"frame {frame_index}")
    if "sample_monotonic_ns" not in timestamps:
        raise KeyError(f"frame {frame_index} missing timestamps.sample_monotonic_ns")
    sample_ns = int(timestamps["sample_monotonic_ns"])
    if sample_ns < 0:
        raise ValueError(f"frame {frame_index} timestamps.sample_monotonic_ns must be non-negative")
    return sample_ns


def _qpos_from_item(item: Mapping[str, Any], frame_index: int, source_key: str, group_key: str, expected_len: int) -> np.ndarray:
    source = _mapping_value(item, source_key, f"frame {frame_index}")
    group = _mapping_value(source, group_key, f"frame {frame_index} {source_key}")
    if "qpos" not in group:
        raise KeyError(f"frame {frame_index} missing {source_key}.{group_key}.qpos")
    return finite_vector(group["qpos"], expected_len, f"frame {frame_index} {source_key}.{group_key}.qpos")


def _pose_from_item(item: Mapping[str, Any], frame_index: int, source_key: str, group_key: str) -> np.ndarray:
    source = _mapping_value(item, source_key, f"frame {frame_index}")
    group = _mapping_value(source, group_key, f"frame {frame_index} {source_key}")
    pose_value = group.get("pose")
    if not isinstance(pose_value, Mapping):
        raise KeyError(f"frame {frame_index} missing {source_key}.{group_key}.pose")
    if "matrix4x4" in pose_value:
        return _finite_pose_matrix(pose_value["matrix4x4"], f"frame {frame_index} {source_key}.{group_key}.pose.matrix4x4")
    if "position" not in pose_value or "rpy" not in pose_value:
        raise ValueError(f"frame {frame_index} {source_key}.{group_key}.pose must contain position/rpy or matrix4x4")
    pose6 = np.concatenate(
        [
            finite_vector(pose_value["position"], 3, f"frame {frame_index} {source_key}.{group_key}.pose.position"),
            finite_vector(pose_value["rpy"], 3, f"frame {frame_index} {source_key}.{group_key}.pose.rpy"),
        ]
    )
    return pose6_to_matrix(pose6)


def _validate_optional_colors(dataset_root: str | Path, episode_index: int, item: Mapping[str, Any], frame_index: int) -> None:
    colors = item.get("colors")
    if not isinstance(colors, Mapping):
        raise KeyError(f"frame {frame_index} missing colors")
    episode_dir = raw_episode_dir_path(dataset_root, episode_index)
    for key in ("head", "left_wrist", "right_wrist"):
        value = colors.get(key)
        if not isinstance(value, str) or not value.strip():
            raise KeyError(f"frame {frame_index} missing colors.{key}")
        image_path = episode_dir / value
        if not image_path.is_file():
            raise FileNotFoundError(f"frame {frame_index} colors.{key} image not found: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"frame {frame_index} colors.{key} image could not be decoded: {image_path}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"frame {frame_index} colors.{key} image must decode to HxWx3, got {image.shape}")


def validate_raw_episode(dataset_root: str | Path, episode_index: int, arm_source: str, motion_repr: str = "qpos") -> dict[str, Any]:
    items = load_raw_episode_items(dataset_root, episode_index)
    previous_sample_ns = None
    pose_complete_count = 0

    for frame_index, item in enumerate(items):
        _validate_optional_colors(dataset_root, episode_index, item, frame_index)
        sample_ns = _sample_monotonic_ns(item, frame_index)
        if previous_sample_ns is not None and sample_ns <= previous_sample_ns:
            raise ValueError(
                f"frame {frame_index} timestamps.sample_monotonic_ns is not strictly increasing: "
                f"{sample_ns} <= {previous_sample_ns}"
            )
        previous_sample_ns = sample_ns

        if arm_source == "action":
            _qpos_from_item(item, frame_index, "actions", "left_arm", 7)
            _qpos_from_item(item, frame_index, "actions", "right_arm", 7)
            _qpos_from_item(item, frame_index, "actions", "left_ee", 1)
            _qpos_from_item(item, frame_index, "actions", "right_ee", 1)
        elif arm_source == "state":
            _qpos_from_item(item, frame_index, "states", "left_arm", 7)
            _qpos_from_item(item, frame_index, "states", "right_arm", 7)
            _qpos_from_item(item, frame_index, "states", "left_ee", 1)
            _qpos_from_item(item, frame_index, "states", "right_ee", 1)
        elif arm_source != "fk_cmd_pose":
            raise ValueError(f"unsupported raw arm_source: {arm_source}")

        if motion_repr == "pose":
            _pose_from_item(item, frame_index, "actions", "left_arm")
            _pose_from_item(item, frame_index, "actions", "right_arm")
            _qpos_from_item(item, frame_index, "actions", "left_ee", 1)
            _qpos_from_item(item, frame_index, "actions", "right_ee", 1)
            pose_complete_count += 1

    first_sample_ns = _sample_monotonic_ns(items[0], 0)
    last_sample_ns = _sample_monotonic_ns(items[-1], len(items) - 1)
    return {
        "frame_count": len(items),
        "sample_monotonic_ns_range": (first_sample_ns, last_sample_ns),
        "pose_complete_count": int(pose_complete_count),
        "episode_dir": str(raw_episode_dir_path(dataset_root, episode_index)),
    }


def load_raw_dry_run_summary(dataset_root: str | Path, episode_index: int, arm_source: str) -> dict[str, Any]:
    motion_repr = "pose" if arm_source == "fk_cmd_pose" else "qpos"
    validation = validate_raw_episode(dataset_root, episode_index, arm_source, motion_repr=motion_repr)
    return {
        "frame_count": int(validation["frame_count"]),
        "sample_monotonic_ns_range": validation["sample_monotonic_ns_range"],
        "pose_complete_count": int(validation["pose_complete_count"]),
        "arm_source": str(arm_source),
        "motion_repr": motion_repr,
        "episode_dir": validation["episode_dir"],
    }


class RawEpisodeInputProvider:
    def __init__(
        self,
        dataset_root: str | Path,
        episode_index: int,
        arm_source: str = "action",
        speed_scale: float = 1.0,
        motion_repr: str = "qpos",
    ):
        self.dataset_root = Path(dataset_root)
        self.episode_index = int(episode_index)
        self.arm_source = str(arm_source)
        self.motion_repr = str(motion_repr)
        self.speed_scale = float(speed_scale)
        if self.arm_source not in {"action", "state", "fk_cmd_pose"}:
            raise ValueError(f"unsupported raw arm_source: {self.arm_source}")
        if self.arm_source == "fk_cmd_pose":
            self.motion_repr = "pose"
        if self.motion_repr not in {"qpos", "pose"}:
            raise ValueError(f"unsupported raw motion_repr: {self.motion_repr}")

        self.validation = validate_raw_episode(
            self.dataset_root,
            self.episode_index,
            self.arm_source,
            self.motion_repr,
        )
        self._items = load_raw_episode_items(self.dataset_root, self.episode_index)
        self._cursor = 0
        self._done = len(self._items) == 0
        self._first_sample_ns = _sample_monotonic_ns(self._items[0], 0)
        self._start_wall_time = None

    def _sleep_until_current_frame(self) -> None:
        if self.speed_scale <= 0.0 or self._cursor >= len(self._items):
            return
        if self._start_wall_time is None:
            self._start_wall_time = time.time()
        if self._cursor == 0:
            return
        frame_sample_ns = _sample_monotonic_ns(self._items[self._cursor], self._cursor)
        target_elapsed = (frame_sample_ns - self._first_sample_ns) / 1e9 / self.speed_scale
        sleep_s = self._start_wall_time + target_elapsed - time.time()
        if sleep_s > 0.0:
            time.sleep(sleep_s)

    def _gripper_q(self, item: Mapping[str, Any], frame_index: int) -> np.ndarray:
        source_key = "actions" if self.arm_source == "action" else "states"
        left_q = _qpos_from_item(item, frame_index, source_key, "left_ee", 1)
        right_q = _qpos_from_item(item, frame_index, source_key, "right_ee", 1)
        return finite_vector([left_q[0], right_q[0]], GRIPPER_SIZE, f"frame {frame_index} gripper_q")

    def _joint_position_motion_intent(self, item: Mapping[str, Any], frame_index: int) -> MotionIntent:
        source_key = "actions" if self.arm_source == "action" else "states"
        left_arm_q = _qpos_from_item(item, frame_index, source_key, "left_arm", 7)
        right_arm_q = _qpos_from_item(item, frame_index, source_key, "right_arm", 7)
        arm_q = finite_vector(
            np.concatenate([left_arm_q, right_arm_q]),
            ARM_SIZE,
            f"frame {frame_index} {source_key}_arm_q",
        )
        sample_ns = _sample_monotonic_ns(item, frame_index)
        return MotionIntent(
            kind="joint_position",
            arm_q=arm_q,
            gripper_q=self._gripper_q(item, frame_index),
            timestamp=float(sample_ns) / 1e9,
            frame_index=frame_index,
            source=f"raw_episode:{self.arm_source}:qpos",
            metadata={
                "episode_index": self.episode_index,
                "arm_source": self.arm_source,
                "motion_repr": self.motion_repr,
            },
        )

    def _pose_motion_intent(self, item: Mapping[str, Any], frame_index: int) -> MotionIntent:
        left_pose = _pose_from_item(item, frame_index, "actions", "left_arm")
        right_pose = _pose_from_item(item, frame_index, "actions", "right_arm")
        sample_ns = _sample_monotonic_ns(item, frame_index)
        return MotionIntent(
            kind="pose",
            left_wrist_pose=left_pose,
            right_wrist_pose=right_pose,
            gripper_q=self._gripper_q(item, frame_index),
            timestamp=float(sample_ns) / 1e9,
            frame_index=frame_index,
            source="raw_episode:action:pose",
            metadata={
                "episode_index": self.episode_index,
                "arm_source": self.arm_source,
                "motion_repr": self.motion_repr,
            },
        )

    def get_sample(self, *args, **kwargs) -> TeleopInputSample | None:
        del args, kwargs
        if self._done:
            return None

        self._sleep_until_current_frame()
        frame_index = self._cursor
        item = self._items[frame_index]
        if self.motion_repr == "qpos":
            motion_intent = self._joint_position_motion_intent(item, frame_index)
            tele_data = _build_offline_tele_data(np.eye(4, dtype=float), np.eye(4, dtype=float), motion_intent.gripper_q)
            tele_data.left_ctrl_triggerValue = dex1_q_to_trigger_value(motion_intent.gripper_q[0])
            tele_data.right_ctrl_triggerValue = dex1_q_to_trigger_value(motion_intent.gripper_q[1])
        else:
            motion_intent = self._pose_motion_intent(item, frame_index)
            tele_data = _build_offline_tele_data(
                motion_intent.left_wrist_pose,
                motion_intent.right_wrist_pose,
                motion_intent.gripper_q,
            )

        is_last = frame_index == len(self._items) - 1
        self._cursor += 1
        self._done = self._cursor >= len(self._items)
        return TeleopInputSample(
            tele_data=tele_data,
            motion_intent=motion_intent,
            done=is_last,
        )

    def has_live_pose_data(self) -> bool:
        return not self.done

    @property
    def done(self) -> bool:
        return self._done
