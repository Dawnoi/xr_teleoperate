#!/usr/bin/env python3
"""Standalone raw episode_xxxx/data.json to LeRobot v2 exporter.

This file intentionally does not import teleop.*.  It preserves the repository
LeRobot v2 schema, including FK columns, by computing G1 29DoF arm FK from the
URDF with a small NumPy kinematics implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CHUNK_INDEX = 0
CHUNK_NAME = f"chunk-{CHUNK_INDEX:03d}"
CAMERA_SLOTS = ("cam0", "cam1", "cam2")
CAMERA_SOURCE_ALIASES = {
    "cam0": ("head",),
    "cam1": ("left_wrist", "wrist_left"),
    "cam2": ("right_wrist", "wrist_right"),
}
CAMERA_SLOT_PRIMARY_SOURCE = {
    "cam0": "head",
    "cam1": "left_wrist",
    "cam2": "right_wrist",
}
CAMERA_SOURCE_TO_SLOT = {
    alias: slot for slot, aliases in CAMERA_SOURCE_ALIASES.items() for alias in aliases
}
CAMERA_SLOT_TO_SOURCE = {
    "cam0": "head",
    "cam1": "left_wrist/wrist_left",
    "cam2": "right_wrist/wrist_right",
}
DEFAULT_OPENCV_FOURCC = "MJPG"
ALIGNMENT_SIDECAR_SCHEMA_VERSION = 1
CONTROL_SIDECAR_SCHEMA_VERSION = 1

LEFT_ARM_LINK_NAMES = [
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
]
RIGHT_ARM_LINK_NAMES = [
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]
LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ARM_JOINT_LABELS = [f"joint{i}" for i in range(1, 8)]
ARM_GROUPS = ("fb", "cmd")
SIDE_GROUPS = ("left", "right")
STATE_VECTOR_SIZE = 16
ARM_VECTOR_SIZE = 14
EE_VECTOR_SIZE = 1

SLOT_TO_RAW_CAMERA_KEY = {
    "cam0": ("head",),
    "cam1": ("left_wrist", "wrist_left"),
    "cam2": ("right_wrist", "wrist_right"),
}
STATE_ACTION_QPOS = (
    ("states", "left_arm", 7),
    ("states", "right_arm", 7),
    ("states", "left_ee", 1),
    ("states", "right_ee", 1),
    ("actions", "left_arm", 7),
    ("actions", "right_arm", 7),
    ("actions", "left_ee", 1),
    ("actions", "right_ee", 1),
)
POSE_PATHS = (
    ("states", "left_arm", "fb", "left"),
    ("states", "right_arm", "fb", "right"),
    ("actions", "left_arm", "cmd", "left"),
    ("actions", "right_arm", "cmd", "right"),
)


def print_progress(message: str) -> None:
    print(f"[EXPORT_PROGRESS] {message}", flush=True)


def parse_bool_flag(value: Any, label: str) -> bool:
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{label} must be one of 0/1/true/false/yes/no/on/off, got {value!r}")


def should_report_frame(frame_number: int, frame_count: int, frame_progress_every: int) -> bool:
    if frame_count <= 0:
        return False
    if frame_number == frame_count:
        return True
    if frame_progress_every <= 0:
        return False
    return frame_number % frame_progress_every == 0


def finite_vector(values: Any, expected_len: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != int(expected_len):
        raise ValueError(f"{label} expected length {expected_len}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or Inf")
    return arr.copy()


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def safe_abs_ms(delta_ns: Any) -> float | None:
    value = safe_int(delta_ns)
    if value is None:
        return None
    return float(abs(value) / 1e6)


def ms_summary(values_ns: list[int]) -> dict[str, Any]:
    if not values_ns:
        return {"count": 0, "avg_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    values_ms = np.asarray(values_ns, dtype=float) / 1e6
    return {
        "count": int(values_ms.size),
        "avg_ms": float(np.mean(values_ms)),
        "p95_ms": float(np.percentile(values_ms, 95)),
        "max_ms": float(np.max(values_ms)),
    }


def struct_type() -> pa.DataType:
    return pa.struct(
        [
            pa.field("path", pa.string()),
            pa.field("timestamp", pa.float64()),
        ]
    )


def build_feature_spec() -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = OrderedDict()
    features["timestamp"] = {"dtype": "float64", "shape": []}
    features["frame_index"] = {"dtype": "int64", "shape": []}
    features["episode_index"] = {"dtype": "int64", "shape": []}
    features["index"] = {"dtype": "int64", "shape": []}
    features["task_index"] = {"dtype": "int64", "shape": []}
    features["observation.state"] = {"dtype": "list<float64>", "length": STATE_VECTOR_SIZE}
    features["action"] = {"dtype": "list<float64>", "length": STATE_VECTOR_SIZE}

    for group in ARM_GROUPS:
        for side in SIDE_GROUPS:
            for label in ARM_JOINT_LABELS:
                features[f"observation.fk.{group}.{side}.{label}"] = {"dtype": "list<float64>", "length": 6}
            features[f"observation.fk.{group}.{side}.gripper_flange"] = {
                "dtype": "list<float64>",
                "length": 6,
            }

    for slot in CAMERA_SLOTS:
        features[f"observation.images.{slot}"] = {
            "dtype": "struct",
            "fields": {
                "path": "string",
                "timestamp": "float64",
            },
        }
    return features


def build_parquet_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("timestamp", pa.float64()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("observation.state", pa.list_(pa.float64())),
        pa.field("action", pa.list_(pa.float64())),
    ]
    for group in ARM_GROUPS:
        for side in SIDE_GROUPS:
            for label in ARM_JOINT_LABELS:
                fields.append(pa.field(f"observation.fk.{group}.{side}.{label}", pa.list_(pa.float64())))
            fields.append(pa.field(f"observation.fk.{group}.{side}.gripper_flange", pa.list_(pa.float64())))
    for slot in CAMERA_SLOTS:
        fields.append(pa.field(f"observation.images.{slot}", struct_type()))
    return pa.schema(fields)


def pick_parquet_compression() -> str | None:
    if pa.Codec.is_available("zstd"):
        return "zstd"
    if pa.Codec.is_available("snappy"):
        return "snappy"
    return None


def column_values_are_finite(column: pa.ChunkedArray) -> bool:
    for chunk in column.chunks:
        for value in chunk.to_pylist():
            if value is None:
                return False
            if isinstance(value, list):
                arr = np.asarray(value, dtype=float)
                if not np.all(np.isfinite(arr)):
                    return False
            elif isinstance(value, dict):
                for nested_value in value.values():
                    if isinstance(nested_value, (int, float, np.floating, np.integer)):
                        if not np.isfinite(float(nested_value)):
                            return False
            elif isinstance(value, (int, float, np.floating, np.integer)):
                if not np.isfinite(float(value)):
                    return False
    return True


def dataset_relpath(path: Path, dataset_root: Path) -> str:
    return os.path.relpath(path, dataset_root).replace(os.sep, "/")


def json_dump_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def jsonl_dump_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def load_raw_episode_data(episode_dir: Path) -> list[dict[str, Any]]:
    data_path = episode_dir / "data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"data.json not found: {data_path}")
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"data.json must contain a top-level data list: {data_path}")
    data = payload["data"]
    if not isinstance(data, list):
        raise ValueError(f"data field must be a list: {data_path}")
    if not data:
        raise ValueError(f"episode has no samples: {episode_dir}")
    for row_index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{episode_dir.name} data[{row_index}] must be an object")
    return data


def list_raw_episode_dirs(input_task_dir: Path) -> list[Path]:
    if not input_task_dir.is_dir():
        raise NotADirectoryError(f"input task dir not found: {input_task_dir}")
    episode_dirs = sorted(
        path for path in input_task_dir.iterdir() if path.is_dir() and path.name.startswith("episode_")
    )
    if not episode_dirs:
        raise FileNotFoundError(f"no episode_xxxx directories found under {input_task_dir}")
    return episode_dirs


def resolve_color_path(episode_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} image path must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = episode_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"{label} image not found: {path}")
    return path


def pick_color_path(colors: Mapping[str, Any], episode_dir: Path, slot: str, frame_index: int) -> tuple[str, Path]:
    for raw_key in SLOT_TO_RAW_CAMERA_KEY[slot]:
        if raw_key in colors:
            label = f"frame {frame_index} colors.{raw_key}"
            return raw_key, resolve_color_path(episode_dir, colors[raw_key], label)
    expected = "/".join(SLOT_TO_RAW_CAMERA_KEY[slot])
    raise KeyError(f"frame {frame_index} missing camera color for {slot}: expected one of {expected}")


def read_color_image(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{label} image could not be decoded: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{label} image must decode to HxWx3, got shape={image.shape}")
    return np.ascontiguousarray(image)


def validate_qpos(item: Mapping[str, Any], frame_index: int) -> None:
    for source_key, group_key, expected_len in STATE_ACTION_QPOS:
        source = item.get(source_key)
        if not isinstance(source, Mapping):
            raise KeyError(f"frame {frame_index} missing {source_key}")
        group = source.get(group_key)
        if not isinstance(group, Mapping):
            raise KeyError(f"frame {frame_index} missing {source_key}.{group_key}")
        qpos_label = f"{source_key}.{group_key}.qpos"
        if "qpos" not in group:
            raise KeyError(f"frame {frame_index} missing {qpos_label}")
        finite_vector(group["qpos"], expected_len, f"frame {frame_index} {qpos_label}")


def qpos_value(item: Mapping[str, Any], source_key: str, group_key: str, expected_len: int, frame_index: int) -> np.ndarray:
    source = item[source_key]
    group = source[group_key]
    return finite_vector(group["qpos"], expected_len, f"frame {frame_index} {source_key}.{group_key}.qpos")


def state_action_vectors(item: Mapping[str, Any], frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_arm_state = qpos_value(item, "states", "left_arm", 7, frame_index)
    right_arm_state = qpos_value(item, "states", "right_arm", 7, frame_index)
    left_ee_state = qpos_value(item, "states", "left_ee", 1, frame_index)
    right_ee_state = qpos_value(item, "states", "right_ee", 1, frame_index)
    left_arm_action = qpos_value(item, "actions", "left_arm", 7, frame_index)
    right_arm_action = qpos_value(item, "actions", "right_arm", 7, frame_index)
    left_ee_action = qpos_value(item, "actions", "left_ee", 1, frame_index)
    right_ee_action = qpos_value(item, "actions", "right_ee", 1, frame_index)

    state_qpos = np.concatenate([left_arm_state, left_ee_state, right_arm_state, right_ee_state])
    action_qpos = np.concatenate([left_arm_action, left_ee_action, right_arm_action, right_ee_action])
    fk_state_qpos = np.concatenate([left_arm_state, right_arm_state])
    fk_action_qpos = np.concatenate([left_arm_action, right_arm_action])
    return state_qpos, action_qpos, fk_state_qpos, fk_action_qpos


def sample_monotonic_ns(item: Mapping[str, Any], frame_index: int) -> int:
    timestamps = item.get("timestamps")
    if not isinstance(timestamps, Mapping):
        raise KeyError(f"frame {frame_index} missing timestamps")
    if "sample_monotonic_ns" not in timestamps:
        raise KeyError(f"frame {frame_index} missing timestamps.sample_monotonic_ns")
    value = int(timestamps["sample_monotonic_ns"])
    if value < 0:
        raise ValueError(f"frame {frame_index} timestamps.sample_monotonic_ns must be non-negative")
    return value


def pose_value(item: Mapping[str, Any], source_key: str, group_key: str) -> Any:
    source = item.get(source_key)
    if not isinstance(source, Mapping):
        return None
    group = source.get(group_key)
    if not isinstance(group, Mapping):
        return None
    return group.get("pose")


def pose6_from_record(value: Any, label: str) -> list[float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if "position" not in value or "rpy" not in value:
        raise ValueError(f"{label} must contain position and rpy")
    position = finite_vector(value["position"], 3, f"{label}.position")
    rpy = finite_vector(value["rpy"], 3, f"{label}.rpy")
    return [float(v) for v in np.concatenate([position, rpy]).tolist()]


def pose_row_if_complete(
    item: Mapping[str, Any],
    episode_index: int,
    frame_index: int,
    timestamp_s: float,
    sample_ns: int,
) -> dict[str, Any] | None:
    raw_values = []
    for source_key, group_key, _pose_group, _side in POSE_PATHS:
        raw_values.append(pose_value(item, source_key, group_key))
    if any(value is None for value in raw_values):
        return None

    row: dict[str, Any] = {
        "episode_index": int(episode_index),
        "frame_index": int(frame_index),
        "timestamp": float(timestamp_s),
        "sample_monotonic_ns": int(sample_ns),
        "fb": {"left": {}, "right": {}},
        "cmd": {"left": {}, "right": {}},
    }
    for source_key, group_key, pose_group, side in POSE_PATHS:
        label = f"{source_key}.{group_key}.pose"
        row[pose_group][side]["gripper_flange"] = pose6_from_record(
            pose_value(item, source_key, group_key),
            label,
        )
    return row


def validate_raw_episode_with_progress(
    episode_dir: Path,
    raw_items: Sequence[Mapping[str, Any]],
    progress_label: str,
    frame_progress_every: int,
    decode_images: bool,
) -> dict[str, Any]:
    shapes_by_slot: dict[str, tuple[int, int, int]] = {}
    previous_sample_ns = None
    complete_pose_count = 0
    frame_count = len(raw_items)

    for frame_index, item in enumerate(raw_items):
        frame_number = frame_index + 1
        validate_qpos(item, frame_index)
        sample_ns = sample_monotonic_ns(item, frame_index)
        if previous_sample_ns is not None and sample_ns <= previous_sample_ns:
            raise ValueError(
                f"frame {frame_index} timestamps.sample_monotonic_ns is not strictly increasing: "
                f"{sample_ns} <= {previous_sample_ns}"
            )
        previous_sample_ns = sample_ns

        colors = item.get("colors")
        if not isinstance(colors, Mapping):
            raise KeyError(f"frame {frame_index} missing colors")
        for slot in CAMERA_SLOTS:
            raw_key, image_path = pick_color_path(colors, episode_dir, slot, frame_index)
            if decode_images:
                image = read_color_image(image_path, f"frame {frame_index} colors.{raw_key}")
                shape = (int(image.shape[0]), int(image.shape[1]), int(image.shape[2]))
                if slot not in shapes_by_slot:
                    shapes_by_slot[slot] = shape
                elif shapes_by_slot[slot] != shape:
                    raise ValueError(
                        f"camera shape mismatch for {slot} in {episode_dir.name}: "
                        f"expected {shapes_by_slot[slot]}, got {shape} at frame {frame_index}"
                    )

        if pose_row_if_complete(item, 0, frame_index, 0.0, sample_ns) is not None:
            complete_pose_count += 1

        if progress_label and should_report_frame(frame_number, frame_count, frame_progress_every):
            print_progress(f"{progress_label} validate frame {frame_number}/{frame_count}")

    return {
        "frame_count": len(raw_items),
        "camera_shapes": {slot: list(shape) for slot, shape in shapes_by_slot.items()},
        "complete_pose_count": int(complete_pose_count),
    }


def normalize_vector3(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    values = [float(part) for part in text.split()]
    if len(values) != 3:
        raise ValueError(f"expected 3-vector, got {text!r}")
    return np.asarray(values, dtype=float)


def translation_matrix(xyz: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return out


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in np.asarray(rpy, dtype=float).reshape(3)]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    out = translation_matrix(xyz)
    out[:3, :3] = rpy_to_matrix(rpy)
    return out


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("joint axis must be non-zero")
    x, y, z = axis / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one_c = 1.0 - c
    rot = np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )
    out = np.eye(4, dtype=float)
    out[:3, :3] = rot
    return out


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    cos_pitch = math.hypot(float(r[0, 0]), float(r[1, 0]))
    if cos_pitch > 1e-12:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        pitch = math.atan2(float(-r[2, 0]), cos_pitch)
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = math.atan2(float(-r[1, 2]), float(r[1, 1]))
        pitch = math.atan2(float(-r[2, 0]), cos_pitch)
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=float)


def pose6_from_transform(transform: np.ndarray) -> list[float]:
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    translation = transform[:3, 3]
    rpy = matrix_to_rpy(transform[:3, :3])
    return [float(v) for v in np.concatenate([translation, rpy]).tolist()]


@dataclass(frozen=True)
class URDFJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


class StandaloneG1Kinematics:
    def __init__(self, urdf_path: Path | str):
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        self.active_joint_index = {
            name: index for index, name in enumerate([*LEFT_ARM_JOINT_NAMES, *RIGHT_ARM_JOINT_NAMES])
        }
        self.joints_by_parent: dict[str, list[URDFJoint]] = defaultdict(list)
        self._load_urdf()

    def _load_urdf(self) -> None:
        root = ET.parse(self.urdf_path).getroot()
        for joint_elem in root.findall("joint"):
            name = joint_elem.attrib["name"]
            joint_type = joint_elem.attrib.get("type", "fixed")
            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")
            if parent_elem is None or child_elem is None:
                raise ValueError(f"joint {name} is missing parent or child")
            origin_elem = joint_elem.find("origin")
            axis_elem = joint_elem.find("axis")
            xyz = normalize_vector3(origin_elem.attrib.get("xyz") if origin_elem is not None else None, (0.0, 0.0, 0.0))
            rpy = normalize_vector3(origin_elem.attrib.get("rpy") if origin_elem is not None else None, (0.0, 0.0, 0.0))
            axis = normalize_vector3(axis_elem.attrib.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0))
            joint = URDFJoint(
                name=name,
                joint_type=joint_type,
                parent=parent_elem.attrib["link"],
                child=child_elem.attrib["link"],
                xyz=xyz,
                rpy=rpy,
                axis=axis,
            )
            self.joints_by_parent[joint.parent].append(joint)

        missing_active = [name for name in self.active_joint_index if name not in self._all_joint_names()]
        if missing_active:
            raise RuntimeError(f"URDF is missing active arm joints: {missing_active}")

    def _all_joint_names(self) -> set[str]:
        return {joint.name for joints in self.joints_by_parent.values() for joint in joints}

    def _joint_motion(self, joint: URDFJoint, q: np.ndarray) -> np.ndarray:
        if joint.joint_type in ("fixed",):
            return np.eye(4, dtype=float)
        if joint.joint_type in ("revolute", "continuous"):
            q_index = self.active_joint_index.get(joint.name)
            angle = 0.0 if q_index is None else float(q[q_index])
            return axis_angle_matrix(joint.axis, angle)
        raise ValueError(f"unsupported joint type in standalone FK: {joint.name} type={joint.joint_type}")

    def link_transforms(self, arm_qpos: Any) -> dict[str, np.ndarray]:
        q = finite_vector(arm_qpos, ARM_VECTOR_SIZE, "arm_qpos")
        transforms: dict[str, np.ndarray] = {}

        def visit(link_name: str, link_transform: np.ndarray) -> None:
            transforms[link_name] = link_transform
            for joint in self.joints_by_parent.get(link_name, []):
                child_transform = link_transform @ origin_matrix(joint.xyz, joint.rpy) @ self._joint_motion(joint, q)
                visit(joint.child, child_transform)

        visit("pelvis", np.eye(4, dtype=float))
        return transforms

    def compute_fk(self, arm_qpos: Any, group: str | None = None) -> dict[str, list[float]]:
        if group is not None and group not in ARM_GROUPS:
            raise ValueError(f"FK group must be one of {ARM_GROUPS}, got {group!r}")
        transforms = self.link_transforms(arm_qpos)
        left_ee = transforms["left_wrist_yaw_link"] @ translation_matrix(np.asarray([0.05, 0.0, 0.0], dtype=float))
        right_ee = transforms["right_wrist_yaw_link"] @ translation_matrix(np.asarray([0.05, 0.0, 0.0], dtype=float))

        groups = ARM_GROUPS if group is None else (group,)
        out: dict[str, list[float]] = {}
        for fk_group in groups:
            for index, link_name in enumerate(LEFT_ARM_LINK_NAMES, start=1):
                out[f"observation.fk.{fk_group}.left.joint{index}"] = pose6_from_transform(transforms[link_name])
            out[f"observation.fk.{fk_group}.left.gripper_flange"] = pose6_from_transform(left_ee)
            for index, link_name in enumerate(RIGHT_ARM_LINK_NAMES, start=1):
                out[f"observation.fk.{fk_group}.right.joint{index}"] = pose6_from_transform(transforms[link_name])
            out[f"observation.fk.{fk_group}.right.gripper_flange"] = pose6_from_transform(right_ee)
        return out


class OpenCVMP4Writer:
    def __init__(self, path: Path, size: tuple[int, int], fps: float, fourcc: str):
        if len(fourcc) != 4:
            raise ValueError(f"OpenCV fourcc must be exactly 4 characters, got {fourcc!r}")
        self.path = Path(path)
        self.width = int(size[0])
        self.height = int(size[1])
        self.fps = float(fps)
        self.fourcc = str(fourcc)
        self.frame_count = 0
        self.released = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*self.fourcc),
            self.fps,
            (self.width, self.height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter failed to open {self.path} fourcc={self.fourcc}")

    def write(self, frame: np.ndarray) -> None:
        if self.released:
            raise RuntimeError("OpenCV mp4 writer already released")
        frame_arr = np.ascontiguousarray(frame)
        expected_shape = (self.height, self.width, 3)
        if frame_arr.shape != expected_shape:
            raise ValueError(f"OpenCV mp4 writer expected frame shape {expected_shape}, got {frame_arr.shape}")
        if frame_arr.dtype != np.uint8:
            frame_arr = frame_arr.astype(np.uint8, copy=False)
        self.writer.write(frame_arr)
        self.frame_count += 1

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.writer.release()
        if self.frame_count <= 0:
            raise RuntimeError(f"OpenCV mp4 writer released without frames: {self.path}")


def camera_meta_for_slot(camera_timestamps: Mapping[str, Any], slot: str) -> tuple[str, Mapping[str, Any]]:
    for source_name in CAMERA_SOURCE_ALIASES[slot]:
        meta = camera_timestamps.get(source_name)
        if isinstance(meta, Mapping):
            return source_name, meta
    return CAMERA_SLOT_PRIMARY_SOURCE[slot], {}


def camera_alignment_payload(source_name: str, meta: Mapping[str, Any], sample_ns: int) -> dict[str, Any]:
    host_recv_monotonic_ns = safe_int(meta.get("host_recv_monotonic_ns"))
    host_monotonic_ns = safe_int(meta.get("host_monotonic_ns"))
    camera_monotonic_ns = host_recv_monotonic_ns if host_recv_monotonic_ns is not None else host_monotonic_ns
    delta_to_sample_ns = safe_int(meta.get("delta_to_sample_ns"))
    if delta_to_sample_ns is None and camera_monotonic_ns is not None:
        delta_to_sample_ns = int(camera_monotonic_ns - sample_ns)

    shape = meta.get("shape")
    if isinstance(shape, tuple):
        shape = list(shape)
    elif not isinstance(shape, list):
        shape = None

    return {
        "source_name": str(meta.get("camera_name") or source_name),
        "frame_seq": safe_int(meta.get("frame_seq")),
        "host_recv_monotonic_ns": host_recv_monotonic_ns,
        "host_monotonic_ns": host_monotonic_ns,
        "camera_monotonic_ns": camera_monotonic_ns,
        "host_recv_wall_time_ns": safe_int(meta.get("host_recv_wall_time_ns")),
        "host_wall_time_ns": safe_int(meta.get("host_wall_time_ns")),
        "source_monotonic_ns": safe_int(meta.get("source_monotonic_ns")),
        "source_wall_time_ns": safe_int(meta.get("source_wall_time_ns")),
        "align_target_monotonic_ns": safe_int(meta.get("align_target_monotonic_ns")),
        "delta_to_sample_ns": delta_to_sample_ns,
        "abs_delta_to_sample_ms": safe_abs_ms(delta_to_sample_ns),
        "shape": shape,
        "protocol": str(meta.get("protocol")) if meta.get("protocol") is not None else None,
        "transport": str(meta.get("transport")) if meta.get("transport") is not None else None,
        "endpoint": str(meta.get("endpoint")) if meta.get("endpoint") is not None else None,
        "read_latency_ms": safe_float(meta.get("read_latency_ms")),
    }


def alignment_timing_payload(meta: Mapping[str, Any]) -> dict[str, Any]:
    meta = meta if isinstance(meta, Mapping) else {}
    return {
        "host_monotonic_ns": safe_int(meta.get("host_monotonic_ns")),
        "delta_to_sample_ns": safe_int(meta.get("delta_to_sample_ns")),
        "abs_delta_to_sample_ms": safe_abs_ms(meta.get("delta_to_sample_ns")),
        "interpolation_mode": str(meta.get("interpolation_mode")) if meta.get("interpolation_mode") is not None else None,
        "support_source_count": safe_int(meta.get("support_source_count")),
        "support_prev_t_ns": safe_int(meta.get("support_prev_t_ns")),
        "support_next_t_ns": safe_int(meta.get("support_next_t_ns")),
        "support_prev_delta_to_sample_ns": safe_int(meta.get("support_prev_delta_to_sample_ns")),
        "support_next_delta_to_sample_ns": safe_int(meta.get("support_next_delta_to_sample_ns")),
        "support_span_ns": safe_int(meta.get("support_span_ns")),
        "support_max_abs_delta_ns": safe_int(meta.get("support_max_abs_delta_ns")),
        "interp_alpha": safe_float(meta.get("interp_alpha")),
    }


def build_alignment_row(
    timestamps: Mapping[str, Any],
    episode_index: int,
    frame_index: int,
    sample_timestamp_s: float,
    sample_ns: int,
) -> dict[str, Any]:
    camera_timestamps = timestamps.get("camera", {}) if isinstance(timestamps, Mapping) else {}
    if not isinstance(camera_timestamps, Mapping):
        camera_timestamps = {}
    cameras = OrderedDict()
    for slot in CAMERA_SLOTS:
        source_name, meta = camera_meta_for_slot(camera_timestamps, slot)
        cameras[slot] = camera_alignment_payload(source_name, meta, sample_ns)

    return {
        "schema_version": ALIGNMENT_SIDECAR_SCHEMA_VERSION,
        "episode_index": int(episode_index),
        "frame_index": int(frame_index),
        "timestamp": float(sample_timestamp_s),
        "timestamp_s": float(sample_timestamp_s),
        "sample_monotonic_ns": int(sample_ns),
        "sample_wall_time_ns": safe_int(timestamps.get("sample_wall_time_ns")) if isinstance(timestamps, Mapping) else None,
        "teleop_input_perf_counter_ns": safe_int(timestamps.get("teleop_input_perf_counter_ns"))
        if isinstance(timestamps, Mapping)
        else None,
        "primary_camera_name": str(timestamps.get("primary_camera_name"))
        if isinstance(timestamps, Mapping) and timestamps.get("primary_camera_name") is not None
        else None,
        "cameras": cameras,
        "state": alignment_timing_payload(timestamps.get("state", {}) if isinstance(timestamps, Mapping) else {}),
        "action": alignment_timing_payload(timestamps.get("action", {}) if isinstance(timestamps, Mapping) else {}),
    }


def build_control_row(
    control_extras: Mapping[str, Any],
    episode_index: int,
    frame_index: int,
    sample_timestamp_s: float,
    sample_ns: int,
) -> dict[str, Any]:
    if "arm_tauff" not in control_extras:
        raise ValueError("control_extras.arm_tauff is required")
    arm_tauff = finite_vector(control_extras.get("arm_tauff"), ARM_VECTOR_SIZE, "control_extras.arm_tauff")
    return {
        "schema_version": CONTROL_SIDECAR_SCHEMA_VERSION,
        "episode_index": int(episode_index),
        "frame_index": int(frame_index),
        "timestamp": float(sample_timestamp_s),
        "sample_monotonic_ns": int(sample_ns),
        "arm_tauff": [float(value) for value in arm_tauff.tolist()],
        "source": "teleop_runtime",
    }


def build_parquet_arrays(
    samples: Sequence[Mapping[str, Any]],
    episode_index: int,
    global_frame_start: int,
) -> OrderedDict[str, pa.Array]:
    columns: dict[str, list[Any]] = OrderedDict()
    for name in build_parquet_schema().names:
        columns[name] = []

    global_index = int(global_frame_start)
    for sample in samples:
        state_qpos = finite_vector(sample["state_qpos"], STATE_VECTOR_SIZE, "state_qpos")
        action_qpos = finite_vector(sample["action_qpos"], STATE_VECTOR_SIZE, "action_qpos")
        columns["timestamp"].append(float(sample["timestamp_s"]))
        columns["frame_index"].append(int(sample["frame_index"]))
        columns["episode_index"].append(int(episode_index))
        columns["index"].append(int(global_index))
        columns["task_index"].append(int(sample["task_index"]))
        columns["observation.state"].append([float(value) for value in state_qpos.tolist()])
        columns["action"].append([float(value) for value in action_qpos.tolist()])
        global_index += 1

        for fk_group in ("fk_fb", "fk_cmd"):
            for key, value in sample[fk_group].items():
                columns[key].append(finite_vector(value, 6, key).tolist())

        for slot in CAMERA_SLOTS:
            columns[f"observation.images.{slot}"].append(
                {
                    "path": sample["images"][slot]["path"],
                    "timestamp": float(sample["images"][slot]["timestamp_s"]),
                }
            )

    arrays: OrderedDict[str, pa.Array] = OrderedDict()
    arrays["timestamp"] = pa.array(columns["timestamp"], type=pa.float64())
    arrays["frame_index"] = pa.array(columns["frame_index"], type=pa.int64())
    arrays["episode_index"] = pa.array(columns["episode_index"], type=pa.int64())
    arrays["index"] = pa.array(columns["index"], type=pa.int64())
    arrays["task_index"] = pa.array(columns["task_index"], type=pa.int64())
    arrays["observation.state"] = pa.array(columns["observation.state"], type=pa.list_(pa.float64()))
    arrays["action"] = pa.array(columns["action"], type=pa.list_(pa.float64()))
    for group in ARM_GROUPS:
        for side in SIDE_GROUPS:
            for label in ARM_JOINT_LABELS:
                key = f"observation.fk.{group}.{side}.{label}"
                arrays[key] = pa.array(columns[key], type=pa.list_(pa.float64()))
            key = f"observation.fk.{group}.{side}.gripper_flange"
            arrays[key] = pa.array(columns[key], type=pa.list_(pa.float64()))
    for slot in CAMERA_SLOTS:
        key = f"observation.images.{slot}"
        arrays[key] = pa.array(columns[key], type=struct_type())
    return arrays


def count_video_frames(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open encoded video for validation: {video_path}")
    count = 0
    ok, frame = capture.read()
    while ok and frame is not None:
        count += 1
        ok, frame = capture.read()
    capture.release()
    return count


def count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"jsonl file not found: {path}")
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def verify_exported_episode(
    output_root: Path,
    episode_index: int,
    expected_rows: int,
    pose_sidecar: bool,
    verify_video_frames: bool,
) -> dict[str, Any]:
    parquet_path = output_root / "data" / CHUNK_NAME / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"exported parquet not found: {parquet_path}")
    table = pq.read_table(parquet_path)
    row_count = int(table.num_rows)
    if row_count != int(expected_rows):
        raise RuntimeError(f"parquet row count mismatch: {row_count} != {expected_rows}")

    rows = table.to_pydict()
    for row_index, values in enumerate(rows["observation.state"]):
        finite_vector(values, 16, f"parquet observation.state[{row_index}]")
    for row_index, values in enumerate(rows["action"]):
        finite_vector(values, 16, f"parquet action[{row_index}]")

    video_counts: dict[str, int] = {}
    for slot in CAMERA_SLOTS:
        video_path = output_root / "videos" / CHUNK_NAME / f"observation.images.{slot}" / f"episode_{episode_index:06d}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"exported video not found: {video_path}")
        if video_path.stat().st_size <= 0:
            raise RuntimeError(f"exported video is empty: {video_path}")
        if verify_video_frames:
            video_counts[slot] = count_video_frames(video_path)
            if video_counts[slot] != int(expected_rows):
                raise RuntimeError(f"{slot} video frame count mismatch: {video_counts[slot]} != {expected_rows}")
        else:
            video_counts[slot] = -1

    alignment_path = output_root / "meta" / "alignment" / f"episode_{episode_index:06d}.jsonl"
    alignment_rows = count_jsonl_rows(alignment_path)
    if alignment_rows != int(expected_rows):
        raise RuntimeError(f"alignment row count mismatch: {alignment_rows} != {expected_rows}")

    pose_path = output_root / "extras" / "raw_pose" / CHUNK_NAME / f"episode_{episode_index:06d}.jsonl"
    pose_rows = 0
    if pose_sidecar:
        pose_rows = count_jsonl_rows(pose_path)
        if pose_rows != int(expected_rows):
            raise RuntimeError(f"raw pose sidecar row count mismatch: {pose_rows} != {expected_rows}")

    return {
        "parquet_rows": row_count,
        "video_frame_verification": bool(verify_video_frames),
        "video_frames": video_counts,
        "alignment_rows": alignment_rows,
        "raw_pose_rows": pose_rows,
    }


class StandaloneLeRobotV2Writer:
    def __init__(self, output_root: Path, task: str, fps: float, urdf_path: Path, verify_video_frames: bool):
        self.output_root = Path(output_root).resolve()
        self.task = str(task)
        self.fps = float(fps)
        self.verify_video_frames = bool(verify_video_frames)
        self.kinematics = StandaloneG1Kinematics(urdf_path)
        self.parquet_schema = build_parquet_schema()
        self.feature_spec = build_feature_spec()
        self.data_dir = self.output_root / "data" / CHUNK_NAME
        self.alignment_dir = self.output_root / "meta" / "alignment"
        self.control_dir = self.output_root / "extras" / "control" / CHUNK_NAME
        self.raw_pose_dir = self.output_root / "extras" / "raw_pose" / CHUNK_NAME
        self.video_root_dir = self.output_root / "videos" / CHUNK_NAME
        self.meta_dir = self.output_root / "meta"
        for path in [self.data_dir, self.alignment_dir, self.control_dir, self.video_root_dir, self.meta_dir]:
            path.mkdir(parents=True, exist_ok=True)
        for slot in CAMERA_SLOTS:
            (self.video_root_dir / f"observation.images.{slot}").mkdir(parents=True, exist_ok=True)
        self.episodes: list[dict[str, Any]] = []
        self.episode_stats: list[dict[str, Any]] = []
        self.tasks = [{"task_index": 0, "task": self.task}]
        self.global_frame_index_next = 0

    def video_final_path(self, episode_index: int, slot: str) -> Path:
        return self.video_root_dir / f"observation.images.{slot}" / f"episode_{episode_index:06d}.mp4"

    def video_temp_path(self, episode_index: int, slot: str) -> Path:
        return self.video_root_dir / f"observation.images.{slot}" / f"episode_{episode_index:06d}.partial.mp4"

    def parquet_final_path(self, episode_index: int) -> Path:
        return self.data_dir / f"episode_{episode_index:06d}.parquet"

    def parquet_temp_path(self, episode_index: int) -> Path:
        return self.data_dir / f"episode_{episode_index:06d}.partial.parquet"

    def alignment_final_path(self, episode_index: int) -> Path:
        return self.alignment_dir / f"episode_{episode_index:06d}.jsonl"

    def control_final_path(self, episode_index: int) -> Path:
        return self.control_dir / f"episode_{episode_index:06d}.jsonl"

    def raw_pose_final_path(self, episode_index: int) -> Path:
        return self.raw_pose_dir / f"episode_{episode_index:06d}.jsonl"

    def build_info_payload(self) -> dict[str, Any]:
        total_frames = sum(int(ep.get("length", 0)) for ep in self.episodes)
        total_episodes = len(self.episodes)
        return {
            "codebase_version": "v2.1",
            "robot_type": "nero_dual_arm",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(self.tasks),
            "total_chunks": 1,
            "total_videos": len(CAMERA_SLOTS),
            "fps": self.fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "depth_camera_ids": [],
            "depth_path": "depth/cam{camera_id}/episode_{episode_index:06d}/frame_{frame_index:06d}.png",
            "features": self.feature_spec,
            "camera_roles": {"0": "head_fpv", "1": "left_wrist", "2": "right_wrist"},
            "camera_name_map": {"0": "head_fpv", "1": "left_wrist", "2": "right_wrist"},
            "source_camera_map": {
                "observation.images.cam0": "head",
                "observation.images.cam1": "left_wrist/wrist_left",
                "observation.images.cam2": "right_wrist/wrist_right",
            },
            "notes": "generated by standalone LeRobot v2 exporter; cam0=head, cam1=left_wrist/wrist_left, cam2=right_wrist/wrist_right",
        }

    def write_meta_files(self) -> None:
        json_dump_atomic(self.meta_dir / "info.json", self.build_info_payload())
        jsonl_dump_atomic(self.meta_dir / "tasks.jsonl", self.tasks)
        jsonl_dump_atomic(self.meta_dir / "episodes.jsonl", self.episodes)
        jsonl_dump_atomic(self.meta_dir / "episodes_stats.jsonl", self.episode_stats)

    def export_episode(
        self,
        episode_dir: Path,
        raw_items: Sequence[Mapping[str, Any]],
        episode_meta: Mapping[str, Any],
        progress_label: str,
        frame_progress_every: int,
    ) -> dict[str, Any]:
        episode_index = len(self.episodes)
        frame_count = len(raw_items)
        first_sample_ns = int(sample_monotonic_ns(raw_items[0], 0))
        samples: list[dict[str, Any]] = []
        alignment_rows: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
        pose_rows: list[dict[str, Any]] = []
        video_writers: dict[str, OpenCVMP4Writer] = {}
        video_frame_counts: dict[str, int] = {slot: 0 for slot in CAMERA_SLOTS}
        video_shapes: dict[str, tuple[int, int, int]] = {}
        process_sample_ns: list[int] = []
        fourcc = os.environ.get("LEROBOT_OPENCV_FOURCC", "").strip() or DEFAULT_OPENCV_FOURCC

        for frame_index, item in enumerate(raw_items):
            process_t0_ns = time.perf_counter_ns()
            frame_number = frame_index + 1
            state_qpos, action_qpos, fk_state_qpos, fk_action_qpos = state_action_vectors(item, frame_index)
            sample_ns = int(sample_monotonic_ns(item, frame_index))
            timestamp_s = float(sample_ns - first_sample_ns) / 1e9
            if timestamp_s < -1e-9:
                raise RuntimeError(f"sample timestamp moved before episode start: {timestamp_s:.9f}s")

            colors = item.get("colors")
            if not isinstance(colors, Mapping):
                raise KeyError(f"frame {frame_index} missing colors")
            slot_frames: dict[str, np.ndarray] = {}
            for slot in CAMERA_SLOTS:
                raw_key, image_path = pick_color_path(colors, episode_dir, slot, frame_index)
                image = read_color_image(image_path, f"frame {frame_index} colors.{raw_key}")
                shape = (int(image.shape[0]), int(image.shape[1]), int(image.shape[2]))
                if slot not in video_shapes:
                    video_shapes[slot] = shape
                    video_writers[slot] = OpenCVMP4Writer(
                        self.video_temp_path(episode_index, slot),
                        size=(shape[1], shape[0]),
                        fps=self.fps,
                        fourcc=fourcc,
                    )
                elif video_shapes[slot] != shape:
                    raise ValueError(
                        f"camera shape mismatch for {slot} in {episode_dir.name}: "
                        f"expected {video_shapes[slot]}, got {shape} at frame {frame_index}"
                    )
                slot_frames[slot] = image

            for slot, frame in slot_frames.items():
                video_writers[slot].write(frame)
                video_frame_counts[slot] += 1

            fk_fb = self.kinematics.compute_fk(fk_state_qpos, "fb")
            fk_cmd = self.kinematics.compute_fk(fk_action_qpos, "cmd")
            samples.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": timestamp_s,
                    "sample_monotonic_ns": sample_ns,
                    "task_index": 0,
                    "state_qpos": state_qpos,
                    "action_qpos": action_qpos,
                    "fk_fb": fk_fb,
                    "fk_cmd": fk_cmd,
                    "images": {
                        slot: {
                            "path": dataset_relpath(self.video_final_path(episode_index, slot), self.output_root),
                            "timestamp_s": timestamp_s,
                        }
                        for slot in CAMERA_SLOTS
                    },
                }
            )
            alignment_rows.append(
                build_alignment_row(
                    item.get("timestamps", {}),
                    episode_index,
                    frame_index,
                    timestamp_s,
                    sample_ns,
                )
            )
            control_extras = item.get("control_extras")
            if isinstance(control_extras, Mapping) and control_extras:
                control_rows.append(
                    build_control_row(control_extras, episode_index, frame_index, timestamp_s, sample_ns)
                )
            pose_row = pose_row_if_complete(item, episode_index, frame_index, timestamp_s, sample_ns)
            if pose_row is not None:
                pose_rows.append(pose_row)

            process_sample_ns.append(time.perf_counter_ns() - process_t0_ns)
            if progress_label and should_report_frame(frame_number, frame_count, frame_progress_every):
                print_progress(f"{progress_label} decode/write frame {frame_number}/{frame_count}")

        finalize_t0_ns = time.perf_counter_ns()
        for writer in video_writers.values():
            writer.release()
        for slot in CAMERA_SLOTS:
            if video_frame_counts.get(slot, 0) != frame_count:
                raise RuntimeError(f"streamed frame count mismatch for {slot}: {video_frame_counts.get(slot, 0)} != {frame_count}")
            if slot not in video_shapes:
                raise RuntimeError(f"missing video writer size for {slot}")

        alignment_tmp = self.alignment_final_path(episode_index).with_suffix(".jsonl.tmp")
        jsonl_dump_atomic(alignment_tmp, alignment_rows)
        if count_jsonl_rows(alignment_tmp) != frame_count:
            raise RuntimeError(f"alignment sidecar row count mismatch for episode {episode_index}")

        control_sidecar = None
        if control_rows:
            if len(control_rows) != frame_count:
                raise RuntimeError(f"control sidecar row count mismatch: {len(control_rows)} != {frame_count}")
            control_tmp = self.control_final_path(episode_index).with_suffix(".jsonl.tmp")
            jsonl_dump_atomic(control_tmp, control_rows)
            control_sidecar = {
                "tmp_path": control_tmp,
                "final_path": self.control_final_path(episode_index),
                "relative_path": dataset_relpath(self.control_final_path(episode_index), self.output_root),
                "num_rows": frame_count,
            }

        pose_missing_count = int(frame_count - len(pose_rows))
        pose_sidecar = pose_missing_count == 0
        pose_skip_reason = "" if pose_sidecar else "raw pose is incomplete for this episode"
        pose_tmp = None
        if pose_sidecar:
            pose_tmp = self.raw_pose_final_path(episode_index).with_suffix(".jsonl.tmp")
            jsonl_dump_atomic(pose_tmp, pose_rows)
            if count_jsonl_rows(pose_tmp) != frame_count:
                raise RuntimeError(f"raw pose sidecar row count mismatch for episode {episode_index}")

        arrays = build_parquet_arrays(samples, episode_index, self.global_frame_index_next)
        table = pa.Table.from_arrays([arrays[name] for name in self.parquet_schema.names], schema=self.parquet_schema)
        parquet_tmp = self.parquet_temp_path(episode_index)
        parquet_compression = pick_parquet_compression()
        pq.write_table(table, parquet_tmp, compression=parquet_compression)
        if int(table.num_rows) != frame_count:
            raise RuntimeError(f"parquet row count mismatch: {table.num_rows} != {frame_count}")
        for column_name in table.column_names:
            if not column_values_are_finite(table[column_name]):
                raise RuntimeError(f"parquet column contains non-finite values: {column_name}")
        if parquet_tmp.stat().st_size <= 0:
            raise RuntimeError("parquet file is empty after write")

        video_stats: dict[str, dict[str, Any]] = {}
        for slot in CAMERA_SLOTS:
            tmp_path = self.video_temp_path(episode_index, slot)
            if not tmp_path.is_file() or tmp_path.stat().st_size <= 0:
                raise RuntimeError(f"missing streamed video file for {slot}: {tmp_path}")
            decoded_count = None
            decoded_shape = None
            if self.verify_video_frames:
                decoded_count = count_video_frames(tmp_path)
                if decoded_count != frame_count:
                    raise RuntimeError(f"decoded frame count mismatch for {slot}: {decoded_count} != {frame_count}")
            video_stats[slot] = {
                "path": dataset_relpath(self.video_final_path(episode_index, slot), self.output_root),
                "expected_frames": frame_count,
                "streamed_frames": int(video_frame_counts.get(slot, 0)),
                "decoded_frames": decoded_count,
                "decode_verification": bool(self.verify_video_frames),
                "shape": list(video_shapes[slot]),
                "decoded_shape": decoded_shape,
                "codec": fourcc,
            }

        os.replace(parquet_tmp, self.parquet_final_path(episode_index))
        for slot in CAMERA_SLOTS:
            os.replace(self.video_temp_path(episode_index, slot), self.video_final_path(episode_index, slot))
        os.replace(alignment_tmp, self.alignment_final_path(episode_index))
        if control_sidecar is not None:
            os.replace(control_sidecar["tmp_path"], control_sidecar["final_path"])
        if pose_tmp is not None:
            os.replace(pose_tmp, self.raw_pose_final_path(episode_index))

        finalize_ns = time.perf_counter_ns() - finalize_t0_ns
        stats = {
            "validation_passed": True,
            "episode_index": int(episode_index),
            "task_index": 0,
            "num_samples": frame_count,
            "parquet": {
                "path": dataset_relpath(self.parquet_final_path(episode_index), self.output_root),
                "row_count": frame_count,
                "compression": parquet_compression or "uncompressed",
            },
            "videos": video_stats,
            "runtime": {
                "enqueue_count": frame_count,
                "accepted_samples": frame_count,
                "skipped_samples": 0,
                "failed_samples": 0,
                "max_queue_depth": 0,
                "queue": {"max_depth": 0},
                "processing": ms_summary(process_sample_ns),
                "rerun": ms_summary([]),
                "finalize": ms_summary([finalize_ns]),
            },
        }
        self.episodes.append({"episode_index": int(episode_index), "tasks": [self.task], "length": frame_count})
        self.episode_stats.append({"episode_index": int(episode_index), "stats": stats})
        self.global_frame_index_next += frame_count
        self.write_meta_files()

        verification = verify_exported_episode(
            self.output_root,
            episode_index,
            frame_count,
            pose_sidecar,
            verify_video_frames=self.verify_video_frames,
        )
        return {
            "source_episode": episode_dir.name,
            "episode_index": episode_index,
            "frame_count": frame_count,
            "camera_shapes": {slot: list(shape) for slot, shape in video_shapes.items()},
            "pose_sidecar": pose_sidecar,
            "pose_missing_frame_count": int(pose_missing_count),
            "pose_skip_reason": pose_skip_reason,
            "precheck": {
                "complete_pose_count": int(episode_meta["complete_pose_count"]),
                "image_decode_prevalidated": bool(episode_meta.get("image_decode_prevalidated", False)),
            },
            "verification": verification,
        }


def prepare_output_root(input_task_dir: Path, output_root: Path, overwrite: bool) -> None:
    if output_root.resolve() == input_task_dir.resolve():
        raise ValueError("output_root must be different from input_task_dir")
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root is not empty; pass --overwrite to replace it: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def default_urdf_path() -> Path:
    script_path = Path(__file__).resolve()
    adjacent = script_path.parent / "assets" / "g1" / "g1_body29_hand14.urdf"
    repo_layout = script_path.parents[1] / "assets" / "g1" / "g1_body29_hand14.urdf"
    if adjacent.is_file():
        return adjacent
    if repo_layout.is_file():
        return repo_layout
    return adjacent


def export_raw_task_dir(
    input_task_dir: str | Path,
    output_root: str | Path,
    task: str,
    fps: float,
    urdf_path: str | Path | None = None,
    overwrite: bool = False,
    progress_every: int = 1,
    frame_progress_every: int = 100,
    strict_image_validate: bool = False,
    verify_export: bool = True,
    verify_video_frames: bool = False,
) -> dict[str, Any]:
    input_task_dir = Path(input_task_dir).resolve()
    output_root = Path(output_root).resolve()
    task_text = str(task).strip()
    if not task_text:
        raise ValueError("task must be a non-empty string")
    frequency = float(fps)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("fps must be a positive finite number")
    progress_every = int(progress_every)
    frame_progress_every = int(frame_progress_every)
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if frame_progress_every < 0:
        raise ValueError("frame_progress_every must be non-negative")
    resolved_urdf_path = Path(urdf_path).resolve() if urdf_path is not None else default_urdf_path()

    episode_dirs = list_raw_episode_dirs(input_task_dir)
    print_progress(f"found {len(episode_dirs)} episodes under {input_task_dir}")
    raw_episode_plan = []
    for episode_number, episode_dir in enumerate(episode_dirs, start=1):
        label = f"[{episode_number}/{len(episode_dirs)}] {episode_dir.name}"
        report_episode = progress_every > 0 and (
            episode_number == 1 or episode_number == len(episode_dirs) or episode_number % progress_every == 0
        )
        if report_episode:
            print_progress(f"{label} loading data.json")
        raw_items = load_raw_episode_data(episode_dir)
        if report_episode:
            mode = "strict image validate" if strict_image_validate else "light precheck"
            print_progress(f"{label} {mode} {len(raw_items)} frames")
        episode_meta = validate_raw_episode_with_progress(
            episode_dir,
            raw_items,
            progress_label=label if report_episode else "",
            frame_progress_every=frame_progress_every,
            decode_images=bool(strict_image_validate),
        )
        episode_meta["image_decode_prevalidated"] = bool(strict_image_validate)
        raw_episode_plan.append((episode_dir, raw_items, episode_meta))
        if report_episode:
            print_progress(f"{label} precheck done")

    print_progress(f"preparing output root {output_root}")
    prepare_output_root(input_task_dir, output_root, overwrite)
    print_progress(f"initializing standalone LeRobot writer urdf={resolved_urdf_path}")
    writer = StandaloneLeRobotV2Writer(
        output_root=output_root,
        task=task_text,
        fps=frequency,
        urdf_path=resolved_urdf_path,
        verify_video_frames=bool(verify_video_frames),
    )
    writer.write_meta_files()

    episode_summaries: list[dict[str, Any]] = []
    for episode_number, (episode_dir, raw_items, episode_meta) in enumerate(raw_episode_plan, start=1):
        label = f"[{episode_number}/{len(raw_episode_plan)}] {episode_dir.name}"
        report_episode = progress_every > 0 and (
            episode_number == 1 or episode_number == len(raw_episode_plan) or episode_number % progress_every == 0
        )
        if report_episode:
            print_progress(f"{label} exporting {len(raw_items)} frames")
        summary = writer.export_episode(
            episode_dir=episode_dir,
            raw_items=raw_items,
            episode_meta=episode_meta,
            progress_label=label if report_episode else "",
            frame_progress_every=frame_progress_every,
        )
        if verify_export:
            summary["verification"] = verify_exported_episode(
                output_root,
                int(summary["episode_index"]),
                int(summary["frame_count"]),
                bool(summary["pose_sidecar"]),
                verify_video_frames=bool(verify_video_frames),
            )
        episode_summaries.append(summary)
        if report_episode:
            print_progress(f"{label} export done")

    summary = {
        "source_root": str(input_task_dir),
        "output_root": str(output_root),
        "task": task_text,
        "fps": frequency,
        "urdf_path": str(resolved_urdf_path),
        "strict_image_validate": bool(strict_image_validate),
        "verify_export": bool(verify_export),
        "verify_video_frames": bool(verify_video_frames),
        "episodes_total": len(episode_dirs),
        "episodes_exported": len(episode_summaries),
        "frames_total": int(sum(item["frame_count"] for item in episode_summaries)),
        "episodes": episode_summaries,
    }
    json_dump_atomic(output_root / "export_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone export of raw episode_xxxx/data.json directories to LeRobot v2."
    )
    parser.add_argument("--input-task-dir", required=True, type=Path, help="Input raw task directory containing episode_xxxx folders.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output LeRobot v2 dataset root.")
    parser.add_argument("--task", required=True, help="Single task prompt written to LeRobot meta/tasks.jsonl.")
    parser.add_argument("--fps", required=True, type=float, help="Nominal LeRobot video/dataset FPS.")
    parser.add_argument("--urdf-path", type=Path, default=None, help="Path to g1_body29_hand14.urdf. Defaults to adjacent assets/g1 or repo assets/g1.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output root.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print episode-level progress every N episodes. Use 0 to disable.")
    parser.add_argument("--frame-progress-every", type=int, default=100, help="Print frame-level progress every N frames for reported episodes.")
    parser.add_argument("--strict-image-validate", default="0", help="Decode every source image during precheck before export. Default 0.")
    parser.add_argument("--verify-export", default="1", help="Verify parquet/alignment/video files after each episode. Default 1.")
    parser.add_argument("--verify-video-frames", default="0", help="Decode exported mp4 files to count frames after writing. Default 0.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = export_raw_task_dir(
        input_task_dir=args.input_task_dir,
        output_root=args.output_root,
        task=args.task,
        fps=args.fps,
        urdf_path=args.urdf_path,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
        frame_progress_every=args.frame_progress_every,
        strict_image_validate=parse_bool_flag(args.strict_image_validate, "--strict-image-validate"),
        verify_export=parse_bool_flag(args.verify_export, "--verify-export"),
        verify_video_frames=parse_bool_flag(args.verify_video_frames, "--verify-video-frames"),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
