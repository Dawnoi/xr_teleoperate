#!/usr/bin/env python3
"""Export legacy raw episodes to the repository's LeRobot v2 dataset format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.recording.lerobot_v2_writer import CAMERA_SLOTS, CHUNK_NAME, LeRobotV2Writer
from data_pipeline.export.g1d_fk import G1DArmFkProvider


SLOT_TO_RAW_CAMERA_KEY = {
    "cam0": ("head",),
    "cam1": ("left_wrist", "wrist_left"),
    "cam2": ("right_wrist", "wrist_right"),
}
POSE_PATHS = (
    ("states", "left_arm", "fb", "left"),
    ("states", "right_arm", "fb", "right"),
    ("actions", "left_arm", "cmd", "left"),
    ("actions", "right_arm", "cmd", "right"),
)
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


def parse_bool_flag(value: Any, label: str) -> bool:
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{label} must be one of 0/1/true/false/yes/no/on/off, got {value!r}")


def finite_vector(values: Any, expected_len: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != int(expected_len):
        raise ValueError(f"{label} expected length {expected_len}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or Inf")
    return arr.copy()


def load_raw_episode_data(episode_dir: Path) -> List[Dict[str, Any]]:
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


def list_raw_episode_dirs(input_task_dir: Path) -> List[Path]:
    if not input_task_dir.is_dir():
        raise NotADirectoryError(f"input task dir not found: {input_task_dir}")
    episode_dirs = [
        path for path in input_task_dir.iterdir()
        if path.is_dir() and path.name.startswith("episode_")
    ]
    episode_dirs = sorted(episode_dirs, key=lambda path: path.name)
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


def read_color_image(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{label} image could not be decoded: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{label} image must decode to HxWx3, got shape={image.shape}")
    return np.ascontiguousarray(image)


def read_color_images_for_frame(item: Mapping[str, Any], episode_dir: Path, frame_index: int) -> Dict[str, np.ndarray]:
    colors = item.get("colors")
    if not isinstance(colors, Mapping):
        raise KeyError(f"frame {frame_index} missing colors")

    images: Dict[str, np.ndarray] = {}
    for slot in CAMERA_SLOTS:
        raw_key, image_path = pick_color_path(colors, episode_dir, slot, frame_index)
        image = read_color_image(image_path, f"frame {frame_index} colors.{raw_key}")
        if slot == "cam0":
            images["head"] = image
        elif slot == "cam1":
            images["left_wrist"] = image
        elif slot == "cam2":
            images["right_wrist"] = image
        else:
            raise KeyError(f"unsupported camera slot: {slot}")
    return images


def pick_color_path(colors: Mapping[str, Any], episode_dir: Path, slot: str, frame_index: int) -> Tuple[str, Path]:
    for raw_key in SLOT_TO_RAW_CAMERA_KEY[slot]:
        if raw_key in colors:
            label = f"frame {frame_index} colors.{raw_key}"
            return raw_key, resolve_color_path(episode_dir, colors[raw_key], label)
    expected = "/".join(SLOT_TO_RAW_CAMERA_KEY[slot])
    raise KeyError(f"frame {frame_index} missing camera color for {slot}: expected one of {expected}")


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


def pose6_from_record(value: Any, label: str) -> List[float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if "position" not in value or "rpy" not in value:
        raise ValueError(f"{label} must contain position and rpy")
    position = finite_vector(value["position"], 3, f"{label}.position")
    rpy = finite_vector(value["rpy"], 3, f"{label}.rpy")
    return [float(v) for v in np.concatenate([position, rpy]).tolist()]


def pose_value(item: Mapping[str, Any], source_key: str, group_key: str) -> Any:
    source = item.get(source_key)
    if not isinstance(source, Mapping):
        return None
    group = source.get(group_key)
    if not isinstance(group, Mapping):
        return None
    return group.get("pose")


def pose_row_if_complete(
    item: Mapping[str, Any],
    episode_index: int,
    frame_index: int,
    timestamp_s: float,
    sample_ns: int,
) -> Dict[str, Any] | None:
    raw_values = []
    for source_key, group_key, _pose_group, _side in POSE_PATHS:
        raw_values.append(pose_value(item, source_key, group_key))
    if any(value is None for value in raw_values):
        return None

    row = {
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


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def preprocess_episode(episode_dir: Path, raw_items: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    return preprocess_episode_with_progress(
        episode_dir,
        raw_items,
        progress_label="",
        frame_progress_every=0,
    )


def preprocess_episode_with_progress(
    episode_dir: Path,
    raw_items: Sequence[Mapping[str, Any]],
    progress_label: str,
    frame_progress_every: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prepared_items: List[Dict[str, Any]] = []
    shapes_by_slot: Dict[str, Tuple[int, int, int]] = {}
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

        images: Dict[str, np.ndarray] = {}
        for slot in CAMERA_SLOTS:
            raw_key, image_path = pick_color_path(colors, episode_dir, slot, frame_index)
            image = read_color_image(image_path, f"frame {frame_index} colors.{raw_key}")
            shape = (int(image.shape[0]), int(image.shape[1]), int(image.shape[2]))
            if slot not in shapes_by_slot:
                shapes_by_slot[slot] = shape
            elif shapes_by_slot[slot] != shape:
                raise ValueError(
                    f"camera shape mismatch for {slot} in {episode_dir.name}: "
                    f"expected {shapes_by_slot[slot]}, got {shape} at frame {frame_index}"
                )
            images[slot] = image

        if pose_row_if_complete(item, 0, frame_index, 0.0, sample_ns) is not None:
            complete_pose_count += 1

        prepared_items.append(
            {
                "colors": {
                    "head": images["cam0"],
                    "left_wrist": images["cam1"],
                    "right_wrist": images["cam2"],
                },
                "states": item["states"],
                "actions": item["actions"],
                "timestamps": item["timestamps"],
                "sample_monotonic_ns": sample_ns,
                "raw_item": item,
            }
        )

        if progress_label and should_report_frame(frame_number, frame_count, frame_progress_every):
            print_progress(f"{progress_label} preprocess frame {frame_number}/{frame_count}")

    return prepared_items, {
        "frame_count": len(prepared_items),
        "camera_shapes": {slot: list(shape) for slot, shape in shapes_by_slot.items()},
        "complete_pose_count": int(complete_pose_count),
    }


def validate_raw_episode(episode_dir: Path, raw_items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return validate_raw_episode_with_progress(
        episode_dir,
        raw_items,
        progress_label="",
        frame_progress_every=0,
        decode_images=True,
    )


def print_progress(message: str) -> None:
    print(f"[EXPORT_PROGRESS] {message}", flush=True)


def should_report_frame(frame_number: int, frame_count: int, frame_progress_every: int) -> bool:
    if frame_count <= 0:
        return False
    if frame_number == frame_count:
        return True
    if frame_progress_every <= 0:
        return False
    return frame_number % frame_progress_every == 0


def validate_raw_episode_with_progress(
    episode_dir: Path,
    raw_items: Sequence[Mapping[str, Any]],
    progress_label: str,
    frame_progress_every: int,
    decode_images: bool = False,
) -> Dict[str, Any]:
    shapes_by_slot: Dict[str, Tuple[int, int, int]] = {}
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


def wait_for_writer(writer: LeRobotV2Writer) -> None:
    writer._item_queue.join()
    while not writer.is_ready():
        time.sleep(0.01)


def count_video_frames(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open encoded video for validation: {video_path}")
    count = 0
    ok, _frame = capture.read()
    while ok:
        count += 1
        ok, _frame = capture.read()
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
    export_fk: bool = False,
) -> Dict[str, Any]:
    return verify_exported_episode_with_options(
        output_root,
        episode_index,
        expected_rows,
        pose_sidecar,
        verify_video_frames=True,
        export_fk=export_fk,
    )


def verify_exported_episode_with_options(
    output_root: Path,
    episode_index: int,
    expected_rows: int,
    pose_sidecar: bool,
    verify_video_frames: bool,
    export_fk: bool,
) -> Dict[str, Any]:
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

    fk_columns = [name for name in table.column_names if name.startswith("observation.fk.")]
    if export_fk:
        if len(fk_columns) != 32:
            raise RuntimeError(f"FK column count mismatch: {len(fk_columns)} != 32")
        for column_name in fk_columns:
            for row_index, values in enumerate(rows[column_name]):
                finite_vector(values, 6, f"parquet {column_name}[{row_index}]")
    elif fk_columns:
        raise RuntimeError("FK columns present while export_fk=0")

    video_counts: Dict[str, int] = {}
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
        "export_fk": bool(export_fk),
    }


def write_pose_sidecar_if_complete(
    output_root: Path,
    episode_index: int,
    prepared_items: Sequence[Mapping[str, Any]],
    complete_pose_count: int,
) -> Tuple[bool, int, str]:
    missing_count = len(prepared_items) - int(complete_pose_count)
    if missing_count > 0:
        return False, missing_count, "raw pose is incomplete for this episode"

    first_sample_ns = int(prepared_items[0]["sample_monotonic_ns"])
    rows = []
    for frame_index, prepared in enumerate(prepared_items):
        sample_ns = int(prepared["sample_monotonic_ns"])
        timestamp_s = float(sample_ns - first_sample_ns) / 1e9
        row = pose_row_if_complete(
            prepared["raw_item"],
            episode_index,
            frame_index,
            timestamp_s,
            sample_ns,
        )
        if row is None:
            raise RuntimeError(f"internal raw pose completeness mismatch at frame {frame_index}")
        rows.append(row)

    pose_path = output_root / "extras" / "raw_pose" / CHUNK_NAME / f"episode_{episode_index:06d}.jsonl"
    write_jsonl_atomic(pose_path, rows)
    return True, 0, ""


def export_prepared_episode(
    writer: LeRobotV2Writer,
    output_root: Path,
    episode_dir: Path,
    prepared_items: Sequence[Mapping[str, Any]],
    preprocess_meta: Mapping[str, Any],
    progress_label: str = "",
    frame_progress_every: int = 0,
) -> Dict[str, Any]:
    if not writer.create_episode():
        raise RuntimeError("LeRobotV2Writer refused to create a new episode")
    episode_index = int(writer._current_episode_index)
    frame_count = len(prepared_items)
    for frame_index, prepared in enumerate(prepared_items):
        writer.add_item(
            colors=prepared["colors"],
            states=prepared["states"],
            actions=prepared["actions"],
            timestamps=prepared["timestamps"],
        )
        frame_number = frame_index + 1
        if progress_label and should_report_frame(frame_number, frame_count, frame_progress_every):
            print_progress(f"{progress_label} queue frame {frame_number}/{frame_count}")
    if progress_label:
        print_progress(f"{progress_label} waiting writer queue")
    writer._item_queue.join()
    if progress_label:
        print_progress(f"{progress_label} saving episode_{episode_index:06d}")
    writer.save_episode()
    wait_for_writer(writer)

    if progress_label:
        print_progress(f"{progress_label} writing pose sidecar")
    pose_sidecar, pose_missing_count, pose_skip_reason = write_pose_sidecar_if_complete(
        output_root,
        episode_index,
        prepared_items,
        int(preprocess_meta["complete_pose_count"]),
    )
    if progress_label:
        print_progress(f"{progress_label} verifying episode_{episode_index:06d}")
    verification = verify_exported_episode(output_root, episode_index, len(prepared_items), pose_sidecar)
    return {
        "source_episode": episode_dir.name,
        "episode_index": episode_index,
        "frame_count": len(prepared_items),
        "camera_shapes": preprocess_meta["camera_shapes"],
        "pose_sidecar": pose_sidecar,
        "pose_missing_frame_count": int(pose_missing_count),
        "pose_skip_reason": pose_skip_reason,
        "verification": verification,
    }


def export_raw_episode(
    writer: LeRobotV2Writer,
    output_root: Path,
    episode_dir: Path,
    raw_items: Sequence[Mapping[str, Any]],
    episode_meta: Mapping[str, Any],
    progress_label: str = "",
    frame_progress_every: int = 0,
    verify_export: bool = True,
    verify_video_frames: bool = False,
    export_fk: bool = False,
) -> Dict[str, Any]:
    if not writer.create_episode():
        raise RuntimeError("LeRobotV2Writer refused to create a new episode")
    episode_index = int(writer._current_episode_index)
    frame_count = len(raw_items)
    first_sample_ns = int(sample_monotonic_ns(raw_items[0], 0))
    pose_rows: List[Dict[str, Any]] = []
    shapes_by_slot: Dict[str, Tuple[int, int, int]] = {}

    for frame_index, item in enumerate(raw_items):
        frame_number = frame_index + 1
        images = read_color_images_for_frame(item, episode_dir, frame_index)
        for raw_name, image in images.items():
            slot = {
                "head": "cam0",
                "left_wrist": "cam1",
                "right_wrist": "cam2",
            }[raw_name]
            shape = (int(image.shape[0]), int(image.shape[1]), int(image.shape[2]))
            if slot not in shapes_by_slot:
                shapes_by_slot[slot] = shape
            elif shapes_by_slot[slot] != shape:
                raise ValueError(
                    f"camera shape mismatch for {slot} in {episode_dir.name}: "
                    f"expected {shapes_by_slot[slot]}, got {shape} at frame {frame_index}"
                )

        sample_ns = int(sample_monotonic_ns(item, frame_index))
        timestamp_s = float(sample_ns - first_sample_ns) / 1e9
        pose_row = pose_row_if_complete(item, episode_index, frame_index, timestamp_s, sample_ns)
        if pose_row is not None:
            pose_rows.append(pose_row)

        writer.add_item(
            colors=images,
            states=item["states"],
            actions=item["actions"],
            timestamps=item["timestamps"],
        )
        if progress_label and should_report_frame(frame_number, frame_count, frame_progress_every):
            print_progress(f"{progress_label} decode/write frame {frame_number}/{frame_count}")

    if progress_label:
        print_progress(f"{progress_label} waiting writer queue")
    writer._item_queue.join()
    if progress_label:
        print_progress(f"{progress_label} saving episode_{episode_index:06d}")
    writer.save_episode()
    wait_for_writer(writer)

    pose_missing_count = int(frame_count - len(pose_rows))
    pose_sidecar = pose_missing_count == 0
    pose_skip_reason = "" if pose_sidecar else "raw pose is incomplete for this episode"
    if pose_sidecar:
        if progress_label:
            print_progress(f"{progress_label} writing pose sidecar")
        pose_path = output_root / "extras" / "raw_pose" / CHUNK_NAME / f"episode_{episode_index:06d}.jsonl"
        write_jsonl_atomic(pose_path, pose_rows)

    verification: Dict[str, Any] = {
        "enabled": bool(verify_export),
        "video_frame_verification": bool(verify_video_frames),
    }
    if verify_export:
        if progress_label:
            print_progress(f"{progress_label} verifying episode_{episode_index:06d}")
        verification = verify_exported_episode_with_options(
            output_root,
            episode_index,
            frame_count,
            pose_sidecar,
            verify_video_frames=verify_video_frames,
            export_fk=export_fk,
        )

    return {
        "source_episode": episode_dir.name,
        "episode_index": episode_index,
        "frame_count": frame_count,
        "camera_shapes": {slot: list(shape) for slot, shape in shapes_by_slot.items()},
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


def export_raw_task_dir(
    input_task_dir: str | Path,
    output_root: str | Path,
    task: str,
    fps: float,
    overwrite: bool = False,
    progress_every: int = 1,
    frame_progress_every: int = 100,
    strict_image_validate: bool = False,
    verify_export: bool = True,
    verify_video_frames: bool = False,
    export_fk: bool = False,
    urdf_path: str | Path | None = None,
) -> Dict[str, Any]:
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

    episode_dirs = list_raw_episode_dirs(input_task_dir)
    print_progress(f"found {len(episode_dirs)} episodes under {input_task_dir}")
    raw_episode_plan = []
    for episode_number, episode_dir in enumerate(episode_dirs, start=1):
        label = f"[{episode_number}/{len(episode_dirs)}] {episode_dir.name}"
        report_episode = progress_every > 0 and (
            episode_number == 1
            or episode_number == len(episode_dirs)
            or episode_number % progress_every == 0
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

    fk_provider = None
    resolved_urdf_path = None
    if export_fk:
        if urdf_path is None or not str(urdf_path).strip():
            raise ValueError("urdf_path is required when export_fk=1")
        resolved_urdf_path = Path(urdf_path).expanduser().resolve()
        fk_provider = G1DArmFkProvider(resolved_urdf_path)

    print_progress(f"preparing output root {output_root}")
    prepare_output_root(input_task_dir, output_root, overwrite)

    print_progress("initializing LeRobot writer")
    writer = LeRobotV2Writer(
        task_dir=str(output_root),
        fk_provider=fk_provider,
        task_goal=task_text,
        frequency=frequency,
        rerun_log=False,
        verify_encoded_video=bool(verify_video_frames),
    )

    episode_summaries: List[Dict[str, Any]] = []
    for episode_number, (episode_dir, raw_items, episode_meta) in enumerate(raw_episode_plan, start=1):
        label = f"[{episode_number}/{len(raw_episode_plan)}] {episode_dir.name}"
        report_episode = progress_every > 0 and (
            episode_number == 1
            or episode_number == len(raw_episode_plan)
            or episode_number % progress_every == 0
        )
        if report_episode:
            print_progress(f"{label} exporting {len(raw_items)} frames")
        episode_summaries.append(
            export_raw_episode(
                writer,
                output_root,
                episode_dir,
                raw_items,
                episode_meta,
                progress_label=label if report_episode else "",
                frame_progress_every=frame_progress_every,
                verify_export=bool(verify_export),
                verify_video_frames=bool(verify_video_frames),
                export_fk=bool(export_fk),
            )
        )
        if report_episode:
            print_progress(f"{label} export done")
    writer.close()

    summary = {
        "source_root": str(input_task_dir),
        "output_root": str(output_root),
        "task": task_text,
        "fps": frequency,
        "strict_image_validate": bool(strict_image_validate),
        "verify_export": bool(verify_export),
        "verify_video_frames": bool(verify_video_frames),
        "export_fk": bool(export_fk),
        "urdf_path": str(resolved_urdf_path) if resolved_urdf_path is not None else "",
        "episodes_total": len(episode_dirs),
        "episodes_exported": len(episode_summaries),
        "frames_total": int(sum(item["frame_count"] for item in episode_summaries)),
        "episodes": episode_summaries,
    }
    write_json_atomic(output_root / "export_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export legacy episode_xxxx/data.json raw data to this repository's LeRobot v2 format."
    )
    parser.add_argument("--input-task-dir", required=True, type=Path, help="Input raw task directory containing episode_xxxx folders.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output LeRobot v2 dataset root.")
    parser.add_argument("--task", required=True, help="Single task prompt written to LeRobot meta/tasks.jsonl.")
    parser.add_argument("--fps", required=True, type=float, help="Nominal LeRobot video/dataset FPS.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output root.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print episode-level progress every N episodes. Use 0 to disable progress lines.",
    )
    parser.add_argument(
        "--frame-progress-every",
        type=int,
        default=100,
        help="Print frame-level progress every N frames inside reported episodes. Use 0 to print only episode boundaries.",
    )
    parser.add_argument(
        "--strict-image-validate",
        default="0",
        help="Decode every source image during precheck before export. Default 0; images are otherwise decoded once during video writing.",
    )
    parser.add_argument(
        "--verify-export",
        default="1",
        help="Verify parquet/alignment/video files after each episode. Default 1.",
    )
    parser.add_argument(
        "--verify-video-frames",
        default="0",
        help="Decode exported mp4 files to count frames after writing. Default 0.",
    )
    parser.add_argument("--export-fk", default="0", help="Export G1D FK columns as xyz+rpy. Default 0.")
    parser.add_argument("--urdf-path", type=Path, help="G1D URDF required when --export-fk=1.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = export_raw_task_dir(
        input_task_dir=args.input_task_dir,
        output_root=args.output_root,
        task=args.task,
        fps=args.fps,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
        frame_progress_every=args.frame_progress_every,
        strict_image_validate=parse_bool_flag(args.strict_image_validate, "--strict-image-validate"),
        verify_export=parse_bool_flag(args.verify_export, "--verify-export"),
        verify_video_frames=parse_bool_flag(args.verify_video_frames, "--verify-video-frames"),
        export_fk=parse_bool_flag(args.export_fk, "--export-fk"),
        urdf_path=args.urdf_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
