"""Nero-facing, control-thread storage operations for XR recordings and exports."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from teleop.ui.episode_store import FINALIZED_JSON_RE


_EPISODE_NAME = re.compile(r"episode_\d+\Z")
_DATASET_NAME = re.compile(r"[^/\\]+\Z")


def update_episode_task_description(
    *,
    root_dir: Path,
    episode_name: str,
    task_description: str,
) -> dict[str, Any]:
    """Atomically update only ``text.desc`` of one finalized XR episode."""
    normalized_episode = _require_episode_name(episode_name)
    normalized_description = _require_task_description(task_description)
    root = Path(root_dir).expanduser().resolve()
    episode_dir = _episode_dir(root, normalized_episode)
    data_path = episode_dir / "data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"episode data.json was not found: {data_path}")
    original_manifest = _data_manifest(data_path)
    validation_path = episode_dir / "validation.json"
    validation = _load_current_validation(validation_path, original_manifest)
    raw = data_path.read_text(encoding="utf-8")
    if FINALIZED_JSON_RE.search(raw) is None:
        raise RuntimeError(f"episode is not finalized yet: {episode_dir.name}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"episode data must be a JSON object: {data_path}")
    if not isinstance(payload.get("data"), list):
        raise ValueError(f"episode data field must be a list: {data_path}")
    text = payload.get("text")
    if not isinstance(text, dict):
        raise ValueError(f"episode text field must be a JSON object: {data_path}")
    previous_description = text.get("desc")
    if not isinstance(previous_description, str):
        raise ValueError(f"episode text.desc must be a string: {data_path}")
    if previous_description == normalized_description:
        return {
            "episode_name": normalized_episode,
            "task_description": normalized_description,
            "changed": False,
        }
    text["desc"] = normalized_description
    _write_json_atomic_durable(data_path, payload)
    if validation is not None:
        validation["source_data"] = _data_manifest(data_path)
        _write_json_atomic_durable(validation_path, validation)
    return {
        "episode_name": normalized_episode,
        "task_description": normalized_description,
        "changed": True,
        "validation_refreshed": validation is not None,
    }


def export_output_details(*, output_root: str, dataset_name: str) -> dict[str, Any]:
    """Read one completed XR LeRobot v2 export without inventing metadata."""
    root, dataset = _resolve_dataset(output_root, dataset_name)
    summary = _read_json_object(dataset / "export_summary.json", required=True)
    info = _read_json_object(dataset / "meta" / "info.json", required=True)
    episodes = _read_export_episodes(dataset)
    frame_count = _require_nonnegative_int(info.get("total_frames"), "info.total_frames")
    return {
        "output_root": str(root),
        "dataset_name": dataset.name,
        "dataset_root": str(dataset),
        "format": "lerobot_v2",
        "info": info,
        "export_summary": summary,
        "episode_count": len(episodes),
        "frame_count": frame_count,
    }


def export_output_episodes(*, output_root: str, dataset_name: str) -> dict[str, Any]:
    """Read exported LeRobot episode rows and their actual task/frame metadata."""
    root, dataset = _resolve_dataset(output_root, dataset_name)
    _read_json_object(dataset / "export_summary.json", required=True)
    return {
        "output_root": str(root),
        "dataset_name": dataset.name,
        "dataset_root": str(dataset),
        "format": "lerobot_v2",
        "episodes": _read_export_episodes(dataset),
    }


def _require_episode_name(value: str) -> str:
    normalized = str(value).strip()
    if _EPISODE_NAME.fullmatch(normalized) is None:
        raise ValueError("episode_name must use episode_NNNN format")
    return normalized


def _require_task_description(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("task_description must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("task_description must be non-empty")
    return normalized


def _episode_dir(root: Path, episode_name: str) -> Path:
    if not root.is_dir():
        raise NotADirectoryError(f"record root is not a directory: {root}")
    episode = (root / episode_name).resolve()
    if episode.parent != root:
        raise ValueError(f"episode path escapes record root: {episode_name}")
    return episode


def _resolve_dataset(output_root: str, dataset_name: str) -> tuple[Path, Path]:
    root = Path(str(output_root).strip()).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"export output_root is not a directory: {root}")
    normalized_name = str(dataset_name).strip()
    if not normalized_name or normalized_name in {".", ".."} or _DATASET_NAME.fullmatch(normalized_name) is None:
        raise ValueError("dataset_name must be one plain directory name")
    dataset = (root / normalized_name).resolve()
    if dataset.parent != root:
        raise ValueError(f"dataset path escapes output_root: {normalized_name}")
    if not dataset.is_dir():
        raise FileNotFoundError(f"export dataset was not found: {dataset}")
    return root, dataset


def _read_json_object(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required export metadata was not found: {path}")
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"export metadata must be a JSON object: {path}")
    return value


def _data_manifest(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _load_current_validation(
    path: Path,
    expected_manifest: Mapping[str, int],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    validation = _read_json_object(path, required=True)
    source_data = validation.get("source_data")
    if not isinstance(source_data, Mapping):
        raise ValueError(f"validation source_data must be an object: {path}")
    for key, expected_value in expected_manifest.items():
        actual_value = source_data.get(key)
        if isinstance(actual_value, bool) or not isinstance(actual_value, int) or actual_value != expected_value:
            raise RuntimeError(
                f"validation.json is stale; revalidate before updating task description: {path}"
            )
    return validation


def _read_export_episodes(dataset: Path) -> list[dict[str, Any]]:
    path = dataset / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"required export episode index was not found: {path}")
    episodes: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"export episode index row must be an object: {path}:{line_number}")
        episode_index = row.get("episode_index")
        length = row.get("length")
        tasks = row.get("tasks")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError(f"export episode_index must be a non-negative integer: {path}:{line_number}")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError(f"export episode length must be a non-negative integer: {path}:{line_number}")
        if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
            raise ValueError(f"export episode tasks must be a string array: {path}:{line_number}")
        episodes.append(
            {
                "name": f"episode_{episode_index:06d}",
                "episode_index": episode_index,
                "frame_count": length,
                "tasks": list(tasks),
            }
        )
    return episodes


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _write_json_atomic_durable(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    os.fsync(directory_fd)
    os.close(directory_fd)
