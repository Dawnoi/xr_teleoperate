from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CAMERA_ID_BY_NAME = {
    "head": 0,
    "left_wrist": 1,
    "wrist_left": 1,
    "right_wrist": 2,
    "wrist_right": 2,
}

CAMERA_NAMES_BY_ID = {
    0: ("head",),
    1: ("left_wrist", "wrist_left"),
    2: ("right_wrist", "wrist_right"),
}

FINALIZED_JSON_RE = re.compile(r"\]\s*}\s*$")


def resolve_root(root_dir: str | None, fallback: str | None = None) -> Path:
    raw = str(root_dir or fallback or ".").strip() or "."
    return Path(raw).expanduser()


def load_episode_json(episode_dir: Path) -> dict[str, Any]:
    data_path = episode_dir / "data.json"
    text = data_path.read_text(encoding="utf-8")
    if FINALIZED_JSON_RE.search(text) is None:
        return {
            "info": {},
            "text": {},
            "data": [],
            "_incomplete_error": f"{data_path} is not finalized yet",
        }
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"episode data must be a JSON object: {data_path}")
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError(f"episode data field must be a list: {data_path}")
    return payload


def episode_items(episode_dir: Path) -> list[dict[str, Any]]:
    payload = load_episode_json(episode_dir)
    return [item for item in payload.get("data", []) if isinstance(item, dict)]


def _sample_time_ns(item: dict[str, Any]) -> int | None:
    timestamps = item.get("timestamps", {})
    if not isinstance(timestamps, dict):
        return None
    value = timestamps.get("sample_monotonic_ns")
    return int(value) if value is not None else None


def _finite_numbers(values: Any, expected_len: int | None = None) -> bool:
    if not isinstance(values, list):
        return False
    if expected_len is not None and len(values) < expected_len:
        return False
    for value in values[: expected_len or len(values)]:
        number = float(value)
        if not math.isfinite(number):
            return False
    return True


def _state_section(item: dict[str, Any], key: str) -> dict[str, Any]:
    states = item.get("states", {})
    if not isinstance(states, dict):
        return {}
    value = states.get(key, {})
    return value if isinstance(value, dict) else {}


def _arm_qpos(item: dict[str, Any], side: str) -> list[float]:
    key = f"{side}_arm"
    section = _state_section(item, key) or _state_section(item, side)
    qpos = section.get("qpos", [])
    return list(qpos) if isinstance(qpos, list) else []


def _gripper_qpos(item: dict[str, Any], side: str) -> float | None:
    section = _state_section(item, f"{side}_ee")
    qpos = section.get("qpos", [])
    if isinstance(qpos, list) and qpos:
        value = float(qpos[0])
        return value if math.isfinite(value) else None
    return None


def _camera_entries(item: dict[str, Any]) -> dict[str, str]:
    colors = item.get("colors", {})
    if not isinstance(colors, dict):
        return {}
    return {str(name): str(path) for name, path in colors.items() if str(path)}


def _camera_id(name: str) -> int:
    if name in CAMERA_ID_BY_NAME:
        return CAMERA_ID_BY_NAME[name]
    return max(CAMERA_ID_BY_NAME.values(), default=2) + 1


def _camera_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names: dict[int, str] = {}
    for item in items:
        for name in _camera_entries(item):
            camera_id = _camera_id(name)
            names.setdefault(camera_id, name)
    return [
        {"camera_id": camera_id, "camera_name": name, "camera_mode": "rgb"}
        for camera_id, name in sorted(names.items())
    ]


def validate_episode(episode_dir: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    loaded = load_episode_json(episode_dir) if payload is None else payload
    items = [item for item in loaded.get("data", []) if isinstance(item, dict)]
    frame_count = len(items)
    errors: list[str] = []
    warnings: list[str] = []
    incomplete_error = loaded.get("_incomplete_error")
    if incomplete_error:
        warnings.append(str(incomplete_error))
    if frame_count <= 0 and not incomplete_error:
        errors.append("episode has no frames")

    camera_counts: dict[str, int] = {}
    for item in items:
        for name, rel_path in _camera_entries(item).items():
            if (episode_dir / rel_path).is_file():
                camera_counts[name] = camera_counts.get(name, 0) + 1
    if frame_count > 0 and not camera_counts:
        errors.append("episode has no camera images")
    for name, count in sorted(camera_counts.items()):
        coverage = count / max(1, frame_count)
        if coverage < 0.95:
            errors.append(f"camera {name} coverage {count}/{frame_count} below 95%")

    robot_counts = {
        "left": {"valid_q8": 0, "valid_gripper": 0},
        "right": {"valid_q8": 0, "valid_gripper": 0},
    }
    for item in items:
        for side in ("left", "right"):
            if _finite_numbers(_arm_qpos(item, side), expected_len=7):
                robot_counts[side]["valid_q8"] += 1
            if _gripper_qpos(item, side) is not None:
                robot_counts[side]["valid_gripper"] += 1
    for side in ("left", "right"):
        if frame_count > 0 and robot_counts[side]["valid_q8"] < frame_count:
            errors.append(f"{side}_arm qpos missing or invalid")

    sample_times = [value for value in (_sample_time_ns(item) for item in items) if value is not None]
    duration_sec = 0.0
    observed_fps = 0.0
    if len(sample_times) >= 2:
        duration_sec = max(0.0, (sample_times[-1] - sample_times[0]) / 1_000_000_000.0)
        if duration_sec > 0:
            observed_fps = (len(sample_times) - 1) / duration_sec
    info = loaded.get("info", {})
    image_info = info.get("image", {}) if isinstance(info, dict) else {}
    expected_fps = float(image_info.get("fps", 0.0) or 0.0) if isinstance(image_info, dict) else 0.0
    checks = {
        "frame_count": {"ok": frame_count > 0, "count": frame_count},
        "camera_coverage": {"ok": not any("camera" in error for error in errors), "counts": camera_counts},
        "robot_state": {"ok": not any("qpos" in error for error in errors), "counts": robot_counts},
        "fps": {"ok": observed_fps > 0.0, "observed": observed_fps, "expected": expected_fps},
    }
    level = "error" if errors else "warning" if warnings else "ok"
    return {
        "checked_at_ns": time.time_ns(),
        "episode_name": episode_dir.name,
        "episode_dir": str(episode_dir),
        "level": level,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "observed_fps": observed_fps,
        "expected_fps": expected_fps,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def summarize_episode(episode_dir: Path) -> dict[str, Any]:
    payload = load_episode_json(episode_dir)
    items = [item for item in payload.get("data", []) if isinstance(item, dict)]
    validation = validate_episode(episode_dir, payload)
    stat = (episode_dir / "data.json").stat()
    return {
        "name": episode_dir.name,
        "path": str(episode_dir),
        "frame_count": int(validation["frame_count"]),
        "duration_sec": float(validation["duration_sec"]),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "validation_level": str(validation["level"]),
        "validation": validation,
        "cameras": _camera_list(items),
    }


def _image_time_ns(path: Path) -> int | None:
    match = re.search(r"_(\d+)\.[^.]+$", path.name)
    return int(match.group(1)) if match else None


def summarize_episode_fast(episode_dir: Path) -> dict[str, Any]:
    data_path = episode_dir / "data.json"
    stat = data_path.stat()
    with data_path.open("rb") as handle:
        handle.seek(max(0, stat.st_size - 2048))
        tail = handle.read().rstrip()
    finalized = tail.endswith(b"}")
    frame_count = 0
    camera_dirs = {
        "head": episode_dir / "colors" / "head",
        "left_wrist": episode_dir / "colors" / "wrist_left",
        "right_wrist": episode_dir / "colors" / "wrist_right",
    }
    camera_counts: dict[str, int] = {}
    camera_times: list[int] = []
    for name, path in camera_dirs.items():
        if not path.is_dir():
            continue
        images = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if images:
            camera_counts[name] = len(images)
            frame_count = max(frame_count, len(images))
            camera_times.extend(value for value in (_image_time_ns(image) for image in images) if value is not None)
    duration_sec = 0.0
    if len(camera_times) >= 2:
        duration_sec = max(0.0, (max(camera_times) - min(camera_times)) / 1_000_000_000.0)
    if not finalized or frame_count <= 0:
        level = "error"
    elif camera_counts and any(count / max(1, frame_count) < 0.95 for count in camera_counts.values()):
        level = "warning"
    else:
        level = "ok"
    return {
        "name": episode_dir.name,
        "path": str(episode_dir),
        "frame_count": int(frame_count),
        "duration_sec": float(duration_sec),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "validation_level": level,
        "validation": {},
        "cameras": [
            {"camera_id": _camera_id(name), "camera_name": name, "camera_mode": "rgb"}
            for name in sorted(camera_counts, key=_camera_id)
        ],
    }


def list_episodes(root_dir: str | None, fallback: str | None = None, limit: int | None = None) -> dict[str, Any]:
    root = resolve_root(root_dir, fallback)
    if not root.exists():
        return {"ok": True, "root_dir": str(root), "episodes": []}
    episode_dirs = [
        path for path in sorted(root.glob("episode_*"), reverse=True)
        if path.is_dir() and (path / "data.json").is_file()
    ]
    if limit is not None and int(limit) > 0:
        episode_dirs = episode_dirs[: int(limit)]
    return {
        "ok": True,
        "root_dir": str(root),
        "episodes": [summarize_episode_fast(path) for path in episode_dirs],
    }


def latest_episode_summary(root_dir: str | None, fallback: str | None = None) -> dict[str, Any] | None:
    root = resolve_root(root_dir, fallback)
    if not root.exists():
        return None
    episode_dirs = [
        path for path in sorted(root.glob("episode_*"), reverse=True)
        if path.is_dir() and (path / "data.json").is_file()
    ]
    if not episode_dirs:
        return None
    return summarize_episode(episode_dirs[0])


@dataclass
class PlaybackSession:
    root_dir: Path | None = None
    episode_dir: Path | None = None
    episode_name: str = ""
    items: list[dict[str, Any]] = field(default_factory=list)
    cameras: list[dict[str, Any]] = field(default_factory=list)
    frame_index: int = 0
    state: str = "disabled"
    play_started_monotonic: float = 0.0
    play_started_frame: int = 0

    def load(self, root_dir: str, episode: str) -> dict[str, Any]:
        root = resolve_root(root_dir)
        episode_dir = root / str(episode)
        if not episode_dir.is_dir():
            raise FileNotFoundError(f"episode directory not found: {episode_dir}")
        items = episode_items(episode_dir)
        self.root_dir = root
        self.episode_dir = episode_dir
        self.episode_name = episode_dir.name
        self.items = items
        self.cameras = _camera_list(items)
        self.frame_index = 0
        self.state = "paused" if items else "empty"
        self.play_started_monotonic = 0.0
        self.play_started_frame = 0
        return {"ok": True, "playback": self.status()}

    def status(self) -> dict[str, Any]:
        self._advance_if_playing()
        total = len(self.items)
        idx = min(max(0, self.frame_index), max(0, total - 1))
        timestamp_ns = _sample_time_ns(self.items[idx]) if total else None
        duration_sec = self._duration_sec()
        current_time_sec = self._frame_time_sec(idx)
        return {
            "enabled": total > 0,
            "state": self.state,
            "episode_name": self.episode_name,
            "episode_dir": str(self.episode_dir or ""),
            "frame_index": idx,
            "total_frames": total,
            "current_time_sec": current_time_sec,
            "duration_sec": duration_sec,
            "timestamp_ns": timestamp_ns,
            "cameras": self.cameras,
        }

    def start(self) -> dict[str, Any]:
        if not self.items:
            raise ValueError("no episode loaded")
        self.state = "playing"
        self.play_started_monotonic = time.monotonic()
        self.play_started_frame = self.frame_index
        return self.status()

    def pause(self) -> dict[str, Any]:
        self._advance_if_playing()
        self.state = "paused" if self.items else "disabled"
        return self.status()

    def stop(self) -> dict[str, Any]:
        self.frame_index = 0
        self.state = "paused" if self.items else "disabled"
        self.play_started_monotonic = 0.0
        self.play_started_frame = 0
        return self.status()

    def seek(self, frame: int) -> dict[str, Any]:
        total = len(self.items)
        self.frame_index = min(max(0, int(frame)), max(0, total - 1))
        self.play_started_monotonic = time.monotonic()
        self.play_started_frame = self.frame_index
        return self.status()

    def image_path(self, camera_id: int, frame: int) -> Path:
        if self.episode_dir is None:
            raise ValueError("no episode loaded")
        total = len(self.items)
        if total <= 0:
            raise ValueError("loaded episode has no frames")
        idx = min(max(0, int(frame)), total - 1)
        colors = _camera_entries(self.items[idx])
        rel_path = ""
        for name in CAMERA_NAMES_BY_ID.get(int(camera_id), ()):
            if name in colors:
                rel_path = colors[name]
                break
        if not rel_path:
            raise FileNotFoundError(f"camera_id={camera_id} has no image at frame={idx}")
        target = (self.episode_dir / rel_path).resolve()
        episode_root = self.episode_dir.resolve()
        if episode_root not in target.parents:
            raise ValueError(f"image path escapes episode directory: {target}")
        if not target.is_file():
            raise FileNotFoundError(f"image file not found: {target}")
        return target

    def curves(self, max_points: int = 900) -> dict[str, Any]:
        total = len(self.items)
        stride = max(1, math.ceil(total / max(1, int(max_points))))
        selected = self.items[::stride]
        left = {f"joint{i + 1}": [] for i in range(7)}
        right = {f"joint{i + 1}": [] for i in range(7)}
        left["gripper_state"] = []
        right["gripper_state"] = []
        frame_indices: list[int] = []
        for item in selected:
            frame_indices.append(int(item.get("idx", len(frame_indices))))
            left_q = _arm_qpos(item, "left")
            right_q = _arm_qpos(item, "right")
            for index in range(7):
                left[f"joint{index + 1}"].append(float(left_q[index]) if len(left_q) > index else None)
                right[f"joint{index + 1}"].append(float(right_q[index]) if len(right_q) > index else None)
            left["gripper_state"].append(_gripper_qpos(item, "left"))
            right["gripper_state"].append(_gripper_qpos(item, "right"))
        return {
            "ok": True,
            "total_frames": total,
            "sample_count": len(selected),
            "sample_stride": stride,
            "frame_indices": frame_indices,
            "left": left,
            "right": right,
        }

    def _duration_sec(self) -> float:
        times = [value for value in (_sample_time_ns(item) for item in self.items) if value is not None]
        if len(times) < 2:
            return 0.0
        return max(0.0, (times[-1] - times[0]) / 1_000_000_000.0)

    def _frame_time_sec(self, frame_index: int) -> float:
        if not self.items:
            return 0.0
        first = _sample_time_ns(self.items[0])
        current = _sample_time_ns(self.items[min(max(0, frame_index), len(self.items) - 1)])
        if first is None or current is None:
            return 0.0
        return max(0.0, (current - first) / 1_000_000_000.0)

    def _advance_if_playing(self) -> None:
        if self.state != "playing" or not self.items:
            return
        elapsed = max(0.0, time.monotonic() - self.play_started_monotonic)
        target_time = self._frame_time_sec(self.play_started_frame) + elapsed
        best = self.frame_index
        for index in range(self.play_started_frame, len(self.items)):
            if self._frame_time_sec(index) <= target_time:
                best = index
            else:
                break
        self.frame_index = best
        if self.frame_index >= len(self.items) - 1:
            self.state = "paused"
