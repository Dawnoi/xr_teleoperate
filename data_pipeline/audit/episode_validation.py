"""Finalized raw-episode validation and durable report writing."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from teleop.ui.episode_store import validate_episode
from teleop.validation.action_semantics import validate_action_semantics
from teleop.validation.mobile_training import validate_mobile_training
from teleop.validation.time_alignment import validate_time_alignment


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTION_SEMANTICS_CONFIG = REPO_ROOT / "configs" / "data_quality" / "g1_29_action_semantics.json"
DEFAULT_TIME_ALIGNMENT_CONFIG = REPO_ROOT / "configs" / "data_quality" / "g1_29_time_alignment.json"
DEFAULT_MOBILE_TRAINING_CONFIG = REPO_ROOT / "configs" / "data_quality" / "g1_d_mobile_training.json"
CAMERA_MANIFEST_ORDER = ("head", "left_wrist", "right_wrist")
CAMERA_NAME_ALIASES = {
    "head": "head",
    "left_wrist": "left_wrist",
    "wrist_left": "left_wrist",
    "right_wrist": "right_wrist",
    "wrist_right": "right_wrist",
}


def _data_manifest(data_path: Path) -> dict[str, int]:
    stat = data_path.stat()
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _read_enabled_cameras(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    info = payload.get("info")
    if not isinstance(info, dict):
        return [], ["data.json info must be an object containing enabled_cameras"]
    raw_cameras = info.get("enabled_cameras")
    if not isinstance(raw_cameras, list):
        return [], ["data.json info.enabled_cameras must be an explicit list"]

    enabled_cameras = []
    errors = []
    for index, raw_name in enumerate(raw_cameras):
        if not isinstance(raw_name, str) or not raw_name.strip():
            errors.append(f"enabled_cameras[{index}] must be a non-empty string")
            continue
        canonical_name = CAMERA_NAME_ALIASES.get(raw_name.strip())
        if canonical_name is None:
            errors.append(f"enabled_cameras[{index}] is unsupported: {raw_name}")
            continue
        if canonical_name in enabled_cameras:
            errors.append(f"enabled_cameras contains duplicate camera: {canonical_name}")
            continue
        enabled_cameras.append(canonical_name)
    return sorted(enabled_cameras, key=CAMERA_MANIFEST_ORDER.index), errors


def _manifest_structural_errors(
    episode_dir: Path,
    items: list[dict[str, Any]],
    enabled_cameras: list[str],
) -> list[str]:
    errors = []
    for item_index, item in enumerate(items):
        colors = item.get("colors")
        timestamps = item.get("timestamps")
        camera_timestamps = timestamps.get("camera") if isinstance(timestamps, dict) else None
        for camera_name in enabled_cameras:
            image_path = colors.get(camera_name) if isinstance(colors, dict) else None
            if not isinstance(image_path, str) or not image_path:
                errors.append(f"item {item_index} missing enabled camera image: {camera_name}")
            elif not (episode_dir / image_path).is_file():
                errors.append(f"item {item_index} missing enabled camera file: {camera_name}: {image_path}")

            camera_meta = camera_timestamps.get(camera_name) if isinstance(camera_timestamps, dict) else None
            if not isinstance(camera_meta, dict):
                errors.append(f"item {item_index} missing enabled camera timestamp: {camera_name}")
                continue
            if camera_meta.get("host_recv_monotonic_ns") is None:
                errors.append(f"item {item_index} missing host receive camera timestamp: {camera_name}")
    return errors


def _report_messages(report: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    if not isinstance(errors, list) or not isinstance(warnings, list):
        raise ValueError("validation report errors and warnings must be lists")
    if errors or warnings:
        return [_message_text(message) for message in errors], [_message_text(message) for message in warnings]

    issues = report.get("issues", [])
    if not isinstance(issues, list):
        raise ValueError("validation report issues must be a list")
    issue_errors = []
    issue_warnings = []
    for issue in issues:
        if not isinstance(issue, dict):
            raise ValueError("validation report issues must contain objects")
        message = _message_text(issue)
        if issue.get("severity") == "error":
            issue_errors.append(message)
        elif issue.get("severity") == "warning":
            issue_warnings.append(message)
    return issue_errors, issue_warnings


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("message", message))
    return str(message)


def _load_time_alignment_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"time alignment config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("time alignment config must be a JSON object")
    return config


def _run_time_alignment(items, enabled_cameras, *, config_path):
    return validate_time_alignment(items, enabled_cameras, _load_time_alignment_config(config_path))


def _run_mobile_training_validation(items, *, episode_info, config_path):
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"mobile training validation config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    return validate_mobile_training(items, config, episode_info=episode_info)


def _parse_error_report(episode_dir: Path, data_path: Path, error: json.JSONDecodeError) -> dict[str, Any]:
    message = f"data.json parse error: {error.msg} at line {error.lineno}, column {error.colno}"
    return {
        "schema_version": 1,
        "checked_at_ns": time.time_ns(),
        "episode_name": episode_dir.name,
        "episode_dir": str(episode_dir),
        "source_data": _data_manifest(data_path),
        "level": "error",
        "frame_count": 0,
        "duration_sec": 0.0,
        "observed_fps": 0.0,
        "expected_fps": 0.0,
        "checks": {"data_json": {"ok": False, "error": message}},
        "manifest": {"enabled_cameras": [], "status": "not_applicable"},
        "time_alignment": {
            "status": "not_applicable",
            "errors": [],
            "warnings": [],
            "not_applicable_reason": "data.json cannot be parsed",
        },
        "action_semantics": {
            "status": "not_applicable",
            "not_applicable_reason": "data.json cannot be parsed",
            "errors": [],
            "warnings": [],
        },
        "errors": [message],
        "warnings": [],
    }


def validate_finalized_episode(
    episode_dir: str | Path,
    *,
    action_semantics_config: str | Path = DEFAULT_ACTION_SEMANTICS_CONFIG,
    time_alignment_config: str | Path = DEFAULT_TIME_ALIGNMENT_CONFIG,
    mobile_training_config: str | Path = DEFAULT_MOBILE_TRAINING_CONFIG,
) -> dict[str, Any]:
    """Validate one finalized episode.

    JSON decoding is the single expected file-boundary failure: it becomes a
    durable validation error so the UI and exporter do not mistake corruption
    for a missing report. Other programming/configuration errors intentionally
    propagate to preserve their traceback.
    """
    episode_path = Path(episode_dir)
    data_path = episode_path / "data.json"
    raw_text = data_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as error:
        return _parse_error_report(episode_path, data_path, error)

    if not isinstance(payload, dict):
        message = "data.json root must be an object"
        return _parse_error_report(episode_path, data_path, json.JSONDecodeError(message, raw_text, 0))

    enabled_cameras, manifest_errors = _read_enabled_cameras(payload)
    if manifest_errors:
        return {
            "schema_version": 1,
            "checked_at_ns": time.time_ns(),
            "episode_name": episode_path.name,
            "episode_dir": str(episode_path),
            "source_data": _data_manifest(data_path),
            "level": "error",
            "frame_count": 0,
            "duration_sec": 0.0,
            "observed_fps": 0.0,
            "expected_fps": 0.0,
            "checks": {"data_json": {"ok": False, "errors": manifest_errors}},
            "manifest": {"enabled_cameras": enabled_cameras, "status": "error", "errors": manifest_errors},
            "time_alignment": {
                "status": "not_applicable",
                "errors": [],
                "warnings": [],
                "not_applicable_reason": "enabled_cameras manifest is invalid",
            },
            "action_semantics": {
                "status": "not_applicable",
                "not_applicable_reason": "enabled_cameras manifest is invalid",
                "errors": [],
                "warnings": [],
            },
            "errors": manifest_errors,
            "warnings": [],
        }

    items = payload.get("data")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        message = "data.json data must be a list of objects"
        return {
            "schema_version": 1,
            "checked_at_ns": time.time_ns(),
            "episode_name": episode_path.name,
            "episode_dir": str(episode_path),
            "source_data": _data_manifest(data_path),
            "level": "error",
            "frame_count": 0,
            "duration_sec": 0.0,
            "observed_fps": 0.0,
            "expected_fps": 0.0,
            "checks": {"data_json": {"ok": False, "error": message}},
            "manifest": {"enabled_cameras": enabled_cameras, "status": "ok"},
            "time_alignment": {
                "status": "not_applicable",
                "errors": [],
                "warnings": [],
                "not_applicable_reason": message,
            },
            "action_semantics": {
                "status": "not_applicable",
                "not_applicable_reason": message,
                "errors": [],
                "warnings": [],
            },
            "errors": [message],
            "warnings": [],
        }

    structural = validate_episode(episode_path, payload)
    structural = dict(structural)
    manifest_errors = _manifest_structural_errors(episode_path, items, enabled_cameras)
    structural_errors = [*list(structural.get("errors", [])), *manifest_errors]
    structural["errors"] = structural_errors
    structural["level"] = "error" if structural_errors else str(structural.get("level", "ok"))
    structural_checks = dict(structural.get("checks", {}))
    structural_checks["camera_manifest"] = {
        "ok": not manifest_errors,
        "enabled_cameras": enabled_cameras,
    }
    structural["checks"] = structural_checks

    semantics = validate_action_semantics(
        items=items,
        config_path=Path(action_semantics_config),
    )
    semantic_errors, semantic_warnings = _report_messages(semantics)
    semantics["errors"] = semantic_errors
    semantics["warnings"] = semantic_warnings
    time_alignment = _run_time_alignment(
        items,
        enabled_cameras,
        config_path=Path(time_alignment_config),
    )
    if not isinstance(time_alignment, dict):
        raise ValueError("time alignment report must be an object")
    alignment_errors, alignment_warnings = _report_messages(time_alignment)
    time_alignment["errors"] = alignment_errors
    time_alignment["warnings"] = alignment_warnings
    mobile_training = _run_mobile_training_validation(
        items,
        episode_info=payload.get("info"),
        config_path=mobile_training_config,
    )
    mobile_errors, mobile_warnings = _report_messages(mobile_training)
    mobile_training["errors"] = mobile_errors
    mobile_training["warnings"] = mobile_warnings

    errors = [*structural_errors, *alignment_errors, *semantic_errors, *mobile_errors]
    warnings = [*list(structural.get("warnings", [])), *alignment_warnings, *semantic_warnings, *mobile_warnings]
    level = "error" if errors else "warning" if warnings else "ok"
    return {
        **structural,
        "schema_version": 1,
        "source_data": _data_manifest(data_path),
        "level": level,
        "manifest": {"enabled_cameras": enabled_cameras, "status": "ok"},
        "structural": structural,
        "time_alignment": time_alignment,
        "action_semantics": semantics,
        "mobile_training": mobile_training,
        "errors": errors,
        "warnings": warnings,
    }


def write_validation_report(episode_dir: str | Path, report: dict[str, Any]) -> Path:
    """Atomically publish a finished report next to the validated data.json."""
    episode_path = Path(episode_dir)
    target = episode_path / "validation.json"
    temporary = episode_path / f".{target.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target
