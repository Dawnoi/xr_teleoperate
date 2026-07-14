from __future__ import annotations

import time
from typing import Any


CAMERA_ID_BY_NAME = {
    "head": 0,
    "left_wrist": 1,
    "right_wrist": 2,
}


def build_web_payload(
    *,
    left_q_fb: list[float] | None = None,
    right_q_fb: list[float] | None = None,
    left_gripper_q_fb: float | None = None,
    right_gripper_q_fb: float | None = None,
    left_gripper_q_cmd: float | None = None,
    right_gripper_q_cmd: float | None = None,
    left_stamp_fb_ns: int | None = None,
    right_stamp_fb_ns: int | None = None,
    left_fk_flange_pose: dict[str, Any] | None = None,
    right_fk_flange_pose: dict[str, Any] | None = None,
    recording_status: dict[str, Any] | None = None,
    active_root_dir: str = "",
    playback_status: dict[str, Any] | None = None,
    convert_status: dict[str, Any] | None = None,
    provider_status: dict[str, Any] | None = None,
    updated_mono: float | None = None,
    teleop_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the same top-level payload shape as the reference data_collector UI."""

    recording = dict(recording_status or {})
    recording_active = bool(recording.get("is_recording", recording.get("active", False)))
    payload = {
        "schema": "data_collector/v1",
        "left": {
            "q_fb": left_q_fb,
            "gripper_q_fb": left_gripper_q_fb,
            "gripper_q_cmd": left_gripper_q_cmd,
            "stamp_fb_ns": left_stamp_fb_ns,
            "fk_flange_pose": left_fk_flange_pose,
        },
        "right": {
            "q_fb": right_q_fb,
            "gripper_q_fb": right_gripper_q_fb,
            "gripper_q_cmd": right_gripper_q_cmd,
            "stamp_fb_ns": right_stamp_fb_ns,
            "fk_flange_pose": right_fk_flange_pose,
        },
        "recording_active": recording_active,
        "recording": {
            "active": recording_active,
            "enabled": bool(recording.get("enabled", True)),
            "phase": str(recording.get("phase", "recording" if recording_active else "idle") or "idle"),
            "session_dir": str(recording.get("session_dir", "") or ""),
            "root_dir": str(recording.get("root_dir", "") or ""),
            "active_root_dir": str(active_root_dir or recording.get("active_root_dir", "") or ""),
            "fps": float(recording.get("fps", 0.0) or 0.0),
            "frame_index": int(recording.get("frame_index", 0) or 0),
            "error": str(recording.get("error", "") or ""),
            "last_alignment": recording.get("last_alignment", {}),
            "last_alert": recording.get("last_alert", {}),
            "alert_seq": int(recording.get("alert_seq", 0) or 0),
            "last_validation": recording.get("last_validation", {}),
        },
        "playback": dict(playback_status or {}),
        "convert": dict(convert_status or {}),
        "provider": dict(provider_status or {"active_provider": "unknown", "real_replay": {"state": "disabled"}}),
        "updated_mono": float(time.monotonic() if updated_mono is None else updated_mono),
    }
    if teleop_status is not None:
        payload["teleop"] = dict(teleop_status)
    return payload


def build_camera_status(
    latest_meta_by_name: dict[str, dict[str, Any] | None],
    *,
    now_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Build /camera/status using field names consumed by the reference frontend."""

    now_ns = int(time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns)
    streams: list[dict[str, Any]] = []
    for camera_name, camera_id in CAMERA_ID_BY_NAME.items():
        meta = latest_meta_by_name.get(camera_name)
        if meta is None:
            continue
        timestamp_ns = _camera_meta_monotonic_ns(meta)
        shape = meta.get("shape") or []
        frame_width = int(shape[0]) if len(shape) >= 1 else 0
        frame_height = int(shape[1]) if len(shape) >= 2 else 0
        channels = int(shape[2]) if len(shape) >= 3 else 0
        frame_seq = int(meta.get("frame_seq", -1) if meta.get("frame_seq") is not None else -1)
        streams.append(
            {
                "camera_id": int(camera_id),
                "camera_name": str(camera_name),
                "camera_role": str(camera_name),
                "camera_mode": "rgb",
                "url": f"/camera/frame?camera_id={int(camera_id)}",
                "shared_seq": frame_seq,
                "shared_timestamp_ns": timestamp_ns,
                "shared_age_ms": int((now_ns - timestamp_ns) / 1_000_000) if timestamp_ns > 0 else -1,
                "requested_capture": None,
                "actual_capture": {
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "channels": channels,
                    "transport": str(meta.get("transport", "")),
                },
            }
        )

    primary = streams[0] if streams else None
    return {
        "url": str(primary["url"]) if primary is not None else "",
        "managed_process_alive": bool(streams),
        "managed_pid": None,
        "camera_id": int(primary["camera_id"]) if primary is not None else None,
        "camera_name": str(primary["camera_name"]) if primary is not None else "",
        "camera_mode": str(primary["camera_mode"]) if primary is not None else "",
        "shared_seq": int(primary["shared_seq"]) if primary is not None else -1,
        "shared_timestamp_ns": int(primary["shared_timestamp_ns"]) if primary is not None else 0,
        "shared_age_ms": int(primary["shared_age_ms"]) if primary is not None else -1,
        "requested_capture": primary["requested_capture"] if primary is not None else None,
        "actual_capture": primary["actual_capture"] if primary is not None else None,
        "active_camera_ids": [int(stream["camera_id"]) for stream in streams],
        "streams": streams,
        "camera_name_map": {str(stream["camera_id"]): stream["camera_name"] for stream in streams},
        "name_presets": [
            {"name": "head", "role": "head"},
            {"name": "left_wrist", "role": "left_wrist"},
            {"name": "right_wrist", "role": "right_wrist"},
        ],
        "last_error": "",
    }


def _camera_meta_monotonic_ns(meta: dict[str, Any]) -> int:
    value = meta.get("host_recv_monotonic_ns")
    if value is None:
        value = meta.get("host_monotonic_ns")
    return int(value or 0)
