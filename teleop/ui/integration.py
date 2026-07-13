from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from teleop.ui.command_bus import UiCommand, UiCommandName
from teleop.ui.payload import build_camera_status, build_web_payload


KEY_BY_COMMAND = {
    UiCommandName.START: "r",
    UiCommandName.STOP: "q",
    UiCommandName.HOME: "h",
    UiCommandName.RECENTER: "c",
    UiCommandName.RECORD_TOGGLE: "s",
    UiCommandName.RECORD_CANCEL: "v",
}


def dispatch_ui_commands(commands: list[UiCommand], on_press: Callable[[str], None]) -> list[str]:
    pressed_keys: list[str] = []
    for command in commands:
        key = KEY_BY_COMMAND.get(command.name)
        if key is None:
            continue
        on_press(key)
        pressed_keys.append(key)
    return pressed_keys


def build_runtime_recording_status(
    *,
    args,
    recorder: Any,
    recording_flow: Any,
    record_running: bool,
) -> dict[str, Any]:
    task_root = Path(str(args.task_dir)) / str(args.task_name)
    flow_state = getattr(recording_flow, "state", None)
    waiting_for_first_frame = bool(getattr(flow_state, "waiting_for_first_frame", False))
    pending_samples = getattr(flow_state, "pending_samples", [])
    pending_count = len(pending_samples) if pending_samples is not None else 0
    active = bool(record_running or waiting_for_first_frame)
    frame_index = int(getattr(recorder, "item_id", -1) or -1) + 1 if recorder is not None else 0
    if frame_index < 0:
        frame_index = 0
    session_dir = str(getattr(recorder, "episode_dir", "") or "")
    phase = "recording" if record_running else "armed" if waiting_for_first_frame else "idle"
    record_start_monotonic_ns = getattr(flow_state, "record_start_monotonic_ns", None)
    return {
        "is_recording": active,
        "active": active,
        "enabled": bool(getattr(args, "record", False)),
        "phase": phase,
        "session_dir": session_dir,
        "root_dir": str(task_root),
        "active_root_dir": str(task_root),
        "fps": float(args.frequency),
        "frame_index": frame_index,
        "error": "",
        "last_alignment": {
            "waiting_for_first_frame": waiting_for_first_frame,
            "pending_samples": int(pending_count),
            "record_start_monotonic_ns": record_start_monotonic_ns,
        },
        "last_alert": {},
        "alert_seq": 0,
        "last_validation": {},
    }


def build_runtime_camera_status(cameras: Any, *, now_monotonic_ns: int | None = None) -> dict[str, Any]:
    latest_meta_by_name: dict[str, dict[str, Any] | None] = {}
    sources = cameras.sources() if cameras is not None else {}
    for name, source in sources.items():
        if source is None:
            latest_meta_by_name[str(name)] = None
            continue
        _, meta = source.get_latest(copy=False)
        latest_meta_by_name[str(name)] = meta
    return build_camera_status(latest_meta_by_name, now_monotonic_ns=now_monotonic_ns)


def build_runtime_web_payload(
    *,
    args,
    recorder: Any,
    recording_flow: Any,
    record_running: bool,
    current_lr_arm_q: Any | None = None,
    current_state_sample_ns: int | None = None,
    started: bool = False,
    ready: bool = False,
    stopping: bool = False,
    provider_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    left_q_fb = None
    right_q_fb = None
    if current_lr_arm_q is not None:
        arm_q = np.asarray(current_lr_arm_q, dtype=float).reshape(-1)
        if arm_q.shape[0] != 14:
            raise ValueError(f"UI arm state must contain 14 joints, got shape={arm_q.shape}")
        left_q_fb = arm_q[:7].tolist()
        right_q_fb = arm_q[-7:].tolist()
    return build_web_payload(
        left_q_fb=left_q_fb,
        right_q_fb=right_q_fb,
        left_stamp_fb_ns=current_state_sample_ns,
        right_stamp_fb_ns=current_state_sample_ns,
        recording_status=build_runtime_recording_status(
            args=args,
            recorder=recorder,
            recording_flow=recording_flow,
            record_running=record_running,
        ),
        active_root_dir=str(Path(str(args.task_dir)) / str(args.task_name)),
        playback_status={"state": "disabled", "error": "playback is not implemented in xr_teleoperate UI"},
        convert_status={"ok": True, "state": "idle", "phase": "idle", "running": False, "message": "ready: LeRobot v2 raw exporter is available"},
        provider_status=provider_status,
        teleop_status={
            "started": bool(started),
            "ready": bool(ready),
            "stopping": bool(stopping),
        },
    )
