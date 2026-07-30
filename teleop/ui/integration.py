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
    base_state_receiver: Any | None = None,
    base_stop_state: dict[str, Any] | None = None,
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
    validation_manager = getattr(recording_flow, "validation_manager", None)
    validation_status = validation_manager.status() if validation_manager is not None else {}
    validation_pending = bool(validation_status.get("pending", False))
    last_validation = validation_status.get("last_validation", {})
    if validation_pending and not active:
        phase = "validating"
    base_enabled = bool(getattr(args, "record_base", False))
    base_status = {
        "enabled": base_enabled,
        "receiver_alive": bool(base_state_receiver.is_alive()) if base_state_receiver is not None else False,
        "odom_topic": str(getattr(args, "base_odom_topic", "") or ""),
        "height_topic": str(getattr(args, "base_height_topic", "") or ""),
        "state_max_age_ms": float(getattr(args, "base_state_max_age_ms", 0.0) or 0.0),
        "action_max_age_ms": float(max(80.0, (2.5 / max(float(args.frequency), 1e-6)) * 1000.0)),
        "stop_confirmed": bool((base_stop_state or {}).get("latched", True)),
        "control_fault": bool((base_stop_state or {}).get("fault", False)),
        "fault_reason": str((base_stop_state or {}).get("error", "") or ""),
    }
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
        "validation_pending": validation_pending,
        "validation_current_episode_dir": validation_status.get("current_episode_dir", ""),
        "validation_queued_episode_dirs": validation_status.get("queued_episode_dirs", []),
        "last_validation": last_validation,
        "base": base_status,
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
    current_left_gripper_q: float | None = None,
    current_right_gripper_q: float | None = None,
    current_left_gripper_cmd: float | None = None,
    current_right_gripper_cmd: float | None = None,
    current_state_sample_ns: int | None = None,
    started: bool = False,
    ready: bool = False,
    stopping: bool = False,
    provider_status: dict[str, Any] | None = None,
    latency_snapshot: dict[str, Any] | None = None,
    timing_snapshot: dict[str, Any] | None = None,
    base_state_receiver: Any | None = None,
    base_stop_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    left_q_fb = None
    right_q_fb = None
    gripper_feedback = (current_left_gripper_q, current_right_gripper_q)
    gripper_command = (current_left_gripper_cmd, current_right_gripper_cmd)
    if (gripper_feedback[0] is None) != (gripper_feedback[1] is None):
        raise ValueError("UI gripper feedback must provide both left and right values")
    if (gripper_command[0] is None) != (gripper_command[1] is None):
        raise ValueError("UI gripper command must provide both left and right values")
    if (gripper_feedback[0] is None) != (gripper_command[0] is None):
        raise ValueError("UI gripper feedback and command must be provided together")
    if current_lr_arm_q is not None:
        arm_q = np.asarray(current_lr_arm_q, dtype=float).reshape(-1)
        if arm_q.shape[0] != 14:
            raise ValueError(f"UI arm state must contain 14 joints, got shape={arm_q.shape}")
        if gripper_feedback[0] is None:
            left_q_fb = arm_q[:7].tolist()
            right_q_fb = arm_q[-7:].tolist()
        else:
            left_gripper_q = float(gripper_feedback[0])
            right_gripper_q = float(gripper_feedback[1])
            left_gripper_cmd = float(gripper_command[0])
            right_gripper_cmd = float(gripper_command[1])
            if not np.isfinite([left_gripper_q, right_gripper_q, left_gripper_cmd, right_gripper_cmd]).all():
                raise ValueError("UI gripper feedback and command must contain finite values")
            left_q_fb = [*arm_q[:7].tolist(), left_gripper_q]
            right_q_fb = [*arm_q[-7:].tolist(), right_gripper_q]
    elif gripper_feedback[0] is not None:
        raise ValueError("UI gripper feedback and command require current dual-arm state")
    return build_web_payload(
        left_q_fb=left_q_fb,
        right_q_fb=right_q_fb,
        left_gripper_q_fb=None if gripper_feedback[0] is None else float(gripper_feedback[0]),
        right_gripper_q_fb=None if gripper_feedback[1] is None else float(gripper_feedback[1]),
        left_gripper_q_cmd=None if gripper_command[0] is None else float(gripper_command[0]),
        right_gripper_q_cmd=None if gripper_command[1] is None else float(gripper_command[1]),
        left_stamp_fb_ns=current_state_sample_ns,
        right_stamp_fb_ns=current_state_sample_ns,
        recording_status=build_runtime_recording_status(
            args=args,
            recorder=recorder,
            recording_flow=recording_flow,
            record_running=record_running,
            base_state_receiver=base_state_receiver,
            base_stop_state=base_stop_state,
        ),
        active_root_dir=str(Path(str(args.task_dir)) / str(args.task_name)),
        playback_status={"state": "disabled", "error": "playback is not implemented in xr_teleoperate UI"},
        convert_status={"ok": True, "state": "idle", "phase": "idle", "running": False, "message": "ready: LeRobot v2 raw exporter is available"},
        provider_status=provider_status,
        teleop_latency=latency_snapshot,
        teleop_timing=timing_snapshot,
        teleop_status={
            "started": bool(started),
            "ready": bool(ready),
            "stopping": bool(stopping),
        },
    )
