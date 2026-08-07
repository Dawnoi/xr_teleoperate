"""机械臂命令流水线：把输入目标转换成安全的关节命令。

Arm command target pipeline extracted from the real teleop loop.

The caller still owns side effects outside arm target generation: publishing arm
commands, mutating START/STOP, and reporting provider feedback to the input
provider. This module only computes the command target, safety limiting, gravity
feed-forward, and the state values that the main loop needs to write back.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
import logging
import time

import numpy as np

from core.control.arm_target_safety import limit_arm_joint_target_velocity
from core.control.arm_workspace_safety import (
    clamp_dual_wrist_poses_to_box,
    clamp_dual_wrist_poses_to_side_workspaces,
    clamp_dual_wrist_poses_to_tapered_workspace,
)
from inference.online_session import online_inference_speed_limit_delta

logger = logging.getLogger(__name__)


class UnsupportedMotionIntentError(ValueError):
    pass


class _NonFiniteVectorError(ValueError):
    pass


@dataclass
class ArmCommandResult:
    """Result of one arm-command pipeline step."""

    sol_q: np.ndarray
    sol_tauff: np.ndarray
    current_hold_q: np.ndarray
    current_hold_tauff: np.ndarray
    provider_feedback: Optional[dict]
    ik_ms: float
    safety_ms: float
    gravity_ms: float
    arm_cmd_input_ms: float
    arm_cmd_takeover_reset_ms: float
    arm_cmd_target_extra_ms: float
    arm_cmd_feedback_gate_ms: float
    arm_cmd_enable_gating_ms: float
    arm_cmd_takeover_settle_ms: float
    arm_cmd_speed_feedback_ms: float
    arm_cmd_hold_update_ms: float
    sol_q_before_speed_limit: np.ndarray
    left_arm_enabled: bool
    right_arm_enabled: bool
    post_home_takeover_armed: bool
    left_takeover_settle_frames: int
    right_takeover_settle_frames: int


def _reset_arm_ik_state(arm_ik: Any, arm_q: Any) -> None:
    arm_q = np.asarray(arm_q, dtype=float).copy()
    if hasattr(arm_ik, "init_data"):
        arm_ik.init_data = arm_q.copy()
    smooth_filter = getattr(arm_ik, "smooth_filter", None)
    if smooth_filter is not None:
        try:
            smooth_filter._data_queue = [arm_q.copy() for _ in range(smooth_filter._window_size)]
            smooth_filter._filtered_data = arm_q.copy()
        except Exception:
            pass


def _require_finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape[0] != int(size):
        raise _NonFiniteVectorError(f"{name} must have shape ({size},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise _NonFiniteVectorError(f"{name} contains non-finite values")
    return arr


def _fatal_feedback(reason: str, **extra: Any) -> dict:
    feedback = {"fatal": True, "reason": reason}
    feedback.update(extra)
    return feedback


def _handle_takeover_reset(
    *,
    arm_ik: Any,
    current_lr_arm_q: np.ndarray,
    any_zero_takeover_this_frame: bool,
    post_home_takeover_armed: bool,
    normalized_head_mode: str,
    sync_reference_to_current_live_pose: Optional[Callable[..., Any]],
) -> bool:
    if not any_zero_takeover_this_frame:
        return bool(post_home_takeover_armed)
    if post_home_takeover_armed and normalized_head_mode in {"head_coupled", "hybrid"}:
        if callable(sync_reference_to_current_live_pose):
            sync_reference_to_current_live_pose(require_live=False)
    _reset_arm_ik_state(arm_ik, current_lr_arm_q)
    return False


def _solve_pose_target(
    *,
    arm_ik: Any,
    motion_intent: Any,
    current_lr_arm_q: np.ndarray,
    current_lr_arm_dq: np.ndarray,
    workspace_limit_enabled: bool,
    workspace_mode: str,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    tapered_workspace_params: Mapping[str, float],
    side_workspaces: Optional[Mapping[str, Mapping[str, Any]]],
    is_online_inference: bool,
    timing_debugger: Any,
    log: Any,
) -> tuple[np.ndarray, np.ndarray, float, Optional[dict]]:
    left_target_pose = motion_intent.left_wrist_pose
    right_target_pose = motion_intent.right_wrist_pose
    provider_feedback = None

    if workspace_limit_enabled:
        original_left_pose = np.asarray(left_target_pose, dtype=float).copy()
        original_right_pose = np.asarray(right_target_pose, dtype=float).copy()
        if side_workspaces is not None:
            left_target_pose, right_target_pose, workspace_was_clamped = clamp_dual_wrist_poses_to_side_workspaces(
                left_target_pose,
                right_target_pose,
                dict(side_workspaces),
            )
        elif workspace_mode == "box":
            left_target_pose, right_target_pose, workspace_was_clamped = clamp_dual_wrist_poses_to_box(
                left_target_pose,
                right_target_pose,
                workspace_min,
                workspace_max,
            )
        else:
            left_target_pose, right_target_pose, workspace_was_clamped = clamp_dual_wrist_poses_to_tapered_workspace(
                left_target_pose,
                right_target_pose,
                float(tapered_workspace_params["z_min"]),
                float(tapered_workspace_params["z_max"]),
                float(tapered_workspace_params["x_min"]),
                float(tapered_workspace_params["x_max_low"]),
                float(tapered_workspace_params["x_max_high"]),
                float(tapered_workspace_params["y_max_low"]),
                float(tapered_workspace_params["y_max_high"]),
            )
        if workspace_was_clamped and is_online_inference:
            left_clamp_delta = float(np.linalg.norm(np.asarray(left_target_pose)[:3, 3] - original_left_pose[:3, 3]))
            right_clamp_delta = float(np.linalg.norm(np.asarray(right_target_pose)[:3, 3] - original_right_pose[:3, 3]))
            provider_feedback = _fatal_feedback(
                "online inference action exceeded workspace",
                left_clamp_delta=left_clamp_delta,
                right_clamp_delta=right_clamp_delta,
            )

    time_ik_start = time.perf_counter()
    sol_q, sol_tauff = arm_ik.solve_ik(
        left_target_pose,
        right_target_pose,
        current_lr_arm_q,
        current_lr_arm_dq,
    )
    ik_dt = time.perf_counter() - time_ik_start
    add_ik = getattr(timing_debugger, "add_ik", None)
    if callable(add_ik):
        add_ik(ik_dt)
    log.debug(f"ik:\t{round(ik_dt, 6)}")
    return sol_q, sol_tauff, ik_dt * 1000.0, provider_feedback


def _solve_motion_target(
    *,
    arm_ik: Any,
    motion_intent: Any,
    current_lr_arm_q: np.ndarray,
    current_lr_arm_dq: np.ndarray,
    current_hold_tauff: np.ndarray,
    control_dt: float,
    workspace_limit_enabled: bool,
    workspace_mode: str,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    tapered_workspace_params: Mapping[str, float],
    side_workspaces: Optional[Mapping[str, Mapping[str, Any]]],
    is_online_inference: bool,
    timing_debugger: Any,
    log: Any,
) -> tuple[np.ndarray, np.ndarray, float, Optional[dict]]:
    motion_kind = str(getattr(motion_intent, "kind", "pose"))
    if motion_kind == "pose":
        return _solve_pose_target(
            arm_ik=arm_ik,
            motion_intent=motion_intent,
            current_lr_arm_q=current_lr_arm_q,
            current_lr_arm_dq=current_lr_arm_dq,
            workspace_limit_enabled=workspace_limit_enabled,
            workspace_mode=workspace_mode,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            tapered_workspace_params=tapered_workspace_params,
            side_workspaces=side_workspaces,
            is_online_inference=is_online_inference,
            timing_debugger=timing_debugger,
            log=log,
        )
    if motion_kind == "joint_position":
        return _require_finite_vector(motion_intent.arm_q, 14, "motion_intent.arm_q"), current_hold_tauff.copy(), 0.0, None
    if motion_kind == "joint_velocity":
        arm_dq = _require_finite_vector(motion_intent.arm_dq, 14, "motion_intent.arm_dq")
        return current_lr_arm_q + arm_dq * float(control_dt), current_hold_tauff.copy(), 0.0, None

    log.error("Unsupported motion intent kind: %s", motion_kind)
    raise UnsupportedMotionIntentError(f"Unsupported motion intent kind: {motion_kind}")


def _apply_arm_enable_gating(
    *,
    sol_q: np.ndarray,
    sol_tauff: np.ndarray,
    current_hold_q: np.ndarray,
    current_hold_tauff: np.ndarray,
    left_arm_enabled: bool,
    right_arm_enabled: bool,
    left_zero_takeover_this_frame: bool,
    right_zero_takeover_this_frame: bool,
    home_return_active: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if ((not left_arm_enabled) or left_zero_takeover_this_frame) and (not home_return_active):
        sol_q[:7] = current_hold_q[:7]
        sol_tauff[:7] = current_hold_tauff[:7]
    if ((not right_arm_enabled) or right_zero_takeover_this_frame) and (not home_return_active):
        sol_q[-7:] = current_hold_q[-7:]
        sol_tauff[-7:] = current_hold_tauff[-7:]
    return sol_q, sol_tauff


def _count_down_takeover_settle_frames(
    *,
    left_zero_takeover_this_frame: bool,
    right_zero_takeover_this_frame: bool,
    left_takeover_rising_edge: bool,
    right_takeover_rising_edge: bool,
    left_takeover_settle_frames: int,
    right_takeover_settle_frames: int,
    takeover_settle_frames: int,
    log: Any,
) -> tuple[int, int]:
    if left_zero_takeover_this_frame:
        if left_takeover_rising_edge:
            log.info(
                f"[TAKEOVER][LEFT] grip rising edge -> zero-delta hold for "
                f"{takeover_settle_frames} frames."
            )
        left_takeover_settle_frames -= 1
    if right_zero_takeover_this_frame:
        if right_takeover_rising_edge:
            log.info(
                f"[TAKEOVER][RIGHT] grip rising edge -> zero-delta hold for "
                f"{takeover_settle_frames} frames."
            )
        right_takeover_settle_frames -= 1
    return left_takeover_settle_frames, right_takeover_settle_frames


def _limit_speed(
    *,
    sol_q: np.ndarray,
    current_lr_arm_q: np.ndarray,
    home_return_active: bool,
    home_return_speed: float,
    max_arm_joint_speed: float,
    frequency: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    safety_start = time.perf_counter()
    sol_q_before_speed_limit = sol_q.copy()
    limited_q = limit_arm_joint_target_velocity(
        sol_q,
        current_lr_arm_q,
        max_joint_speed=(float(home_return_speed) if home_return_active else float(max_arm_joint_speed)),
        control_frequency=float(frequency),
        synchronize_per_arm=not home_return_active,
    )
    safety_ms = (time.perf_counter() - safety_start) * 1000.0
    return limited_q, sol_q_before_speed_limit, safety_ms


def _online_speed_limit_feedback(
    *,
    sol_q_before_speed_limit: np.ndarray,
    sol_q: np.ndarray,
    max_arm_joint_speed: float,
    frequency: float,
) -> Optional[dict]:
    speed_limit_delta = online_inference_speed_limit_delta(
        target_q=sol_q_before_speed_limit,
        limited_q=sol_q,
        max_arm_joint_speed=float(max_arm_joint_speed),
        frequency=float(frequency),
    )
    if speed_limit_delta is None:
        return None
    return _fatal_feedback(
        "online inference action exceeded joint speed limit",
        speed_limit_delta=speed_limit_delta,
    )


def build_arm_command(
    *,
    arm_ik: Any,
    motion_intent: Any,
    current_lr_arm_q: Any,
    current_lr_arm_dq: Any,
    current_hold_q: Any,
    current_hold_tauff: Any,
    left_arm_enabled: bool,
    right_arm_enabled: bool,
    home_return_active: bool,
    home_target_q: Any,
    control_dt: float,
    input_provider: str,
    frequency: float,
    max_arm_joint_speed: float,
    home_return_speed: float,
    workspace_limit_enabled: bool,
    workspace_mode: str,
    workspace_min: Any,
    workspace_max: Any,
    tapered_workspace_params: Mapping[str, float],
    compute_arm_gravity_tauff: Callable[[Any, np.ndarray], np.ndarray],
    side_workspaces: Optional[Mapping[str, Mapping[str, Any]]] = None,
    timing_debugger: Any = None,
    any_zero_takeover_this_frame: bool = False,
    left_zero_takeover_this_frame: bool = False,
    right_zero_takeover_this_frame: bool = False,
    left_takeover_rising_edge: bool = False,
    right_takeover_rising_edge: bool = False,
    left_takeover_settle_frames: int = 0,
    right_takeover_settle_frames: int = 0,
    takeover_settle_frames: int = 0,
    post_home_takeover_armed: bool = False,
    normalized_head_mode: str = "",
    sync_reference_to_current_live_pose: Optional[Callable[..., Any]] = None,
    log: Optional[Any] = None,
) -> ArmCommandResult:
    """Build one arm command using the real teleop loop's current semantics.

    ``provider_feedback`` is returned to the caller. This function deliberately
    does not call ``tv_wrapper.report_control_feedback``.
    """

    arm_cmd_input_start = time.perf_counter()
    log = logger if log is None else log
    input_provider = str(input_provider)
    is_online_inference = input_provider == "online_inference"

    current_lr_arm_q = np.asarray(current_lr_arm_q, dtype=float)
    current_lr_arm_dq = np.asarray(current_lr_arm_dq, dtype=float)
    current_hold_q = np.asarray(current_hold_q, dtype=float)
    current_hold_tauff = np.asarray(current_hold_tauff, dtype=float)
    workspace_min = np.asarray(workspace_min, dtype=float)
    workspace_max = np.asarray(workspace_max, dtype=float)
    arm_cmd_input_ms = (time.perf_counter() - arm_cmd_input_start) * 1000.0

    ik_ms = 0.0
    provider_feedback = None

    arm_cmd_takeover_reset_start = time.perf_counter()
    post_home_takeover_armed = _handle_takeover_reset(
        arm_ik=arm_ik,
        current_lr_arm_q=current_lr_arm_q,
        any_zero_takeover_this_frame=any_zero_takeover_this_frame,
        post_home_takeover_armed=post_home_takeover_armed,
        normalized_head_mode=normalized_head_mode,
        sync_reference_to_current_live_pose=sync_reference_to_current_live_pose,
    )
    arm_cmd_takeover_reset_ms = (time.perf_counter() - arm_cmd_takeover_reset_start) * 1000.0

    arm_cmd_target_start = time.perf_counter()
    if home_return_active:
        sol_q = np.asarray(home_target_q, dtype=float).copy()
        sol_tauff = current_hold_tauff.copy()
    elif left_arm_enabled or right_arm_enabled:
        try:
            sol_q, sol_tauff, ik_ms, provider_feedback = _solve_motion_target(
                arm_ik=arm_ik,
                motion_intent=motion_intent,
                current_lr_arm_q=current_lr_arm_q,
                current_lr_arm_dq=current_lr_arm_dq,
                current_hold_tauff=current_hold_tauff,
                control_dt=control_dt,
                workspace_limit_enabled=workspace_limit_enabled,
                workspace_mode=workspace_mode,
                workspace_min=workspace_min,
                workspace_max=workspace_max,
                tapered_workspace_params=tapered_workspace_params,
                side_workspaces=side_workspaces,
                is_online_inference=is_online_inference,
                timing_debugger=timing_debugger,
                log=log,
            )
        except _NonFiniteVectorError as exc:
            if not is_online_inference:
                raise
            provider_feedback = _fatal_feedback(
                "online inference produced non-finite IK/control target",
                error=str(exc),
            )
            sol_q = current_hold_q.copy()
            sol_tauff = current_hold_tauff.copy()

        sol_q = np.asarray(sol_q, dtype=float)
        sol_tauff = np.asarray(sol_tauff, dtype=float)
        if is_online_inference and (not np.all(np.isfinite(sol_q)) or not np.all(np.isfinite(sol_tauff))):
            provider_feedback = _fatal_feedback("online inference produced non-finite IK/control target")
    else:
        sol_q = current_hold_q.copy()
        sol_tauff = current_hold_tauff.copy()
    arm_cmd_target_ms = (time.perf_counter() - arm_cmd_target_start) * 1000.0
    arm_cmd_target_extra_ms = max(0.0, arm_cmd_target_ms - float(ik_ms))

    arm_cmd_feedback_gate_start = time.perf_counter()
    if is_online_inference and provider_feedback is not None:
        sol_q = current_hold_q.copy()
        sol_tauff = current_hold_tauff.copy()
        left_arm_enabled = False
        right_arm_enabled = False
    arm_cmd_feedback_gate_ms = (time.perf_counter() - arm_cmd_feedback_gate_start) * 1000.0

    arm_cmd_enable_gating_start = time.perf_counter()
    sol_q, sol_tauff = _apply_arm_enable_gating(
        sol_q=sol_q,
        sol_tauff=sol_tauff,
        current_hold_q=current_hold_q,
        current_hold_tauff=current_hold_tauff,
        left_arm_enabled=left_arm_enabled,
        right_arm_enabled=right_arm_enabled,
        left_zero_takeover_this_frame=left_zero_takeover_this_frame,
        right_zero_takeover_this_frame=right_zero_takeover_this_frame,
        home_return_active=home_return_active,
    )
    arm_cmd_enable_gating_ms = (time.perf_counter() - arm_cmd_enable_gating_start) * 1000.0

    arm_cmd_takeover_settle_start = time.perf_counter()
    left_takeover_settle_frames, right_takeover_settle_frames = _count_down_takeover_settle_frames(
        left_zero_takeover_this_frame=left_zero_takeover_this_frame,
        right_zero_takeover_this_frame=right_zero_takeover_this_frame,
        left_takeover_rising_edge=left_takeover_rising_edge,
        right_takeover_rising_edge=right_takeover_rising_edge,
        left_takeover_settle_frames=left_takeover_settle_frames,
        right_takeover_settle_frames=right_takeover_settle_frames,
        takeover_settle_frames=takeover_settle_frames,
        log=log,
    )
    arm_cmd_takeover_settle_ms = (time.perf_counter() - arm_cmd_takeover_settle_start) * 1000.0

    sol_q, sol_q_before_speed_limit, safety_ms = _limit_speed(
        sol_q=sol_q,
        current_lr_arm_q=current_lr_arm_q,
        home_return_active=home_return_active,
        home_return_speed=home_return_speed,
        max_arm_joint_speed=max_arm_joint_speed,
        frequency=frequency,
    )

    arm_cmd_speed_feedback_start = time.perf_counter()
    if is_online_inference and provider_feedback is None:
        provider_feedback = _online_speed_limit_feedback(
            sol_q_before_speed_limit=sol_q_before_speed_limit,
            sol_q=sol_q,
            max_arm_joint_speed=float(max_arm_joint_speed),
            frequency=float(frequency),
        )
        if provider_feedback is not None:
            sol_q = current_hold_q.copy()
    arm_cmd_speed_feedback_ms = (time.perf_counter() - arm_cmd_speed_feedback_start) * 1000.0

    gravity_start = time.perf_counter()
    sol_tauff = compute_arm_gravity_tauff(arm_ik, sol_q)
    gravity_ms = (time.perf_counter() - gravity_start) * 1000.0

    arm_cmd_hold_update_start = time.perf_counter()
    current_hold_q = np.asarray(sol_q, dtype=float).copy()
    current_hold_tauff = np.asarray(sol_tauff, dtype=float).copy()
    arm_cmd_hold_update_ms = (time.perf_counter() - arm_cmd_hold_update_start) * 1000.0

    return ArmCommandResult(
        sol_q=current_hold_q.copy(),
        sol_tauff=current_hold_tauff.copy(),
        current_hold_q=current_hold_q,
        current_hold_tauff=current_hold_tauff,
        provider_feedback=provider_feedback,
        ik_ms=ik_ms,
        safety_ms=safety_ms,
        gravity_ms=gravity_ms,
        arm_cmd_input_ms=arm_cmd_input_ms,
        arm_cmd_takeover_reset_ms=arm_cmd_takeover_reset_ms,
        arm_cmd_target_extra_ms=arm_cmd_target_extra_ms,
        arm_cmd_feedback_gate_ms=arm_cmd_feedback_gate_ms,
        arm_cmd_enable_gating_ms=arm_cmd_enable_gating_ms,
        arm_cmd_takeover_settle_ms=arm_cmd_takeover_settle_ms,
        arm_cmd_speed_feedback_ms=arm_cmd_speed_feedback_ms,
        arm_cmd_hold_update_ms=arm_cmd_hold_update_ms,
        sol_q_before_speed_limit=sol_q_before_speed_limit,
        left_arm_enabled=bool(left_arm_enabled),
        right_arm_enabled=bool(right_arm_enabled),
        post_home_takeover_armed=bool(post_home_takeover_armed),
        left_takeover_settle_frames=int(left_takeover_settle_frames),
        right_takeover_settle_frames=int(right_takeover_settle_frames),
    )
