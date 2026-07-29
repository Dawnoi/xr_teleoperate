"""Base command handling extracted from the real teleop main loop.

This module intentionally keeps the current control semantics flat: compute the
same base command values, send them through the existing controller object, and
return the flags/metrics that the main loop needs to apply global state changes.
"""

from dataclasses import dataclass
from typing import Any, Optional
import math
import time

from core.input.base import BaseCommandIntent


WAIST_YAW_MINIMUM_RAD = -2.7053
WAIST_YAW_MAXIMUM_RAD = 2.7053

@dataclass
class BaseCommandResult:
    base_control_mode: str = "none"
    base_control_ms: float = 0.0
    base_move_ms: float = 0.0
    base_height_ms: float = 0.0
    base_misc_ms: float = 0.0
    base_async_cycle_avg_ms: Optional[float] = None
    base_async_move_avg_ms: Optional[float] = None
    base_async_height_avg_ms: Optional[float] = None
    base_async_queue_avg_ms: Optional[float] = None
    base_async_publish_hz: Optional[float] = None
    base_vx: float = 0.0
    base_vy: float = 0.0
    base_wz: float = 0.0
    base_z: float = 0.0
    stop_requested: bool = False
    should_continue_frame: bool = False
    base_stop_latched: bool = False
    base_stop_fault: bool = False
    base_stop_error: str = ""


@dataclass(frozen=True)
class BaseStopResult:
    confirmed: bool
    controller: str
    ack: str = ""
    error: str = ""


def _apply_deadzone(value: float, deadzone: float) -> float:
    return 0.0 if abs(value) < deadzone else value


def stop_base_command(*, args: Any, loco_wrapper: Any = None, agv_bridge: Any = None) -> BaseStopResult:
    if not args.base_motion:
        return BaseStopResult(confirmed=True, controller="disabled")
    controller = str(getattr(args, "base_controller", "none") or "none")
    if controller == "none":
        return BaseStopResult(confirmed=True, controller=controller)
    if controller == "loco":
        if loco_wrapper is None:
            raise RuntimeError("loco_wrapper is required for --base-controller loco")
        loco_wrapper.Move(0.0, 0.0, 0.0)
        return BaseStopResult(confirmed=True, controller=controller, ack="MOVE 0 0 0")
    if controller == "g1d_agv":
        if agv_bridge is None:
            raise RuntimeError("agv_bridge is required for --base-controller g1d_agv")
        ack = agv_bridge.stop_sync()
        confirmed = ack == "OK STOP 0 0"
        if confirmed:
            return BaseStopResult(confirmed=True, controller=controller, ack=ack)
        timing = agv_bridge.get_timing_snapshot()
        last_error = str(timing.get("last_error") or "")
        error = last_error or f"unexpected G1D AGV STOP acknowledgement: {ack!r}"
        return BaseStopResult(confirmed=False, controller=controller, ack=str(ack or ""), error=error)
    raise ValueError(f"unsupported base_controller: {controller}")


def resolve_runtime_base_command_source(
    *,
    active_input_provider: str,
    is_ui_raw_replay: bool,
    raw_replay_base_source: str,
) -> str | None:
    """Select provider-owned base input only while its UI provider is active."""
    if str(active_input_provider) == "online_inference":
        return "provider"
    if bool(is_ui_raw_replay) and str(raw_replay_base_source) == "action":
        return "provider"
    return None


def _validate_provider_base_intent(
    *,
    args: Any,
    intent: BaseCommandIntent,
    loco_wrapper: Any,
    agv_bridge: Any,
) -> None:
    limits = (
        ("vx", "base_max_vx"),
        ("vy", "base_max_vy"),
        ("wz", "base_max_wz"),
        ("z", "base_max_z"),
    )
    for axis, limit_name in limits:
        limit = float(getattr(args, limit_name))
        if not math.isfinite(limit) or limit <= 0.0:
            stop_result = stop_base_command(args=args, loco_wrapper=loco_wrapper, agv_bridge=agv_bridge)
            if not stop_result.confirmed:
                raise RuntimeError(f"base STOP was not confirmed: {stop_result.error}")
            raise ValueError(f"{limit_name} must be positive and finite, got {limit!r}")
        value = float(getattr(intent, axis))
        if abs(value) > limit:
            stop_result = stop_base_command(args=args, loco_wrapper=loco_wrapper, agv_bridge=agv_bridge)
            if not stop_result.confirmed:
                raise RuntimeError(f"base STOP was not confirmed: {stop_result.error}")
            raise ValueError(
                f"provider base command {axis}={value:.6g} exceeds {limit_name} "
                f"(--base-max-{axis})={limit:.6g}"
            )


def _controller_base_intent(args: Any, tele_data: Any, *, home_return_active: bool) -> BaseCommandIntent:
    if home_return_active:
        return BaseCommandIntent(source="controller")

    left_stick_x = _apply_deadzone(float(tele_data.left_ctrl_thumbstickValue[0]), args.base_stick_deadzone)
    left_stick_y = _apply_deadzone(float(tele_data.left_ctrl_thumbstickValue[1]), args.base_stick_deadzone)
    right_stick_x = _apply_deadzone(float(tele_data.right_ctrl_thumbstickValue[0]), args.base_stick_deadzone)
    right_stick_y = _apply_deadzone(float(tele_data.right_ctrl_thumbstickValue[1]), args.base_stick_deadzone)

    if args.base_controller == "loco":
        return BaseCommandIntent(
            vx=-left_stick_y * args.base_max_vx,
            vy=-left_stick_x * args.base_max_vy,
            wz=-right_stick_x * args.base_max_wz,
            z=0.0,
            source="controller:loco",
        )
    if args.base_controller == "g1d_agv":
        return BaseCommandIntent(
            vx=left_stick_y * args.base_max_vx,
            vy=0.0,
            wz=-left_stick_x * args.base_max_wz,
            z=right_stick_y * args.base_max_z,
            source="controller:g1d_agv",
        )
    return BaseCommandIntent(source="controller:none")


def _base_intent_for_source(
    *,
    args: Any,
    source: str,
    tele_data: Any,
    home_return_active: bool,
    base_intent: BaseCommandIntent | None,
) -> BaseCommandIntent | None:
    if str(getattr(args, "base_controller", "none")) == "none" or source == "none":
        return None
    if source == "controller":
        if str(getattr(args, "input_mode", "hand")) != "controller":
            return None
        return _controller_base_intent(args, tele_data, home_return_active=home_return_active)
    if source == "provider":
        if home_return_active:
            return BaseCommandIntent(source="provider:home_return")
        if base_intent is None:
            raise RuntimeError("base_command_source=provider requires sample.base_intent")
        return base_intent
    raise ValueError(f"unsupported base_command_source: {source}")


def map_base_command(
    *,
    args: Any,
    tele_data: Any,
    home_return_active: bool,
    base_intent: BaseCommandIntent | None,
    base_command_source: str,
) -> BaseCommandIntent:
    """Map one input frame to a nominal body command without sending hardware I/O."""
    intent = _base_intent_for_source(
        args=args,
        source=base_command_source,
        tele_data=tele_data,
        home_return_active=home_return_active,
        base_intent=base_intent,
    )
    return BaseCommandIntent(source="none") if intent is None else intent


def map_manual_torso_yaw_rate(*, args: Any, tele_data: Any, home_return_active: bool) -> float:
    """Map the right controller X axis to a bounded manual torso yaw rate.

    This is an independent waist-motor command. It does not participate in
    arm IK and does not share an axis with the G1D base yaw, which uses the
    left controller X axis.
    """
    if home_return_active:
        return 0.0
    if str(getattr(args, "input_mode", "hand")) != "controller":
        return 0.0
    right_stick_x = _apply_deadzone(
        float(tele_data.right_ctrl_thumbstickValue[0]),
        float(args.base_stick_deadzone),
    )
    return -right_stick_x * float(args.mobile_max_torso_yaw_rate)


def integrate_manual_torso_yaw_target(
    *,
    current_target_rad: float,
    yaw_rate_radps: float,
    dt: float,
) -> tuple[float, bool]:
    """Integrate the independent waist command and enforce hardware limits."""
    values = {
        "current_target_rad": current_target_rad,
        "yaw_rate_radps": yaw_rate_radps,
        "dt": dt,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")
    raw_target = float(current_target_rad) + float(yaw_rate_radps) * float(dt)
    target = min(max(raw_target, WAIST_YAW_MINIMUM_RAD), WAIST_YAW_MAXIMUM_RAD)
    return target, not math.isclose(target, raw_target, abs_tol=1e-12)


def apply_base_command(
    *,
    args: Any,
    tele_data: Any,
    home_return_active: bool,
    loco_wrapper: Any = None,
    agv_bridge: Any = None,
    timing_debugger: Any = None,
    base_intent: BaseCommandIntent | None = None,
    base_provider_active: bool = True,
    base_stop_latched: bool = False,
    base_stop_fault: bool = False,
    base_command_source: str | None = None,
) -> BaseCommandResult:
    """Apply one frame of base control and return main-loop state/metrics.

    The caller owns global state updates. In particular, right-controller A is
    reported as ``stop_requested`` instead of directly mutating START/STOP, and
    the double-thumbstick Damp branch is reported as ``should_continue_frame``.
    """

    base_control_start = time.perf_counter()
    result = BaseCommandResult()
    controller = str(getattr(args, "base_controller", "none") or "none")
    source = str(base_command_source or getattr(args, "base_command_source", "controller") or "controller")

    if tele_data.right_ctrl_aButton:
        result.stop_requested = True

    if not args.base_motion:
        result.base_control_ms = (time.perf_counter() - base_control_start) * 1000.0
        result.base_misc_ms = result.base_control_ms
        return result

    if base_stop_fault:
        stop_result = stop_base_command(args=args, loco_wrapper=loco_wrapper, agv_bridge=agv_bridge)
        result.base_stop_latched = stop_result.confirmed
        result.base_stop_fault = not stop_result.confirmed
        result.base_stop_error = stop_result.error
        result.base_control_mode = "base_stop_fault" if result.base_stop_fault else "base_stop_recovered"
        result.should_continue_frame = True
        result.base_control_ms = (time.perf_counter() - base_control_start) * 1000.0
        result.base_misc_ms = result.base_control_ms
        return result

    if source == "provider" and not bool(base_provider_active):
        if not base_stop_latched:
            stop_result = stop_base_command(args=args, loco_wrapper=loco_wrapper, agv_bridge=agv_bridge)
            result.base_stop_latched = stop_result.confirmed
            result.base_stop_fault = not stop_result.confirmed
            result.base_stop_error = stop_result.error
        else:
            result.base_stop_latched = True
        result.base_control_mode = "provider_inactive"
        result.base_control_ms = (time.perf_counter() - base_control_start) * 1000.0
        result.base_misc_ms = result.base_control_ms
        return result

    if (
        source == "provider"
        and controller != "none"
        and not home_return_active
        and base_intent is None
    ):
        stop_result = stop_base_command(args=args, loco_wrapper=loco_wrapper, agv_bridge=agv_bridge)
        if not stop_result.confirmed:
            raise RuntimeError(f"base STOP was not confirmed: {stop_result.error}")
        raise RuntimeError("base_command_source=provider requires sample.base_intent")

    if (
        controller == "loco"
        and source == "controller"
        and str(getattr(args, "input_mode", "hand")) == "controller"
    ):
        if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
            if loco_wrapper is None:
                raise RuntimeError("loco_wrapper is required for --base-controller loco")
            loco_wrapper.Damp()
            time.sleep(0.05)
            result.base_control_mode = "loco"
            result.should_continue_frame = True
            result.base_control_ms = (time.perf_counter() - base_control_start) * 1000.0
            result.base_misc_ms = max(
                0.0,
                result.base_control_ms - result.base_move_ms - result.base_height_ms,
            )
            return result

    intent = _base_intent_for_source(
        args=args,
        source=source,
        tele_data=tele_data,
        home_return_active=home_return_active,
        base_intent=base_intent,
    )
    if source == "provider" and intent is not None:
        _validate_provider_base_intent(
            args=args,
            intent=intent,
            loco_wrapper=loco_wrapper,
            agv_bridge=agv_bridge,
        )
    if intent is not None:
        result.base_vx = float(intent.vx)
        result.base_vy = float(intent.vy)
        result.base_wz = float(intent.wz)
        result.base_z = float(intent.z)

    if controller == "none" or intent is None:
        result.base_control_mode = "none"
    elif controller == "loco":
        result.base_control_mode = "loco"
        if loco_wrapper is None:
            raise RuntimeError("loco_wrapper is required for --base-controller loco")
        base_move_start = time.perf_counter()
        loco_wrapper.Move(result.base_vx, result.base_vy, result.base_wz)
        result.base_move_ms = (time.perf_counter() - base_move_start) * 1000.0
    elif controller == "g1d_agv":
        result.base_control_mode = "g1d_agv_async"
        if agv_bridge is None:
            raise RuntimeError("agv_bridge is required for --base-controller g1d_agv")
        agv_timing_snapshot = agv_bridge.get_timing_snapshot()
        if not bool(agv_timing_snapshot.get("healthy", True)):
            stop_result = stop_base_command(args=args, loco_wrapper=loco_wrapper, agv_bridge=agv_bridge)
            result.base_stop_latched = stop_result.confirmed
            result.base_stop_fault = True
            result.base_stop_error = str(agv_timing_snapshot.get("fault_reason") or stop_result.error)
            result.base_control_mode = "base_stop_fault"
            result.should_continue_frame = True
            result.base_control_ms = (time.perf_counter() - base_control_start) * 1000.0
            result.base_misc_ms = result.base_control_ms
            return result
        agv_send_start = time.perf_counter()
        base_move_start = time.perf_counter()
        agv_bridge.set_target(result.base_vx, result.base_vy, result.base_wz, result.base_z)
        result.base_move_ms = (time.perf_counter() - base_move_start) * 1000.0
        if timing_debugger is not None:
            timing_debugger.add_agv(time.perf_counter() - agv_send_start)
        agv_timing_snapshot = agv_bridge.get_timing_snapshot()
        last_error = str(agv_timing_snapshot.get("last_error") or "")
        if last_error:
            stop_result = stop_base_command(args=args, loco_wrapper=loco_wrapper, agv_bridge=agv_bridge)
            result.base_stop_latched = stop_result.confirmed
            result.base_stop_fault = True
            result.base_stop_error = last_error
            result.base_control_mode = "base_stop_fault"
            result.should_continue_frame = True
            result.base_control_ms = (time.perf_counter() - base_control_start) * 1000.0
            result.base_misc_ms = result.base_control_ms
            return result
        result.base_async_publish_hz = float(agv_timing_snapshot.get("publish_hz", 0.0))
        move_stats = agv_timing_snapshot.get("move_stats") or {}
        height_stats = agv_timing_snapshot.get("height_stats") or {}
        cycle_stats = agv_timing_snapshot.get("cycle_stats") or {}
        queue_stats = agv_timing_snapshot.get("queue_delay_stats") or {}
        result.base_async_move_avg_ms = move_stats.get("avg_ms")
        result.base_async_height_avg_ms = height_stats.get("avg_ms")
        result.base_async_cycle_avg_ms = cycle_stats.get("avg_ms")
        result.base_async_queue_avg_ms = queue_stats.get("avg_ms")
    else:
        raise ValueError(f"unsupported base_controller: {controller}")

    result.base_control_ms = (time.perf_counter() - base_control_start) * 1000.0
    result.base_misc_ms = max(
        0.0,
        result.base_control_ms - result.base_move_ms - result.base_height_ms,
    )
    return result
