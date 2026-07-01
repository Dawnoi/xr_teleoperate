"""Base command handling extracted from the real teleop main loop.

This module intentionally keeps the current control semantics flat: compute the
same base command values, send them through the existing controller object, and
return the flags/metrics that the main loop needs to apply global state changes.
"""

from dataclasses import dataclass
from typing import Any, Optional
import time


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


def _apply_deadzone(value: float, deadzone: float) -> float:
    return 0.0 if abs(value) < deadzone else value


def apply_base_command(
    *,
    args: Any,
    tele_data: Any,
    home_return_active: bool,
    loco_wrapper: Any = None,
    agv_bridge: Any = None,
    timing_debugger: Any = None,
) -> BaseCommandResult:
    """Apply one frame of base control and return main-loop state/metrics.

    The caller owns global state updates. In particular, right-controller A is
    reported as ``stop_requested`` instead of directly mutating START/STOP, and
    the double-thumbstick Damp branch is reported as ``should_continue_frame``.
    """

    base_control_start = time.perf_counter()
    result = BaseCommandResult()

    if args.input_mode == "controller" and args.motion:
        result.base_control_mode = "loco"
        if tele_data.right_ctrl_aButton:
            result.stop_requested = True

        if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
            if loco_wrapper is None:
                raise RuntimeError("loco_wrapper is required for --motion base control")
            loco_wrapper.Damp()
            time.sleep(0.05)
            result.should_continue_frame = True
            result.base_control_ms = (time.perf_counter() - base_control_start) * 1000.0
            result.base_misc_ms = max(
                0.0,
                result.base_control_ms - result.base_move_ms - result.base_height_ms,
            )
            return result

        left_stick_x = float(tele_data.left_ctrl_thumbstickValue[0])
        left_stick_y = float(tele_data.left_ctrl_thumbstickValue[1])
        right_stick_x = float(tele_data.right_ctrl_thumbstickValue[0])

        if home_return_active:
            result.base_vx = 0.0
            result.base_vy = 0.0
            result.base_wz = 0.0
        else:
            left_stick_x = _apply_deadzone(left_stick_x, args.base_stick_deadzone)
            left_stick_y = _apply_deadzone(left_stick_y, args.base_stick_deadzone)
            right_stick_x = _apply_deadzone(right_stick_x, args.base_stick_deadzone)

            # Match Unitree loco semantics:
            #   left stick  -> body-frame x/y velocity
            #   right stick -> body-frame angular z velocity
            result.base_vx = -left_stick_y * args.base_max_vx
            result.base_vy = -left_stick_x * args.base_max_vy
            result.base_wz = -right_stick_x * args.base_max_wz

        if loco_wrapper is None:
            raise RuntimeError("loco_wrapper is required for --motion base control")
        base_move_start = time.perf_counter()
        loco_wrapper.Move(result.base_vx, result.base_vy, result.base_wz)
        result.base_move_ms = (time.perf_counter() - base_move_start) * 1000.0

    elif args.input_mode == "controller" and args.base_controller == "g1d_agv":
        result.base_control_mode = "g1d_agv_async"
        if tele_data.right_ctrl_aButton:
            result.stop_requested = True

        left_stick_x = float(tele_data.left_ctrl_thumbstickValue[0])
        left_stick_y = float(tele_data.left_ctrl_thumbstickValue[1])
        right_stick_x = float(tele_data.right_ctrl_thumbstickValue[0])
        right_stick_y = float(tele_data.right_ctrl_thumbstickValue[1])

        left_stick_x = _apply_deadzone(left_stick_x, args.base_stick_deadzone)
        left_stick_y = _apply_deadzone(left_stick_y, args.base_stick_deadzone)
        right_stick_x = _apply_deadzone(right_stick_x, args.base_stick_deadzone)
        right_stick_y = _apply_deadzone(right_stick_y, args.base_stick_deadzone)

        if home_return_active:
            result.base_vx = 0.0
            result.base_vy = 0.0
            result.base_wz = 0.0
            result.base_z = 0.0
        else:
            # Backported from the official unitree_sdk2 G1D example path:
            #   AgvClient.Move(vx, vy, vyaw)
            #   AgvClient.HeightAdjust(vz)
            # Note: the official G1D AGV header comments that vy is currently
            # ignored by the AGV side. For practical teleop on G1D, we therefore
            # map left stick X to yaw so it behaves like the official remote:
            #   left stick up/down -> forward/backward
            #   left stick left/right -> in-place turn
            #   right stick up/down -> column height adjust
            result.base_vx = left_stick_y * args.base_max_vx
            result.base_vy = 0.0
            result.base_wz = -left_stick_x * args.base_max_wz
            result.base_z = right_stick_y * args.base_max_z

        if agv_bridge is not None:
            agv_send_start = time.perf_counter()
            base_move_start = time.perf_counter()
            agv_bridge.set_target(result.base_vx, result.base_vy, result.base_wz, result.base_z)
            result.base_move_ms = (time.perf_counter() - base_move_start) * 1000.0
            if timing_debugger is not None:
                timing_debugger.add_agv(time.perf_counter() - agv_send_start)
            try:
                agv_timing_snapshot = agv_bridge.get_timing_snapshot()
            except Exception:
                agv_timing_snapshot = None
            if agv_timing_snapshot is not None:
                result.base_async_publish_hz = float(agv_timing_snapshot.get("publish_hz", 0.0))
                move_stats = agv_timing_snapshot.get("move_stats") or {}
                height_stats = agv_timing_snapshot.get("height_stats") or {}
                cycle_stats = agv_timing_snapshot.get("cycle_stats") or {}
                queue_stats = agv_timing_snapshot.get("queue_delay_stats") or {}
                result.base_async_move_avg_ms = move_stats.get("avg_ms")
                result.base_async_height_avg_ms = height_stats.get("avg_ms")
                result.base_async_cycle_avg_ms = cycle_stats.get("avg_ms")
                result.base_async_queue_avg_ms = queue_stats.get("avg_ms")

    result.base_control_ms = (time.perf_counter() - base_control_start) * 1000.0
    result.base_misc_ms = max(
        0.0,
        result.base_control_ms - result.base_move_ms - result.base_height_ms,
    )
    return result
