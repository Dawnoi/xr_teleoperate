"""操作员状态流：集中处理 home、deadman 和接管边沿状态。

Operator/home/deadman state for the real teleop control loop.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class OperatorStateResult:
    tele_data: Any
    left_arm_enabled: bool
    right_arm_enabled: bool
    home_return_active: bool
    home_wait_grip_release: bool
    post_home_takeover_armed: bool
    left_takeover_settle_frames: int
    right_takeover_settle_frames: int
    left_takeover_rising_edge: bool
    right_takeover_rising_edge: bool
    left_zero_takeover_this_frame: bool
    right_zero_takeover_this_frame: bool
    any_zero_takeover_this_frame: bool
    takeover_logic_ms: float


class OperatorStateFlow:
    def __init__(self, *, takeover_settle_frames: int, log):
        self.takeover_settle_frames = int(takeover_settle_frames)
        self.log = log
        self.prev_home_button_pressed = False
        self.home_wait_grip_release = False
        self.prev_left_arm_enabled = None
        self.prev_right_arm_enabled = None
        self.prev_left_grip_pressed = False
        self.prev_right_grip_pressed = False
        self.left_takeover_settle_frames = 0
        self.right_takeover_settle_frames = 0

    def apply(
        self,
        *,
        args,
        operator_runtime,
        tele_data,
        started: bool,
        on_press: Callable[[str], None],
        motion_intent,
        home_return_active: bool,
        normalized_head_mode: str,
        tv_wrapper,
        arm_ik,
        current_hold_q,
        reset_arm_ik_state: Callable[[Any, Any], None],
        timer,
        ) -> OperatorStateResult:
        start_time = timer()
        tele_data = operator_runtime.apply_to_tele_data(
            tele_data,
            started=bool(started),
            on_press=on_press,
        )

        home_return_active = self._apply_home_button(tele_data, home_return_active)
        left_arm_enabled, right_arm_enabled = self._resolve_enabled_arms(args, tele_data, motion_intent)
        left_arm_enabled, right_arm_enabled, post_home_takeover_armed = self._apply_home_release_gate(
            left_arm_enabled=left_arm_enabled,
            right_arm_enabled=right_arm_enabled,
            tele_data=tele_data,
            normalized_head_mode=normalized_head_mode,
            tv_wrapper=tv_wrapper,
            arm_ik=arm_ik,
            current_hold_q=current_hold_q,
            reset_arm_ik_state=reset_arm_ik_state,
        )
        self._log_deadman_change(args, left_arm_enabled, right_arm_enabled)
        (
            left_takeover_rising_edge,
            right_takeover_rising_edge,
            left_zero_takeover_this_frame,
            right_zero_takeover_this_frame,
            any_zero_takeover_this_frame,
        ) = self._update_takeover_edges(tele_data)

        return OperatorStateResult(
            tele_data=tele_data,
            left_arm_enabled=left_arm_enabled,
            right_arm_enabled=right_arm_enabled,
            home_return_active=home_return_active,
            home_wait_grip_release=self.home_wait_grip_release,
            post_home_takeover_armed=post_home_takeover_armed,
            left_takeover_settle_frames=self.left_takeover_settle_frames,
            right_takeover_settle_frames=self.right_takeover_settle_frames,
            left_takeover_rising_edge=left_takeover_rising_edge,
            right_takeover_rising_edge=right_takeover_rising_edge,
            left_zero_takeover_this_frame=left_zero_takeover_this_frame,
            right_zero_takeover_this_frame=right_zero_takeover_this_frame,
            any_zero_takeover_this_frame=any_zero_takeover_this_frame,
            takeover_logic_ms=(timer() - start_time) * 1000.0,
        )

    def sync_from_arm_command(self, *, left_takeover_settle_frames: int, right_takeover_settle_frames: int):
        self.left_takeover_settle_frames = int(left_takeover_settle_frames)
        self.right_takeover_settle_frames = int(right_takeover_settle_frames)

    def _apply_home_button(self, tele_data, home_return_active: bool) -> bool:
        home_button_pressed = bool(tele_data.left_ctrl_bButton)
        if home_button_pressed and not self.prev_home_button_pressed:
            self.log.info("[HOME] left Y pressed -> returning both arms to ready/calibration pose with speed limit.")
            home_return_active = True
            self.home_wait_grip_release = True
        self.prev_home_button_pressed = home_button_pressed
        return home_return_active

    def _resolve_enabled_arms(self, args, tele_data, motion_intent) -> tuple[bool, bool]:
        if args.input_mode == "controller" and args.controller_deadman == "grip":
            left_arm_enabled = bool(tele_data.left_ctrl_squeeze)
            right_arm_enabled = bool(tele_data.right_ctrl_squeeze)
        else:
            left_arm_enabled = True
            right_arm_enabled = True

        provider_enabled_arms = None
        if motion_intent is not None:
            provider_enabled_arms = motion_intent.metadata.get("enabled_arms")
        if provider_enabled_arms is not None:
            provider_enabled_set = {str(side) for side in provider_enabled_arms}
            left_arm_enabled = left_arm_enabled and ("left" in provider_enabled_set)
            right_arm_enabled = right_arm_enabled and ("right" in provider_enabled_set)
        return left_arm_enabled, right_arm_enabled

    def _apply_home_release_gate(
        self,
        *,
        left_arm_enabled: bool,
        right_arm_enabled: bool,
        tele_data,
        normalized_head_mode: str,
        tv_wrapper,
        arm_ik,
        current_hold_q,
        reset_arm_ik_state: Callable[[Any, Any], None],
    ) -> tuple[bool, bool, bool]:
        if not self.home_wait_grip_release:
            return left_arm_enabled, right_arm_enabled, False
        if bool(tele_data.left_ctrl_squeeze) or bool(tele_data.right_ctrl_squeeze):
            return False, False, False

        self.home_wait_grip_release = False
        if normalized_head_mode in {"head_coupled", "hybrid"}:
            tv_wrapper.sync_reference_to_current_live_pose(require_live=False)
            self.log.info("[HOME] reference synced to current live pose after home return.")
        reset_arm_ik_state(arm_ik, current_hold_q)
        self.log.info("[HOME] IK state reset at current home pose.")
        self.log.info("[HOME] grip released -> teleop re-enabled.")
        return left_arm_enabled, right_arm_enabled, True

    def _log_deadman_change(self, args, left_arm_enabled: bool, right_arm_enabled: bool) -> None:
        if left_arm_enabled == self.prev_left_arm_enabled and right_arm_enabled == self.prev_right_arm_enabled:
            return
        self.log.info(
            f"[DEADMAN] left_enabled={left_arm_enabled} right_enabled={right_arm_enabled} "
            f"(controller_deadman={args.controller_deadman})"
        )
        self.prev_left_arm_enabled = left_arm_enabled
        self.prev_right_arm_enabled = right_arm_enabled

    def _update_takeover_edges(self, tele_data) -> tuple[bool, bool, bool, bool, bool]:
        left_grip_pressed = bool(tele_data.left_ctrl_squeeze)
        right_grip_pressed = bool(tele_data.right_ctrl_squeeze)
        left_takeover_rising_edge = left_grip_pressed and (not self.prev_left_grip_pressed)
        right_takeover_rising_edge = right_grip_pressed and (not self.prev_right_grip_pressed)
        if left_takeover_rising_edge:
            self.left_takeover_settle_frames = self.takeover_settle_frames
        if right_takeover_rising_edge:
            self.right_takeover_settle_frames = self.takeover_settle_frames

        left_zero_takeover_this_frame = self.left_takeover_settle_frames > 0
        right_zero_takeover_this_frame = self.right_takeover_settle_frames > 0
        self.prev_left_grip_pressed = left_grip_pressed
        self.prev_right_grip_pressed = right_grip_pressed
        return (
            left_takeover_rising_edge,
            right_takeover_rising_edge,
            left_zero_takeover_this_frame,
            right_zero_takeover_this_frame,
            left_zero_takeover_this_frame or right_zero_takeover_this_frame,
        )
