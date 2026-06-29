import logging_mp

from teleop.operator.keybinds import resolve_operator_controller_buttons


logger_mp = logging_mp.getLogger(__name__)


class OperatorRuntime:
    def __init__(self, *, record_enabled: bool):
        self.record_enabled = bool(record_enabled)
        self._prev_left_primary_pressed = False
        self._prev_right_secondary_pressed = False
        self._home_requested = False

    def request_home(self, source: str) -> None:
        self._home_requested = True
        logger_mp.info("[HOME] %s requested home return; injecting controller home pulse.", source)

    def home_requested(self) -> bool:
        return bool(self._home_requested)

    def warn_record_disabled(self, shortcut: str) -> None:
        logger_mp.warning(
            "[RECORD] ignored %s: recording is disabled (--record not set).",
            shortcut,
        )

    def apply_to_tele_data(self, tele_data, *, started: bool, on_press):
        command = resolve_operator_controller_buttons(
            left_primary_pressed=bool(tele_data.left_ctrl_aButton),
            prev_left_primary_pressed=self._prev_left_primary_pressed,
            right_secondary_pressed=bool(tele_data.right_ctrl_bButton),
            prev_right_secondary_pressed=self._prev_right_secondary_pressed,
            started=bool(started),
            record_enabled=self.record_enabled,
        )
        self._prev_left_primary_pressed = bool(tele_data.left_ctrl_aButton)
        self._prev_right_secondary_pressed = bool(tele_data.right_ctrl_bButton)

        if command.log_message is not None:
            if command.log_level == "warning":
                logger_mp.warning(command.log_message)
            else:
                logger_mp.info(command.log_message)
        if command.set_record_toggle:
            on_press("s")
        if command.set_record_cancel:
            on_press("v")

        if self._home_requested:
            tele_data.left_ctrl_bButton = True
            self._home_requested = False

        return tele_data
