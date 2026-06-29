from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorKeyCommand:
    set_record_toggle: bool | None = None
    set_record_cancel: bool | None = None
    log_level: str | None = None
    log_message: str | None = None


def resolve_operator_controller_buttons(
    *,
    left_primary_pressed: bool,
    prev_left_primary_pressed: bool,
    right_secondary_pressed: bool,
    prev_right_secondary_pressed: bool,
    started: bool,
    record_enabled: bool,
) -> OperatorKeyCommand:
    left_primary_rising = bool(left_primary_pressed) and (not bool(prev_left_primary_pressed))
    right_secondary_rising = bool(right_secondary_pressed) and (not bool(prev_right_secondary_pressed))

    if left_primary_rising and right_secondary_rising:
        return OperatorKeyCommand(
            log_level="warning",
            log_message=(
                "[CONTROLLER] simultaneous left X and right B presses ignored: "
                "record toggle and record cancel cannot be triggered in the same frame."
            ),
        )

    if left_primary_rising:
        if not started:
            return OperatorKeyCommand(
                log_level="warning",
                log_message="[RECORD_TOGGLE] ignored controller shortcut [left X]: teleop has not started yet.",
            )
        if not record_enabled:
            return OperatorKeyCommand(
                log_level="warning",
                log_message="[RECORD_TOGGLE] ignored controller shortcut [left X]: recording is disabled (--record not set).",
            )
        return OperatorKeyCommand(set_record_toggle=True)

    if right_secondary_rising:
        if not started:
            return OperatorKeyCommand(
                log_level="warning",
                log_message="[RECORD_CANCEL] ignored controller shortcut [right B]: teleop has not started yet.",
            )
        if not record_enabled:
            return OperatorKeyCommand(
                log_level="warning",
                log_message="[RECORD_CANCEL] ignored controller shortcut [right B]: recording is disabled (--record not set).",
            )
        return OperatorKeyCommand(set_record_cancel=True)

    return OperatorKeyCommand()
