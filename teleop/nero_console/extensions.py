"""XR-owned contracts for Nero UI extension actions.

The Nero Host transports ``extension.invoke`` generically.  This module owns
the XR allowlist and translates only accepted actions into existing high-level
control-loop intents.
"""

from __future__ import annotations

from typing import Any

from teleop.ui.command_bus import UiCommandName


XR_TELEOPERATION_EXTENSION_ID = "xr.teleoperation"
XR_HEAD_REFERENCE_RESET_ACTION = "head_reference.reset"
XR_AUTHORITY_HOLD_ACTION = "authority.hold"
XR_AUTHORITY_XR_ACTION = "authority.xr"
XR_HOME_ACTION = "home"
XR_RECORD_CANCEL_ACTION = "record.cancel"
XR_INFERENCE_START_ACTION = "inference.start"
XR_INFERENCE_STOP_ACTION = "inference.stop"

_XR_INFERENCE_PROTOCOL_PROFILES = {"mobile_tcp23", "mobile_pelvis_planar22", "mobile_joint_base"}
_MAX_INFERENCE_PROMPT_LENGTH = 2_048

_XR_TELEOPERATION_ACTIONS: dict[str, tuple[UiCommandName, dict[str, str]]] = {
    XR_HEAD_REFERENCE_RESET_ACTION: (UiCommandName.RECENTER, {}),
    XR_AUTHORITY_HOLD_ACTION: (
        UiCommandName.SET_PROVIDER_HOLD,
        {"reason": "extension:xr.teleoperation"},
    ),
    XR_AUTHORITY_XR_ACTION: (
        UiCommandName.SET_PROVIDER_XR,
        {"reason": "extension:xr.teleoperation"},
    ),
    XR_HOME_ACTION: (UiCommandName.HOME, {}),
    XR_RECORD_CANCEL_ACTION: (UiCommandName.RECORD_CANCEL, {}),
}


def map_xr_teleoperation_extension_action(
    params: dict[str, Any],
) -> tuple[UiCommandName, dict[str, Any]] | None:
    """Map the first XR extension action to an existing main-loop intent.

    Extension payloads intentionally have a closed shape.  The web extension
    cannot select an arbitrary command or pass undocumented options into the
    control loop.
    """
    if not isinstance(params, dict) or set(params) != {"extension_id", "action", "extra"}:
        return None
    if not isinstance(params["extension_id"], str) or params["extension_id"] != XR_TELEOPERATION_EXTENSION_ID:
        return None
    if not isinstance(params["action"], str):
        return None
    if not isinstance(params["extra"], dict):
        return None
    action = params["action"]
    extra = params["extra"]
    if action == XR_INFERENCE_START_ACTION:
        if set(extra) != {"prompt", "protocol_profile"}:
            return None
        prompt = extra.get("prompt")
        protocol_profile = extra.get("protocol_profile")
        if not isinstance(prompt, str) or not isinstance(protocol_profile, str):
            return None
        prompt = prompt.strip()
        protocol_profile = protocol_profile.strip()
        if not prompt or len(prompt) > _MAX_INFERENCE_PROMPT_LENGTH:
            return None
        if protocol_profile not in _XR_INFERENCE_PROTOCOL_PROFILES:
            return None
        return UiCommandName.START_ONLINE_INFERENCE, {
            "prompt": prompt,
            "protocol_profile": protocol_profile,
        }
    if action == XR_INFERENCE_STOP_ACTION:
        if extra:
            return None
        return UiCommandName.STOP_ONLINE_INFERENCE, {}
    if extra:
        return None
    mapped_action = _XR_TELEOPERATION_ACTIONS.get(action)
    if mapped_action is None:
        return None
    command, payload = mapped_action
    return command, dict(payload)
