from __future__ import annotations

import time
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import cv2

from robot_ui_platform import (
    CommandOutcome,
    CommandStatus,
    UiCapabilities,
    UiCommandCapability,
    UiIntent,
    UiSnapshot,
)
from teleop.ui.command_bus import UiCommandBus, UiCommandName
from teleop.ui.state_store import UiStateStore as LegacyUiStateStore


@dataclass(frozen=True, slots=True)
class XrTeleoperateUiContext:
    """Existing XR runtime services exposed to the UI adapter.

    The adapter owns no robot-control object. It only translates a validated
    high-level UI intent into the command bus already consumed by the control
    loop on its robot thread.
    """

    command_bus: UiCommandBus
    state_store: LegacyUiStateStore
    camera_status_getter: Callable[[], dict[str, Any]]
    camera_frame_getter: Callable[[int], tuple[Any, dict[str, Any] | None]]
    recenter_enabled: bool


class XrTeleoperateBackend:
    backend_id = "xr_teleoperate"
    backend_name = "XR Teleoperate"

    _CAMERA_ID_BY_NAME = {
        "head": 0,
        "left_wrist": 1,
        "right_wrist": 2,
    }
    _POSITIVE_FLOAT_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")

    def __init__(self, context: XrTeleoperateUiContext) -> None:
        self._context = context

    def capabilities(self) -> UiCapabilities:
        state = self._legacy_state()
        recording = self._mapping(state.get("recording"))
        provider = self._mapping(state.get("provider"))
        teleop = self._mapping(state.get("teleop"))
        recording_active = self._recording_active(recording)
        validation_pending = bool(recording.get("validation_pending", False))
        recording_enabled = bool(recording.get("enabled", False))
        started = bool(teleop.get("started", False))
        stopping = bool(teleop.get("stopping", False))
        active_provider = str(provider.get("active_provider", "unknown"))

        return UiCapabilities(
            backend_id=self.backend_id,
            backend_name=self.backend_name,
            commands=(
                self._capability("control.start", not stopping, "teleop is stopping"),
                self._capability("control.stop", not stopping, "teleop is already stopping"),
                self._capability("control.home", started, "teleop has not started"),
                self._capability(
                    "control.recenter",
                    started and self._context.recenter_enabled,
                    "head reference recentering is unavailable in the current teleop mode"
                    if not self._context.recenter_enabled
                    else "teleop has not started",
                ),
                self._capability(
                    "record.start",
                    recording_enabled and not recording_active and not validation_pending,
                    self._record_start_disabled_reason(recording_enabled, recording_active, validation_pending),
                ),
                self._capability("record.stop", recording_active, "recording is not active or armed"),
                self._capability("record.cancel", recording_active, "recording is not active or armed"),
                self._capability(
                    "record.root.set",
                    recording_enabled and not recording_active and not validation_pending,
                    self._record_start_disabled_reason(recording_enabled, recording_active, validation_pending),
                    required=("root_dir",),
                ),
                self._capability(
                    "provider.hold",
                    not recording_active,
                    "recording is active or armed",
                ),
                self._capability(
                    "provider.xr",
                    not recording_active,
                    "recording is active or armed",
                ),
                self._capability(
                    "replay.start",
                    started and not recording_active and active_provider in {"hold", "raw_replay"},
                    self._replay_start_disabled_reason(started, recording_active, active_provider),
                    required=("dataset_root", "episode_name", "base_source"),
                ),
                self._capability(
                    "replay.stop",
                    active_provider == "raw_replay",
                    "raw replay is not active",
                ),
                self._capability(
                    "inference.start",
                    not recording_active and active_provider in {"hold", "xr_live"},
                    self._inference_start_disabled_reason(recording_active, active_provider),
                    required=("prompt",),
                ),
                self._capability(
                    "inference.stop",
                    active_provider == "online_inference",
                    "online inference is not active",
                ),
            ),
        )

    def consume_intent(self, intent: UiIntent) -> CommandOutcome:
        command, payload, rejection = self._translate(intent)
        if rejection:
            return CommandOutcome(
                command_id=intent.id,
                status=CommandStatus.REJECTED,
                code="intent_rejected",
                message=rejection,
            )
        if command is None:
            raise RuntimeError(f"intent {intent.type!r} has neither a command nor a rejection")
        submitted = self._context.command_bus.submit(command, source=intent.source, payload=payload)
        return CommandOutcome(
            command_id=intent.id,
            status=CommandStatus.SUCCEEDED,
            code="accepted_by_control_layer",
            message="intent was accepted and queued for the existing control thread",
            details={
                "legacy_command": submitted.name.value,
                "submitted_monotonic_ns": submitted.created_monotonic_ns,
            },
        )

    def snapshot(self) -> UiSnapshot:
        legacy_state = self._legacy_state()
        recording = self._mapping(legacy_state.get("recording"))
        provider = self._mapping(legacy_state.get("provider"))
        playback = self._mapping(legacy_state.get("playback"))
        teleop = self._mapping(legacy_state.get("teleop"))
        alerts = self._alerts(recording, provider)
        return UiSnapshot(
            backend_id=self.backend_id,
            runtime={
                "state": self._runtime_state(teleop, provider),
                "legacy_schema": str(legacy_state.get("schema", "unknown")),
            },
            control=teleop,
            recording=recording,
            inference=self._mapping(provider.get("online_inference")),
            replay={"provider": provider.get("real_replay", {}), "playback": playback},
            cameras=self._context.camera_status_getter(),
            episodes={},
            alerts=alerts,
            robot_detail={
                "left": self._mapping(legacy_state.get("left")),
                "right": self._mapping(legacy_state.get("right")),
                "provider": provider,
            },
            updated_monotonic_ns=time.monotonic_ns(),
        )

    def get_preview(self, stream_id: str) -> tuple[bytes, str] | None:
        camera_id = self._CAMERA_ID_BY_NAME.get(str(stream_id))
        if camera_id is None:
            return None
        frame, _ = self._context.camera_frame_getter(camera_id)
        if frame is None:
            return None
        encoded_ok, encoded = cv2.imencode(".jpg", frame)
        if not encoded_ok:
            raise RuntimeError(f"failed to encode preview stream {stream_id!r} as JPEG")
        return encoded.tobytes(), "image/jpeg"

    def _translate(self, intent: UiIntent) -> tuple[UiCommandName | None, dict[str, Any], str]:
        state = self._legacy_state()
        recording = self._mapping(state.get("recording"))
        provider = self._mapping(state.get("provider"))
        teleop = self._mapping(state.get("teleop"))
        payload = dict(intent.payload or {})
        recording_active = self._recording_active(recording)
        validation_pending = bool(recording.get("validation_pending", False))
        recording_enabled = bool(recording.get("enabled", False))
        active_provider = str(provider.get("active_provider", "unknown"))
        started = bool(teleop.get("started", False))
        stopping = bool(teleop.get("stopping", False))

        if intent.type == "control.start":
            return (None, {}, "teleop is already stopping") if stopping else (UiCommandName.START, {}, "")
        if intent.type == "control.stop":
            return (None, {}, "teleop is already stopping") if stopping else (UiCommandName.STOP, {}, "")
        if intent.type == "control.home":
            return (UiCommandName.HOME, {}, "") if started else (None, {}, "teleop has not started")
        if intent.type == "control.recenter":
            if not started:
                return None, {}, "teleop has not started"
            if not self._context.recenter_enabled:
                return None, {}, "head reference recentering is unavailable in the current teleop mode"
            return UiCommandName.RECENTER, {}, ""
        if intent.type == "record.start":
            reason = self._record_start_disabled_reason(recording_enabled, recording_active, validation_pending)
            return (None, {}, reason) if reason else (UiCommandName.RECORD_TOGGLE, {}, "")
        if intent.type == "record.stop":
            return (UiCommandName.RECORD_TOGGLE, {}, "") if recording_active else (None, {}, "recording is not active or armed")
        if intent.type == "record.cancel":
            return (UiCommandName.RECORD_CANCEL, {}, "") if recording_active else (None, {}, "recording is not active or armed")
        if intent.type == "record.root.set":
            reason = self._record_start_disabled_reason(recording_enabled, recording_active, validation_pending)
            root_dir = str(payload.get("root_dir", "")).strip()
            if reason:
                return None, {}, reason
            if not root_dir:
                return None, {}, "record.root.set requires a non-empty root_dir"
            return UiCommandName.SET_RECORD_ROOT, {"root_dir": root_dir}, ""
        if intent.type == "provider.hold":
            return (None, {}, "recording is active or armed") if recording_active else (UiCommandName.SET_PROVIDER_HOLD, {"reason": "ui_intent"}, "")
        if intent.type == "provider.xr":
            return (None, {}, "recording is active or armed") if recording_active else (UiCommandName.SET_PROVIDER_XR, {"reason": "ui_intent"}, "")
        if intent.type == "replay.start":
            reason = self._replay_start_disabled_reason(started, recording_active, active_provider)
            if reason:
                return None, {}, reason
            return self._replay_start_command(payload)
        if intent.type == "replay.stop":
            return (UiCommandName.STOP_RAW_REPLAY, {"reason": "ui_intent"}, "") if active_provider == "raw_replay" else (None, {}, "raw replay is not active")
        if intent.type == "inference.start":
            reason = self._inference_start_disabled_reason(recording_active, active_provider)
            prompt = str(payload.get("prompt", "")).strip()
            if reason:
                return None, {}, reason
            if not prompt:
                return None, {}, "inference.start requires a non-empty prompt"
            return UiCommandName.START_ONLINE_INFERENCE, {"prompt": prompt}, ""
        if intent.type == "inference.stop":
            return (UiCommandName.STOP_ONLINE_INFERENCE, {"reason": "ui_intent"}, "") if active_provider == "online_inference" else (None, {}, "online inference is not active")
        return None, {}, f"unsupported XR UI intent: {intent.type}"

    @staticmethod
    def _capability(type: str, enabled: bool, disabled_reason: str, required: tuple[str, ...] = ()) -> UiCommandCapability:
        return UiCommandCapability(
            type=type,
            enabled=enabled,
            disabled_reason="" if enabled else disabled_reason,
            parameter_schema={"required": list(required)} if required else {},
        )

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"expected legacy UI state mapping, got {type(value).__name__}")
        return dict(value)

    def _legacy_state(self) -> dict[str, Any]:
        _, state = self._context.state_store.snapshot()
        return state

    @staticmethod
    def _recording_active(recording: Mapping[str, Any]) -> bool:
        alignment = recording.get("last_alignment")
        armed = bool(alignment.get("waiting_for_first_frame", False)) if isinstance(alignment, Mapping) else False
        return bool(recording.get("active", False)) or armed or str(recording.get("phase", "")) == "armed"

    @staticmethod
    def _record_start_disabled_reason(enabled: bool, active: bool, validation_pending: bool) -> str:
        if not enabled:
            return "recording is disabled; restart with --record"
        if active:
            return "recording is already active or armed"
        if validation_pending:
            return "episode validation is still running"
        return ""

    @staticmethod
    def _replay_start_disabled_reason(started: bool, recording_active: bool, provider: str) -> str:
        if not started:
            return "teleop has not started"
        if recording_active:
            return "recording is active or armed"
        if provider not in {"hold", "raw_replay"}:
            return f"real replay requires provider=hold, got {provider or 'unknown'}"
        return ""

    @staticmethod
    def _inference_start_disabled_reason(recording_active: bool, provider: str) -> str:
        if recording_active:
            return "recording is active or armed"
        if provider == "raw_replay":
            return "online inference cannot start while raw replay is active"
        if provider == "online_inference":
            return "online inference is already active"
        if provider not in {"hold", "xr_live"}:
            return f"online inference cannot start from provider={provider or 'unknown'}"
        return ""

    @staticmethod
    def _runtime_state(teleop: Mapping[str, Any], provider: Mapping[str, Any]) -> str:
        if bool(teleop.get("stopping", False)):
            return "stopping"
        if not bool(teleop.get("ready", False)):
            return "initializing"
        return str(provider.get("active_provider", "ready"))

    @staticmethod
    def _alerts(recording: Mapping[str, Any], provider: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        alerts: list[Mapping[str, Any]] = []
        recording_error = str(recording.get("error", "")).strip()
        if recording_error:
            alerts.append({"source": "recording", "message": recording_error})
        base = recording.get("base")
        if isinstance(base, Mapping) and bool(base.get("control_fault", False)):
            alerts.append({"source": "base", "message": str(base.get("fault_reason") or "base STOP is not confirmed")})
        inference = provider.get("online_inference")
        if isinstance(inference, Mapping) and str(inference.get("error", "")).strip():
            alerts.append({"source": "inference", "message": str(inference["error"])})
        replay = provider.get("real_replay")
        if isinstance(replay, Mapping) and str(replay.get("error", "")).strip():
            alerts.append({"source": "replay", "message": str(replay["error"])})
        return tuple(alerts)

    @classmethod
    def _replay_start_command(cls, payload: Mapping[str, Any]) -> tuple[UiCommandName | None, dict[str, Any], str]:
        dataset_root = str(payload.get("dataset_root", "")).strip()
        episode_name = str(payload.get("episode_name", "")).strip()
        base_source = str(payload.get("base_source", "")).strip()
        arm_source = str(payload.get("arm_source", "action")).strip()
        if not dataset_root:
            return None, {}, "replay.start requires dataset_root"
        if not episode_name:
            return None, {}, "replay.start requires episode_name"
        tail = episode_name.rsplit("_", 1)[-1]
        if not tail.isdigit():
            return None, {}, "replay.start episode_name must end with a numeric episode index"
        if base_source not in {"none", "action"}:
            return None, {}, "replay.start base_source must be one of: none, action"
        if arm_source not in {"action", "state", "fk_cmd_pose"}:
            return None, {}, "replay.start arm_source must be one of: action, state, fk_cmd_pose"
        speed_scale_text = str(payload.get("speed_scale", 1.0)).strip()
        if cls._POSITIVE_FLOAT_RE.fullmatch(speed_scale_text) is None:
            return None, {}, "replay.start speed_scale must be a positive finite number"
        speed_scale = float(speed_scale_text)
        if speed_scale <= 0.0:
            return None, {}, "replay.start speed_scale must be positive"
        return (
            UiCommandName.START_RAW_REPLAY,
            {
                "dataset_root": dataset_root,
                "episode_index": int(tail),
                "episode_name": episode_name,
                "arm_source": arm_source,
                "base_source": base_source,
                "speed_scale": speed_scale,
            },
            "",
        )
