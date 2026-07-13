from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class ActiveProviderKind(str, Enum):
    XR_LIVE = "xr_live"
    HOLD = "hold"
    RAW_REPLAY = "raw_replay"
    ONLINE_INFERENCE = "online_inference"


@dataclass
class RealReplayStatus:
    state: str = "idle"
    dataset_root: str = ""
    episode_name: str = ""
    episode_index: int = -1
    arm_source: str = "action"
    speed_scale: float = 1.0
    frame_index: int = -1
    error: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "dataset_root": self.dataset_root,
            "episode_name": self.episode_name,
            "episode_index": self.episode_index,
            "arm_source": self.arm_source,
            "speed_scale": self.speed_scale,
            "frame_index": self.frame_index,
            "error": self.error,
            "reason": self.reason,
        }


@dataclass
class OnlineInferenceStatus:
    state: str = "idle"
    prompt: str = ""
    error: str = ""
    reason: str = ""
    runtime_debug: dict[str, Any] | None = None

    def as_dict(self, provider: Any = None) -> dict[str, Any]:
        debug = {}
        if provider is not None:
            get_debug_snapshot = getattr(provider, "get_debug_snapshot", None)
            if callable(get_debug_snapshot):
                debug = get_debug_snapshot()
        return {
            "state": self.state,
            "prompt": self.prompt,
            "error": self.error,
            "reason": self.reason,
            "debug": debug,
            "runtime_debug": dict(self.runtime_debug or {}),
        }


def _default_replay_provider_factory(**kwargs):
    from core.input.raw_offline import RawEpisodeInputProvider

    return RawEpisodeInputProvider(**kwargs)


class TeleopProviderRuntime:
    """Runtime provider switch for UI-driven XR, hold, replay, and inference control."""

    def __init__(
        self,
        *,
        live_provider: Any,
        live_provider_name: str = "xr",
        replay_provider_factory: Callable[..., Any] | None = None,
        online_provider_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._live_provider = live_provider
        self._live_provider_name = str(live_provider_name or "xr")
        self._replay_provider_factory = replay_provider_factory or _default_replay_provider_factory
        self._online_provider_factory = online_provider_factory
        self._raw_provider = None
        self._online_provider = None
        self._active_provider_kind = ActiveProviderKind.XR_LIVE
        self._real_replay = RealReplayStatus()
        self._online_inference = OnlineInferenceStatus()
        self._last_error = ""
        self._last_reason = "startup"

    @property
    def active_provider_kind(self) -> ActiveProviderKind:
        return self._active_provider_kind

    def active_provider(self):
        if self._active_provider_kind == ActiveProviderKind.XR_LIVE:
            return self._live_provider
        if self._active_provider_kind == ActiveProviderKind.RAW_REPLAY:
            return self._raw_provider
        if self._active_provider_kind == ActiveProviderKind.ONLINE_INFERENCE:
            return self._online_provider
        return None

    def active_input_provider_name(self) -> str:
        if self._active_provider_kind == ActiveProviderKind.RAW_REPLAY:
            return "lerobot_offline"
        if self._active_provider_kind == ActiveProviderKind.XR_LIVE:
            return self._live_provider_name
        if self._active_provider_kind == ActiveProviderKind.ONLINE_INFERENCE:
            return "online_inference"
        return "hold"

    def set_hold(self, *, reason: str = "") -> None:
        if self._active_provider_kind == ActiveProviderKind.ONLINE_INFERENCE:
            self._close_online_provider()
            self._online_inference.state = "stopped"
            self._online_inference.reason = str(reason or "hold")
        self._active_provider_kind = ActiveProviderKind.HOLD
        self._last_reason = str(reason or "hold")
        if self._real_replay.state == "running":
            self._real_replay.state = "stopped"
            self._real_replay.reason = self._last_reason

    def set_live(self, *, reason: str = "") -> None:
        self._raw_provider = None
        if self._active_provider_kind == ActiveProviderKind.ONLINE_INFERENCE:
            self._close_online_provider()
            self._online_inference.state = "stopped"
            self._online_inference.reason = str(reason or "xr_live")
        self._active_provider_kind = ActiveProviderKind.XR_LIVE
        self._last_reason = str(reason or "xr_live")
        if self._real_replay.state == "running":
            self._real_replay.state = "stopped"
            self._real_replay.reason = self._last_reason

    def start_raw_replay(
        self,
        *,
        dataset_root: str | Path,
        episode_index: int,
        episode_name: str = "",
        arm_source: str = "action",
        speed_scale: float = 1.0,
    ) -> None:
        if self._active_provider_kind != ActiveProviderKind.HOLD:
            raise RuntimeError("raw replay can only start while active provider is hold")
        source = str(arm_source or "action")
        if source not in {"action", "state", "fk_cmd_pose"}:
            raise ValueError(f"unsupported raw replay arm_source: {source}")
        scale = float(speed_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("raw replay speed_scale must be positive and finite")

        root = str(dataset_root)
        index = int(episode_index)
        provider = self._replay_provider_factory(
            dataset_root=root,
            episode_index=index,
            arm_source=source,
            speed_scale=scale,
        )
        self._raw_provider = provider
        self._active_provider_kind = ActiveProviderKind.RAW_REPLAY
        self._last_error = ""
        self._last_reason = "raw_replay_start"
        self._real_replay = RealReplayStatus(
            state="running",
            dataset_root=root,
            episode_name=str(episode_name or f"episode_{index:04d}"),
            episode_index=index,
            arm_source=source,
            speed_scale=scale,
            frame_index=-1,
        )

    def stop_raw_replay(self, *, reason: str = "") -> None:
        self._raw_provider = None
        self._active_provider_kind = ActiveProviderKind.HOLD
        self._last_reason = str(reason or "raw_replay_stop")
        if self._real_replay.state == "running":
            self._real_replay.state = "stopped"
        self._real_replay.reason = self._last_reason

    def start_online_inference(self, *, prompt: str) -> None:
        if self._active_provider_kind != ActiveProviderKind.HOLD:
            raise RuntimeError("online inference can only start while active provider is hold")
        if self._online_provider_factory is None:
            raise RuntimeError("online inference provider factory is not configured")

        text = str(prompt or "").strip()
        if not text:
            raise ValueError("online inference prompt is required")
        provider = self._online_provider_factory(prompt=text)
        self._online_provider = provider
        self._active_provider_kind = ActiveProviderKind.ONLINE_INFERENCE
        self._online_inference = OnlineInferenceStatus(
            state="running",
            prompt=text,
            reason="online_inference_start",
        )
        self._last_error = ""
        self._last_reason = "online_inference_start"

    def stop_online_inference(self, *, reason: str = "") -> None:
        self._close_online_provider()
        self._active_provider_kind = ActiveProviderKind.HOLD
        self._online_inference.state = "stopped"
        self._online_inference.reason = str(reason or "online_inference_stop")
        self._last_reason = self._online_inference.reason

    def fail_online_inference(self, message: str) -> None:
        self._close_online_provider()
        self._active_provider_kind = ActiveProviderKind.HOLD
        self._online_inference.state = "error"
        self._online_inference.error = str(message)
        self._online_inference.reason = "online_inference_error"
        self._last_error = self._online_inference.error
        self._last_reason = self._online_inference.reason

    def note_online_inference_runtime_debug(self, debug: dict[str, Any]) -> None:
        if self._active_provider_kind != ActiveProviderKind.ONLINE_INFERENCE:
            return
        self._online_inference.runtime_debug = dict(debug)

    def _close_online_provider(self) -> None:
        provider = self._online_provider
        self._online_provider = None
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    def finish_raw_replay(self, *, reason: str = "") -> None:
        self._raw_provider = None
        self._active_provider_kind = ActiveProviderKind.HOLD
        self._last_reason = str(reason or "raw_replay_finished")
        self._real_replay.state = "finished"
        self._real_replay.reason = self._last_reason

    def fail_raw_replay(self, message: str) -> None:
        self._raw_provider = None
        self._active_provider_kind = ActiveProviderKind.HOLD
        self._last_error = str(message)
        self._last_reason = "raw_replay_error"
        self._real_replay.state = "error"
        self._real_replay.error = self._last_error
        self._real_replay.reason = self._last_reason

    def note_sample(self, sample: Any) -> None:
        if self._active_provider_kind != ActiveProviderKind.RAW_REPLAY or sample is None:
            return
        motion_intent = getattr(sample, "motion_intent", None)
        frame_index = getattr(motion_intent, "frame_index", None)
        if frame_index is not None:
            self._real_replay.frame_index = int(frame_index)

    def status(self) -> dict[str, Any]:
        return {
            "active_provider": self._active_provider_kind.value,
            "input_provider": self.active_input_provider_name(),
            "last_error": self._last_error,
            "last_reason": self._last_reason,
            "real_replay": self._real_replay.as_dict(),
            "online_inference": self._online_inference.as_dict(self._online_provider),
        }
