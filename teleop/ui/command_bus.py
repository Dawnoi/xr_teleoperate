from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from threading import Event, Lock
from typing import Any
from uuid import uuid4


class UiCommandName(str, Enum):
    START = "start"
    STOP = "stop"
    HOME = "home"
    RECENTER = "recenter"
    RECORD_TOGGLE = "record_toggle"
    RECORD_CANCEL = "record_cancel"
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"
    SET_RECORD_ROOT = "set_record_root"
    SET_PROVIDER_HOLD = "set_provider_hold"
    SET_PROVIDER_XR = "set_provider_xr"
    START_RAW_REPLAY = "start_raw_replay"
    STOP_RAW_REPLAY = "stop_raw_replay"
    START_ONLINE_INFERENCE = "start_online_inference"
    STOP_ONLINE_INFERENCE = "stop_online_inference"
    LOAD_OFFLINE_REPLAY = "load_offline_replay"
    START_OFFLINE_REPLAY = "start_offline_replay"
    PAUSE_OFFLINE_REPLAY = "pause_offline_replay"
    RESUME_OFFLINE_REPLAY = "resume_offline_replay"
    STOP_OFFLINE_REPLAY = "stop_offline_replay"
    SEEK_OFFLINE_REPLAY = "seek_offline_replay"
    OFFLINE_REPLAY_CURVES = "offline_replay_curves"
    OFFLINE_REPLAY_IMAGE = "offline_replay_image"
    LOAD_ONLINE_REPLAY = "load_online_replay"
    DELETE_EPISODES = "delete_episodes"
    UPDATE_EPISODE_TASK_DESCRIPTION = "update_episode_task_description"
    START_EXPORT = "start_export"
    EXPORT_OUTPUTS = "export_outputs"
    EXPORT_OUTPUT_GET = "export_output_get"
    EXPORT_OUTPUT_EPISODES_GET = "export_output_episodes_get"


class UiCommandCompletionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class UiCommand:
    request_id: str
    name: UiCommandName
    created_monotonic_ns: int
    source: str = "web"
    payload: dict[str, Any] | None = None
    track_completion: bool = False


@dataclass(frozen=True)
class UiCommandCompletion:
    request_id: str
    status: UiCommandCompletionStatus
    message: str
    completed_monotonic_ns: int
    details: dict[str, Any] | None = None


@dataclass
class _PendingCommand:
    completion_event: Event
    completion: UiCommandCompletion | None = None


class UiCommandBus:
    """Thread-safe command queue with optional control-thread completion receipts."""

    def __init__(self) -> None:
        self._queue: Queue[UiCommand] = Queue()
        self._completion_lock = Lock()
        self._pending: dict[str, _PendingCommand] = {}
        self._cancelled_request_ids: set[str] = set()
        self._online_inference_stop_lock = Lock()
        self._online_inference_stop_count = 0
        self._raw_replay_stop_lock = Lock()
        self._raw_replay_stop_count = 0

    def submit(
        self,
        name: UiCommandName | str,
        *,
        source: str = "web",
        payload: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> UiCommand:
        command_name = name if isinstance(name, UiCommandName) else UiCommandName(str(name))
        if payload is not None and not isinstance(payload, dict):
            raise TypeError("UI command payload must be a dict when provided")
        normalized_source = str(source).strip()
        if not normalized_source:
            raise ValueError("UI command source must be non-empty")
        normalized_request_id = str(request_id or uuid4()).strip()
        if not normalized_request_id:
            raise ValueError("UI command request_id must be non-empty")
        track_completion = request_id is not None
        command = UiCommand(
            request_id=normalized_request_id,
            name=command_name,
            created_monotonic_ns=int(time.monotonic_ns()),
            source=normalized_source,
            payload=dict(payload) if payload is not None else None,
            track_completion=track_completion,
        )
        if track_completion:
            with self._completion_lock:
                if normalized_request_id in self._pending:
                    raise ValueError(f"duplicate UI command request_id: {normalized_request_id}")
                self._pending[normalized_request_id] = _PendingCommand(completion_event=Event())
        if command.name == UiCommandName.STOP_ONLINE_INFERENCE:
            with self._online_inference_stop_lock:
                self._online_inference_stop_count += 1
        if command.name == UiCommandName.STOP_RAW_REPLAY:
            with self._raw_replay_stop_lock:
                self._raw_replay_stop_count += 1
        self._queue.put(command)
        return command

    def complete(
        self,
        command: UiCommand,
        *,
        status: UiCommandCompletionStatus,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> UiCommandCompletion:
        if not command.track_completion:
            raise ValueError("completion requested for a command without tracked request_id")
        if not isinstance(status, UiCommandCompletionStatus):
            raise TypeError("UI command completion status must be UiCommandCompletionStatus")
        normalized_message = str(message).strip()
        if not normalized_message:
            raise ValueError("UI command completion message must be non-empty")
        if details is not None and not isinstance(details, dict):
            raise TypeError("UI command completion details must be a dict when provided")
        completion = UiCommandCompletion(
            request_id=command.request_id,
            status=status,
            message=normalized_message,
            completed_monotonic_ns=int(time.monotonic_ns()),
            details=dict(details) if details is not None else None,
        )
        with self._completion_lock:
            pending = self._pending.get(command.request_id)
            if pending is None:
                raise RuntimeError(f"UI command request_id is not pending: {command.request_id}")
            if pending.completion is not None:
                raise RuntimeError(f"UI command request_id already completed: {command.request_id}")
            pending.completion = completion
            pending.completion_event.set()
        return completion

    def succeed(self, command: UiCommand, message: str, *, details: dict[str, Any] | None = None) -> UiCommandCompletion:
        return self.complete(command, status=UiCommandCompletionStatus.SUCCEEDED, message=message, details=details)

    def reject(self, command: UiCommand, message: str, *, details: dict[str, Any] | None = None) -> UiCommandCompletion:
        return self.complete(command, status=UiCommandCompletionStatus.REJECTED, message=message, details=details)

    def fail(self, command: UiCommand, message: str, *, details: dict[str, Any] | None = None) -> UiCommandCompletion:
        return self.complete(command, status=UiCommandCompletionStatus.FAILED, message=message, details=details)

    def wait_for_completion(self, request_id: str, *, timeout_sec: float) -> UiCommandCompletion | None:
        normalized_request_id = str(request_id).strip()
        if not normalized_request_id:
            raise ValueError("UI command request_id must be non-empty")
        normalized_timeout_sec = float(timeout_sec)
        if normalized_timeout_sec <= 0.0:
            raise ValueError("UI command completion timeout must be positive")
        with self._completion_lock:
            pending = self._pending.get(normalized_request_id)
            if pending is None:
                raise KeyError(f"unknown UI command request_id: {normalized_request_id}")
            completion_event = pending.completion_event
        if not completion_event.wait(timeout=normalized_timeout_sec):
            return None
        with self._completion_lock:
            completion = self._pending[normalized_request_id].completion
        if completion is None:
            raise RuntimeError(f"UI command completion event set without a receipt: {normalized_request_id}")
        return completion

    def completion_for(self, request_id: str) -> UiCommandCompletion | None:
        normalized_request_id = str(request_id).strip()
        if not normalized_request_id:
            raise ValueError("UI command request_id must be non-empty")
        with self._completion_lock:
            pending = self._pending.get(normalized_request_id)
            if pending is None:
                raise KeyError(f"unknown UI command request_id: {normalized_request_id}")
            return pending.completion

    def has_pending_completion(self, request_id: str) -> bool:
        normalized_request_id = str(request_id).strip()
        if not normalized_request_id:
            raise ValueError("UI command request_id must be non-empty")
        with self._completion_lock:
            return normalized_request_id in self._pending

    def forget_completion(self, request_id: str) -> None:
        normalized_request_id = str(request_id).strip()
        if not normalized_request_id:
            raise ValueError("UI command request_id must be non-empty")
        with self._completion_lock:
            self._pending.pop(normalized_request_id, None)

    def cancel(self, request_id: str) -> None:
        """Prevent a timed-out tracked command from reaching the control loop.

        A Provider timeout is terminal from the caller's perspective.  The
        queued intent must therefore be discarded instead of being applied
        after the caller has received an error response.
        """
        normalized_request_id = str(request_id).strip()
        if not normalized_request_id:
            raise ValueError("UI command request_id must be non-empty")
        with self._completion_lock:
            if normalized_request_id not in self._pending:
                raise KeyError(f"unknown UI command request_id: {normalized_request_id}")
            self._cancelled_request_ids.add(normalized_request_id)

    def is_cancelled(self, request_id: str) -> bool:
        normalized_request_id = str(request_id).strip()
        if not normalized_request_id:
            raise ValueError("UI command request_id must be non-empty")
        with self._completion_lock:
            return normalized_request_id in self._cancelled_request_ids

    def drain(self, *, max_commands: int = 64) -> list[UiCommand]:
        limit = int(max_commands)
        if limit <= 0:
            raise ValueError("max_commands must be positive")
        commands: list[UiCommand] = []
        drain_count = min(limit, int(self._queue.qsize()))
        for _ in range(drain_count):
            command = self._queue.get_nowait()
            with self._completion_lock:
                cancelled = command.request_id in self._cancelled_request_ids
                if cancelled:
                    self._cancelled_request_ids.remove(command.request_id)
            if cancelled:
                if command.name == UiCommandName.STOP_ONLINE_INFERENCE:
                    with self._online_inference_stop_lock:
                        self._online_inference_stop_count -= 1
                if command.name == UiCommandName.STOP_RAW_REPLAY:
                    with self._raw_replay_stop_lock:
                        self._raw_replay_stop_count -= 1
                continue
            commands.append(command)
            if command.name == UiCommandName.STOP_ONLINE_INFERENCE:
                with self._online_inference_stop_lock:
                    self._online_inference_stop_count -= 1
            if command.name == UiCommandName.STOP_RAW_REPLAY:
                with self._raw_replay_stop_lock:
                    self._raw_replay_stop_count -= 1
        return commands

    def online_inference_stop_requested(self) -> bool:
        with self._online_inference_stop_lock:
            return self._online_inference_stop_count > 0

    def raw_replay_stop_requested(self) -> bool:
        with self._raw_replay_stop_lock:
            return self._raw_replay_stop_count > 0
