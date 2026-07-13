from __future__ import annotations

import time
from threading import Lock
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Any


class UiCommandName(str, Enum):
    START = "start"
    STOP = "stop"
    HOME = "home"
    RECENTER = "recenter"
    RECORD_TOGGLE = "record_toggle"
    RECORD_CANCEL = "record_cancel"
    SET_PROVIDER_HOLD = "set_provider_hold"
    SET_PROVIDER_XR = "set_provider_xr"
    START_RAW_REPLAY = "start_raw_replay"
    STOP_RAW_REPLAY = "stop_raw_replay"
    START_ONLINE_INFERENCE = "start_online_inference"
    STOP_ONLINE_INFERENCE = "stop_online_inference"


@dataclass(frozen=True)
class UiCommand:
    name: UiCommandName
    created_monotonic_ns: int
    source: str = "web"
    payload: dict[str, Any] | None = None


class UiCommandBus:
    """Thread-safe one-way command queue from the web UI into the teleop loop."""

    def __init__(self) -> None:
        self._queue: Queue[UiCommand] = Queue()
        self._online_inference_stop_lock = Lock()
        self._online_inference_stop_count = 0

    def submit(
        self,
        name: UiCommandName | str,
        *,
        source: str = "web",
        payload: dict[str, Any] | None = None,
    ) -> UiCommand:
        command_name = name if isinstance(name, UiCommandName) else UiCommandName(str(name))
        if payload is not None and not isinstance(payload, dict):
            raise TypeError("UI command payload must be a dict when provided")
        command = UiCommand(
            name=command_name,
            created_monotonic_ns=int(time.monotonic_ns()),
            source=str(source or "web"),
            payload=dict(payload) if payload is not None else None,
        )
        if command.name == UiCommandName.STOP_ONLINE_INFERENCE:
            with self._online_inference_stop_lock:
                self._online_inference_stop_count += 1
        self._queue.put(command)
        return command

    def drain(self, *, max_commands: int = 64) -> list[UiCommand]:
        limit = int(max_commands)
        if limit <= 0:
            raise ValueError("max_commands must be positive")

        commands: list[UiCommand] = []
        drain_count = min(limit, int(self._queue.qsize()))
        for _ in range(drain_count):
            command = self._queue.get_nowait()
            commands.append(command)
            if command.name == UiCommandName.STOP_ONLINE_INFERENCE:
                with self._online_inference_stop_lock:
                    self._online_inference_stop_count -= 1
        return commands

    def online_inference_stop_requested(self) -> bool:
        with self._online_inference_stop_lock:
            return self._online_inference_stop_count > 0
