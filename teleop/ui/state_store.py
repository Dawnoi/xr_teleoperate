from __future__ import annotations

import copy
import threading
from typing import Any


class UiStateStore:
    """Thread-safe latest-state cache for web UI polling and SSE."""

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._version = 0
        self._state: dict[str, Any] = {}
        if initial_state is not None:
            self.update(initial_state)

    def update(self, state: dict[str, Any]) -> int:
        if not isinstance(state, dict):
            raise TypeError("UI state must be a dict")
        with self._lock:
            self._state = copy.deepcopy(state)
            self._version += 1
            return self._version

    def snapshot(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            return self._version, copy.deepcopy(self._state)
