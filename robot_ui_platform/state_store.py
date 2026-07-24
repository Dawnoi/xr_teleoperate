from __future__ import annotations

import threading
from dataclasses import replace

from robot_ui_platform.contracts import UiSnapshot


class UiStateStore:
    """Thread-safe latest snapshot store for polling and SSE."""

    def __init__(self, initial_snapshot: UiSnapshot) -> None:
        self._condition = threading.Condition()
        self._snapshot = initial_snapshot.with_version(0)

    def publish(self, snapshot: UiSnapshot) -> UiSnapshot:
        with self._condition:
            next_version = self._snapshot.version + 1
            self._snapshot = replace(snapshot.with_version(next_version))
            self._condition.notify_all()
            return self._snapshot

    def snapshot(self) -> UiSnapshot:
        with self._condition:
            return self._snapshot

    def wait_for_update(self, version: int, timeout_sec: float) -> UiSnapshot:
        if float(timeout_sec) <= 0.0:
            raise ValueError("timeout_sec must be positive")
        with self._condition:
            if self._snapshot.version == int(version):
                self._condition.wait(timeout=float(timeout_sec))
            return self._snapshot
