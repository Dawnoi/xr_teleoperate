from __future__ import annotations

import threading
from collections import deque

from robot_ui_platform.contracts import CommandOutcome, CommandStatus, UiIntent


class UiCommandStore:
    """Thread-safe command lifecycle store.

    HTTP handlers only enqueue intents. A backend control thread drains and
    completes them after delegating to the existing robot runtime.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: deque[UiIntent] = deque()
        self._outcomes: dict[str, CommandOutcome] = {}
        self._intents: dict[str, UiIntent] = {}

    def submit(self, intent: UiIntent) -> CommandOutcome:
        with self._lock:
            existing = self._outcomes.get(intent.id)
            if existing is not None:
                return existing
            queued = CommandOutcome.queued(intent)
            self._intents[intent.id] = intent
            self._outcomes[intent.id] = queued
            self._pending.append(intent)
            return queued

    def drain(self, *, max_commands: int = 64) -> list[UiIntent]:
        if int(max_commands) <= 0:
            raise ValueError("max_commands must be positive")
        with self._lock:
            commands: list[UiIntent] = []
            while self._pending and len(commands) < int(max_commands):
                intent = self._pending.popleft()
                self._outcomes[intent.id] = CommandOutcome(
                    command_id=intent.id,
                    status=CommandStatus.RUNNING,
                )
                commands.append(intent)
            return commands

    def complete(self, outcome: CommandOutcome) -> CommandOutcome:
        if outcome.status not in {
            CommandStatus.SUCCEEDED,
            CommandStatus.REJECTED,
            CommandStatus.FAILED,
        }:
            raise ValueError("only terminal command outcomes may complete an intent")
        with self._lock:
            if outcome.command_id not in self._intents:
                raise KeyError(f"unknown UI command id: {outcome.command_id}")
            self._outcomes[outcome.command_id] = outcome
            return outcome

    def outcome(self, command_id: str) -> CommandOutcome | None:
        with self._lock:
            return self._outcomes.get(str(command_id))
