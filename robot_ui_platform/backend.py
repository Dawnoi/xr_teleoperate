from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from robot_ui_platform.contracts import CommandOutcome, UiCapabilities, UiIntent, UiSnapshot


class UiBackend(Protocol):
    """Robot-specific adapter consumed by the shared UI host."""

    def capabilities(self) -> UiCapabilities: ...

    def consume_intent(self, intent: UiIntent) -> CommandOutcome: ...

    def snapshot(self) -> UiSnapshot: ...

    def get_preview(self, stream_id: str) -> tuple[bytes, str] | None: ...


UiBackendBuilder = Callable[[Any], UiBackend]


class UiBackendFactory:
    """Registry-backed factory for robot repository backend adapters."""

    _builders: dict[str, UiBackendBuilder] = {}

    @classmethod
    def register(cls, backend_id: str, builder: UiBackendBuilder) -> None:
        normalized_id = str(backend_id).strip()
        if not normalized_id:
            raise ValueError("backend_id must not be empty")
        if normalized_id in cls._builders:
            raise ValueError(f"UI backend already registered: {normalized_id}")
        cls._builders[normalized_id] = builder

    @classmethod
    def create(cls, backend_id: str, runtime_context: Any) -> UiBackend:
        normalized_id = str(backend_id).strip()
        builder = cls._builders.get(normalized_id)
        if builder is None:
            known = ", ".join(sorted(cls._builders)) or "<none>"
            raise ValueError(f"unknown UI backend {normalized_id!r}; registered backends: {known}")
        backend = builder(runtime_context)
        capabilities = backend.capabilities()
        if capabilities.backend_id != normalized_id:
            raise ValueError(
                f"UI backend factory id {normalized_id!r} does not match adapter capability id "
                f"{capabilities.backend_id!r}"
            )
        return backend

    @classmethod
    def registered_ids(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._builders))
