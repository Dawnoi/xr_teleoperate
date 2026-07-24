from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class CommandStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


_TERMINAL_COMMAND_STATUSES = frozenset(
    {CommandStatus.SUCCEEDED, CommandStatus.REJECTED, CommandStatus.FAILED}
)


def _mapping(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


@dataclass(frozen=True, slots=True)
class UiIntent:
    """A high-level request from the browser to a robot backend.

    Intent payloads may describe operations such as ``record.start`` or
    ``inference.stop``. They must never carry raw joint commands.
    """

    type: str
    payload: Mapping[str, Any] | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "web"
    created_monotonic_ns: int = field(default_factory=time.monotonic_ns)
    expected_snapshot_version: int | None = None

    def __post_init__(self) -> None:
        intent_type = str(self.type).strip()
        if not intent_type or "." not in intent_type:
            raise ValueError("UI intent type must use '<domain>.<operation>' form")
        if not str(self.id).strip():
            raise ValueError("UI intent id must not be empty")
        if int(self.created_monotonic_ns) <= 0:
            raise ValueError("UI intent created_monotonic_ns must be positive")
        if self.expected_snapshot_version is not None and int(self.expected_snapshot_version) < 0:
            raise ValueError("UI intent expected_snapshot_version must be non-negative")
        object.__setattr__(self, "type", intent_type)
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "source", str(self.source or "web"))
        object.__setattr__(self, "created_monotonic_ns", int(self.created_monotonic_ns))
        object.__setattr__(self, "payload", _mapping(self.payload, "UI intent payload"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UiIntent":
        if not isinstance(value, Mapping):
            raise TypeError("UI intent request must be a JSON object")
        return cls(
            id=str(value.get("id") or str(uuid.uuid4())),
            type=str(value.get("type") or ""),
            payload=_mapping(value.get("payload"), "UI intent payload"),
            source=str(value.get("source") or "web"),
            created_monotonic_ns=int(value.get("created_monotonic_ns") or time.monotonic_ns()),
            expected_snapshot_version=value.get("expected_snapshot_version"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "payload": dict(self.payload or {}),
            "source": self.source,
            "created_monotonic_ns": self.created_monotonic_ns,
            "expected_snapshot_version": self.expected_snapshot_version,
        }


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    command_id: str
    status: CommandStatus
    code: str = ""
    message: str = ""
    details: Mapping[str, Any] | None = None
    completed_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        if not str(self.command_id).strip():
            raise ValueError("command outcome command_id must not be empty")
        if self.status in _TERMINAL_COMMAND_STATUSES and self.completed_monotonic_ns is None:
            object.__setattr__(self, "completed_monotonic_ns", time.monotonic_ns())
        if self.completed_monotonic_ns is not None and int(self.completed_monotonic_ns) <= 0:
            raise ValueError("completed_monotonic_ns must be positive")
        object.__setattr__(self, "command_id", str(self.command_id))
        object.__setattr__(self, "code", str(self.code))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "details", _mapping(self.details, "command outcome details"))

    @classmethod
    def queued(cls, intent: UiIntent) -> "CommandOutcome":
        return cls(command_id=intent.id, status=CommandStatus.QUEUED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "status": self.status.value,
            "code": self.code,
            "message": self.message,
            "details": dict(self.details or {}),
            "completed_monotonic_ns": self.completed_monotonic_ns,
        }


@dataclass(frozen=True, slots=True)
class UiCommandCapability:
    type: str
    enabled: bool
    disabled_reason: str = ""
    parameter_schema: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        intent_type = str(self.type).strip()
        if not intent_type or "." not in intent_type:
            raise ValueError("capability type must use '<domain>.<operation>' form")
        if self.enabled and self.disabled_reason:
            raise ValueError("enabled capability must not contain a disabled_reason")
        if not self.enabled and not str(self.disabled_reason).strip():
            raise ValueError("disabled capability must explain why it is unavailable")
        object.__setattr__(self, "type", intent_type)
        object.__setattr__(self, "disabled_reason", str(self.disabled_reason))
        object.__setattr__(self, "parameter_schema", _mapping(self.parameter_schema, "capability parameter_schema"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "parameter_schema": dict(self.parameter_schema or {}),
        }


@dataclass(frozen=True, slots=True)
class UiCapabilities:
    backend_id: str
    backend_name: str
    commands: tuple[UiCommandCapability, ...]
    extensions: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not str(self.backend_id).strip():
            raise ValueError("backend_id must not be empty")
        if not str(self.backend_name).strip():
            raise ValueError("backend_name must not be empty")
        command_types = [command.type for command in self.commands]
        if len(command_types) != len(set(command_types)):
            raise ValueError("capability command types must be unique")
        object.__setattr__(self, "backend_id", str(self.backend_id))
        object.__setattr__(self, "backend_name", str(self.backend_name))
        object.__setattr__(self, "extensions", _mapping(self.extensions, "capability extensions"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "backend_name": self.backend_name,
            "commands": [command.as_dict() for command in self.commands],
            "extensions": dict(self.extensions or {}),
        }


@dataclass(frozen=True, slots=True)
class UiSnapshot:
    backend_id: str
    runtime: Mapping[str, Any]
    control: Mapping[str, Any] = field(default_factory=dict)
    recording: Mapping[str, Any] = field(default_factory=dict)
    inference: Mapping[str, Any] = field(default_factory=dict)
    replay: Mapping[str, Any] = field(default_factory=dict)
    cameras: Mapping[str, Any] = field(default_factory=dict)
    episodes: Mapping[str, Any] = field(default_factory=dict)
    alerts: tuple[Mapping[str, Any], ...] = ()
    robot_detail: Mapping[str, Any] = field(default_factory=dict)
    updated_monotonic_ns: int = field(default_factory=time.monotonic_ns)
    version: int = 0

    def __post_init__(self) -> None:
        if not str(self.backend_id).strip():
            raise ValueError("snapshot backend_id must not be empty")
        if int(self.updated_monotonic_ns) <= 0:
            raise ValueError("snapshot updated_monotonic_ns must be positive")
        if int(self.version) < 0:
            raise ValueError("snapshot version must be non-negative")
        object.__setattr__(self, "backend_id", str(self.backend_id))
        object.__setattr__(self, "runtime", _mapping(self.runtime, "snapshot runtime"))
        object.__setattr__(self, "control", _mapping(self.control, "snapshot control"))
        object.__setattr__(self, "recording", _mapping(self.recording, "snapshot recording"))
        object.__setattr__(self, "inference", _mapping(self.inference, "snapshot inference"))
        object.__setattr__(self, "replay", _mapping(self.replay, "snapshot replay"))
        object.__setattr__(self, "cameras", _mapping(self.cameras, "snapshot cameras"))
        object.__setattr__(self, "episodes", _mapping(self.episodes, "snapshot episodes"))
        object.__setattr__(self, "robot_detail", _mapping(self.robot_detail, "snapshot robot_detail"))
        object.__setattr__(self, "alerts", tuple(_mapping(alert, "snapshot alert") for alert in self.alerts))
        object.__setattr__(self, "updated_monotonic_ns", int(self.updated_monotonic_ns))
        object.__setattr__(self, "version", int(self.version))

    def with_version(self, version: int) -> "UiSnapshot":
        return UiSnapshot(
            backend_id=self.backend_id,
            runtime=self.runtime,
            control=self.control,
            recording=self.recording,
            inference=self.inference,
            replay=self.replay,
            cameras=self.cameras,
            episodes=self.episodes,
            alerts=self.alerts,
            robot_detail=self.robot_detail,
            updated_monotonic_ns=self.updated_monotonic_ns,
            version=version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "runtime": dict(self.runtime),
            "control": dict(self.control),
            "recording": dict(self.recording),
            "inference": dict(self.inference),
            "replay": dict(self.replay),
            "cameras": dict(self.cameras),
            "episodes": dict(self.episodes),
            "alerts": [dict(alert) for alert in self.alerts],
            "robot_detail": dict(self.robot_detail),
            "updated_monotonic_ns": self.updated_monotonic_ns,
            "version": self.version,
        }
