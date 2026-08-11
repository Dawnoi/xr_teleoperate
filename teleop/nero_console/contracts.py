"""Pure-Python builders and validators for the Nero Web Console v1 transport."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


JsonObject = dict[str, Any]

CONFIG_SCHEMA = "nero.web.console-config/v1"
REPLY_SCHEMA = "nero.console.reply/v1"
SNAPSHOT_SCHEMAS = {
    "collector": "nero.web.collector.snapshot/v1",
    "vla": "nero.web.vla.snapshot/v1",
}

_PROVIDER_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_MODES = frozenset(SNAPSHOT_SCHEMAS)


def validate_provider_id(provider_id: str) -> str:
    """Validate and return a Nero v1 provider identifier."""
    if not isinstance(provider_id, str):
        raise TypeError("provider_id must be a string")
    if not _PROVIDER_ID_PATTERN.fullmatch(provider_id):
        raise ValueError(
            "provider_id must match '[A-Za-z][A-Za-z0-9_]*'"
        )
    return provider_id


def provider_namespace(provider_id: str) -> str:
    """Return the fixed ROS namespace for a validated provider identifier."""
    return f"/nero_console/v1/{validate_provider_id(provider_id)}"


def _require_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    if mode not in _MODES:
        raise ValueError("mode must be either 'vla' or 'collector'")
    return mode


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _copy_object(value: Mapping[str, Any], field_name: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a JSON object")
    return dict(value)


def canonical_json(value: object) -> str:
    """Return deterministic JSON and reject non-finite or non-JSON values."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class RevisionTracker:
    """Issue a stable revision until the supplied browser-visible state changes."""

    def __init__(self, instance_id: str) -> None:
        self._instance_id = _require_non_empty_string(instance_id, "instance_id")
        self._fingerprint: str | None = None
        self._sequence = 0

    @property
    def current(self) -> str | None:
        """Return the latest issued revision, if a state has been observed."""
        if self._fingerprint is None:
            return None
        return self._format_revision(self._fingerprint)

    def revision_for(self, state: Mapping[str, Any]) -> str:
        """Return the current revision, advancing only when canonical state changes."""
        encoded_state = canonical_json(_copy_object(state, "state"))
        fingerprint = hashlib.sha256(encoded_state.encode("utf-8")).hexdigest()
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self._sequence += 1
        return self._format_revision(fingerprint)

    def _format_revision(self, fingerprint: str) -> str:
        return f"{self._instance_id}-{self._sequence}-{fingerprint[:16]}"


def build_console_config(
    *,
    mode: str,
    instance: Mapping[str, Any],
    config_revision: str,
    capabilities: Sequence[str],
    robot: Mapping[str, Any],
    streams: Sequence[Mapping[str, Any]],
) -> JsonObject:
    """Build a valid ``nero.web.console-config/v1`` object."""
    validated_mode = _require_mode(mode)
    instance_object = _copy_object(instance, "instance")
    _require_non_empty_string(instance_object.get("id"), "instance.id")
    if not isinstance(capabilities, Sequence) or isinstance(capabilities, (str, bytes)):
        raise TypeError("capabilities must be a JSON array of strings")
    validated_capabilities = [
        _require_non_empty_string(capability, "capabilities item")
        for capability in capabilities
    ]
    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes)):
        raise TypeError("streams must be a JSON array of objects")
    stream_objects = [_copy_object(stream, "streams item") for stream in streams]
    for stream in stream_objects:
        _require_non_empty_string(stream.get("stream_id"), "streams item.stream_id")

    config = {
        "schema": CONFIG_SCHEMA,
        "mode": validated_mode,
        "instance": instance_object,
        "config_revision": _require_non_empty_string(config_revision, "config_revision"),
        "capabilities": validated_capabilities,
        "robot": _copy_object(robot, "robot"),
        "streams": stream_objects,
    }
    canonical_json(config)
    return config


def build_snapshot(
    *,
    mode: str,
    instance_id: str,
    state_revision: str,
    updated_monotonic_ms: int,
    lifecycle: Mapping[str, Any],
    health: Mapping[str, Any],
    data: Mapping[str, Any],
) -> JsonObject:
    """Build a complete mode-specific Nero v1 snapshot object."""
    if isinstance(updated_monotonic_ms, bool) or not isinstance(updated_monotonic_ms, int):
        raise TypeError("updated_monotonic_ms must be an integer")
    if updated_monotonic_ms < 0:
        raise ValueError("updated_monotonic_ms must be non-negative")
    snapshot = {
        "schema": SNAPSHOT_SCHEMAS[_require_mode(mode)],
        "mode": mode,
        "instance_id": _require_non_empty_string(instance_id, "instance_id"),
        "state_revision": _require_non_empty_string(state_revision, "state_revision"),
        "updated_monotonic_ms": updated_monotonic_ms,
        "lifecycle": _copy_object(lifecycle, "lifecycle"),
        "health": _copy_object(health, "health"),
        "data": _copy_object(data, "data"),
    }
    canonical_json(snapshot)
    return snapshot


def build_reply(
    *,
    request_id: str,
    ok: bool,
    code: str = "",
    message: str = "",
    state_revision: str = "",
    result: Mapping[str, Any] | None = None,
) -> JsonObject:
    """Build one terminal reply for a Nero v1 command."""
    if not isinstance(ok, bool):
        raise TypeError("ok must be a boolean")
    reply = {
        "schema": REPLY_SCHEMA,
        "request_id": _require_non_empty_string(request_id, "request_id"),
        "ok": ok,
        "code": _require_non_empty_string(code, "code") if code != "" else "",
        "message": _require_non_empty_string(message, "message") if message != "" else "",
        "state_revision": _require_non_empty_string(state_revision, "state_revision") if state_revision != "" else "",
        "result": {} if result is None else _copy_object(result, "result"),
    }
    canonical_json(reply)
    return reply


def validate_command_basics(
    command: Mapping[str, Any],
    *,
    require_expected_revision: bool = True,
) -> JsonObject:
    """Validate required transport fields and return a detached command object."""
    command_object = _copy_object(command, "command")
    request_id = _require_non_empty_string(command_object.get("request_id"), "command.request_id")
    action = _require_non_empty_string(command_object.get("action"), "command.action")
    if "." not in action or action.startswith(".") or action.endswith("."):
        raise ValueError("command.action must use '<domain>.<operation>' form")

    expected_revision = command_object.get("expected_revision")
    if require_expected_revision:
        expected_revision = _require_non_empty_string(
            expected_revision, "command.expected_revision"
        )
    elif expected_revision is not None:
        expected_revision = _require_non_empty_string(
            expected_revision, "command.expected_revision"
        )

    params = _copy_object(command_object.get("params"), "command.params")
    validated = dict(command_object)
    validated["request_id"] = request_id
    validated["action"] = action
    validated["params"] = params
    if expected_revision is not None:
        validated["expected_revision"] = expected_revision
    canonical_json(validated)
    return validated
