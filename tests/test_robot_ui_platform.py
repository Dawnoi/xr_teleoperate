from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from robot_ui_platform import (
    CommandOutcome,
    CommandStatus,
    UiBackendFactory,
    UiCapabilities,
    UiCommandCapability,
    UiHost,
    UiIntent,
    UiSnapshot,
)


@dataclass
class DemoBackend:
    backend_id: str = "demo"

    def capabilities(self) -> UiCapabilities:
        return UiCapabilities(
            backend_id=self.backend_id,
            backend_name="Demo backend",
            commands=(UiCommandCapability(type="record.start", enabled=True),),
        )

    def consume_intent(self, intent: UiIntent) -> CommandOutcome:
        return CommandOutcome(
            command_id=intent.id,
            status=CommandStatus.SUCCEEDED,
            code="demo_completed",
        )

    def snapshot(self) -> UiSnapshot:
        return UiSnapshot(backend_id=self.backend_id, runtime={"state": "ready"})

    def get_preview(self, stream_id: str):
        return None


def test_backend_factory_creates_registered_adapter_once():
    original_builders = dict(UiBackendFactory._builders)
    UiBackendFactory._builders.clear()
    UiBackendFactory.register("demo", lambda context: DemoBackend(backend_id=context))

    backend = UiBackendFactory.create("demo", "demo")

    assert backend.capabilities().backend_id == "demo"
    assert UiBackendFactory.registered_ids() == ("demo",)
    UiBackendFactory._builders.clear()
    UiBackendFactory._builders.update(original_builders)


def test_command_store_idempotently_returns_first_submission():
    backend = DemoBackend()
    host = UiHost(backend=backend, initial_snapshot=backend.snapshot())
    intent = UiIntent(id="same-intent", type="record.start")

    first = host.command_store.submit(intent)
    second = host.command_store.submit(intent)

    assert first == second
    assert host.drain_intents() == [intent]
    completed = host.complete_intent(backend.consume_intent(intent))
    assert completed.status is CommandStatus.SUCCEEDED
    assert host.command_store.outcome(intent.id) == completed


def test_http_host_queues_intent_and_publishes_completed_status():
    backend = DemoBackend()
    host = UiHost(backend=backend, initial_snapshot=backend.snapshot(), port=0)
    host.start()
    base_url = f"http://127.0.0.1:{host.port}"

    request_body = json.dumps({"id": "record-1", "type": "record.start"}).encode("utf-8")
    request = Request(
        f"{base_url}/api/v1/intents",
        data=request_body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=2.0) as response:
        queued = json.loads(response.read())

    assert queued["status"] == "queued"
    intent = host.drain_intents()[0]
    host.complete_intent(backend.consume_intent(intent))
    with urlopen(f"{base_url}/api/v1/commands/record-1", timeout=2.0) as response:
        completed = json.loads(response.read())

    assert completed["status"] == "succeeded"
    host.stop()


def test_http_host_serves_the_shared_ui_shell():
    backend = DemoBackend()
    host = UiHost(backend=backend, initial_snapshot=backend.snapshot(), port=0)
    host.start()
    base_url = f"http://127.0.0.1:{host.port}"

    with urlopen(f"{base_url}/", timeout=2.0) as response:
        index = response.read().decode("utf-8")
    with urlopen(f"{base_url}/assets/app.js", timeout=2.0) as response:
        app_js = response.read().decode("utf-8")

    assert "UNIFIED ROBOT UI" in index
    assert "/api/v1/intents" in app_js
    host.stop()


def test_snapshot_version_conflict_rejects_stale_intent():
    backend = DemoBackend()
    host = UiHost(backend=backend, initial_snapshot=backend.snapshot(), port=0)
    host.start()
    host.publish_snapshot(UiSnapshot(backend_id="demo", runtime={"state": "running"}))
    base_url = f"http://127.0.0.1:{host.port}"
    request = Request(
        f"{base_url}/api/v1/intents",
        data=json.dumps(
            {
                "id": "stale-record",
                "type": "record.start",
                "expected_snapshot_version": 0,
            }
        ).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=2.0)

    assert error.value.code == 409
    assert host.drain_intents() == []
    host.stop()
