from __future__ import annotations

import numpy as np

from robot_ui_platform import CommandStatus, UiIntent
from teleop.ui.command_bus import UiCommandBus, UiCommandName
from teleop.ui.state_store import UiStateStore
from teleop.ui_backend.xr_teleoperate import XrTeleoperateBackend, XrTeleoperateUiContext


def build_backend(state: dict) -> tuple[XrTeleoperateBackend, UiCommandBus]:
    command_bus = UiCommandBus()
    context = XrTeleoperateUiContext(
        command_bus=command_bus,
        state_store=UiStateStore(state),
        camera_status_getter=lambda: {
            "streams": [
                {
                    "camera_name": "head",
                    "shared_age_ms": 1,
                }
            ]
        },
        camera_frame_getter=lambda camera_id: (np.zeros((12, 16, 3), dtype=np.uint8), {"camera_id": camera_id}),
        recenter_enabled=False,
    )
    return XrTeleoperateBackend(context), command_bus


def test_xr_backend_translates_high_level_inference_intent_to_existing_command_bus():
    backend, command_bus = build_backend(
        {
            "schema": "data_collector/v1",
            "recording": {"enabled": True, "active": False, "phase": "idle"},
            "provider": {"active_provider": "hold", "online_inference": {}},
            "teleop": {"started": True, "ready": True, "stopping": False},
        }
    )

    outcome = backend.consume_intent(UiIntent(id="inference-1", type="inference.start", payload={"prompt": "pick cube"}))

    assert outcome.status is CommandStatus.SUCCEEDED
    assert outcome.code == "accepted_by_control_layer"
    command = command_bus.drain()
    assert len(command) == 1
    assert command[0].name is UiCommandName.START_ONLINE_INFERENCE
    assert command[0].payload == {"prompt": "pick cube"}


def test_xr_backend_rejects_invalid_intents_without_enqueuing_control_commands():
    backend, command_bus = build_backend(
        {
            "recording": {"enabled": True, "active": True, "phase": "recording"},
            "provider": {"active_provider": "raw_replay"},
            "teleop": {"started": True, "ready": True, "stopping": False},
        }
    )

    outcome = backend.consume_intent(UiIntent(id="inference-2", type="inference.start", payload={"prompt": "pick cube"}))

    assert outcome.status is CommandStatus.REJECTED
    assert outcome.message == "recording is active or armed"
    assert command_bus.drain() == []


def test_xr_backend_exposes_normalized_snapshot_and_jpeg_preview():
    backend, _ = build_backend(
        {
            "schema": "data_collector/v1",
            "left": {"q_fb": [0.0] * 8},
            "right": {"q_fb": [0.0] * 8},
            "recording": {"enabled": True, "active": False, "phase": "idle"},
            "provider": {"active_provider": "xr_live", "online_inference": {}},
            "teleop": {"started": True, "ready": True, "stopping": False},
        }
    )

    snapshot = backend.snapshot()
    preview = backend.get_preview("head")

    assert snapshot.backend_id == "xr_teleoperate"
    assert snapshot.runtime["state"] == "xr_live"
    assert snapshot.cameras["streams"][0]["camera_name"] == "head"
    assert preview is not None
    assert preview[1] == "image/jpeg"
    assert preview[0].startswith(b"\xff\xd8")
