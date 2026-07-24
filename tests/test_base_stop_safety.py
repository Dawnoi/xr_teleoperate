from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from teleop.control_flow.base_command import apply_base_command


def _bridge_class():
    path = Path(__file__).resolve().parents[1] / "core" / "control" / "g1d_agv_bridge.py"
    spec = importlib.util.spec_from_file_location("g1d_agv_bridge_safety_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load bridge module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.G1DAgvBridge


class FakeAgvBridge:
    def __init__(self, acknowledgements: list[str]) -> None:
        self._acknowledgements = list(acknowledgements)
        self.stop_calls = 0
        self.targets: list[tuple[float, float, float, float]] = []

    def stop_sync(self) -> str:
        self.stop_calls += 1
        return self._acknowledgements.pop(0)

    def set_target(self, vx: float, vy: float, vyaw: float, vz: float) -> str:
        self.targets.append((vx, vy, vyaw, vz))
        return "QUEUED TARGET"

    def get_timing_snapshot(self) -> dict[str, object]:
        return {"last_error": "STOP rejected", "healthy": True}


def base_args() -> SimpleNamespace:
    return SimpleNamespace(
        input_mode="controller",
        base_motion=True,
        base_controller="g1d_agv",
        base_command_source="provider",
        base_stick_deadzone=0.0,
        base_max_vx=0.3,
        base_max_vy=0.3,
        base_max_wz=0.3,
        base_max_z=1.0,
    )


def tele_data() -> SimpleNamespace:
    return SimpleNamespace(
        right_ctrl_aButton=False,
        left_ctrl_thumbstick=False,
        right_ctrl_thumbstick=False,
        left_ctrl_thumbstickValue=(0.0, 0.0),
        right_ctrl_thumbstickValue=(0.0, 0.0),
    )


def test_failed_stop_is_not_latched_and_next_frame_retries_stop_without_motion_target():
    bridge = FakeAgvBridge(["ERR STOP -1 0", "OK STOP 0 0"])

    first = apply_base_command(
        args=base_args(),
        tele_data=tele_data(),
        home_return_active=False,
        agv_bridge=bridge,
        base_provider_active=False,
        base_stop_latched=False,
    )
    second = apply_base_command(
        args=base_args(),
        tele_data=tele_data(),
        home_return_active=False,
        agv_bridge=bridge,
        base_stop_fault=first.base_stop_fault,
    )

    assert first.base_stop_latched is False
    assert first.base_stop_fault is True
    assert first.base_stop_error == "STOP rejected"
    assert second.base_stop_latched is True
    assert second.base_stop_fault is False
    assert second.base_control_mode == "base_stop_recovered"
    assert bridge.stop_calls == 2
    assert bridge.targets == []


def test_dead_worker_fault_is_visible_and_rejects_new_motion_targets():
    bridge = object.__new__(_bridge_class())
    bridge.process = SimpleNamespace(poll=lambda: None)
    bridge._io_lock = threading.Lock()
    bridge._cmd_lock = threading.Lock()
    bridge._cmd_cond = threading.Condition(bridge._cmd_lock)
    bridge._running = True
    bridge._worker_thread = threading.Thread(target=lambda: None)
    bridge._fault_reason = ""
    bridge._fault_monotonic_ns = 0
    bridge._command_generation = 0
    bridge._pending_target = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "vz": 0.0, "stamp_ns": 0, "generation": 0}
    bridge._has_pending = False

    with pytest.raises(RuntimeError, match="worker exited unexpectedly"):
        bridge.set_target(0.1, 0.0, 0.0, 0.0)

    assert bridge._fault_reason == "G1D AGV bridge worker exited unexpectedly"


def test_bridge_health_fault_remains_visible_after_stop_is_acknowledged():
    bridge = FakeAgvBridge(["OK STOP 0 0"])
    bridge.get_timing_snapshot = lambda: {
        "last_error": "worker exited unexpectedly",
        "healthy": False,
        "fault_reason": "worker exited unexpectedly",
    }

    result = apply_base_command(
        args=base_args(),
        tele_data=tele_data(),
        home_return_active=False,
        agv_bridge=bridge,
        base_provider_active=True,
        base_intent=SimpleNamespace(vx=0.1, vy=0.0, wz=0.0, z=0.0),
    )

    assert result.base_stop_latched is True
    assert result.base_stop_fault is True
    assert result.base_stop_error == "worker exited unexpectedly"
    assert result.base_control_mode == "base_stop_fault"
    assert bridge.stop_calls == 1
    assert bridge.targets == []


def test_generated_g1d_scene_uses_a_portable_relative_mesh_directory():
    scene = Path("assets/.generated/g1_d_mobile_scene.xml").read_text(encoding="utf-8")

    assert 'meshdir="../g1_d"' in scene
    assert "/home/" not in scene
    assert "/data/" not in scene
