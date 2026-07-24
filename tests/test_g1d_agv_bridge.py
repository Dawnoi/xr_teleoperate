import importlib.util
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest


def _bridge_module():
    path = Path(__file__).resolve().parents[1] / "core" / "control" / "g1d_agv_bridge.py"
    spec = importlib.util.spec_from_file_location("g1d_agv_bridge_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load bridge module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bridge_class():
    return _bridge_module().G1DAgvBridge


@pytest.fixture
def bridge_factory():
    bridge_cls = _bridge_class()
    bridges = []

    def make(child_code: str, timeout: float = 0.05):
        process = subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        bridge = object.__new__(bridge_cls)
        bridge.process = process
        bridge.response_timeout_sec = timeout
        bridge._io_lock = threading.Lock()
        bridge._cmd_lock = threading.Lock()
        bridge._cmd_cond = threading.Condition(bridge._cmd_lock)
        bridge._stdout_buffer = b""
        bridge._last_ack = ""
        bridge._last_error = ""
        bridge._command_generation = 0
        bridge._running = True
        bridge._stderr_thread = None
        bridge._worker_thread = None
        bridge._pending_target = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "vz": 0.0, "stamp_ns": 0}
        bridge._has_pending = False
        bridge._last_sent_move = (None, None, None)
        bridge._last_sent_height = None
        bridge._last_sent_move_ns = 0
        bridge._last_sent_height_ns = 0
        bridge._stderr_lines = []
        bridges.append(bridge)
        return bridge

    yield make

    for bridge in bridges:
        process = bridge.process
        if process is not None and process.poll() is None:
            process.kill()
        if process is not None:
            process.wait()


def test_success_response_is_returned_and_has_no_error(bridge_factory):
    bridge = bridge_factory("import sys; sys.stdin.readline(); print('OK MOVE 0', flush=True)")

    assert bridge._send_sync("MOVE 0.1 0.0 0.0") == "OK MOVE 0"
    assert bridge._last_ack == "OK MOVE 0"
    assert bridge._last_error == ""


def test_err_response_is_returned_and_marks_last_error(bridge_factory):
    bridge = bridge_factory("import sys; sys.stdin.readline(); print('ERR MOVE -7', flush=True)")

    assert bridge._send_sync("MOVE 0.1 0.0 0.0") == "ERR MOVE -7"
    assert "ERR MOVE -7" in bridge._last_error


def test_timeout_marks_last_error_and_returns_without_blocking(bridge_factory):
    bridge = bridge_factory("import sys; sys.stdin.readline(); import time; time.sleep(10)", timeout=0.02)
    start = time.monotonic()

    assert bridge._send_sync("MOVE 0.1 0.0 0.0") is None
    assert time.monotonic() - start < 0.5
    assert "response timeout" in bridge._last_error


def test_eof_marks_last_error(bridge_factory):
    bridge = bridge_factory("import sys; sys.stdin.readline()")

    assert bridge._send_sync("MOVE 0.1 0.0 0.0") is None
    assert "EOF" in bridge._last_error


def test_nonzero_child_returncode_marks_last_error(bridge_factory):
    bridge = bridge_factory("import sys; sys.stdin.readline(); sys.exit(23)")

    assert bridge._send_sync("MOVE 0.1 0.0 0.0") is None
    assert "returncode=23" in bridge._last_error


def test_close_bounds_no_response_and_does_not_raise(bridge_factory):
    bridge = bridge_factory(
        "import sys; sys.stdin.readline(); import time; time.sleep(10)",
        timeout=0.02,
    )
    start = time.monotonic()

    bridge.close()

    assert time.monotonic() - start < 1.0
    assert bridge.process is None


def test_startup_timeout_terminates_child_process(monkeypatch):
    module = _bridge_module()
    bridge_cls = module.G1DAgvBridge
    started_processes = []
    original_popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        started_processes.append(process)
        return process

    monkeypatch.setattr(module, "BIN_PATH", Path(sys.executable))
    monkeypatch.setattr(module.subprocess, "Popen", tracked_popen)

    with pytest.raises(RuntimeError, match="failed to become ready"):
        bridge_cls(auto_build=False, response_timeout_sec=0.02)

    assert len(started_processes) == 1
    assert started_processes[0].poll() is not None


def test_cpp_ack_is_err_for_nonzero_return_codes():
    source = (
        Path(__file__).resolve().parents[1] / "core" / "control" / "cpp" / "g1d_agv_bridge.cpp"
    ).read_text()

    assert 'ret == 0 ? "OK MOVE " : "ERR MOVE "' in source
    assert 'ret == 0 ? "OK HEIGHT " : "ERR HEIGHT "' in source
    assert 'ret_move == 0 && ret_height == 0' in source
    assert 'stop_ok ? "OK STOP " : "ERR STOP "' in source


def test_stop_sync_invalidates_queued_target_and_confirms_stop():
    bridge_cls = _bridge_class()
    bridge = object.__new__(bridge_cls)
    bridge._cmd_lock = threading.Lock()
    bridge._cmd_cond = threading.Condition(bridge._cmd_lock)
    bridge._running = True
    bridge.process = object()
    bridge._command_generation = 7
    bridge._pending_target = {"vx": 0.2, "vy": 0.0, "vyaw": 0.1, "vz": 0.0, "stamp_ns": 1}
    bridge._has_pending = True
    bridge._last_sent_move = (0.2, 0.0, 0.1)
    bridge._last_sent_height = 0.0
    bridge._last_sent_move_ns = 10
    bridge._last_sent_height_ns = 10
    bridge._send_sync = Mock(return_value="OK STOP 0 0")

    result = bridge.stop_sync()

    assert result == "OK STOP 0 0"
    bridge._send_sync.assert_called_once_with("STOP", allow_fault=True)
    assert bridge._command_generation == 8
    assert bridge._has_pending is False
    assert bridge._pending_target["vx"] == 0.0
    assert bridge._pending_target["vy"] == 0.0
    assert bridge._pending_target["vyaw"] == 0.0
    assert bridge._pending_target["vz"] == 0.0


def test_stop_sync_returns_err_without_raising_and_marks_last_error():
    bridge_cls = _bridge_class()
    bridge = object.__new__(bridge_cls)
    bridge._cmd_lock = threading.Lock()
    bridge._cmd_cond = threading.Condition(bridge._cmd_lock)
    bridge._running = True
    bridge.process = object()
    bridge._command_generation = 7
    bridge._pending_target = {"vx": 0.2, "vy": 0.0, "vyaw": 0.1, "vz": 0.0, "stamp_ns": 1}
    bridge._has_pending = True
    bridge._last_sent_move = (0.2, 0.0, 0.1)
    bridge._last_sent_height = 0.0
    bridge._last_sent_move_ns = 10
    bridge._last_sent_height_ns = 10
    bridge._last_error = ""
    bridge._send_sync = Mock(return_value="ERR STOP -1 0")

    result = bridge.stop_sync()

    assert result == "ERR STOP -1 0"
    assert bridge._last_error == "STOP: child response 'ERR STOP -1 0'"
    assert bridge._command_generation == 8
