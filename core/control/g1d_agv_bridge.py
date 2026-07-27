from __future__ import annotations

import math
import os
import select
import subprocess
import threading
import time
from collections import deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_SRC = REPO_ROOT / "core" / "control" / "cpp" / "g1d_agv_bridge.cpp"
BIN_DIR = REPO_ROOT / "cache" / "bin"
BIN_PATH = BIN_DIR / "g1d_agv_bridge"
UNITREE_SDK2_ROOT = (
    REPO_ROOT.parent / "unitree_sdk2_official_latest"
    if (REPO_ROOT.parent / "unitree_sdk2_official_latest").exists()
    else REPO_ROOT.parent / "unitree_sdk2"
)
UNITREE_INCLUDE = UNITREE_SDK2_ROOT / "include"
UNITREE_LIB = UNITREE_SDK2_ROOT / "lib" / "x86_64" / "libunitree_sdk2.a"
UNITREE_TP_INCLUDE = UNITREE_SDK2_ROOT / "thirdparty" / "include"
UNITREE_TP_LIB_DIR = UNITREE_SDK2_ROOT / "thirdparty" / "lib" / "x86_64"


MOVE_EPS = 1e-5
HEIGHT_EPS = 1e-5
MOVE_KEEPALIVE_SEC = 0.10
HEIGHT_KEEPALIVE_SEC = 0.10
RESPONSE_TIMEOUT_SEC = 1.0
PROCESS_EXIT_TIMEOUT_SEC = 2.0
PROCESS_POLL_SEC = 0.01
HEALTH_MONITOR_SEC = 0.05



def build_g1d_agv_bridge(force_rebuild: bool = False) -> Path:
    if not CPP_SRC.exists():
        raise FileNotFoundError(f"Missing C++ source: {CPP_SRC}")
    if not UNITREE_INCLUDE.exists() or not UNITREE_LIB.exists():
        raise FileNotFoundError(
            "unitree_sdk2 headers/static library not found locally. "
            f"Expected: {UNITREE_INCLUDE} and {UNITREE_LIB}"
        )

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    if (
        BIN_PATH.exists()
        and not force_rebuild
        and BIN_PATH.stat().st_mtime >= max(CPP_SRC.stat().st_mtime, UNITREE_LIB.stat().st_mtime)
    ):
        return BIN_PATH

    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        str(CPP_SRC),
        "-o",
        str(BIN_PATH),
        "-I",
        str(UNITREE_INCLUDE),
        "-I",
        str(UNITREE_TP_INCLUDE),
        "-I",
        str(UNITREE_TP_INCLUDE / "ddscxx"),
        "-L",
        str(UNITREE_TP_LIB_DIR),
        f"-Wl,-rpath,{UNITREE_TP_LIB_DIR}",
        str(UNITREE_LIB),
        "-lddsc",
        "-lddscxx",
        "-lpthread",
    ]
    subprocess.run(cmd, check=True)
    return BIN_PATH



def _summary_stats(values):
    if not values:
        return None
    arr = list(values)
    arr_f = [float(v) for v in arr]
    arr_f.sort()
    n = len(arr_f)
    return {
        "avg_ms": sum(arr_f) / n,
        "p95_ms": arr_f[min(n - 1, max(0, int(round(0.95 * (n - 1)))))],
        "max_ms": arr_f[-1],
    }


class G1DAgvBridge:
    def __init__(
        self,
        network_interface: str | None = None,
        auto_build: bool = True,
        response_timeout_sec: float = RESPONSE_TIMEOUT_SEC,
    ):
        self.network_interface = network_interface or ""
        self.response_timeout_sec = float(response_timeout_sec)
        if not math.isfinite(self.response_timeout_sec) or self.response_timeout_sec <= 0.0:
            raise ValueError(f"response_timeout_sec must be positive, got {response_timeout_sec!r}")
        self.process: subprocess.Popen[bytes] | None = None
        self._io_lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._cmd_cond = threading.Condition(self._cmd_lock)
        self._running = True
        self._command_generation = 0
        self._pending_target = {
            "vx": 0.0,
            "vy": 0.0,
            "vyaw": 0.0,
            "vz": 0.0,
            "stamp_ns": 0,
            "generation": self._command_generation,
        }
        self._has_pending = False
        self._last_sent_move = (None, None, None)
        self._last_sent_height = None
        self._last_sent_move_ns = 0
        self._last_sent_height_ns = 0
        self._move_send_ms = deque(maxlen=400)
        self._height_send_ms = deque(maxlen=400)
        self._cycle_ms = deque(maxlen=400)
        self._queue_delay_ms = deque(maxlen=400)
        self._send_count = 0
        self._send_start_ns = time.perf_counter_ns()
        self._last_ack = ""
        self._last_error = ""
        self._fault_reason = ""
        self._fault_monotonic_ns = 0
        self._stderr_lines = deque(maxlen=50)
        self._stdout_buffer = b""
        self._stderr_thread = None
        self._worker_thread = None
        self._health_thread = None

        if auto_build:
            build_g1d_agv_bridge(force_rebuild=False)
        self._start()

    def _start(self) -> None:
        args = [str(BIN_PATH)]
        if self.network_interface:
            args.append(self.network_interface)
        env = os.environ.copy()
        ld_library_path = env.get("LD_LIBRARY_PATH", "")
        extra = str(UNITREE_TP_LIB_DIR)
        env["LD_LIBRARY_PATH"] = extra if not ld_library_path else f"{extra}:{ld_library_path}"
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        ready = self._readline_with_timeout("startup READY")
        if ready != "READY":
            self._terminate_startup_process()
            raise RuntimeError(
                f"G1D AGV bridge failed to become ready: stdout={ready!r} error={self._last_error!r}"
            )

        if self.process.stderr is not None:
            self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
            self._stderr_thread.start()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()

    def _terminate_startup_process(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        deadline = time.monotonic() + PROCESS_EXIT_TIMEOUT_SEC
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(PROCESS_POLL_SEC)
        if process.poll() is None:
            process.kill()
            deadline = time.monotonic() + PROCESS_EXIT_TIMEOUT_SEC
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(PROCESS_POLL_SEC)
        self.process = None

    def _drain_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        stderr_buffer = b""
        stderr_fd = self.process.stderr.fileno()
        while self._running:
            readable, _, _ = select.select([stderr_fd], [], [], 0.2)
            if not readable:
                continue
            chunk = os.read(stderr_fd, 4096)
            if not chunk:
                break
            stderr_buffer += chunk
            while b"\n" in stderr_buffer:
                raw_line, stderr_buffer = stderr_buffer.split(b"\n", 1)
                msg = raw_line.decode("utf-8", errors="replace").strip()
                if msg:
                    with self._cmd_lock:
                        self._stderr_lines.append(msg)
                        self._last_error = msg

    def _set_last_error(self, message: str) -> None:
        with self._cmd_lock:
            self._last_error = str(message)

    def _readline_with_timeout(self, operation: str) -> str | None:
        deadline = time.monotonic() + self.response_timeout_sec
        while True:
            newline_index = self._stdout_buffer.find(b"\n")
            if newline_index >= 0:
                raw_line = self._stdout_buffer[:newline_index]
                self._stdout_buffer = self._stdout_buffer[newline_index + 1 :]
                return raw_line.decode("utf-8", errors="replace").rstrip("\r")

            process = self.process
            if process is None or process.stdout is None:
                self._set_last_error(f"{operation}: child stdout is unavailable")
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._set_last_error(f"{operation}: response timeout after {self.response_timeout_sec:.3f}s")
                return None

            stdout_fd = process.stdout.fileno()
            readable, _, _ = select.select([stdout_fd], [], [], remaining)
            if not readable:
                returncode = process.poll()
                if returncode is not None:
                    self._set_last_error(f"{operation}: child exited before response; returncode={returncode}")
                else:
                    self._set_last_error(f"{operation}: response timeout after {self.response_timeout_sec:.3f}s")
                return None
            chunk = os.read(stdout_fd, 4096)
            if not chunk:
                self._set_last_error(
                    f"{operation}: EOF while waiting for response; returncode={process.poll()}"
                )
                return None
            self._stdout_buffer += chunk

    @staticmethod
    def _ack_is_success(command: str, ack: str | None) -> bool:
        if command in {"MOVE", "HEIGHT"}:
            return ack == f"OK {command} 0"
        if command == "STOP":
            return ack == "OK STOP 0 0"
        if command == "QUIT":
            return ack == "BYE"
        return False

    def _mark_fault(self, reason: str) -> None:
        message = str(reason).strip()
        if not message:
            raise ValueError("bridge fault reason must not be empty")
        with self._cmd_lock:
            if not getattr(self, "_fault_reason", ""):
                self._fault_reason = message
                self._fault_monotonic_ns = time.monotonic_ns()
            self._last_error = message

    def _refresh_health(self) -> None:
        process = self.process
        if process is None:
            self._mark_fault("G1D AGV bridge process is unavailable")
            return
        poll = getattr(process, "poll", None)
        if poll is not None:
            returncode = poll()
            if returncode is not None:
                self._mark_fault(f"G1D AGV bridge process exited: returncode={returncode}")
                return
        worker = self._worker_thread
        if worker is not None and not worker.is_alive() and self._running:
            self._mark_fault("G1D AGV bridge worker exited unexpectedly")

    def _health_loop(self) -> None:
        while self._running:
            self._refresh_health()
            with self._cmd_lock:
                if getattr(self, "_fault_reason", ""):
                    return
            time.sleep(HEALTH_MONITOR_SEC)

    def _send_sync(
        self,
        line: str,
        *,
        expected_generation: int | None = None,
        allow_fault: bool = False,
    ) -> str | None:
        command = line.split(maxsplit=1)[0]
        self._refresh_health()
        with self._cmd_lock:
            if getattr(self, "_fault_reason", "") and not allow_fault:
                raise RuntimeError(f"G1D AGV bridge is faulted: {self._fault_reason}")
        with self._io_lock:
            with self._cmd_lock:
                if expected_generation is not None and int(expected_generation) != self._command_generation:
                    return None
            process = self.process
            if process is None or process.stdin is None or process.stdout is None:
                self._mark_fault(f"{command}: child process is not running")
                return None
            returncode = process.poll()
            if returncode is not None:
                self._mark_fault(f"{command}: child process exited; returncode={returncode}")
                return None
            try:
                process.stdin.write((line + "\n").encode("utf-8"))
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._mark_fault(f"G1D AGV bridge pipe write failed: {exc}")
                raise
            ack = self._readline_with_timeout(command)
        if ack is None:
            self._mark_fault(self._last_error or f"{command}: no response")
            return None
        with self._cmd_lock:
            self._last_ack = ack
        if not self._ack_is_success(command, ack):
            self._mark_fault(f"{command}: child response {ack!r}")
        return ack

    def _worker_loop(self) -> None:
        while True:
            with self._cmd_cond:
                while self._running and not self._has_pending:
                    self._cmd_cond.wait(timeout=0.2)
                if not self._running:
                    return
                target = dict(self._pending_target)
                self._has_pending = False

            cycle_start_ns = time.perf_counter_ns()
            queue_delay_ms = max(0.0, (cycle_start_ns - int(target.get("stamp_ns") or cycle_start_ns)) / 1e6)
            move_ms = 0.0
            height_ms = 0.0

            move_tuple = (
                float(target["vx"]),
                float(target["vy"]),
                float(target["vyaw"]),
            )
            height_value = float(target["vz"])
            target_generation = int(target["generation"])

            if self._should_send_move(move_tuple):
                start_ns = time.perf_counter_ns()
                move_ack = self._send_sync(
                    f"MOVE {move_tuple[0]:.6f} {move_tuple[1]:.6f} {move_tuple[2]:.6f}",
                    expected_generation=target_generation,
                )
                if not self._ack_is_success("MOVE", move_ack):
                    if self._target_was_superseded_without_fault(target_generation):
                        continue
                    return
                move_ms = (time.perf_counter_ns() - start_ns) / 1e6
                self._last_sent_move = move_tuple
                self._last_sent_move_ns = time.perf_counter_ns()

            if self._should_send_height(height_value):
                start_ns = time.perf_counter_ns()
                height_ack = self._send_sync(
                    f"HEIGHT {height_value:.6f}",
                    expected_generation=target_generation,
                )
                if not self._ack_is_success("HEIGHT", height_ack):
                    if self._target_was_superseded_without_fault(target_generation):
                        continue
                    return
                height_ms = (time.perf_counter_ns() - start_ns) / 1e6
                self._last_sent_height = height_value
                self._last_sent_height_ns = time.perf_counter_ns()

            cycle_ms = (time.perf_counter_ns() - cycle_start_ns) / 1e6
            with self._cmd_lock:
                self._move_send_ms.append(float(move_ms))
                self._height_send_ms.append(float(height_ms))
                self._cycle_ms.append(float(cycle_ms))
                self._queue_delay_ms.append(float(queue_delay_ms))
                self._send_count += 1

    def _target_was_superseded_without_fault(self, target_generation: int) -> bool:
        with self._cmd_lock:
            return (
                int(target_generation) != self._command_generation
                and not bool(self._fault_reason)
            )

    def _should_send_move(self, move_tuple) -> bool:
        prev = self._last_sent_move
        if prev[0] is None:
            return True
        changed = any(abs(float(a) - float(b)) > MOVE_EPS for a, b in zip(move_tuple, prev))
        if changed:
            return True
        if self._last_sent_move_ns <= 0:
            return True
        return (time.perf_counter_ns() - self._last_sent_move_ns) >= int(MOVE_KEEPALIVE_SEC * 1e9)

    def _should_send_height(self, height_value: float) -> bool:
        prev = self._last_sent_height
        if prev is None:
            return True
        changed = abs(float(height_value) - float(prev)) > HEIGHT_EPS
        if changed:
            return True
        if self._last_sent_height_ns <= 0:
            return True
        return (time.perf_counter_ns() - self._last_sent_height_ns) >= int(HEIGHT_KEEPALIVE_SEC * 1e9)

    def set_target(self, vx: float, vy: float, vyaw: float, vz: float = 0.0) -> str:
        stamp_ns = time.perf_counter_ns()
        self._refresh_health()
        with self._cmd_cond:
            if not self._running:
                raise RuntimeError("G1D AGV bridge is closed")
            if getattr(self, "_fault_reason", ""):
                raise RuntimeError(f"G1D AGV bridge is faulted: {self._fault_reason}")
            self._command_generation += 1
            self._pending_target = {
                "vx": float(vx),
                "vy": float(vy),
                "vyaw": float(vyaw),
                "vz": float(vz),
                "stamp_ns": stamp_ns,
                "generation": self._command_generation,
            }
            self._has_pending = True
            self._cmd_cond.notify()
        return "QUEUED TARGET"

    def move(self, vx: float, vy: float, vyaw: float) -> str:
        with self._cmd_lock:
            vz = float(self._pending_target.get("vz", 0.0))
        return self.set_target(vx, vy, vyaw, vz)

    def height_adjust(self, vz: float) -> str:
        with self._cmd_lock:
            vx = float(self._pending_target.get("vx", 0.0))
            vy = float(self._pending_target.get("vy", 0.0))
            vyaw = float(self._pending_target.get("vyaw", 0.0))
        return self.set_target(vx, vy, vyaw, vz)

    def stop_sync(self) -> str | None:
        with self._cmd_cond:
            if not self._running:
                self._set_last_error("STOP: bridge is already closed")
                return None
            self._command_generation += 1
            self._pending_target = {
                "vx": 0.0,
                "vy": 0.0,
                "vyaw": 0.0,
                "vz": 0.0,
                "stamp_ns": time.perf_counter_ns(),
                "generation": self._command_generation,
            }
            self._has_pending = False
            self._cmd_cond.notify_all()

        ack = self._send_sync("STOP", allow_fault=True)
        if not self._ack_is_success("STOP", ack):
            if ack is None:
                self._mark_fault(self._last_error or "STOP: no successful response")
            else:
                self._mark_fault(f"STOP: child response {ack!r}")
            return ack
        with self._cmd_lock:
            self._last_sent_move = (0.0, 0.0, 0.0)
            self._last_sent_height = 0.0
            self._last_sent_move_ns = time.perf_counter_ns()
            self._last_sent_height_ns = time.perf_counter_ns()
        return ack

    def stop(self) -> str | None:
        return self.stop_sync()

    def get_timing_snapshot(self):
        self._refresh_health()
        with self._cmd_lock:
            elapsed_ns = max(1, time.perf_counter_ns() - self._send_start_ns)
            publish_hz = self._send_count / (elapsed_ns / 1e9)
            return {
                "publish_hz": float(publish_hz),
                "move_stats": _summary_stats(self._move_send_ms),
                "height_stats": _summary_stats(self._height_send_ms),
                "cycle_stats": _summary_stats(self._cycle_ms),
                "queue_delay_stats": _summary_stats(self._queue_delay_ms),
                "last_ack": self._last_ack,
                "last_error": self._last_error,
                "healthy": not bool(getattr(self, "_fault_reason", "")),
                "fault_reason": getattr(self, "_fault_reason", ""),
                "fault_monotonic_ns": int(getattr(self, "_fault_monotonic_ns", 0)),
                "worker_alive": self._worker_thread.is_alive() if self._worker_thread is not None else False,
                "process_alive": self.process is not None and (
                    getattr(self.process, "poll", None) is None or self.process.poll() is None
                ),
                "stderr_tail": list(self._stderr_lines),
                "pending_target": dict(self._pending_target),
                "has_pending": bool(self._has_pending),
            }

    def close(self) -> None:
        if self.process is None:
            return
        self.stop_sync()
        with self._cmd_cond:
            self._running = False
            self._has_pending = False
            self._cmd_cond.notify_all()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
        health_thread = getattr(self, "_health_thread", None)
        if health_thread is not None:
            health_thread.join(timeout=1.0)
        process = self.process
        if process is None:
            return
        ack = self._send_sync("QUIT", allow_fault=True)
        if not self._ack_is_success("QUIT", ack):
            self._set_last_error(f"QUIT: failed acknowledgement {ack!r}; terminating child")
        if process.poll() is None:
            process.terminate()
            deadline = time.monotonic() + PROCESS_EXIT_TIMEOUT_SEC
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(PROCESS_POLL_SEC)
        if process.poll() is None:
            self._set_last_error("QUIT: child ignored terminate; killing")
            process.kill()
            deadline = time.monotonic() + PROCESS_EXIT_TIMEOUT_SEC
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(PROCESS_POLL_SEC)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
        self.process = None


__all__ = ["G1DAgvBridge", "build_g1d_agv_bridge", "BIN_PATH"]
