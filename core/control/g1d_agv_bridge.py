from __future__ import annotations

import os
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
    def __init__(self, network_interface: str | None = None, auto_build: bool = True):
        self.network_interface = network_interface or ""
        self.process: subprocess.Popen[str] | None = None
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
        self._stderr_lines = deque(maxlen=50)
        self._stderr_thread = None
        self._worker_thread = None

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
            text=True,
            bufsize=1,
            env=env,
        )
        ready = self.process.stdout.readline().strip() if self.process.stdout else ""
        if ready != "READY":
            stderr_msg = ""
            if self.process.stderr:
                try:
                    stderr_msg = self.process.stderr.readline().strip()
                except Exception:
                    stderr_msg = ""
            raise RuntimeError(f"G1D AGV bridge failed to become ready: stdout={ready!r} stderr={stderr_msg!r}")

        if self.process.stderr is not None:
            self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
            self._stderr_thread.start()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _drain_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        while self._running:
            line = self.process.stderr.readline()
            if not line:
                break
            msg = line.strip()
            if msg:
                with self._cmd_lock:
                    self._stderr_lines.append(msg)
                    self._last_error = msg

    def _send_sync(self, line: str, *, expected_generation: int | None = None) -> str | None:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("G1D AGV bridge process is not running")
        with self._io_lock:
            with self._cmd_lock:
                if expected_generation is not None and int(expected_generation) != self._command_generation:
                    return None
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
            ack = self.process.stdout.readline().strip()
        with self._cmd_lock:
            self._last_ack = ack
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

            try:
                if self._should_send_move(move_tuple):
                    start_ns = time.perf_counter_ns()
                    move_ack = self._send_sync(
                        f"MOVE {move_tuple[0]:.6f} {move_tuple[1]:.6f} {move_tuple[2]:.6f}",
                        expected_generation=target_generation,
                    )
                    if move_ack is None:
                        continue
                    move_ms = (time.perf_counter_ns() - start_ns) / 1e6
                    self._last_sent_move = move_tuple
                    self._last_sent_move_ns = time.perf_counter_ns()

                if self._should_send_height(height_value):
                    start_ns = time.perf_counter_ns()
                    height_ack = self._send_sync(
                        f"HEIGHT {height_value:.6f}",
                        expected_generation=target_generation,
                    )
                    if height_ack is None:
                        continue
                    height_ms = (time.perf_counter_ns() - start_ns) / 1e6
                    self._last_sent_height = height_value
                    self._last_sent_height_ns = time.perf_counter_ns()
            except Exception as exc:
                with self._cmd_lock:
                    self._last_error = str(exc)
                return

            cycle_ms = (time.perf_counter_ns() - cycle_start_ns) / 1e6
            with self._cmd_lock:
                self._move_send_ms.append(float(move_ms))
                self._height_send_ms.append(float(height_ms))
                self._cycle_ms.append(float(cycle_ms))
                self._queue_delay_ms.append(float(queue_delay_ms))
                self._send_count += 1

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
        with self._cmd_cond:
            if not self._running:
                raise RuntimeError("G1D AGV bridge is closed")
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

    def stop_sync(self) -> str:
        with self._cmd_cond:
            if not self._running:
                raise RuntimeError("G1D AGV bridge is closed")
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

        ack = self._send_sync("STOP")
        if ack is None or not ack.startswith("OK STOP "):
            raise RuntimeError(f"G1D AGV bridge STOP failed: ack={ack!r}")
        with self._cmd_lock:
            self._last_sent_move = (0.0, 0.0, 0.0)
            self._last_sent_height = 0.0
            self._last_sent_move_ns = time.perf_counter_ns()
            self._last_sent_height_ns = time.perf_counter_ns()
        return ack

    def stop(self) -> str:
        return self.stop_sync()

    def get_timing_snapshot(self):
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
        ack = self._send_sync("QUIT")
        if ack != "BYE":
            raise RuntimeError(f"G1D AGV bridge QUIT failed: ack={ack!r}")
        self.process.wait(timeout=2.0)
        self.process = None


__all__ = ["G1DAgvBridge", "build_g1d_agv_bridge", "BIN_PATH"]
