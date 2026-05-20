from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_SRC = REPO_ROOT / "teleop" / "cpp" / "g1d_agv_bridge.cpp"
BIN_DIR = REPO_ROOT / "teleop" / "bin"
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


class G1DAgvBridge:
    def __init__(self, network_interface: str | None = None, auto_build: bool = True):
        self.network_interface = network_interface or ""
        self.process: subprocess.Popen[str] | None = None
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

    def _send(self, line: str) -> str:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("G1D AGV bridge process is not running")
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        return self.process.stdout.readline().strip()

    def move(self, vx: float, vy: float, vyaw: float) -> str:
        return self._send(f"MOVE {vx:.6f} {vy:.6f} {vyaw:.6f}")

    def height_adjust(self, vz: float) -> str:
        return self._send(f"HEIGHT {vz:.6f}")

    def stop(self) -> str:
        return self._send("STOP")

    def close(self) -> None:
        if self.process is None:
            return
        try:
            self._send("QUIT")
        except Exception:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=2.0)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self.process = None


__all__ = ["G1DAgvBridge", "build_g1d_agv_bridge", "BIN_PATH"]
