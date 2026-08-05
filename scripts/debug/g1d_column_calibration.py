#!/usr/bin/env python3
"""Interactive G1-D column calibration with live DDS height feedback.

Run this only when no teleop, replay, or other AGV controller is running.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import signal
import statistics
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _field(message, name: str):
    value = getattr(message, name)
    return value() if callable(value) else value


def _finite_float(value, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{field_name} must be finite, got {value!r}")
    return result


class HeightMonitor:
    def __init__(self, topic: str, *, history_size: int = 600):
        self.topic = str(topic)
        self._lock = threading.Lock()
        self._samples: deque[tuple[int, float, float, float]] = deque(maxlen=int(history_size))

    def callback(self, message) -> None:
        sample = (
            time.monotonic_ns(),
            _finite_float(_field(message, "x"), f"{self.topic}.x"),
            _finite_float(_field(message, "y"), f"{self.topic}.y"),
            _finite_float(_field(message, "z"), f"{self.topic}.z"),
        )
        with self._lock:
            self._samples.append(sample)

    def latest(self) -> tuple[int, float, float, float] | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def stable_y(self, *, window_sec: float, maximum_span_m: float) -> float:
        now_ns = time.monotonic_ns()
        earliest_ns = now_ns - int(float(window_sec) * 1e9)
        with self._lock:
            values = [sample[2] for sample in self._samples if sample[0] >= earliest_ns]
        if len(values) < 2:
            raise RuntimeError(
                f"not enough height samples in the last {window_sec:.2f}s; wait for DDS feedback"
            )
        span = max(values) - min(values)
        if span > float(maximum_span_m):
            raise RuntimeError(
                f"column is not stable: raw_height_y span={span:.6f}m exceeds "
                f"limit={maximum_span_m:.6f}m; release movement and wait before marking"
            )
        return float(statistics.median(values))


def _active_controller_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    prohibited_tokens = (
        "teleop/real/teleop_hand_and_arm.py",
        "g1d_agv_bridge",
        "g1d_wbc_marker.py",
        "whole_body_controller",
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if any(token in line for token in prohibited_tokens)
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive G1-D column calibration using rt/hispeed_state.y."
    )
    parser.add_argument(
        "--network-interface",
        default=os.environ.get("NETWORK_INTERFACE", ""),
        help="Unitree DDS interface. Defaults to NETWORK_INTERFACE.",
    )
    parser.add_argument("--height-topic", default="rt/hispeed_state")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.15,
        help="Absolute normalized HeightAdjust command while W/S is held. Must be in (0, 0.25].",
    )
    parser.add_argument(
        "--deadman-timeout-ms",
        type=float,
        default=250.0,
        help="Stop movement this long after the final W/S keyboard event.",
    )
    parser.add_argument(
        "--height-timeout-ms",
        type=float,
        default=500.0,
        help="Abort and stop if rt/hispeed_state is older than this value.",
    )
    parser.add_argument(
        "--endpoint-window-sec",
        type=float,
        default=0.75,
        help="Recent stable feedback window used when marking lower/upper endpoints.",
    )
    parser.add_argument(
        "--endpoint-max-span-m",
        type=float,
        default=0.0008,
        help="Maximum raw y span accepted while marking an endpoint.",
    )
    parser.add_argument(
        "--endpoint-safety-margin-m",
        type=float,
        default=0.001,
        help="Outward raw-height margin added to both endpoint values in suggested teleop arguments.",
    )
    args = parser.parse_args()
    if not args.network_interface:
        raise ValueError("--network-interface is required (or set NETWORK_INTERFACE)")
    if not 0.0 < float(args.speed) <= 0.25:
        raise ValueError("--speed must be in (0, 0.25]")
    if float(args.deadman_timeout_ms) <= 0.0:
        raise ValueError("--deadman-timeout-ms must be positive")
    if float(args.height_timeout_ms) <= 0.0:
        raise ValueError("--height-timeout-ms must be positive")
    if float(args.endpoint_window_sec) <= 0.0:
        raise ValueError("--endpoint-window-sec must be positive")
    if float(args.endpoint_max_span_m) <= 0.0:
        raise ValueError("--endpoint-max-span-m must be positive")
    if float(args.endpoint_safety_margin_m) < 0.0:
        raise ValueError("--endpoint-safety-margin-m must be non-negative")
    return args


def _signal_exit(signum, _frame) -> None:
    raise SystemExit(f"received signal {signum}; stopping column calibration")


def _print_endpoint_summary(lower: float | None, upper: float | None, *, safety_margin_m: float) -> None:
    print()
    if lower is None:
        print("lower endpoint: not marked")
    else:
        print(f"lower endpoint raw_height_y: {lower:+.9f}")
    if upper is None:
        print("upper endpoint: not marked")
    else:
        print(f"upper endpoint raw_height_y: {upper:+.9f}")
    if lower is None or upper is None:
        return
    if upper <= lower:
        raise RuntimeError(
            f"invalid endpoints: upper={upper:+.9f} must be greater than lower={lower:+.9f}"
        )
    print(f"raw range: {upper - lower:.9f} m")
    minimum = lower - float(safety_margin_m)
    maximum = upper + float(safety_margin_m)
    print(f"Suggested teleop arguments (outward safety margin {safety_margin_m * 1e3:.3f}mm):")
    print(f"  --mobile-height-raw-minimum {minimum:.9f} \\")
    print(f"  --mobile-height-raw-maximum {maximum:.9f}")


def main() -> None:
    args = _parse_args()
    running = _active_controller_processes()
    if running:
        raise RuntimeError(
            "refusing to start while another mobile controller is running:\n  "
            + "\n  ".join(running)
        )
    if not sys.stdin.isatty():
        raise RuntimeError("column calibration requires an interactive TTY")

    from core.control.g1d_agv_bridge import G1DAgvBridge
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_

    signal.signal(signal.SIGINT, _signal_exit)
    signal.signal(signal.SIGTERM, _signal_exit)
    signal.signal(signal.SIGHUP, _signal_exit)
    signal.signal(signal.SIGTSTP, _signal_exit)

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)
    monitor = HeightMonitor(args.height_topic)
    subscriber = ChannelSubscriber(args.height_topic, Point32_)
    subscriber.Init(monitor.callback, 10)
    bridge = G1DAgvBridge(network_interface=args.network_interface, auto_build=True)

    stdin_fd = sys.stdin.fileno()
    terminal_settings = termios.tcgetattr(stdin_fd)
    lower_endpoint: float | None = None
    upper_endpoint: float | None = None
    last_motion_key_ns = 0
    active_command = 0.0
    deadman_timeout_ns = int(float(args.deadman_timeout_ms) * 1e6)
    height_timeout_ns = int(float(args.height_timeout_ms) * 1e6)

    print("G1-D column calibration")
    print("Stop all teleop/replay/WBC before use. Keep clear of the moving column.")
    print("Hold W to raise, S to lower. Space stops. L marks lower, U marks upper, P prints, Q exits.")

    try:
        tty.setcbreak(stdin_fd)
        while True:
            now_ns = time.monotonic_ns()
            latest = monitor.latest()
            if latest is None:
                status = "waiting for rt/hispeed_state"
            else:
                age_ms = (now_ns - latest[0]) / 1e6
                if age_ms > float(args.height_timeout_ms):
                    raise RuntimeError(
                        f"rt/hispeed_state is stale: age_ms={age_ms:.1f} "
                        f"timeout_ms={args.height_timeout_ms:.1f}"
                    )
                status = (
                    f"raw x={latest[1]:+.6f} y={latest[2]:+.6f} z={latest[3]:+.6f} "
                    f"age={age_ms:5.1f}ms"
                )

            readable, _, _ = select.select([sys.stdin], [], [], 0.02)
            if readable:
                key = sys.stdin.read(1).lower()
                if key == "w":
                    active_command = float(args.speed)
                    last_motion_key_ns = now_ns
                elif key == "s":
                    active_command = -float(args.speed)
                    last_motion_key_ns = now_ns
                elif key == " ":
                    active_command = 0.0
                    bridge.set_target(0.0, 0.0, 0.0, 0.0)
                elif key == "l":
                    if active_command != 0.0:
                        raise RuntimeError("stop the column before marking the lower endpoint")
                    lower_endpoint = monitor.stable_y(
                        window_sec=args.endpoint_window_sec,
                        maximum_span_m=args.endpoint_max_span_m,
                    )
                    print(f"\nmarked lower raw_height_y={lower_endpoint:+.9f}")
                elif key == "u":
                    if active_command != 0.0:
                        raise RuntimeError("stop the column before marking the upper endpoint")
                    upper_endpoint = monitor.stable_y(
                        window_sec=args.endpoint_window_sec,
                        maximum_span_m=args.endpoint_max_span_m,
                    )
                    print(f"\nmarked upper raw_height_y={upper_endpoint:+.9f}")
                elif key == "p":
                    _print_endpoint_summary(
                        lower_endpoint,
                        upper_endpoint,
                        safety_margin_m=float(args.endpoint_safety_margin_m),
                    )
                elif key == "q":
                    break
                elif key:
                    print(f"\nunknown key {key!r}; use W/S/Space/L/U/P/Q")

            if active_command != 0.0 and now_ns - last_motion_key_ns > deadman_timeout_ns:
                active_command = 0.0
                bridge.set_target(0.0, 0.0, 0.0, 0.0)

            if active_command != 0.0:
                bridge.set_target(0.0, 0.0, 0.0, active_command)

            command_name = "raise" if active_command > 0.0 else "lower" if active_command < 0.0 else "stopped"
            print(f"\r{status}  command={command_name:7s}", end="", flush=True)

            if latest is not None and now_ns - latest[0] > height_timeout_ns:
                raise RuntimeError("height feedback timeout")

    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, terminal_settings)
        print("\nStopping G1-D column calibration.")
        bridge.stop_sync()
        bridge.close()
        subscriber.Close()

    _print_endpoint_summary(
        lower_endpoint,
        upper_endpoint,
        safety_margin_m=float(args.endpoint_safety_margin_m),
    )


if __name__ == "__main__":
    main()
