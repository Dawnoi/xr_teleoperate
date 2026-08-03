#!/usr/bin/env python3
"""Monitor G1-D odometry and column-height DDS arrival timing."""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass

def _field(message, name: str):
    value = getattr(message, name)
    return value() if callable(value) else value


def _stamp_ns(message) -> int:
    header = _field(message, "header")
    stamp = _field(header, "stamp")
    return int(_field(stamp, "sec")) * 1_000_000_000 + int(_field(stamp, "nanosec"))


@dataclass
class StreamStats:
    name: str
    callback_count: int = 0
    last_callback_ns: int | None = None
    last_header_ns: int | None = None
    last_gap_ms: float = 0.0
    max_gap_ms: float = 0.0
    max_header_gap_ms: float = 0.0
    last_warn_ns: int = 0

    def update(self, header_ns: int | None) -> None:
        now_ns = time.monotonic_ns()
        if self.last_callback_ns is not None:
            self.last_gap_ms = (now_ns - self.last_callback_ns) / 1e6
            self.max_gap_ms = max(self.max_gap_ms, self.last_gap_ms)
        if header_ns is not None and self.last_header_ns is not None:
            header_gap_ms = (header_ns - self.last_header_ns) / 1e6
            self.max_header_gap_ms = max(self.max_header_gap_ms, header_gap_ms)
        self.callback_count += 1
        self.last_callback_ns = now_ns
        self.last_header_ns = header_ns

    def age_ms(self) -> float | None:
        if self.last_callback_ns is None:
            return None
        return (time.monotonic_ns() - self.last_callback_ns) / 1e6


def _read_rx_dropped(interface: str) -> int:
    path = f"/sys/class/net/{interface}/statistics/rx_dropped"
    with open(path, "r", encoding="ascii") as stream:
        return int(stream.read().strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor rt/agv/odom and rt/hispeed_state timing.")
    parser.add_argument("--network-interface", default=os.environ.get("NETWORK_INTERFACE", ""))
    parser.add_argument("--duration-sec", type=float, default=300.0)
    parser.add_argument("--warn-gap-ms", type=float, default=200.0)
    parser.add_argument("--report-period-sec", type=float, default=0.5)
    args = parser.parse_args()
    if not args.network_interface:
        raise ValueError("--network-interface is required or NETWORK_INTERFACE must be set")
    if args.duration_sec <= 0.0 or args.warn_gap_ms <= 0.0 or args.report_period_sec <= 0.0:
        raise ValueError("duration, warn gap, and report period must be positive")
    return args


def main() -> None:
    args = _parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_
    from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)

    odom = StreamStats("odom")
    height = StreamStats("height")

    def on_odom(message) -> None:
        odom.update(_stamp_ns(message))
        if odom.last_gap_ms > args.warn_gap_ms:
            now_ns = time.monotonic_ns()
            if now_ns - odom.last_warn_ns >= 1_000_000_000:
                print(
                    f"\n[ODOM_GAP] callback_gap_ms={odom.last_gap_ms:.1f} "
                    f"header_gap_ms={odom.max_header_gap_ms:.1f}",
                    flush=True,
                )
                odom.last_warn_ns = now_ns

    def on_height(message) -> None:
        height.update(None)
        if height.last_gap_ms > args.warn_gap_ms:
            now_ns = time.monotonic_ns()
            if now_ns - height.last_warn_ns >= 1_000_000_000:
                print(f"\n[HEIGHT_GAP] callback_gap_ms={height.last_gap_ms:.1f}", flush=True)
                height.last_warn_ns = now_ns

    odom_subscriber = ChannelSubscriber("rt/agv/odom", Odometry_)
    height_subscriber = ChannelSubscriber("rt/hispeed_state", Point32_)
    odom_subscriber.Init(on_odom, 10)
    height_subscriber.Init(on_height, 10)

    initial_rx_dropped = _read_rx_dropped(args.network_interface)
    start = time.monotonic()
    next_report = start
    print(
        f"monitoring for {args.duration_sec:.1f}s; "
        f"warn_gap_ms={args.warn_gap_ms:.1f}; "
        f"rx_dropped_start={initial_rx_dropped}",
        flush=True,
    )

    while time.monotonic() - start < args.duration_sec:
        now = time.monotonic()
        if now >= next_report:
            rx_dropped = _read_rx_dropped(args.network_interface)
            print(
                f"\r[STATUS] odom_count={odom.callback_count} "
                f"odom_age_ms={odom.age_ms() if odom.age_ms() is not None else math.nan:6.1f} "
                f"odom_last_gap_ms={odom.last_gap_ms:6.1f} "
                f"odom_max_gap_ms={odom.max_gap_ms:6.1f} "
                f"height_count={height.callback_count} "
                f"height_age_ms={height.age_ms() if height.age_ms() is not None else math.nan:6.1f} "
                f"height_max_gap_ms={height.max_gap_ms:6.1f} "
                f"rx_dropped_delta={rx_dropped - initial_rx_dropped}",
                end="",
                flush=True,
            )
            next_report = now + args.report_period_sec
        time.sleep(0.02)

    print()
    print(
        f"summary: odom_count={odom.callback_count} "
        f"odom_max_callback_gap_ms={odom.max_gap_ms:.3f} "
        f"odom_max_header_gap_ms={odom.max_header_gap_ms:.3f} "
        f"height_count={height.callback_count} "
        f"height_max_callback_gap_ms={height.max_gap_ms:.3f} "
        f"rx_dropped_delta={_read_rx_dropped(args.network_interface) - initial_rx_dropped}",
        flush=True,
    )
    odom_subscriber.Close()
    height_subscriber.Close()


if __name__ == "__main__":
    main()
