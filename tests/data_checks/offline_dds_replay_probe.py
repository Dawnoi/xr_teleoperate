#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.input.teleop_input_provider import LeRobotOfflineInputProvider, validate_lerobot_offline_episode  # noqa: E402


def _format_vector(values: Iterable[float], precision: int = 4) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{value:.{precision}f}" for value in arr) + "]"


def _iter_samples(args):
    validate_lerobot_offline_episode(
        args.dataset_root,
        args.episode_index,
        args.arm_source,
    )
    provider = LeRobotOfflineInputProvider(
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        arm_source=args.arm_source,
        speed_scale=0.0 if args.no_timing else args.speed_scale,
        arm_ik=None,
    )
    count = 0
    while True:
        if args.max_frames > 0 and count >= args.max_frames:
            break
        sample = provider.get_sample()
        if sample is None:
            break
        yield sample
        count += 1


def _print_sample(sample, prefix: str = "[OFFLINE]") -> None:
    intent = sample.motion_intent
    if intent.kind == "joint_position":
        print(
            f"{prefix} frame={intent.frame_index} "
            f"timestamp={intent.timestamp:.6f} "
            f"kind={intent.kind} "
            f"arm_q={_format_vector(intent.arm_q)} "
            f"gripper_q={_format_vector(intent.gripper_q) if intent.gripper_q is not None else 'None'}"
        )
        return
    print(
        f"{prefix} frame={intent.frame_index} "
        f"timestamp={intent.timestamp:.6f} "
        f"kind={intent.kind} "
        "arm_q=None gripper_q=None"
    )


def _initialize_robot_runtime(network_interface: str | None, motion: bool, max_arm_joint_speed: float):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from teleop.robot_control.robot_arm import G1_29_ArmController

    ChannelFactoryInitialize(0, networkInterface=network_interface)
    arm_ctrl = G1_29_ArmController(motion_mode=motion, simulation_mode=False)
    arm_ctrl.arm_velocity_limit = float(max_arm_joint_speed)
    return arm_ctrl


def _publish_episode(args, arm_ctrl, samples) -> None:
    sent = 0
    zero_tauff = np.zeros(14, dtype=float)
    for sample in samples:
        intent = sample.motion_intent
        _print_sample(sample, prefix="[DDS_SMOKE]")
        if intent.kind != "joint_position":
            print(f"[DDS_SMOKE][WARN] skip frame={intent.frame_index}: expected joint_position, got {intent.kind}")
            continue
        arm_q = np.asarray(intent.arm_q, dtype=float).reshape(14)
        arm_ctrl.ctrl_dual_arm(arm_q, zero_tauff)
        sent += 1
        time.sleep(max(0.0, 1.0 / max(args.publish_frequency, 1e-6)))

    time.sleep(max(0.05, args.settle_time))
    snapshot = None
    if hasattr(arm_ctrl, "get_timing_snapshot"):
        snapshot = arm_ctrl.get_timing_snapshot()
    print(f"[DDS_SMOKE] queued_frames={sent}")
    print(f"[DDS_SMOKE] timing={snapshot}")


def run(args) -> int:
    samples = list(_iter_samples(args))
    if not samples:
        raise RuntimeError("no samples loaded from offline episode")

    print(
        f"[OFFLINE] dataset_root={args.dataset_root} episode={args.episode_index} "
        f"arm_source={args.arm_source} samples={len(samples)} dds_smoke={args.dds_smoke}"
    )
    for sample in samples:
        _print_sample(sample)

    if not args.dds_smoke:
        print("[OFFLINE] DDS disabled. Add --dds-smoke to initialize Unitree DDS and send these targets.")
        return 0

    if args.arm_source != "action":
        raise ValueError("--dds-smoke currently expects --arm-source action for joint_position targets")
    arm_ctrl = _initialize_robot_runtime(
        network_interface=args.network_interface,
        motion=args.motion,
        max_arm_joint_speed=args.max_arm_joint_speed,
    )
    _publish_episode(args, arm_ctrl, samples)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect offline LeRobot joint targets and optionally smoke-test Unitree DDS arm publishing."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--arm-source", choices=["action", "state", "fk_cmd_pose"], default="action")
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--no-timing", action="store_true", help="Do not sleep according to offline timestamps.")
    parser.add_argument("--dds-smoke", action="store_true", help="Initialize Unitree DDS and send arm targets.")
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--max-arm-joint-speed", type=float, default=1.5)
    parser.add_argument("--publish-frequency", type=float, default=30.0)
    parser.add_argument("--settle-time", type=float, default=0.2)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
