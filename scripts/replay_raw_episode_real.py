#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.utils.raw_episode_replay import load_raw_dry_run_summary


def load_dry_run_summary(dataset_root: str | Path, episode_index: int, arm_source: str) -> dict:
    return load_raw_dry_run_summary(dataset_root, episode_index, arm_source)


def print_dry_run_summary(summary: dict) -> None:
    print(
        f"[RAW_REPLAY] frames={summary['frame_count']} "
        f"sample_monotonic_ns={summary['sample_monotonic_ns_range'][0]}->{summary['sample_monotonic_ns_range'][1]} "
        f"motion_repr={summary['motion_repr']} "
        f"arm_source={summary['arm_source']} "
        f"episode_dir={summary['episode_dir']}"
    )


def teleop_entry_path() -> Path:
    return (REPO_ROOT / "teleop" / "teleop_hand_and_arm.py").resolve()


def build_subprocess_command(
    *,
    dataset_root: str,
    episode_index: int,
    network_interface: Optional[str],
    max_arm_joint_speed: float,
    speed_scale: float,
    motion: bool,
    arm_source: str,
    end_action: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(teleop_entry_path()),
        "--input-mode",
        "controller",
        "--arm",
        "G1_29",
        "--ee",
        "dex1",
        "--input-provider",
        "lerobot_offline",
        "--offline-replay-dataset-root",
        str(dataset_root),
        "--offline-replay-episode-index",
        str(episode_index),
        "--offline-replay-arm-source",
        str(arm_source),
        "--offline-replay-speed-scale",
        str(speed_scale),
        "--offline-replay-end-action",
        str(end_action),
        "--max-arm-joint-speed",
        str(max_arm_joint_speed),
        "--controller-deadman",
        "grip",
        "--head-reference-mode",
        "fixed_per_grip",
        "--controller-mapping-mode",
        "anchored_safe",
        "--controller-orientation-mode",
        "relative",
        "--base-controller",
        "none",
        "--headless",
        "--auto-start",
    ]
    if network_interface:
        cmd.extend(["--network-interface", str(network_interface)])
    if motion:
        cmd.append("--motion")
    return cmd


def run_replay(args) -> int:
    summary = load_raw_dry_run_summary(args.dataset_root, args.episode_index, args.arm_source)
    print_dry_run_summary(summary)
    if args.dry_run:
        return 0

    cmd = build_subprocess_command(
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        network_interface=args.network_interface,
        max_arm_joint_speed=args.max_arm_joint_speed,
        speed_scale=args.speed_scale,
        motion=args.motion,
        arm_source=args.arm_source,
        end_action=args.end_action,
    )
    if args.no_gripper:
        cmd.append("--no-gripper")
    return int(subprocess.call(cmd))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a raw episode_xxxx/data.json recording through teleop_hand_and_arm.py.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--max-arm-joint-speed", type=float, default=1.5)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--motion", action="store_true")
    parser.add_argument(
        "--arm-source",
        choices=["action", "state", "fk_cmd_pose"],
        default="action",
    )
    parser.add_argument(
        "--end-action",
        choices=["home", "hold"],
        default="home",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
