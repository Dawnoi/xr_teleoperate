#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


CHUNK_NAME = "chunk-000"
ACTION_SIZE = 16

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.utils.teleop_input_provider import validate_lerobot_offline_episode  # noqa: E402


@dataclass
class ActionSplit:
    left_arm7: np.ndarray
    left_gripper: float
    right_arm7: np.ndarray
    right_gripper: float

    @property
    def arm_q(self) -> np.ndarray:
        return np.concatenate([self.left_arm7, self.right_arm7])

    @property
    def gripper_q(self) -> np.ndarray:
        return np.asarray([self.left_gripper, self.right_gripper], dtype=float)


def finite_vector(values, expected_len: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != expected_len:
        raise ValueError(f"{label} expected length {expected_len}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or Inf")
    return arr


def split_action(action_values) -> ActionSplit:
    action = finite_vector(action_values, ACTION_SIZE, "action")
    return ActionSplit(
        left_arm7=action[0:7].copy(),
        left_gripper=float(action[7]),
        right_arm7=action[8:15].copy(),
        right_gripper=float(action[15]),
    )


def episode_parquet_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / CHUNK_NAME / f"episode_{episode_index:06d}.parquet"


def control_sidecar_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "extras" / "control" / CHUNK_NAME / f"episode_{episode_index:06d}.jsonl"


def load_dry_run_summary(dataset_root: str | Path, episode_index: int, arm_source: str) -> dict:
    import pyarrow.parquet as pq

    root = Path(dataset_root)
    parquet_path = episode_parquet_path(root, episode_index)
    validation = validate_lerobot_offline_episode(root, episode_index, arm_source)

    action_shape = None
    schema_names = set(pq.read_schema(parquet_path).names)
    if "action" in schema_names:
        action_table = pq.read_table(parquet_path, columns=["action"])
        actions = action_table["action"].to_pylist()
        if actions:
            action_shape = tuple(finite_vector(actions[0], ACTION_SIZE, "action").shape)

    summary = {
        "frame_count": int(validation["frame_count"]),
        "timestamp_range": validation["timestamp_range"],
        "action_shape": action_shape,
        "sidecar": "present" if control_sidecar_path(root, episode_index).exists() else "missing",
        "arm_source": arm_source,
    }
    return summary


def print_dry_run_summary(summary: dict) -> None:
    start_ts, end_ts = summary["timestamp_range"]
    print(
        f"[REPLAY] frames={summary['frame_count']} "
        f"timestamp={start_ts:.6f}->{end_ts:.6f} "
        f"action_shape={summary['action_shape']} "
        f"sidecar={summary['sidecar']} "
        f"arm_source={summary['arm_source']}"
    )


def teleop_entry_path() -> Path:
    return (Path(__file__).resolve().parent / "teleop_hand_and_arm.py").resolve()


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
    summary = load_dry_run_summary(args.dataset_root, args.episode_index, args.arm_source)
    print_dry_run_summary(summary)
    if args.use_recorded_tauff:
        print("[REPLAY] use_recorded_tauff requested; ignored in wrapper, main loop uses runtime tauff/default handling.")
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
    parser = argparse.ArgumentParser(description="Replay a LeRobot episode through teleop_hand_and_arm.py.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--max-arm-joint-speed", type=float, default=0.5)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--use-recorded-tauff", action="store_true")
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
