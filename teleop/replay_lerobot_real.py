#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


CHUNK_NAME = "chunk-000"
ACTION_SIZE = 16
ARM_SIZE = 14


def warn(message: str) -> None:
    print(f"[REPLAY][WARN] {message}")


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

    @property
    def left_arm(self) -> np.ndarray:
        return self.left_arm7

    @property
    def right_arm(self) -> np.ndarray:
        return self.right_arm7


@dataclass
class ReplayFrame:
    frame_index: int
    timestamp: float
    action: ActionSplit
    recorded_tauff: Optional[np.ndarray]


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


def load_recorded_tauff_by_frame(dataset_root: Path, episode_index: int) -> dict[int, np.ndarray]:
    path = control_sidecar_path(dataset_root, episode_index)
    if not path.exists():
        warn(f"control sidecar missing at {path}; falling back to runtime tauff.")
        return {}
    out: dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                frame_index = int(row["frame_index"])
                out[frame_index] = finite_vector(row.get("arm_tauff"), ARM_SIZE, f"{path}:{line_no}:arm_tauff")
            except Exception as exc:
                warn(f"invalid recorded tauff in {path}:{line_no}: {exc}; falling back for that row.")
                continue
    return out


def load_replay_frames(dataset_root: str | Path, episode_index: int, use_recorded_tauff: bool = False) -> List[ReplayFrame]:
    import pyarrow.parquet as pq

    root = Path(dataset_root)
    parquet_path = episode_parquet_path(root, episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(f"episode parquet not found: {parquet_path}")

    table = pq.read_table(parquet_path, columns=["timestamp", "frame_index", "action"])
    timestamps = table["timestamp"].to_pylist()
    frame_indices = table["frame_index"].to_pylist()
    actions = table["action"].to_pylist()
    sidecar_exists = control_sidecar_path(root, episode_index).exists()
    recorded_tauff = load_recorded_tauff_by_frame(root, episode_index) if use_recorded_tauff else {}

    frames: List[ReplayFrame] = []
    missing_recorded_tauff = 0
    for timestamp, frame_index, action in zip(timestamps, frame_indices, actions):
        idx = int(frame_index)
        frame_recorded_tauff = recorded_tauff.get(idx)
        if use_recorded_tauff and frame_recorded_tauff is None:
            missing_recorded_tauff += 1
        frames.append(
            ReplayFrame(
                frame_index=idx,
                timestamp=float(timestamp),
                action=split_action(action),
                recorded_tauff=frame_recorded_tauff,
            )
        )
    if use_recorded_tauff and sidecar_exists and missing_recorded_tauff:
        warn(
            f"recorded tauff missing for {missing_recorded_tauff} frame(s); "
            "falling back to runtime tauff for those frames."
        )
    return frames


def compute_gravity_tauff(arm_ik, arm_q: np.ndarray) -> np.ndarray:
    import pinocchio as pin

    model = arm_ik.reduced_robot.model
    data = arm_ik.reduced_robot.data
    q = finite_vector(arm_q, ARM_SIZE, "arm_q")
    dq = np.zeros(model.nv, dtype=float)
    ddq = np.zeros(model.nv, dtype=float)
    return np.asarray(pin.rnea(model, data, q, dq, ddq), dtype=float).reshape(-1)


class Dex1DirectPublisher:
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
        from teleop.robot_control.robot_hand_unitree import (
            kTopicGripperLeftCommand,
            kTopicGripperRightCommand,
        )

        self.left_msg = MotorCmds_()
        self.left_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.right_msg = MotorCmds_()
        self.right_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        for msg in (self.left_msg, self.right_msg):
            msg.cmds[0].dq = 0.0
            msg.cmds[0].tau = 0.0
            msg.cmds[0].kp = 5.0
            msg.cmds[0].kd = 0.05
        self.left_pub = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.left_pub.Init()
        self.right_pub = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.right_pub.Init()

    def publish(self, gripper_q: Iterable[float]) -> None:
        q = finite_vector(gripper_q, 2, "gripper_q")
        self.left_msg.cmds[0].q = float(q[0])
        self.right_msg.cmds[0].q = float(q[1])
        self.left_pub.Write(self.left_msg)
        self.right_pub.Write(self.right_msg)


def initialize_robot_runtime(network_interface: Optional[str], motion: bool):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
    from teleop.utils.motion_switcher import MotionSwitcher

    ChannelFactoryInitialize(0, networkInterface=network_interface)
    if not motion:
        motion_switcher = MotionSwitcher()
        motion_switcher.Enter_Debug_Mode()
    arm_ik = G1_29_ArmIK()
    arm_ctrl = G1_29_ArmController(motion_mode=motion, simulation_mode=False)
    return arm_ik, arm_ctrl


def sleep_until_frame(start_wall: float, first_timestamp: float, frame_timestamp: float, speed_scale: float) -> None:
    if speed_scale <= 0:
        return
    target_elapsed = (frame_timestamp - first_timestamp) / speed_scale
    sleep_s = start_wall + target_elapsed - time.time()
    if sleep_s > 0:
        time.sleep(sleep_s)


def run_replay(args) -> int:
    frames = load_replay_frames(args.dataset_root, args.episode_index, args.use_recorded_tauff)
    if not frames:
        raise RuntimeError("episode has no frames")

    sidecar = control_sidecar_path(Path(args.dataset_root), args.episode_index)
    recorded_count = sum(1 for frame in frames if frame.recorded_tauff is not None)
    print(
        f"[REPLAY] episode={args.episode_index} frames={len(frames)} "
        f"timestamp={frames[0].timestamp:.6f}->{frames[-1].timestamp:.6f} "
        f"action_shape=({ACTION_SIZE},) sidecar={'present' if sidecar.exists() else 'missing'} "
        f"recorded_tauff_frames={recorded_count}"
    )
    if args.dry_run:
        return 0

    arm_ik, arm_ctrl = initialize_robot_runtime(args.network_interface, args.motion)
    gripper_pub = None if args.no_gripper else Dex1DirectPublisher()
    arm_ctrl.arm_velocity_limit = float(args.max_arm_joint_speed)
    print("[REPLAY] moving arms home before replay...")
    arm_ctrl.ctrl_dual_arm_go_home()

    first_timestamp = frames[0].timestamp
    start_wall = time.time()
    try:
        for frame in frames:
            sleep_until_frame(start_wall, first_timestamp, frame.timestamp, args.speed_scale)
            arm_q = frame.action.arm_q
            if args.use_recorded_tauff and frame.recorded_tauff is not None:
                tauff = frame.recorded_tauff
            else:
                tauff = compute_gravity_tauff(arm_ik, arm_q)
            tauff = finite_vector(tauff, ARM_SIZE, "tauff")
            arm_ctrl.ctrl_dual_arm(arm_q, tauff)
            if gripper_pub is not None:
                gripper_pub.publish(frame.action.gripper_q)
    except KeyboardInterrupt:
        print("[REPLAY] interrupted")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a LeRobot episode on the real G1_29 + Dex1 robot.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--use-recorded-tauff", action="store_true")
    parser.add_argument("--max-arm-joint-speed", type=float, default=0.5)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--motion", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
