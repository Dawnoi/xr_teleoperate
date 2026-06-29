#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_G1_XML = REPO_ROOT / "assets/g1/g1_body29_hand14.xml"
DEFAULT_DEX1_XML = REPO_ROOT / "assets/.generated/g1_29dof_mode_15_with_dex1_1_scene.xml"
DEFAULT_G1D_XML = REPO_ROOT / "assets/g1_d/g1_d_scene.xml"

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

DEX1_JOINT_NAMES = {
    "left": ["left_dex1_finger_joint_1", "left_dex1_finger_joint_2"],
    "right": ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"],
}

DEX1_OPEN_Q = 0.0245
DEX1_CLOSED_Q = 0.0
DEX1_REAL_MAX_Q = 5.40


def bootstrap_extra_paths() -> None:
    extras = [
        "/home/dx/miniconda3/envs/tv/lib/python3.10/site-packages",
        "/home/dx/miniconda3/envs/tv/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages",
        "/home/dx/miniconda3/envs/gmr/lib/python3.10/site-packages",
    ]
    for path in reversed(extras):
        if os.path.isdir(path):
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)


bootstrap_extra_paths()

import mujoco as mj
import mujoco.viewer as mjv
import glfw

from teleop.utils.g1d_mujoco_builder import prepare_g1d_mobile_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a raw episode_xxxx/data.json action trajectory in a MuJoCo viewer.",
    )
    parser.add_argument("--dataset", default="utils/data/multi_cam_record", help="Path to the raw multi_cam_record directory.")
    parser.add_argument("--episode", type=int, default=None, help="Single episode index, e.g. 200 for episode_0200.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Explicit episode indices to replay in order.")
    parser.add_argument("--episode-range", type=int, nargs=2, metavar=("START", "END"), help="Inclusive episode index range to replay.")
    parser.add_argument("--source", choices=["actions", "states"], default="actions", help="Replay qpos from this data.json field.")
    parser.add_argument("--frequency", type=float, default=30.0, help="Replay frequency when --realtime is not using recorded timestamps.")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier. Must be positive.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to replay.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to replay. 0 means all remaining frames.")
    parser.add_argument("--gap-s", type=float, default=0.5, help="Seconds to wait between episodes when not paused.")
    parser.add_argument("--pause-between-episodes", action="store_true", help="Wait for Enter before each episode after the first.")
    parser.add_argument("--headless", action="store_true", help="Run mj_forward without opening a viewer.")
    parser.add_argument("--realtime", action="store_true", help="Use timestamps.sample_monotonic_ns spacing instead of fixed --frequency.")
    parser.add_argument("--camera-distance", type=float, default=3.0, help="Viewer camera distance.")
    parser.add_argument("--camera-azimuth", type=float, default=-135.0, help="Viewer camera azimuth in degrees.")
    parser.add_argument("--camera-elevation", type=float, default=-18.0, help="Viewer camera elevation in degrees.")
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.95],
        metavar=("X", "Y", "Z"),
        help="Viewer camera lookat point.",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Disable viewer text overlay.")
    parser.add_argument("--xml", type=str, default=None, help="Optional MuJoCo XML path override.")
    parser.add_argument("--ee", choices=["none", "dex1"], default="dex1", help="Whether to animate Dex1 gripper joints.")
    parser.add_argument(
        "--viewer-robot",
        choices=["g1", "g1_d", "g1_d_mobile"],
        default="g1_d_mobile",
        help='MuJoCo model selection. "g1_d_mobile" uses the generated local mobile G1D scene.',
    )
    return parser.parse_args()


def resolve_episode_indices(args: argparse.Namespace) -> list[int]:
    indices: list[int] = []
    if args.episode is not None:
        indices.append(int(args.episode))
    if args.episodes:
        indices.extend(int(value) for value in args.episodes)
    if args.episode_range is not None:
        start, end = [int(value) for value in args.episode_range]
        if end < start:
            raise ValueError("--episode-range END must be >= START")
        indices.extend(range(start, end + 1))
    if not indices:
        raise ValueError("one of --episode, --episodes, or --episode-range is required")

    seen = set()
    ordered = []
    for index in indices:
        if index < 0:
            raise ValueError(f"episode index must be non-negative, got {index}")
        if index not in seen:
            ordered.append(index)
            seen.add(index)
    return ordered


def resolve_xml_path(args: argparse.Namespace) -> Path:
    if args.xml:
        return Path(args.xml).expanduser().resolve()
    if args.viewer_robot == "g1_d_mobile":
        return prepare_g1d_mobile_scene(use_dex1=(args.ee == "dex1"))
    if args.viewer_robot == "g1_d" and DEFAULT_G1D_XML.exists():
        return DEFAULT_G1D_XML
    if args.ee == "dex1" and DEFAULT_DEX1_XML.exists():
        return DEFAULT_DEX1_XML
    return DEFAULT_G1_XML.resolve()


def episode_dir_path(dataset: str | Path, episode_index: int) -> Path:
    root = Path(dataset).expanduser()
    if root.name.startswith("episode_") and (root / "data.json").is_file():
        return root
    return root / f"episode_{int(episode_index):04d}"


def load_items(dataset: str | Path, episode_index: int) -> list[dict[str, Any]]:
    episode_dir = episode_dir_path(dataset, episode_index)
    data_path = episode_dir / "data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"data.json not found: {data_path}")
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"data.json must contain top-level data list: {data_path}")
    items = payload["data"]
    if not items:
        raise ValueError(f"episode has no frames: {data_path}")
    for frame_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"frame {frame_index} must be an object")
    return items


def finite_qpos(value: Any, expected_len: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != expected_len:
        actual_len = len(value) if isinstance(value, list) else "not-list"
        raise ValueError(f"{label} must be length {expected_len}, got {actual_len}")
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains non-finite values")
    return arr


def qpos_from_item(item: dict[str, Any], frame_index: int, source: str, group: str, expected_len: int) -> np.ndarray:
    source_value = item.get(source)
    if not isinstance(source_value, dict):
        raise KeyError(f"frame {frame_index} missing {source}")
    group_value = source_value.get(group)
    if not isinstance(group_value, dict) or "qpos" not in group_value:
        raise KeyError(f"frame {frame_index} missing {source}.{group}.qpos")
    return finite_qpos(group_value["qpos"], expected_len, f"frame {frame_index} {source}.{group}.qpos")


def sample_ns(item: dict[str, Any], frame_index: int) -> int:
    timestamps = item.get("timestamps")
    if not isinstance(timestamps, dict) or "sample_monotonic_ns" not in timestamps:
        raise KeyError(f"frame {frame_index} missing timestamps.sample_monotonic_ns")
    value = int(timestamps["sample_monotonic_ns"])
    if value < 0:
        raise ValueError(f"frame {frame_index} sample_monotonic_ns must be non-negative")
    return value


def joint_qpos_indices(model: mj.MjModel, joint_names: list[str]) -> list[int]:
    qpos_indices = []
    for name in joint_names:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in model: {name}")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
    return qpos_indices


def joint_qpos_indices_if_present(model: mj.MjModel, joint_names: list[str]) -> list[int] | None:
    qpos_indices = []
    for name in joint_names:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            return None
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
    return qpos_indices


def gripper_state_to_mujoco_q(gripper_q: float) -> float:
    open_ratio = float(np.clip(float(gripper_q) / DEX1_REAL_MAX_Q, 0.0, 1.0))
    return DEX1_CLOSED_Q + (DEX1_OPEN_Q - DEX1_CLOSED_Q) * open_ratio


def set_frame_qpos(
    data: mj.MjData,
    base_qpos: np.ndarray,
    arm_qpos_indices: list[int],
    dex1_qpos_indices: dict[str, list[int] | None] | None,
    item: dict[str, Any],
    frame_index: int,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_arm = qpos_from_item(item, frame_index, source, "left_arm", 7)
    right_arm = qpos_from_item(item, frame_index, source, "right_arm", 7)
    arm_q = np.concatenate([left_arm, right_arm])
    left_ee = qpos_from_item(item, frame_index, source, "left_ee", 1)
    right_ee = qpos_from_item(item, frame_index, source, "right_ee", 1)

    data.qpos[:] = base_qpos
    for idx, qpos_idx in enumerate(arm_qpos_indices):
        data.qpos[qpos_idx] = arm_q[idx]

    if dex1_qpos_indices is not None:
        left_q = gripper_state_to_mujoco_q(float(left_ee[0]))
        right_q = gripper_state_to_mujoco_q(float(right_ee[0]))
        for qpos_idx in dex1_qpos_indices["left"] or []:
            data.qpos[qpos_idx] = left_q
        for qpos_idx in dex1_qpos_indices["right"] or []:
            data.qpos[qpos_idx] = right_q

    if data.qvel is not None and data.qvel.size:
        data.qvel[:] = 0.0
    if data.ctrl is not None and data.ctrl.size:
        data.ctrl[:] = 0.0
    return arm_q, np.array([float(left_ee[0]), float(right_ee[0])], dtype=float)


def frame_sleep_s(items: list[dict[str, Any]], frame_index: int, start_frame: int, frequency: float, realtime: bool, speed: float) -> float:
    if not realtime or frame_index <= start_frame:
        return 1.0 / frequency / speed
    prev_ns = sample_ns(items[frame_index - 1], frame_index - 1)
    cur_ns = sample_ns(items[frame_index], frame_index)
    delta_s = max(0.0, float(cur_ns - prev_ns) / 1e9)
    return delta_s / speed


def validate_args(args: argparse.Namespace) -> None:
    if args.frequency <= 0.0:
        raise ValueError("--frequency must be positive")
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.gap_s < 0.0:
        raise ValueError("--gap-s must be non-negative")
    if args.camera_distance <= 0.0:
        raise ValueError("--camera-distance must be positive")
    if len(args.camera_lookat) != 3 or not all(math.isfinite(float(value)) for value in args.camera_lookat):
        raise ValueError("--camera-lookat must contain three finite values")


def configure_viewer_camera(viewer: Any, args: argparse.Namespace) -> None:
    viewer.cam.distance = float(args.camera_distance)
    viewer.cam.azimuth = float(args.camera_azimuth)
    viewer.cam.elevation = float(args.camera_elevation)
    viewer.cam.lookat[:] = np.asarray(args.camera_lookat, dtype=float)


def update_overlay(
    viewer: Any,
    args: argparse.Namespace,
    episode_index: int,
    episode_position: int,
    episode_total: int,
    frame_index: int,
    end_frame: int,
    gripper_q: np.ndarray,
) -> None:
    if args.no_overlay:
        return
    viewer.set_texts(
        (
            mj.mjtFontScale.mjFONTSCALE_150,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            (
                f"episode_{episode_index:04d} ({episode_position}/{episode_total})\n"
                f"frame {frame_index}/{end_frame - 1}\n"
                f"source={args.source}"
            ),
            (
                f"left_ee={gripper_q[0]:.4f}\n"
                f"right_ee={gripper_q[1]:.4f}"
            ),
        )
    )


def replay(args: argparse.Namespace) -> None:
    validate_args(args)
    episode_indices = resolve_episode_indices(args)


    xml_path = resolve_xml_path(args)
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    base_qpos = data.qpos.copy()

    arm_qpos_indices = joint_qpos_indices(model, ARM_JOINT_NAMES)
    dex1_qpos_indices = None
    if args.ee == "dex1":
        dex1_qpos_indices = {
            side: joint_qpos_indices_if_present(model, names)
            for side, names in DEX1_JOINT_NAMES.items()
        }
        if dex1_qpos_indices["left"] is None or dex1_qpos_indices["right"] is None:
            raise ValueError(f"XML has no complete Dex1 joint set: {xml_path}")

    print(
        f"[RAW_MUJOCO_REPLAY] dataset={args.dataset} episodes={','.join(f'{idx:04d}' for idx in episode_indices)} "
        f"source={args.source} viewer_robot={args.viewer_robot} ee={args.ee} xml={xml_path} headless={args.headless}"
    )

    stop_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal stop_requested
        if keycode == glfw.KEY_Q:
            stop_requested = True
            print("[RAW_MUJOCO_REPLAY] Q pressed, stopping.")

    def step_frame(items: list[dict[str, Any]], frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        arm_q, gripper_q = set_frame_qpos(
            data=data,
            base_qpos=base_qpos,
            arm_qpos_indices=arm_qpos_indices,
            dex1_qpos_indices=dex1_qpos_indices,
            item=items[frame_index],
            frame_index=frame_index,
            source=args.source,
        )
        mj.mj_forward(model, data)
        return arm_q, gripper_q

    if args.headless:
        for episode_index in episode_indices:
            items = load_items(args.dataset, episode_index)
            if args.start_frame >= len(items):
                raise ValueError(f"--start-frame {args.start_frame} is outside episode_{episode_index:04d} length {len(items)}")
            end_frame = len(items) if args.max_frames == 0 else min(len(items), args.start_frame + args.max_frames)
            last_arm_q = None
            last_gripper_q = None
            for frame_index in range(args.start_frame, end_frame):
                last_arm_q, last_gripper_q = step_frame(items, frame_index)
            print(
                f"[RAW_MUJOCO_REPLAY] episode_{episode_index:04d} headless complete: "
                f"frames={end_frame - args.start_frame} "
                f"last_arm_norm={float(np.linalg.norm(last_arm_q)):.6f} "
                f"last_gripper=({last_gripper_q[0]:.6f}, {last_gripper_q[1]:.6f})"
            )
        return

    with mjv.launch_passive(model, data, key_callback=key_callback, show_left_ui=False, show_right_ui=False) as viewer:
        configure_viewer_camera(viewer, args)
        for episode_position, episode_index in enumerate(episode_indices, start=1):
            if stop_requested or not viewer.is_running():
                break
            if episode_position > 1:
                if args.pause_between_episodes:
                    input(f"[RAW_MUJOCO_REPLAY] press Enter to replay episode_{episode_index:04d}...")
                elif args.gap_s > 0.0:
                    time.sleep(args.gap_s)

            items = load_items(args.dataset, episode_index)
            if args.start_frame >= len(items):
                raise ValueError(f"--start-frame {args.start_frame} is outside episode_{episode_index:04d} length {len(items)}")
            end_frame = len(items) if args.max_frames == 0 else min(len(items), args.start_frame + args.max_frames)
            print(
                f"[RAW_MUJOCO_REPLAY] episode_{episode_index:04d} "
                f"({episode_position}/{len(episode_indices)}) frames={len(items)} replay_range=[{args.start_frame}, {end_frame})",
                flush=True,
            )
            for frame_index in range(args.start_frame, end_frame):
                if stop_requested or not viewer.is_running():
                    break
                _, gripper_q = step_frame(items, frame_index)
                if frame_index == args.start_frame or frame_index % 30 == 0 or frame_index + 1 == end_frame:
                    print(
                        f"[RAW_MUJOCO_REPLAY] episode_{episode_index:04d} frame={frame_index}/{end_frame - 1} "
                        f"right_ee={gripper_q[1]:.6f}",
                        flush=True,
                    )
                update_overlay(
                    viewer=viewer,
                    args=args,
                    episode_index=episode_index,
                    episode_position=episode_position,
                    episode_total=len(episode_indices),
                    frame_index=frame_index,
                    end_frame=end_frame,
                    gripper_q=gripper_q,
                )
                viewer.sync()
                time.sleep(frame_sleep_s(items, frame_index, args.start_frame, args.frequency, args.realtime, args.speed))
            if stop_requested or not viewer.is_running():
                break
        if viewer.is_running():
            print("[RAW_MUJOCO_REPLAY] batch complete. Close viewer or press Q.")
            while viewer.is_running() and not stop_requested:
                viewer.sync()
                time.sleep(0.05)


def main() -> None:
    args = parse_args()
    replay(args)


if __name__ == "__main__":
    main()
