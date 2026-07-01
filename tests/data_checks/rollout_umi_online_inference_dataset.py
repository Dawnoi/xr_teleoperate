#!/usr/bin/env python3
"""Teacher-forcing rollout probe for a Pika/UMI online inference server.

For each selected dataset frame, this tool sends the original camera images and
dataset state to the server, stores the full returned action chunk, and compares
the first N actions in that chunk against the dataset frames between this sample
and the next sample. It does not command the robot.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.data_checks.probe_umi_online_inference_dataset import (  # noqa: E402
    build_observation,
    item_gripper,
    item_pose7,
    load_episode_items,
    parse_action_steps,
    send_reset_then_observation,
    xyz_delta,
)


def rollout_frame_indices(
    episode_len: int,
    n_obs_steps: int,
    chunk_size: int,
    start_frame: int | None,
    end_frame: int | None,
    stride: int | None,
) -> list[int]:
    if episode_len <= 0:
        raise ValueError("episode_len must be positive")
    if n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    resolved_stride = int(chunk_size if stride is None else stride)
    if resolved_stride <= 0:
        raise ValueError("stride must be positive")

    first_allowed = n_obs_steps - 1
    last_allowed_exclusive = episode_len - chunk_size + 1
    first = first_allowed if start_frame is None else max(first_allowed, int(start_frame))
    last = last_allowed_exclusive if end_frame is None else min(last_allowed_exclusive, int(end_frame))
    indices = list(range(first, last, resolved_stride))
    if not indices:
        raise ValueError(
            "no rollout frames selected: "
            f"episode_len={episode_len}, n_obs_steps={n_obs_steps}, chunk_size={chunk_size}, "
            f"start_frame={start_frame}, end_frame={end_frame}, stride={resolved_stride}"
        )
    return indices


def summarize_rollout_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot summarize empty rollout samples")
    xyz_errors = []
    gripper_errors = []
    for sample_index, sample in enumerate(samples):
        dataset_poses = sample["dataset_poses"]
        dataset_grippers = sample["dataset_grippers"]
        expanded_actions = sample["expanded_actions"]
        if not isinstance(dataset_poses, list) or not isinstance(dataset_grippers, list) or not isinstance(expanded_actions, list):
            raise ValueError(f"samples[{sample_index}] must contain list dataset_poses, dataset_grippers, expanded_actions")
        if len(dataset_poses) != len(dataset_grippers) or len(dataset_poses) != len(expanded_actions):
            raise ValueError(
                f"samples[{sample_index}] expanded lengths mismatch: "
                f"{len(dataset_poses)} poses, {len(dataset_grippers)} grippers, {len(expanded_actions)} actions"
            )
        for chunk_index, (dataset_pose, dataset_gripper, selected_action) in enumerate(
            zip(dataset_poses, dataset_grippers, expanded_actions)
        ):
            if not isinstance(dataset_pose, list) or len(dataset_pose) != 7:
                raise ValueError(f"samples[{sample_index}].dataset_poses[{chunk_index}] must be pose7")
            if not isinstance(selected_action, list) or len(selected_action) != 8:
                raise ValueError(f"samples[{sample_index}].expanded_actions[{chunk_index}] must be action8")
            xyz_errors.append(xyz_delta(selected_action[:7], dataset_pose))
            gripper_errors.append(abs(float(selected_action[7]) - float(dataset_gripper)))

    return {
        "num_samples": len(samples),
        "num_expanded_points": len(xyz_errors),
        "mean_xyz_error_m": float(sum(xyz_errors) / len(xyz_errors)),
        "max_xyz_error_m": float(max(xyz_errors)),
        "mean_abs_gripper_error": float(sum(gripper_errors) / len(gripper_errors)),
        "max_abs_gripper_error": float(max(gripper_errors)),
    }


def build_rollout_output(
    *,
    dataset_root: Path,
    episode_name: str,
    episode_index: int,
    arm_side: str,
    host: str,
    port: int,
    n_obs_steps: int,
    camera_order: list[str],
    image_layout: str,
    jpeg_quality: int,
    chunk_size: int,
    stride: int,
    start_frame: int | None,
    end_frame: int | None,
    meta: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "teacher_forcing_rollout",
        "dataset_root": str(dataset_root),
        "episode": episode_name,
        "episode_index": int(episode_index),
        "arm_side": arm_side,
        "host": host,
        "port": int(port),
        "n_obs_steps": int(n_obs_steps),
        "camera_order": camera_order,
        "image_layout": image_layout,
        "jpeg_quality": int(jpeg_quality),
        "chunk_size": int(chunk_size),
        "stride": int(stride),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "meta_camera_keys": meta.get("camera_keys", {}),
        "summary": summarize_rollout_samples(samples),
        "samples": samples,
    }


def write_rollout_json(
    output_json: Path,
    *,
    dataset_root: Path,
    episode_name: str,
    episode_index: int,
    arm_side: str,
    host: str,
    port: int,
    n_obs_steps: int,
    camera_order: list[str],
    image_layout: str,
    jpeg_quality: int,
    chunk_size: int,
    stride: int,
    start_frame: int | None,
    end_frame: int | None,
    meta: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    output = build_rollout_output(
        dataset_root=dataset_root,
        episode_name=episode_name,
        episode_index=episode_index,
        arm_side=arm_side,
        host=host,
        port=port,
        n_obs_steps=n_obs_steps,
        camera_order=camera_order,
        image_layout=image_layout,
        jpeg_quality=jpeg_quality,
        chunk_size=chunk_size,
        stride=stride,
        start_frame=start_frame,
        end_frame=end_frame,
        meta=meta,
        samples=samples,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def request_action(
    *,
    host: str,
    port: int,
    timeout_sec: float,
    observation: dict[str, Any],
) -> dict[str, Any]:
    with socket.create_connection((host, int(port)), timeout=float(timeout_sec)) as sock:
        sock.settimeout(float(timeout_sec))
        return send_reset_then_observation(sock, observation=observation)


def run_rollout(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    camera_order = [part.strip() for part in str(args.camera_order).split(",") if part.strip()]
    if not camera_order:
        raise ValueError("camera_order must contain at least one camera")

    episode_dir, meta, tcp_items, gripper_items = load_episode_items(dataset_root, args.episode, args.arm_side)
    frame_indices = rollout_frame_indices(
        episode_len=len(tcp_items),
        n_obs_steps=args.n_obs_steps,
        chunk_size=args.chunk_size,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
    )

    samples: list[dict[str, Any]] = []
    partial_json = output_json.with_suffix(output_json.suffix + ".partial")

    def process_frame(ordinal: int, frame_index: int, sock: socket.socket | None) -> None:
            observation, obs_indices = build_observation(
                episode_dir=episode_dir,
                tcp_items=tcp_items,
                gripper_items=gripper_items,
                arm_side=args.arm_side,
                frame_index=frame_index,
                n_obs_steps=args.n_obs_steps,
                camera_order=camera_order,
                image_layout=args.image_layout,
                jpeg_quality=args.jpeg_quality,
            )
            if sock is None:
                response = request_action(
                    host=args.host,
                    port=int(args.port),
                    timeout_sec=float(args.timeout_sec),
                    observation=observation,
                )
            else:
                response = send_reset_then_observation(
                    sock,
                    observation=observation,
                )
            action_steps = parse_action_steps(response, args.arm_side)
            if args.chunk_size > len(action_steps):
                raise IndexError(
                    f"chunk_size={args.chunk_size} exceeds returned action chunk length {len(action_steps)} "
                    f"at frame_index={frame_index}"
                )

            dataset_poses = []
            dataset_grippers = []
            expanded_actions = []
            expanded_errors = []
            expanded_gripper_errors = []
            for action_offset in range(args.chunk_size):
                target_index = frame_index + action_offset
                selected_action = action_steps[action_offset]
                dataset_pose = item_pose7(tcp_items, target_index)
                dataset_gripper = item_gripper(gripper_items, target_index)
                dataset_poses.append(dataset_pose)
                dataset_grippers.append(dataset_gripper)
                expanded_actions.append(selected_action)
                expanded_errors.append(xyz_delta(selected_action[:7], dataset_pose))
                expanded_gripper_errors.append(float(selected_action[7]) - float(dataset_gripper))
            sample = {
                "ordinal": ordinal,
                "frame_index": frame_index,
                "target_start_index": frame_index,
                "target_end_index": frame_index + args.chunk_size - 1,
                "obs_indices": obs_indices,
                "chunk_size": args.chunk_size,
                "dataset_poses": dataset_poses,
                "dataset_grippers": dataset_grippers,
                "expanded_actions": expanded_actions,
                "action_steps": action_steps,
                "mean_xyz_error_m": float(sum(expanded_errors) / len(expanded_errors)),
                "max_xyz_error_m": float(max(expanded_errors)),
                "mean_gripper_error": float(sum(expanded_gripper_errors) / len(expanded_gripper_errors)),
            }
            samples.append(sample)
            if args.progress_every > 0 and (ordinal == 0 or (ordinal + 1) % args.progress_every == 0):
                print(
                    f"sample {ordinal + 1}/{len(frame_indices)} "
                    f"frame={frame_index} mean_xyz_error_m={sample['mean_xyz_error_m']:.6f} "
                    f"mean_gripper_error={sample['mean_gripper_error']:.6f}"
                )
            if args.sleep_sec > 0.0:
                time.sleep(float(args.sleep_sec))

    try:
        if args.connect_per_frame:
            for ordinal, frame_index in enumerate(frame_indices):
                process_frame(ordinal, frame_index, None)
                if args.checkpoint_every > 0 and (ordinal + 1) % args.checkpoint_every == 0:
                    write_rollout_json(
                        partial_json,
                        dataset_root=dataset_root,
                        episode_name=episode_dir.name,
                        episode_index=args.episode,
                        arm_side=args.arm_side,
                        host=args.host,
                        port=int(args.port),
                        n_obs_steps=args.n_obs_steps,
                        camera_order=camera_order,
                        image_layout=args.image_layout,
                        jpeg_quality=args.jpeg_quality,
                        chunk_size=args.chunk_size,
                        stride=args.stride,
                        start_frame=args.start_frame,
                        end_frame=args.end_frame,
                        meta=meta,
                        samples=samples,
                    )
                    print("checkpoint:", partial_json)
        else:
            with socket.create_connection((args.host, int(args.port)), timeout=float(args.timeout_sec)) as sock:
                sock.settimeout(float(args.timeout_sec))
                for ordinal, frame_index in enumerate(frame_indices):
                    response = None
                    observation, obs_indices = build_observation(
                        episode_dir=episode_dir,
                        tcp_items=tcp_items,
                        gripper_items=gripper_items,
                        arm_side=args.arm_side,
                        frame_index=frame_index,
                        n_obs_steps=args.n_obs_steps,
                        camera_order=camera_order,
                        image_layout=args.image_layout,
                        jpeg_quality=args.jpeg_quality,
                    )
                    response = send_reset_then_observation(
                        sock,
                        observation=observation,
                    )
                    action_steps = parse_action_steps(response, args.arm_side)
                    if args.chunk_size > len(action_steps):
                        raise IndexError(
                            f"chunk_size={args.chunk_size} exceeds returned action chunk length {len(action_steps)} "
                            f"at frame_index={frame_index}"
                        )
                    dataset_poses = []
                    dataset_grippers = []
                    expanded_actions = []
                    expanded_errors = []
                    expanded_gripper_errors = []
                    for action_offset in range(args.chunk_size):
                        target_index = frame_index + action_offset
                        selected_action = action_steps[action_offset]
                        dataset_pose = item_pose7(tcp_items, target_index)
                        dataset_gripper = item_gripper(gripper_items, target_index)
                        dataset_poses.append(dataset_pose)
                        dataset_grippers.append(dataset_gripper)
                        expanded_actions.append(selected_action)
                        expanded_errors.append(xyz_delta(selected_action[:7], dataset_pose))
                        expanded_gripper_errors.append(float(selected_action[7]) - float(dataset_gripper))
                    sample = {
                        "ordinal": ordinal,
                        "frame_index": frame_index,
                        "target_start_index": frame_index,
                        "target_end_index": frame_index + args.chunk_size - 1,
                        "obs_indices": obs_indices,
                        "chunk_size": args.chunk_size,
                        "dataset_poses": dataset_poses,
                        "dataset_grippers": dataset_grippers,
                        "expanded_actions": expanded_actions,
                        "action_steps": action_steps,
                        "mean_xyz_error_m": float(sum(expanded_errors) / len(expanded_errors)),
                        "max_xyz_error_m": float(max(expanded_errors)),
                        "mean_gripper_error": float(sum(expanded_gripper_errors) / len(expanded_gripper_errors)),
                    }
                    samples.append(sample)
                    if args.progress_every > 0 and (ordinal == 0 or (ordinal + 1) % args.progress_every == 0):
                        print(
                            f"sample {ordinal + 1}/{len(frame_indices)} "
                            f"frame={frame_index} mean_xyz_error_m={sample['mean_xyz_error_m']:.6f} "
                            f"mean_gripper_error={sample['mean_gripper_error']:.6f}"
                        )
                    if args.sleep_sec > 0.0:
                        time.sleep(float(args.sleep_sec))
                    if args.checkpoint_every > 0 and (ordinal + 1) % args.checkpoint_every == 0:
                        write_rollout_json(
                            partial_json,
                            dataset_root=dataset_root,
                            episode_name=episode_dir.name,
                            episode_index=args.episode,
                            arm_side=args.arm_side,
                            host=args.host,
                            port=int(args.port),
                            n_obs_steps=args.n_obs_steps,
                            camera_order=camera_order,
                            image_layout=args.image_layout,
                            jpeg_quality=args.jpeg_quality,
                            chunk_size=args.chunk_size,
                            stride=args.stride,
                            start_frame=args.start_frame,
                            end_frame=args.end_frame,
                            meta=meta,
                            samples=samples,
                        )
                        print("checkpoint:", partial_json)
    except Exception:
        if samples:
            write_rollout_json(
                partial_json,
                dataset_root=dataset_root,
                episode_name=episode_dir.name,
                episode_index=args.episode,
                arm_side=args.arm_side,
                host=args.host,
                port=int(args.port),
                n_obs_steps=args.n_obs_steps,
                camera_order=camera_order,
                image_layout=args.image_layout,
                jpeg_quality=args.jpeg_quality,
                chunk_size=args.chunk_size,
                stride=args.stride,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                meta=meta,
                samples=samples,
            )
            print("partial_wrote:", partial_json)
        raise

    output = write_rollout_json(
        output_json,
        dataset_root=dataset_root,
        episode_name=episode_dir.name,
        episode_index=args.episode,
        arm_side=args.arm_side,
        host=args.host,
        port=int(args.port),
        n_obs_steps=args.n_obs_steps,
        camera_order=camera_order,
        image_layout=args.image_layout,
        jpeg_quality=args.jpeg_quality,
        chunk_size=args.chunk_size,
        stride=args.stride,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        meta=meta,
        samples=samples,
    )
    print("summary:", json.dumps(output["summary"], ensure_ascii=False))
    print("wrote:", output_json)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run teacher-forcing rollout over exported UMI/DP frames using original images and dataset state."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--arm-side", choices=["left", "right"], default="right")
    parser.add_argument("--host", default="192.168.100.27")
    parser.add_argument("--port", type=int, default=8007)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--camera-order", default="cam0,cam1,cam2")
    parser.add_argument("--image-layout", choices=["step-major", "camera-major"], default="step-major")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--connect-per-frame", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    run_rollout(parse_args())


if __name__ == "__main__":
    main()
