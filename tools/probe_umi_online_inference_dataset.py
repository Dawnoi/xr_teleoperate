#!/usr/bin/env python3
"""Probe a Pika/UMI online inference server with exported UMI/DP frames.

This tool does not command the robot. It reads an exported episode folder,
builds the same newline-delimited observation JSON used by online inference,
sends it to the TCP server, and compares the returned action with nearby
dataset TCP/gripper targets.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import math
from pathlib import Path
import socket
from typing import Any

from PIL import Image


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} contains NaN or Inf")
    return out


def finite_float_list(values: Any, expected_len: int, label: str) -> list[float]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    if len(values) != expected_len:
        raise ValueError(f"{label} expected length {expected_len}, got {len(values)}")
    return [finite_float(value, f"{label}[{idx}]") for idx, value in enumerate(values)]


def item_pose7(items: list[dict[str, Any]], index: int) -> list[float]:
    item = items[index]
    xyz = finite_float_list(item.get("xyz"), 3, f"tcp[{index}].xyz")
    quat = finite_float_list(item.get("quat_xyzw"), 4, f"tcp[{index}].quat_xyzw")
    return xyz + quat


def item_gripper(items: list[dict[str, Any]], index: int) -> float:
    return finite_float(items[index].get("position"), f"gripper[{index}].position")


def encode_image_jpeg_base64(path: Path, quality: int) -> str:
    image = Image.open(path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def sorted_camera_frames(episode_dir: Path, camera_name: str) -> list[Path]:
    camera_dir = episode_dir / "camera" / camera_name
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"missing camera directory: {camera_dir}")
    frames = sorted(camera_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no PNG frames found in {camera_dir}")
    return frames


def build_images(
    episode_dir: Path,
    camera_order: list[str],
    obs_indices: list[int],
    jpeg_quality: int,
    image_layout: str,
) -> list[str]:
    frame_paths = {camera: sorted_camera_frames(episode_dir, camera) for camera in camera_order}
    for camera, frames in frame_paths.items():
        max_index = max(obs_indices)
        if max_index >= len(frames):
            raise IndexError(f"{camera} has {len(frames)} frames, cannot read index {max_index}")

    images: list[str] = []
    if image_layout == "step-major":
        for index in obs_indices:
            for camera in camera_order:
                images.append(encode_image_jpeg_base64(frame_paths[camera][index], jpeg_quality))
    elif image_layout == "camera-major":
        for camera in camera_order:
            for index in obs_indices:
                images.append(encode_image_jpeg_base64(frame_paths[camera][index], jpeg_quality))
    else:
        raise ValueError("image_layout must be 'step-major' or 'camera-major'")
    return images


def send_json_line(sock: socket.socket, payload: dict[str, Any]) -> None:
    line = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    sock.sendall(line + b"\n")


def recv_json_line(sock: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("server closed TCP connection before newline JSON response")
        chunks.append(chunk)
        joined = b"".join(chunks)
        newline_index = joined.find(b"\n")
        if newline_index >= 0:
            line = joined[:newline_index]
            return json.loads(line.decode("utf-8"))


def recv_action_response(sock: socket.socket) -> dict[str, Any]:
    while True:
        payload = recv_json_line(sock)
        if payload.get("type") == "reset_ack":
            print("reset_ack:", json.dumps(payload, ensure_ascii=False))
            continue
        if payload.get("type") == "action":
            return payload
        raise ValueError(f"unexpected response type: {payload.get('type')!r}, payload={payload!r}")


def send_reset_then_observation(
    sock: socket.socket,
    *,
    observation: dict[str, Any],
) -> dict[str, Any]:
    send_json_line(sock, {"type": "reset"})
    reset_payload = recv_json_line(sock)
    if reset_payload.get("type") != "reset_ack":
        raise ValueError(f"unexpected reset response type: {reset_payload.get('type')!r}, payload={reset_payload!r}")
    print("reset:", json.dumps(reset_payload, ensure_ascii=False))
    send_json_line(sock, observation)
    return recv_action_response(sock)


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


def parse_action_steps(payload: dict[str, Any], arm_side: str) -> list[list[float]]:
    action_key = f"action_{arm_side[0]}"
    raw_steps = payload.get(action_key)
    if raw_steps is None:
        raw_steps = payload.get("action")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"missing non-empty {action_key} or action in server response")

    steps: list[list[float]] = []
    for step_index, raw_step in enumerate(raw_steps):
        steps.append(finite_float_list(raw_step, 8, f"{action_key}[{step_index}]"))
    return steps


def xyz_delta(a_pose7: list[float], b_pose7: list[float]) -> float:
    return math.sqrt(sum((a_pose7[idx] - b_pose7[idx]) ** 2 for idx in range(3)))


def print_pose(label: str, pose7: list[float]) -> None:
    xyz = ", ".join(f"{value:.6f}" for value in pose7[:3])
    quat = ", ".join(f"{value:.6f}" for value in pose7[3:7])
    print(f"{label}: xyz=[{xyz}] quat_xyzw=[{quat}]")


def load_episode_items(dataset_root: Path, episode: int, arm_side: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_dir = dataset_root / f"episode_{int(episode):06d}"
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"missing episode directory: {episode_dir}")
    meta = load_json(episode_dir / "meta.json")
    tcp_items = load_json(episode_dir / "tcp" / f"{arm_side}.json")["items"]
    gripper_items = load_json(episode_dir / "gripper" / f"{arm_side}.json")["items"]
    if len(tcp_items) != len(gripper_items):
        raise ValueError(f"tcp/gripper length mismatch: {len(tcp_items)} vs {len(gripper_items)}")
    return episode_dir, meta, tcp_items, gripper_items


def build_observation(
    episode_dir: Path,
    tcp_items: list[dict[str, Any]],
    gripper_items: list[dict[str, Any]],
    arm_side: str,
    frame_index: int,
    n_obs_steps: int,
    camera_order: list[str],
    image_layout: str,
    jpeg_quality: int,
) -> tuple[dict[str, Any], list[int]]:
    if n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")
    if frame_index < n_obs_steps - 1:
        raise ValueError(f"frame_index={frame_index} is too early for n_obs_steps={n_obs_steps}")
    if frame_index >= len(tcp_items):
        raise IndexError(f"frame_index={frame_index} out of range for episode length {len(tcp_items)}")

    obs_indices = list(range(frame_index - n_obs_steps + 1, frame_index + 1))
    poses = [item_pose7(tcp_items, index) for index in obs_indices]
    grippers = [item_gripper(gripper_items, index) for index in obs_indices]
    images = build_images(
        episode_dir=episode_dir,
        camera_order=camera_order,
        obs_indices=obs_indices,
        jpeg_quality=jpeg_quality,
        image_layout=image_layout,
    )

    arm_key = "arm_r" if arm_side == "right" else "arm_l"
    return (
        {
            "type": "observation",
            arm_key: {
                "images": images,
                "poses": poses,
                "grippers": grippers,
                "init_pose": item_pose7(tcp_items, 0),
                "arm_current_pose": poses[-1],
            },
        },
        obs_indices,
    )


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    camera_order = [part.strip() for part in str(args.camera_order).split(",") if part.strip()]
    if not camera_order:
        raise ValueError("camera_order must contain at least one camera")

    episode_dir, meta, tcp_items, gripper_items = load_episode_items(dataset_root, args.episode, args.arm_side)
    observation, obs_indices = build_observation(
        episode_dir=episode_dir,
        tcp_items=tcp_items,
        gripper_items=gripper_items,
        arm_side=args.arm_side,
        frame_index=args.frame_index,
        n_obs_steps=args.n_obs_steps,
        camera_order=camera_order,
        image_layout=args.image_layout,
        jpeg_quality=args.jpeg_quality,
    )

    if args.frame_index + args.compare_horizon >= len(tcp_items):
        raise IndexError(
            f"frame_index + compare_horizon exceeds episode length: {args.frame_index} + "
            f"{args.compare_horizon} >= {len(tcp_items)}"
        )

    response = request_action(
        host=args.host,
        port=int(args.port),
        timeout_sec=float(args.timeout_sec),
        observation=observation,
    )

    action_steps = parse_action_steps(response, args.arm_side)
    current_pose = item_pose7(tcp_items, args.frame_index)
    current_gripper = item_gripper(gripper_items, args.frame_index)

    print("dataset_root:", dataset_root)
    print("episode:", episode_dir.name)
    print("meta.camera_keys:", json.dumps(meta.get("camera_keys", {}), ensure_ascii=False))
    print("arm_side:", args.arm_side)
    print("frame_index:", args.frame_index)
    print("obs_indices:", obs_indices)
    print("camera_order:", camera_order)
    print("image_layout:", args.image_layout)
    print("n_images_sent:", len(observation["arm_r" if args.arm_side == "right" else "arm_l"]["images"]))
    print()
    print_pose("dataset current", current_pose)
    print("dataset current gripper:", f"{current_gripper:.6f}")
    print_pose("server action[0]", action_steps[0][:7])
    print("server action[0] gripper:", f"{action_steps[0][7]:.6f}")
    print("server action[0] xyz_delta_vs_current_m:", f"{xyz_delta(action_steps[0][:7], current_pose):.6f}")
    print()
    print("comparison candidates:")
    for offset in range(0, args.compare_horizon + 1):
        dataset_index = args.frame_index + offset
        dataset_pose = item_pose7(tcp_items, dataset_index)
        dataset_gripper = item_gripper(gripper_items, dataset_index)
        action_index = min(offset, len(action_steps) - 1)
        action_step = action_steps[action_index]
        print(
            f"  offset={offset:02d} dataset_idx={dataset_index:06d} "
            f"action_idx={action_index:02d} "
            f"xyz_delta_m={xyz_delta(action_step[:7], dataset_pose):.6f} "
            f"dataset_gripper={dataset_gripper:.6f} "
            f"server_gripper={action_step[7]:.6f} "
            f"gripper_delta={action_step[7] - dataset_gripper:.6f}"
        )

    summary = {
        "dataset_root": str(dataset_root),
        "episode": episode_dir.name,
        "arm_side": args.arm_side,
        "frame_index": int(args.frame_index),
        "obs_indices": obs_indices,
        "camera_order": camera_order,
        "image_layout": args.image_layout,
        "current_pose": current_pose,
        "current_gripper": current_gripper,
        "response": response,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote:", args.output_json)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send exported UMI/DP dataset observations to an online inference server and compare returned actions."
    )
    parser.add_argument("--dataset-root", default="/home/luopengcheng/Programs/xr_teleoperate/utils/data/test")
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument("--frame-index", type=int, default=225)
    parser.add_argument("--arm-side", choices=["left", "right"], default="right")
    parser.add_argument("--host", default="192.168.100.27")
    parser.add_argument("--port", type=int, default=8007)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--camera-order", default="cam0,cam1,cam2")
    parser.add_argument("--image-layout", choices=["step-major", "camera-major"], default="step-major")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--compare-horizon", type=int, default=16)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    run_probe(parse_args())


if __name__ == "__main__":
    main()
