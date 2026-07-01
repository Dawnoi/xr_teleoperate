#!/usr/bin/env python3
"""Render a MuJoCo diagnostic view for a UMI online-inference probe.

The script is intentionally read-only with respect to robot control. It loads a
saved single-window probe JSON or teacher-forcing rollout JSON, overlays the
dataset TCP trajectory and the server action trajectory in the same MuJoCo
scene, and writes a PNG plus an MP4. It does not run IK or publish commands.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from teleop.sim.g1d_mujoco_builder import prepare_g1d_mobile_scene


RGBA_DATASET = np.array([0.05, 0.55, 1.00, 1.0], dtype=np.float32)
RGBA_SERVER = np.array([1.00, 0.20, 0.12, 1.0], dtype=np.float32)
RGBA_CURRENT = np.array([1.00, 0.90, 0.05, 1.0], dtype=np.float32)
RGBA_END = np.array([0.15, 1.00, 0.35, 1.0], dtype=np.float32)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} contains NaN or Inf")
    return out


def finite_pose7(values: Any, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != 7:
        raise ValueError(f"{label} must be a 7D pose [x, y, z, qx, qy, qz, qw]")
    return [finite_float(value, f"{label}[{idx}]") for idx, value in enumerate(values)]


def load_dataset_trajectory(dataset_root: Path, episode: int, arm_side: str, frame_index: int, horizon: int) -> tuple[np.ndarray, list[float]]:
    episode_dir = dataset_root / f"episode_{episode:06d}"
    tcp_path = episode_dir / "tcp" / f"{arm_side}.json"
    gripper_path = episode_dir / "gripper" / f"{arm_side}.json"
    if not tcp_path.is_file():
        raise FileNotFoundError(f"missing TCP JSON: {tcp_path}")
    if not gripper_path.is_file():
        raise FileNotFoundError(f"missing gripper JSON: {gripper_path}")

    tcp_items = load_json(tcp_path)["items"]
    gripper_items = load_json(gripper_path)["items"]
    end_index = frame_index + horizon
    if end_index > len(tcp_items):
        raise IndexError(f"frame_index + horizon exceeds TCP length: {frame_index} + {horizon} > {len(tcp_items)}")
    if end_index > len(gripper_items):
        raise IndexError(
            f"frame_index + horizon exceeds gripper length: {frame_index} + {horizon} > {len(gripper_items)}"
        )

    xyz = []
    grippers = []
    for idx in range(frame_index, end_index):
        pose7 = finite_pose7(tcp_items[idx].get("xyz", []) + tcp_items[idx].get("quat_xyzw", []), f"tcp[{idx}]")
        xyz.append(pose7[:3])
        grippers.append(finite_float(gripper_items[idx].get("position"), f"gripper[{idx}].position"))
    return np.asarray(xyz, dtype=np.float64), grippers


def load_probe_trajectory(probe_json: Path, arm_side: str, horizon: int) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    payload = load_json(probe_json)
    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"probe JSON missing response object: {probe_json}")

    action_key = f"action_{arm_side[0]}"
    raw_steps = response.get(action_key)
    if raw_steps is None:
        raw_steps = response.get("action")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"probe JSON missing non-empty {action_key} or action")

    if horizon > len(raw_steps):
        raise IndexError(f"requested horizon={horizon}, but probe only has {len(raw_steps)} action steps")

    xyz = []
    grippers = []
    for idx in range(horizon):
        step = raw_steps[idx]
        if not isinstance(step, list) or len(step) != 8:
            raise ValueError(f"{action_key}[{idx}] must be an 8D action")
        values = [finite_float(value, f"{action_key}[{idx}][{col}]") for col, value in enumerate(step)]
        xyz.append(values[:3])
        grippers.append(values[7])
    return np.asarray(xyz, dtype=np.float64), grippers, payload


def load_rollout_trajectories(rollout_json: Path) -> tuple[np.ndarray, list[float], np.ndarray, list[float], dict[str, Any]]:
    payload = load_json(rollout_json)
    if payload.get("type") != "teacher_forcing_rollout":
        raise ValueError(f"rollout JSON has unexpected type: {payload.get('type')!r}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"rollout JSON must contain non-empty samples: {rollout_json}")

    dataset_xyz = []
    dataset_grippers = []
    server_xyz = []
    server_grippers = []
    for sample_index, sample in enumerate(samples):
        dataset_poses = sample.get("dataset_poses")
        dataset_gripper_values = sample.get("dataset_grippers")
        expanded_actions = sample.get("expanded_actions")
        if not isinstance(dataset_poses, list) or not isinstance(dataset_gripper_values, list) or not isinstance(expanded_actions, list):
            raise ValueError(
                f"samples[{sample_index}] must contain dataset_poses, dataset_grippers, expanded_actions lists"
            )
        if len(dataset_poses) != len(dataset_gripper_values) or len(dataset_poses) != len(expanded_actions):
            raise ValueError(
                f"samples[{sample_index}] expanded lengths mismatch: "
                f"{len(dataset_poses)} poses, {len(dataset_gripper_values)} grippers, {len(expanded_actions)} actions"
            )
        for chunk_index, (dataset_pose_raw, dataset_gripper_raw, action_raw) in enumerate(
            zip(dataset_poses, dataset_gripper_values, expanded_actions)
        ):
            dataset_pose = finite_pose7(dataset_pose_raw, f"samples[{sample_index}].dataset_poses[{chunk_index}]")
            if not isinstance(action_raw, list) or len(action_raw) != 8:
                raise ValueError(f"samples[{sample_index}].expanded_actions[{chunk_index}] must be an 8D action")
            action_values = [
                finite_float(value, f"samples[{sample_index}].expanded_actions[{chunk_index}][{idx}]")
                for idx, value in enumerate(action_raw)
            ]
            dataset_xyz.append(dataset_pose[:3])
            dataset_grippers.append(
                finite_float(dataset_gripper_raw, f"samples[{sample_index}].dataset_grippers[{chunk_index}]")
            )
            server_xyz.append(action_values[:3])
            server_grippers.append(action_values[7])
    return (
        np.asarray(dataset_xyz, dtype=np.float64),
        dataset_grippers,
        np.asarray(server_xyz, dtype=np.float64),
        server_grippers,
        payload,
    )


def set_free_camera(
    model: mj.MjModel,
    data: mj.MjData,
    center: np.ndarray,
    radius: float,
    distance_scale: float,
    min_distance: float,
) -> mj.MjvCamera:
    camera = mj.MjvCamera()
    mj.mjv_defaultFreeCamera(model, camera)
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center
    camera.distance = max(float(min_distance), radius * float(distance_scale))
    camera.azimuth = -118.0
    camera.elevation = -23.0
    mj.mj_forward(model, data)
    return camera


def add_sphere(scene: mj.MjvScene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(f"MuJoCo scene geom capacity exceeded: {scene.ngeom} >= {scene.maxgeom}")
    mj.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mj.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    scene.ngeom += 1


def add_segment(scene: mj.MjvScene, start: np.ndarray, end: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(f"MuJoCo scene geom capacity exceeded: {scene.ngeom} >= {scene.maxgeom}")
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        mj.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    mj.mjv_connector(
        geom,
        mj.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    geom.rgba[:] = rgba
    scene.ngeom += 1


def overlay_trajectory(scene: mj.MjvScene, points: np.ndarray, rgba: np.ndarray, radius: float) -> None:
    if points.shape[0] < 1 or points.shape[1] != 3:
        raise ValueError(f"trajectory points must have shape (N, 3), got {points.shape}")
    for idx in range(points.shape[0] - 1):
        add_segment(scene, points[idx], points[idx + 1], radius, rgba)
    for idx, point in enumerate(points):
        point_radius = radius * (1.9 if idx in {0, points.shape[0] - 1} else 1.2)
        point_rgba = RGBA_CURRENT if idx == 0 else RGBA_END if idx == points.shape[0] - 1 else rgba
        add_sphere(scene, point, point_radius, point_rgba)


def draw_text_panel(
    rgb: np.ndarray,
    *,
    title: str,
    dataset_points: np.ndarray,
    server_points: np.ndarray,
    dataset_grippers: list[float],
    server_grippers: list[float],
    probe_json: Path,
) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    panel_w = 430
    panel_h = 186
    margin = 14
    draw.rectangle([margin, margin, margin + panel_w, margin + panel_h], fill=(0, 0, 0, 176))

    dataset_d = float(np.linalg.norm(dataset_points[-1] - dataset_points[0]))
    server_d = float(np.linalg.norm(server_points[-1] - server_points[0]))
    end_err = float(np.linalg.norm(server_points[-1] - dataset_points[-1]))
    mean_err = float(np.mean(np.linalg.norm(server_points - dataset_points, axis=1)))

    lines = [
        title,
        f"blue dataset TCP: d15={dataset_d:.3f} m, gripper {dataset_grippers[0]:.3f}->{dataset_grippers[-1]:.3f}",
        f"red server action: d15={server_d:.3f} m, gripper {server_grippers[0]:.3f}->{server_grippers[-1]:.3f}",
        f"mean xyz error={mean_err:.3f} m, end xyz error={end_err:.3f} m",
        "yellow=start, green=end",
        f"probe={probe_json}",
    ]
    y = margin + 12
    for line in lines:
        draw.text((margin + 12, y), line, fill=(255, 255, 255, 255), font=font)
        y += 25
    return np.asarray(image)


def render_frame(
    renderer: mj.Renderer,
    model: mj.MjModel,
    data: mj.MjData,
    camera: mj.MjvCamera,
    dataset_points: np.ndarray,
    server_points: np.ndarray,
    dataset_grippers: list[float],
    server_grippers: list[float],
    probe_json: Path,
    title: str,
    progress: float,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    scene = renderer.scene
    overlay_trajectory(scene, dataset_points, RGBA_DATASET, radius=0.006)
    overlay_trajectory(scene, server_points, RGBA_SERVER, radius=0.005)

    max_index = dataset_points.shape[0] - 1
    marker_index = int(round(float(np.clip(progress, 0.0, 1.0)) * max_index))
    add_sphere(scene, dataset_points[marker_index], 0.018, RGBA_DATASET)
    add_sphere(scene, server_points[marker_index], 0.016, RGBA_SERVER)

    rgb = renderer.render()
    return draw_text_panel(
        rgb,
        title=title,
        dataset_points=dataset_points,
        server_points=server_points,
        dataset_grippers=dataset_grippers,
        server_grippers=server_grippers,
        probe_json=probe_json,
    )


def body_position(model: mj.MjModel, data: mj.MjData, body_name: str) -> np.ndarray:
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {body_name}")
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy()


def display_points_in_mujoco_world(
    dataset_points: np.ndarray,
    server_points: np.ndarray,
    model: mj.MjModel,
    data: mj.MjData,
    arm_side: str,
) -> tuple[np.ndarray, np.ndarray]:
    wrist_body = "right_wrist_yaw_link" if arm_side == "right" else "left_wrist_yaw_link"
    world_home = body_position(model, data, wrist_body) + np.array([0.05, 0.0, 0.0], dtype=np.float64)
    display_offset = world_home - dataset_points[0]
    return dataset_points + display_offset, server_points + display_offset


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("cannot write MP4 with no frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open OpenCV VideoWriter: {path}")
    for frame in frames:
        if frame.shape != frames[0].shape:
            raise ValueError(f"all frames must have shape {frames[0].shape}, got {frame.shape}")
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MuJoCo overlay for a saved UMI online-inference probe.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument("--frame-index", type=int, default=225)
    parser.add_argument("--arm-side", choices=["left", "right"], default="right")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--video-frames", type=int, default=40)
    parser.add_argument("--output-dir", default="/tmp/xr_teleoperate_mujoco_probe")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--camera-distance-scale", type=float, default=4.0)
    parser.add_argument("--camera-min-distance", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    probe_json = Path(args.probe_json).expanduser().resolve()
    if args.horizon < 2:
        raise ValueError("horizon must be at least 2")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    if args.video_frames <= 0:
        raise ValueError("video_frames must be positive")

    probe_payload = load_json(probe_json)
    if probe_payload.get("type") == "teacher_forcing_rollout":
        dataset_points, dataset_grippers, server_points, server_grippers, probe_payload = load_rollout_trajectories(probe_json)
        args.episode = int(probe_payload["episode_index"])
        args.arm_side = str(probe_payload["arm_side"])
    else:
        dataset_points, dataset_grippers = load_dataset_trajectory(
            dataset_root=dataset_root,
            episode=args.episode,
            arm_side=args.arm_side,
            frame_index=args.frame_index,
            horizon=args.horizon,
        )
        server_points, server_grippers, probe_payload = load_probe_trajectory(
            probe_json=probe_json,
            arm_side=args.arm_side,
            horizon=args.horizon,
        )

    model = mj.MjModel.from_xml_path(str(prepare_g1d_mobile_scene(use_dex1=True)))
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    dataset_display_points, server_display_points = display_points_in_mujoco_world(
        dataset_points=dataset_points,
        server_points=server_points,
        model=model,
        data=data,
        arm_side=args.arm_side,
    )

    all_points = np.vstack([dataset_display_points, server_display_points])
    center = np.mean(all_points, axis=0)
    center[2] += 0.18
    radius = float(np.max(np.linalg.norm(all_points - np.mean(all_points, axis=0), axis=1)) + 0.25)
    camera = set_free_camera(
        model,
        data,
        center=center,
        radius=radius,
        distance_scale=args.camera_distance_scale,
        min_distance=args.camera_min_distance,
    )

    renderer = mj.Renderer(model, height=int(args.height), width=int(args.width))
    if probe_payload.get("type") == "teacher_forcing_rollout":
        title = (
            f"UMI rollout episode={args.episode:06d} n={len(probe_payload['samples'])} "
            f"layout={probe_payload.get('image_layout', 'unknown')} stride={probe_payload.get('stride')}"
        )
    else:
        title = (
            f"UMI probe episode={args.episode:06d} frame={args.frame_index} "
            f"layout={probe_payload.get('image_layout', 'unknown')}"
        )

    preview = render_frame(
        renderer,
        model,
        data,
        camera,
        dataset_display_points,
        server_display_points,
        dataset_grippers,
        server_grippers,
        probe_json,
        title,
        progress=1.0,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    prefix = args.output_prefix or f"umi_probe_ep{args.episode:06d}_frame{args.frame_index}_{args.arm_side}"
    png_path = output_dir / f"{prefix}.png"
    mp4_path = output_dir / f"{prefix}.mp4"
    write_png(png_path, preview)

    frames = [
        render_frame(
            renderer,
            model,
            data,
            camera,
            dataset_display_points,
            server_display_points,
            dataset_grippers,
            server_grippers,
            probe_json,
            title,
            progress=idx / max(1, args.video_frames - 1),
        )
        for idx in range(args.video_frames)
    ]
    write_mp4(mp4_path, frames, fps=args.fps)

    print(f"dataset_start={dataset_points[0].tolist()} dataset_end={dataset_points[-1].tolist()}")
    print(f"server_start={server_points[0].tolist()} server_end={server_points[-1].tolist()}")
    print(f"wrote_png={png_path}")
    print(f"wrote_mp4={mp4_path}")


if __name__ == "__main__":
    main()
