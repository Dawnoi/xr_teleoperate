#!/usr/bin/env python3
"""Build legacy pelvis-frame EEF pose manifest from a mobile LeRobot export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


POSE_KEYS = (
    ("fb", "left", "gripper_flange"),
    ("fb", "right", "gripper_flange"),
    ("cmd", "left", "gripper_flange"),
    ("cmd", "right", "gripper_flange"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument("--minimum-source-episode", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_pose(entry: dict, key: tuple[str, str, str], *, source: Path, line_index: int) -> list[float]:
    value: object = entry
    for name in key:
        if not isinstance(value, dict) or name not in value:
            raise KeyError(f"{source}:{line_index}: missing pose {'/'.join(key)}")
        value = value[name]
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f"{source}:{line_index}: {'/'.join(key)} must be six finite values")
    return pose.tolist()


def main() -> None:
    args = parse_args()
    export_root = args.export_root.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    summary_path = export_root / "export_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = summary.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{summary_path}: expected episodes list")

    selected: list[tuple[int, str, int]] = []
    for entry in episodes:
        if not isinstance(entry, dict):
            raise ValueError(f"{summary_path}: episode entry must be an object")
        source_episode = entry.get("source_episode")
        export_index = entry.get("episode_index")
        if not isinstance(source_episode, str) or not isinstance(export_index, int):
            raise ValueError(f"{summary_path}: episode requires source_episode and episode_index")
        if not source_episode.startswith("episode_"):
            raise ValueError(f"{summary_path}: invalid source episode name {source_episode!r}")
        raw_index = int(source_episode.split("_")[-1])
        if raw_index >= args.minimum_source_episode:
            selected.append((export_index, source_episode, raw_index))
    selected.sort()
    if not selected:
        raise ValueError("selection contains no exported episodes")
    selected_indices = np.asarray([item[0] for item in selected], dtype=np.int64)
    if not np.array_equal(selected_indices, np.arange(selected_indices[0], selected_indices[-1] + 1)):
        raise ValueError("selected export episode indices must be contiguous")

    frame_offsets = [0]
    feedback_left: list[list[float]] = []
    feedback_right: list[list[float]] = []
    command_left: list[list[float]] = []
    command_right: list[list[float]] = []
    for export_index, _source_episode, _raw_index in selected:
        pose_path = export_root / "extras/raw_pose" / f"chunk-{export_index // 1000:03d}" / f"episode_{export_index:06d}.jsonl"
        if not pose_path.is_file():
            raise FileNotFoundError(f"missing raw pose sidecar: {pose_path}")
        line_count = 0
        with pose_path.open(encoding="utf-8") as stream:
            for line_index, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"{pose_path}:{line_index}: blank line is not allowed")
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError(f"{pose_path}:{line_index}: expected JSON object")
                if entry.get("episode_index") != export_index or entry.get("frame_index") != line_count:
                    raise ValueError(f"{pose_path}:{line_index}: unexpected episode/frame index")
                feedback_left.append(require_pose(entry, POSE_KEYS[0], source=pose_path, line_index=line_index))
                feedback_right.append(require_pose(entry, POSE_KEYS[1], source=pose_path, line_index=line_index))
                command_left.append(require_pose(entry, POSE_KEYS[2], source=pose_path, line_index=line_index))
                command_right.append(require_pose(entry, POSE_KEYS[3], source=pose_path, line_index=line_index))
                line_count += 1
        if line_count == 0:
            raise ValueError(f"{pose_path}: empty pose sidecar")
        frame_offsets.append(frame_offsets[-1] + line_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_episode_index=selected_indices,
        source_raw_episode_index=np.asarray([item[2] for item in selected], dtype=np.int64),
        frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
        feedback_left_pose6=np.asarray(feedback_left, dtype=np.float64),
        feedback_right_pose6=np.asarray(feedback_right, dtype=np.float64),
        command_left_pose6=np.asarray(command_left, dtype=np.float64),
        command_right_pose6=np.asarray(command_right, dtype=np.float64),
    )
    print(
        f"created {output_path}: export episodes={selected_indices[0]}..{selected_indices[-1]} "
        f"raw episodes={selected[0][2]}..{selected[-1][2]} frames={frame_offsets[-1]}"
    )


if __name__ == "__main__":
    main()
