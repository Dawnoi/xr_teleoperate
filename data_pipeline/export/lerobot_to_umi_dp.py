#!/usr/bin/env python3
"""Export LeRobot v2 episodes to a UMI/DP-friendly frame folder format.

The exporter preserves the recorded sample order:

    parquet row i == mp4 decoded frame i == alignment jsonl row i

The MP4 files do not carry the real collection wall-clock timestamp per
frame, so PNG names are derived from the alignment sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

CAMERA_SLOTS = ("cam0", "cam1", "cam2")
CHUNK_NAME = "chunk-000"
TCP_COLUMNS = {
    "cmd": {
        "left": "observation.fk.cmd.left.gripper_flange",
        "right": "observation.fk.cmd.right.gripper_flange",
    },
    "fb": {
        "left": "observation.fk.fb.left.gripper_flange",
        "right": "observation.fk.fb.right.gripper_flange",
    },
}
GRIPPER_COLUMNS = {
    "action": "action",
    "state": "observation.state",
}


@dataclass(frozen=True)
class EpisodeData:
    episode_index: int
    timestamps: List[float]
    frame_indices: List[int]
    left_tcp: List[List[float]]
    right_tcp: List[List[float]]
    left_gripper: List[float]
    right_gripper: List[float]

    @property
    def length(self) -> int:
        return len(self.frame_indices)


def require_finite_float(value: Any, label: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} contains NaN or Inf")
    return out


def finite_vector(values: Any, expected_len: int, label: str) -> List[float]:
    if values is None:
        raise ValueError(f"{label} is missing")
    if not isinstance(values, (list, tuple)):
        try:
            values = list(values)
        except Exception as exc:
            raise ValueError(f"{label} must be a sequence") from exc
    if len(values) != expected_len:
        raise ValueError(f"{label} expected length {expected_len}, got {len(values)}")
    return [require_finite_float(value, f"{label}[{idx}]") for idx, value in enumerate(values)]


def format_wall_time_ns(wall_time_ns: int, *, timezone_name: str = "local") -> str:
    ns = int(wall_time_ns)
    seconds = ns // 1_000_000_000
    millis = (ns % 1_000_000_000) // 1_000_000
    if timezone_name == "utc":
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    else:
        dt = datetime.fromtimestamp(seconds)
    return f"{dt:%Y%m%d_%H%M%S}_{millis:03d}"


def png_name(frame_index: int, wall_time_ns: int, *, timezone_name: str = "local") -> str:
    return f"{int(frame_index):06d}_{format_wall_time_ns(wall_time_ns, timezone_name=timezone_name)}.png"


def rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> List[float]:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5
    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("invalid quaternion norm")
    return [qx / norm, qy / norm, qz / norm, qw / norm]


def pose6_to_xyz_quat_xyzw(pose6: Sequence[float]) -> Dict[str, List[float]]:
    values = finite_vector(pose6, 6, "pose6")
    quat = rpy_to_quat_xyzw(values[3], values[4], values[5])
    return {
        "xyz": values[:3],
        "quat_xyzw": quat,
    }


def episode_parquet_path(lerobot_root: Path, episode_index: int) -> Path:
    return lerobot_root / "data" / CHUNK_NAME / f"episode_{int(episode_index):06d}.parquet"


def episode_alignment_path(lerobot_root: Path, episode_index: int) -> Path:
    return lerobot_root / "meta" / "alignment" / f"episode_{int(episode_index):06d}.jsonl"


def episode_video_path(lerobot_root: Path, episode_index: int, slot: str) -> Path:
    return (
        lerobot_root
        / "videos"
        / CHUNK_NAME
        / f"observation.images.{slot}"
        / f"episode_{int(episode_index):06d}.mp4"
    )


def list_episode_indices(lerobot_root: Path) -> List[int]:
    data_dir = lerobot_root / "data" / CHUNK_NAME
    if not data_dir.is_dir():
        raise FileNotFoundError(f"LeRobot data directory not found: {data_dir}")
    indices = []
    for path in sorted(data_dir.glob("episode_*.parquet")):
        stem = path.stem
        try:
            indices.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    if not indices:
        raise FileNotFoundError(f"no episode_*.parquet files found in {data_dir}")
    return indices


def read_episode_data(
    root: Path,
    episode_index: int,
    tcp_source: str,
    gripper_source: str,
) -> EpisodeData:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read LeRobot parquet files") from exc

    if tcp_source not in TCP_COLUMNS:
        raise ValueError(f"tcp_source must be one of {sorted(TCP_COLUMNS)}, got {tcp_source!r}")
    if gripper_source not in GRIPPER_COLUMNS:
        raise ValueError(f"gripper_source must be one of {sorted(GRIPPER_COLUMNS)}, got {gripper_source!r}")

    parquet_path = episode_parquet_path(root, episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(f"episode parquet not found: {parquet_path}")

    left_tcp_column = TCP_COLUMNS[tcp_source]["left"]
    right_tcp_column = TCP_COLUMNS[tcp_source]["right"]
    gripper_column = GRIPPER_COLUMNS[gripper_source]
    columns = [
        "timestamp",
        "frame_index",
        left_tcp_column,
        right_tcp_column,
        gripper_column,
    ]
    schema_names = set(pq.read_schema(parquet_path).names)
    missing = [column for column in columns if column not in schema_names]
    if missing:
        raise KeyError(f"missing required column(s) in {parquet_path}: {', '.join(missing)}")

    table = pq.read_table(parquet_path, columns=columns)
    rows = table.to_pydict()
    length = table.num_rows
    if length <= 0:
        raise ValueError(f"episode has no rows: {parquet_path}")

    timestamps = [require_finite_float(value, f"timestamp[{idx}]") for idx, value in enumerate(rows["timestamp"])]
    frame_indices = [int(value) for value in rows["frame_index"]]
    expected_indices = list(range(length))
    if frame_indices != expected_indices:
        raise ValueError(f"frame_index must be contiguous 0..{length - 1}, got {frame_indices[:5]}...")

    left_tcp = [finite_vector(value, 6, f"{left_tcp_column}[{idx}]") for idx, value in enumerate(rows[left_tcp_column])]
    right_tcp = [finite_vector(value, 6, f"{right_tcp_column}[{idx}]") for idx, value in enumerate(rows[right_tcp_column])]
    gripper_vectors = [finite_vector(value, 16, f"{gripper_column}[{idx}]") for idx, value in enumerate(rows[gripper_column])]
    left_gripper = [float(value[7]) for value in gripper_vectors]
    right_gripper = [float(value[15]) for value in gripper_vectors]

    return EpisodeData(
        episode_index=int(episode_index),
        timestamps=timestamps,
        frame_indices=frame_indices,
        left_tcp=left_tcp,
        right_tcp=right_tcp,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"jsonl file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"jsonl row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def read_alignment_rows(root: Path, episode_index: int, expected_len: int) -> List[Dict[str, Any]]:
    path = episode_alignment_path(root, episode_index)
    rows = read_jsonl(path)
    if len(rows) != expected_len:
        raise ValueError(f"alignment row count mismatch for episode {episode_index}: {len(rows)} != {expected_len}")
    for idx, row in enumerate(rows):
        frame_index = row.get("frame_index")
        if int(frame_index) != idx:
            raise ValueError(f"alignment frame_index mismatch at row {idx}: {frame_index!r}")
    return rows


def wall_time_for_row(row: Dict[str, Any], slot: str, timestamp_source: str) -> int:
    if timestamp_source == "sample":
        value = row.get("sample_wall_time_ns")
    elif timestamp_source == "camera":
        cameras = row.get("cameras") or {}
        camera_meta = cameras.get(slot) or {}
        value = camera_meta.get("host_recv_wall_time_ns")
        if value is None:
            value = row.get("sample_wall_time_ns")
    else:
        raise ValueError(f"timestamp_source must be 'sample' or 'camera', got {timestamp_source!r}")
    if value is None:
        raise ValueError(f"missing wall-clock timestamp for {slot} frame_index={row.get('frame_index')}")
    return int(value)


def decode_video_frames(path: Path, expected_frames: int) -> List[Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python/cv2 is required to decode LeRobot mp4 videos") from exc

    if not path.exists():
        raise FileNotFoundError(f"video file not found: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video: {path}")
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if len(frames) != expected_frames:
            raise ValueError(f"video frame count mismatch for {path}: {len(frames)} != {expected_frames}")
        return frames
    finally:
        capture.release()


def write_png(path: Path, frame: Any) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python/cv2 is required to write PNG files") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"failed to write PNG: {path}")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def build_tcp_payload(
    episode: EpisodeData,
    side: str,
    poses: Sequence[Sequence[float]],
    alignment_rows: Sequence[Dict[str, Any]],
    tcp_source: str,
    timezone_name: str,
) -> Dict[str, Any]:
    items = []
    for idx, pose6 in enumerate(poses):
        wall_time_ns = wall_time_for_row(alignment_rows[idx], "cam0", "sample")
        pose = pose6_to_xyz_quat_xyzw(pose6)
        items.append(
            {
                "frame_index": int(episode.frame_indices[idx]),
                "timestamp": float(episode.timestamps[idx]),
                "wall_time_ns": int(wall_time_ns),
                "timestamp_name": format_wall_time_ns(wall_time_ns, timezone_name=timezone_name),
                "xyz": pose["xyz"],
                "quat_xyzw": pose["quat_xyzw"],
            }
        )
    return {
        "schema_version": 1,
        "source": "lerobot_v2",
        "side": side,
        "pose_source": TCP_COLUMNS[tcp_source][side],
        "timestamp_source": "sample_wall_time_ns",
        "rotation": "quat_xyzw",
        "items": items,
    }


def build_gripper_payload(
    episode: EpisodeData,
    side: str,
    positions: Sequence[float],
    alignment_rows: Sequence[Dict[str, Any]],
    gripper_source: str,
    timezone_name: str,
) -> Dict[str, Any]:
    index = 7 if side == "left" else 15
    items = []
    for idx, position in enumerate(positions):
        wall_time_ns = wall_time_for_row(alignment_rows[idx], "cam0", "sample")
        items.append(
            {
                "frame_index": int(episode.frame_indices[idx]),
                "timestamp": float(episode.timestamps[idx]),
                "wall_time_ns": int(wall_time_ns),
                "timestamp_name": format_wall_time_ns(wall_time_ns, timezone_name=timezone_name),
                "position": require_finite_float(position, f"{side}_gripper[{idx}]"),
            }
        )
    return {
        "schema_version": 1,
        "source": "lerobot_v2",
        "side": side,
        "position_source": f"{GRIPPER_COLUMNS[gripper_source]}[{index}]",
        "timestamp_source": "sample_wall_time_ns",
        "items": items,
    }


def prepare_episode_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output episode directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def export_episode(
    lerobot_root: Path,
    output_root: Path,
    episode_index: int,
    tcp_source: str,
    gripper_source: str,
    timestamp_source: str,
    timezone_name: str,
    overwrite: bool,
) -> Dict[str, Any]:
    lerobot_root = Path(lerobot_root)
    output_root = Path(output_root)
    episode = read_episode_data(lerobot_root, episode_index, tcp_source, gripper_source)
    alignment_rows = read_alignment_rows(lerobot_root, episode_index, episode.length)

    episode_dir = output_root / f"episode_{episode_index:06d}"
    prepare_episode_output_dir(episode_dir, overwrite=overwrite)

    for slot in CAMERA_SLOTS:
        video_path = episode_video_path(lerobot_root, episode_index, slot)
        frames = decode_video_frames(video_path, episode.length)
        for idx, frame in enumerate(frames):
            wall_time_ns = wall_time_for_row(alignment_rows[idx], slot, timestamp_source)
            out_path = episode_dir / "camera" / slot / png_name(
                episode.frame_indices[idx],
                wall_time_ns,
                timezone_name=timezone_name,
            )
            write_png(out_path, frame)

    write_json_atomic(
        episode_dir / "tcp" / "left.json",
        build_tcp_payload(
            episode,
            "left",
            episode.left_tcp,
            alignment_rows,
            tcp_source=tcp_source,
            timezone_name=timezone_name,
        ),
    )
    write_json_atomic(
        episode_dir / "tcp" / "right.json",
        build_tcp_payload(
            episode,
            "right",
            episode.right_tcp,
            alignment_rows,
            tcp_source=tcp_source,
            timezone_name=timezone_name,
        ),
    )
    write_json_atomic(
        episode_dir / "gripper" / "left.json",
        build_gripper_payload(
            episode,
            "left",
            episode.left_gripper,
            alignment_rows,
            gripper_source=gripper_source,
            timezone_name=timezone_name,
        ),
    )
    write_json_atomic(
        episode_dir / "gripper" / "right.json",
        build_gripper_payload(
            episode,
            "right",
            episode.right_gripper,
            alignment_rows,
            gripper_source=gripper_source,
            timezone_name=timezone_name,
        ),
    )
    write_json_atomic(
        episode_dir / "meta.json",
        {
            "schema_version": 1,
            "source": "lerobot_v2",
            "episode_index": int(episode_index),
            "length": int(episode.length),
            "tcp_source": tcp_source,
            "gripper_source": gripper_source,
            "camera_timestamp_source": timestamp_source,
            "tcp_gripper_timestamp_source": "sample_wall_time_ns",
            "timezone": timezone_name,
            "quaternion_order": "xyzw",
            "order_contract": "parquet row i == mp4 decoded frame i == alignment jsonl row i",
            "camera_keys": {
                "cam0": "observation.images.cam0",
                "cam1": "observation.images.cam1",
                "cam2": "observation.images.cam2",
            },
        },
    )
    return {
        "episode_index": int(episode_index),
        "frames": int(episode.length),
        "output_dir": str(episode_dir),
    }


def parse_episode_indices(values: Sequence[str] | None, lerobot_root: Path) -> List[int]:
    if not values:
        return list_episode_indices(lerobot_root)
    indices = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            indices.append(int(part))
    if not indices:
        raise ValueError("--episodes did not include any episode indices")
    return sorted(dict.fromkeys(indices))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v2 parquet/mp4 episodes to UMI/DP camera/tcp/gripper folders.",
    )
    parser.add_argument("--lerobot-root", required=True, type=Path, help="Input LeRobot v2 dataset root.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output UMI/DP dataset root.")
    parser.add_argument("--episodes", nargs="*", help="Episode indices to export, e.g. 0 1 2 or 0,1,2. Default: all.")
    parser.add_argument("--tcp-source", choices=sorted(TCP_COLUMNS), default="cmd", help="TCP source: cmd action FK or fb state FK.")
    parser.add_argument(
        "--gripper-source",
        choices=sorted(GRIPPER_COLUMNS),
        default="action",
        help="Gripper source: action command or observation.state feedback.",
    )
    parser.add_argument(
        "--timestamp-source",
        choices=("sample", "camera"),
        default="sample",
        help="PNG timestamp source. sample gives one timestamp shared by action/TCP/all cameras; camera uses per-camera receive wall time.",
    )
    parser.add_argument(
        "--timezone",
        choices=("local", "utc"),
        default="local",
        help="Timezone used when formatting wall-clock PNG timestamp names.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output episode directories.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    lerobot_root = args.lerobot_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    episode_indices = parse_episode_indices(args.episodes, lerobot_root)
    summaries = []
    for episode_index in episode_indices:
        summary = export_episode(
            lerobot_root=lerobot_root,
            output_root=output_root,
            episode_index=episode_index,
            tcp_source=args.tcp_source,
            gripper_source=args.gripper_source,
            timestamp_source=args.timestamp_source,
            timezone_name=args.timezone,
            overwrite=bool(args.overwrite),
        )
        summaries.append(summary)
        print(
            f"exported episode_{episode_index:06d}: "
            f"frames={summary['frames']} output={summary['output_dir']}"
        )

    write_json_atomic(
        output_root / "export_summary.json",
        {
            "schema_version": 1,
            "source_root": str(lerobot_root),
            "episodes": summaries,
            "total_episodes": len(summaries),
            "total_frames": sum(int(summary["frames"]) for summary in summaries),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
