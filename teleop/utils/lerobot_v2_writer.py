from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pinocchio as pin
import pyarrow as pa
import pyarrow.parquet as pq

import logging_mp

from .episode_writer import ZMQRawCameraReceiver, canonical_color_key
from .rerun_visualizer import RerunLogger

logger_mp = logging_mp.getLogger(__name__)

CHUNK_INDEX = 0
CHUNK_NAME = f"chunk-{CHUNK_INDEX:03d}"
CAMERA_SLOTS = ("cam0", "cam1", "cam2")
CAMERA_SOURCE_ALIASES = {
    "cam0": ("head",),
    "cam1": ("left_wrist", "wrist_left"),
    "cam2": ("right_wrist", "wrist_right"),
}
CAMERA_SLOT_PRIMARY_SOURCE = {
    "cam0": "head",
    "cam1": "left_wrist",
    "cam2": "right_wrist",
}
CAMERA_SOURCE_TO_SLOT = {
    alias: slot for slot, aliases in CAMERA_SOURCE_ALIASES.items() for alias in aliases
}
CAMERA_SLOT_TO_SOURCE = {
    "cam0": "head",
    "cam1": "left_wrist/wrist_left",
    "cam2": "right_wrist/wrist_right",
}
DEFAULT_VIDEO_BACKEND = "opencv"
DEFAULT_OPENCV_FOURCCS = ("MJPG", "mp4v", "avc1")
DEFAULT_VIDEO_CRF = 18
DEFAULT_VIDEO_PRESET = "veryfast"
DEFAULT_VIDEO_PROFILE = "baseline"
DEFAULT_VIDEO_LEVEL = "3.0"
ALIGNMENT_SIDECAR_SCHEMA_VERSION = 1
CONTROL_SIDECAR_SCHEMA_VERSION = 1

LEFT_ARM_LINK_NAMES = [
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
]
RIGHT_ARM_LINK_NAMES = [
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]
ARM_JOINT_LABELS = [f"joint{i}" for i in range(1, 8)]
ARM_GROUPS = ("fb", "cmd")
SIDE_GROUPS = ("left", "right")
STATE_VECTOR_SIZE = 16
ARM_VECTOR_SIZE = 14
EE_VECTOR_SIZE = 1


def _normalize_text(value: Any, fallback: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else fallback


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_dump_atomic(path: str, payload: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _jsonl_dump_atomic(path: str, rows: List[Dict[str, Any]]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp_path, path)


def _dataset_relpath(path: str, dataset_root: str) -> str:
    return os.path.relpath(path, dataset_root).replace(os.sep, "/")


def _decode_fourcc(value: float | int) -> str:
    try:
        fourcc_int = int(value)
    except Exception:
        return ""
    chars = [
        chr((fourcc_int >> (8 * i)) & 0xFF)
        for i in range(4)
    ]
    decoded = "".join(chars)
    if decoded.strip("\x00"):
        return decoded
    return ""


def _struct_type() -> pa.DataType:
    return pa.struct(
        [
            pa.field("path", pa.string()),
            pa.field("timestamp", pa.float64()),
        ]
    )


def _finite_list(values: Any, expected_len: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != expected_len:
        raise ValueError(f"{label} expected length {expected_len}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or Inf")
    return arr


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_value(nested_value) for nested_value in value]
    if isinstance(value, list):
        return [_json_safe_value(nested_value) for nested_value in value]
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _pose6_from_se3(se3: pin.SE3) -> List[float]:
    translation = np.asarray(se3.translation, dtype=float).reshape(3)
    rpy = np.asarray(pin.rpy.matrixToRpy(np.asarray(se3.rotation, dtype=float)), dtype=float).reshape(3)
    return [float(translation[0]), float(translation[1]), float(translation[2]), float(rpy[0]), float(rpy[1]), float(rpy[2])]


def _make_feature_spec() -> Dict[str, Dict[str, Any]]:
    features: Dict[str, Dict[str, Any]] = OrderedDict()
    features["timestamp"] = {"dtype": "float64", "shape": []}
    features["frame_index"] = {"dtype": "int64", "shape": []}
    features["episode_index"] = {"dtype": "int64", "shape": []}
    features["index"] = {"dtype": "int64", "shape": []}
    features["task_index"] = {"dtype": "int64", "shape": []}
    features["observation.state"] = {"dtype": "list<float64>", "length": STATE_VECTOR_SIZE}
    features["action"] = {"dtype": "list<float64>", "length": STATE_VECTOR_SIZE}

    for group in ARM_GROUPS:
        for side in SIDE_GROUPS:
            for label in ARM_JOINT_LABELS:
                key = f"observation.fk.fb.{side}.{label}" if group == "fb" else f"observation.fk.cmd.{side}.{label}"
                features[key] = {"dtype": "list<float64>", "length": 6}
            key = f"observation.fk.fb.{side}.gripper_flange" if group == "fb" else f"observation.fk.cmd.{side}.gripper_flange"
            features[key] = {"dtype": "list<float64>", "length": 6}

    for slot in CAMERA_SLOTS:
        features[f"observation.images.{slot}"] = {
            "dtype": "struct",
            "fields": {
                "path": "string",
                "timestamp": "float64",
            },
        }
    return features


def _build_parquet_schema() -> pa.Schema:
    fields: List[pa.Field] = [
        pa.field("timestamp", pa.float64()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("observation.state", pa.list_(pa.float64())),
        pa.field("action", pa.list_(pa.float64())),
    ]
    for group in ARM_GROUPS:
        for side in SIDE_GROUPS:
            for label in ARM_JOINT_LABELS:
                key = f"observation.fk.fb.{side}.{label}" if group == "fb" else f"observation.fk.cmd.{side}.{label}"
                fields.append(pa.field(key, pa.list_(pa.float64())))
            key = f"observation.fk.fb.{side}.gripper_flange" if group == "fb" else f"observation.fk.cmd.{side}.gripper_flange"
            fields.append(pa.field(key, pa.list_(pa.float64())))
    for slot in CAMERA_SLOTS:
        fields.append(pa.field(f"observation.images.{slot}", _struct_type()))
    return pa.schema(fields)


def _pick_parquet_compression() -> Optional[str]:
    for codec in ("zstd", "snappy"):
        try:
            if pa.Codec.is_available(codec):
                return codec
        except Exception:
            continue
    return None


def _column_values_are_finite(column: pa.ChunkedArray) -> bool:
    for chunk in column.chunks:
        values = chunk.to_pylist()
        for value in values:
            if value is None:
                return False
            if isinstance(value, list):
                arr = np.asarray(value, dtype=float)
                if not np.all(np.isfinite(arr)):
                    return False
            elif isinstance(value, dict):
                for nested_value in value.values():
                    if isinstance(nested_value, (int, float, np.floating, np.integer)):
                        if not np.isfinite(float(nested_value)):
                            return False
            elif isinstance(value, (int, float, np.floating, np.integer)):
                if not np.isfinite(float(value)):
                    return False
    return True


def _video_encoder_settings() -> Dict[str, Any]:
    backend = (os.environ.get("LEROBOT_VIDEO_BACKEND", "").strip() or DEFAULT_VIDEO_BACKEND).lower()
    if backend in ("cv2", "opencv"):
        backend = "opencv"
    if backend not in ("opencv", "ffmpeg"):
        raise ValueError(f"LEROBOT_VIDEO_BACKEND must be 'opencv' or 'ffmpeg', got {backend!r}")

    if backend == "opencv":
        fourcc_raw = os.environ.get("LEROBOT_OPENCV_FOURCC", "").strip()
        fourccs = [fourcc_raw] if fourcc_raw else list(DEFAULT_OPENCV_FOURCCS)
        fourccs = [str(fourcc).strip() for fourcc in fourccs if str(fourcc).strip()]
        for fourcc in fourccs:
            if len(fourcc) != 4:
                raise ValueError(f"OpenCV fourcc must be exactly 4 characters, got {fourcc!r}")
        if not fourccs:
            fourccs = list(DEFAULT_OPENCV_FOURCCS)
        return {
            "backend": "opencv",
            "codec": fourccs[0],
            "container": "mp4",
            "fourccs": fourccs,
        }

    ffmpeg_bin = os.environ.get("LEROBOT_FFMPEG_BIN", "").strip() or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg binary not found. Install ffmpeg with libx264 in the active environment or set LEROBOT_FFMPEG_BIN."
        )

    crf_raw = os.environ.get("LEROBOT_VIDEO_CRF", "").strip()
    if crf_raw:
        crf = int(crf_raw)
    else:
        crf = DEFAULT_VIDEO_CRF
    if crf < 0 or crf > 51:
        raise ValueError(f"LEROBOT_VIDEO_CRF must be within [0, 51], got {crf}")

    preset = os.environ.get("LEROBOT_VIDEO_PRESET", "").strip() or DEFAULT_VIDEO_PRESET
    if not preset:
        preset = DEFAULT_VIDEO_PRESET

    profile = (os.environ.get("LEROBOT_VIDEO_PROFILE", "").strip() or DEFAULT_VIDEO_PROFILE).lower()
    if profile not in ("baseline", "main"):
        raise ValueError(f"LEROBOT_VIDEO_PROFILE must be 'baseline' or 'main', got {profile!r}")

    level = os.environ.get("LEROBOT_VIDEO_LEVEL", "").strip() or DEFAULT_VIDEO_LEVEL
    if not level:
        level = DEFAULT_VIDEO_LEVEL

    return {
        "backend": "ffmpeg",
        "codec": "libx264",
        "pix_fmt": "yuv420p",
        "crf": int(crf),
        "preset": preset,
        "profile": profile,
        "level": level,
        "ffmpeg_bin": ffmpeg_bin,
    }


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _safe_abs_ms(delta_ns: Any) -> Optional[float]:
    value = _safe_int(delta_ns)
    if value is None:
        return None
    return float(abs(value) / 1e6)


def _ms_summary(values_ns: List[int]) -> Dict[str, Any]:
    if not values_ns:
        return {"count": 0, "avg_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    values_ms = np.asarray(values_ns, dtype=float) / 1e6
    return {
        "count": int(values_ms.size),
        "avg_ms": float(np.mean(values_ms)),
        "p95_ms": float(np.percentile(values_ms, 95)),
        "max_ms": float(np.max(values_ms)),
    }


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def _abs_delta_ms_summary(values_ns: List[int]) -> Dict[str, Any]:
    values_ms = [float(abs(int(value))) / 1e6 for value in values_ns if value is not None]
    if not values_ms:
        return {
            "count": 0,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values_ms),
        "p50": _percentile(values_ms, 50),
        "p95": _percentile(values_ms, 95),
        "p99": _percentile(values_ms, 99),
        "max": float(max(values_ms)),
        "mean": float(np.mean(np.asarray(values_ms, dtype=float))),
    }


def _frame_seq_gap_summary(frame_seq_values: List[Optional[int]]) -> Dict[str, Any]:
    present = [int(value) for value in frame_seq_values if value is not None]
    gaps: List[Dict[str, int]] = []
    duplicate_count = 0
    out_of_order_count = 0
    for prev, cur in zip(present, present[1:]):
        step = cur - prev
        if step == 0:
            duplicate_count += 1
        elif step < 0:
            out_of_order_count += 1
        elif step > 1:
            gaps.append({"prev": int(prev), "current": int(cur), "missing": int(step - 1)})
    return {
        "count": len(present),
        "missing_count": len(frame_seq_values) - len(present),
        "gap_count": len(gaps),
        "missing_frames_estimate": int(sum(gap["missing"] for gap in gaps)),
        "duplicate_count": duplicate_count,
        "out_of_order_count": out_of_order_count,
        "first": present[0] if present else None,
        "last": present[-1] if present else None,
        "gaps": gaps[:20],
    }


class _OpenCVMP4Writer:
    def __init__(
        self,
        path: str,
        size: Tuple[int, int],
        fps: float,
        fourccs: List[str],
    ):
        self.path = path
        self.width = int(size[0])
        self.height = int(size[1])
        self.fps = float(fps)
        self.fourcc = ""
        self._writer: Optional[cv2.VideoWriter] = None
        self._released = False
        self._frame_count = 0
        self._open(fourccs)

    def _open(self, fourccs: List[str]) -> None:
        last_error = ""
        for fourcc in fourccs:
            try:
                writer = cv2.VideoWriter(
                    self.path,
                    cv2.VideoWriter_fourcc(*fourcc),
                    self.fps,
                    (self.width, self.height),
                )
            except Exception as exc:
                last_error = str(exc)
                continue
            if writer.isOpened():
                self._writer = writer
                self.fourcc = str(fourcc)
                logger_mp.info(
                    "[LeRobotV2Writer] OpenCV mp4 writer opened %s fourcc=%s size=%dx%d fps=%.3f",
                    self.path,
                    self.fourcc,
                    self.width,
                    self.height,
                    self.fps,
                )
                return
            try:
                writer.release()
            except Exception:
                pass
        raise RuntimeError(
            f"OpenCV VideoWriter failed to open {self.path} with fourcc candidates {fourccs}. {last_error}"
        )

    def write(self, frame: np.ndarray) -> None:
        if self._released:
            raise RuntimeError("OpenCV mp4 writer already released")
        if self._writer is None or not self._writer.isOpened():
            raise RuntimeError("OpenCV mp4 writer is not initialized")

        frame_arr = np.ascontiguousarray(frame)
        expected_shape = (self.height, self.width, 3)
        if frame_arr.shape != expected_shape:
            raise ValueError(f"OpenCV mp4 writer expected frame shape {expected_shape}, got {frame_arr.shape}")
        if frame_arr.dtype != np.uint8:
            frame_arr = frame_arr.astype(np.uint8, copy=False)
        self._writer.write(frame_arr)
        self._frame_count += 1

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._writer is not None:
            self._writer.release()
        if self._frame_count == 0:
            raise RuntimeError("OpenCV mp4 writer released without any written frames")


class _FFmpegMP4Writer:
    def __init__(
        self,
        path: str,
        size: Tuple[int, int],
        fps: float,
        ffmpeg_bin: str,
        preset: str,
        crf: int,
        profile: str,
        level: str,
    ):
        self.path = path
        self.width = int(size[0])
        self.height = int(size[1])
        self.fps = float(fps)
        self.ffmpeg_bin = ffmpeg_bin
        self.preset = str(preset)
        self.crf = int(crf)
        self.profile = str(profile)
        self.level = str(level)
        self._process: Optional[subprocess.Popen] = None
        self._released = False
        self._frame_count = 0
        self._stderr_text = ""
        self._start_process()

    def _command(self) -> List[str]:
        return [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            f"{self.fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-profile:v",
            self.profile,
            "-level:v",
            self.level,
            "-bf",
            "0",
            "-refs",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            self.path,
        ]

    def _start_process(self) -> None:
        cmd = self._command()
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"ffmpeg binary not found at {self.ffmpeg_bin}; install ffmpeg with libx264 in the active environment."
            ) from exc
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg writer failed to create stdin pipe")
        logger_mp.info(
            "[LeRobotV2Writer] ffmpeg writer opened %s codec=libx264 profile=%s level=%s crf=%d preset=%s size=%dx%d fps=%.3f",
            self.path,
            self.profile,
            self.level,
            self.crf,
            self.preset,
            self.width,
            self.height,
            self.fps,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._released:
            raise RuntimeError("ffmpeg writer already released")
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("ffmpeg writer is not initialized")
        if self._process.poll() is not None:
            raise RuntimeError(self._build_process_error("ffmpeg process exited before frame write"))

        frame_arr = np.ascontiguousarray(frame)
        expected_shape = (self.height, self.width, 3)
        if frame_arr.shape != expected_shape:
            raise ValueError(f"ffmpeg writer expected frame shape {expected_shape}, got {frame_arr.shape}")
        if frame_arr.dtype != np.uint8:
            frame_arr = frame_arr.astype(np.uint8, copy=False)
        try:
            self._process.stdin.write(frame_arr.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(self._build_process_error("ffmpeg stdin pipe closed while writing frame")) from exc
        self._frame_count += 1

    def _build_process_error(self, prefix: str) -> str:
        stderr_text = self._stderr_text.strip()
        if stderr_text:
            return f"{prefix}: {stderr_text}"
        if self._process is not None and self._process.returncode is not None:
            return f"{prefix}: returncode={self._process.returncode}"
        return prefix

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._process is None:
            return
        try:
            if self._process.stdin is not None and not self._process.stdin.closed:
                self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            try:
                self._process.kill()
            except Exception:
                pass
            self._process.wait(timeout=30.0)
        except Exception as exc:
            try:
                self._process.kill()
            except Exception:
                pass
            raise RuntimeError(f"ffmpeg release failed: {exc}") from exc
        stderr_data = b""
        try:
            if self._process.stderr is not None:
                stderr_data = self._process.stderr.read() or b""
        except Exception:
            stderr_data = b""
        self._stderr_text = (stderr_data or b"").decode("utf-8", errors="replace")
        if self._process.returncode not in (0, None):
            raise RuntimeError(self._build_process_error("ffmpeg encoder exited with failure"))
        if self._frame_count == 0:
            raise RuntimeError("ffmpeg encoder released without any written frames")


class LeRobotV2Writer:
    """Background writer for the LeRobot v2 three-camera dataset format."""

    def __init__(
        self,
        task_dir: str,
        arm_ik,
        task_goal: Optional[str] = None,
        task_desc: Optional[str] = None,
        task_steps: Optional[str] = None,
        frequency: float = 30.0,
        image_size: Optional[List[int]] = None,
        rerun_log: bool = False,
    ):
        self.dataset_root = os.path.abspath(task_dir)
        self.frequency = float(frequency)
        self.image_size = image_size
        self.rerun_log = rerun_log
        self.task_goal = _normalize_text(task_goal, fallback=_normalize_text(task_desc, fallback=_normalize_text(task_steps, fallback="unnamed task")))
        self.task_desc = _normalize_text(task_desc)
        self.task_steps = _normalize_text(task_steps)

        if arm_ik is None or not hasattr(arm_ik, "reduced_robot"):
            raise ValueError("LeRobotV2Writer requires an arm_ik with reduced_robot.model for FK generation.")
        self._fk_model = arm_ik.reduced_robot.model
        self._fk_data = self._fk_model.createData()
        self._fk_lock = threading.Lock()
        self._fk_frame_ids = self._resolve_fk_frame_ids()

        self._parquet_schema = _build_parquet_schema()
        self._feature_spec = _make_feature_spec()

        self.meta_dir = os.path.join(self.dataset_root, "meta")
        self.alignment_dir = os.path.join(self.meta_dir, "alignment")
        self.control_dir = os.path.join(self.dataset_root, "extras", "control", CHUNK_NAME)
        self.data_dir = os.path.join(self.dataset_root, "data", CHUNK_NAME)
        self.video_root_dir = os.path.join(self.dataset_root, "videos", CHUNK_NAME)
        _safe_mkdir(self.meta_dir)
        _safe_mkdir(self.alignment_dir)
        _safe_mkdir(self.control_dir)
        _safe_mkdir(self.data_dir)
        _safe_mkdir(self.video_root_dir)
        for slot in CAMERA_SLOTS:
            _safe_mkdir(self._video_slot_dir(slot))

        self._meta_info_path = os.path.join(self.meta_dir, "info.json")
        self._meta_tasks_path = os.path.join(self.meta_dir, "tasks.jsonl")
        self._meta_episodes_path = os.path.join(self.meta_dir, "episodes.jsonl")
        self._meta_stats_path = os.path.join(self.meta_dir, "episodes_stats.jsonl")

        self._tasks: List[Dict[str, Any]] = []
        self._episodes: List[Dict[str, Any]] = []
        self._episode_stats: List[Dict[str, Any]] = []
        self._task_index_by_text: Dict[str, int] = {}
        self._load_existing_meta()

        self._state_lock = threading.Lock()
        self._item_queue: Queue = Queue()
        self._stop_worker = False
        self._save_requested = False
        self._finalizing = False
        self._current_episode_active = False
        self._current_episode_index = -1
        self._current_task_index = -1
        self._current_task_text = self.task_goal
        self._current_samples: List[Dict[str, Any]] = []
        self._current_alignment_rows: List[Dict[str, Any]] = []
        self._current_control_rows: List[Dict[str, Any]] = []
        self._current_frame_shapes: Dict[str, Tuple[int, int, int]] = {}
        self._current_failure_reason: Optional[str] = None
        self._current_episode_paths: Dict[str, Any] = {}
        self._current_first_sample_monotonic_ns: Optional[int] = None
        self._current_item_idx = -1
        self._current_runtime_stats: Dict[str, Any] = self._new_runtime_stats()
        self._current_video_writers: Dict[str, Any] = {}
        self._current_video_writer_sizes: Dict[str, Tuple[int, int]] = {}
        self._current_video_writer_fourcc: Dict[str, str] = {}
        self._current_video_frame_counts: Dict[str, int] = {slot: 0 for slot in CAMERA_SLOTS}
        self.online_logger = None
        self._active_task_registered = False
        self.is_available = True
        self._next_episode_index = self._compute_next_episode_index()
        self._global_frame_index_next = sum(int(ep.get("length", 0)) for ep in self._episodes)
        if not all(
            os.path.exists(path)
            for path in (
                self._meta_info_path,
                self._meta_tasks_path,
                self._meta_episodes_path,
                self._meta_stats_path,
            )
        ):
            self._write_meta_files()
        self.worker_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.worker_thread.start()
        logger_mp.info(
            "[LeRobotV2Writer] ready at %s (episodes=%d, frames=%d, tasks=%d)",
            self.dataset_root,
            len(self._episodes),
            self._global_frame_index_next,
            len(self._tasks),
        )
        if self.rerun_log:
            logger_mp.info("[LeRobotV2Writer] Rerun live logging enabled; logger will be created per episode.")

    def _video_slot_dir(self, slot: str) -> str:
        return os.path.join(self.video_root_dir, f"observation.images.{slot}")

    def _video_final_path(self, episode_index: int, slot: str) -> str:
        return os.path.join(self._video_slot_dir(slot), f"episode_{episode_index:06d}.mp4")

    def _video_temp_path(self, episode_index: int, slot: str) -> str:
        return os.path.join(self._video_slot_dir(slot), f"episode_{episode_index:06d}.partial.mp4")

    def _parquet_final_path(self, episode_index: int) -> str:
        return os.path.join(self.data_dir, f"episode_{episode_index:06d}.parquet")

    def _parquet_temp_path(self, episode_index: int) -> str:
        return os.path.join(self.data_dir, f"episode_{episode_index:06d}.partial.parquet")

    def _alignment_final_path(self, episode_index: int) -> str:
        return os.path.join(self.alignment_dir, f"episode_{episode_index:06d}.jsonl")

    def _alignment_temp_path(self, episode_index: int) -> str:
        return os.path.join(self.alignment_dir, f"episode_{episode_index:06d}.partial.jsonl")

    def _control_final_path(self, episode_index: int) -> str:
        return os.path.join(self.control_dir, f"episode_{episode_index:06d}.jsonl")

    def _control_temp_path(self, episode_index: int) -> str:
        return os.path.join(self.control_dir, f"episode_{episode_index:06d}.partial.jsonl")

    def _resolve_fk_frame_ids(self) -> Dict[str, int]:
        frame_map = OrderedDict()
        for idx, frame_name in enumerate(LEFT_ARM_LINK_NAMES, start=1):
            key = f"observation.fk.fb.left.joint{idx}"
            frame_map[key] = frame_name
        frame_map["observation.fk.fb.left.gripper_flange"] = "L_ee"
        for idx, frame_name in enumerate(RIGHT_ARM_LINK_NAMES, start=1):
            key = f"observation.fk.fb.right.joint{idx}"
            frame_map[key] = frame_name
        frame_map["observation.fk.fb.right.gripper_flange"] = "R_ee"
        for idx, frame_name in enumerate(LEFT_ARM_LINK_NAMES, start=1):
            key = f"observation.fk.cmd.left.joint{idx}"
            frame_map[key] = frame_name
        frame_map["observation.fk.cmd.left.gripper_flange"] = "L_ee"
        for idx, frame_name in enumerate(RIGHT_ARM_LINK_NAMES, start=1):
            key = f"observation.fk.cmd.right.joint{idx}"
            frame_map[key] = frame_name
        frame_map["observation.fk.cmd.right.gripper_flange"] = "R_ee"

        resolved: Dict[str, int] = {}
        for key, frame_name in frame_map.items():
            frame_id = self._fk_model.getFrameId(frame_name)
            if frame_id < 0 or frame_id >= self._fk_model.nframes:
                raise RuntimeError(f"FK frame '{frame_name}' required by {key} was not found in the reduced model.")
            if self._fk_model.frames[frame_id].name != frame_name:
                raise RuntimeError(f"FK frame '{frame_name}' lookup returned mismatched frame id {frame_id}.")
            resolved[key] = int(frame_id)
        return resolved

    def _load_existing_meta(self) -> None:
        if os.path.exists(self._meta_tasks_path):
            with open(self._meta_tasks_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    task_text = _normalize_text(row.get("task"))
                    task_index = int(row.get("task_index", len(self._tasks)))
                    self._tasks.append({"task_index": task_index, "task": task_text})
                    self._task_index_by_text[task_text] = task_index

        if os.path.exists(self._meta_episodes_path):
            with open(self._meta_episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    episode_index = int(row.get("episode_index", len(self._episodes)))
                    tasks = row.get("tasks", [])
                    if not tasks:
                        tasks = [self.task_goal]
                    self._episodes.append(
                        {
                            "episode_index": episode_index,
                            "tasks": tasks,
                            "length": int(row.get("length", 0)),
                        }
                    )

        if os.path.exists(self._meta_stats_path):
            with open(self._meta_stats_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self._episode_stats.append(row)

        if not self._episodes:
            episode_lengths = self._scan_existing_episode_lengths()
            for episode_index, length in episode_lengths:
                self._episodes.append(
                    {
                        "episode_index": episode_index,
                        "tasks": [self.task_goal],
                        "length": length,
                    }
                )

        if not self._tasks and self._episodes:
            self._tasks.append({"task_index": 0, "task": self.task_goal})
            self._task_index_by_text[self.task_goal] = 0

        if not self._episode_stats and self._episodes:
            for episode in self._episodes:
                self._episode_stats.append({"episode_index": int(episode["episode_index"]), "stats": {}})

    def _scan_existing_episode_lengths(self) -> List[Tuple[int, int]]:
        results: List[Tuple[int, int]] = []
        if not os.path.isdir(self.data_dir):
            return results
        pattern = re.compile(r"^episode_(\d{6})\.parquet$")
        for name in sorted(os.listdir(self.data_dir)):
            match = pattern.match(name)
            if not match:
                continue
            episode_index = int(match.group(1))
            path = os.path.join(self.data_dir, name)
            try:
                metadata = pq.read_metadata(path)
                results.append((episode_index, int(metadata.num_rows)))
            except Exception as exc:
                logger_mp.warning("[LeRobotV2Writer] failed to read parquet metadata from %s: %s", path, exc)
        return results

    def _compute_next_episode_index(self) -> int:
        max_index = -1
        for episode in self._episodes:
            max_index = max(max_index, int(episode.get("episode_index", -1)))
        for episode_index, _ in self._scan_existing_episode_lengths():
            max_index = max(max_index, episode_index)
        return max_index + 1

    def _task_index_for_text(self, task_text: str) -> Tuple[int, bool]:
        normalized = _normalize_text(task_text, fallback=self.task_goal)
        if normalized in self._task_index_by_text:
            return self._task_index_by_text[normalized], True
        return len(self._tasks), False

    def _commit_task_if_needed(self, task_text: str, task_index: int, already_registered: bool) -> None:
        if already_registered:
            return
        row = {"task_index": int(task_index), "task": task_text}
        self._tasks.append(row)
        self._task_index_by_text[task_text] = int(task_index)

    def _build_info_payload(self) -> Dict[str, Any]:
        total_frames = sum(int(ep.get("length", 0)) for ep in self._episodes)
        total_episodes = len(self._episodes)
        total_tasks = len(self._tasks)
        return {
            "codebase_version": "v2.1",
            "robot_type": "nero_dual_arm",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_chunks": 1,
            "total_videos": len(CAMERA_SLOTS),
            "fps": self.frequency,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "depth_camera_ids": [],
            "depth_path": "depth/cam{camera_id}/episode_{episode_index:06d}/frame_{frame_index:06d}.png",
            "features": self._feature_spec,
            "camera_roles": {"0": "head_fpv", "1": "left_wrist", "2": "right_wrist"},
            "camera_name_map": {"0": "head_fpv", "1": "left_wrist", "2": "right_wrist"},
            "source_camera_map": {
                "observation.images.cam0": "head",
                "observation.images.cam1": "left_wrist/wrist_left",
                "observation.images.cam2": "right_wrist/wrist_right",
            },
            "notes": "generated by LeRobotV2Writer; cam0=head, cam1=left_wrist/wrist_left, cam2=right_wrist/wrist_right",
        }

    def _write_meta_files(self) -> None:
        _json_dump_atomic(self._meta_info_path, self._build_info_payload())
        _jsonl_dump_atomic(self._meta_tasks_path, self._tasks)
        _jsonl_dump_atomic(self._meta_episodes_path, self._episodes)
        _jsonl_dump_atomic(self._meta_stats_path, self._episode_stats)

    def _new_runtime_stats(self) -> Dict[str, Any]:
        return {
            "enqueue_count": 0,
            "accepted_samples": 0,
            "skipped_samples": 0,
            "failed_samples": 0,
            "max_queue_depth": 0,
            "process_sample_ns": [],
            "rerun_log_ns": [],
            "finalize_ns": [],
        }

    def _runtime_stats_payload(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "enqueue_count": int(stats.get("enqueue_count", 0)),
            "accepted_samples": int(stats.get("accepted_samples", 0)),
            "skipped_samples": int(stats.get("skipped_samples", 0)),
            "failed_samples": int(stats.get("failed_samples", 0)),
            "max_queue_depth": int(stats.get("max_queue_depth", 0)),
            "queue": {
                "max_depth": int(stats.get("max_queue_depth", 0)),
            },
            "processing": _ms_summary(list(stats.get("process_sample_ns", []) or [])),
            "rerun": _ms_summary(list(stats.get("rerun_log_ns", []) or [])),
            "finalize": _ms_summary(list(stats.get("finalize_ns", []) or [])),
        }

    def _open_rerun_logger(self, episode_index: int) -> None:
        if not self.rerun_log:
            self.online_logger = None
            return
        if self.online_logger is not None:
            return
        try:
            self.online_logger = RerunLogger(
                prefix="online/",
                IdxRangeBoundary=60,
                memory_limit="300MB",
                rrd_path=None,
                spawn_viewer=True,
            )
        except Exception as exc:
            self.online_logger = None
            self.rerun_log = False
            logger_mp.warning("[LeRobotV2Writer] failed to create Rerun live logger; disable live viewer logging: %s", exc)

    def _close_rerun_logger(self) -> None:
        if self.online_logger is None:
            return
        try:
            self.online_logger.close()
        except Exception as exc:
            logger_mp.warning("[LeRobotV2Writer] failed to close Rerun live logger: %s", exc)
        finally:
            self.online_logger = None

    def _log_rerun_item(self, item_data: Dict[str, Any], item_idx: int) -> None:
        if not self.rerun_log or self.online_logger is None:
            return
        try:
            rerun_item = dict(item_data)
            rerun_item["idx"] = int(item_idx)
            rerun_item["colors"] = {
                key: value.copy() if hasattr(value, "copy") else value
                for key, value in (item_data.get("colors", {}) or {}).items()
            }
            self.online_logger.log_item_data(rerun_item)
        except Exception as exc:
            self._close_rerun_logger()
            self.rerun_log = False
            logger_mp.warning("[LeRobotV2Writer] Rerun live logging failed; disable live viewer logging: %s", exc)

    def _video_writer(self, path: str, size: Tuple[int, int], fps: float) -> Tuple[Any, str]:
        settings = _video_encoder_settings()
        if settings["backend"] == "opencv":
            writer = _OpenCVMP4Writer(
                path=path,
                size=size,
                fps=fps,
                fourccs=list(settings["fourccs"]),
            )
            return writer, str(writer.fourcc or settings["codec"])

        writer = _FFmpegMP4Writer(
            path=path,
            size=size,
            fps=fps,
            ffmpeg_bin=str(settings["ffmpeg_bin"]),
            preset=str(settings["preset"]),
            crf=int(settings["crf"]),
            profile=str(settings["profile"]),
            level=str(settings["level"]),
        )
        return writer, str(settings["codec"])

    def _video_encoder_config(self) -> Dict[str, Any]:
        return _video_encoder_settings()

    def _ensure_video_writer(self, slot: str, frame: np.ndarray) -> Any:
        writer = self._current_video_writers.get(slot)
        shape = (int(frame.shape[1]), int(frame.shape[0]))
        if writer is not None:
            expected = self._current_video_writer_sizes.get(slot)
            if expected is not None and expected != shape:
                raise ValueError(f"camera shape mismatch for {slot}: expected {(expected[1], expected[0], 3)}, got {frame.shape}")
            return writer

        if self._current_episode_index < 0:
            raise RuntimeError("video writer requested without an active episode")
        video_tmp_path = self._video_temp_path(self._current_episode_index, slot)
        writer, actual_fourcc = self._video_writer(video_tmp_path, shape, self.frequency)
        self._current_video_writers[slot] = writer
        self._current_video_writer_sizes[slot] = shape
        self._current_video_writer_fourcc[slot] = actual_fourcc
        self._current_video_frame_counts[slot] = 0
        return writer

    def _write_video_frames(self, slot_frames: Dict[str, np.ndarray]) -> None:
        for slot, frame in slot_frames.items():
            writer = self._ensure_video_writer(slot, frame)
            writer.write(np.ascontiguousarray(frame))
            self._current_video_frame_counts[slot] = int(self._current_video_frame_counts.get(slot, 0)) + 1

    def _release_video_writers(self) -> None:
        for writer in self._current_video_writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self._current_video_writers = {}

    def _extract_camera_frames(self, colors: Dict[str, Any]) -> Dict[str, np.ndarray]:
        slot_frames: Dict[str, np.ndarray] = {}
        for color_key, frame in (colors or {}).items():
            canonical_key = canonical_color_key(color_key)
            slot = CAMERA_SOURCE_TO_SLOT.get(canonical_key)
            if slot is None:
                continue
            if frame is None:
                continue
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"camera frame for {slot} must have shape HxWx3, got {getattr(frame, 'shape', None)}")
            slot_frames[slot] = np.ascontiguousarray(frame)
        return slot_frames

    def _extract_arm_vectors(self, states: Dict[str, Any], actions: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not states or not actions:
            raise ValueError("states/actions are required")

        def _get_qpos(source: Dict[str, Any], arm_key: str, expected_len: int, label: str) -> np.ndarray:
            arm = source.get(arm_key) or {}
            qpos = arm.get("qpos")
            if qpos is None:
                raise KeyError(f"{label} missing qpos")
            return _finite_list(qpos, expected_len, label)

        left_arm_state = _get_qpos(states, "left_arm", 7, "states.left_arm.qpos")
        right_arm_state = _get_qpos(states, "right_arm", 7, "states.right_arm.qpos")
        left_ee_state = _get_qpos(states, "left_ee", 1, "states.left_ee.qpos")
        right_ee_state = _get_qpos(states, "right_ee", 1, "states.right_ee.qpos")

        left_arm_action = _get_qpos(actions, "left_arm", 7, "actions.left_arm.qpos")
        right_arm_action = _get_qpos(actions, "right_arm", 7, "actions.right_arm.qpos")
        left_ee_action = _get_qpos(actions, "left_ee", 1, "actions.left_ee.qpos")
        right_ee_action = _get_qpos(actions, "right_ee", 1, "actions.right_ee.qpos")

        left_arm_qpos = np.concatenate([left_arm_state, left_ee_state])
        right_arm_qpos = np.concatenate([right_arm_state, right_ee_state])
        left_arm_action_qpos = np.concatenate([left_arm_action, left_ee_action])
        right_arm_action_qpos = np.concatenate([right_arm_action, right_ee_action])
        return left_arm_qpos, right_arm_qpos, left_arm_action_qpos, right_arm_action_qpos

    def _compute_fk_for_qpos(self, arm_qpos: np.ndarray, group: str) -> Dict[str, List[float]]:
        if group not in ARM_GROUPS:
            raise ValueError(f"FK group must be one of {ARM_GROUPS}, got {group!r}")
        q = np.asarray(arm_qpos, dtype=float).reshape(-1)
        if q.shape[0] != ARM_VECTOR_SIZE:
            raise ValueError(f"arm qpos expected length {ARM_VECTOR_SIZE}, got {q.shape[0]}")
        if not np.all(np.isfinite(q)):
            raise ValueError("arm qpos contains NaN or Inf")
        out: Dict[str, List[float]] = {}
        with self._fk_lock:
            pin.framesForwardKinematics(self._fk_model, self._fk_data, q)
            pin.updateFramePlacements(self._fk_model, self._fk_data)
            for key, frame_id in self._fk_frame_ids.items():
                if key.startswith(f"observation.fk.{group}."):
                    out[key] = _pose6_from_se3(self._fk_data.oMf[frame_id])
        return out

    def _camera_meta_for_slot(self, camera_timestamps: Dict[str, Any], slot: str) -> Tuple[str, Dict[str, Any]]:
        for source_name in CAMERA_SOURCE_ALIASES[slot]:
            meta = camera_timestamps.get(source_name)
            if isinstance(meta, dict):
                return source_name, meta
        return CAMERA_SLOT_PRIMARY_SOURCE[slot], {}

    def _camera_alignment_payload(self, source_name: str, meta: Dict[str, Any], sample_monotonic_ns: int) -> Dict[str, Any]:
        host_recv_monotonic_ns = _safe_int(meta.get("host_recv_monotonic_ns"))
        host_monotonic_ns = _safe_int(meta.get("host_monotonic_ns"))
        camera_monotonic_ns = host_recv_monotonic_ns if host_recv_monotonic_ns is not None else host_monotonic_ns
        delta_to_sample_ns = _safe_int(meta.get("delta_to_sample_ns"))
        if delta_to_sample_ns is None and camera_monotonic_ns is not None:
            delta_to_sample_ns = int(camera_monotonic_ns - sample_monotonic_ns)

        shape = meta.get("shape")
        if isinstance(shape, tuple):
            shape = list(shape)
        elif not isinstance(shape, list):
            shape = None

        return {
            "source_name": str(meta.get("camera_name") or source_name),
            "frame_seq": _safe_int(meta.get("frame_seq")),
            "host_recv_monotonic_ns": host_recv_monotonic_ns,
            "host_monotonic_ns": host_monotonic_ns,
            "camera_monotonic_ns": camera_monotonic_ns,
            "host_recv_wall_time_ns": _safe_int(meta.get("host_recv_wall_time_ns")),
            "host_wall_time_ns": _safe_int(meta.get("host_wall_time_ns")),
            "source_monotonic_ns": _safe_int(meta.get("source_monotonic_ns")),
            "source_wall_time_ns": _safe_int(meta.get("source_wall_time_ns")),
            "align_target_monotonic_ns": _safe_int(meta.get("align_target_monotonic_ns")),
            "delta_to_sample_ns": delta_to_sample_ns,
            "abs_delta_to_sample_ms": _safe_abs_ms(delta_to_sample_ns),
            "shape": shape,
            "protocol": str(meta.get("protocol")) if meta.get("protocol") is not None else None,
            "transport": str(meta.get("transport")) if meta.get("transport") is not None else None,
            "endpoint": str(meta.get("endpoint")) if meta.get("endpoint") is not None else None,
            "read_latency_ms": _safe_float(meta.get("read_latency_ms")),
        }

    def _alignment_timing_payload(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        meta = meta if isinstance(meta, dict) else {}
        return {
            "host_monotonic_ns": _safe_int(meta.get("host_monotonic_ns")),
            "delta_to_sample_ns": _safe_int(meta.get("delta_to_sample_ns")),
            "abs_delta_to_sample_ms": _safe_abs_ms(meta.get("delta_to_sample_ns")),
            "interpolation_mode": str(meta.get("interpolation_mode")) if meta.get("interpolation_mode") is not None else None,
            "support_source_count": _safe_int(meta.get("support_source_count")),
            "support_prev_t_ns": _safe_int(meta.get("support_prev_t_ns")),
            "support_next_t_ns": _safe_int(meta.get("support_next_t_ns")),
            "support_prev_delta_to_sample_ns": _safe_int(meta.get("support_prev_delta_to_sample_ns")),
            "support_next_delta_to_sample_ns": _safe_int(meta.get("support_next_delta_to_sample_ns")),
            "support_span_ns": _safe_int(meta.get("support_span_ns")),
            "support_max_abs_delta_ns": _safe_int(meta.get("support_max_abs_delta_ns")),
            "interp_alpha": _safe_float(meta.get("interp_alpha")),
        }

    def _build_alignment_row(
        self,
        timestamps: Dict[str, Any],
        episode_index: int,
        frame_index: int,
        sample_timestamp_s: float,
        sample_monotonic_ns: int,
    ) -> Dict[str, Any]:
        camera_timestamps = timestamps.get("camera", {}) if isinstance(timestamps, dict) else {}
        if not isinstance(camera_timestamps, dict):
            camera_timestamps = {}
        cameras = OrderedDict()
        for slot in CAMERA_SLOTS:
            source_name, meta = self._camera_meta_for_slot(camera_timestamps, slot)
            cameras[slot] = self._camera_alignment_payload(source_name, meta, sample_monotonic_ns)

        return {
            "schema_version": ALIGNMENT_SIDECAR_SCHEMA_VERSION,
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "timestamp": float(sample_timestamp_s),
            "timestamp_s": float(sample_timestamp_s),
            "sample_monotonic_ns": int(sample_monotonic_ns),
            "sample_wall_time_ns": _safe_int(timestamps.get("sample_wall_time_ns")) if isinstance(timestamps, dict) else None,
            "teleop_input_perf_counter_ns": _safe_int(timestamps.get("teleop_input_perf_counter_ns")) if isinstance(timestamps, dict) else None,
            "primary_camera_name": str(timestamps.get("primary_camera_name")) if isinstance(timestamps, dict) and timestamps.get("primary_camera_name") is not None else None,
            "cameras": cameras,
            "state": self._alignment_timing_payload(timestamps.get("state", {}) if isinstance(timestamps, dict) else {}),
            "action": self._alignment_timing_payload(timestamps.get("action", {}) if isinstance(timestamps, dict) else {}),
        }

    def _write_alignment_sidecar(self, episode_index: int, rows: List[Dict[str, Any]], expected_rows: int) -> Dict[str, Any]:
        if len(rows) != expected_rows:
            raise RuntimeError(f"alignment sidecar row count mismatch: {len(rows)} != {expected_rows}")

        sidecar_tmp = self._alignment_temp_path(episode_index)
        sidecar_final = self._alignment_final_path(episode_index)
        _safe_mkdir(os.path.dirname(sidecar_tmp))
        _jsonl_dump_atomic(sidecar_tmp, rows)

        line_count = 0
        with open(sidecar_tmp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line_count += 1
        if line_count != expected_rows:
            raise RuntimeError(f"alignment sidecar line count mismatch: {line_count} != {expected_rows}")
        if os.path.getsize(sidecar_tmp) <= 0:
            raise RuntimeError("alignment sidecar is empty after write")

        return {
            "tmp_path": sidecar_tmp,
            "final_path": sidecar_final,
            "relative_path": _dataset_relpath(sidecar_final, self.dataset_root),
            "num_rows": line_count,
        }

    def _control_tauff_from_extras(self, control_extras: Dict[str, Any]) -> np.ndarray:
        if not isinstance(control_extras, dict):
            raise ValueError("control_extras must be a dict")
        if "arm_tauff" not in control_extras:
            raise ValueError("control_extras.arm_tauff is required")
        return _finite_list(control_extras.get("arm_tauff"), ARM_VECTOR_SIZE, "control_extras.arm_tauff")

    def _build_control_row(
        self,
        control_extras: Dict[str, Any],
        episode_index: int,
        frame_index: int,
        sample_timestamp_s: float,
        sample_monotonic_ns: int,
    ) -> Dict[str, Any]:
        arm_tauff = self._control_tauff_from_extras(control_extras)
        return {
            "schema_version": CONTROL_SIDECAR_SCHEMA_VERSION,
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "timestamp": float(sample_timestamp_s),
            "sample_monotonic_ns": int(sample_monotonic_ns),
            "arm_tauff": [float(value) for value in arm_tauff.tolist()],
            "source": "teleop_runtime",
        }

    def _write_control_sidecar(self, episode_index: int, rows: List[Dict[str, Any]], expected_rows: int) -> Dict[str, Any]:
        if len(rows) != expected_rows:
            raise RuntimeError(f"control sidecar row count mismatch: {len(rows)} != {expected_rows}")

        sidecar_tmp = self._control_temp_path(episode_index)
        sidecar_final = self._control_final_path(episode_index)
        _safe_mkdir(os.path.dirname(sidecar_tmp))
        _jsonl_dump_atomic(sidecar_tmp, rows)

        line_count = 0
        with open(sidecar_tmp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line_count += 1
        if line_count != expected_rows:
            raise RuntimeError(f"control sidecar line count mismatch: {line_count} != {expected_rows}")
        if os.path.getsize(sidecar_tmp) <= 0:
            raise RuntimeError("control sidecar is empty after write")

        return {
            "tmp_path": sidecar_tmp,
            "final_path": sidecar_final,
            "relative_path": _dataset_relpath(sidecar_final, self.dataset_root),
            "num_rows": line_count,
        }

    def _normalize_and_store_item(self, item_data: Dict[str, Any]) -> None:
        if self._current_failure_reason is not None:
            return

        colors = item_data.get("colors", {}) or {}
        states = item_data.get("states", {}) or {}
        actions = item_data.get("actions", {}) or {}
        timestamps = item_data.get("timestamps", {}) or {}
        control_extras = item_data.get("control_extras")

        try:
            slot_frames = self._extract_camera_frames(colors)
        except ValueError as exc:
            with self._state_lock:
                self._current_failure_reason = str(exc)
            logger_mp.warning("[LeRobotV2Writer] episode failed: %s", exc)
            return

        missing_slots = [slot for slot in CAMERA_SLOTS if slot not in slot_frames]
        if missing_slots:
            logger_mp.warning(
                "[LeRobotV2Writer] skip sample: missing camera frame(s) %s",
                ", ".join(missing_slots),
            )
            return

        try:
            left_arm_qpos, right_arm_qpos, left_arm_action_qpos, right_arm_action_qpos = self._extract_arm_vectors(states, actions)
        except (KeyError, ValueError) as exc:
            logger_mp.warning("[LeRobotV2Writer] skip sample: %s", exc)
            return

        sample_monotonic_ns = timestamps.get("sample_monotonic_ns")
        if sample_monotonic_ns is None:
            raise ValueError("timestamps.sample_monotonic_ns is required")
        sample_monotonic_ns = int(sample_monotonic_ns)

        try:
            fk_fb = self._compute_fk_for_qpos(np.concatenate([left_arm_qpos[:7], right_arm_qpos[:7]]), "fb")
            fk_cmd = self._compute_fk_for_qpos(
                np.concatenate([left_arm_action_qpos[:7], right_arm_action_qpos[:7]]),
                "cmd",
            )
        except Exception as exc:
            with self._state_lock:
                self._current_failure_reason = f"FK computation failed: {exc}"
            logger_mp.warning("[LeRobotV2Writer] episode failed: FK computation failed: %s", exc)
            return

        with self._state_lock:
            if self._current_first_sample_monotonic_ns is None:
                self._current_first_sample_monotonic_ns = sample_monotonic_ns
            sample_timestamp_s = float(sample_monotonic_ns - self._current_first_sample_monotonic_ns) / 1e9
            if sample_timestamp_s < -1e-9:
                self._current_failure_reason = (
                    f"sample timestamp moved before episode start: {sample_timestamp_s:.9f}s"
                )
                logger_mp.warning("[LeRobotV2Writer] episode failed: %s", self._current_failure_reason)
                return

            frame_index = len(self._current_samples)
            expected_shapes = dict(self._current_frame_shapes)
            for slot, frame in slot_frames.items():
                shape = (int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2]))
                if slot not in expected_shapes:
                    expected_shapes[slot] = shape
                elif expected_shapes[slot] != shape:
                    self._current_failure_reason = (
                        f"camera shape mismatch for {slot}: expected {expected_shapes[slot]}, got {shape}"
                    )
                    logger_mp.warning("[LeRobotV2Writer] episode failed: %s", self._current_failure_reason)
                    return
            episode_index = self._current_episode_index
            task_index = int(self._current_task_index)

        try:
            self._write_video_frames(slot_frames)
        except Exception as exc:
            with self._state_lock:
                self._current_failure_reason = f"video streaming failed: {exc}"
            logger_mp.warning("[LeRobotV2Writer] episode failed: video streaming failed: %s", exc)
            return

        sample = {
            "frame_index": frame_index,
            "timestamp_s": sample_timestamp_s,
            "sample_monotonic_ns": sample_monotonic_ns,
            "task_index": task_index,
            "state_qpos": np.concatenate([left_arm_qpos[:7], left_arm_qpos[7:8], right_arm_qpos[:7], right_arm_qpos[7:8]]),
            "action_qpos": np.concatenate([left_arm_action_qpos[:7], left_arm_action_qpos[7:8], right_arm_action_qpos[:7], right_arm_action_qpos[7:8]]),
            "fk_fb": fk_fb,
            "fk_cmd": fk_cmd,
            "images": {
                slot: {
                    "path": _dataset_relpath(
                        self._video_final_path(episode_index, slot),
                        self.dataset_root,
                    ),
                    "timestamp_s": sample_timestamp_s,
                }
                for slot in CAMERA_SLOTS
            },
        }
        alignment_row = self._build_alignment_row(
            timestamps,
            episode_index,
            frame_index,
            sample_timestamp_s,
            sample_monotonic_ns,
        )
        control_row = None
        if control_extras:
            try:
                control_row = self._build_control_row(
                    control_extras,
                    episode_index,
                    frame_index,
                    sample_timestamp_s,
                    sample_monotonic_ns,
                )
            except ValueError as exc:
                with self._state_lock:
                    self._current_failure_reason = str(exc)
                logger_mp.warning("[LeRobotV2Writer] episode failed: %s", exc)
                return
        with self._state_lock:
            if self._current_failure_reason is not None:
                return
            self._current_frame_shapes = expected_shapes
            self._current_samples.append(sample)
            self._current_alignment_rows.append(alignment_row)
            if control_row is not None:
                self._current_control_rows.append(control_row)

    def _sample_is_valid(self, sample: Dict[str, Any]) -> None:
        state_qpos = np.asarray(sample["state_qpos"], dtype=float)
        action_qpos = np.asarray(sample["action_qpos"], dtype=float)
        if state_qpos.shape[0] != STATE_VECTOR_SIZE or action_qpos.shape[0] != STATE_VECTOR_SIZE:
            raise ValueError("state/action vector size mismatch")
        if not np.all(np.isfinite(state_qpos)):
            raise ValueError("state vector contains NaN or Inf")
        if not np.all(np.isfinite(action_qpos)):
            raise ValueError("action vector contains NaN or Inf")

        for fk_group in ("fk_fb", "fk_cmd"):
            fk_map = sample[fk_group]
            for key in fk_map:
                fk_arr = np.asarray(fk_map[key], dtype=float)
                if fk_arr.shape[0] != 6:
                    raise ValueError(f"{key} expected length 6")
                if not np.all(np.isfinite(fk_arr)):
                    raise ValueError(f"{key} contains NaN or Inf")

        for slot in CAMERA_SLOTS:
            image_meta = sample["images"][slot]
            path = image_meta.get("path")
            timestamp_s = image_meta.get("timestamp_s")
            if not isinstance(path, str) or not path:
                raise ValueError(f"{slot} image path is missing")
            if timestamp_s is None or not np.isfinite(float(timestamp_s)):
                raise ValueError(f"{slot} image timestamp is invalid")

    def _build_parquet_arrays(self, samples: List[Dict[str, Any]]) -> Dict[str, pa.Array]:
        columns: Dict[str, List[Any]] = OrderedDict()
        columns["timestamp"] = []
        columns["frame_index"] = []
        columns["episode_index"] = []
        columns["index"] = []
        columns["task_index"] = []
        columns["observation.state"] = []
        columns["action"] = []
        for group in ARM_GROUPS:
            for side in SIDE_GROUPS:
                for label in ARM_JOINT_LABELS:
                    key = f"observation.fk.fb.{side}.{label}" if group == "fb" else f"observation.fk.cmd.{side}.{label}"
                    columns[key] = []
                key = f"observation.fk.fb.{side}.gripper_flange" if group == "fb" else f"observation.fk.cmd.{side}.gripper_flange"
                columns[key] = []
        for slot in CAMERA_SLOTS:
            columns[f"observation.images.{slot}"] = []

        global_index = self._global_frame_index_next
        for sample in samples:
            self._sample_is_valid(sample)
            columns["timestamp"].append(float(sample["timestamp_s"]))
            columns["frame_index"].append(int(sample["frame_index"]))
            columns["episode_index"].append(int(self._current_episode_index))
            columns["index"].append(int(global_index))
            columns["task_index"].append(int(sample["task_index"]))
            global_index += 1
            columns["observation.state"].append(np.asarray(sample["state_qpos"], dtype=float).tolist())
            columns["action"].append(np.asarray(sample["action_qpos"], dtype=float).tolist())

            for key, value in sample["fk_fb"].items():
                columns[key].append(np.asarray(value, dtype=float).tolist())
            for key, value in sample["fk_cmd"].items():
                columns[key].append(np.asarray(value, dtype=float).tolist())

            for slot in CAMERA_SLOTS:
                columns[f"observation.images.{slot}"].append(
                    {
                        "path": sample["images"][slot]["path"],
                        "timestamp": float(sample["images"][slot]["timestamp_s"]),
                    }
                )

        arrays: Dict[str, pa.Array] = OrderedDict()
        arrays["timestamp"] = pa.array(columns["timestamp"], type=pa.float64())
        arrays["frame_index"] = pa.array(columns["frame_index"], type=pa.int64())
        arrays["episode_index"] = pa.array(columns["episode_index"], type=pa.int64())
        arrays["index"] = pa.array(columns["index"], type=pa.int64())
        arrays["task_index"] = pa.array(columns["task_index"], type=pa.int64())
        arrays["observation.state"] = pa.array(columns["observation.state"], type=pa.list_(pa.float64()))
        arrays["action"] = pa.array(columns["action"], type=pa.list_(pa.float64()))
        for group in ARM_GROUPS:
            for side in SIDE_GROUPS:
                for label in ARM_JOINT_LABELS:
                    key = f"observation.fk.fb.{side}.{label}" if group == "fb" else f"observation.fk.cmd.{side}.{label}"
                    arrays[key] = pa.array(columns[key], type=pa.list_(pa.float64()))
                key = f"observation.fk.fb.{side}.gripper_flange" if group == "fb" else f"observation.fk.cmd.{side}.gripper_flange"
                arrays[key] = pa.array(columns[key], type=pa.list_(pa.float64()))
        for slot in CAMERA_SLOTS:
            arrays[f"observation.images.{slot}"] = pa.array(columns[f"observation.images.{slot}"], type=_struct_type())
        return arrays

    def _write_episode_artifacts(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise RuntimeError("cannot write an empty episode")

        for sample in samples:
            self._sample_is_valid(sample)

        parquet_tmp = self._parquet_temp_path(self._current_episode_index)
        parquet_final = self._parquet_final_path(self._current_episode_index)
        video_tmp_paths = {slot: self._video_temp_path(self._current_episode_index, slot) for slot in CAMERA_SLOTS}
        video_final_paths = {slot: self._video_final_path(self._current_episode_index, slot) for slot in CAMERA_SLOTS}
        alignment_rows = list(self._current_alignment_rows)
        alignment_sidecar = self._write_alignment_sidecar(
            self._current_episode_index,
            alignment_rows,
            len(samples),
        )
        control_rows = list(self._current_control_rows)
        control_sidecar = None
        if control_rows:
            control_sidecar = self._write_control_sidecar(
                self._current_episode_index,
                control_rows,
                len(samples),
            )

        paths = [parquet_tmp, parquet_final, alignment_sidecar["tmp_path"], alignment_sidecar["final_path"], *video_tmp_paths.values(), *video_final_paths.values()]
        if control_sidecar is not None:
            paths.extend([control_sidecar["tmp_path"], control_sidecar["final_path"]])

        for path in paths:
            parent = os.path.dirname(path)
            if parent:
                _safe_mkdir(parent)

        try:
            self._release_video_writers()

            for slot in CAMERA_SLOTS:
                if self._current_video_frame_counts.get(slot, 0) != len(samples):
                    raise RuntimeError(
                        f"streamed frame count mismatch for {slot}: "
                        f"{self._current_video_frame_counts.get(slot, 0)} != {len(samples)}"
                    )
                if slot not in self._current_video_writer_sizes:
                    raise RuntimeError(f"missing video writer size for {slot}")
                if not os.path.exists(video_tmp_paths[slot]):
                    raise RuntimeError(f"missing streamed video file for {slot}: {video_tmp_paths[slot]}")

            arrays = self._build_parquet_arrays(samples)
            table = pa.Table.from_arrays([arrays[name] for name in self._parquet_schema.names], schema=self._parquet_schema)
            parquet_compression = _pick_parquet_compression()
            pq.write_table(table, parquet_tmp, compression=parquet_compression)

            parquet_row_count = int(table.num_rows)
            if parquet_row_count != len(samples):
                raise RuntimeError(f"parquet row count mismatch: {parquet_row_count} != {len(samples)}")
            for column_name in table.column_names:
                if not _column_values_are_finite(table[column_name]):
                    raise RuntimeError(f"parquet column contains non-finite values: {column_name}")

            video_stats: Dict[str, Dict[str, Any]] = {}
            for slot in CAMERA_SLOTS:
                capture = cv2.VideoCapture(video_tmp_paths[slot])
                if not capture.isOpened():
                    raise RuntimeError(f"failed to open encoded video for validation: {video_tmp_paths[slot]}")
                decoded_count = 0
                decoded_shape: Optional[Tuple[int, int, int]] = None
                while True:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    decoded_count += 1
                    decoded_shape = (int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2]) if frame.ndim == 3 else 1)
                capture.release()
                if decoded_count != len(samples):
                    raise RuntimeError(
                        f"decoded frame count mismatch for {slot}: {decoded_count} != {len(samples)}"
                    )
                video_stats[slot] = {
                    "path": _dataset_relpath(video_final_paths[slot], self.dataset_root),
                    "expected_frames": len(samples),
                    "decoded_frames": decoded_count,
                    "shape": list(self._current_video_writer_sizes[slot][::-1]) + [3],
                    "decoded_shape": list(decoded_shape) if decoded_shape is not None else None,
                    "codec": self._current_video_writer_fourcc.get(slot, ""),
                }

            with open(parquet_tmp, "rb") as f:
                parquet_bytes = f.read()
            if len(parquet_bytes) == 0:
                raise RuntimeError("parquet file is empty after write")

            os.replace(parquet_tmp, parquet_final)
            for slot in CAMERA_SLOTS:
                os.replace(video_tmp_paths[slot], video_final_paths[slot])
            os.replace(alignment_sidecar["tmp_path"], alignment_sidecar["final_path"])
            if control_sidecar is not None:
                os.replace(control_sidecar["tmp_path"], control_sidecar["final_path"])

            stats = {
                "validation_passed": True,
                "episode_index": int(self._current_episode_index),
                "task_index": int(self._current_task_index),
                "num_samples": len(samples),
                "parquet": {
                    "path": _dataset_relpath(parquet_final, self.dataset_root),
                    "row_count": parquet_row_count,
                    "compression": parquet_compression or "uncompressed",
                },
                "videos": video_stats,
            }
            return stats
        finally:
            self._release_video_writers()

    def _cleanup_unregistered_artifacts(self, episode_index: Optional[int] = None) -> None:
        if episode_index is None:
            episode_index = self._current_episode_index
        if episode_index < 0:
            return
        paths = [
            self._parquet_temp_path(episode_index),
            self._parquet_final_path(episode_index),
            self._alignment_temp_path(episode_index),
            self._alignment_final_path(episode_index),
            self._control_temp_path(episode_index),
            self._control_final_path(episode_index),
        ]
        for slot in CAMERA_SLOTS:
            paths.append(self._video_temp_path(episode_index, slot))
            paths.append(self._video_final_path(episode_index, slot))
        for path in paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def _reset_episode_state(self) -> None:
        self._close_rerun_logger()
        self._release_video_writers()
        self._current_episode_active = False
        self._current_episode_index = -1
        self._current_task_index = -1
        self._current_task_text = self.task_goal
        self._current_samples = []
        self._current_alignment_rows = []
        self._current_control_rows = []
        self._current_frame_shapes = {}
        self._current_failure_reason = None
        self._current_episode_paths = {}
        self._current_first_sample_monotonic_ns = None
        self._current_item_idx = -1
        self._current_runtime_stats = self._new_runtime_stats()
        self._current_video_writer_sizes = {}
        self._current_video_writer_fourcc = {}
        self._current_video_frame_counts = {slot: 0 for slot in CAMERA_SLOTS}
        self._save_requested = False
        self._active_task_registered = False
        self.is_available = True

    def _finalize_episode(self) -> None:
        with self._state_lock:
            if not self._current_episode_active:
                self._save_requested = False
                self.is_available = True
                return
            episode_index = self._current_episode_index
            task_index = self._current_task_index
            task_text = self._current_task_text
            samples = list(self._current_samples)
            failed_reason = self._current_failure_reason
            already_registered = self._active_task_registered
            runtime_stats = dict(self._current_runtime_stats)

        if failed_reason is not None:
            logger_mp.warning("[LeRobotV2Writer] drop episode %d because it failed: %s", episode_index, failed_reason)
            self._cleanup_unregistered_artifacts(episode_index)
            with self._state_lock:
                self._reset_episode_state()
            return

        if not samples:
            logger_mp.warning("[LeRobotV2Writer] drop empty episode %d", episode_index)
            self._cleanup_unregistered_artifacts(episode_index)
            with self._state_lock:
                self._reset_episode_state()
            return

        try:
            finalize_t0_ns = time.perf_counter_ns()
            stats = self._write_episode_artifacts(samples)
            finalize_t1_ns = time.perf_counter_ns()
            runtime_stats.setdefault("finalize_ns", []).append(int(finalize_t1_ns - finalize_t0_ns))
            stats["runtime"] = self._runtime_stats_payload(runtime_stats)
        except Exception as exc:
            logger_mp.error("[LeRobotV2Writer] failed to save episode %d: %s", episode_index, exc)
            self._cleanup_unregistered_artifacts(episode_index)
            with self._state_lock:
                self._current_failure_reason = str(exc)
                self._reset_episode_state()
            return

        with self._state_lock:
            self._commit_task_if_needed(task_text, task_index, already_registered)
            self._episodes.append(
                {
                    "episode_index": int(episode_index),
                    "tasks": [task_text],
                    "length": len(samples),
                }
            )
            self._episode_stats.append(
                {
                    "episode_index": int(episode_index),
                    "stats": stats,
                }
            )
            self._global_frame_index_next += len(samples)
            self._reset_episode_state()

        try:
            self._write_meta_files()
        except Exception as exc:
            logger_mp.error("[LeRobotV2Writer] episode %d saved, but failed to update meta files: %s", episode_index, exc)
            return

        logger_mp.info(
            "[LeRobotV2Writer] saved episode %06d with %d samples (task_index=%d)",
            episode_index,
            len(samples),
            task_index,
        )

    def is_ready(self) -> bool:
        return bool(self.is_available)

    def create_episode(self) -> bool:
        with self._state_lock:
            if not self.is_available:
                logger_mp.info("[LeRobotV2Writer] writer is busy; wait for save to finish before creating a new episode.")
                return False
            episode_index = self._next_episode_index
            self._next_episode_index += 1
            task_index, already_registered = self._task_index_for_text(self.task_goal)
            self._current_episode_active = True
            self._current_episode_index = int(episode_index)
            self._current_task_index = int(task_index)
            self._current_task_text = self.task_goal
            self._current_samples = []
            self._current_alignment_rows = []
            self._current_control_rows = []
            self._current_frame_shapes = {}
            self._current_failure_reason = None
            self._current_first_sample_monotonic_ns = None
            self._current_item_idx = -1
            self._current_runtime_stats = self._new_runtime_stats()
            self._current_video_writers = {}
            self._current_video_writer_sizes = {}
            self._current_video_writer_fourcc = {}
            self._current_video_frame_counts = {slot: 0 for slot in CAMERA_SLOTS}
            self._current_episode_paths = {
                "parquet": self._parquet_final_path(episode_index),
                "parquet_tmp": self._parquet_temp_path(episode_index),
                "alignment": self._alignment_final_path(episode_index),
                "alignment_tmp": self._alignment_temp_path(episode_index),
                "control": self._control_final_path(episode_index),
                "control_tmp": self._control_temp_path(episode_index),
                "videos": {slot: self._video_final_path(episode_index, slot) for slot in CAMERA_SLOTS},
                "videos_tmp": {slot: self._video_temp_path(episode_index, slot) for slot in CAMERA_SLOTS},
            }
            self._active_task_registered = already_registered
            self._save_requested = False
            self.is_available = False
        logger_mp.info(
            "[LeRobotV2Writer] created episode %06d for task_index=%d",
            episode_index,
            task_index,
        )
        return True

    def add_item(
        self,
        colors=None,
        depths=None,
        states=None,
        actions=None,
        tactiles=None,
        audios=None,
        sim_state=None,
        timestamps=None,
        control_extras=None,
    ):
        with self._state_lock:
            if not self._current_episode_active or self.is_available:
                logger_mp.warning("[LeRobotV2Writer] drop sample because no active episode is ready.")
                return
        item_data = {
            "colors": colors or {},
            "depths": depths or {},
            "states": states or {},
            "actions": actions or {},
            "tactiles": tactiles or {},
            "audios": audios or {},
            "sim_state": sim_state,
            "timestamps": timestamps or {},
            "control_extras": control_extras or {},
        }
        self._item_queue.put(item_data)
        with self._state_lock:
            self._current_runtime_stats["enqueue_count"] += 1
            self._current_runtime_stats["max_queue_depth"] = max(
                int(self._current_runtime_stats.get("max_queue_depth", 0)),
                int(self._item_queue.qsize()),
            )

    def process_queue(self) -> None:
        while not self._stop_worker or not self._item_queue.empty():
            try:
                item_data = self._item_queue.get(timeout=0.2)
            except Empty:
                item_data = None
            if item_data is not None:
                try:
                    should_log_rerun = False
                    item_idx = -1
                    with self._state_lock:
                        sample_count_before = len(self._current_samples)
                    process_t0_ns = time.perf_counter_ns()
                    self._normalize_and_store_item(item_data)
                    process_t1_ns = time.perf_counter_ns()
                    with self._state_lock:
                        sample_count_after = len(self._current_samples)
                        self._current_runtime_stats["process_sample_ns"].append(int(process_t1_ns - process_t0_ns))
                        if (
                            self._current_failure_reason is None
                            and self._current_episode_active
                            and sample_count_after > sample_count_before
                        ):
                            self._current_runtime_stats["accepted_samples"] += 1
                            self._current_item_idx += 1
                            item_idx = self._current_item_idx
                            should_log_rerun = True
                        elif self._current_failure_reason is not None:
                            self._current_runtime_stats["failed_samples"] += 1
                        else:
                            self._current_runtime_stats["skipped_samples"] += 1
                    if should_log_rerun:
                        self._open_rerun_logger(self._current_episode_index)
                        rerun_t0_ns = time.perf_counter_ns()
                        self._log_rerun_item(item_data, item_idx)
                        rerun_t1_ns = time.perf_counter_ns()
                        with self._state_lock:
                            self._current_runtime_stats["rerun_log_ns"].append(int(rerun_t1_ns - rerun_t0_ns))
                except Exception as exc:
                    with self._state_lock:
                        self._current_failure_reason = str(exc)
                    logger_mp.error("[LeRobotV2Writer] failed to process sample: %s", exc)
                finally:
                    self._item_queue.task_done()
            with self._state_lock:
                should_save = self._save_requested and not self._finalizing and self._item_queue.empty()
                if should_save:
                    self._finalizing = True
                else:
                    should_save = False
            if should_save:
                try:
                    self._finalize_episode()
                finally:
                    with self._state_lock:
                        self._finalizing = False

    def save_episode(self) -> None:
        with self._state_lock:
            if not self._current_episode_active:
                logger_mp.warning("[LeRobotV2Writer] save requested without an active episode.")
                return
            self._save_requested = True
        logger_mp.info("[LeRobotV2Writer] save requested for the active episode.")

    def close(self) -> None:
        with self._state_lock:
            if self._current_episode_active and not self._save_requested:
                self._save_requested = True
        self._item_queue.join()
        while True:
            with self._state_lock:
                busy = not self.is_available or self._finalizing or self._save_requested
            if not busy:
                break
            time.sleep(0.05)
        self._stop_worker = True
        self.worker_thread.join(timeout=5.0)
        self._close_rerun_logger()


EpisodeWriter = LeRobotV2Writer

__all__ = [
    "EpisodeWriter",
    "LeRobotV2Writer",
    "ZMQRawCameraReceiver",
]
