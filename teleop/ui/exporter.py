from __future__ import annotations

import copy
import json
import math
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_UI_URDF_PATH = str(Path(__file__).resolve().parents[2] / "assets/g1_d/g1_d.urdf")
EPISODE_LENGTH_WARNING_MIN_SAMPLES = 4
EPISODE_LENGTH_WARNING_IQR_SCALE = 1.5


@dataclass(frozen=True)
class UiExportRequest:
    source_root: Path
    output_root: Path
    dataset_name: str
    task: str
    fps: float
    format_version: str = "v2"
    export_mode: str = "new"
    export_video: bool = False
    export_fk: bool = False
    export_verify: bool = True
    selected_episodes: tuple[str, ...] = ()
    urdf_path: str = DEFAULT_UI_URDF_PATH


class UiExportManager:
    """Background LeRobot export task used by the reference-shaped web UI."""

    def __init__(self, export_raw_task_dir: Callable[..., dict[str, Any]] | None = None) -> None:
        self._export_raw_task_dir = export_raw_task_dir
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._run_id = 0
        self._status = self._idle_status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._status)

    def start(self, request: UiExportRequest) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                status = copy.deepcopy(self._status)
                status["ok"] = False
                status["error"] = "export is already running"
                status["message"] = "export is already running"
                return status

        validation_error = self._validate_static_request(request)
        if validation_error:
            with self._lock:
                self._status = self._error_status(validation_error)
                return copy.deepcopy(self._status)

        source_root = Path(request.source_root).expanduser().resolve()
        output_parent = Path(request.output_root).expanduser().resolve()
        dataset_name = str(request.dataset_name).strip()
        output_dataset_root = output_parent / dataset_name
        path_error = self._validate_output_path(source_root, output_parent, output_dataset_root)
        if path_error:
            with self._lock:
                self._status = self._error_status(path_error)
                return copy.deepcopy(self._status)
        try:
            episode_names, frame_counts = self._resolve_episode_plan(source_root, request.selected_episodes)
        except Exception as exc:
            with self._lock:
                self._status = self._error_status(str(exc))
                self._status["traceback"] = traceback.format_exc()
                return copy.deepcopy(self._status)
        empty_episode_names = [name for name, frame_count in frame_counts.items() if frame_count <= 0]
        if empty_episode_names:
            with self._lock:
                self._status = self._error_status(
                    "selected episodes have no samples: " + ", ".join(empty_episode_names)
                )
                return copy.deepcopy(self._status)
        episode_length_reference, episode_length_warnings = _episode_length_warnings(frame_counts)
        total_frames = int(sum(frame_counts.values()))
        if total_frames <= 0:
            with self._lock:
                self._status = self._error_status(f"selected episodes contain no frames under {source_root}")
                return copy.deepcopy(self._status)

        normalized = UiExportRequest(
            source_root=source_root,
            output_root=output_parent,
            dataset_name=dataset_name,
            task=str(request.task).strip(),
            fps=float(request.fps),
            format_version=str(request.format_version).strip().lower(),
            export_mode=str(request.export_mode).strip().lower(),
            export_video=bool(request.export_video),
            export_fk=bool(request.export_fk),
            export_verify=bool(request.export_verify),
            selected_episodes=tuple(episode_names),
            urdf_path=str(request.urdf_path or DEFAULT_UI_URDF_PATH).strip() or DEFAULT_UI_URDF_PATH,
        )

        with self._lock:
            self._run_id += 1
            run_id = self._run_id
            self._status = {
                "ok": True,
                "state": "running",
                "phase": "exporting",
                "running": True,
                "message": f"export started: {len(episode_names)} episodes -> {output_dataset_root}",
                "processed_frames": 0,
                "total_frames": total_frames,
                "output_total_frames": 0,
                "append_base_frames": 0,
                "total_episodes": len(episode_names),
                "processed_episodes": 0,
                "source_root": str(source_root),
                "output_root": str(output_dataset_root),
                "dataset_name": dataset_name,
                "selected_episodes": list(episode_names),
                "episode_frame_counts": dict(frame_counts),
                "episode_length_reference": episode_length_reference,
                "episode_length_warnings": episode_length_warnings,
                "format_version": normalized.format_version,
                "export_mode": normalized.export_mode,
                "export_video": normalized.export_video,
                "export_fk": normalized.export_fk,
                "export_verify": normalized.export_verify,
                "urdf_path": normalized.urdf_path,
                "start_time_ns": time.time_ns(),
                "end_time_ns": 0,
                "export_verification": {},
            }
            thread = threading.Thread(
                target=self._run_export,
                args=(run_id, normalized, output_dataset_root, episode_names, total_frames),
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return copy.deepcopy(self._status)

    def _run_export(
        self,
        run_id: int,
        request: UiExportRequest,
        output_dataset_root: Path,
        episode_names: tuple[str, ...],
        total_frames: int,
    ) -> None:
        # This is the only exception boundary: without it, a background-thread failure would leave the UI stuck at running.
        try:
            input_task_dir = self._build_input_task_dir(request.source_root, episode_names)
            with input_task_dir as selected_source_root:
                export_summary = self._call_exporter(request, selected_source_root, output_dataset_root)
            processed_frames = int(export_summary.get("frames_total", total_frames) or total_frames)
            processed_episodes = int(export_summary.get("episodes_exported", len(episode_names)) or len(episode_names))
            with self._lock:
                if run_id != self._run_id:
                    return
                self._status.update(
                    {
                        "ok": True,
                        "state": "done",
                        "phase": "done",
                        "running": False,
                        "message": f"export done: {processed_episodes} episodes, {processed_frames} frames",
                        "processed_frames": processed_frames,
                        "output_total_frames": processed_frames,
                        "processed_episodes": processed_episodes,
                        "end_time_ns": time.time_ns(),
                        "export_summary": export_summary,
                        "export_verification": self._verification_status(request, export_summary),
                    }
                )
        except Exception as exc:
            with self._lock:
                if run_id != self._run_id:
                    return
                self._status.update(
                    {
                        "ok": False,
                        "state": "error",
                        "phase": "error",
                        "running": False,
                        "message": "export failed",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "end_time_ns": time.time_ns(),
                    }
                )

    def _call_exporter(self, request: UiExportRequest, input_task_dir: Path, output_dataset_root: Path) -> dict[str, Any]:
        export_raw_task_dir = self._export_raw_task_dir
        if export_raw_task_dir is None:
            from data_pipeline.export.raw_to_lerobot_v2 import export_raw_task_dir as default_export_raw_task_dir

            export_raw_task_dir = default_export_raw_task_dir
        result = export_raw_task_dir(
            input_task_dir=input_task_dir,
            output_root=output_dataset_root,
            task=request.task,
            fps=request.fps,
            overwrite=request.export_mode == "replace",
            progress_every=1,
            frame_progress_every=100,
            strict_image_validate=False,
            verify_export=request.export_verify,
            verify_video_frames=request.export_video,
            export_fk=request.export_fk,
            urdf_path=Path(request.urdf_path).expanduser().resolve(),
        )
        if not isinstance(result, dict):
            raise TypeError(f"raw_to_lerobot_v2 exporter must return dict, got {type(result).__name__}")
        return result

    def _build_input_task_dir(self, source_root: Path, episode_names: tuple[str, ...]):
        source_episode_names = tuple(path.name for path in sorted(source_root.iterdir()) if path.is_dir() and path.name.startswith("episode_"))
        if episode_names == source_episode_names:
            return _ExistingTaskDir(source_root)
        return _SelectedEpisodeTaskDir(source_root, episode_names)

    @staticmethod
    def _validate_static_request(request: UiExportRequest) -> str:
        dataset_name = str(request.dataset_name).strip()
        if not str(request.source_root).strip():
            return "source_root is required"
        if not str(request.output_root).strip():
            return "output_root is required"
        if not dataset_name:
            return "dataset_name is required"
        if dataset_name in {".", ".."}:
            return f"dataset_name must be a plain dataset directory name, got {dataset_name!r}"
        if "/" in dataset_name or "\\" in dataset_name:
            return f"dataset_name must be a directory name, got {dataset_name!r}"
        if not str(request.task).strip():
            return "default_task must be a non-empty task description"
        fps = float(request.fps)
        if not math.isfinite(fps) or fps <= 0.0:
            return "fps must be a positive finite number"
        format_version = str(request.format_version).strip().lower()
        if format_version != "v2":
            return f"format_version={format_version!r} is not implemented; xr_teleoperate UI currently exports LeRobot v2 only"
        export_mode = str(request.export_mode).strip().lower()
        if export_mode == "append":
            return "export_mode=append is not implemented by raw_to_lerobot_v2; use new or replace"
        if export_mode not in {"new", "replace"}:
            return f"export_mode must be new/replace/append, got {export_mode!r}"
        if bool(request.export_fk):
            urdf_path = Path(str(request.urdf_path).strip()).expanduser()
            if not urdf_path.is_file():
                return f"urdf_path does not exist or is not a file: {urdf_path}"
        return ""

    @staticmethod
    def _validate_output_path(source_root: Path, output_parent: Path, output_dataset_root: Path) -> str:
        output_parent_resolved = output_parent.resolve()
        output_dataset_resolved = output_dataset_root.resolve()
        source_resolved = source_root.resolve()
        if not _path_contains(output_parent_resolved, output_dataset_resolved):
            return f"output dataset root escapes output_root: {output_dataset_resolved}"
        if _paths_overlap(source_resolved, output_dataset_resolved):
            return (
                "output dataset root overlaps source_root; choose a separate output_root "
                f"(source_root={source_resolved}, output={output_dataset_resolved})"
            )
        return ""

    def _resolve_episode_plan(self, source_root: Path, selected_episodes: tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, int]]:
        if not source_root.is_dir():
            raise NotADirectoryError(f"source_root is not a directory: {source_root}")
        requested_names = self._normalize_episode_names(selected_episodes)
        if not requested_names:
            requested_names = tuple(path.name for path in sorted(source_root.iterdir()) if path.is_dir() and path.name.startswith("episode_"))
        if not requested_names:
            raise FileNotFoundError(f"no episode_xxxx directories found under {source_root}")

        frame_counts: dict[str, int] = {}
        for episode_name in requested_names:
            episode_dir = source_root / episode_name
            if not episode_dir.is_dir():
                raise FileNotFoundError(f"selected episode not found: {episode_dir}")
            frame_counts[episode_name] = self._count_episode_frames(episode_dir)
        return requested_names, frame_counts

    @staticmethod
    def _normalize_episode_names(selected_episodes: tuple[str, ...]) -> tuple[str, ...]:
        names = []
        seen = set()
        for raw_name in selected_episodes:
            name = str(raw_name).strip()
            if not name:
                continue
            if "/" in name or "\\" in name or name in {".", ".."}:
                raise ValueError(f"invalid episode name: {name!r}")
            if name not in seen:
                names.append(name)
                seen.add(name)
        return tuple(names)

    @staticmethod
    def _count_episode_frames(episode_dir: Path) -> int:
        data_path = episode_dir / "data.json"
        if not data_path.is_file():
            raise FileNotFoundError(f"data.json not found: {data_path}")
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError(f"data.json must contain a top-level data list: {data_path}")
        return len(payload["data"])

    @staticmethod
    def _verification_status(request: UiExportRequest, export_summary: dict[str, Any]) -> dict[str, Any]:
        if not request.export_verify:
            return {}
        episode_errors = []
        for item in export_summary.get("episodes", []) or []:
            verification = item.get("verification", {}) if isinstance(item, dict) else {}
            if isinstance(verification, dict) and verification.get("enabled") is False:
                episode_errors.append(f"{item.get('source_episode', '-')}: verification disabled")
        return {
            "ok": len(episode_errors) == 0,
            "errors": episode_errors,
            "summary_path": str(Path(str(export_summary.get("output_root", ""))) / "export_summary.json")
            if export_summary.get("output_root")
            else "",
            "samples": [],
        }

    @staticmethod
    def _idle_status() -> dict[str, Any]:
        return {
            "ok": True,
            "state": "idle",
            "phase": "idle",
            "running": False,
            "message": "ready: LeRobot v2 raw exporter is available",
            "processed_frames": 0,
            "total_frames": 0,
            "output_total_frames": 0,
            "append_base_frames": 0,
            "export_video": False,
            "export_fk": False,
            "export_verify": False,
            "export_verification": {},
        }

    @staticmethod
    def _error_status(error: str) -> dict[str, Any]:
        return {
            "ok": False,
            "state": "error",
            "phase": "error",
            "running": False,
            "message": "export rejected",
            "error": str(error),
            "processed_frames": 0,
            "total_frames": 0,
            "output_total_frames": 0,
            "append_base_frames": 0,
            "export_verification": {},
            "end_time_ns": time.time_ns(),
        }


class _ExistingTaskDir:
    def __init__(self, path: Path) -> None:
        self._path = path

    def __enter__(self) -> Path:
        return self._path

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> bool:
        return False


class _SelectedEpisodeTaskDir:
    def __init__(self, source_root: Path, episode_names: tuple[str, ...]) -> None:
        self._source_root = source_root
        self._episode_names = episode_names
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="xr_ui_export_")
        view_root = Path(self._temp_dir.name)
        for episode_name in self._episode_names:
            source_episode = self._source_root / episode_name
            target_episode = view_root / episode_name
            target_episode.symlink_to(source_episode, target_is_directory=True)
        return view_root

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> bool:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
        return False


def _path_contains(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


def _paths_overlap(left: Path, right: Path) -> bool:
    return _path_contains(left, right) or _path_contains(right, left)


def _episode_length_warnings(frame_counts: dict[str, int]) -> tuple[dict[str, object], list[dict[str, object]]]:
    values = sorted(int(frame_count) for frame_count in frame_counts.values() if frame_count > 0)
    reference: dict[str, object] = {
        "method": "iqr",
        "sample_count": len(values),
        "minimum_sample_count": EPISODE_LENGTH_WARNING_MIN_SAMPLES,
    }
    if len(values) < EPISODE_LENGTH_WARNING_MIN_SAMPLES:
        return reference, []

    q1 = _percentile(values, 0.25)
    median = _percentile(values, 0.50)
    q3 = _percentile(values, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - EPISODE_LENGTH_WARNING_IQR_SCALE * iqr
    upper_bound = q3 + EPISODE_LENGTH_WARNING_IQR_SCALE * iqr
    reference.update(
        {
            "median_frame_count": median,
            "iqr_lower_bound": lower_bound,
            "iqr_upper_bound": upper_bound,
        }
    )
    warnings: list[dict[str, object]] = []
    for episode_name, frame_count in frame_counts.items():
        if frame_count < lower_bound:
            warnings.append({"kind": "short", "episode": episode_name, "frame_count": int(frame_count)})
        elif frame_count > upper_bound:
            warnings.append({"kind": "long", "episode": episode_name, "frame_count": int(frame_count)})
    return reference, warnings


def _percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * float(pct)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    upper_weight = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - upper_weight)
        + sorted_values[upper_index] * upper_weight
    )
