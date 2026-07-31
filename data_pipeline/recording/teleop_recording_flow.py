"""遥操作业录制流程：管理 episode 生命周期、时间对齐和样本写入。

Recording flow used by the real teleop entrypoint.

This module owns the mutable recording state and the per-frame alignment/write
logic so ``teleop_hand_and_arm.py`` can stay as the runtime coordinator.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from data_pipeline.recording.alignment import (
    build_alignment_timestamp_entry,
    camera_frame_identity,
    camera_meta_monotonic_ns,
    interpolate_base_height_timed_sample_strict,
    interpolate_base_state_timed_sample_strict,
    interpolate_slam_tf_timed_sample_strict,
    interpolate_timed_sample_strict,
    nearest_timed_sample,
    timed_buffer_bounds,
)
from data_pipeline.recording.base_recording import build_base_action_record, build_base_state_record
from teleop.control_flow.mobile_manipulation_coordinator import column_position_from_raw_height


_CAMERA_NAME_ALIASES = {
    "head": "head",
    "left_wrist": "left_wrist",
    "wrist_left": "left_wrist",
    "right_wrist": "right_wrist",
    "wrist_right": "right_wrist",
}
_CAMERA_MANIFEST_ORDER = ("head", "left_wrist", "right_wrist")


def _enabled_camera_manifest(camera_sources: dict[str, Any]) -> list[str]:
    if not isinstance(camera_sources, dict):
        raise TypeError("camera_sources must be the explicit camera source mapping")

    enabled = []
    for camera_name, source in camera_sources.items():
        if source is None:
            continue
        canonical_name = _CAMERA_NAME_ALIASES.get(str(camera_name).strip())
        if canonical_name is None:
            raise ValueError(f"unsupported camera source name: {camera_name}")
        if canonical_name in enabled:
            raise ValueError(f"duplicate camera source name: {canonical_name}")
        enabled.append(canonical_name)
    return sorted(enabled, key=_CAMERA_MANIFEST_ORDER.index)


@dataclass
class RecordingCommandResult:
    record_running: bool
    record_toggle: bool
    record_cancel: bool


@dataclass
class RecordingFrameResult:
    record_running: bool
    ready: bool
    should_continue_frame: bool = False


@dataclass
class TeleopRecordingState:
    record_start_monotonic_ns: int | None = None
    waiting_for_first_frame: bool = False
    pending_samples: deque = field(default_factory=deque)
    last_wait_log_ns: int = 0
    last_enqueued_primary_frame_id: object | None = None
    finalizing: bool = False
    canceling: bool = False
    save_requested: bool = False
    max_pending_sample_count: int = 0
    processed_sample_count: int = 0
    dropped_sample_count: int = 0
    last_alignment_worker_ms: float = 0.0
    last_alignment_worker_status: str = "idle"
    camera_feeder_enqueued_sample_count: int = 0
    camera_feeder_last_frame_seq: int | None = None
    camera_feeder_last_sample_monotonic_ns: int | None = None
    camera_feeder_status: str = "idle"


@dataclass(frozen=True)
class _RecordingSourceSnapshot:
    camera_sources: dict[str, Any]
    state_history: deque
    action_history: deque
    base_state_history: deque | None
    base_height_history: deque | None
    base_action_history: deque | None
    slam_tf_history: deque | None
    teleop_input_perf_counter_ns: int
    record_min_timestamp_ns: int
    arm_ik: Any
    get_wrist_poses: Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]]
    get_wrist_poses_base_link: Callable[[Any, np.ndarray, float, float], tuple[np.ndarray, np.ndarray]] | None
    get_dex1_tcp_poses_base_link: Callable[[np.ndarray, float, float], tuple[np.ndarray, np.ndarray]] | None
    sim_state_subscriber: Any
    source_version: int


class TeleopRecordingFlow:
    """Owns teleop episode lifecycle, camera alignment and frame writes."""

    def __init__(
        self,
        *,
        args,
        recorder,
        log,
        reset_callback: Callable[[], None] | None = None,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        dual_gripper_data_lock=None,
        dual_gripper_state_array=None,
        dual_gripper_action_array=None,
        validation_manager=None,
    ):
        self.args = args
        self.recorder = recorder
        self.log = log
        self.reset_callback = reset_callback
        self.dual_hand_data_lock = dual_hand_data_lock
        self.dual_hand_state_array = dual_hand_state_array
        self.dual_hand_action_array = dual_hand_action_array
        self.dual_gripper_data_lock = dual_gripper_data_lock
        self.dual_gripper_state_array = dual_gripper_state_array
        self.dual_gripper_action_array = dual_gripper_action_array
        self.validation_manager = validation_manager
        self.state = TeleopRecordingState()
        self._condition = threading.Condition()
        self._latest_sources: _RecordingSourceSnapshot | None = None
        self._worker_shutdown_requested = False
        self._episode_generation = 0
        self._worker: threading.Thread | None = None
        self._camera_feeder: threading.Thread | None = None
        self._camera_sources: dict[str, Any] = {}
        self._static_camera_sources: dict[str, Any] | None = None
        self._static_arm_ik: Any = None
        self._static_get_wrist_poses: Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None
        self._static_get_wrist_poses_base_link: Callable[[Any, np.ndarray, float, float], tuple[np.ndarray, np.ndarray]] | None = None
        self._static_get_dex1_tcp_poses_base_link: Callable[[np.ndarray, float, float], tuple[np.ndarray, np.ndarray]] | None = None
        self._static_sim_state_subscriber: Any = None

        frequency = max(float(args.frequency), 1e-6)
        base_history_size = max(32, int(getattr(args, "base_history_size", 512)))
        self._state_history = deque(maxlen=max(32, int(frequency * 6)))
        self._action_history = deque(maxlen=max(32, int(frequency * 6)))
        self._base_state_history = deque(maxlen=base_history_size)
        self._base_height_history = deque(maxlen=base_history_size)
        self._base_action_history = deque(maxlen=base_history_size)
        self._slam_tf_history = deque(maxlen=base_history_size * 5)
        self._history_after_t_ns = {
            "state": None,
            "action": None,
            "base_state": None,
            "base_height": None,
            "base_action": None,
            "slam_tf": None,
        }
        self._source_version = 0
        self.camera_align_max_delta_ns = int(max(20_000_000, (1.5 / frequency) * 1e9))
        self.state_align_max_delta_ns = int(max(20_000_000, (1.25 / frequency) * 1e9))
        self.action_align_max_delta_ns = self.state_align_max_delta_ns
        self.state_nearest_fallback_max_delta_ns = int(max(80_000_000, (2.5 / frequency) * 1e9))
        self.action_nearest_fallback_max_delta_ns = self.state_nearest_fallback_max_delta_ns
        self.record_future_wait_timeout_ns = int(max(120_000_000, (2.0 / frequency) * 1e9))
        self.pending_sample_timeout_ns = int(1_000_000_000)
        self.max_pending_samples = max(120, int(frequency * 8.0))
        self.record_base = bool(getattr(args, "record_base", False))
        self.record_slam_map_pose = bool(getattr(args, "record_slam_map_pose", False))
        self.record_base_velocity_only = bool(getattr(args, "record_base_velocity_only", False))
        self._record_mobile_training_state = bool(
            getattr(args, "record_mobile_training_state", False)
        )
        if self.record_base_velocity_only and not self.record_base:
            raise ValueError("record_base_velocity_only requires record_base")
        if self.record_base_velocity_only and self.record_slam_map_pose:
            raise ValueError("record_base_velocity_only cannot be combined with record_slam_map_pose")
        self.base_state_max_delta_ns = int(max(1_000_000, float(getattr(args, "base_state_max_age_ms", 131.578947)) * 1e6))
        self.base_height_max_delta_ns = int(150.0 * 1e6)
        self.slam_tf_max_delta_ns = int(max(1_000_000, float(getattr(args, "slam_tf_max_age_ms", 125.0)) * 1e6))
        self.base_height_required = bool(str(getattr(args, "base_height_topic", "/hispeed_state") or "").strip())

    def close(self) -> None:
        """Stop recording workers before camera and writer resources close."""
        with self._condition:
            if self._worker is None and self._camera_feeder is None:
                return
            episode_has_not_started = bool(
                self.state.waiting_for_first_frame
                or (
                    self.state.record_start_monotonic_ns is not None
                    and self._latest_sources is None
                    and not self.state.pending_samples
                    and self.state.processed_sample_count == 0
                )
            )
            if episode_has_not_started:
                # An armed episode has no valid sample boundary yet.  Preserve the
                # normal stop semantics: discard it instead of saving an empty episode.
                self._episode_generation += 1
                self.state.waiting_for_first_frame = False
                self.state.finalizing = False
                self.state.canceling = True
                self.state.save_requested = False
                self.state.pending_samples.clear()
                self.state.last_enqueued_primary_frame_id = None
                self.state.last_alignment_worker_status = "canceling_on_shutdown"
            elif self.state.record_start_monotonic_ns is not None and not self.state.canceling:
                self.state.waiting_for_first_frame = False
                self.state.finalizing = True
                self.state.last_alignment_worker_status = "finalizing_on_shutdown"
            self._worker_shutdown_requested = True
            self._condition.notify_all()
        if self._camera_feeder is not None:
            self._camera_feeder.join()
        if self._camera_feeder is not None and self._camera_feeder.is_alive():
            raise RuntimeError("teleop recording camera feeder did not stop")
        if self._worker is not None:
            self._worker.join()
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError("teleop recording alignment worker did not stop")

    def _ensure_worker_started(self) -> None:
        with self._condition:
            if self._worker is not None and not self._worker.is_alive():
                raise RuntimeError("teleop recording alignment worker cannot be restarted after exiting")
            if self._camera_feeder is not None and not self._camera_feeder.is_alive():
                raise RuntimeError("teleop recording camera feeder cannot be restarted after exiting")
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._run_alignment_worker,
                    name="teleop_record_alignment",
                    daemon=True,
                )
                self._worker.start()
            if self._camera_feeder is None:
                self._camera_feeder = threading.Thread(
                    target=self._run_camera_feeder,
                    name="teleop_record_camera_feeder",
                    daemon=True,
                )
                self._camera_feeder.start()

    def assert_worker_healthy(self) -> None:
        if self._worker is not None and not self._worker.is_alive():
            raise RuntimeError(
                "teleop recording alignment worker exited unexpectedly; "
                "inspect the worker traceback above"
            )
        if self._camera_feeder is not None and not self._camera_feeder.is_alive():
            raise RuntimeError(
                "teleop recording camera feeder exited unexpectedly; "
                "inspect the feeder traceback above"
            )

    def status_snapshot(self) -> dict[str, object]:
        with self._condition:
            pending_count = len(self.state.pending_samples)
            oldest_pending_age_ms = 0.0
            if self.state.pending_samples:
                oldest_pending_age_ms = max(
                    0.0,
                    (time.monotonic_ns() - int(self.state.pending_samples[0]["sample_monotonic_ns"])) / 1e6,
                )
            return {
                "waiting_for_first_frame": bool(self.state.waiting_for_first_frame),
                "pending_samples": int(pending_count),
                "max_pending_samples": int(self.state.max_pending_sample_count),
                "oldest_pending_age_ms": float(oldest_pending_age_ms),
                "processed_sample_count": int(self.state.processed_sample_count),
                "dropped_sample_count": int(self.state.dropped_sample_count),
                "last_alignment_worker_ms": float(self.state.last_alignment_worker_ms),
                "last_alignment_worker_status": str(self.state.last_alignment_worker_status),
                "worker_alive": bool(self._worker is None or self._worker.is_alive()),
                "camera_feeder_alive": bool(self._camera_feeder is None or self._camera_feeder.is_alive()),
                "camera_feeder_status": str(self.state.camera_feeder_status),
                "camera_feeder_enqueued_sample_count": int(self.state.camera_feeder_enqueued_sample_count),
                "camera_feeder_last_frame_seq": self.state.camera_feeder_last_frame_seq,
                "camera_feeder_last_sample_monotonic_ns": self.state.camera_feeder_last_sample_monotonic_ns,
                "finalizing": bool(self.state.finalizing),
                "canceling": bool(self.state.canceling),
                "save_requested": bool(self.state.save_requested),
                "record_start_monotonic_ns": self.state.record_start_monotonic_ns,
            }

    def needs_source_updates(self) -> bool:
        """Return whether an episode can still consume source history updates."""
        with self._condition:
            return bool(
                not self.state.canceling
                and (
                    self.state.record_start_monotonic_ns is not None
                    or self.state.waiting_for_first_frame
                    or self.state.finalizing
                )
            )

    def handle_commands(
        self,
        *,
        record_running: bool,
        record_toggle: bool,
        record_cancel: bool,
        camera_sources: dict[str, Any],
    ) -> RecordingCommandResult:
        """Consume operator recording commands and update episode lifecycle state."""
        self.assert_worker_healthy()
        if record_cancel:
            record_cancel = False
            if record_running or self.state.waiting_for_first_frame or self.state.finalizing:
                record_running = False
                self._request_cancel()
                self.log.info("[RECORD_CANCEL] requested; alignment worker will discard pending samples before canceling the episode.")
            else:
                self.log.info("[RECORD_CANCEL] ignored: no active recording to cancel.")

        if record_toggle:
            record_toggle = False
            if not record_running and not self.state.waiting_for_first_frame and not self.state.finalizing:
                if self.validation_manager is not None and self.validation_manager.is_busy():
                    status = self.validation_manager.status()
                    active_episode = status.get("current_episode_dir") or "queued episode"
                    self.log.warning(
                        "[RECORD_VALIDATE] recording start rejected: validation is still running for %s.",
                        active_episode,
                    )
                    return RecordingCommandResult(
                        record_running=False,
                        record_toggle=record_toggle,
                        record_cancel=record_cancel,
                    )
                enabled_cameras = _enabled_camera_manifest(camera_sources)
                self._validate_camera_sources_for_feeder(camera_sources)
                if self.recorder.create_episode(enabled_cameras=enabled_cameras):
                    self._ensure_worker_started()
                    with self._condition:
                        self._episode_generation += 1
                        self.state.record_start_monotonic_ns = int(time.monotonic_ns())
                        self.state.pending_samples.clear()
                        self.state.last_enqueued_primary_frame_id = None
                        self.state.waiting_for_first_frame = True
                        self.state.finalizing = False
                        self.state.canceling = False
                        self.state.save_requested = False
                        self.state.max_pending_sample_count = 0
                        self.state.processed_sample_count = 0
                        self.state.dropped_sample_count = 0
                        self.state.last_alignment_worker_ms = 0.0
                        self.state.last_alignment_worker_status = "armed"
                        self.state.camera_feeder_enqueued_sample_count = 0
                        self.state.camera_feeder_last_frame_seq = None
                        self.state.camera_feeder_last_sample_monotonic_ns = None
                        self.state.camera_feeder_status = "waiting_for_first_frame"
                        self._camera_sources = dict(camera_sources)
                        self._latest_sources = None
                        self._condition.notify_all()
                    record_running = True
                    self.log.info("[RECORD_ALIGN] episode armed, waiting for first post-start camera frame before recording.")
                else:
                    self.log.error("Failed to create episode. Recording not started.")
            else:
                record_running = False
                self._request_finalize()
                self.log.info("[RECORD_ALIGN] recording stop requested; waiting for pending samples to resolve before saving episode.")

        return RecordingCommandResult(
            record_running=record_running,
            record_toggle=record_toggle,
            record_cancel=record_cancel,
        )

    @staticmethod
    def _validate_camera_sources_for_feeder(camera_sources: dict[str, Any]) -> None:
        for camera_name, source in camera_sources.items():
            if source is None:
                continue
            wait_next = getattr(source, "wait_for_next_frame", None)
            if not callable(wait_next):
                raise RuntimeError(
                    f"camera source {camera_name!r} does not implement wait_for_next_frame(); "
                    "camera-driven recording cannot start"
                )
            get_nearest = getattr(source, "get_nearest", None)
            if not callable(get_nearest):
                raise RuntimeError(
                    f"camera source {camera_name!r} does not implement get_nearest(); "
                    "camera-driven recording cannot establish the post-start boundary"
                )

    def publish_sources(
        self,
        *,
        camera_sources: dict[str, Any],
        state_history: deque,
        action_history: deque,
        teleop_input_perf_counter_ns: int,
        arm_ik,
        get_wrist_poses: Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]],
        get_wrist_poses_base_link: Callable[[Any, np.ndarray, float, float], tuple[np.ndarray, np.ndarray]] | None = None,
        get_dex1_tcp_poses_base_link: Callable[[np.ndarray, float, float], tuple[np.ndarray, np.ndarray]] | None = None,
        sim_state_subscriber=None,
        base_state_history: deque | None = None,
        base_height_history: deque | None = None,
        base_action_history: deque | None = None,
        slam_tf_history: deque | None = None,
    ) -> None:
        """Publish a point-in-time source snapshot for the camera-driven workers."""
        self.assert_worker_healthy()

        with self._condition:
            waiting_for_first_frame = bool(self.state.waiting_for_first_frame)
            finalizing = bool(self.state.finalizing)
            active = self.state.record_start_monotonic_ns is not None
        if not (active or waiting_for_first_frame or finalizing):
            return

        with self._condition:
            record_min_timestamp_ns = int(self.state.record_start_monotonic_ns or time.monotonic_ns())
        self._publish_source_snapshot(
            camera_sources=camera_sources,
            state_history=state_history,
            action_history=action_history,
            base_state_history=base_state_history,
            base_height_history=base_height_history,
            base_action_history=base_action_history,
            slam_tf_history=slam_tf_history,
            teleop_input_perf_counter_ns=teleop_input_perf_counter_ns,
            record_min_timestamp_ns=record_min_timestamp_ns,
            arm_ik=arm_ik,
            get_wrist_poses=get_wrist_poses,
            get_wrist_poses_base_link=get_wrist_poses_base_link,
            get_dex1_tcp_poses_base_link=get_dex1_tcp_poses_base_link,
            sim_state_subscriber=sim_state_subscriber,
        )

    def _reset_active_state(self):
        with self._condition:
            self.state.waiting_for_first_frame = False
            self.state.record_start_monotonic_ns = None
            self.state.pending_samples.clear()
            self.state.last_enqueued_primary_frame_id = None
            self.state.finalizing = False
            self.state.canceling = False
            self.state.save_requested = False
            self.state.camera_feeder_status = "idle"
            self._latest_sources = None
            self._camera_sources = {}
            self._static_camera_sources = None
            self._static_arm_ik = None
            self._static_get_wrist_poses = None
            self._static_get_wrist_poses_base_link = None
            self._static_get_dex1_tcp_poses_base_link = None
            self._static_sim_state_subscriber = None
            self._state_history.clear()
            self._action_history.clear()
            self._base_state_history.clear()
            self._base_height_history.clear()
            self._base_action_history.clear()
            self._slam_tf_history.clear()
            for key in self._history_after_t_ns:
                self._history_after_t_ns[key] = None
            self._source_version += 1
            self._condition.notify_all()

    def _request_finalize(self) -> None:
        with self._condition:
            if self.state.waiting_for_first_frame:
                self._episode_generation += 1
                self._reset_active_state()
                self.recorder.cancel_episode()
                self.log.info("[RECORD_ALIGN] recording stopped before the first synchronized camera frame; episode canceled.")
                return
            self.state.finalizing = True
            self.state.last_alignment_worker_status = "finalizing"
            self._condition.notify_all()

    def _request_cancel(self) -> None:
        with self._condition:
            self._episode_generation += 1
            self.state.waiting_for_first_frame = False
            self.state.finalizing = False
            self.state.canceling = True
            self.state.save_requested = False
            self.state.pending_samples.clear()
            self.state.last_enqueued_primary_frame_id = None
            self.state.last_alignment_worker_status = "canceling"
            self.state.camera_feeder_status = "canceling"
            self._condition.notify_all()

    def _publish_source_snapshot(
        self,
        *,
        camera_sources,
        state_history,
        action_history,
        base_state_history,
        base_height_history,
        base_action_history,
        slam_tf_history,
        teleop_input_perf_counter_ns,
        record_min_timestamp_ns,
        arm_ik,
        get_wrist_poses,
        get_wrist_poses_base_link,
        get_dex1_tcp_poses_base_link,
        sim_state_subscriber,
    ) -> None:
        with self._condition:
            if self._static_camera_sources is None:
                self._static_camera_sources = dict(self._camera_sources or camera_sources)
                self._static_arm_ik = arm_ik
                self._static_get_wrist_poses = get_wrist_poses
                self._static_get_wrist_poses_base_link = get_wrist_poses_base_link
                self._static_get_dex1_tcp_poses_base_link = get_dex1_tcp_poses_base_link
                self._static_sim_state_subscriber = sim_state_subscriber
            else:
                if self._static_camera_sources.keys() != camera_sources.keys():
                    raise RuntimeError("recording camera sources changed during an active episode")
                for camera_name, source in camera_sources.items():
                    if self._static_camera_sources.get(camera_name) is not source:
                        raise RuntimeError(f"recording camera source changed during an active episode: {camera_name}")
                if self._static_arm_ik is not arm_ik:
                    raise RuntimeError("recording arm IK dependency changed during an active episode")
                if self._static_get_wrist_poses is not get_wrist_poses:
                    raise RuntimeError("recording wrist FK dependency changed during an active episode")

            changed = False
            changed |= self._ingest_new_history("state", state_history, self._state_history)
            changed |= self._ingest_new_history("action", action_history, self._action_history)
            changed |= self._ingest_new_history("base_state", base_state_history, self._base_state_history)
            changed |= self._ingest_new_history("base_height", base_height_history, self._base_height_history)
            changed |= self._ingest_new_history("base_action", base_action_history, self._base_action_history)
            changed |= self._ingest_new_history("slam_tf", slam_tf_history, self._slam_tf_history)
            if changed:
                self._source_version += 1

            source_version = int(self._source_version)
            static_camera_sources = self._static_camera_sources
            static_arm_ik = self._static_arm_ik
            static_get_wrist_poses = self._static_get_wrist_poses
            static_get_wrist_poses_base_link = self._static_get_wrist_poses_base_link
            static_get_dex1_tcp_poses_base_link = self._static_get_dex1_tcp_poses_base_link
            static_sim_state_subscriber = self._static_sim_state_subscriber

        snapshot = _RecordingSourceSnapshot(
            camera_sources=static_camera_sources,
            state_history=self._state_history,
            action_history=self._action_history,
            base_state_history=self._base_state_history,
            base_height_history=self._base_height_history,
            base_action_history=self._base_action_history,
            slam_tf_history=self._slam_tf_history,
            teleop_input_perf_counter_ns=int(teleop_input_perf_counter_ns),
            record_min_timestamp_ns=int(record_min_timestamp_ns),
            arm_ik=static_arm_ik,
            get_wrist_poses=static_get_wrist_poses,
            get_wrist_poses_base_link=static_get_wrist_poses_base_link,
            get_dex1_tcp_poses_base_link=static_get_dex1_tcp_poses_base_link,
            sim_state_subscriber=static_sim_state_subscriber,
            source_version=source_version,
        )
        with self._condition:
            had_snapshot = self._latest_sources is not None
            self._latest_sources = snapshot
            if not had_snapshot or changed:
                self._condition.notify_all()

    def _ingest_new_history(self, name: str, incoming: deque | None, destination: deque) -> bool:
        if incoming is None:
            return False
        after_t_ns = self._history_after_t_ns[name]
        new_entries = []
        for entry in reversed(incoming):
            if not isinstance(entry, dict) or "t_ns" not in entry:
                raise RuntimeError(f"recording {name} history entry is missing t_ns")
            t_ns = int(entry["t_ns"])
            if after_t_ns is not None and t_ns <= int(after_t_ns):
                break
            new_entries.append(entry)

        changed = False
        for entry in reversed(new_entries):
            t_ns = int(entry["t_ns"])
            if after_t_ns is not None and t_ns <= int(after_t_ns):
                raise RuntimeError(
                    f"recording {name} history timestamps must be strictly increasing: "
                    f"last t_ns={after_t_ns}, new t_ns={t_ns}"
                )
            destination.append(entry)
            self._history_after_t_ns[name] = t_ns
            after_t_ns = t_ns
            changed = True
        return changed

    def source_history_cursors(self) -> dict[str, int | None]:
        with self._condition:
            return dict(self._history_after_t_ns)

    def _read_end_effector_sample(self):
        args = self.args
        if (not args.no_gripper) and args.ee == "dex3" and args.input_mode == "hand":
            with self.dual_hand_data_lock:
                return {
                    "left_ee_state": list(self.dual_hand_state_array[:7]),
                    "right_ee_state": list(self.dual_hand_state_array[-7:]),
                    "left_hand_action": list(self.dual_hand_action_array[:7]),
                    "right_hand_action": list(self.dual_hand_action_array[-7:]),
                }
        if (not args.no_gripper) and args.ee == "dex1" and args.input_mode in {"hand", "controller"}:
            with self.dual_gripper_data_lock:
                return {
                    "left_ee_state": [self.dual_gripper_state_array[0]],
                    "right_ee_state": [self.dual_gripper_state_array[1]],
                    "left_hand_action": [self.dual_gripper_action_array[0]],
                    "right_hand_action": [self.dual_gripper_action_array[1]],
                }
        if (
            (not args.no_gripper)
            and args.ee in {"inspire_dfx", "inspire_ftp", "brainco"}
            and args.input_mode == "hand"
        ):
            with self.dual_hand_data_lock:
                return {
                    "left_ee_state": list(self.dual_hand_state_array[:6]),
                    "right_ee_state": list(self.dual_hand_state_array[-6:]),
                    "left_hand_action": list(self.dual_hand_action_array[:6]),
                    "right_hand_action": list(self.dual_hand_action_array[-6:]),
                }
        return {
            "left_ee_state": [],
            "right_ee_state": [],
            "left_hand_action": [],
            "right_hand_action": [],
        }

    def _maybe_start_after_first_camera_frame(self, record_running: bool, camera_sources: dict[str, Any]) -> RecordingFrameResult:
        required_sources = [(name, src) for name, src in camera_sources.items() if src is not None]
        if not required_sources:
            with self._condition:
                if (
                    not self.state.waiting_for_first_frame
                    or self.state.canceling
                    or self._worker_shutdown_requested
                ):
                    return RecordingFrameResult(
                        record_running=False,
                        ready=True,
                        should_continue_frame=True,
                    )
                self.state.waiting_for_first_frame = False
                self.state.last_alignment_worker_status = "recording"
                self._condition.notify_all()
            self.log.info("[RECORD_ALIGN] no camera source configured, recording starts immediately.")
            return RecordingFrameResult(record_running=True, ready=True)

        wait_target_ns = int(time.monotonic_ns())
        first_frame_times = []
        for _, source in required_sources:
            _, meta = source.get_nearest(
                wait_target_ns,
                max_delta_ns=None,
                min_monotonic_ns=self.state.record_start_monotonic_ns,
                copy=False,
            )
            meta_ts = camera_meta_monotonic_ns(meta)
            if meta is None or meta_ts is None:
                now_ns = int(time.monotonic_ns())
                with self._condition:
                    if now_ns - self.state.last_wait_log_ns > 1_000_000_000:
                        self.log.info("[RECORD_ALIGN] waiting for first post-start camera frame...")
                        self.state.last_wait_log_ns = now_ns
                return RecordingFrameResult(record_running=record_running, ready=True, should_continue_frame=True)
            first_frame_times.append(meta_ts)

        with self._condition:
            if (
                not self.state.waiting_for_first_frame
                or self.state.canceling
                or self._worker_shutdown_requested
            ):
                return RecordingFrameResult(
                    record_running=False,
                    ready=True,
                    should_continue_frame=True,
                )
            self.state.record_start_monotonic_ns = int(max(first_frame_times))
            self.state.waiting_for_first_frame = False
            self.state.last_alignment_worker_status = "recording"
            self._condition.notify_all()
        self.log.info(
            "[RECORD_ALIGN] first post-start camera frame received, recording begins at monotonic_ns=%d",
            self.state.record_start_monotonic_ns,
        )
        return RecordingFrameResult(record_running=True, ready=True, should_continue_frame=True)

    def _enqueue_primary_camera_sample(
        self,
        *,
        camera_sources: dict[str, Any],
        record_min_timestamp_ns: int,
        ee_sample: dict[str, list],
        teleop_input_perf_counter_ns: int,
    ):
        primary_camera_name = None
        primary_source = None
        for camera_name in ("head", "left_wrist", "right_wrist"):
            source = camera_sources.get(camera_name)
            if source is not None:
                primary_camera_name = camera_name
                primary_source = source
                break
        if primary_source is None:
            return

        primary_frame, primary_meta = primary_source.get_latest(copy=False)
        if primary_frame is None or primary_meta is None:
            return

        self._enqueue_primary_camera_frame(
            primary_camera_name=primary_camera_name,
            primary_frame=primary_frame,
            primary_meta=primary_meta,
            record_min_timestamp_ns=record_min_timestamp_ns,
            ee_sample=ee_sample,
            teleop_input_perf_counter_ns=teleop_input_perf_counter_ns,
        )

    def _enqueue_primary_camera_frame(
        self,
        *,
        primary_camera_name: str,
        primary_frame,
        primary_meta: dict,
        record_min_timestamp_ns: int,
        ee_sample: dict[str, list],
        teleop_input_perf_counter_ns: int,
    ) -> None:
        primary_ts_value = camera_meta_monotonic_ns(primary_meta)
        if primary_ts_value is None:
            raise RuntimeError("primary camera metadata is missing host monotonic timestamp")
        primary_ts = int(primary_ts_value)
        primary_frame_seq = primary_meta.get("frame_seq")
        primary_frame_id = camera_frame_identity(primary_camera_name, primary_meta)
        if primary_frame_id is None:
            raise RuntimeError(f"primary camera frame identity is unavailable: camera={primary_camera_name}")
        with self._condition:
            already_pending = False
            if self.state.pending_samples:
                last_pending = self.state.pending_samples[-1]
                already_pending = (
                    last_pending.get("primary_camera_name") == primary_camera_name
                    and last_pending.get("frame_seq") == primary_frame_seq
                    and last_pending.get("sample_monotonic_ns") == primary_ts
                )
            if (
                not already_pending
                and primary_ts >= int(record_min_timestamp_ns)
                and primary_frame_id is not None
                and primary_frame_id != self.state.last_enqueued_primary_frame_id
                and not self.state.finalizing
                and not self.state.canceling
            ):
                if len(self.state.pending_samples) >= self.max_pending_samples:
                    raise RuntimeError(
                        "RECORD_PENDING_QUEUE_OVERFLOW "
                        f"pending={len(self.state.pending_samples)} limit={self.max_pending_samples}"
                    )
                self.state.pending_samples.append(
                    {
                        "generation": int(self._episode_generation),
                        "enqueue_wall_time_ns": int(time.time_ns()),
                        "sample_monotonic_ns": primary_ts,
                        "primary_camera_name": primary_camera_name,
                        "primary_frame": primary_frame,
                        "primary_meta": primary_meta,
                        "teleop_input_perf_counter_ns": int(teleop_input_perf_counter_ns),
                        "frame_seq": primary_frame_seq,
                        **ee_sample,
                    }
                )
                self.state.last_enqueued_primary_frame_id = primary_frame_id
                self.state.max_pending_sample_count = max(
                    int(self.state.max_pending_sample_count),
                    int(len(self.state.pending_samples)),
                )
                self._condition.notify_all()

    @staticmethod
    def _camera_receiver_cursor(meta: dict) -> int:
        value = meta.get("receiver_frame_seq", meta.get("frame_seq"))
        if value is None:
            raise RuntimeError("camera frame metadata is missing receiver_frame_seq/frame_seq")
        return int(value)

    def _run_camera_feeder(self) -> None:
        """Turn every primary-camera arrival into one pending recording sample."""
        observed_generation = -1
        primary_cursor: int | None = None
        while True:
            with self._condition:
                if self._worker_shutdown_requested:
                    return
                generation = int(self._episode_generation)
                waiting_for_first_frame = bool(self.state.waiting_for_first_frame)
                record_start_ns = self.state.record_start_monotonic_ns
                finalizing = bool(self.state.finalizing)
                canceling = bool(self.state.canceling)
                camera_sources = dict(self._camera_sources)
                source_snapshot = self._latest_sources
                if record_start_ns is None or finalizing or canceling:
                    self.state.camera_feeder_status = "idle" if record_start_ns is None else "finalizing"
                    self._condition.wait()
                    continue

            if generation != observed_generation:
                observed_generation = generation
                primary_cursor = None

            if waiting_for_first_frame:
                start_result = self._maybe_start_after_first_camera_frame(True, camera_sources)
                if start_result.should_continue_frame:
                    time.sleep(0.005)
                    continue
                primary_cursor = None
                continue

            primary_camera_name = next(
                (name for name in ("head", "left_wrist", "right_wrist") if camera_sources.get(name) is not None),
                None,
            )
            if primary_camera_name is None:
                with self._condition:
                    self.state.camera_feeder_status = "recording_without_camera"
                    if (
                        generation == int(self._episode_generation)
                        and self.state.record_start_monotonic_ns is not None
                        and not self.state.finalizing
                        and not self.state.canceling
                        and not self._worker_shutdown_requested
                    ):
                        self._condition.wait()
                continue
            if source_snapshot is None:
                with self._condition:
                    self.state.camera_feeder_status = "waiting_for_control_history"
                    if (
                        self._latest_sources is None
                        and generation == int(self._episode_generation)
                        and not self.state.finalizing
                        and not self.state.canceling
                    ):
                        self._condition.wait()
                continue

            source = camera_sources[primary_camera_name]
            frame, meta = source.wait_for_next_frame(
                after_receiver_frame_seq=primary_cursor,
                min_monotonic_ns=int(record_start_ns),
                timeout_sec=0.05,
                copy=False,
            )
            if frame is None or meta is None:
                continue
            primary_cursor = self._camera_receiver_cursor(meta)
            sample_timestamp_ns = camera_meta_monotonic_ns(meta)
            if sample_timestamp_ns is None:
                raise RuntimeError(f"primary camera frame has no host timestamp: camera={primary_camera_name}")

            with self._condition:
                if generation != int(self._episode_generation) or self.state.finalizing or self.state.canceling:
                    continue
                source_snapshot = self._latest_sources
                if source_snapshot is None:
                    continue
            self._enqueue_primary_camera_frame(
                primary_camera_name=primary_camera_name,
                primary_frame=frame,
                primary_meta=meta,
                record_min_timestamp_ns=int(record_start_ns),
                ee_sample=self._read_end_effector_sample(),
                teleop_input_perf_counter_ns=int(source_snapshot.teleop_input_perf_counter_ns),
            )
            with self._condition:
                self.state.camera_feeder_enqueued_sample_count += 1
                self.state.camera_feeder_last_frame_seq = int(primary_cursor)
                self.state.camera_feeder_last_sample_monotonic_ns = int(sample_timestamp_ns)
                self.state.camera_feeder_status = "recording"

    def _run_alignment_worker(self) -> None:
        while True:
            with self._condition:
                if self.state.canceling:
                    self.state.canceling = False
                    self._latest_sources = None
                    action = "cancel"
                    pending = None
                    sources = None
                elif self.state.finalizing and not self.state.pending_samples:
                    if not self.state.save_requested:
                        self.state.save_requested = True
                        action = "save"
                        pending = None
                        sources = None
                    elif self.recorder.is_ready():
                        self._reset_active_state()
                        self.state.last_alignment_worker_status = "idle"
                        if self._worker_shutdown_requested:
                            return
                        self._condition.wait()
                        continue
                    else:
                        self._condition.wait(timeout=0.01)
                        continue
                elif self.state.pending_samples and self._latest_sources is not None:
                    pending = self.state.pending_samples[0]
                    sources = self._latest_sources
                    action = "process"
                elif self._worker_shutdown_requested:
                    return
                else:
                    self._condition.wait()
                    continue

            if action == "cancel":
                self.recorder.cancel_episode()
                if self.reset_callback is not None:
                    self.reset_callback()
                self._reset_active_state()
                self.log.info("[RECORD_CANCEL] alignment worker canceled episode after pending samples were discarded.")
                continue
            if action == "save":
                self.recorder.save_episode()
                if self.reset_callback is not None:
                    self.reset_callback()
                self.log.info("[RECORD_ALIGN] all pending samples resolved; episode save requested.")
                continue

            started_ns = time.perf_counter_ns()
            status = self._process_pending_sample(pending, sources)
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1e6
            with self._condition:
                self.state.last_alignment_worker_ms = float(elapsed_ms)
                self.state.last_alignment_worker_status = str(status)
                if status == "wait":
                    waiting_source_version = int(self._source_version)
                    waiting_generation = int(self._episode_generation)
                    waiting_finalizing = bool(self.state.finalizing)
                    waiting_pending = self.state.pending_samples[0] if self.state.pending_samples else None
                    wait_deadline_ns = (
                        int(pending["sample_monotonic_ns"])
                        + int(self.record_future_wait_timeout_ns)
                    )
                    while True:
                        if self._worker_shutdown_requested:
                            return
                        if self.state.canceling or bool(self.state.finalizing) != waiting_finalizing:
                            break
                        if int(self._source_version) != waiting_source_version:
                            break
                        if int(self._episode_generation) != waiting_generation:
                            break
                        if not self.state.pending_samples or self.state.pending_samples[0] is not waiting_pending:
                            break
                        remaining_ns = wait_deadline_ns - int(time.monotonic_ns())
                        if remaining_ns <= 0:
                            break
                        self._condition.wait(timeout=remaining_ns / 1e9)
                    continue
                if self.state.pending_samples and self.state.pending_samples[0] is pending:
                    self.state.pending_samples.popleft()
                    if status == "drop":
                        self.state.dropped_sample_count += 1
                    else:
                        self.state.processed_sample_count += 1
                self._condition.notify_all()

    def _pending_generation_is_current(self, pending: dict) -> bool:
        with self._condition:
            return (
                int(pending["generation"]) == int(self._episode_generation)
                and not self.state.canceling
            )

    def _process_pending_sample(self, pending: dict, sources: _RecordingSourceSnapshot) -> str:
        with self._condition:
            if not self._pending_generation_is_current(pending):
                return "drop"
            end_of_stream = bool(self.state.finalizing or self.state.canceling or self._worker_shutdown_requested)
            (
                status,
                aligned_state,
                aligned_action,
                aligned_base_state,
                aligned_base_height,
                aligned_base_action,
            ) = self._resolve_pending_alignment(
                state_history=sources.state_history,
                action_history=sources.action_history,
                base_state_history=sources.base_state_history,
                base_height_history=sources.base_height_history,
                base_action_history=sources.base_action_history,
                slam_tf_history=sources.slam_tf_history,
                sample_monotonic_ns=int(pending["sample_monotonic_ns"]),
                record_min_timestamp_ns=sources.record_min_timestamp_ns,
                end_of_stream=end_of_stream,
            )
        if status != "ready":
            return status
        sample_monotonic_ns = int(pending["sample_monotonic_ns"])

        colors, depths, camera_timestamps = self._aligned_camera_frames(
            pending=pending,
            camera_sources=sources.camera_sources,
            sample_monotonic_ns=sample_monotonic_ns,
            record_min_timestamp_ns=sources.record_min_timestamp_ns,
        )
        states, actions = self._build_state_action_payload(
            pending=pending,
            aligned_state=aligned_state,
            aligned_action=aligned_action,
            aligned_base_state=aligned_base_state,
            aligned_base_height=aligned_base_height,
            aligned_base_action=aligned_base_action,
            arm_ik=sources.arm_ik,
            get_wrist_poses=sources.get_wrist_poses,
            get_wrist_poses_base_link=sources.get_wrist_poses_base_link,
            get_dex1_tcp_poses_base_link=sources.get_dex1_tcp_poses_base_link,
        )
        if not self._pending_generation_is_current(pending):
            return "drop"
        self._add_record_item(
            pending=pending,
            sample_monotonic_ns=sample_monotonic_ns,
            aligned_state=aligned_state,
            aligned_action=aligned_action,
            aligned_base_state=aligned_base_state,
            aligned_base_height=aligned_base_height,
            aligned_base_action=aligned_base_action,
            colors=colors,
            depths=depths,
            states=states,
            actions=actions,
            camera_timestamps=camera_timestamps,
            sim_state_subscriber=sources.sim_state_subscriber,
        )
        return "recorded"

    def _resolve_pending_alignment(
        self,
        *,
        state_history: deque,
        action_history: deque,
        base_state_history: deque | None,
        base_height_history: deque | None,
        base_action_history: deque | None,
        slam_tf_history: deque | None,
        sample_monotonic_ns: int,
        record_min_timestamp_ns: int,
        end_of_stream: bool = False,
    ) -> tuple[str, dict | None, dict | None, dict | None, dict | None, dict | None]:
        state_earliest, state_latest = timed_buffer_bounds(state_history, min_timestamp_ns=record_min_timestamp_ns)
        action_earliest, action_latest = timed_buffer_bounds(action_history, min_timestamp_ns=record_min_timestamp_ns)
        sample_age_ns = int(time.monotonic_ns()) - int(sample_monotonic_ns)

        state_watermark = self._watermark_status(
            stream_name="state",
            earliest=state_earliest,
            latest=state_latest,
            target_ns=sample_monotonic_ns,
            record_min_timestamp_ns=record_min_timestamp_ns,
            sample_age_ns=sample_age_ns,
            end_of_stream=end_of_stream,
        )
        action_watermark = self._watermark_status(
            stream_name="action",
            earliest=action_earliest,
            latest=action_latest,
            target_ns=sample_monotonic_ns,
            record_min_timestamp_ns=record_min_timestamp_ns,
            sample_age_ns=sample_age_ns,
            end_of_stream=end_of_stream,
        )
        if state_watermark == "wait" or action_watermark == "wait":
            return "wait", None, None, None, None, None

        aligned_state = self._aligned_sample(
            state_history,
            sample_monotonic_ns,
            min_timestamp_ns=record_min_timestamp_ns,
        )
        aligned_action = self._aligned_sample(
            action_history,
            sample_monotonic_ns,
            min_timestamp_ns=record_min_timestamp_ns,
        )
        if aligned_state is None:
            raise RuntimeError(
                "RECORD_ALIGN_MISSING_REQUIRED_STREAM "
                f"stream=state target_monotonic_ns={sample_monotonic_ns}"
            )
        if aligned_action is None:
            raise RuntimeError(
                "RECORD_ALIGN_MISSING_REQUIRED_STREAM "
                f"stream=action target_monotonic_ns={sample_monotonic_ns}"
            )

        base_status, aligned_base_state, aligned_base_height, aligned_base_action = self._resolve_base_alignment(
            base_state_history=base_state_history,
            base_height_history=base_height_history,
            base_action_history=base_action_history,
            slam_tf_history=slam_tf_history,
            sample_monotonic_ns=sample_monotonic_ns,
            record_min_timestamp_ns=record_min_timestamp_ns,
            sample_age_ns=sample_age_ns,
            end_of_stream=end_of_stream,
        )
        if base_status != "ready":
            return base_status, None, None, None, None, None
        return "ready", aligned_state, aligned_action, aligned_base_state, aligned_base_height, aligned_base_action

    def _watermark_status(
        self,
        *,
        stream_name: str,
        earliest: int | None,
        latest: int | None,
        target_ns: int,
        record_min_timestamp_ns: int,
        sample_age_ns: int,
        end_of_stream: bool,
    ) -> str:
        if earliest is None or latest is None:
            if end_of_stream or sample_age_ns > self.record_future_wait_timeout_ns:
                raise RuntimeError(
                    "RECORD_ALIGN_MISSING_REQUIRED_STREAM "
                    f"stream={stream_name} target_monotonic_ns={target_ns} "
                    f"record_min_timestamp_ns={record_min_timestamp_ns}"
                )
            return "wait"
        if target_ns < earliest:
            return "ready" if end_of_stream or sample_age_ns > self.record_future_wait_timeout_ns else "wait"
        if target_ns > latest:
            if end_of_stream or sample_age_ns > self.record_future_wait_timeout_ns:
                return "ready"
            return "wait"
        return "ready"

    def _resolve_base_alignment(
        self,
        *,
        base_state_history: deque | None,
        base_height_history: deque | None,
        base_action_history: deque | None,
        sample_monotonic_ns: int,
        record_min_timestamp_ns: int,
        sample_age_ns: int,
        slam_tf_history: deque | None = None,
        end_of_stream: bool = False,
    ) -> tuple[str, dict | None, dict | None, dict | None]:
        if not self.record_base:
            return "ready", None, None, None
        if base_state_history is None or base_action_history is None:
            raise RuntimeError("--record-base requires base_state_history and base_action_history")
        if self.record_slam_map_pose and slam_tf_history is None:
            raise RuntimeError("--record-slam-map-pose requires slam_tf_history")
        if self.base_height_required and base_height_history is None:
            raise RuntimeError("--record-base requires base_height_history when --base-height-topic is set")

        base_state_earliest, base_state_latest = timed_buffer_bounds(
            base_state_history,
            min_timestamp_ns=record_min_timestamp_ns,
        )
        base_state_watermark = self._watermark_status(
            stream_name="base_state",
            earliest=base_state_earliest,
            latest=base_state_latest,
            target_ns=sample_monotonic_ns,
            record_min_timestamp_ns=record_min_timestamp_ns,
            sample_age_ns=sample_age_ns,
            end_of_stream=end_of_stream,
        )
        if base_state_watermark != "ready":
            return base_state_watermark, None, None, None

        if self.base_height_required:
            height_earliest, height_latest = timed_buffer_bounds(
                base_height_history,
                min_timestamp_ns=record_min_timestamp_ns,
            )
            watermark = self._watermark_status(
                stream_name="base_height",
                earliest=height_earliest,
                latest=height_latest,
                target_ns=sample_monotonic_ns,
                record_min_timestamp_ns=record_min_timestamp_ns,
                sample_age_ns=sample_age_ns,
                end_of_stream=end_of_stream,
            )
            if watermark != "ready":
                return watermark, None, None, None

        if self.record_slam_map_pose:
            slam_earliest, slam_latest = timed_buffer_bounds(
                slam_tf_history,
                min_timestamp_ns=record_min_timestamp_ns,
            )
            watermark = self._watermark_status(
                stream_name="slam_tf",
                earliest=slam_earliest,
                latest=slam_latest,
                target_ns=sample_monotonic_ns,
                record_min_timestamp_ns=record_min_timestamp_ns,
                sample_age_ns=sample_age_ns,
                end_of_stream=end_of_stream,
            )
            if watermark != "ready":
                return watermark, None, None, None

        aligned_base_state = self._interpolated_or_unbounded_nearest(
            interpolate_base_state_timed_sample_strict,
            base_state_history,
            sample_monotonic_ns,
            min_timestamp_ns=record_min_timestamp_ns,
        )
        aligned_base_action = self._aligned_sample(
            base_action_history,
            sample_monotonic_ns,
            min_timestamp_ns=record_min_timestamp_ns,
        )
        aligned_base_height = None
        aligned_slam_tf = None
        if self.base_height_required:
            aligned_base_height = self._interpolated_or_unbounded_nearest(
                interpolate_base_height_timed_sample_strict,
                base_height_history,
                sample_monotonic_ns,
                min_timestamp_ns=record_min_timestamp_ns,
            )
        if self.record_slam_map_pose:
            aligned_slam_tf = self._interpolated_or_unbounded_nearest(
                interpolate_slam_tf_timed_sample_strict,
                slam_tf_history,
                sample_monotonic_ns,
                min_timestamp_ns=record_min_timestamp_ns,
            )

        if aligned_base_state is None:
            raise RuntimeError(
                "RECORD_ALIGN_MISSING_REQUIRED_STREAM "
                f"stream=base_state target_monotonic_ns={sample_monotonic_ns}"
            )
        if aligned_base_action is None:
            raise RuntimeError(
                "RECORD_ALIGN_MISSING_REQUIRED_STREAM "
                f"stream=base_action target_monotonic_ns={sample_monotonic_ns}"
            )
        if self.base_height_required and aligned_base_height is None:
            raise RuntimeError(
                "RECORD_ALIGN_MISSING_REQUIRED_STREAM "
                f"stream=base_height target_monotonic_ns={sample_monotonic_ns}"
            )
        if self.record_slam_map_pose and aligned_slam_tf is None:
            raise RuntimeError(
                "RECORD_ALIGN_MISSING_REQUIRED_STREAM "
                f"stream=slam_tf target_monotonic_ns={sample_monotonic_ns}"
            )

        if aligned_slam_tf is not None:
            aligned_base_state["slam_map_pose"] = aligned_slam_tf
        return "ready", aligned_base_state, aligned_base_height, aligned_base_action

    @staticmethod
    def _interpolated_or_unbounded_nearest(interpolator, buffer, target_ns, *, min_timestamp_ns):
        aligned = interpolator(
            buffer,
            target_ns,
            max_delta_ns=None,
            min_timestamp_ns=min_timestamp_ns,
        )
        if aligned is not None:
            return aligned
        aligned = nearest_timed_sample(
            buffer,
            target_ns,
            max_delta_ns=None,
            min_timestamp_ns=min_timestamp_ns,
        )
        if aligned is not None:
            aligned["interpolation_mode"] = "nearest_fallback"
        return aligned

    def _add_record_item(
        self,
        *,
        pending,
        sample_monotonic_ns: int,
        aligned_state: dict,
        aligned_action: dict,
        aligned_base_state: dict | None,
        aligned_base_height: dict | None,
        aligned_base_action: dict | None,
        colors,
        depths,
        states,
        actions,
        camera_timestamps,
        sim_state_subscriber=None,
    ) -> None:
        timestamps = {
            "sample_wall_time_ns": int(time.time_ns()),
            "sample_monotonic_ns": int(sample_monotonic_ns),
            "teleop_input_perf_counter_ns": int(pending["teleop_input_perf_counter_ns"]),
            "primary_camera_name": pending["primary_camera_name"],
            "state": build_alignment_timestamp_entry(aligned_state, sample_monotonic_ns),
            "action": build_alignment_timestamp_entry(aligned_action, sample_monotonic_ns),
            "camera": camera_timestamps,
        }
        if self.record_base:
            timestamps["base_state"] = self._alignment_timestamp_with_source(aligned_base_state, sample_monotonic_ns)
            slam_map_pose = aligned_base_state.get("slam_map_pose")
            if slam_map_pose is not None:
                timestamps["slam_tf"] = self._alignment_timestamp_with_source(slam_map_pose, sample_monotonic_ns)
            timestamps["base_action"] = self._alignment_timestamp_with_source(aligned_base_action, sample_monotonic_ns)
            if aligned_base_height is not None:
                timestamps["base_height"] = self._alignment_timestamp_with_source(aligned_base_height, sample_monotonic_ns)
        aligned_arm_tauff = np.asarray(aligned_action["tauff"], dtype=float).reshape(-1)
        if aligned_arm_tauff.shape[0] != 14 or not np.all(np.isfinite(aligned_arm_tauff)):
            raise RuntimeError(
                "RECORD_ALIGN_INVALID_ARM_TAUFF "
                f"target_monotonic_ns={sample_monotonic_ns} expected_shape=(14,) "
                f"actual_shape={tuple(aligned_arm_tauff.shape)}"
            )
        control_extras = {"arm_tauff": aligned_arm_tauff.tolist()}

        if self.args.sim:
            sim_state = sim_state_subscriber.read_data() if sim_state_subscriber is not None else None
            self.recorder.add_item(
                colors=colors,
                depths=depths,
                states=states,
                actions=actions,
                sim_state=sim_state,
                timestamps=timestamps,
                control_extras=control_extras,
            )
        else:
            self.recorder.add_item(
                colors=colors,
                depths=depths,
                states=states,
                actions=actions,
                timestamps=timestamps,
                control_extras=control_extras,
            )

    def _alignment_timestamp_with_source(self, aligned_entry: dict, sample_monotonic_ns: int):
        entry = build_alignment_timestamp_entry(aligned_entry, sample_monotonic_ns)
        source_topic = aligned_entry.get("source_topic")
        if source_topic:
            entry["source_topic"] = str(source_topic)
        source_stamp_ns = aligned_entry.get("source_stamp_ns")
        if source_stamp_ns is not None:
            entry["source_stamp_ns"] = int(source_stamp_ns)
        return entry

    def _aligned_sample(self, buffer, target_ns, *, min_timestamp_ns):
        """Align without using distance thresholds as a recording decision.

        A pair of valid supports is always interpolated.  A one-sided support is
        returned as an explicitly marked, unbounded nearest fallback after the
        caller's future-wait policy has reached its deadline.
        """
        aligned = interpolate_timed_sample_strict(
            buffer,
            target_ns,
            max_delta_ns=None,
            min_timestamp_ns=min_timestamp_ns,
        )
        if aligned is not None:
            return aligned
        aligned = nearest_timed_sample(
            buffer,
            target_ns,
            max_delta_ns=None,
            min_timestamp_ns=min_timestamp_ns,
        )
        if aligned is not None:
            aligned["interpolation_mode"] = "nearest_fallback"
        return aligned

    def _aligned_camera_frames(self, *, pending, camera_sources, sample_monotonic_ns, record_min_timestamp_ns):
        colors = {pending["primary_camera_name"]: pending["primary_frame"]}
        depths = {}
        camera_timestamps = {pending["primary_camera_name"]: pending["primary_meta"]}
        for camera_name in ("head", "left_wrist", "right_wrist"):
            source = camera_sources.get(camera_name)
            if source is None or pending["primary_camera_name"] == camera_name:
                continue
            frame, meta = source.get_nearest(
                sample_monotonic_ns,
                max_delta_ns=self.camera_align_max_delta_ns,
                min_monotonic_ns=record_min_timestamp_ns,
                copy=True,
            )
            if frame is not None:
                colors[camera_name] = frame
                camera_timestamps[camera_name] = meta
        return colors, depths, camera_timestamps

    def _aligned_end_effector_qpos(self, aligned_sample: dict, key: str, *, sample_kind: str) -> list[float]:
        value = aligned_sample.get(key)
        if value is None:
            if bool(self.args.no_gripper):
                return []
            raise RuntimeError(f"aligned {sample_kind} is missing end-effector field {key!r}")
        qpos = np.asarray(value, dtype=float).reshape(-1)
        if not np.all(np.isfinite(qpos)):
            raise RuntimeError(f"aligned {sample_kind} end-effector field {key!r} contains non-finite values")
        return qpos.tolist()

    def _build_state_action_payload(
        self,
        *,
        pending,
        aligned_state,
        aligned_action,
        aligned_base_state,
        aligned_base_height,
        aligned_base_action,
        arm_ik,
        get_wrist_poses,
        get_wrist_poses_base_link,
        get_dex1_tcp_poses_base_link,
    ):
        aligned_lr_arm_q = np.asarray(aligned_state["q"], dtype=float)
        aligned_sol_q = np.asarray(aligned_action["q"], dtype=float)
        left_arm_state = aligned_lr_arm_q[:7]
        right_arm_state = aligned_lr_arm_q[-7:]
        left_arm_action = aligned_sol_q[:7]
        right_arm_action = aligned_sol_q[-7:]

        record_arm_repr = self.args.record_arm_repr
        need_arm_pose = record_arm_repr in {"pose", "both"}
        left_state_pose = right_state_pose = None
        left_action_pose = right_action_pose = None
        if need_arm_pose:
            left_state_pose, right_state_pose = get_wrist_poses(arm_ik, aligned_lr_arm_q)
            left_action_pose, right_action_pose = get_wrist_poses(arm_ik, aligned_sol_q)

        left_arm_state_entry = {
            "qpos": left_arm_state.tolist() if record_arm_repr in {"qpos", "both"} else [],
            "qvel": [],
            "torque": [],
        }
        right_arm_state_entry = {
            "qpos": right_arm_state.tolist() if record_arm_repr in {"qpos", "both"} else [],
            "qvel": [],
            "torque": [],
        }
        left_arm_action_entry = {
            "qpos": left_arm_action.tolist() if record_arm_repr in {"qpos", "both"} else [],
            "qvel": [],
            "torque": [],
        }
        right_arm_action_entry = {
            "qpos": right_arm_action.tolist() if record_arm_repr in {"qpos", "both"} else [],
            "qvel": [],
            "torque": [],
        }
        if need_arm_pose:
            left_arm_state_entry["pose"] = pose_matrix_to_record(left_state_pose)
            right_arm_state_entry["pose"] = pose_matrix_to_record(right_state_pose)
            left_arm_action_entry["pose"] = pose_matrix_to_record(left_action_pose)
            right_arm_action_entry["pose"] = pose_matrix_to_record(right_action_pose)

        if self._record_mobile_training_state:
            if aligned_base_height is None:
                raise RuntimeError("mobile training state requires aligned column height")
            if get_wrist_poses_base_link is None:
                raise RuntimeError("mobile training state requires base_link wrist kinematics")
            if get_dex1_tcp_poses_base_link is None:
                raise RuntimeError("mobile training state requires calibrated Dex1 TCP FK")
            raw_height = float(aligned_base_height["height"]["z"])
            column_height_m = column_position_from_raw_height(
                raw_height=raw_height,
                raw_minimum=float(self.args.mobile_height_raw_minimum),
                raw_maximum=float(self.args.mobile_height_raw_maximum),
                column_travel_m=float(self.args.mobile_column_travel_m),
            )
            waist_yaw = _finite_scalar(aligned_state.get("waist_yaw"), "state.waist_yaw")
            waist_yaw_target = _finite_scalar(
                aligned_action.get("waist_yaw_target"),
                "action.waist_yaw_target",
            )
            left_state_base_pose, right_state_base_pose = get_wrist_poses_base_link(
                arm_ik,
                aligned_lr_arm_q,
                column_height_m,
                waist_yaw,
            )
            left_action_base_pose, right_action_base_pose = get_wrist_poses_base_link(
                arm_ik,
                aligned_sol_q,
                column_height_m,
                waist_yaw_target,
            )
            left_state_tcp_pose, right_state_tcp_pose = get_dex1_tcp_poses_base_link(
                aligned_lr_arm_q,
                column_height_m,
                waist_yaw,
            )
            left_action_tcp_pose, right_action_tcp_pose = get_dex1_tcp_poses_base_link(
                aligned_sol_q,
                column_height_m,
                waist_yaw_target,
            )
            left_arm_state_entry["pose_base_link"] = pose_matrix_to_record(
                left_state_base_pose,
                frame_id="base_link",
                child_frame_id="left_ee",
            )
            right_arm_state_entry["pose_base_link"] = pose_matrix_to_record(
                right_state_base_pose,
                frame_id="base_link",
                child_frame_id="right_ee",
            )
            left_arm_action_entry["pose_base_link"] = pose_matrix_to_record(
                left_action_base_pose,
                frame_id="base_link",
                child_frame_id="left_ee_target",
            )
            right_arm_action_entry["pose_base_link"] = pose_matrix_to_record(
                right_action_base_pose,
                frame_id="base_link",
                child_frame_id="right_ee_target",
            )
            left_arm_state_entry["pose_base_link_tcp"] = pose_matrix_to_record(
                left_state_tcp_pose,
                frame_id="base_link",
                child_frame_id="left_dex1_tcp",
            )
            right_arm_state_entry["pose_base_link_tcp"] = pose_matrix_to_record(
                right_state_tcp_pose,
                frame_id="base_link",
                child_frame_id="right_dex1_tcp",
            )
            left_arm_action_entry["pose_base_link_tcp"] = pose_matrix_to_record(
                left_action_tcp_pose,
                frame_id="base_link",
                child_frame_id="left_dex1_tcp_target",
            )
            right_arm_action_entry["pose_base_link_tcp"] = pose_matrix_to_record(
                right_action_tcp_pose,
                frame_id="base_link",
                child_frame_id="right_dex1_tcp_target",
            )

        left_ee_state = self._aligned_end_effector_qpos(
            aligned_state,
            "left_ee_state",
            sample_kind="state",
        )
        right_ee_state = self._aligned_end_effector_qpos(
            aligned_state,
            "right_ee_state",
            sample_kind="state",
        )
        left_ee_action = self._aligned_end_effector_qpos(
            aligned_action,
            "left_hand_action",
            sample_kind="action",
        )
        right_ee_action = self._aligned_end_effector_qpos(
            aligned_action,
            "right_hand_action",
            sample_kind="action",
        )
        states = {
            "left_arm": left_arm_state_entry,
            "right_arm": right_arm_state_entry,
            "left_ee": {"qpos": left_ee_state, "qvel": [], "torque": []},
            "right_ee": {"qpos": right_ee_state, "qvel": [], "torque": []},
        }
        actions = {
            "left_arm": left_arm_action_entry,
            "right_arm": right_arm_action_entry,
            "left_ee": {"qpos": left_ee_action, "qvel": [], "torque": []},
            "right_ee": {"qpos": right_ee_action, "qvel": [], "torque": []},
        }
        if self.record_base:
            states["base"] = build_base_state_record(
                aligned_base_state,
                aligned_base_height,
                include_world_pose=not self.record_base_velocity_only,
            )
            actions["base"] = build_base_action_record(aligned_base_action)
            if self._record_mobile_training_state:
                states["base"]["column_height_m"] = column_height_m
                states["base"]["waist_yaw"] = waist_yaw
                actions["base"]["waist_yaw_target"] = waist_yaw_target
        return states, actions


def pose_matrix_to_record(pose_mat, *, frame_id: str | None = None, child_frame_id: str | None = None):
    from scipy.spatial.transform import Rotation as R

    pose = np.asarray(pose_mat, dtype=float)
    rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False)
    record = {
        "position": pose[:3, 3].tolist(),
        "rpy": rpy.tolist(),
        "rotation_matrix": pose[:3, :3].tolist(),
        "matrix4x4": pose.tolist(),
    }
    if frame_id is not None:
        record["frame_id"] = str(frame_id)
    if child_frame_id is not None:
        record["child_frame_id"] = str(child_frame_id)
    return record


def _finite_scalar(value, field_name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return result
