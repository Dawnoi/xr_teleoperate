"""遥操作业录制流程：管理 episode 生命周期、时间对齐和样本写入。

Recording flow used by the real teleop entrypoint.

This module owns the mutable recording state and the per-frame alignment/write
logic so ``teleop_hand_and_arm.py`` can stay as the runtime coordinator.
"""

from __future__ import annotations

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

        frequency = max(float(args.frequency), 1e-6)
        self.camera_align_max_delta_ns = int(max(20_000_000, (1.5 / frequency) * 1e9))
        self.state_align_max_delta_ns = int(max(20_000_000, (1.25 / frequency) * 1e9))
        self.action_align_max_delta_ns = self.state_align_max_delta_ns
        self.state_nearest_fallback_max_delta_ns = int(max(80_000_000, (2.5 / frequency) * 1e9))
        self.action_nearest_fallback_max_delta_ns = self.state_nearest_fallback_max_delta_ns
        self.record_future_wait_timeout_ns = int(max(120_000_000, (2.0 / frequency) * 1e9))
        self.pending_sample_timeout_ns = int(1_000_000_000)
        self.record_base = bool(getattr(args, "record_base", False))
        self.record_slam_map_pose = bool(getattr(args, "record_slam_map_pose", False))
        self._record_mobile_training_state = bool(
            getattr(args, "record_mobile_training_state", False)
        )
        self.base_state_max_delta_ns = int(max(1_000_000, float(getattr(args, "base_state_max_age_ms", 131.578947)) * 1e6))
        self.base_height_max_delta_ns = int(150.0 * 1e6)
        self.slam_tf_max_delta_ns = int(max(1_000_000, float(getattr(args, "slam_tf_max_age_ms", 125.0)) * 1e6))
        self.base_height_required = bool(str(getattr(args, "base_height_topic", "/hispeed_state") or "").strip())

    def handle_commands(
        self,
        *,
        record_running: bool,
        record_toggle: bool,
        record_cancel: bool,
        camera_sources: dict[str, Any],
    ) -> RecordingCommandResult:
        """Consume operator recording commands and update episode lifecycle state."""
        if record_cancel:
            record_cancel = False
            if record_running or self.state.waiting_for_first_frame:
                record_running = False
                self._reset_active_state()
                self.recorder.cancel_episode()
                self.log.info("[RECORD_CANCEL] active episode canceled; next recording will reuse the episode index.")
            else:
                self.log.info("[RECORD_CANCEL] ignored: no active recording to cancel.")

        if record_toggle:
            record_toggle = False
            if not record_running and not self.state.waiting_for_first_frame:
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
                if self.recorder.create_episode(enabled_cameras=enabled_cameras):
                    self.state.record_start_monotonic_ns = int(time.monotonic_ns())
                    self.state.pending_samples.clear()
                    self.state.last_enqueued_primary_frame_id = None
                    self.state.waiting_for_first_frame = True
                    self.log.info("[RECORD_ALIGN] episode armed, waiting for first post-start camera frame before recording.")
                else:
                    self.log.error("Failed to create episode. Recording not started.")
            else:
                record_running = False
                self._reset_active_state()
                self.recorder.save_episode()
                if self.reset_callback is not None:
                    self.reset_callback()

        return RecordingCommandResult(
            record_running=record_running,
            record_toggle=record_toggle,
            record_cancel=record_cancel,
        )

    def process_frame(
        self,
        *,
        record_running: bool,
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
    ) -> RecordingFrameResult:
        """Align the current frame to camera time and append items to the recorder."""
        ready = self.recorder.is_ready()
        ee_sample = self._read_end_effector_sample()

        if not (record_running or self.state.waiting_for_first_frame):
            return RecordingFrameResult(record_running=record_running, ready=ready)

        if self.state.waiting_for_first_frame:
            wait_result = self._maybe_start_after_first_camera_frame(record_running, camera_sources)
            record_running = wait_result.record_running
            if wait_result.should_continue_frame:
                return RecordingFrameResult(record_running=record_running, ready=ready, should_continue_frame=True)

        record_min_timestamp_ns = int(self.state.record_start_monotonic_ns or time.monotonic_ns())
        self._enqueue_primary_camera_sample(
            camera_sources=camera_sources,
            record_min_timestamp_ns=record_min_timestamp_ns,
            ee_sample=ee_sample,
            teleop_input_perf_counter_ns=teleop_input_perf_counter_ns,
        )
        self._write_pending_samples(
            camera_sources=camera_sources,
            state_history=state_history,
            action_history=action_history,
            base_state_history=base_state_history,
            base_height_history=base_height_history,
            base_action_history=base_action_history,
            slam_tf_history=slam_tf_history,
            record_min_timestamp_ns=record_min_timestamp_ns,
            arm_ik=arm_ik,
            get_wrist_poses=get_wrist_poses,
            get_wrist_poses_base_link=get_wrist_poses_base_link,
            get_dex1_tcp_poses_base_link=get_dex1_tcp_poses_base_link,
            sim_state_subscriber=sim_state_subscriber,
        )
        return RecordingFrameResult(record_running=record_running, ready=ready)

    def _reset_active_state(self):
        self.state.waiting_for_first_frame = False
        self.state.record_start_monotonic_ns = None
        self.state.pending_samples.clear()
        self.state.last_enqueued_primary_frame_id = None

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
            self.state.waiting_for_first_frame = False
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
                if now_ns - self.state.last_wait_log_ns > 1_000_000_000:
                    self.log.info("[RECORD_ALIGN] waiting for first post-start camera frame...")
                    self.state.last_wait_log_ns = now_ns
                return RecordingFrameResult(record_running=record_running, ready=True, should_continue_frame=True)
            first_frame_times.append(meta_ts)

        self.state.record_start_monotonic_ns = int(max(first_frame_times))
        self.state.waiting_for_first_frame = False
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

        primary_frame, primary_meta = primary_source.get_latest(copy=True)
        if primary_frame is None or primary_meta is None:
            return

        primary_ts = int(camera_meta_monotonic_ns(primary_meta))
        primary_frame_seq = (primary_meta or {}).get("frame_seq")
        primary_frame_id = camera_frame_identity(primary_camera_name, primary_meta)
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
        ):
            self.state.pending_samples.append(
                {
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

    def _write_pending_samples(
        self,
        *,
        camera_sources: dict[str, Any],
        state_history: deque,
        action_history: deque,
        base_state_history: deque | None,
        base_height_history: deque | None,
        base_action_history: deque | None,
        slam_tf_history: deque | None,
        record_min_timestamp_ns: int,
        arm_ik,
        get_wrist_poses: Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]],
        get_wrist_poses_base_link: Callable[[Any, np.ndarray, float, float], tuple[np.ndarray, np.ndarray]] | None,
        get_dex1_tcp_poses_base_link: Callable[[np.ndarray, float, float], tuple[np.ndarray, np.ndarray]] | None,
        sim_state_subscriber=None,
    ):
        while self.state.pending_samples:
            pending = self.state.pending_samples[0]
            sample_monotonic_ns = int(pending["sample_monotonic_ns"])
            (
                status,
                aligned_state,
                aligned_action,
                aligned_base_state,
                aligned_base_height,
                aligned_base_action,
            ) = self._resolve_pending_alignment(
                state_history=state_history,
                action_history=action_history,
                base_state_history=base_state_history,
                base_height_history=base_height_history,
                base_action_history=base_action_history,
                slam_tf_history=slam_tf_history,
                sample_monotonic_ns=sample_monotonic_ns,
                record_min_timestamp_ns=record_min_timestamp_ns,
            )
            if status == "wait":
                break
            if status == "drop":
                self.state.pending_samples.popleft()
                continue

            colors, depths, camera_timestamps = self._aligned_camera_frames(
                pending=pending,
                camera_sources=camera_sources,
                sample_monotonic_ns=sample_monotonic_ns,
                record_min_timestamp_ns=record_min_timestamp_ns,
            )
            states, actions = self._build_state_action_payload(
                pending=pending,
                aligned_state=aligned_state,
                aligned_action=aligned_action,
                aligned_base_state=aligned_base_state,
                aligned_base_height=aligned_base_height,
                aligned_base_action=aligned_base_action,
                arm_ik=arm_ik,
                get_wrist_poses=get_wrist_poses,
                get_wrist_poses_base_link=get_wrist_poses_base_link,
                get_dex1_tcp_poses_base_link=get_dex1_tcp_poses_base_link,
            )
            if not self._add_record_item(
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
                sim_state_subscriber=sim_state_subscriber,
            ):
                self.state.pending_samples.popleft()
                continue
            self.state.pending_samples.popleft()

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
    ) -> tuple[str, dict | None, dict | None, dict | None, dict | None, dict | None]:
        state_earliest, state_latest = timed_buffer_bounds(state_history, min_timestamp_ns=record_min_timestamp_ns)
        action_earliest, action_latest = timed_buffer_bounds(action_history, min_timestamp_ns=record_min_timestamp_ns)
        sample_age_ns = int(time.monotonic_ns()) - int(sample_monotonic_ns)

        if state_earliest is None or action_earliest is None:
            return "wait", None, None, None, None, None
        if sample_monotonic_ns < state_earliest or sample_monotonic_ns < action_earliest:
            self.log.warning("[RECORD_ALIGN] drop pending sample: target timestamp fell out of state/action history window.")
            return "drop", None, None, None, None, None
        if state_latest is None or action_latest is None:
            return "wait", None, None, None, None, None
        if sample_monotonic_ns > state_latest or sample_monotonic_ns > action_latest:
            if sample_age_ns <= self.record_future_wait_timeout_ns:
                return "wait", None, None, None, None, None

        aligned_state = self._aligned_sample(
            state_history,
            sample_monotonic_ns,
            strict_max_delta_ns=self.state_align_max_delta_ns,
            fallback_max_delta_ns=self.state_nearest_fallback_max_delta_ns,
            min_timestamp_ns=record_min_timestamp_ns,
        )
        aligned_action = self._aligned_sample(
            action_history,
            sample_monotonic_ns,
            strict_max_delta_ns=self.action_align_max_delta_ns,
            fallback_max_delta_ns=self.action_nearest_fallback_max_delta_ns,
            min_timestamp_ns=record_min_timestamp_ns,
        )
        if aligned_state is None:
            if sample_age_ns > self.pending_sample_timeout_ns:
                self.log.warning("[RECORD_ALIGN] drop pending sample: no interpolated state found at primary camera timestamp.")
                return "drop", None, None, None, None, None
            return "wait", None, None, None, None, None
        if aligned_action is None:
            if sample_age_ns > self.pending_sample_timeout_ns:
                self.log.warning("[RECORD_ALIGN] drop pending sample: no interpolated action found at primary camera timestamp.")
                return "drop", None, None, None, None, None
            return "wait", None, None, None, None, None

        base_status, aligned_base_state, aligned_base_height, aligned_base_action = self._resolve_base_alignment(
            base_state_history=base_state_history,
            base_height_history=base_height_history,
            base_action_history=base_action_history,
            slam_tf_history=slam_tf_history,
            sample_monotonic_ns=sample_monotonic_ns,
            record_min_timestamp_ns=record_min_timestamp_ns,
            sample_age_ns=sample_age_ns,
        )
        if base_status != "ready":
            return base_status, None, None, None, None, None
        return "ready", aligned_state, aligned_action, aligned_base_state, aligned_base_height, aligned_base_action

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
    ) -> tuple[str, dict | None, dict | None, dict | None]:
        if not self.record_base:
            return "ready", None, None, None
        if base_state_history is None or base_action_history is None:
            raise RuntimeError("--record-base requires base_state_history and base_action_history")
        if self.record_slam_map_pose and slam_tf_history is None:
            raise RuntimeError("--record-slam-map-pose requires slam_tf_history")
        if self.base_height_required and base_height_history is None:
            raise RuntimeError("--record-base requires base_height_history when --base-height-topic is set")

        aligned_base_state = interpolate_base_state_timed_sample_strict(
            base_state_history,
            sample_monotonic_ns,
            max_delta_ns=self.base_state_max_delta_ns,
            min_timestamp_ns=record_min_timestamp_ns,
        )
        aligned_base_action = self._aligned_sample(
            base_action_history,
            sample_monotonic_ns,
            strict_max_delta_ns=self.action_align_max_delta_ns,
            fallback_max_delta_ns=self.action_nearest_fallback_max_delta_ns,
            min_timestamp_ns=record_min_timestamp_ns,
        )
        aligned_base_height = None
        aligned_slam_tf = None
        if self.base_height_required:
            aligned_base_height = interpolate_base_height_timed_sample_strict(
                base_height_history,
                sample_monotonic_ns,
                max_delta_ns=self.base_height_max_delta_ns,
                min_timestamp_ns=record_min_timestamp_ns,
            )
        if self.record_slam_map_pose:
            aligned_slam_tf = interpolate_slam_tf_timed_sample_strict(
                slam_tf_history,
                sample_monotonic_ns,
                max_delta_ns=self.slam_tf_max_delta_ns,
                min_timestamp_ns=record_min_timestamp_ns,
            )

        if aligned_base_state is None:
            if sample_age_ns > self.pending_sample_timeout_ns:
                self.log.warning("[RECORD_ALIGN] drop pending sample: no aligned base state found at primary camera timestamp.")
                return "drop", None, None, None
            return "wait", None, None, None
        if aligned_base_action is None:
            if sample_age_ns > self.pending_sample_timeout_ns:
                self.log.error(
                    "[RECORD_ALIGN] drop pending sample: no base action aligned to the primary camera timestamp "
                    "under the arm-action alignment policy."
                )
                return "drop", None, None, None
            return "wait", None, None, None
        if self.base_height_required and aligned_base_height is None:
            if sample_age_ns > self.pending_sample_timeout_ns:
                self.log.warning("[RECORD_ALIGN] drop pending sample: no aligned base height found at primary camera timestamp.")
                return "drop", None, None, None
            return "wait", None, None, None
        if self.record_slam_map_pose and aligned_slam_tf is None:
            if sample_age_ns > self.pending_sample_timeout_ns:
                self.log.warning("[RECORD_ALIGN] drop pending sample: no aligned SLAM TF found at primary camera timestamp.")
                return "drop", None, None, None
            return "wait", None, None, None

        if aligned_slam_tf is not None:
            aligned_base_state["slam_map_pose"] = aligned_slam_tf
        return "ready", aligned_base_state, aligned_base_height, aligned_base_action

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
    ) -> bool:
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
            self.log.warning("[RECORD_ALIGN] drop pending sample: invalid aligned arm tauff for control sidecar.")
            return False
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
        return True

    def _alignment_timestamp_with_source(self, aligned_entry: dict, sample_monotonic_ns: int):
        entry = build_alignment_timestamp_entry(aligned_entry, sample_monotonic_ns)
        source_topic = aligned_entry.get("source_topic")
        if source_topic:
            entry["source_topic"] = str(source_topic)
        source_stamp_ns = aligned_entry.get("source_stamp_ns")
        if source_stamp_ns is not None:
            entry["source_stamp_ns"] = int(source_stamp_ns)
        return entry

    def _aligned_sample(self, buffer, target_ns, *, strict_max_delta_ns, fallback_max_delta_ns, min_timestamp_ns):
        aligned = interpolate_timed_sample_strict(
            buffer,
            target_ns,
            max_delta_ns=strict_max_delta_ns,
            min_timestamp_ns=min_timestamp_ns,
        )
        if aligned is not None:
            return aligned
        aligned = nearest_timed_sample(
            buffer,
            target_ns,
            max_delta_ns=fallback_max_delta_ns,
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

        states = {
            "left_arm": left_arm_state_entry,
            "right_arm": right_arm_state_entry,
            "left_ee": {"qpos": pending["left_ee_state"], "qvel": [], "torque": []},
            "right_ee": {"qpos": pending["right_ee_state"], "qvel": [], "torque": []},
        }
        actions = {
            "left_arm": left_arm_action_entry,
            "right_arm": right_arm_action_entry,
            "left_ee": {"qpos": pending["left_hand_action"], "qvel": [], "torque": []},
            "right_ee": {"qpos": pending["right_hand_action"], "qvel": [], "torque": []},
        }
        if self.record_base:
            states["base"] = build_base_state_record(aligned_base_state, aligned_base_height)
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
