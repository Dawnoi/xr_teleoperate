"""Recording flow used by the real teleop entrypoint.

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
    interpolate_timed_sample_strict,
    nearest_timed_sample,
    timed_buffer_bounds,
)


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
        self.state = TeleopRecordingState()

        frequency = max(float(args.frequency), 1e-6)
        self.camera_align_max_delta_ns = int(max(20_000_000, (1.5 / frequency) * 1e9))
        self.state_align_max_delta_ns = int(max(20_000_000, (1.25 / frequency) * 1e9))
        self.action_align_max_delta_ns = self.state_align_max_delta_ns
        self.state_nearest_fallback_max_delta_ns = int(max(80_000_000, (2.5 / frequency) * 1e9))
        self.action_nearest_fallback_max_delta_ns = self.state_nearest_fallback_max_delta_ns
        self.record_future_wait_timeout_ns = int(max(120_000_000, (2.0 / frequency) * 1e9))
        self.pending_sample_timeout_ns = int(1_000_000_000)

    def handle_commands(self, *, record_running: bool, record_toggle: bool, record_cancel: bool) -> RecordingCommandResult:
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
                if self.recorder.create_episode():
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
        sim_state_subscriber=None,
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
            record_min_timestamp_ns=record_min_timestamp_ns,
            arm_ik=arm_ik,
            get_wrist_poses=get_wrist_poses,
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
        record_min_timestamp_ns: int,
        arm_ik,
        get_wrist_poses: Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]],
        sim_state_subscriber=None,
    ):
        while self.state.pending_samples:
            pending = self.state.pending_samples[0]
            sample_monotonic_ns = int(pending["sample_monotonic_ns"])
            state_earliest, state_latest = timed_buffer_bounds(state_history, min_timestamp_ns=record_min_timestamp_ns)
            action_earliest, action_latest = timed_buffer_bounds(action_history, min_timestamp_ns=record_min_timestamp_ns)
            now_mono_ns = int(time.monotonic_ns())
            sample_age_ns = now_mono_ns - sample_monotonic_ns

            if state_earliest is None or action_earliest is None:
                break
            if sample_monotonic_ns < state_earliest or sample_monotonic_ns < action_earliest:
                self.log.warning("[RECORD_ALIGN] drop pending sample: target timestamp fell out of state/action history window.")
                self.state.pending_samples.popleft()
                continue
            if state_latest is None or action_latest is None:
                break

            waiting_for_future_coverage = sample_monotonic_ns > state_latest or sample_monotonic_ns > action_latest
            if waiting_for_future_coverage and sample_age_ns <= self.record_future_wait_timeout_ns:
                break

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
                    self.state.pending_samples.popleft()
                    continue
                break
            if aligned_action is None:
                if sample_age_ns > self.pending_sample_timeout_ns:
                    self.log.warning("[RECORD_ALIGN] drop pending sample: no interpolated action found at primary camera timestamp.")
                    self.state.pending_samples.popleft()
                    continue
                break

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
                arm_ik=arm_ik,
                get_wrist_poses=get_wrist_poses,
            )
            timestamps = {
                "sample_wall_time_ns": int(time.time_ns()),
                "sample_monotonic_ns": sample_monotonic_ns,
                "teleop_input_perf_counter_ns": int(pending["teleop_input_perf_counter_ns"]),
                "primary_camera_name": pending["primary_camera_name"],
                "state": build_alignment_timestamp_entry(aligned_state, sample_monotonic_ns),
                "action": build_alignment_timestamp_entry(aligned_action, sample_monotonic_ns),
                "camera": camera_timestamps,
            }
            aligned_arm_tauff = np.asarray(aligned_action["tauff"], dtype=float).reshape(-1)
            if aligned_arm_tauff.shape[0] != 14 or not np.all(np.isfinite(aligned_arm_tauff)):
                self.log.warning("[RECORD_ALIGN] drop pending sample: invalid aligned arm tauff for control sidecar.")
                self.state.pending_samples.popleft()
                continue
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
            self.state.pending_samples.popleft()

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

    def _build_state_action_payload(self, *, pending, aligned_state, aligned_action, arm_ik, get_wrist_poses):
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
        return states, actions


def pose_matrix_to_record(pose_mat):
    from scipy.spatial.transform import Rotation as R

    pose = np.asarray(pose_mat, dtype=float)
    rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False)
    return {
        "position": pose[:3, 3].tolist(),
        "rpy": rpy.tolist(),
        "rotation_matrix": pose[:3, :3].tolist(),
        "matrix4x4": pose.tolist(),
    }
