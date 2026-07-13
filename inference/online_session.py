from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import logging
import numpy as np

from inference.transport import encode_jpeg_base64, parse_action_chunk
from inference.pi05_protocol import (
    PI05_IMAGE_ROLES,
    build_pi05_observation_payload,
    pi05_action_sequence_to_pose7_chunks,
    validate_pi05_action_sequence,
)
from inference.pose_transform import matrix_to_pose7_xyzw, matrix_to_pose9_rot6d, pose7_xyzw_to_matrix

logger = logging.getLogger(__name__)


def _monotonic_ns() -> int:
    import time

    return time.monotonic_ns()


def _perf_counter_ns() -> int:
    import time

    return time.perf_counter_ns()


@dataclass
class OnlineInferenceConfig:
    arm_side: str
    protocol_profile: str = "pika_pose7"
    task_prompt: str = ""
    n_obs_steps: int = 2
    camera_freq: float = 30.0
    action_step_sec: float = 0.10
    chunk_step_mode: str = "per_tick"
    interpolation_interval_sec: float = 0.01
    post_action_delay_ms: int = 75
    max_post_action_delay_ms: int = 5000
    response_timeout_sec: float = 2.0
    jpeg_quality: int = 85
    enable_motion: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.protocol_profile = str(self.protocol_profile or "pika_pose7").strip()
        if self.protocol_profile not in {"pika_pose7", "pi05_dual_arm_20d"}:
            raise ValueError(f"unsupported protocol_profile: {self.protocol_profile!r}")
        if self.arm_side not in {"left", "right", "both"}:
            raise ValueError(f"unsupported arm_side: {self.arm_side!r}")
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        if self.action_step_sec <= 0.0:
            raise ValueError("action_step_sec must be positive")
        if self.chunk_step_mode not in {"timed", "per_tick"}:
            raise ValueError(f"unsupported chunk_step_mode: {self.chunk_step_mode!r}")
        if self.interpolation_interval_sec <= 0.0:
            raise ValueError("interpolation_interval_sec must be positive")
        if self.response_timeout_sec <= 0.0:
            raise ValueError("response_timeout_sec must be positive")
        if self.post_action_delay_ms < 0:
            raise ValueError("post_action_delay_ms must be non-negative")
        if self.max_post_action_delay_ms < 0:
            raise ValueError("max_post_action_delay_ms must be non-negative")
        if self.post_action_delay_ms > self.max_post_action_delay_ms:
            raise ValueError("post_action_delay_ms exceeds max_post_action_delay_ms")
        if self.camera_freq <= 0.0:
            raise ValueError("camera_freq must be positive")


@dataclass
class CameraSample:
    name: str
    frame: Any
    host_monotonic_ns: int


@dataclass
class RobotStateSample:
    host_monotonic_ns: int
    left_pose: Any
    right_pose: Any
    left_gripper_width: float
    right_gripper_width: float


@dataclass
class OnlineInferenceStep:
    left_pose: np.ndarray
    right_pose: np.ndarray
    left_gripper_width: float
    right_gripper_width: float
    enabled_arms: List[str]
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def online_inference_speed_limit_delta(
    *,
    target_q: Any,
    limited_q: Any,
    max_arm_joint_speed: float,
    frequency: float,
) -> Optional[float]:
    target = np.asarray(target_q, dtype=float).reshape(-1)
    limited = np.asarray(limited_q, dtype=float).reshape(-1)
    if target.shape != limited.shape:
        raise ValueError("joint speed check requires target and limited vectors with identical shape")
    if target.size == 0:
        raise ValueError("joint speed check requires non-empty joint vectors")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(limited)):
        raise ValueError("joint speed check vectors must be finite")

    speed_limit_delta = float(np.max(np.abs(target - limited)))
    allowed_delta = float(max_arm_joint_speed) / max(float(frequency), 1e-6)
    if speed_limit_delta <= max(0.10, allowed_delta * 2.0):
        return None
    return speed_limit_delta


class OnlineInferenceSession:
    def __init__(
        self,
        config: OnlineInferenceConfig,
        transport: Any,
        pose_transformer: Optional[Any] = None,
        clock_ns: Optional[Callable[[], int]] = None,
        trace_clock_ns: Optional[Callable[[], int]] = None,
        transport_reset_done: bool = False,
    ) -> None:
        self.config = config
        self.transport = transport
        self.pose_transformer = pose_transformer
        self.clock_ns = clock_ns or _monotonic_ns
        self.trace_clock_ns = trace_clock_ns or _perf_counter_ns

        if self.config.enable_motion and not self.config.dry_run and self.pose_transformer is None:
            raise ValueError("pose_transformer is required when online inference motion is enabled")

        self.status = "collecting_observation"
        self.error: Optional[str] = None

        self._state_history: List[RobotStateSample] = []
        self._camera_history_by_name: Dict[str, List[CameraSample]] = {}
        self._camera_order: List[str] = []
        self._required_camera_names: Optional[List[str]] = None
        self._init_poses_server: Dict[str, List[float]] = {}
        self._last_observation_debug: Dict[str, Any] = {}
        self._last_action_debug: Dict[str, Any] = {}
        self._observation_seq = 0
        self._current_observation_seq = 0
        self._current_observation_send_ns: Optional[int] = None
        self._current_observation_send_perf_ns: Optional[int] = None
        self._chunk_seq = 0
        self._current_chunk_seq = 0
        self._current_action_recv_ns: Optional[int] = None
        self._current_action_recv_perf_ns: Optional[int] = None

        self._waiting_since_ns: Optional[int] = None
        self._collecting_since_ns: Optional[int] = None

        self._current_chunk_left: Optional[List[np.ndarray]] = None
        self._current_chunk_right: Optional[List[np.ndarray]] = None
        self._current_chunk_index = 0
        self._current_chunk_size = 0
        self._current_post_action_delay_ns = int(self.config.post_action_delay_ms * 1_000_000)
        self._post_action_ready_after_ns: Optional[int] = None
        self._post_action_entered_ns: Optional[int] = None
        self._current_step_start_ns: Optional[int] = None
        self._current_chunk_started_ns: Optional[int] = None
        self._current_step_duration_ns = max(1, int(self.config.action_step_sec * 1_000_000_000))
        self._observation_step_ns = max(1, int((1.0 / self.config.camera_freq) * 1_000_000_000))
        self.max_camera_delta_ns = max(1, int(max(1.5 / self.config.camera_freq, 0.05) * 1_000_000_000))
        self._max_history_count = max(16, self.config.n_obs_steps * 8)
        self._observation_ready_timeout_ns = int(max(5.0, self.config.response_timeout_sec * 2.0) * 1_000_000_000)

        self._transport_reset_done = bool(transport_reset_done)

    def get_debug_snapshot(self) -> Dict[str, Any]:
        last_observation = dict(self._last_observation_debug)
        if last_observation:
            last_observation.setdefault("observation_seq", int(self._current_observation_seq))
            if self._current_observation_send_ns is not None:
                last_observation.setdefault("sent_monotonic_ns", int(self._current_observation_send_ns))

        last_action = dict(self._last_action_debug)
        if self._current_chunk_seq > 0:
            last_action.setdefault("chunk_seq", int(self._current_chunk_seq))
            last_action.setdefault("chunk_index", int(self._current_chunk_index))
            last_action.setdefault("chunk_size", int(self._current_chunk_size))
        if self.status == "executing_chunk":
            last_action.update(
                {
                    "chunk_seq": int(self._current_chunk_seq),
                    "chunk_index": int(self._current_chunk_index),
                    "chunk_size": int(self._current_chunk_size),
                }
            )

        post_action_delay = {}
        if self.status == "post_action_delay" and self._post_action_ready_after_ns is not None:
            remaining_ns = max(0, int(self._post_action_ready_after_ns) - int(self.clock_ns()))
            post_action_delay = {
                "ready_after_monotonic_ns": int(self._post_action_ready_after_ns),
                "remaining_ms": remaining_ns / 1e6,
            }
        return {
            "arm_side": self.config.arm_side,
            "protocol_profile": self.config.protocol_profile,
            "status": self.status,
            "error": self.error,
            "last_observation": last_observation,
            "last_action": last_action,
            "post_action_delay": post_action_delay,
        }

    def set_required_camera_names(self, names: Sequence[str]) -> None:
        required = [str(name) for name in names if str(name)]
        if not required:
            return
        self._required_camera_names = list(dict.fromkeys(required))
        self._camera_order = list(self._required_camera_names)

    def tick(
        self,
        state_sample: RobotStateSample,
        camera_samples: Sequence[CameraSample],
    ) -> OnlineInferenceStep:
        self._append_history(state_sample, camera_samples)

        if self.status == "failed":
            return self._hold_step(state_sample, "failed", {"error": self.error})

        if not self._transport_connected():
            self._transport_reset_done = False
            self._fail("transport disconnected")
            return self._hold_step(state_sample, "failed", {"error": self.error})

        if self.status == "waiting_action":
            action_step = self._poll_action_and_maybe_execute(state_sample)
            if action_step is not None:
                return action_step

        if self.status == "executing_chunk":
            return self._emit_action_step(state_sample)

        if self.status == "post_action_delay":
            now_ns = int(self.clock_ns())
            if not self._post_action_delay_satisfied(now_ns, state_sample, camera_samples):
                timeout_base_ns = self._current_chunk_started_ns or self._post_action_ready_after_ns or self._post_action_entered_ns
                if timeout_base_ns is not None and now_ns - int(timeout_base_ns) >= self._observation_ready_timeout_ns:
                    self._fail("post_action_delay observation coverage timeout")
                    return self._hold_step(state_sample, "failed", {"error": self.error})
                return self._hold_step(
                    state_sample,
                    "post_action_delay",
                    {"ready_after_ns": self._post_action_ready_after_ns},
                )
            self.status = "collecting_observation"

        if not self._history_ready():
            now_ns = int(self.clock_ns())
            if self._collecting_since_ns is None:
                self._collecting_since_ns = now_ns
            elif now_ns - self._collecting_since_ns > self._observation_ready_timeout_ns:
                self._fail("observation history timeout")
                return self._hold_step(state_sample, "failed", {"error": self.error})
            return self._hold_step(
                state_sample,
                "collecting_observation",
                {"history_len": len(self._state_history)},
            )

        try:
            self._ensure_transport_reset()
            observation = self._build_observation_message()
            self._log_current_observation_point(observation)
            observation_debug = self._make_observation_debug(observation)
            observation_send_ns = int(self.clock_ns())
            observation_send_perf_ns = int(self.trace_clock_ns())
            self.transport.send_json(observation)
            self._waiting_since_ns = observation_send_ns
            self._current_observation_send_perf_ns = observation_send_perf_ns
            self._observation_seq += 1
            self._current_observation_seq = self._observation_seq
            self._current_observation_send_ns = int(self._waiting_since_ns)
            observation_debug.update(
                {
                    "observation_seq": int(self._current_observation_seq),
                    "sent_monotonic_ns": int(self._current_observation_send_ns),
                }
            )
            self._last_observation_debug = observation_debug
            self._collecting_since_ns = None
            self.status = "waiting_action"
        except Exception as exc:
            self._fail(f"failed to send observation: {exc}")
            return self._hold_step(state_sample, "failed", {"error": self.error})
        return self._hold_step(
            state_sample,
            "waiting_action",
            {"history_len": len(self._state_history)},
        )

    def report_control_feedback(self, feedback: dict) -> None:
        if isinstance(feedback, dict) and feedback.get("fatal"):
            reason = str(feedback.get("reason", "fatal control feedback"))
            self._fail(reason)

    def _append_history(
        self,
        state_sample: RobotStateSample,
        camera_samples: Sequence[CameraSample],
    ) -> None:
        self._state_history.append(state_sample)
        if len(self._state_history) > self._max_history_count:
            self._state_history = self._state_history[-self._max_history_count :]

        for sample in camera_samples:
            name = str(sample.name)
            if name not in self._camera_order:
                self._camera_order.append(name)
            history = self._camera_history_by_name.setdefault(name, [])
            history.append(sample)
            if len(history) > self._max_history_count:
                self._camera_history_by_name[name] = history[-self._max_history_count :]

    def _transport_connected(self) -> bool:
        return bool(self.transport.is_connected())

    def _history_ready(self) -> bool:
        try:
            self._select_observation_window()
            return True
        except ValueError:
            return False

    def _ensure_transport_reset(self) -> None:
        if self._transport_reset_done:
            return
        clear_pending_rx = getattr(self.transport, "clear_pending_rx", None)
        if callable(clear_pending_rx):
            clear_pending_rx()
        reset = getattr(self.transport, "reset", None)
        if callable(reset):
            reset()
        else:
            send_json = getattr(self.transport, "send_json", None)
            if callable(send_json):
                send_json({"type": "reset"})
        if callable(clear_pending_rx):
            clear_pending_rx()
        self._transport_reset_done = True
        self._init_poses_server = {}
        self._last_observation_debug = {}
        self._last_action_debug = {}

    def _build_observation_message(self) -> Dict[str, Any]:
        if self.config.protocol_profile == "pi05_dual_arm_20d":
            return self._build_pi05_observation_message()
        message: Dict[str, Any] = {"type": "observation"}
        if self.config.arm_side in {"left", "both"}:
            message["arm_l"] = self._build_arm_observation("left")
        if self.config.arm_side in {"right", "both"}:
            message["arm_r"] = self._build_arm_observation("right")
        return message

    def _build_arm_observation(self, side: str) -> Dict[str, Any]:
        state_window, camera_window = self._select_observation_window()
        poses: List[List[float]] = []
        grippers: List[float] = []
        for sample in state_window:
            poses.append(self._observation_pose_to_server(side, self._pose_for_side(sample, side)))
            grippers.append(float(self._gripper_for_side(sample, side)))

        images: List[str] = []
        for step_index in range(self.config.n_obs_steps):
            for camera_name in self._camera_order:
                sample = camera_window.get(camera_name, [None] * self.config.n_obs_steps)[step_index]
                if sample is None:
                    raise ValueError(f"missing camera sample for {camera_name}")
                images.append(encode_jpeg_base64(np.asarray(sample.frame), quality=self.config.jpeg_quality))

        if not images and self._camera_order:
            raise ValueError("camera history is not ready")
        if not self._camera_order:
            raise ValueError("no camera sources available")

        init_pose = self._init_poses_server.get(side)
        if init_pose is None:
            init_pose = list(poses[-1])
            self._init_poses_server[side] = init_pose

        return {
            "images": images,
            "poses": poses,
            "grippers": grippers,
            "init_pose": list(init_pose),
            "arm_current_pose": list(poses[-1]),
        }

    def _build_pi05_observation_message(self) -> Dict[str, Any]:
        state_window, camera_window = self._select_observation_window()
        poses_left = [self._observation_pose9_for_pi05("left", self._pose_for_side(sample, "left")) for sample in state_window]
        poses_right = [
            self._observation_pose9_for_pi05("right", self._pose_for_side(sample, "right")) for sample in state_window
        ]
        grippers_left = [[float(self._gripper_for_side(sample, "left"))] for sample in state_window]
        grippers_right = [[float(self._gripper_for_side(sample, "right"))] for sample in state_window]
        images = self._build_pi05_images(camera_window)
        return build_pi05_observation_payload(
            images=images,
            poses_left=poses_left,
            grippers_left=grippers_left,
            poses_right=poses_right,
            grippers_right=grippers_right,
            prompt=self.config.task_prompt,
        )

    def _build_pi05_images(self, camera_window: Dict[str, List[CameraSample]]) -> Dict[str, bytes | str]:
        if not self._camera_order:
            raise ValueError("no camera sources available")
        images: Dict[str, bytes | str] = {role: "" for role in PI05_IMAGE_ROLES}
        for camera_name in self._camera_order:
            role = self._pi05_role_for_camera_name(camera_name)
            if role is None:
                continue
            samples = camera_window.get(camera_name)
            if not samples:
                raise ValueError(f"missing camera sample for {camera_name}")
            jpeg = self._encode_jpeg_bytes(np.asarray(samples[-1].frame), self.config.jpeg_quality)
            images[role] = jpeg
        right_hand = images.get("right_hand")
        third_front = images.get("third_front")
        if (not isinstance(third_front, (bytes, bytearray)) or not third_front) and isinstance(
            right_hand, (bytes, bytearray)
        ) and right_hand:
            # 兼容当前 8017 服务端：在未提供独立 front view 时，用右手图填 third_front。
            images["third_front"] = bytes(right_hand)
        for role in ("head_fpv", "right_hand", "third_front"):
            value = images.get(role)
            if not isinstance(value, (bytes, bytearray)) or not value:
                raise ValueError(f"pi05 observation requires non-empty image role: {role}")
        return images

    @staticmethod
    def _pi05_role_for_camera_name(camera_name: str) -> Optional[str]:
        normalized = str(camera_name).strip().lower()
        mapping = {
            "front": "third_front",
            "third_front": "third_front",
            "head": "head_fpv",
            "head_fpv": "head_fpv",
            "left": "left_hand",
            "left_wrist": "left_hand",
            "left_hand": "left_hand",
            "right": "right_hand",
            "right_wrist": "right_hand",
            "right_hand": "right_hand",
        }
        return mapping.get(normalized)

    @staticmethod
    def _encode_jpeg_bytes(image: np.ndarray, quality: int) -> bytes:
        import cv2

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("image must have shape HxWx3")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise ValueError("failed to encode JPEG")
        return bytes(encoded.tobytes())

    def _observation_pose9_for_pi05(self, side: str, pose: Any) -> List[float]:
        if self.pose_transformer is not None:
            server_pose7 = self.pose_transformer.observation_to_server(side, pose)
            matrix = pose7_xyzw_to_matrix(server_pose7)
        else:
            matrix = np.asarray(pose, dtype=np.float64).tolist()
        return list(matrix_to_pose9_rot6d(matrix))

    def _log_current_observation_point(self, observation: Dict[str, Any]) -> None:
        if self.config.protocol_profile == "pi05_dual_arm_20d":
            poses_left = observation.get("poses_left")
            poses_right = observation.get("poses_right")
            images = observation.get("images")
            if not isinstance(poses_left, list) or not poses_left or len(poses_left[-1]) != 9:
                raise ValueError("pi05 poses_left must be non-empty pose9 history")
            if not isinstance(poses_right, list) or not poses_right or len(poses_right[-1]) != 9:
                raise ValueError("pi05 poses_right must be non-empty pose9 history")
            if not isinstance(images, dict):
                raise ValueError("pi05 images must be a role mapping")
            logger.info(
                "[ONLINE_INFERENCE][OBS][pi05] camera_order=%s image_roles=%s n_left=%d n_right=%d "
                "left_current_pose9=%s right_current_pose9=%s prompt=%r",
                list(self._camera_order),
                sorted(str(key) for key, value in images.items() if value),
                len(poses_left),
                len(poses_right),
                [float(value) for value in poses_left[-1]],
                [float(value) for value in poses_right[-1]],
                self.config.task_prompt,
            )
            return
        for arm_key in ("arm_l", "arm_r"):
            arm_payload = observation.get(arm_key)
            if not isinstance(arm_payload, dict):
                continue
            pose = arm_payload.get("arm_current_pose")
            grippers = arm_payload.get("grippers")
            if not isinstance(pose, list) or len(pose) != 7:
                raise ValueError(f"{arm_key}.arm_current_pose must be pose7")
            if not isinstance(grippers, list) or not grippers:
                raise ValueError(f"{arm_key}.grippers must be non-empty")
            images = arm_payload.get("images")
            if not isinstance(images, list):
                raise ValueError(f"{arm_key}.images must be a list")
            logger.info(
                "[ONLINE_INFERENCE][OBS] %s camera_order=%s n_images=%d n_poses=%d "
                "current_pose=%s gripper_history=%s current_gripper=%.6f",
                arm_key,
                list(self._camera_order),
                len(images),
                len(arm_payload.get("poses") or []),
                [float(value) for value in pose],
                [float(value) for value in grippers],
                float(grippers[-1]),
            )

    def _select_observation_window(self) -> tuple[List[RobotStateSample], Dict[str, List[CameraSample]]]:
        if len(self._state_history) < self.config.n_obs_steps:
            raise ValueError("state history is not ready")

        camera_names = list(self._required_camera_names or self._camera_order)
        if not camera_names:
            raise ValueError("no camera sources available")
        for name in camera_names:
            if name not in self._camera_history_by_name or not self._camera_history_by_name[name]:
                raise ValueError("camera history is not ready")

        self._camera_order = camera_names
        state_latest_ns = max(int(sample.host_monotonic_ns) for sample in self._state_history)
        camera_latest_ns = min(
            max(int(sample.host_monotonic_ns) for sample in self._camera_history_by_name[name])
            for name in camera_names
        )
        target_latest_ns = min(state_latest_ns, camera_latest_ns)
        target_times = [
            int(target_latest_ns - (self.config.n_obs_steps - 1 - index) * self._observation_step_ns)
            for index in range(self.config.n_obs_steps)
        ]
        first_target_ns = int(target_times[0])
        last_target_ns = int(target_times[-1])

        state_earliest_ns = min(int(sample.host_monotonic_ns) for sample in self._state_history)
        if state_earliest_ns > first_target_ns or state_latest_ns < last_target_ns:
            raise ValueError("state history does not cover observation window")

        for name in camera_names:
            history = self._camera_history_by_name[name]
            camera_earliest_ns = min(int(sample.host_monotonic_ns) for sample in history)
            camera_latest_for_name_ns = max(int(sample.host_monotonic_ns) for sample in history)
            if camera_earliest_ns > first_target_ns or camera_latest_for_name_ns < last_target_ns:
                raise ValueError(f"camera history does not cover observation window: {name}")

        state_window: List[RobotStateSample] = []
        for target_ns in target_times:
            state_sample = self._nearest_state(target_ns)
            if state_sample is None:
                raise ValueError("state history does not cover observation window")
            state_window.append(state_sample)

        camera_window: Dict[str, List[CameraSample]] = {}
        for name in camera_names:
            history = self._camera_history_by_name[name]
            selected_samples: List[CameraSample] = []
            for target_ns in target_times:
                camera_sample = self._nearest_camera(history, target_ns)
                if camera_sample is None:
                    raise ValueError(f"camera history does not cover observation window: {name}")
                selected_samples.append(camera_sample)
            camera_window[name] = selected_samples
        return state_window, camera_window

    def _nearest_state(self, target_ns: int) -> Optional[RobotStateSample]:
        best_sample = None
        best_delta = None
        for sample in self._state_history:
            delta = abs(int(sample.host_monotonic_ns) - int(target_ns))
            if delta > self.max_camera_delta_ns:
                continue
            if best_delta is None or delta < best_delta:
                best_sample = sample
                best_delta = delta
        return best_sample

    def _nearest_camera(self, history: Sequence[CameraSample], target_ns: int) -> Optional[CameraSample]:
        best_sample = None
        best_delta = None
        for sample in history:
            delta = abs(int(sample.host_monotonic_ns) - int(target_ns))
            if delta > self.max_camera_delta_ns:
                continue
            if best_delta is None or delta < best_delta:
                best_sample = sample
                best_delta = delta
        return best_sample

    @staticmethod
    def _pose_for_side(sample: RobotStateSample, side: str) -> Any:
        return sample.left_pose if side == "left" else sample.right_pose

    @staticmethod
    def _gripper_for_side(sample: RobotStateSample, side: str) -> float:
        return float(sample.left_gripper_width if side == "left" else sample.right_gripper_width)

    def _observation_pose_to_server(self, side: str, pose: Any) -> List[float]:
        try:
            if self.pose_transformer is not None:
                return list(self.pose_transformer.observation_to_server(side, pose))
            return list(matrix_to_pose7_xyzw(np.asarray(pose, dtype=np.float64).tolist()))
        except Exception as exc:
            self._fail(f"failed to transform observation pose for {side}: {exc}")
            raise

    def _poll_action_and_maybe_execute(self, state_sample: RobotStateSample) -> Optional[OnlineInferenceStep]:
        try:
            payload = self.transport.recv_json_nonblocking()
        except Exception as exc:
            self._transport_reset_done = False
            self._fail(f"action recv failed: {exc}")
            return self._hold_step(state_sample, "failed", {"error": self.error})
        if payload is None:
            if self._waiting_since_ns is not None:
                timeout_ns = int(self.config.response_timeout_sec * 1_000_000_000)
                if int(self.clock_ns()) - self._waiting_since_ns > timeout_ns:
                    self._fail("action response timeout")
                    return self._hold_step(state_sample, "failed", {"error": self.error})
            return self._hold_step(
                state_sample,
                "waiting_action",
                {},
            )

        if isinstance(payload, dict) and payload.get("type") == "reset_ack":
            return self._hold_step(
                state_sample,
                "waiting_action",
                {"ignored_response_type": "reset_ack"},
            )

        action_recv_ns = int(self.clock_ns())
        action_recv_perf_ns = int(self.trace_clock_ns())
        try:
            self._load_action_payload(
                payload,
                action_recv_ns=action_recv_ns,
                action_recv_perf_ns=action_recv_perf_ns,
            )
        except Exception as exc:
            self._fail(f"invalid action response: {exc}")
            return self._hold_step(state_sample, "failed", {"error": self.error})
        return self._emit_action_step(state_sample)

    def _load_action_payload(
        self,
        payload: Any,
        action_recv_ns: Optional[int] = None,
        action_recv_perf_ns: Optional[int] = None,
    ) -> None:
        if not isinstance(payload, dict):
            raise ValueError("action payload must be an object")

        if self.config.protocol_profile == "pi05_dual_arm_20d":
            actions = validate_pi05_action_sequence(payload, expected_dim=20)
            left_steps, right_steps = pi05_action_sequence_to_pose7_chunks(actions, self.config.arm_side)
        else:
            parsed = parse_action_chunk(payload, self.config.arm_side)
            left_steps = parsed.left
            right_steps = parsed.right

        if self.config.arm_side == "both":
            if left_steps is None or right_steps is None:
                raise ValueError("arm_side='both' requires action_l/action_r chunks")
            chunk_size = max(len(left_steps), len(right_steps))
        elif self.config.arm_side == "left":
            if left_steps is None:
                raise ValueError("missing left action chunk")
            chunk_size = len(left_steps)
        else:
            if right_steps is None:
                raise ValueError("missing right action chunk")
            chunk_size = len(right_steps)

        delay_ms = payload.get("post_action_delay_ms", self.config.post_action_delay_ms)
        delay_ms_value = int(delay_ms)
        if delay_ms_value < 0:
            raise ValueError("post_action_delay_ms must be non-negative")
        if delay_ms_value > self.config.max_post_action_delay_ms:
            raise ValueError("post_action_delay_ms exceeds max_post_action_delay_ms")

        self._chunk_seq += 1
        self._current_chunk_seq = self._chunk_seq
        self._last_action_debug = self._make_action_debug(left_steps, right_steps, chunk_size)
        self._last_action_debug.update(
            {
                "chunk_seq": int(self._current_chunk_seq),
                "chunk_index": 0,
                "chunk_size": int(chunk_size),
            }
        )
        self._log_action_chunk(left_steps, right_steps, chunk_size)
        self._current_chunk_left = left_steps
        self._current_chunk_right = right_steps
        self._current_chunk_index = 0
        self._current_chunk_size = chunk_size
        self._current_post_action_delay_ns = delay_ms_value * 1_000_000
        self._current_action_recv_ns = int(action_recv_ns) if action_recv_ns is not None else int(self.clock_ns())
        self._current_action_recv_perf_ns = (
            int(action_recv_perf_ns) if action_recv_perf_ns is not None else int(self.trace_clock_ns())
        )
        self._current_step_start_ns = int(self._current_action_recv_ns)
        self._current_chunk_started_ns = int(self._current_step_start_ns)
        self._current_step_duration_ns = max(1, int(self.config.action_step_sec * 1_000_000_000))

        self._waiting_since_ns = None
        self.status = "executing_chunk"

    def _log_action_chunk(
        self,
        left_steps: Optional[List[np.ndarray]],
        right_steps: Optional[List[np.ndarray]],
        chunk_size: int,
    ) -> None:
        for side, steps in (("left", left_steps), ("right", right_steps)):
            if not steps:
                continue
            action_array = np.asarray(steps, dtype=np.float64)
            if action_array.ndim != 2 or action_array.shape[1] != 8:
                raise ValueError(f"{side} action chunk must have shape Nx8, got {action_array.shape}")
            first_step = action_array[0]
            last_step = action_array[-1]
            grippers = action_array[:, 7]
            logger.info(
                "[ONLINE_INFERENCE][ACTION] %s chunk_size=%d "
                "first_pose=%s first_gripper=%.6f "
                "last_pose=%s last_gripper=%.6f "
                "gripper_min=%.6f gripper_max=%.6f",
                side,
                int(chunk_size),
                [float(value) for value in first_step[:7].tolist()],
                float(first_step[7]),
                [float(value) for value in last_step[:7].tolist()],
                float(last_step[7]),
                float(np.min(grippers)),
                float(np.max(grippers)),
            )

    def _make_observation_debug(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.protocol_profile == "pi05_dual_arm_20d":
            poses_left = observation.get("poses_left") or []
            poses_right = observation.get("poses_right") or []
            grippers_left = observation.get("grippers_left") or []
            grippers_right = observation.get("grippers_right") or []
            images = observation.get("images") or {}
            return {
                "profile": "pi05_dual_arm_20d",
                "prompt": str(observation.get("prompt", "")),
                "image_roles": sorted(str(key) for key, value in images.items() if value),
                "left": {
                    "n_poses": len(poses_left) if isinstance(poses_left, list) else 0,
                    "arm_current_pose": self._float_list_or_none(poses_left[-1] if poses_left else None),
                    "last_gripper": self._nested_last_float_or_none(grippers_left),
                },
                "right": {
                    "n_poses": len(poses_right) if isinstance(poses_right, list) else 0,
                    "arm_current_pose": self._float_list_or_none(poses_right[-1] if poses_right else None),
                    "last_gripper": self._nested_last_float_or_none(grippers_right),
                },
            }
        debug: Dict[str, Any] = {}
        for key, side in (("arm_l", "left"), ("arm_r", "right")):
            arm_payload = observation.get(key)
            if not isinstance(arm_payload, dict):
                continue
            poses = arm_payload.get("poses") or []
            grippers = arm_payload.get("grippers") or []
            arm_current_pose = arm_payload.get("arm_current_pose")
            init_pose = arm_payload.get("init_pose")
            debug[side] = {
                "n_poses": len(poses) if isinstance(poses, list) else 0,
                "n_images": len(arm_payload.get("images") or []),
                "arm_current_pose": self._float_list_or_none(arm_current_pose),
                "init_pose": self._float_list_or_none(init_pose),
                "last_gripper": float(grippers[-1]) if isinstance(grippers, list) and grippers else None,
            }
        return debug

    def _make_action_debug(
        self,
        left_steps: Optional[List[np.ndarray]],
        right_steps: Optional[List[np.ndarray]],
        chunk_size: int,
    ) -> Dict[str, Any]:
        debug: Dict[str, Any] = {}
        if self.config.protocol_profile == "pi05_dual_arm_20d":
            debug["profile"] = "pi05_dual_arm_20d"
        for side, steps in (("left", left_steps), ("right", right_steps)):
            if not steps:
                continue
            first_step = np.asarray(steps[0], dtype=np.float64)
            side_observation = self._last_observation_debug.get(side, {})
            arm_current_pose = side_observation.get("arm_current_pose")
            delta_xyz_m = None
            if arm_current_pose is not None and len(arm_current_pose) >= 3:
                current_xyz = np.asarray(arm_current_pose[:3], dtype=np.float64)
                action_xyz = np.asarray(first_step[:3], dtype=np.float64)
                delta_xyz_m = float(np.linalg.norm(action_xyz - current_xyz))
            debug[side] = {
                "chunk_size": int(chunk_size),
                "first_step": [float(value) for value in first_step.tolist()],
                "delta_xyz_m": delta_xyz_m,
            }
        return debug

    @staticmethod
    def _float_list_or_none(values: Any) -> Optional[List[float]]:
        if not isinstance(values, (list, tuple)):
            return None
        return [float(value) for value in values]

    @staticmethod
    def _nested_last_float_or_none(values: Any) -> Optional[float]:
        if not isinstance(values, (list, tuple)) or not values:
            return None
        last = values[-1]
        if isinstance(last, (list, tuple)) and last:
            return float(last[0])
        return float(last)

    def _emit_action_step(self, state_sample: RobotStateSample) -> OnlineInferenceStep:
        if self._current_chunk_index >= self._current_chunk_size:
            self._enter_post_action_delay()
            return self._hold_step(
                state_sample,
                "post_action_delay",
                {"ready_after_ns": self._post_action_ready_after_ns},
            )

        now_ns = int(self.clock_ns())
        self._advance_due_steps(now_ns)
        if self._current_chunk_index >= self._current_chunk_size:
            self._enter_post_action_delay()
            if self._post_action_ready_after_ns is not None:
                timeout_base_ns = self._current_chunk_started_ns or self._post_action_ready_after_ns
                if now_ns - int(timeout_base_ns) >= self._observation_ready_timeout_ns:
                    self._fail("post_action_delay observation coverage timeout")
                    return self._hold_step(state_sample, "failed", {"error": self.error})
            return self._hold_step(
                state_sample,
                "post_action_delay",
                {"ready_after_ns": self._post_action_ready_after_ns},
            )

        step_index = self._current_chunk_index
        left_pose, right_pose, left_gripper_width, right_gripper_width = self._build_action_output(
            state_sample,
            step_index,
            now_ns,
        )
        if self.config.chunk_step_mode == "per_tick":
            self._current_step_start_ns = now_ns
            self._current_chunk_index += 1
        metadata = self._make_action_step_metadata(
            step_index=step_index,
            step_output_ns=now_ns,
            step_output_perf_ns=int(self.trace_clock_ns()),
        )
        return OnlineInferenceStep(
            left_pose=left_pose,
            right_pose=right_pose,
            left_gripper_width=left_gripper_width,
            right_gripper_width=right_gripper_width,
            enabled_arms=self._enabled_arms_for_action(),
            status="executing_chunk",
            metadata=metadata,
        )

    def _make_action_step_metadata(
        self,
        *,
        step_index: int,
        step_output_ns: int,
        step_output_perf_ns: int,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "chunk_index": int(step_index),
            "chunk_size": int(self._current_chunk_size),
            "online_chunk_index": int(step_index),
            "online_chunk_size": int(self._current_chunk_size),
            "online_step_output_ns": int(step_output_ns),
            "online_step_output_perf_ns": int(step_output_perf_ns),
            "online_observation_seq": int(self._current_observation_seq),
            "online_chunk_seq": int(self._current_chunk_seq),
            "online_chunk_step_mode": self.config.chunk_step_mode,
        }
        if self._current_observation_send_ns is not None:
            metadata["online_obs_send_ns"] = int(self._current_observation_send_ns)
        if self._current_observation_send_perf_ns is not None:
            metadata["online_obs_send_perf_ns"] = int(self._current_observation_send_perf_ns)
        if self._current_action_recv_ns is not None:
            metadata["online_action_recv_ns"] = int(self._current_action_recv_ns)
        if self._current_action_recv_perf_ns is not None:
            metadata["online_action_recv_perf_ns"] = int(self._current_action_recv_perf_ns)
        if self._current_observation_send_perf_ns is not None and self._current_action_recv_perf_ns is not None:
            metadata["online_obs_send_to_action_recv_ms"] = (
                int(self._current_action_recv_perf_ns) - int(self._current_observation_send_perf_ns)
            ) / 1e6
        return metadata

    def _advance_due_steps(self, now_ns: int) -> None:
        if self._current_step_start_ns is None:
            self._current_step_start_ns = now_ns
        step_duration_ns = self._current_step_duration_ns
        while (
            self._current_chunk_index < self._current_chunk_size
            and now_ns - int(self._current_step_start_ns) >= step_duration_ns
        ):
            self._current_step_start_ns = int(self._current_step_start_ns) + step_duration_ns
            self._current_chunk_index += 1
            if self._current_chunk_index >= self._current_chunk_size:
                break

    def _enter_post_action_delay(self) -> None:
        self.status = "post_action_delay"
        self._post_action_entered_ns = int(self.clock_ns())
        if self._current_step_start_ns is None:
            self._current_step_start_ns = int(self.clock_ns())
        self._post_action_ready_after_ns = int(self._current_step_start_ns) + self._current_post_action_delay_ns
        self._current_chunk_left = None
        self._current_chunk_right = None
        self._current_chunk_index = 0
        self._current_chunk_size = 0
        self._current_step_start_ns = None
        self._current_action_recv_ns = None
        self._current_action_recv_perf_ns = None

    def _build_action_output(
        self,
        state_sample: RobotStateSample,
        step_index: int,
        now_ns: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        left_pose = np.asarray(state_sample.left_pose, dtype=np.float64).copy()
        right_pose = np.asarray(state_sample.right_pose, dtype=np.float64).copy()
        left_gripper_width = float(state_sample.left_gripper_width)
        right_gripper_width = float(state_sample.right_gripper_width)

        left_current = None
        right_current = None
        if self._current_chunk_left is not None and step_index < len(self._current_chunk_left):
            left_current = self._current_chunk_left[step_index]
        if self._current_chunk_right is not None and step_index < len(self._current_chunk_right):
            right_current = self._current_chunk_right[step_index]

        next_index = step_index + 1
        left_next = self._current_chunk_left[next_index] if self._current_chunk_left is not None and next_index < len(self._current_chunk_left) else None
        right_next = self._current_chunk_right[next_index] if self._current_chunk_right is not None and next_index < len(self._current_chunk_right) else None

        if left_current is not None:
            left_pose, left_gripper_width = self._interpolate_step(
                side="left",
                current_step=left_current,
                next_step=left_next,
                now_ns=now_ns,
            )
        if right_current is not None:
            right_pose, right_gripper_width = self._interpolate_step(
                side="right",
                current_step=right_current,
                next_step=right_next,
                now_ns=now_ns,
            )
        return left_pose, right_pose, left_gripper_width, right_gripper_width

    def _interpolate_step(
        self,
        side: str,
        current_step: np.ndarray,
        next_step: Optional[np.ndarray],
        now_ns: int,
    ) -> tuple[np.ndarray, float]:
        current_pose = self._action_pose_to_unitree(side, current_step[:7])
        current_gripper = float(current_step[7])
        if next_step is None or self.config.chunk_step_mode == "per_tick":
            return current_pose, current_gripper

        if self._current_step_start_ns is None:
            alpha = 0.0
        else:
            elapsed_ns = max(0, now_ns - int(self._current_step_start_ns))
            alpha = float(elapsed_ns) / float(self._current_step_duration_ns)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            if self.config.interpolation_interval_sec > 0.0:
                quantum = float(self.config.interpolation_interval_sec / self.config.action_step_sec)
                if quantum > 0.0:
                    alpha = float(np.clip(round(alpha / quantum) * quantum, 0.0, 1.0))

        blended_pose7 = self._blend_pose7_xyzw(current_step[:7], next_step[:7], alpha)
        blended_pose = self._action_pose_to_unitree(side, blended_pose7)
        blended_gripper = float((1.0 - alpha) * float(current_step[7]) + alpha * float(next_step[7]))
        return blended_pose, blended_gripper

    @staticmethod
    def _blend_pose7_xyzw(current_pose7: Any, next_pose7: Any, alpha: float) -> np.ndarray:
        current = np.asarray(current_pose7, dtype=np.float64)
        next_pose = np.asarray(next_pose7, dtype=np.float64)
        if current.shape != (7,) or next_pose.shape != (7,):
            raise ValueError("pose7 must have shape (7,)")
        xyz = (1.0 - alpha) * current[:3] + alpha * next_pose[:3]
        quat_a = current[3:7].copy()
        quat_b = next_pose[3:7].copy()
        if float(np.dot(quat_a, quat_b)) < 0.0:
            quat_b = -quat_b
        quat = (1.0 - alpha) * quat_a + alpha * quat_b
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 1e-9:
            quat = quat_a
            quat_norm = float(np.linalg.norm(quat))
        quat = quat / quat_norm
        return np.asarray([xyz[0], xyz[1], xyz[2], quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)

    def _action_pose_to_unitree(self, side: str, pose7: Any) -> np.ndarray:
        try:
            if self.pose_transformer is not None:
                return np.asarray(self.pose_transformer.action_to_unitree(side, pose7), dtype=np.float64)
            return np.asarray(pose7_xyzw_to_matrix(np.asarray(pose7, dtype=np.float64).tolist()), dtype=np.float64)
        except Exception as exc:
            self._fail(f"failed to transform action pose for {side}: {exc}")
            raise

    def _enabled_arms_for_action(self) -> List[str]:
        if not self.config.enable_motion or self.config.dry_run:
            return []
        if self.config.arm_side == "both":
            return ["left", "right"]
        return [self.config.arm_side]

    def _post_action_delay_satisfied(
        self,
        now_ns: int,
        state_sample: RobotStateSample,
        camera_samples: Sequence[CameraSample],
    ) -> bool:
        if self._post_action_ready_after_ns is None:
            return True
        if now_ns < self._post_action_ready_after_ns:
            return False
        if int(state_sample.host_monotonic_ns) < self._post_action_ready_after_ns:
            return False
        camera_names = list(self._required_camera_names or self._camera_order)
        if not camera_names:
            return False
        for name in camera_names:
            history = self._camera_history_by_name.get(name) or []
            if not history:
                return False
            if max(int(sample.host_monotonic_ns) for sample in history) < self._post_action_ready_after_ns:
                return False
        return True

    def _hold_step(
        self,
        state_sample: RobotStateSample,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OnlineInferenceStep:
        return OnlineInferenceStep(
            left_pose=np.asarray(state_sample.left_pose, dtype=np.float64).copy(),
            right_pose=np.asarray(state_sample.right_pose, dtype=np.float64).copy(),
            left_gripper_width=float(state_sample.left_gripper_width),
            right_gripper_width=float(state_sample.right_gripper_width),
            enabled_arms=[],
            status=status,
            metadata=metadata or {},
        )

    def _fail(self, reason: str) -> None:
        self.status = "failed"
        self.error = str(reason)
        self._waiting_since_ns = None
        self._current_chunk_left = None
        self._current_chunk_right = None
        self._current_chunk_index = 0
        self._current_chunk_size = 0
        self._current_chunk_started_ns = None
        self._post_action_ready_after_ns = None
        self._post_action_entered_ns = None
        self._current_step_start_ns = None
        self._collecting_since_ns = None
