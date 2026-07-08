from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from inference.online_session import (
    CameraSample,
    OnlineInferenceConfig,
    OnlineInferenceSession,
    OnlineInferenceStep,
    RobotStateSample,
    _monotonic_ns,
    _perf_counter_ns,
)
from inference.pi05_protocol import pi05_action_sequence_to_pose7_chunks, validate_pi05_action_sequence


@dataclass(init=False)
class RTCConfig:
    horizon: int
    request_step_s: int
    predicted_delay_steps_d: int

    def __init__(
        self,
        horizon: int,
        request_step_s: int,
        predicted_delay_steps_d: int | None = None,
        delay_steps_d: int | None = None,
    ) -> None:
        if predicted_delay_steps_d is None:
            if delay_steps_d is None:
                raise TypeError("RTCConfig requires predicted_delay_steps_d")
            predicted_delay_steps_d = delay_steps_d
        elif delay_steps_d is not None and int(predicted_delay_steps_d) != int(delay_steps_d):
            raise ValueError("RTCConfig got conflicting predicted_delay_steps_d and delay_steps_d")
        self.horizon = horizon
        self.request_step_s = request_step_s
        self.predicted_delay_steps_d = predicted_delay_steps_d
        self.__post_init__()

    def __post_init__(self) -> None:
        self.horizon = int(self.horizon)
        self.request_step_s = int(self.request_step_s)
        self.predicted_delay_steps_d = int(self.predicted_delay_steps_d)
        if self.horizon <= 0:
            raise ValueError("rtc horizon must be positive")
        if not 0 < self.request_step_s < self.horizon:
            raise ValueError("rtc request_step_s must satisfy 0 < s < horizon")
        if not 0 <= self.predicted_delay_steps_d < self.request_step_s:
            raise ValueError("rtc predicted_delay_steps_d must satisfy 0 <= d < s")
        if self.request_step_s + self.predicted_delay_steps_d >= self.horizon:
            raise ValueError("rtc request_step_s + predicted_delay_steps_d must be < horizon")

    @property
    def delay_steps_d(self) -> int:
        """Deprecated compatibility name. This is a predicted delay, not a fixed takeover cursor."""
        return int(self.predicted_delay_steps_d)


@dataclass
class OnlineInferenceRTCConfig(OnlineInferenceConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.protocol_profile != "pi05_dual_arm_20d":
            raise ValueError("online_inference_rtc requires protocol_profile='pi05_dual_arm_20d'")
        if self.arm_side != "both":
            raise ValueError("online_inference_rtc requires arm_side='both'")
        if self.chunk_step_mode != "per_tick":
            raise ValueError("online_inference_rtc requires chunk_step_mode='per_tick'")


class AsyncInferenceTransport:
    def __init__(self, transport: Any) -> None:
        self.transport = transport
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="online-inference-rtc")
        self._future: Future | None = None

    def is_connected(self) -> bool:
        return bool(self.transport.is_connected())

    def reset(self) -> None:
        if self._future is not None and not self._future.done():
            raise RuntimeError("cannot reset online inference transport while an async request is pending")
        self._future = None
        self.transport.reset()

    def clear_pending_rx(self) -> None:
        clear_pending_rx = getattr(self.transport, "clear_pending_rx", None)
        if callable(clear_pending_rx):
            clear_pending_rx()

    def send_json_async(self, payload: dict) -> None:
        if self._future is not None and not self._future.done():
            raise RuntimeError("online inference async request already pending")
        self._future = self._executor.submit(self._send_and_recv, payload)

    def recv_json_nonblocking(self):
        if self._future is None or not self._future.done():
            return None
        future = self._future
        self._future = None
        return future.result()

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _send_and_recv(self, payload: dict):
        self.transport.send_json(payload)
        return self.transport.recv_json_nonblocking()


class OnlineInferenceRTCSession(OnlineInferenceSession):
    def __init__(
        self,
        config: OnlineInferenceRTCConfig,
        rtc_config: RTCConfig,
        transport: Any,
        pose_transformer: Optional[Any] = None,
        clock_ns: Optional[Callable[[], int]] = None,
        trace_clock_ns: Optional[Callable[[], int]] = None,
    ) -> None:
        if not callable(getattr(transport, "send_json_async", None)):
            raise ValueError("online_inference_rtc transport must provide send_json_async(payload)")
        super().__init__(
            config=config,
            transport=transport,
            pose_transformer=pose_transformer,
            clock_ns=clock_ns or _monotonic_ns,
            trace_clock_ns=trace_clock_ns or _perf_counter_ns,
        )
        self.rtc_config = rtc_config
        self._current_chunk_pose20: np.ndarray | None = None
        self._rtc_request_pending = False
        self._rtc_request_seq = 0
        self._rtc_request_chunk_seq: int | None = None
        self._rtc_request_step_s: int | None = None
        self._rtc_request_predicted_d: int | None = None
        self._rtc_request_sent_ns: int | None = None
        self._rtc_apply_next_metadata: dict[str, Any] = {}

    def tick(
        self,
        state_sample: RobotStateSample,
        camera_samples: list[CameraSample] | tuple[CameraSample, ...],
    ) -> OnlineInferenceStep:
        self._append_history(state_sample, camera_samples)

        if self.status == "failed":
            return self._hold_step(state_sample, "failed", {"error": self.error})
        if not self._transport_connected():
            self._transport_reset_done = False
            self._fail("transport disconnected")
            return self._hold_step(state_sample, "failed", {"error": self.error})

        if self.status == "waiting_action":
            payload = self.transport.recv_json_nonblocking()
            if payload is None:
                if self._waiting_since_ns is not None and self._response_timed_out(self._waiting_since_ns):
                    self._fail("RTC initial action response timeout")
                    return self._hold_step(state_sample, "failed", {"error": self.error})
                return self._hold_step(state_sample, "waiting_action", {})
            if isinstance(payload, dict) and payload.get("type") == "reset_ack":
                return self._hold_step(state_sample, "waiting_action", {"ignored_response_type": "reset_ack"})
            self._load_pose20_action_payload(payload)
            return self._emit_action_step(state_sample)

        if self.status == "executing_chunk":
            if self._rtc_request_pending:
                payload = self.transport.recv_json_nonblocking()
                if payload is not None:
                    self._apply_rtc_action_payload(payload)
                elif self._rtc_request_sent_ns is not None and self._response_timed_out(self._rtc_request_sent_ns):
                    self._fail("RTC pending action response timeout")
                    return self._hold_step(state_sample, "failed", {"error": self.error})
            return self._emit_action_step(state_sample)

        if self.status == "post_action_delay":
            return super().tick(state_sample=state_sample, camera_samples=camera_samples)

        if not self._history_ready():
            return self._hold_step(state_sample, "collecting_observation", {"history_len": len(self._state_history)})

        self._ensure_transport_reset()
        observation = self._build_observation_message()
        self._log_current_observation_point(observation)
        self._last_observation_debug = self._make_observation_debug(observation)
        observation_send_ns = int(self.clock_ns())
        observation_send_perf_ns = int(self.trace_clock_ns())
        self.transport.send_json_async(observation)
        self._waiting_since_ns = observation_send_ns
        self._current_observation_send_perf_ns = observation_send_perf_ns
        self._observation_seq += 1
        self._current_observation_seq = self._observation_seq
        self._current_observation_send_ns = int(self._waiting_since_ns)
        self._collecting_since_ns = None
        self.status = "waiting_action"
        return self._hold_step(state_sample, "waiting_action", {"history_len": len(self._state_history)})

    def _load_pose20_action_payload(
        self,
        payload: Any,
        action_recv_ns: Optional[int] = None,
        action_recv_perf_ns: Optional[int] = None,
    ) -> None:
        actions = validate_pi05_action_sequence(payload, expected_dim=20)
        expected_shape = (self.rtc_config.horizon, 20)
        if actions.shape != expected_shape:
            raise ValueError(f"online_inference_rtc action chunk expected shape {expected_shape}, got {actions.shape}")
        left_steps, right_steps = pi05_action_sequence_to_pose7_chunks(actions, self.config.arm_side)
        self._last_action_debug = self._make_action_debug(left_steps, right_steps, int(actions.shape[0]))
        self._log_action_chunk(left_steps, right_steps, int(actions.shape[0]))
        self._current_chunk_left = left_steps
        self._current_chunk_right = right_steps
        self._current_chunk_pose20 = actions.astype(np.float32, copy=True)
        self._current_chunk_index = 0
        self._current_chunk_size = int(actions.shape[0])
        self._current_action_recv_ns = int(action_recv_ns) if action_recv_ns is not None else int(self.clock_ns())
        self._current_action_recv_perf_ns = (
            int(action_recv_perf_ns) if action_recv_perf_ns is not None else int(self.trace_clock_ns())
        )
        self._current_step_start_ns = int(self._current_action_recv_ns)
        self._current_chunk_started_ns = int(self._current_step_start_ns)
        self._current_step_duration_ns = max(1, int(self.config.action_step_sec * 1_000_000_000))
        self._chunk_seq += 1
        self._current_chunk_seq = self._chunk_seq
        self._waiting_since_ns = None
        self._rtc_request_pending = False
        self._rtc_request_chunk_seq = None
        self._rtc_request_step_s = None
        self._rtc_request_predicted_d = None
        self._rtc_request_sent_ns = None
        self.status = "executing_chunk"

    def get_debug_snapshot(self) -> dict[str, Any]:
        snapshot = super().get_debug_snapshot()
        snapshot["rtc"] = {
            "enabled": True,
            "pending": bool(self._rtc_request_pending),
            "request_seq": int(self._rtc_request_seq),
            "horizon": int(self.rtc_config.horizon),
            "s": int(self.rtc_config.request_step_s),
            "d": int(self.rtc_config.predicted_delay_steps_d),
            "predicted_d": int(self.rtc_config.predicted_delay_steps_d),
            "predicted_delay_steps": int(self.rtc_config.predicted_delay_steps_d),
            "current_chunk_seq": int(self._current_chunk_seq),
            "current_chunk_index": int(self._current_chunk_index),
            "has_current_chunk": self._current_chunk_pose20 is not None,
            "request_sent_ns": self._rtc_request_sent_ns,
        }
        return snapshot

    def _apply_rtc_action_payload(self, payload: Any) -> None:
        if self._rtc_request_step_s is None:
            raise RuntimeError("RTC action response arrived without a recorded request step")
        predicted_d = int(
            self._rtc_request_predicted_d
            if self._rtc_request_predicted_d is not None
            else self.rtc_config.predicted_delay_steps_d
        )
        request_s = int(self._rtc_request_step_s)
        old_takeover_index = int(self._current_chunk_index)
        actual_d = int(old_takeover_index - request_s)
        if actual_d < 0:
            raise RuntimeError(
                f"RTC response cursor moved before request step: current={old_takeover_index}, request_s={request_s}"
            )
        self._load_pose20_action_payload(payload)
        if actual_d >= self._current_chunk_size:
            raise RuntimeError(
                f"RTC response arrived too late for new chunk: actual_d={actual_d}, chunk_size={self._current_chunk_size}"
            )
        self._current_chunk_index = actual_d
        self._rtc_apply_next_metadata = {
            "online_rtc_applied": True,
            "online_rtc_s": request_s,
            "online_rtc_d": predicted_d,
            "online_rtc_predicted_d": predicted_d,
            "online_rtc_predicted_delay_steps": predicted_d,
            "online_rtc_actual_d": actual_d,
            "online_rtc_actual_delay_steps": actual_d,
            "online_rtc_takeover_old_index": old_takeover_index,
        }

    def _emit_action_step(self, state_sample: RobotStateSample) -> OnlineInferenceStep:
        if (
            self._rtc_request_pending
            and self._current_chunk_size > 0
            and self._current_chunk_index >= self._current_chunk_size
        ):
            self._fail("RTC response did not arrive before chunk exhaustion")
            return self._hold_step(state_sample, "failed", {"error": self.error})
        step = super()._emit_action_step(state_sample)
        if step.status == "executing_chunk":
            step.metadata.update(
                {
                    "online_rtc_enabled": True,
                    "online_rtc_pending": bool(self._rtc_request_pending),
                    "online_rtc_request_seq": int(self._rtc_request_seq),
                    "online_rtc_horizon": int(self.rtc_config.horizon),
                }
            )
            if self._rtc_apply_next_metadata:
                step.metadata.update(self._rtc_apply_next_metadata)
                self._rtc_apply_next_metadata = {}
            self._maybe_send_rtc_request()
        return step

    def _maybe_send_rtc_request(self) -> None:
        if self._current_chunk_pose20 is None:
            return
        if self._rtc_request_pending:
            return
        if self._rtc_request_chunk_seq == self._current_chunk_seq:
            return
        if self._current_chunk_index < self.rtc_config.request_step_s:
            return
        observation = self._build_observation_message()
        observation["rtc"] = {
            "enabled": True,
            "old_chunk": self._current_chunk_pose20.astype(float).tolist(),
            "s": int(self.rtc_config.request_step_s),
            "d": int(self.rtc_config.predicted_delay_steps_d),
        }
        self._log_current_observation_point(observation)
        self._rtc_request_seq += 1
        self._rtc_request_pending = True
        self._rtc_request_chunk_seq = int(self._current_chunk_seq)
        self._rtc_request_step_s = int(self.rtc_config.request_step_s)
        self._rtc_request_predicted_d = int(self.rtc_config.predicted_delay_steps_d)
        self._rtc_request_sent_ns = int(self.clock_ns())
        self.transport.send_json_async(observation)

    def _response_timed_out(self, sent_ns: int) -> bool:
        timeout_ns = int(float(self.config.response_timeout_sec) * 1_000_000_000)
        return int(self.clock_ns()) - int(sent_ns) > timeout_ns
