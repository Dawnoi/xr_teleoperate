import json
import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import logging_mp

logger_mp = logging_mp.getLogger(__name__)


@dataclass
class TraceRecord:
    seq: int
    t_recv_ns: int
    recv_q: np.ndarray
    t_pub_ns: Optional[int] = None
    t_exec_ns: Optional[int] = None
    t_exec_thread_ns: Optional[int] = None
    status: str = "pending"
    fields: Dict[str, object] = field(default_factory=dict)


class SimpleLatencyTracker:
    """
    Single-flight latency tracker with per-stage breakdown fields.

    The active sample tracks one command path at a time so recorded timings are
    easy to attribute:
        receive -> per-stage processing -> DDS publish -> motion detected
    """

    def __init__(
        self,
        output_path: Optional[str],
        summary_every: int = 1,
        log_each_trace: bool = True,
        timeout_s: float = 2.0,
        ui_memory_provider_scope: Optional[set[str]] = None,
    ):
        self.output_path = str(output_path) if output_path else None
        self.summary_every = max(1, int(summary_every))
        self.log_each_trace = bool(log_each_trace)
        self.ui_memory_provider_scope = (
            {str(provider_name) for provider_name in ui_memory_provider_scope}
            if ui_memory_provider_scope is not None
            else None
        )
        self.timeout_ns = int(float(timeout_s) * 1e9)
        self._lock = threading.Lock()
        self._next_seq = 0
        self._active: Optional[TraceRecord] = None
        self._latest_payload: Optional[Dict[str, object]] = None
        self._completed = 0
        self._dropped = 0

        if self.output_path is not None:
            out_dir = os.path.dirname(os.path.abspath(self.output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

    def can_start_new_trace(self) -> bool:
        with self._lock:
            return self._active is None

    def tracks_input_provider(self, input_provider: str) -> bool:
        if self.ui_memory_provider_scope is None:
            return True
        return str(input_provider) in self.ui_memory_provider_scope

    def begin_trace(self, recv_ts_ns: int, recv_q, extra: Optional[Dict[str, object]] = None) -> Optional[int]:
        recv_q = np.asarray(recv_q, dtype=float).copy()
        with self._lock:
            if self._active is not None:
                return None
            self._next_seq += 1
            self._active = TraceRecord(
                seq=self._next_seq,
                t_recv_ns=int(recv_ts_ns),
                recv_q=recv_q,
                fields=dict(extra or {}),
            )
            return self._active.seq

    def set_fields(self, seq: Optional[int], **fields) -> None:
        if not seq:
            return
        with self._lock:
            if self._active is None or self._active.seq != int(seq):
                return
            for key, value in fields.items():
                self._active.fields[key] = value

    def get_snapshot(self) -> Dict[str, Optional[Dict[str, object]]]:
        with self._lock:
            current = self._build_payload(self._active) if self._active is not None else None
            latest = deepcopy(self._latest_payload) if self._latest_payload is not None else None
        return {"current": current, "latest": latest}

    def mark_publish(
        self,
        seq: Optional[int],
        publish_ts_ns: Optional[int] = None,
        **fields,
    ) -> None:
        if not seq:
            return
        with self._lock:
            if self._active is None or self._active.seq != int(seq):
                return
            if self._active.t_pub_ns is None:
                self._active.t_pub_ns = int(publish_ts_ns) if publish_ts_ns is not None else time.perf_counter_ns()
            for key, value in fields.items():
                self._active.fields[key] = value

    def _motion_trigger_values(self, active: TraceRecord, current_q, current_dq):
        current_q = np.asarray(current_q, dtype=float)
        current_dq = np.asarray(current_dq, dtype=float)
        if active.recv_q is not None and current_q.shape != active.recv_q.shape:
            raise ValueError(
                f"latency trace q shape mismatch: recv_q={active.recv_q.shape}, current_q={current_q.shape}"
            )
        q_delta = float(np.max(np.abs(current_q - active.recv_q))) if active.recv_q is not None else 0.0
        dq_peak = float(np.max(np.abs(current_dq))) if current_dq.size else 0.0
        return q_delta, dq_peak

    def maybe_mark_execute_thread(
        self,
        current_q,
        current_dq,
        q_threshold: float,
        dq_threshold: float,
        detect_ts_ns: Optional[int] = None,
    ):
        record_seq = None
        with self._lock:
            active = self._active
            if active is None or active.t_pub_ns is None or active.t_exec_thread_ns is not None:
                return None
            q_delta, dq_peak = self._motion_trigger_values(active, current_q, current_dq)
            if q_delta < float(q_threshold) and dq_peak < float(dq_threshold):
                return None
            active.t_exec_thread_ns = int(detect_ts_ns) if detect_ts_ns is not None else time.perf_counter_ns()
            active.fields["q_delta_thread_trigger"] = q_delta
            active.fields["dq_peak_thread_trigger"] = dq_peak
            record_seq = active.seq
        return record_seq

    def maybe_mark_execute(
        self,
        current_q,
        current_dq,
        q_threshold: float,
        dq_threshold: float,
        detect_ts_ns: Optional[int] = None,
    ):
        record = None
        with self._lock:
            active = self._active
            if active is None or active.t_pub_ns is None or active.t_exec_ns is not None:
                return None
            q_delta, dq_peak = self._motion_trigger_values(active, current_q, current_dq)
            if q_delta < float(q_threshold) and dq_peak < float(dq_threshold):
                return None
            active.t_exec_ns = int(detect_ts_ns) if detect_ts_ns is not None else time.perf_counter_ns()
            active.status = "completed"
            active.fields["q_delta_trigger"] = q_delta
            active.fields["dq_peak_trigger"] = dq_peak
            record = active
            self._active = None
            self._completed += 1
        if record is not None:
            self._finalize_record(record)
        return record.seq if record is not None else None

    def maybe_timeout(self):
        record = None
        with self._lock:
            active = self._active
            if active is None:
                return None
            now_ns = time.perf_counter_ns()
            if now_ns - active.t_recv_ns < self.timeout_ns:
                return None
            active.status = "timeout"
            if active.t_exec_ns is None:
                active.t_exec_ns = now_ns
            record = active
            self._active = None
            self._dropped += 1
        if record is not None:
            self._finalize_record(record)
        return record.seq if record is not None else None

    def _finalize_record(self, record: TraceRecord):
        payload = self._build_payload(record)
        with self._lock:
            self._latest_payload = deepcopy(payload)

        if self.output_path is not None:
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if self.log_each_trace:
            if record.status == "completed":
                logger_mp.info(
                    "[LATENCY] seq=%s recv->pub=%.2f ms | pub->exec=%.2f ms | recv->exec=%.2f ms | tele=%.2f base=%.2f(move=%.2f,height=%.2f,misc=%.2f) ik=%.2f queue=%.2f wait=%.2f dds=%.3f ms",
                    payload["seq"],
                    payload["recv_to_pub_ms"] or -1.0,
                    payload["pub_to_exec_ms"] or -1.0,
                    payload["recv_to_exec_ms"] or -1.0,
                    float(payload.get("tele_fetch_ms") or 0.0),
                    float(payload.get("base_control_ms") or 0.0),
                    float(payload.get("base_move_ms") or 0.0),
                    float(payload.get("base_height_ms") or 0.0),
                    float(payload.get("base_misc_ms") or 0.0),
                    float(payload.get("ik_ms") or 0.0),
                    float(payload.get("enqueue_to_publish_ms") or 0.0),
                    float(payload.get("controller_wait_ms") or 0.0),
                    float(payload.get("dds_write_ms") or 0.0),
                )
            else:
                logger_mp.warning(
                    "[LATENCY] seq=%s timeout status=%s recv->pub=%s ms recv->exec=%s ms",
                    payload["seq"],
                    payload["status"],
                    "%.2f" % payload["recv_to_pub_ms"] if payload["recv_to_pub_ms"] is not None else "N/A",
                    "%.2f" % payload["recv_to_exec_ms"] if payload["recv_to_exec_ms"] is not None else "N/A",
                )

        total = self._completed + self._dropped
        if self.output_path is not None and total % self.summary_every == 0:
            self._log_summary()

    def _build_payload(self, record: TraceRecord) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "seq": int(record.seq),
            "status": record.status,
            "t_recv_ns": int(record.t_recv_ns),
            "t_pub_ns": int(record.t_pub_ns) if record.t_pub_ns is not None else None,
            "t_exec_ns": int(record.t_exec_ns) if record.t_exec_ns is not None else None,
            "t_exec_thread_ns": int(record.t_exec_thread_ns) if record.t_exec_thread_ns is not None else None,
            "recv_to_pub_ms": self._delta_ms(record.t_recv_ns, record.t_pub_ns),
            "pub_to_exec_ms": self._delta_ms(record.t_pub_ns, record.t_exec_ns),
            "recv_to_exec_ms": self._delta_ms(record.t_recv_ns, record.t_exec_ns),
            "pub_to_exec_thread_ms": self._delta_ms(record.t_pub_ns, record.t_exec_thread_ns),
            "recv_to_exec_thread_ms": self._delta_ms(record.t_recv_ns, record.t_exec_thread_ns),
        }
        payload.update(record.fields)
        self._add_online_inference_derived_fields(payload)

        enqueue_to_publish_ms = payload.get("enqueue_to_publish_ms")
        dds_write_ms = payload.get("dds_write_ms")
        if enqueue_to_publish_ms is not None:
            if dds_write_ms is not None:
                payload["controller_wait_ms"] = max(
                    0.0,
                    float(enqueue_to_publish_ms) - float(dds_write_ms),
                )
            else:
                payload["controller_wait_ms"] = float(enqueue_to_publish_ms)
        else:
            payload["controller_wait_ms"] = None

        # Derived breakdown helpers.
        known_pre_publish_keys = [
            "takeover_logic_ms",
            "end_effector_command_ms",
            "base_control_ms",
            "arm_cmd_input_ms",
            "arm_cmd_takeover_reset_ms",
            "arm_cmd_target_extra_ms",
            "ik_ms",
            "arm_cmd_feedback_gate_ms",
            "arm_cmd_enable_gating_ms",
            "arm_cmd_takeover_settle_ms",
            "safety_ms",
            "arm_cmd_speed_feedback_ms",
            "gravity_ms",
            "arm_cmd_hold_update_ms",
            "operator_sync_ms",
            "provider_feedback_ms",
            "latency_trace_prepare_ms",
            "action_history_append_ms",
            "ctrl_dual_arm_call_ms",
            "enqueue_to_publish_ms",
        ]
        known_pre_publish = 0.0
        for key in known_pre_publish_keys:
            value = payload.get(key)
            if value is not None:
                known_pre_publish += float(value)
        payload["known_pre_publish_ms"] = known_pre_publish if payload["recv_to_pub_ms"] is not None else None
        if payload["recv_to_pub_ms"] is not None:
            payload["unaccounted_pre_publish_ms"] = max(
                0.0,
                float(payload["recv_to_pub_ms"]) - known_pre_publish,
            )
        else:
            payload["unaccounted_pre_publish_ms"] = None
        if payload["recv_to_exec_ms"] is not None:
            known_post_receive = known_pre_publish + float(payload.get("pub_to_exec_ms") or 0.0)
            payload["known_post_receive_ms"] = known_post_receive
            payload["unaccounted_post_receive_ms"] = max(
                0.0,
                float(payload["recv_to_exec_ms"]) - known_post_receive,
            )
            tele_fetch = float(payload.get("tele_fetch_ms") or 0.0)
            payload["fetch_to_exec_ms"] = tele_fetch + float(payload["recv_to_exec_ms"])
        else:
            payload["known_post_receive_ms"] = None
            payload["unaccounted_post_receive_ms"] = None
            payload["fetch_to_exec_ms"] = None
        return payload

    def _log_summary(self):
        try:
            completed = []
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if item.get("status") == "completed" and item.get("recv_to_exec_ms") is not None:
                        completed.append(item)
            if not completed:
                return

            def _p(arr_key):
                arr = np.array([float(item[arr_key]) for item in completed if item.get(arr_key) is not None], dtype=float)
                if arr.size == 0:
                    return None
                return arr

            recv_to_pub = _p("recv_to_pub_ms")
            pub_to_exec = _p("pub_to_exec_ms")
            pub_to_exec_thread = _p("pub_to_exec_thread_ms")
            recv_to_exec = _p("recv_to_exec_ms")
            recv_to_exec_thread = _p("recv_to_exec_thread_ms")
            tele_fetch = _p("tele_fetch_ms")
            ik = _p("ik_ms")
            takeover = _p("takeover_logic_ms")
            base = _p("base_control_ms")
            base_move = _p("base_move_ms")
            base_height = _p("base_height_ms")
            base_misc = _p("base_misc_ms")
            safety = _p("safety_ms")
            gravity = _p("gravity_ms")
            queue = _p("enqueue_to_publish_ms")
            wait = _p("controller_wait_ms")
            dds = _p("dds_write_ms")
            unknown_pre = _p("unaccounted_pre_publish_ms")
            logger_mp.info(
                "[LATENCY][SUMMARY] samples=%d recv->pub P50/P95=%.2f/%.2f ms | pub->exec(main) P50/P95=%.2f/%.2f ms | recv->exec(main) P50/P95=%.2f/%.2f ms",
                len(completed),
                np.percentile(recv_to_pub, 50), np.percentile(recv_to_pub, 95),
                np.percentile(pub_to_exec, 50), np.percentile(pub_to_exec, 95),
                np.percentile(recv_to_exec, 50), np.percentile(recv_to_exec, 95),
            )
            if pub_to_exec_thread is not None and recv_to_exec_thread is not None:
                logger_mp.info(
                    "[LATENCY][LOWSTATE_THREAD] pub->exec(thread) P50/P95=%.2f/%.2f ms | recv->exec(thread) P50/P95=%.2f/%.2f ms",
                    np.percentile(pub_to_exec_thread, 50), np.percentile(pub_to_exec_thread, 95),
                    np.percentile(recv_to_exec_thread, 50), np.percentile(recv_to_exec_thread, 95),
                )
            if any(arr is not None for arr in [tele_fetch, takeover, base, ik, safety, gravity, queue, wait, dds, unknown_pre]):
                logger_mp.info(
                    "[LATENCY][BREAKDOWN] tele=%.2f | takeover=%.2f | base=%.2f(move=%.2f,height=%.2f,misc=%.2f) | ik=%.2f | safety=%.2f | gravity=%.2f | queue=%.2f | wait=%.2f | dds=%.3f | unknown_pre=%.2f ms",
                    float(np.mean(tele_fetch)) if tele_fetch is not None else 0.0,
                    float(np.mean(takeover)) if takeover is not None else 0.0,
                    float(np.mean(base)) if base is not None else 0.0,
                    float(np.mean(base_move)) if base_move is not None else 0.0,
                    float(np.mean(base_height)) if base_height is not None else 0.0,
                    float(np.mean(base_misc)) if base_misc is not None else 0.0,
                    float(np.mean(ik)) if ik is not None else 0.0,
                    float(np.mean(safety)) if safety is not None else 0.0,
                    float(np.mean(gravity)) if gravity is not None else 0.0,
                    float(np.mean(queue)) if queue is not None else 0.0,
                    float(np.mean(wait)) if wait is not None else 0.0,
                    float(np.mean(dds)) if dds is not None else 0.0,
                    float(np.mean(unknown_pre)) if unknown_pre is not None else 0.0,
                )
        except Exception as e:
            logger_mp.warning(f"[LATENCY] failed to summarize latency trace file: {e}")

    @staticmethod
    def _delta_ms(start_ns: Optional[int], end_ns: Optional[int]):
        if start_ns is None or end_ns is None:
            return None
        return (int(end_ns) - int(start_ns)) / 1e6

    @classmethod
    def _add_online_inference_derived_fields(cls, payload: Dict[str, object]) -> None:
        obs_send_perf_ns = payload.get("online_obs_send_perf_ns")
        action_recv_perf_ns = payload.get("online_action_recv_perf_ns")
        step_output_perf_ns = payload.get("online_step_output_perf_ns")
        provider_return_perf_ns = payload.get("online_provider_output_perf_ns")
        if provider_return_perf_ns is None and step_output_perf_ns is not None:
            provider_return_perf_ns = payload.get("t_recv_ns")
            payload["online_provider_output_perf_ns"] = provider_return_perf_ns

        payload["online_obs_send_to_action_recv_ms"] = cls._delta_ms(obs_send_perf_ns, action_recv_perf_ns)
        payload["online_action_recv_to_step_output_ms"] = cls._delta_ms(action_recv_perf_ns, step_output_perf_ns)
        payload["online_step_output_to_provider_return_ms"] = cls._delta_ms(step_output_perf_ns, provider_return_perf_ns)
        payload["online_provider_return_to_pub_ms"] = cls._delta_ms(provider_return_perf_ns, payload.get("t_pub_ns"))

        payload["online_step_output_trace_ns"] = int(step_output_perf_ns) if step_output_perf_ns is not None else None
        payload["online_step_output_to_pub_ms"] = cls._delta_ms(step_output_perf_ns, payload.get("t_pub_ns"))
        payload["online_step_output_to_exec_ms"] = cls._delta_ms(step_output_perf_ns, payload.get("t_exec_ns"))
        payload["online_step_output_to_exec_thread_ms"] = cls._delta_ms(step_output_perf_ns, payload.get("t_exec_thread_ns"))
        payload["online_obs_send_to_publish_ms"] = cls._delta_ms(obs_send_perf_ns, payload.get("t_pub_ns"))
        payload["online_action_recv_to_publish_ms"] = cls._delta_ms(action_recv_perf_ns, payload.get("t_pub_ns"))
        payload["online_action_recv_to_exec_ms"] = cls._delta_ms(action_recv_perf_ns, payload.get("t_exec_ns"))
        payload["online_action_recv_to_exec_thread_ms"] = cls._delta_ms(action_recv_perf_ns, payload.get("t_exec_thread_ns"))
        payload["online_obs_send_to_exec_ms"] = cls._delta_ms(obs_send_perf_ns, payload.get("t_exec_ns"))
        payload["online_obs_send_to_exec_thread_ms"] = cls._delta_ms(obs_send_perf_ns, payload.get("t_exec_thread_ns"))
