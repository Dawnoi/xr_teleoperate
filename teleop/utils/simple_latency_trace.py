import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import logging_mp

logger_mp = logging_mp.getLogger(__name__)


@dataclass
class TraceRecord:
    seq: int
    t_recv_ns: int
    t_pub_ns: Optional[int] = None
    t_exec_ns: Optional[int] = None
    recv_q: Optional[np.ndarray] = None
    status: str = "pending"


class SimpleLatencyTracker:
    """
    Minimal single-flight latency tracker.

    One trace is tracked at a time so the reported path is easy to interpret:
        收到输入 -> DDS下发 -> 执行响应
    """

    def __init__(
        self,
        output_path: str,
        summary_every: int = 1,
        log_each_trace: bool = True,
        timeout_s: float = 2.0,
    ):
        self.output_path = output_path
        self.summary_every = max(1, int(summary_every))
        self.log_each_trace = bool(log_each_trace)
        self.timeout_ns = int(float(timeout_s) * 1e9)
        self._lock = threading.Lock()
        self._next_seq = 0
        self._active: Optional[TraceRecord] = None
        self._completed = 0
        self._dropped = 0

        out_dir = os.path.dirname(os.path.abspath(self.output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def can_start_new_trace(self) -> bool:
        with self._lock:
            return self._active is None

    def begin_trace(self, recv_ts_ns: int, recv_q) -> Optional[int]:
        recv_q = np.asarray(recv_q, dtype=float).copy()
        with self._lock:
            if self._active is not None:
                return None
            self._next_seq += 1
            self._active = TraceRecord(
                seq=self._next_seq,
                t_recv_ns=int(recv_ts_ns),
                recv_q=recv_q,
            )
            return self._active.seq

    def mark_publish(self, seq: Optional[int]) -> None:
        if not seq:
            return
        with self._lock:
            if self._active is None or self._active.seq != int(seq):
                return
            if self._active.t_pub_ns is None:
                self._active.t_pub_ns = time.perf_counter_ns()

    def maybe_mark_execute(self, current_q, current_dq, q_threshold: float, dq_threshold: float):
        current_q = np.asarray(current_q, dtype=float)
        current_dq = np.asarray(current_dq, dtype=float)
        record = None
        with self._lock:
            active = self._active
            if active is None or active.t_pub_ns is None or active.t_exec_ns is not None:
                return None
            q_delta = float(np.max(np.abs(current_q - active.recv_q))) if active.recv_q is not None else 0.0
            dq_peak = float(np.max(np.abs(current_dq))) if current_dq.size else 0.0
            if q_delta < float(q_threshold) and dq_peak < float(dq_threshold):
                return None
            active.t_exec_ns = time.perf_counter_ns()
            active.status = "completed"
            record = active
            self._active = None
            self._completed += 1
        if record is not None:
            self._finalize_record(record, q_delta=q_delta, dq_peak=dq_peak)
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
            self._finalize_record(record, q_delta=None, dq_peak=None)
        return record.seq if record is not None else None

    def _finalize_record(self, record: TraceRecord, q_delta, dq_peak):
        payload = {
            "seq": int(record.seq),
            "status": record.status,
            "t_recv_ns": int(record.t_recv_ns),
            "t_pub_ns": int(record.t_pub_ns) if record.t_pub_ns is not None else None,
            "t_exec_ns": int(record.t_exec_ns) if record.t_exec_ns is not None else None,
            "recv_to_pub_ms": self._delta_ms(record.t_recv_ns, record.t_pub_ns),
            "pub_to_exec_ms": self._delta_ms(record.t_pub_ns, record.t_exec_ns),
            "recv_to_exec_ms": self._delta_ms(record.t_recv_ns, record.t_exec_ns),
            "q_delta_trigger": q_delta,
            "dq_peak_trigger": dq_peak,
        }
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if self.log_each_trace:
            if record.status == "completed":
                logger_mp.info(
                    "[LATENCY] seq=%s 收到输入->DDS下发=%.2f ms | DDS下发->执行响应=%.2f ms | 收到输入->执行响应=%.2f ms",
                    payload["seq"],
                    payload["recv_to_pub_ms"] or -1.0,
                    payload["pub_to_exec_ms"] or -1.0,
                    payload["recv_to_exec_ms"] or -1.0,
                )
            else:
                logger_mp.warning(
                    "[LATENCY] seq=%s 超时未完成，状态=%s recv_to_pub=%s ms recv_to_exec=%s ms",
                    payload["seq"],
                    payload["status"],
                    "%.2f" % payload["recv_to_pub_ms"] if payload["recv_to_pub_ms"] is not None else "N/A",
                    "%.2f" % payload["recv_to_exec_ms"] if payload["recv_to_exec_ms"] is not None else "N/A",
                )

        total = self._completed + self._dropped
        if total % self.summary_every == 0:
            self._log_summary()

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
            recv_to_exec = np.array([item["recv_to_exec_ms"] for item in completed], dtype=float)
            recv_to_pub = np.array([item["recv_to_pub_ms"] for item in completed], dtype=float)
            pub_to_exec = np.array([item["pub_to_exec_ms"] for item in completed], dtype=float)
            logger_mp.info(
                "[LATENCY][SUMMARY] samples=%d recv->pub P50/P95=%.2f/%.2f ms | pub->exec P50/P95=%.2f/%.2f ms | recv->exec P50/P95=%.2f/%.2f ms",
                len(completed),
                np.percentile(recv_to_pub, 50),
                np.percentile(recv_to_pub, 95),
                np.percentile(pub_to_exec, 50),
                np.percentile(pub_to_exec, 95),
                np.percentile(recv_to_exec, 50),
                np.percentile(recv_to_exec, 95),
            )
        except Exception as e:
            logger_mp.warning(f"[LATENCY] failed to summarize latency trace file: {e}")

    @staticmethod
    def _delta_ms(start_ns: Optional[int], end_ns: Optional[int]):
        if start_ns is None or end_ns is None:
            return None
        return (int(end_ns) - int(start_ns)) / 1e6
