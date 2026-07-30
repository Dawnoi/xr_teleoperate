from __future__ import annotations

from typing import Any

from teleop.debug.latency_trace import SimpleLatencyTracker


def setup_latency_tracker(args: Any, arm_ctrl: Any, log: Any) -> SimpleLatencyTracker | None:
    file_trace_enabled = bool(getattr(args, "latency_trace", False))
    memory_trace_enabled = bool(getattr(args, "ui", False))
    if not file_trace_enabled and not memory_trace_enabled:
        return None

    set_tracker = getattr(arm_ctrl, "set_latency_tracker", None)
    set_thresholds = getattr(arm_ctrl, "set_latency_exec_thresholds", None)
    if not callable(set_tracker) or not callable(set_thresholds):
        raise RuntimeError("arm controller does not provide required latency trace hooks")

    tracker = SimpleLatencyTracker(
        output_path=str(args.latency_trace_path) if file_trace_enabled else None,
        summary_every=int(args.latency_summary_every),
        log_each_trace=file_trace_enabled,
        timeout_s=float(args.latency_timeout),
        ui_memory_provider_scope=(
            {"xr", "online_inference", "lerobot_offline"}
            if memory_trace_enabled and not file_trace_enabled
            else None
        ),
    )
    set_tracker(tracker)
    set_thresholds(float(args.latency_exec_q_threshold), float(args.latency_exec_dq_threshold))
    log.info(
        "[LATENCY] tracing enabled: mode=%s command_threshold=%.4f rad, exec_q_threshold=%.4f rad, exec_dq_threshold=%.4f rad",
        "jsonl" if file_trace_enabled else "ui_memory",
        float(args.latency_command_threshold),
        float(args.latency_exec_q_threshold),
        float(args.latency_exec_dq_threshold),
    )
    return tracker
