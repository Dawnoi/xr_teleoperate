"""Pure host-monotonic timestamp validation for finalized episode items."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any


REQUIRED_CONFIG_KEYS = (
    "schema_version",
    "frame_interval_warning_scale",
    "alignment_p95_warning_scale",
    "alignment_p99_error_scale",
    "error_min_count",
    "error_min_fraction",
    "error_consecutive_count",
    "camera_reuse_warning_min_count",
    "camera_reuse_warning_min_fraction",
    "camera_reuse_error_min_count",
    "camera_reuse_error_min_fraction",
    "camera_reuse_error_consecutive_count",
    "action_support_warning_scale",
    "action_support_error_scale",
    "action_support_span_warning_scale",
    "action_support_span_error_scale",
)
STATUS_ORDER = {"ok": 0, "warning": 1, "error": 2}


def _issue(code: str, severity: str, message: str, **details: Any) -> dict[str, Any]:
    result = {"code": code, "severity": severity, "message": message}
    result.update(details)
    return result


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _longest_run(flags: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _status_from_issues(issues: Sequence[Mapping[str, Any]], default: str = "ok") -> str:
    status = default
    for issue in issues:
        severity = str(issue.get("severity", "error"))
        if STATUS_ORDER.get(severity, STATUS_ORDER["error"]) > STATUS_ORDER.get(status, 0):
            status = severity
    return status


def _validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("time alignment config must be a JSON object")
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"time alignment config is missing required keys: {', '.join(missing)}")
    if not isinstance(config["schema_version"], int) or isinstance(config["schema_version"], bool) or config["schema_version"] != 1:
        raise ValueError("time alignment config schema_version must be integer 1")
    positive_scales = (
        "frame_interval_warning_scale",
        "alignment_p99_error_scale",
        "action_support_warning_scale",
        "action_support_error_scale",
        "action_support_span_warning_scale",
        "action_support_span_error_scale",
    )
    non_negative_scales = ("alignment_p95_warning_scale",)
    for key in (*positive_scales, *non_negative_scales):
        value = config[key]
        minimum = 0.0 if key in non_negative_scales else 0.0
        if not _is_finite_number(value) or float(value) < minimum or (key in positive_scales and float(value) == minimum):
            raise ValueError(f"time alignment config {key} must be finite and positive/non-negative")
    for key in (
        "error_min_count",
        "error_consecutive_count",
        "camera_reuse_warning_min_count",
        "camera_reuse_error_min_count",
        "camera_reuse_error_consecutive_count",
    ):
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"time alignment config {key} must be an integer >= 1")
    for key in (
        "error_min_fraction",
        "camera_reuse_warning_min_fraction",
        "camera_reuse_error_min_fraction",
    ):
        fraction = config[key]
        if not _is_finite_number(fraction) or not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"time alignment config {key} must be in [0, 1]")
    return dict(config)


def _read_sample_times(
    items: Sequence[Any],
) -> tuple[list[float | None], list[dict[str, Any]], bool]:
    times: list[float | None] = []
    issues: list[dict[str, Any]] = []
    strictly_increasing = True
    previous: float | None = None
    for frame_index, item in enumerate(items):
        timestamps = item.get("timestamps") if isinstance(item, Mapping) else None
        value = timestamps.get("sample_monotonic_ns") if isinstance(timestamps, Mapping) else None
        path = "timestamps.sample_monotonic_ns"
        if value is None:
            issues.append(_issue("missing_sample_timestamp", "error", f"frame {frame_index} is missing {path}", frame_index=frame_index, path=path))
            times.append(None)
            strictly_increasing = False
            continue
        if not _is_finite_number(value) or float(value) < 0:
            issues.append(_issue("invalid_sample_timestamp", "error", f"frame {frame_index} has invalid {path}", frame_index=frame_index, path=path, value=value))
            times.append(None)
            strictly_increasing = False
            continue
        current = float(value)
        if previous is not None and current <= previous:
            code = "repeated_sample_timestamp" if current == previous else "backward_sample_timestamp"
            issues.append(_issue(code, "error", f"frame {frame_index} sample timestamps must be strictly increasing", frame_index=frame_index, path=path, previous=previous, current=current))
            strictly_increasing = False
        previous = current
        times.append(current)
    return times, issues, strictly_increasing


def _read_signal_times(
    items: Sequence[Any],
    source_name: str,
) -> tuple[list[float | None], list[dict[str, Any]], bool]:
    times: list[float | None] = []
    issues: list[dict[str, Any]] = []
    strictly_increasing = True
    previous: float | None = None
    path = f"timestamps.{source_name}.host_monotonic_ns"
    for frame_index, item in enumerate(items):
        timestamps = item.get("timestamps") if isinstance(item, Mapping) else None
        source = timestamps.get(source_name) if isinstance(timestamps, Mapping) else None
        value = source.get("host_monotonic_ns") if isinstance(source, Mapping) else None
        if value is None:
            issues.append(_issue(f"missing_{source_name}_timestamp", "error", f"frame {frame_index} is missing {path}", frame_index=frame_index, path=path))
            times.append(None)
            strictly_increasing = False
            continue
        if not _is_finite_number(value) or float(value) < 0:
            issues.append(_issue(f"invalid_{source_name}_timestamp", "error", f"frame {frame_index} has invalid {path}", frame_index=frame_index, path=path, value=value))
            times.append(None)
            strictly_increasing = False
            continue
        current = float(value)
        if previous is not None and current <= previous:
            code = f"repeated_{source_name}_timestamp" if current == previous else f"backward_{source_name}_timestamp"
            issues.append(_issue(code, "error", f"{path} must be strictly increasing", frame_index=frame_index, path=path, previous=previous, current=current))
            strictly_increasing = False
        previous = current
        times.append(current)
    return times, issues, strictly_increasing


def _read_camera(
    items: Sequence[Any],
    camera_name: str,
) -> tuple[list[float | None], list[dict[str, Any]], bool, list[bool]]:
    times: list[float | None] = []
    issues: list[dict[str, Any]] = []
    strictly_increasing = True
    previous: float | None = None
    previous_image_path: str | None = None
    previous_frame_seq: int | None = None
    reused_flags: list[bool] = []
    path = f"timestamps.camera.{camera_name}.host_recv_monotonic_ns"
    for frame_index, item in enumerate(items):
        colors = item.get("colors") if isinstance(item, Mapping) else None
        image_path = colors.get(camera_name) if isinstance(colors, Mapping) else None
        if not isinstance(image_path, str) or not image_path:
            issues.append(_issue("missing_camera_path", "error", f"frame {frame_index} is missing colors.{camera_name}", frame_index=frame_index, path=f"colors.{camera_name}"))

        timestamps = item.get("timestamps") if isinstance(item, Mapping) else None
        cameras = timestamps.get("camera") if isinstance(timestamps, Mapping) else None
        metadata = cameras.get(camera_name) if isinstance(cameras, Mapping) else None
        value = metadata.get("host_recv_monotonic_ns") if isinstance(metadata, Mapping) else None
        frame_seq = metadata.get("frame_seq") if isinstance(metadata, Mapping) else None
        if frame_seq is not None and (not isinstance(frame_seq, int) or isinstance(frame_seq, bool) or frame_seq < 0):
            issues.append(_issue("invalid_camera_frame_seq", "error", f"frame {frame_index} has invalid timestamps.camera.{camera_name}.frame_seq", frame_index=frame_index, path=f"timestamps.camera.{camera_name}.frame_seq", value=frame_seq))
            frame_seq = None
        if value is None:
            issues.append(_issue("missing_camera_timestamp", "error", f"frame {frame_index} is missing {path}", frame_index=frame_index, path=path))
            times.append(None)
            reused_flags.append(False)
            strictly_increasing = False
            continue
        if not _is_finite_number(value) or float(value) < 0:
            issues.append(_issue("invalid_camera_timestamp", "error", f"frame {frame_index} has invalid {path}", frame_index=frame_index, path=path, value=value))
            times.append(None)
            reused_flags.append(False)
            strictly_increasing = False
            continue
        current = float(value)
        reused = False
        if previous is not None and current < previous:
            issues.append(_issue("backward_camera_timestamp", "error", f"{path} must not move backward", frame_index=frame_index, path=path, previous=previous, current=current))
            strictly_increasing = False
        elif previous is not None and current == previous:
            same_frame_seq = frame_seq is not None and frame_seq == previous_frame_seq
            same_image_path = isinstance(image_path, str) and image_path and image_path == previous_image_path
            if same_frame_seq or same_image_path:
                reused = True
            else:
                issues.append(_issue("repeated_camera_timestamp", "error", f"{path} repeats for a different camera image", frame_index=frame_index, path=path, previous=previous, current=current))
                strictly_increasing = False
        previous = current
        previous_image_path = image_path if isinstance(image_path, str) and image_path else None
        previous_frame_seq = frame_seq
        times.append(current)
        reused_flags.append(reused)
    return times, issues, strictly_increasing, reused_flags


def _sample_interval_report(
    sample_times: Sequence[float | None],
    sample_times_valid: bool,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], float | None]:
    report: dict[str, Any] = {
        "status": "ok",
        "frame_count": len(sample_times),
        "interval_count": max(0, len(sample_times) - 1),
        "duration_ns": None,
        "duration_s": None,
        "T_ns": None,
        "T_s": None,
        "average_fps": None,
        "interval_median_ns": None,
        "interval_p95_ns": None,
        "interval_max_ns": None,
        "long_gap_threshold_ns": None,
        "long_gap_count": 0,
        "longest_consecutive_long_gap_run": 0,
        "offending_frame_indices": [],
        "issues": [],
    }
    if not sample_times_valid:
        report["status"] = "error"
        return report, [], None
    valid_times = [float(value) for value in sample_times if value is not None]
    if len(valid_times) < 2:
        return report, [], None
    intervals = [current - previous for previous, current in zip(valid_times, valid_times[1:])]
    mean_interval = sum(intervals) / len(intervals)
    warning_scale = float(config["frame_interval_warning_scale"])
    threshold = mean_interval * warning_scale
    long_flags = [interval > threshold for interval in intervals]
    offending = [frame_index for frame_index, flag in enumerate(long_flags, start=1) if flag]
    report.update(
        {
            "duration_ns": valid_times[-1] - valid_times[0],
            "duration_s": (valid_times[-1] - valid_times[0]) / 1e9,
            "T_ns": mean_interval,
            "T_s": mean_interval / 1e9,
            "average_fps": 1e9 / mean_interval,
            "interval_median_ns": _percentile(intervals, 0.5),
            "interval_p95_ns": _percentile(intervals, 0.95),
            "interval_max_ns": max(intervals),
            "long_gap_threshold_ns": threshold,
            "long_gap_count": len(offending),
            "longest_consecutive_long_gap_run": _longest_run(long_flags),
            "offending_frame_indices": offending,
        }
    )
    min_error_count = max(int(config["error_min_count"]), math.ceil(len(intervals) * float(config["error_min_fraction"])))
    severity = "ok"
    if offending:
        severity = "warning"
        report["issues"].append(_issue("long_sample_interval", "warning", "sample interval exceeds configured warning threshold", frame_indices=offending, threshold_ns=threshold))
    if len(offending) >= min_error_count or report["longest_consecutive_long_gap_run"] >= int(config["error_consecutive_count"]):
        severity = "error"
        report["issues"].append(_issue("persistent_long_sample_interval", "error", "sample intervals exceed the configured error policy", frame_indices=offending, threshold_ns=threshold))
    report["status"] = severity
    return report, list(report["issues"]), mean_interval


def _source_report(
    source_name: str,
    sample_times: Sequence[float | None],
    source_times: Sequence[float | None],
    source_issues: Sequence[Mapping[str, Any]],
    sample_interval_ns: float | None,
    config: Mapping[str, Any],
    camera_reuse_flags: Sequence[bool] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[float] = []
    for sample_time, source_time in zip(sample_times, source_times):
        if sample_time is None or source_time is None:
            continue
        errors.append(abs(sample_time - source_time))
    report: dict[str, Any] = {
        "status": _status_from_issues(source_issues),
        "sample_count": len(errors),
        "mean_error_ns": sum(errors) / len(errors) if errors else None,
        "p95_error_ns": _percentile(errors, 0.95),
        "p99_error_ns": _percentile(errors, 0.99),
        "max_error_ns": max(errors) if errors else None,
        "alignment_warning_threshold_ns": None,
        "alignment_error_threshold_ns": None,
        "above_warning_threshold_count": 0,
        "above_error_threshold_count": 0,
        "warning_frame_indices": [],
        "error_frame_indices": [],
        "longest_consecutive_error_run": 0,
        "offending_frame_indices": [],
        "camera_frame_reuse_count": None,
        "camera_frame_reuse_fraction": None,
        "longest_consecutive_camera_frame_reuse_run": None,
        "camera_frame_reuse_indices": None,
        "issues": list(source_issues),
    }
    severity = report["status"]
    if camera_reuse_flags is not None:
        if len(camera_reuse_flags) != len(sample_times):
            raise ValueError("camera reuse flags must have one entry per sample")
        reuse_indices = [frame_index for frame_index, reused in enumerate(camera_reuse_flags) if reused]
        reuse_count = len(reuse_indices)
        report.update(
            {
                "camera_frame_reuse_count": reuse_count,
                "camera_frame_reuse_fraction": reuse_count / len(sample_times) if sample_times else 0.0,
                "longest_consecutive_camera_frame_reuse_run": _longest_run(camera_reuse_flags),
                "camera_frame_reuse_indices": reuse_indices,
            }
        )
        warning_count = max(
            int(config["camera_reuse_warning_min_count"]),
            math.ceil(len(sample_times) * float(config["camera_reuse_warning_min_fraction"])),
        )
        error_count = max(
            int(config["camera_reuse_error_min_count"]),
            math.ceil(len(sample_times) * float(config["camera_reuse_error_min_fraction"])),
        )
        if reuse_count >= warning_count:
            if STATUS_ORDER.get(severity, 2) < STATUS_ORDER["error"]:
                severity = "warning"
            report["issues"].append(_issue("camera_frame_reuse", "warning", f"{source_name} reuses an already received camera image", source=source_name, frame_indices=reuse_indices, count=reuse_count, fraction=report["camera_frame_reuse_fraction"]))
        if reuse_count >= error_count or report["longest_consecutive_camera_frame_reuse_run"] >= int(config["camera_reuse_error_consecutive_count"]):
            severity = "error"
            report["issues"].append(_issue("persistent_camera_frame_reuse", "error", f"{source_name} camera image reuse exceeds the configured error policy", source=source_name, frame_indices=reuse_indices, count=reuse_count, fraction=report["camera_frame_reuse_fraction"]))
    if sample_interval_ns is None:
        report["status"] = "error" if source_issues else "warning"
        return report, list(report["issues"])
    warning_threshold = sample_interval_ns * float(config["alignment_p95_warning_scale"])
    error_threshold = sample_interval_ns * float(config["alignment_p99_error_scale"])
    warning_flags = [False] * len(sample_times)
    error_flags = [False] * len(sample_times)
    error_index = 0
    for frame_index, (sample_time, source_time) in enumerate(zip(sample_times, source_times)):
        if sample_time is None or source_time is None:
            continue
        error = errors[error_index]
        error_index += 1
        warning_flags[frame_index] = error > warning_threshold
        error_flags[frame_index] = error > error_threshold
    warning_indices = [index for index, flag in enumerate(warning_flags) if flag]
    error_indices = [index for index, flag in enumerate(error_flags) if flag]
    report.update(
        {
            "alignment_warning_threshold_ns": warning_threshold,
            "alignment_error_threshold_ns": error_threshold,
            "above_warning_threshold_count": len(warning_indices),
            "above_error_threshold_count": len(error_indices),
            "warning_frame_indices": warning_indices,
            "error_frame_indices": error_indices,
            "longest_consecutive_error_run": _longest_run(error_flags),
            "offending_frame_indices": warning_indices,
        }
    )
    if report["p95_error_ns"] is not None and report["p95_error_ns"] > warning_threshold:
        severity = "warning" if STATUS_ORDER.get(severity, 2) < STATUS_ORDER["error"] else severity
        report["issues"].append(_issue("alignment_p95_warning", "warning", f"{source_name} P95 alignment error exceeds the configured warning threshold", source=source_name, frame_indices=warning_indices, threshold_ns=warning_threshold))
    min_error_count = max(int(config["error_min_count"]), math.ceil(len(sample_times) * float(config["error_min_fraction"])))
    persistent_error = (
        report["p99_error_ns"] is not None
        and report["p99_error_ns"] > error_threshold
        and len(error_indices) >= min_error_count
    ) or report["longest_consecutive_error_run"] >= int(config["error_consecutive_count"])
    if persistent_error:
        severity = "error"
        report["issues"].append(_issue("persistent_alignment_error", "error", f"{source_name} exceeds the configured alignment error policy", source=source_name, frame_indices=error_indices, threshold_ns=error_threshold))
    report["status"] = severity
    return report, list(report["issues"])


def _action_support_report(
    items: Sequence[Mapping[str, Any]],
    sample_interval_ns: float | None,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    modes: list[str] = []
    support_nearest_abs_delta_ns: list[float] = []
    support_max_abs_delta_ns: list[float] = []
    support_span_ns: list[float] = []
    issues: list[dict[str, Any]] = []
    for frame_index, item in enumerate(items):
        timestamps = item.get("timestamps") if isinstance(item, Mapping) else None
        action = timestamps.get("action") if isinstance(timestamps, Mapping) else None
        if not isinstance(action, Mapping):
            issues.append(_issue("missing_action_support", "error", f"frame {frame_index} is missing timestamps.action support metadata", frame_index=frame_index))
            continue
        mode = action.get("interpolation_mode")
        if mode not in {"exact", "linear", "nearest_fallback"}:
            issues.append(_issue("invalid_action_interpolation_mode", "error", f"frame {frame_index} has invalid action interpolation_mode", frame_index=frame_index, value=mode))
            continue
        max_delta = action.get("support_max_abs_delta_ns")
        span = action.get("support_span_ns")
        if not _is_finite_number(max_delta) or float(max_delta) < 0:
            issues.append(_issue("invalid_action_support_max_delta", "error", f"frame {frame_index} has invalid timestamps.action.support_max_abs_delta_ns", frame_index=frame_index, value=max_delta))
            continue
        if not _is_finite_number(span) or float(span) < 0:
            issues.append(_issue("invalid_action_support_span", "error", f"frame {frame_index} has invalid timestamps.action.support_span_ns", frame_index=frame_index, value=span))
            continue
        if mode == "linear" and float(span) < float(max_delta):
            issues.append(_issue("invalid_action_support_geometry", "error", f"frame {frame_index} has support_span_ns smaller than support_max_abs_delta_ns", frame_index=frame_index, support_span_ns=span, support_max_abs_delta_ns=max_delta))
            continue
        modes.append(str(mode))
        support_nearest_abs_delta_ns.append(
            float(span) - float(max_delta) if mode == "linear" else float(max_delta)
        )
        support_max_abs_delta_ns.append(float(max_delta))
        support_span_ns.append(float(span))

    report: dict[str, Any] = {
        "status": _status_from_issues(issues),
        "sample_count": len(items),
        "valid_sample_count": len(modes),
        "linear_count": modes.count("linear"),
        "exact_count": modes.count("exact"),
        "nearest_fallback_count": modes.count("nearest_fallback"),
        "nearest_fallback_frame_indices": [index for index, mode in enumerate(modes) if mode == "nearest_fallback"],
        "support_nearest_abs_delta_mean_ns": sum(support_nearest_abs_delta_ns) / len(support_nearest_abs_delta_ns) if support_nearest_abs_delta_ns else None,
        "support_nearest_abs_delta_p95_ns": _percentile(support_nearest_abs_delta_ns, 0.95),
        "support_nearest_abs_delta_p99_ns": _percentile(support_nearest_abs_delta_ns, 0.99),
        "support_nearest_abs_delta_max_ns": max(support_nearest_abs_delta_ns) if support_nearest_abs_delta_ns else None,
        "support_max_abs_delta_mean_ns": sum(support_max_abs_delta_ns) / len(support_max_abs_delta_ns) if support_max_abs_delta_ns else None,
        "support_max_abs_delta_p95_ns": _percentile(support_max_abs_delta_ns, 0.95),
        "support_max_abs_delta_p99_ns": _percentile(support_max_abs_delta_ns, 0.99),
        "support_max_abs_delta_max_ns": max(support_max_abs_delta_ns) if support_max_abs_delta_ns else None,
        "support_span_mean_ns": sum(support_span_ns) / len(support_span_ns) if support_span_ns else None,
        "support_span_p95_ns": _percentile(support_span_ns, 0.95),
        "support_span_p99_ns": _percentile(support_span_ns, 0.99),
        "support_span_max_ns": max(support_span_ns) if support_span_ns else None,
        "warning_frame_indices": [],
        "error_frame_indices": [],
        "issues": issues,
    }
    if len(modes) != len(items):
        report["status"] = "error"
        return report, list(report["issues"])
    if sample_interval_ns is None:
        report["status"] = "warning"
        return report, list(report["issues"])

    delta_warning_threshold = sample_interval_ns * float(config["action_support_warning_scale"])
    delta_error_threshold = sample_interval_ns * float(config["action_support_error_scale"])
    span_warning_threshold = sample_interval_ns * float(config["action_support_span_warning_scale"])
    span_error_threshold = sample_interval_ns * float(config["action_support_span_error_scale"])
    warning_flags = [
        nearest_delta > delta_warning_threshold or span > span_warning_threshold
        for nearest_delta, span in zip(support_nearest_abs_delta_ns, support_span_ns)
    ]
    error_flags = [
        nearest_delta > delta_error_threshold or span > span_error_threshold
        for nearest_delta, span in zip(support_nearest_abs_delta_ns, support_span_ns)
    ]
    warning_indices = [index for index, flag in enumerate(warning_flags) if flag]
    error_indices = [index for index, flag in enumerate(error_flags) if flag]
    fallback_indices = report["nearest_fallback_frame_indices"]
    report.update(
        {
            "support_max_abs_delta_warning_threshold_ns": delta_warning_threshold,
            "support_max_abs_delta_error_threshold_ns": delta_error_threshold,
            "support_span_warning_threshold_ns": span_warning_threshold,
            "support_span_error_threshold_ns": span_error_threshold,
            "warning_frame_indices": warning_indices,
            "error_frame_indices": error_indices,
            "longest_consecutive_error_run": _longest_run(error_flags),
            "longest_consecutive_nearest_fallback_run": _longest_run([mode == "nearest_fallback" for mode in modes]),
        }
    )
    severity = report["status"]
    if warning_indices:
        severity = "warning" if STATUS_ORDER.get(severity, 2) < STATUS_ORDER["error"] else severity
        report["issues"].append(_issue("action_support_warning", "warning", "action interpolation support exceeds the configured warning policy", frame_indices=warning_indices))
    if fallback_indices:
        severity = "warning" if STATUS_ORDER.get(severity, 2) < STATUS_ORDER["error"] else severity
        report["issues"].append(_issue("action_nearest_fallback", "warning", "action used nearest-frame fallback instead of exact or linear interpolation", frame_indices=fallback_indices))
    min_error_count = max(int(config["error_min_count"]), math.ceil(len(items) * float(config["error_min_fraction"])))
    persistent_error = (
        (
            report["support_nearest_abs_delta_p99_ns"] is not None
            and report["support_nearest_abs_delta_p99_ns"] > delta_error_threshold
            and len(error_indices) >= min_error_count
        )
        or (
            report["support_span_p99_ns"] is not None
            and report["support_span_p99_ns"] > span_error_threshold
            and len(error_indices) >= min_error_count
        )
        or report["longest_consecutive_error_run"] >= int(config["error_consecutive_count"])
        or len(fallback_indices) >= min_error_count
        or report["longest_consecutive_nearest_fallback_run"] >= int(config["error_consecutive_count"])
    )
    if persistent_error:
        severity = "error"
        report["issues"].append(_issue("persistent_action_support_error", "error", "action interpolation support exceeds the configured error policy", frame_indices=sorted(set(error_indices + fallback_indices))))
    report["status"] = severity
    return report, list(report["issues"])


def validate_time_alignment(
    items: Sequence[Mapping[str, Any]],
    enabled_cameras: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure host-monotonic calculation with an explicit config snapshot.

    ``items`` are already parsed episode records. Configuration file loading
    and report persistence belong to the audit layer, not this module.
    """

    validated_config = _validate_config(config)
    config_snapshot = copy.deepcopy(validated_config)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "error",
        "timestamp_source": "host_monotonic_only",
        "sample_count": len(items) if isinstance(items, Sequence) and not isinstance(items, (str, bytes)) else 0,
        "enabled_cameras": [],
        "config": config_snapshot,
        "sample_interval": {},
        "action_support": {},
        "sources": {},
        "issues": [],
        "errors": [],
        "warnings": [],
    }
    issues: list[dict[str, Any]] = []
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        issues.append(_issue("invalid_items", "error", "items must be a sequence of parsed episode objects"))
        report["sample_interval"] = {"status": "error", "frame_count": 0, "issues": issues}
        report["issues"] = list(issues)
        report["errors"] = issues
        return report
    if not items:
        issues.append(_issue("empty_items", "error", "episode items must not be empty"))
    if not isinstance(enabled_cameras, Sequence) or isinstance(enabled_cameras, (str, bytes)):
        issues.append(_issue("invalid_enabled_cameras", "error", "enabled_cameras must be an ordered sequence of camera names"))
        cameras: list[str] = []
    else:
        cameras = list(enabled_cameras)
        if any(not isinstance(name, str) or not name for name in cameras):
            issues.append(_issue("invalid_enabled_camera_name", "error", "enabled_cameras must contain non-empty strings"))
            cameras = [name for name in cameras if isinstance(name, str) and name]
        if len(cameras) != len(set(cameras)):
            issues.append(_issue("duplicate_enabled_camera", "error", "enabled_cameras must not contain duplicates"))
            cameras = list(dict.fromkeys(cameras))
    report["enabled_cameras"] = cameras
    sample_times, sample_issues, sample_times_valid = _read_sample_times(items)
    issues.extend(sample_issues)
    interval, interval_issues, sample_interval_ns = _sample_interval_report(sample_times, sample_times_valid and bool(items), validated_config)
    interval["issues"] = [*sample_issues, *interval_issues]
    report["sample_interval"] = interval
    issues.extend(interval_issues)
    action_support, action_support_issues = _action_support_report(items, sample_interval_ns, validated_config)
    report["action_support"] = action_support
    issues.extend(action_support_issues)

    source_specs: list[tuple[str, list[float | None], list[dict[str, Any]], list[bool] | None]] = []
    for source_name in ("state", "action"):
        source_times, source_issues, _ = _read_signal_times(items, source_name)
        source_specs.append((source_name, source_times, source_issues, None))
    for camera_name in cameras:
        source_times, source_issues, _, camera_reuse_flags = _read_camera(items, camera_name)
        source_specs.append((f"camera.{camera_name}", source_times, source_issues, camera_reuse_flags))
    for source_name, source_times, source_issues, camera_reuse_flags in source_specs:
        source_report, source_report_issues = _source_report(source_name, sample_times, source_times, source_issues, sample_interval_ns, validated_config, camera_reuse_flags)
        report["sources"][source_name] = source_report
        issues.extend(source_report_issues)

    report["errors"] = [issue for issue in issues if issue["severity"] == "error"]
    report["warnings"] = [issue for issue in issues if issue["severity"] == "warning"]
    report["issues"] = list(issues)
    report["status"] = "error" if report["errors"] else "warning" if report["warnings"] else "ok"
    return report


__all__ = [
    "validate_time_alignment",
]
