"""Timestamp alignment helpers for recording data streams."""

from collections import deque

import numpy as np


def append_timed_sample(buffer: deque, timestamp_ns: int, **payload):
    entry = {"t_ns": int(timestamp_ns)}
    entry.update(payload)
    buffer.append(entry)


def nearest_timed_sample(buffer: deque, target_ns: int, max_delta_ns: int | None = None, min_timestamp_ns: int | None = None):
    best = None
    best_abs_delta = None
    target_ns = int(target_ns)
    for entry in reversed(buffer):
        t_ns = int(entry["t_ns"])
        if min_timestamp_ns is not None and t_ns < int(min_timestamp_ns):
            continue
        abs_delta = abs(t_ns - target_ns)
        if max_delta_ns is not None and abs_delta > int(max_delta_ns):
            continue
        if best_abs_delta is None or abs_delta < best_abs_delta:
            best = entry
            best_abs_delta = abs_delta
    if best is None:
        return None
    result = dict(best)
    result["delta_to_target_ns"] = int(best["t_ns"] - target_ns)
    result["target_monotonic_ns"] = target_ns
    return result


def interpolate_timed_sample(
    buffer: deque,
    target_ns: int,
    max_delta_ns: int | None = None,
    min_timestamp_ns: int | None = None,
):
    target_ns = int(target_ns)
    prev_entry = None
    next_entry = None
    for entry in reversed(buffer):
        t_ns = int(entry["t_ns"])
        if min_timestamp_ns is not None and t_ns < int(min_timestamp_ns):
            continue
        if t_ns <= target_ns:
            prev_entry = entry
            break
    for entry in buffer:
        t_ns = int(entry["t_ns"])
        if min_timestamp_ns is not None and t_ns < int(min_timestamp_ns):
            continue
        if t_ns >= target_ns:
            next_entry = entry
            break

    if prev_entry is None and next_entry is None:
        return None

    if prev_entry is not None and int(prev_entry["t_ns"]) == target_ns:
        result = dict(prev_entry)
        result["delta_to_target_ns"] = 0
        result["target_monotonic_ns"] = target_ns
        result["interpolation_mode"] = "exact"
        return result
    if next_entry is not None and int(next_entry["t_ns"]) == target_ns:
        result = dict(next_entry)
        result["delta_to_target_ns"] = 0
        result["target_monotonic_ns"] = target_ns
        result["interpolation_mode"] = "exact"
        return result

    if prev_entry is None or next_entry is None:
        fallback = nearest_timed_sample(
            buffer,
            target_ns,
            max_delta_ns=max_delta_ns,
            min_timestamp_ns=min_timestamp_ns,
        )
        if fallback is not None:
            fallback["interpolation_mode"] = "nearest_fallback"
        return fallback

    prev_t = int(prev_entry["t_ns"])
    next_t = int(next_entry["t_ns"])
    if next_t <= prev_t:
        fallback = nearest_timed_sample(
            buffer,
            target_ns,
            max_delta_ns=max_delta_ns,
            min_timestamp_ns=min_timestamp_ns,
        )
        if fallback is not None:
            fallback["interpolation_mode"] = "nearest_nonmonotonic"
        return fallback

    prev_delta = target_ns - prev_t
    next_delta = next_t - target_ns
    if max_delta_ns is not None and (
        prev_delta > int(max_delta_ns) or next_delta > int(max_delta_ns)
    ):
        return None

    alpha = float(target_ns - prev_t) / float(next_t - prev_t)
    result = {
        "t_ns": target_ns,
        "target_monotonic_ns": target_ns,
        "delta_to_target_ns": 0,
        "interpolation_mode": "linear",
        "interp_prev_t_ns": prev_t,
        "interp_next_t_ns": next_t,
        "interp_alpha": alpha,
    }
    all_keys = set(prev_entry.keys()) | set(next_entry.keys())
    for key in all_keys:
        if key == "t_ns":
            continue
        prev_val = prev_entry.get(key)
        next_val = next_entry.get(key)
        if prev_val is None and next_val is None:
            continue
        if prev_val is None:
            result[key] = next_val
            continue
        if next_val is None:
            result[key] = prev_val
            continue
        try:
            prev_arr = np.asarray(prev_val, dtype=float)
            next_arr = np.asarray(next_val, dtype=float)
            if prev_arr.shape == next_arr.shape and prev_arr.ndim >= 1:
                interp_arr = prev_arr + alpha * (next_arr - prev_arr)
                result[key] = interp_arr
                continue
        except Exception:
            pass
        result[key] = prev_val if abs(target_ns - prev_t) <= abs(next_t - target_ns) else next_val
    return result


def camera_meta_monotonic_ns(meta):
    if meta is None:
        return None
    value = meta.get("host_recv_monotonic_ns")
    if value is None:
        value = meta.get("host_monotonic_ns")
    return int(value) if value is not None else None


def camera_frame_identity(camera_name: str, meta):
    if meta is None:
        return None
    frame_seq = meta.get("frame_seq")
    if frame_seq is not None:
        return (str(camera_name), "seq", int(frame_seq))
    meta_ts = camera_meta_monotonic_ns(meta)
    if meta_ts is not None:
        return (str(camera_name), "ts", int(meta_ts))
    return None


def build_alignment_timestamp_entry(aligned_entry: dict, sample_monotonic_ns: int):
    sample_monotonic_ns = int(sample_monotonic_ns)
    t_ns = int(aligned_entry["t_ns"])
    delta_to_target_ns = int(aligned_entry.get("delta_to_target_ns", t_ns - sample_monotonic_ns))
    interpolation_mode = str(aligned_entry.get("interpolation_mode", "unknown"))
    entry = {
        "host_monotonic_ns": t_ns,
        "delta_to_sample_ns": delta_to_target_ns,
        "interpolation_mode": interpolation_mode,
    }

    if interpolation_mode == "linear":
        prev_t_ns = aligned_entry.get("interp_prev_t_ns")
        next_t_ns = aligned_entry.get("interp_next_t_ns")
        if prev_t_ns is not None:
            prev_t_ns = int(prev_t_ns)
        if next_t_ns is not None:
            next_t_ns = int(next_t_ns)
        prev_delta_ns = (prev_t_ns - sample_monotonic_ns) if prev_t_ns is not None else None
        next_delta_ns = (next_t_ns - sample_monotonic_ns) if next_t_ns is not None else None
        support_span_ns = (
            int(next_t_ns - prev_t_ns)
            if prev_t_ns is not None and next_t_ns is not None
            else None
        )
        support_max_abs_delta_ns = max(
            abs(prev_delta_ns) if prev_delta_ns is not None else 0,
            abs(next_delta_ns) if next_delta_ns is not None else 0,
        )
        entry.update(
            {
                "interp_prev_t_ns": prev_t_ns,
                "interp_next_t_ns": next_t_ns,
                "interp_alpha": float(aligned_entry.get("interp_alpha", 0.0)),
                "support_source_count": 2,
                "support_prev_t_ns": prev_t_ns,
                "support_next_t_ns": next_t_ns,
                "support_prev_delta_to_sample_ns": prev_delta_ns,
                "support_next_delta_to_sample_ns": next_delta_ns,
                "support_span_ns": support_span_ns,
                "support_max_abs_delta_ns": int(support_max_abs_delta_ns),
            }
        )
    else:
        source_t_ns = t_ns
        source_delta_ns = delta_to_target_ns
        entry.update(
            {
                "support_source_count": 1,
                "support_source_t_ns": source_t_ns,
                "support_prev_t_ns": source_t_ns,
                "support_next_t_ns": source_t_ns,
                "support_prev_delta_to_sample_ns": source_delta_ns,
                "support_next_delta_to_sample_ns": source_delta_ns,
                "support_span_ns": 0,
                "support_max_abs_delta_ns": abs(source_delta_ns),
            }
        )
    return entry


def timed_buffer_bounds(buffer: deque, min_timestamp_ns: int | None = None):
    earliest = None
    latest = None
    for entry in buffer:
        t_ns = int(entry["t_ns"])
        if min_timestamp_ns is not None and t_ns < int(min_timestamp_ns):
            continue
        if earliest is None:
            earliest = t_ns
        latest = t_ns
    return earliest, latest


def interpolate_timed_sample_strict(
    buffer: deque,
    target_ns: int,
    max_delta_ns: int | None = None,
    min_timestamp_ns: int | None = None,
):
    target_ns = int(target_ns)
    prev_entry = None
    next_entry = None
    for entry in reversed(buffer):
        t_ns = int(entry["t_ns"])
        if min_timestamp_ns is not None and t_ns < int(min_timestamp_ns):
            continue
        if t_ns <= target_ns:
            prev_entry = entry
            break
    for entry in buffer:
        t_ns = int(entry["t_ns"])
        if min_timestamp_ns is not None and t_ns < int(min_timestamp_ns):
            continue
        if t_ns >= target_ns:
            next_entry = entry
            break

    if prev_entry is None and next_entry is None:
        return None
    if prev_entry is not None and int(prev_entry["t_ns"]) == target_ns:
        result = dict(prev_entry)
        result["delta_to_target_ns"] = 0
        result["target_monotonic_ns"] = target_ns
        result["interpolation_mode"] = "exact"
        return result
    if next_entry is not None and int(next_entry["t_ns"]) == target_ns:
        result = dict(next_entry)
        result["delta_to_target_ns"] = 0
        result["target_monotonic_ns"] = target_ns
        result["interpolation_mode"] = "exact"
        return result
    if prev_entry is None or next_entry is None:
        return None
    prev_t = int(prev_entry["t_ns"])
    next_t = int(next_entry["t_ns"])
    if next_t <= prev_t:
        return None
    prev_delta = target_ns - prev_t
    next_delta = next_t - target_ns
    if max_delta_ns is not None and (
        prev_delta > int(max_delta_ns) or next_delta > int(max_delta_ns)
    ):
        return None
    alpha = float(target_ns - prev_t) / float(next_t - prev_t)
    result = {
        "t_ns": target_ns,
        "target_monotonic_ns": target_ns,
        "delta_to_target_ns": 0,
        "interpolation_mode": "linear",
        "interp_prev_t_ns": prev_t,
        "interp_next_t_ns": next_t,
        "interp_alpha": alpha,
    }
    all_keys = set(prev_entry.keys()) | set(next_entry.keys())
    for key in all_keys:
        if key == "t_ns":
            continue
        prev_val = prev_entry.get(key)
        next_val = next_entry.get(key)
        if prev_val is None and next_val is None:
            continue
        if prev_val is None:
            result[key] = next_val
            continue
        if next_val is None:
            result[key] = prev_val
            continue
        try:
            prev_arr = np.asarray(prev_val, dtype=float)
            next_arr = np.asarray(next_val, dtype=float)
            if prev_arr.shape == next_arr.shape and prev_arr.ndim >= 1:
                interp_arr = prev_arr + alpha * (next_arr - prev_arr)
                result[key] = interp_arr
                continue
        except Exception:
            pass
        result[key] = prev_val if abs(target_ns - prev_t) <= abs(next_t - target_ns) else next_val
    return result
