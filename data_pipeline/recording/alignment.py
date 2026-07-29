"""Timestamp alignment helpers for recording data streams."""

import copy
import math
from collections import deque
from collections.abc import Mapping
from numbers import Real

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


def hold_last_timed_sample(
    buffer: deque,
    target_ns: int,
    max_age_ns: int | None = None,
    min_timestamp_ns: int | None = None,
):
    target_ns = int(target_ns)
    for entry in reversed(buffer):
        t_ns = int(entry["t_ns"])
        if min_timestamp_ns is not None and t_ns < int(min_timestamp_ns):
            continue
        if t_ns > target_ns:
            continue
        age_ns = target_ns - t_ns
        if max_age_ns is not None and age_ns > int(max_age_ns):
            return None
        result = dict(entry)
        result["delta_to_target_ns"] = int(t_ns - target_ns)
        result["target_monotonic_ns"] = target_ns
        result["interpolation_mode"] = "hold_last"
        return result
    return None


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
        if isinstance(prev_val, Real) and isinstance(next_val, Real):
            result[key] = float(prev_val) + alpha * (float(next_val) - float(prev_val))
            continue
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
        if isinstance(prev_val, Real) and isinstance(next_val, Real):
            result[key] = float(prev_val) + alpha * (float(next_val) - float(prev_val))
            continue
        result[key] = prev_val if abs(target_ns - prev_t) <= abs(next_t - target_ns) else next_val
    return result


def _finite_float(value, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return result


def _mapping(value, field_name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _same_metadata(first: Mapping, second: Mapping, keys: tuple[str, ...], field_name: str) -> dict:
    result = {}
    for key in keys:
        first_value = first.get(key)
        second_value = second.get(key)
        if first_value != second_value:
            raise RuntimeError(
                f"{field_name}.{key} changes between interpolation supports: "
                f"{first_value!r} != {second_value!r}"
            )
        result[key] = copy.deepcopy(first_value)
    return result


def _normalized_quaternion(value, field_name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field_name} must be a length-4 quaternion")
    quaternion = [_finite_float(component, f"{field_name}[{index}]") for index, component in enumerate(value)]
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm < 1e-12:
        raise ValueError(f"{field_name} has zero norm")
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"{field_name} must have unit norm, got {norm:.9f}")
    return [component / norm for component in quaternion]


def _slerp_quaternion_xyzw(first: list[float], second: list[float], alpha: float) -> list[float]:
    dot = sum(left * right for left, right in zip(first, second))
    if dot < 0.0:
        second = [-component for component in second]
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = [left + alpha * (right - left) for left, right in zip(first, second)]
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        first_weight = math.sin((1.0 - alpha) * theta) / sin_theta
        second_weight = math.sin(alpha * theta) / sin_theta
        result = [first_weight * left + second_weight * right for left, right in zip(first, second)]
    norm = math.sqrt(sum(component * component for component in result))
    if norm < 1e-12:
        raise RuntimeError("SLERP produced a zero-norm quaternion")
    return [component / norm for component in result]


def _interpolate_scalar(first, second, alpha: float, field_name: str) -> float:
    first_value = _finite_float(first, f"{field_name}.prev")
    second_value = _finite_float(second, f"{field_name}.next")
    return first_value + alpha * (second_value - first_value)


def _interpolate_yaw_shortest(first, second, alpha: float, field_name: str) -> float:
    first_value = _finite_float(first, f"{field_name}.prev")
    second_value = _finite_float(second, f"{field_name}.next")
    delta = math.atan2(math.sin(second_value - first_value), math.cos(second_value - first_value))
    return math.atan2(math.sin(first_value + alpha * delta), math.cos(first_value + alpha * delta))


def _interpolate_pose(first: Mapping, second: Mapping, alpha: float, field_name: str) -> dict:
    metadata_keys = ("frame_id", "source_topic")
    if "child_frame_id" in first or "child_frame_id" in second:
        metadata_keys += ("child_frame_id",)
    result = {
        "x": _interpolate_scalar(first.get("x"), second.get("x"), alpha, f"{field_name}.x"),
        "y": _interpolate_scalar(first.get("y"), second.get("y"), alpha, f"{field_name}.y"),
        "z": _interpolate_scalar(first.get("z"), second.get("z"), alpha, f"{field_name}.z"),
        "yaw": _interpolate_yaw_shortest(first.get("yaw"), second.get("yaw"), alpha, f"{field_name}.yaw"),
        "quat_xyzw": _slerp_quaternion_xyzw(
            _normalized_quaternion(first.get("quat_xyzw"), f"{field_name}.prev.quat_xyzw"),
            _normalized_quaternion(second.get("quat_xyzw"), f"{field_name}.next.quat_xyzw"),
            alpha,
        ),
    }
    result.update(_same_metadata(first, second, metadata_keys, field_name))
    return result


def _support_entry(buffer: deque, timestamp_ns: int, min_timestamp_ns: int | None, field_name: str) -> Mapping:
    for entry in buffer:
        if min_timestamp_ns is not None and int(entry["t_ns"]) < int(min_timestamp_ns):
            continue
        if int(entry["t_ns"]) == int(timestamp_ns):
            return entry
    raise RuntimeError(f"{field_name} interpolation support timestamp {timestamp_ns} is absent from its history")


def _linear_supports(buffer: deque, aligned: Mapping, min_timestamp_ns: int | None, field_name: str) -> tuple[Mapping, Mapping, float] | None:
    if aligned.get("interpolation_mode") != "linear":
        return None
    previous = _support_entry(buffer, int(aligned["interp_prev_t_ns"]), min_timestamp_ns, field_name)
    following = _support_entry(buffer, int(aligned["interp_next_t_ns"]), min_timestamp_ns, field_name)
    return previous, following, float(aligned["interp_alpha"])


def interpolate_base_state_timed_sample_strict(
    buffer: deque,
    target_ns: int,
    max_delta_ns: int | None = None,
    min_timestamp_ns: int | None = None,
):
    """Interpolate measured odom pose and feedback velocity to one camera timestamp."""
    aligned = interpolate_timed_sample_strict(buffer, target_ns, max_delta_ns, min_timestamp_ns)
    if aligned is None:
        return None
    supports = _linear_supports(buffer, aligned, min_timestamp_ns, "base_state")
    if supports is None:
        return aligned
    previous, following, alpha = supports
    previous_pose = _mapping(previous.get("world_pose"), "base_state.prev.world_pose")
    following_pose = _mapping(following.get("world_pose"), "base_state.next.world_pose")
    previous_velocity = _mapping(previous.get("velocity"), "base_state.prev.velocity")
    following_velocity = _mapping(following.get("velocity"), "base_state.next.velocity")
    result = dict(aligned)
    result.pop("source_stamp_ns", None)
    result["world_pose"] = _interpolate_pose(previous_pose, following_pose, alpha, "base_state.world_pose")
    velocity = {
        key: _interpolate_scalar(previous_velocity.get(key), following_velocity.get(key), alpha, f"base_state.velocity.{key}")
        for key in ("vx", "vy", "vz", "wz")
    }
    velocity.update(
        _same_metadata(
            previous_velocity,
            following_velocity,
            ("frame_id", "linear_unit", "angular_unit", "source_topic"),
            "base_state.velocity",
        )
    )
    result["velocity"] = velocity
    result["source_topic"] = _same_metadata(previous, following, ("source_topic",), "base_state")["source_topic"]
    return result


def interpolate_base_height_timed_sample_strict(
    buffer: deque,
    target_ns: int,
    max_delta_ns: int | None = None,
    min_timestamp_ns: int | None = None,
):
    """Interpolate measured column height to one camera timestamp."""
    aligned = interpolate_timed_sample_strict(buffer, target_ns, max_delta_ns, min_timestamp_ns)
    if aligned is None:
        return None
    supports = _linear_supports(buffer, aligned, min_timestamp_ns, "base_height")
    if supports is None:
        return aligned
    previous, following, alpha = supports
    previous_height = _mapping(previous.get("height"), "base_height.prev.height")
    following_height = _mapping(following.get("height"), "base_height.next.height")
    result = dict(aligned)
    result.pop("source_stamp_ns", None)
    height = {
        "z": _interpolate_scalar(previous_height.get("z"), following_height.get("z"), alpha, "base_height.height.z"),
    }
    height.update(_same_metadata(previous_height, following_height, ("source_topic",), "base_height.height"))
    result["height"] = height
    result["source_topic"] = _same_metadata(previous, following, ("source_topic",), "base_height")["source_topic"]
    return result


def interpolate_slam_tf_timed_sample_strict(
    buffer: deque,
    target_ns: int,
    max_delta_ns: int | None = None,
    min_timestamp_ns: int | None = None,
):
    """Interpolate slamware_map->base_link pose while retaining both raw TF supports."""
    aligned = interpolate_timed_sample_strict(buffer, target_ns, max_delta_ns, min_timestamp_ns)
    if aligned is None:
        return None
    supports = _linear_supports(buffer, aligned, min_timestamp_ns, "slam_tf")
    if supports is None:
        return aligned
    previous, following, alpha = supports
    pose = _interpolate_pose(previous, following, alpha, "slam_tf.pose")
    pose.update(
        _same_metadata(
            previous,
            following,
            ("source_child_frame_id", "source_to_base_link_identity_assumed"),
            "slam_tf.pose",
        )
    )
    previous_tf_age_ms = _finite_float(previous.get("tf_age_ms"), "slam_tf.prev.tf_age_ms")
    following_tf_age_ms = _finite_float(following.get("tf_age_ms"), "slam_tf.next.tf_age_ms")
    pose["tf_age_ms"] = max(previous_tf_age_ms, following_tf_age_ms)
    pose["tf_age_ms_semantics"] = "max_support_age_ms"
    pose["interpolation_support"] = {
        "prev_tf_header_stamp_ns": int(previous["tf_header_stamp_ns"]),
        "next_tf_header_stamp_ns": int(following["tf_header_stamp_ns"]),
        "prev_tf_lookup_wall_time_ns": int(previous["tf_lookup_wall_time_ns"]),
        "next_tf_lookup_wall_time_ns": int(following["tf_lookup_wall_time_ns"]),
        "prev_tf_lookup_monotonic_ns": int(previous["tf_lookup_monotonic_ns"]),
        "next_tf_lookup_monotonic_ns": int(following["tf_lookup_monotonic_ns"]),
        "prev_source_chain": copy.deepcopy(previous.get("source_chain")),
        "next_source_chain": copy.deepcopy(following.get("source_chain")),
    }
    result = dict(aligned)
    result.pop("source_stamp_ns", None)
    for key in ("tf_header_stamp_ns", "tf_lookup_wall_time_ns", "tf_lookup_monotonic_ns", "source_chain"):
        result.pop(key, None)
    result.update(pose)
    return result
