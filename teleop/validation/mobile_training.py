"""Validation for raw episodes collected with the mobile-training schema."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _issue(code: str, severity: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "details": details}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(_finite_number(item) for item in value)


def _finite_matrix(value: Any, rows: int, columns: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == rows
        and all(_finite_vector(row, columns) for row in value)
    )


def _section(item: Mapping[str, Any], top_level: str, name: str) -> dict[str, Any]:
    parent = item.get(top_level)
    if not isinstance(parent, Mapping):
        return {}
    section = parent.get(name)
    return dict(section) if isinstance(section, Mapping) else {}


def _mobile_schema_present(item: Mapping[str, Any]) -> bool:
    states = item.get("states")
    if not isinstance(states, Mapping):
        return False
    base = states.get("base")
    return isinstance(base, Mapping)


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("mobile training validation config must be an object")
    required_text = ("map_frame", "base_frame")
    for key in required_text:
        if not isinstance(config.get(key), str) or not str(config[key]).strip():
            raise ValueError(f"mobile training validation config.{key} must be a non-empty string")
    velocity_frames = config.get("velocity_frames")
    if (
        not isinstance(velocity_frames, list)
        or not velocity_frames
        or any(not isinstance(frame, str) or not frame for frame in velocity_frames)
    ):
        raise ValueError("mobile training validation config.velocity_frames must be a non-empty string list")
    limits = config.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("mobile training validation config.limits must be an object")
    timing = config.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError("mobile training validation config.timing must be an object")
    required_timing = (
        "base_state_period_ms",
        "base_height_period_ms",
        "slam_tf_period_ms",
        "base_action_period_ms",
        "max_alignment_periods",
        "slam_tf_max_alignment_periods",
        "slam_tf_max_age_periods",
        "slam_tf_max_future_periods",
        "source_gap_long_threshold_ms",
        "source_gap_error_consecutive_count",
        "source_gap_error_total_count",
    )
    validated_timing = {}
    for key in required_timing:
        value = timing.get(key)
        if not _finite_number(value):
            raise ValueError(f"mobile training validation config.timing.{key} must be finite")
        validated_timing[key] = float(value)
    for key in required_timing[:-1]:
        if validated_timing[key] <= 0.0:
            raise ValueError(f"mobile training validation config.timing.{key} must be positive")
    if validated_timing["slam_tf_max_future_periods"] < 0.0:
        raise ValueError("mobile training validation config.timing.slam_tf_max_future_periods must be non-negative")
    for key in ("source_gap_error_consecutive_count", "source_gap_error_total_count"):
        if validated_timing[key] <= 0.0 or not validated_timing[key].is_integer():
            raise ValueError(f"mobile training validation config.timing.{key} must be a positive integer")

    required_limits = (
        "map_speed_warning_mps",
        "map_yaw_rate_warning_radps",
        "column_minimum_m",
        "column_maximum_m",
        "waist_yaw_minimum_rad",
        "waist_yaw_maximum_rad",
    )
    validated_limits = {}
    for key in required_limits:
        value = limits.get(key)
        if not _finite_number(value):
            raise ValueError(f"mobile training validation config.limits.{key} must be finite")
        validated_limits[key] = float(value)
    if validated_limits["column_maximum_m"] <= validated_limits["column_minimum_m"]:
        raise ValueError("mobile training validation column limits must be ordered")
    if validated_limits["waist_yaw_maximum_rad"] <= validated_limits["waist_yaw_minimum_rad"]:
        raise ValueError("mobile training validation waist yaw limits must be ordered")
    max_alignment_ms = validated_timing["max_alignment_periods"]
    validated_limits.update(
        {
            "base_state_max_delta_ms": validated_timing["base_state_period_ms"] * max_alignment_ms,
            "base_height_max_delta_ms": validated_timing["base_height_period_ms"] * max_alignment_ms,
            "base_action_max_support_delta_ms": validated_timing["base_action_period_ms"] * max_alignment_ms,
            "slam_tf_max_delta_ms": (
                validated_timing["slam_tf_period_ms"] * validated_timing["slam_tf_max_alignment_periods"]
            ),
            "slam_tf_max_age_ms": (
                validated_timing["slam_tf_period_ms"] * validated_timing["slam_tf_max_age_periods"]
            ),
            "slam_tf_max_future_ms": (
                validated_timing["slam_tf_period_ms"] * validated_timing["slam_tf_max_future_periods"]
            ),
            "source_gap_long_threshold_ms": validated_timing["source_gap_long_threshold_ms"],
            "source_gap_error_consecutive_count": int(validated_timing["source_gap_error_consecutive_count"]),
            "source_gap_error_total_count": int(validated_timing["source_gap_error_total_count"]),
        }
    )
    return {
        "map_frame": str(config["map_frame"]),
        "base_frame": str(config["base_frame"]),
        "velocity_frames": list(velocity_frames),
        "allow_identity_source_to_base_link": bool(config.get("allow_identity_source_to_base_link", False)),
        "timing": validated_timing,
        "limits": validated_limits,
    }


def _validate_pose(
    pose: dict[str, Any],
    *,
    frame_id: str,
    child_frame_id: str,
    frame_index: int,
    field: str,
    issues: list[dict[str, Any]],
) -> None:
    if not pose:
        issues.append(_issue("MOBILE_MISSING_POSE", "error", f"frame {frame_index} missing {field}", frame_index=frame_index))
        return
    if pose.get("frame_id") != frame_id or pose.get("child_frame_id") != child_frame_id:
        issues.append(
            _issue(
                "MOBILE_POSE_FRAME_MISMATCH",
                "error",
                f"frame {frame_index} {field} must be {frame_id}->{child_frame_id}",
                frame_index=frame_index,
                observed_frame_id=pose.get("frame_id"),
                observed_child_frame_id=pose.get("child_frame_id"),
            )
        )
    if not _finite_vector(pose.get("position"), 3):
        issues.append(_issue("MOBILE_BAD_POSE_POSITION", "error", f"frame {frame_index} {field}.position must be finite xyz", frame_index=frame_index))
    if not _finite_vector(pose.get("rpy"), 3):
        issues.append(_issue("MOBILE_BAD_POSE_RPY", "error", f"frame {frame_index} {field}.rpy must be finite roll/pitch/yaw", frame_index=frame_index))
    if not _finite_matrix(pose.get("rotation_matrix"), 3, 3):
        issues.append(_issue("MOBILE_BAD_POSE_ROTATION", "error", f"frame {frame_index} {field}.rotation_matrix must be finite 3x3", frame_index=frame_index))
    if not _finite_matrix(pose.get("matrix4x4"), 4, 4):
        issues.append(_issue("MOBILE_BAD_POSE_MATRIX", "error", f"frame {frame_index} {field}.matrix4x4 must be finite 4x4", frame_index=frame_index))


def _validate_alignment(
    timestamps: dict[str, Any],
    *,
    key: str,
    max_abs_delta_ms: float,
    require_past: bool,
    frame_index: int,
    issues: list[dict[str, Any]],
) -> float | None:
    entry = timestamps.get(key)
    if not isinstance(entry, Mapping) or not _finite_number(entry.get("delta_to_sample_ns")):
        issues.append(_issue("MOBILE_MISSING_ALIGNMENT", "error", f"frame {frame_index} missing {key} alignment", frame_index=frame_index))
        return None
    delta_ms = float(entry["delta_to_sample_ns"]) / 1e6
    interpolation_mode = str(entry.get("interpolation_mode", ""))
    if interpolation_mode not in {_EDGE_FUTURE_MODE, _EDGE_HOLD_MODE} and abs(delta_ms) > max_abs_delta_ms:
        issues.append(_issue("MOBILE_ALIGNMENT_EXCEEDED", "error", f"frame {frame_index} {key} delta {delta_ms:.1f}ms exceeds {max_abs_delta_ms:.1f}ms", frame_index=frame_index, delta_ms=delta_ms, limit_ms=max_abs_delta_ms))
    if require_past and delta_ms > 0.0:
        issues.append(_issue("MOBILE_ACTION_ALIGNMENT_FUTURE", "error", f"frame {frame_index} base action must be hold-last, got future delta {delta_ms:.1f}ms", frame_index=frame_index, delta_ms=delta_ms))
    return delta_ms


def _validate_base_action_alignment(
    timestamps: dict[str, Any],
    *,
    max_support_delta_ms: float,
    frame_index: int,
    issues: list[dict[str, Any]],
) -> float | None:
    entry = timestamps.get("base_action")
    if not isinstance(entry, Mapping):
        issues.append(_issue("MOBILE_MISSING_ALIGNMENT", "error", f"frame {frame_index} missing base_action alignment", frame_index=frame_index))
        return None
    interpolation_mode = str(entry.get("interpolation_mode", ""))
    if interpolation_mode not in {"exact", "linear", "nearest_fallback", "hold_last", _EDGE_FUTURE_MODE, _EDGE_HOLD_MODE}:
        issues.append(
            _issue(
                "MOBILE_BASE_ACTION_ALIGNMENT_MODE",
                "error",
                f"frame {frame_index} base action must use the arm-action alignment policy, got {interpolation_mode!r}",
                frame_index=frame_index,
                interpolation_mode=interpolation_mode,
            )
        )
    if interpolation_mode in {"hold_last", _EDGE_FUTURE_MODE, _EDGE_HOLD_MODE}:
        sample_time_ns = timestamps.get("sample_monotonic_ns")
        source_time_ns = entry.get("support_source_t_ns")
        if not _finite_number(sample_time_ns):
            issues.append(
                _issue(
                    "MOBILE_BASE_ACTION_HOLD_LAST_TIMESTAMP",
                    "error",
                    f"frame {frame_index} hold-last base action requires finite sample_monotonic_ns",
                    frame_index=frame_index,
                )
            )
        elif not _finite_number(source_time_ns):
            issues.append(
                _issue(
                    "MOBILE_BASE_ACTION_HOLD_LAST_TIMESTAMP",
                    "error",
                    f"frame {frame_index} hold-last base action requires finite support_source_t_ns",
                    frame_index=frame_index,
                )
            )
        elif interpolation_mode == "hold_last" and float(source_time_ns) > float(sample_time_ns):
            issues.append(
                _issue(
                    "MOBILE_BASE_ACTION_HOLD_LAST_FUTURE",
                    "error",
                    f"frame {frame_index} hold-last base action source must not be later than the sample",
                    frame_index=frame_index,
                    source_time_ns=int(source_time_ns),
                    sample_time_ns=int(sample_time_ns),
                )
            )
        elif interpolation_mode == _EDGE_FUTURE_MODE and float(source_time_ns) < float(sample_time_ns):
            issues.append(
                _issue(
                    "MOBILE_BASE_ACTION_EDGE_FUTURE_PAST",
                    "error",
                    f"frame {frame_index} start-edge base action source must not precede the sample",
                    frame_index=frame_index,
                    source_time_ns=int(source_time_ns),
                    sample_time_ns=int(sample_time_ns),
                )
            )
    support_delta_ns = entry.get("support_max_abs_delta_ns")
    if not _finite_number(support_delta_ns):
        issues.append(_issue("MOBILE_MISSING_ALIGNMENT", "error", f"frame {frame_index} missing base_action support delta", frame_index=frame_index))
        return None
    support_delta_ms = abs(float(support_delta_ns)) / 1e6
    if support_delta_ms > max_support_delta_ms:
        issues.append(
            _issue(
                "MOBILE_ALIGNMENT_EXCEEDED",
                "error",
                f"frame {frame_index} base_action support delta {support_delta_ms:.1f}ms exceeds {max_support_delta_ms:.1f}ms",
                frame_index=frame_index,
                delta_ms=support_delta_ms,
                limit_ms=max_support_delta_ms,
            )
        )
    return support_delta_ms


_EDGE_FUTURE_MODE = "edge_nearest_future"
_EDGE_HOLD_MODE = "edge_hold_last"
_INTERNAL_ALIGNMENT_MODES = {"", "exact", "linear", "nearest_fallback", "hold_last"}


def _source_gap_report(
    items: Sequence[Mapping[str, Any]],
    *,
    key: str,
    long_threshold_ms: float,
    error_consecutive_count: int,
    error_total_count: int,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize unique two-sided source intervals, never image-frame multiplicity."""
    support_pairs: dict[tuple[int, int], list[int]] = {}
    modes: list[tuple[int, str]] = []
    for frame_index, item in enumerate(items):
        timestamps = item.get("timestamps") if isinstance(item, Mapping) else None
        entry = timestamps.get(key) if isinstance(timestamps, Mapping) else None
        if not isinstance(entry, Mapping):
            continue
        mode = str(entry.get("interpolation_mode", ""))
        modes.append((frame_index, mode))
        if mode != "linear":
            continue
        previous = entry.get("support_prev_t_ns")
        following = entry.get("support_next_t_ns")
        if not _finite_number(previous) or not _finite_number(following):
            issues.append(_issue("MOBILE_SOURCE_GAP_METADATA", "error", f"frame {frame_index} {key} linear alignment is missing support timestamps", frame_index=frame_index, key=key))
            continue
        previous_ns = int(previous)
        following_ns = int(following)
        if following_ns <= previous_ns:
            issues.append(_issue("MOBILE_SOURCE_GAP_METADATA", "error", f"frame {frame_index} {key} has invalid support interval", frame_index=frame_index, key=key, support_prev_t_ns=previous_ns, support_next_t_ns=following_ns))
            continue
        support_pairs.setdefault((previous_ns, following_ns), []).append(frame_index)

    normal_seen = False
    hold_seen = False
    edge_future_frames: list[int] = []
    edge_hold_frames: list[int] = []
    for frame_index, mode in modes:
        if mode == _EDGE_FUTURE_MODE:
            edge_future_frames.append(frame_index)
            if normal_seen or hold_seen:
                issues.append(_issue("MOBILE_EDGE_ALIGNMENT_INTERIOR", "error", f"frame {frame_index} {key} uses start-edge support after an internal sample", frame_index=frame_index, key=key, interpolation_mode=mode))
            continue
        if mode == _EDGE_HOLD_MODE:
            edge_hold_frames.append(frame_index)
            hold_seen = True
            continue
        if mode not in _INTERNAL_ALIGNMENT_MODES:
            issues.append(_issue("MOBILE_ALIGNMENT_MODE_INVALID", "error", f"frame {frame_index} {key} has invalid interpolation_mode {mode!r}", frame_index=frame_index, key=key, interpolation_mode=mode))
            continue
        if hold_seen:
            issues.append(_issue("MOBILE_EDGE_ALIGNMENT_INTERIOR", "error", f"frame {frame_index} {key} has an internal sample after end-edge support", frame_index=frame_index, key=key, interpolation_mode=mode))
        normal_seen = True

    intervals = []
    previous_interval_end_ns: int | None = None
    consecutive_run = 0
    longest_consecutive_run = 0
    long_intervals = []
    threshold_ns = int(long_threshold_ms * 1e6)
    for (previous_ns, following_ns), frame_indices in sorted(support_pairs.items()):
        gap_ns = following_ns - previous_ns
        is_long = gap_ns > threshold_ns
        interval = {
            "support_prev_t_ns": previous_ns,
            "support_next_t_ns": following_ns,
            "gap_ns": gap_ns,
            "frame_indices": frame_indices,
            "is_long": is_long,
        }
        intervals.append(interval)
        if is_long:
            long_intervals.append(interval)
            consecutive_run = consecutive_run + 1 if previous_interval_end_ns == previous_ns else 1
            longest_consecutive_run = max(longest_consecutive_run, consecutive_run)
        else:
            consecutive_run = 0
        previous_interval_end_ns = following_ns

    report = {
        "long_gap_threshold_ms": float(long_threshold_ms),
        "unique_interpolated_source_interval_count": len(intervals),
        "long_gap_count": len(long_intervals),
        "long_gap_intervals": long_intervals,
        "longest_consecutive_long_gap_run": longest_consecutive_run,
        "edge_nearest_future_frame_indices": edge_future_frames,
        "edge_hold_last_frame_indices": edge_hold_frames,
    }
    if long_intervals:
        issues.append(_issue("MOBILE_SOURCE_LONG_GAP", "warning", f"{key} has {len(long_intervals)} unique source gap(s) above {long_threshold_ms:.1f}ms", key=key, long_gap_count=len(long_intervals), long_gap_intervals=long_intervals))
    if (
        len(long_intervals) >= int(error_total_count)
        or longest_consecutive_run >= int(error_consecutive_count)
    ):
        issues.append(_issue("MOBILE_SOURCE_LONG_GAP_PERSISTENT", "error", f"{key} source gaps exceed the episode policy", key=key, long_gap_count=len(long_intervals), longest_consecutive_long_gap_run=longest_consecutive_run, error_total_count=int(error_total_count), error_consecutive_count=int(error_consecutive_count)))
    return report


def _validate_tcp_metadata(info: Any, issues: list[dict[str, Any]]) -> None:
    if not isinstance(info, Mapping):
        issues.append(_issue("MOBILE_MISSING_TCP_METADATA", "error", "episode info is required for mobile_tcp_v1"))
        return
    expected_scalars = {
        "eef_pose_frame": "dex1_tcp",
        "eef_pose_parent_frame": "base_link",
        "eef_pose_unit": "m/rad",
    }
    for key, expected in expected_scalars.items():
        if info.get(key) != expected:
            issues.append(_issue("MOBILE_TCP_METADATA_MISMATCH", "error", f"episode info.{key} must be {expected!r}", observed=info.get(key)))
    for key in ("robot_fk_urdf", "eef_model_urdf"):
        value = info.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(_issue("MOBILE_TCP_METADATA_MISSING_PATH", "error", f"episode info.{key} must be a non-empty model path"))
    expected_xyz = [0.1201, 0.0, 0.0]
    expected_rpy = [0.0, 0.0, 0.0]
    for key, expected in (
        ("left_wrist_to_tcp_xyz_m", expected_xyz),
        ("left_wrist_to_tcp_rpy_rad", expected_rpy),
        ("right_wrist_to_tcp_xyz_m", expected_xyz),
        ("right_wrist_to_tcp_rpy_rad", expected_rpy),
    ):
        value = info.get(key)
        if not _finite_vector(value, 3):
            issues.append(_issue("MOBILE_TCP_METADATA_BAD_EXTRINSIC", "error", f"episode info.{key} must be a finite length-3 vector"))
        elif any(abs(float(actual) - expected_value) > 1e-12 for actual, expected_value in zip(value, expected)):
            issues.append(_issue("MOBILE_TCP_METADATA_MISMATCH", "error", f"episode info.{key} must equal {expected}", observed=value))


def _validate_base_observation_mode(episode_info: Any, issues: list[dict[str, Any]]) -> bool:
    """Return whether this episode intentionally excludes all global base poses."""
    if not isinstance(episode_info, Mapping):
        return False
    mode = episode_info.get("mobile_base_observation_mode")
    if mode is None:
        return False
    if mode == "global_slam_map_and_velocity":
        return False
    if mode != "velocity_only_base_link":
        issues.append(
            _issue(
                "MOBILE_BASE_OBSERVATION_MODE_INVALID",
                "error",
                "episode info.mobile_base_observation_mode is invalid",
                observed=mode,
            )
        )
        return False
    if episode_info.get("mobile_base_absolute_pose_recorded") is not False:
        issues.append(
            _issue(
                "MOBILE_BASE_OBSERVATION_MODE_INVALID",
                "error",
                "velocity_only_base_link must declare mobile_base_absolute_pose_recorded=false",
            )
        )
    if episode_info.get("mobile_base_velocity_frame") != "base_link":
        issues.append(
            _issue(
                "MOBILE_BASE_OBSERVATION_MODE_INVALID",
                "error",
                "velocity_only_base_link must declare mobile_base_velocity_frame=base_link",
            )
        )
    return True


def _validate_global_slam_pose(
    *,
    states_base: dict[str, Any],
    timestamps: dict[str, Any],
    validated: dict[str, Any],
    limits: dict[str, Any],
    frame_index: int,
    issues: list[dict[str, Any]],
    previous_map_pose: tuple[float, float, float, int] | None,
) -> tuple[tuple[float, float, float, int] | None, int, float, float, float]:
    slam_map_pose = states_base.get("slam_map_pose")
    if not isinstance(slam_map_pose, Mapping):
        issues.append(_issue("MOBILE_MISSING_SLAM_POSE", "error", f"frame {frame_index} missing states.base.slam_map_pose", frame_index=frame_index))
        slam_map_pose = {}
    if slam_map_pose.get("frame_id") != validated["map_frame"] or slam_map_pose.get("child_frame_id") != validated["base_frame"]:
        issues.append(_issue("MOBILE_SLAM_FRAME_MISMATCH", "error", f"frame {frame_index} slam pose must be {validated['map_frame']}->{validated['base_frame']}", frame_index=frame_index))

    identity_assumption_count = 0
    max_slam_tf_age_ms = 0.0
    max_map_speed_mps = 0.0
    max_map_yaw_rate_radps = 0.0
    if not _finite_vector(slam_map_pose.get("quat_xyzw"), 4) or not all(_finite_number(slam_map_pose.get(key)) for key in ("x", "y", "z", "yaw")):
        issues.append(_issue("MOBILE_BAD_SLAM_POSE", "error", f"frame {frame_index} slam pose must contain finite xyz/yaw/quaternion", frame_index=frame_index))
    else:
        quaternion_norm = math.sqrt(sum(float(value) ** 2 for value in slam_map_pose["quat_xyzw"]))
        if abs(quaternion_norm - 1.0) > 0.01:
            issues.append(_issue("MOBILE_BAD_SLAM_QUATERNION", "error", f"frame {frame_index} slam quaternion norm is {quaternion_norm:.4f}", frame_index=frame_index))
        sample_time = timestamps.get("sample_monotonic_ns")
        if _finite_number(sample_time):
            current_map_pose = (float(slam_map_pose["x"]), float(slam_map_pose["y"]), float(slam_map_pose["yaw"]), int(sample_time))
            if previous_map_pose is not None and current_map_pose[3] > previous_map_pose[3]:
                dt_s = (current_map_pose[3] - previous_map_pose[3]) / 1e9
                max_map_speed_mps = math.hypot(current_map_pose[0] - previous_map_pose[0], current_map_pose[1] - previous_map_pose[1]) / dt_s
                yaw_delta = math.atan2(math.sin(current_map_pose[2] - previous_map_pose[2]), math.cos(current_map_pose[2] - previous_map_pose[2]))
                max_map_yaw_rate_radps = abs(yaw_delta) / dt_s
            previous_map_pose = current_map_pose
    if not isinstance(slam_map_pose.get("source_child_frame_id"), str) or not slam_map_pose.get("source_child_frame_id"):
        issues.append(_issue("MOBILE_MISSING_SLAM_SOURCE_FRAME", "error", f"frame {frame_index} missing slam source_child_frame_id", frame_index=frame_index))
    if not isinstance(slam_map_pose.get("source_to_base_link_identity_assumed"), bool):
        issues.append(_issue("MOBILE_BAD_SLAM_EXTRINSIC_FLAG", "error", f"frame {frame_index} slam identity-extrinsic flag must be boolean", frame_index=frame_index))
    if slam_map_pose.get("source_to_base_link_identity_assumed") is True:
        identity_assumption_count = 1
    tf_age_ms = slam_map_pose.get("tf_age_ms")
    if not _finite_number(tf_age_ms):
        issues.append(_issue("MOBILE_MISSING_SLAM_TF_AGE", "error", f"frame {frame_index} missing finite slam TF age", frame_index=frame_index))
    else:
        max_slam_tf_age_ms = float(tf_age_ms)
        if max_slam_tf_age_ms > limits["slam_tf_max_age_ms"]:
            issues.append(_issue("MOBILE_SLAM_TF_STALE", "error", f"frame {frame_index} slam TF age {max_slam_tf_age_ms:.1f}ms exceeds {limits['slam_tf_max_age_ms']:.1f}ms", frame_index=frame_index, tf_age_ms=max_slam_tf_age_ms, limit_ms=limits["slam_tf_max_age_ms"]))
        if max_slam_tf_age_ms < -limits["slam_tf_max_future_ms"]:
            issues.append(_issue("MOBILE_SLAM_TF_FUTURE", "error", f"frame {frame_index} slam TF is {-max_slam_tf_age_ms:.1f}ms in the future; verify ROS host clock synchronization", frame_index=frame_index, tf_age_ms=max_slam_tf_age_ms, limit_ms=limits["slam_tf_max_future_ms"]))
    return previous_map_pose, identity_assumption_count, max_slam_tf_age_ms, max_map_speed_mps, max_map_yaw_rate_radps


def validate_mobile_training(
    items: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    episode_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit fields required to train local EEF plus global mobile-base policies."""
    validated = _validate_config(config)
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("mobile training validation items must be a sequence")
    present_count = sum(_mobile_schema_present(item) for item in items if isinstance(item, Mapping))
    if present_count == 0:
        return {
            "status": "not_applicable",
            "frame_count": len(items),
            "present_frame_count": 0,
            "issues": [],
            "errors": [],
            "warnings": [],
        }

    issues: list[dict[str, Any]] = []
    _validate_tcp_metadata(episode_info, issues)
    velocity_only_base = _validate_base_observation_mode(episode_info, issues)
    limits = validated["limits"]
    max_deltas_ms = {"base_state": 0.0, "base_height": 0.0, "base_action": 0.0, "slam_tf": 0.0}
    max_slam_tf_age_ms = 0.0
    identity_assumption_count = 0
    previous_map_pose: tuple[float, float, float, int] | None = None
    max_map_speed_mps = 0.0
    max_map_yaw_rate_radps = 0.0

    for frame_index, item in enumerate(items):
        if not isinstance(item, Mapping):
            issues.append(_issue("MOBILE_BAD_ITEM", "error", f"frame {frame_index} is not an object", frame_index=frame_index))
            continue
        states_base = _section(item, "states", "base")
        actions_base = _section(item, "actions", "base")
        timestamps = item.get("timestamps")
        timestamps = dict(timestamps) if isinstance(timestamps, Mapping) else {}

        pose_specs = (
            ("states", "left_arm", "left_ee", "states.left_arm.pose_base_link"),
            ("states", "right_arm", "right_ee", "states.right_arm.pose_base_link"),
            ("actions", "left_arm", "left_ee_target", "actions.left_arm.pose_base_link"),
            ("actions", "right_arm", "right_ee_target", "actions.right_arm.pose_base_link"),
        )
        for top_level, arm_name, child_frame_id, field in pose_specs:
            pose = _section(item, top_level, arm_name).get("pose_base_link")
            _validate_pose(
                pose if isinstance(pose, dict) else {},
                frame_id=validated["base_frame"],
                child_frame_id=child_frame_id,
                frame_index=frame_index,
                field=field,
                issues=issues,
            )

        tcp_pose_specs = (
            ("states", "left_arm", "left_dex1_tcp", "states.left_arm.pose_base_link_tcp"),
            ("states", "right_arm", "right_dex1_tcp", "states.right_arm.pose_base_link_tcp"),
            ("actions", "left_arm", "left_dex1_tcp_target", "actions.left_arm.pose_base_link_tcp"),
            ("actions", "right_arm", "right_dex1_tcp_target", "actions.right_arm.pose_base_link_tcp"),
        )
        for top_level, arm_name, child_frame_id, field in tcp_pose_specs:
            pose = _section(item, top_level, arm_name).get("pose_base_link_tcp")
            _validate_pose(
                pose if isinstance(pose, dict) else {},
                frame_id=validated["base_frame"],
                child_frame_id=child_frame_id,
                frame_index=frame_index,
                field=field,
                issues=issues,
            )

        if velocity_only_base:
            if "slam_map_pose" in states_base:
                issues.append(_issue("MOBILE_UNEXPECTED_GLOBAL_POSE", "error", f"frame {frame_index} velocity-only episode must not contain states.base.slam_map_pose", frame_index=frame_index))
            if "world_pose" in states_base:
                issues.append(_issue("MOBILE_UNEXPECTED_GLOBAL_POSE", "error", f"frame {frame_index} velocity-only episode must not contain states.base.world_pose", frame_index=frame_index))
        else:
            previous_map_pose, identity_count, tf_age_ms, map_speed_mps, map_yaw_rate_radps = _validate_global_slam_pose(
                states_base=states_base,
                timestamps=timestamps,
                validated=validated,
                limits=limits,
                frame_index=frame_index,
                issues=issues,
                previous_map_pose=previous_map_pose,
            )
            identity_assumption_count += identity_count
            max_slam_tf_age_ms = max(max_slam_tf_age_ms, tf_age_ms)
            max_map_speed_mps = max(max_map_speed_mps, map_speed_mps)
            max_map_yaw_rate_radps = max(max_map_yaw_rate_radps, map_yaw_rate_radps)

        velocity = states_base.get("velocity")
        if not isinstance(velocity, Mapping) or not all(_finite_number(velocity.get(key)) for key in ("vx", "vy", "wz")):
            issues.append(_issue("MOBILE_BAD_BASE_VELOCITY", "error", f"frame {frame_index} base velocity vx/vy/wz must be finite", frame_index=frame_index))
        elif velocity.get("frame_id") not in validated["velocity_frames"] or velocity.get("linear_unit") != "m/s" or velocity.get("angular_unit") != "rad/s":
            issues.append(_issue("MOBILE_BASE_VELOCITY_SEMANTICS", "error", f"frame {frame_index} base velocity frame or units are invalid", frame_index=frame_index))

        column_height = states_base.get("column_height_m")
        waist_yaw = states_base.get("waist_yaw")
        if not _finite_number(column_height) or not limits["column_minimum_m"] <= float(column_height) <= limits["column_maximum_m"]:
            issues.append(_issue("MOBILE_BAD_COLUMN_STATE", "error", f"frame {frame_index} column_height_m is missing or out of range", frame_index=frame_index))
        if not _finite_number(waist_yaw) or not limits["waist_yaw_minimum_rad"] <= float(waist_yaw) <= limits["waist_yaw_maximum_rad"]:
            issues.append(_issue("MOBILE_BAD_WAIST_STATE", "error", f"frame {frame_index} waist_yaw is missing or out of range", frame_index=frame_index))

        if not all(_finite_number(actions_base.get(key)) for key in ("vx_cmd", "vy_cmd", "wz_cmd", "z_cmd", "waist_yaw_target")):
            issues.append(_issue("MOBILE_BAD_BASE_ACTION", "error", f"frame {frame_index} base action must contain finite vx/vy/wz/z/waist target", frame_index=frame_index))
        elif not limits["waist_yaw_minimum_rad"] <= float(actions_base["waist_yaw_target"]) <= limits["waist_yaw_maximum_rad"]:
            issues.append(_issue("MOBILE_BAD_WAIST_ACTION", "error", f"frame {frame_index} waist_yaw_target is out of range", frame_index=frame_index))
        elif actions_base.get("frame_id") != validated["base_frame"] or actions_base.get("linear_unit") != "m/s" or actions_base.get("angular_unit") != "rad/s" or actions_base.get("z_cmd_unit") != "normalized":
            issues.append(_issue("MOBILE_BASE_ACTION_SEMANTICS", "error", f"frame {frame_index} base action frame or units are invalid", frame_index=frame_index))

        alignment_specs = [
            ("base_state", limits["base_state_max_delta_ms"], False),
            ("base_height", limits["base_height_max_delta_ms"], False),
        ]
        if not velocity_only_base:
            alignment_specs.append(("slam_tf", limits["slam_tf_max_delta_ms"], False))
        for key, limit, require_past in alignment_specs:
            delta_ms = _validate_alignment(timestamps, key=key, max_abs_delta_ms=limit, require_past=require_past, frame_index=frame_index, issues=issues)
            if delta_ms is not None:
                max_deltas_ms[key] = max(max_deltas_ms[key], abs(delta_ms))
        base_action_delta_ms = _validate_base_action_alignment(
            timestamps,
            max_support_delta_ms=limits["base_action_max_support_delta_ms"],
            frame_index=frame_index,
            issues=issues,
        )
        if base_action_delta_ms is not None:
            max_deltas_ms["base_action"] = max(max_deltas_ms["base_action"], base_action_delta_ms)

    if not velocity_only_base and identity_assumption_count:
        severity = "warning" if validated["allow_identity_source_to_base_link"] else "error"
        issues.append(_issue("MOBILE_IDENTITY_EXTRINSIC_ASSUMPTION", severity, f"{identity_assumption_count}/{len(items)} frames use source_to_base_link_identity_assumed", frame_count=identity_assumption_count))
    if not velocity_only_base and max_map_speed_mps > limits["map_speed_warning_mps"]:
        issues.append(_issue("MOBILE_MAP_SPEED_JUMP", "warning", f"map pose max speed {max_map_speed_mps:.3f}m/s exceeds {limits['map_speed_warning_mps']:.3f}m/s", max_speed_mps=max_map_speed_mps))
    if not velocity_only_base and max_map_yaw_rate_radps > limits["map_yaw_rate_warning_radps"]:
        issues.append(_issue("MOBILE_MAP_YAW_JUMP", "warning", f"map pose max yaw rate {max_map_yaw_rate_radps:.3f}rad/s exceeds {limits['map_yaw_rate_warning_radps']:.3f}rad/s", max_yaw_rate_radps=max_map_yaw_rate_radps))

    source_gap_reports = {
        "base_state": _source_gap_report(
            items,
            key="base_state",
            long_threshold_ms=limits["source_gap_long_threshold_ms"],
            error_consecutive_count=limits["source_gap_error_consecutive_count"],
            error_total_count=limits["source_gap_error_total_count"],
            issues=issues,
        )
    }
    if not velocity_only_base:
        source_gap_reports["slam_tf"] = _source_gap_report(
            items,
            key="slam_tf",
            long_threshold_ms=limits["source_gap_long_threshold_ms"],
            error_consecutive_count=limits["source_gap_error_consecutive_count"],
            error_total_count=limits["source_gap_error_total_count"],
            issues=issues,
        )

    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    return {
        "status": "error" if errors else "warning" if warnings else "ok",
        "frame_count": len(items),
        "present_frame_count": present_count,
        "base_observation_mode": "velocity_only_base_link" if velocity_only_base else "global_slam_map_and_velocity",
        "identity_extrinsic_assumption_frames": identity_assumption_count,
        "timing": dict(validated["timing"]),
        "limits": dict(limits),
        "max_alignment_delta_ms": max_deltas_ms,
        "max_slam_tf_age_ms": max_slam_tf_age_ms,
        "max_map_speed_mps": max_map_speed_mps,
        "max_map_yaw_rate_radps": max_map_yaw_rate_radps,
        "source_gap_reports": source_gap_reports,
        "issues": issues,
        "errors": errors,
        "warnings": warnings,
    }


__all__ = ["validate_mobile_training"]
