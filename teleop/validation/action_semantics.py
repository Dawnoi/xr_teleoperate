"""Independent validation of action semantics for recorded teleoperation items.

The kernel consumes already decoded episode items.  It deliberately does not read
``data.json`` and never infers robot limits or thresholds from observed values.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ARM_SIDES = ("left", "right")
STATUS_ORDER = {"not_applicable": 0, "ok": 1, "warning": 2, "error": 3}

REQUIRED_THRESHOLDS = (
    "near_limit_margin_rad",
    "near_limit_warning_ratio",
    "near_limit_error_ratio",
    "action_step_warning_rad",
    "action_step_error_rad",
    "action_velocity_warning_rad_s",
    "action_velocity_error_rad_s",
    "action_acceleration_warning_rad_s2",
    "action_acceleration_error_rad_s2",
    "motion_error_count",
    "gripper_jump_diagnostic_rad",
    "action_event_min_step_rad",
    "feedback_event_min_step_rad",
    "response_window_s",
    "response_lag_warning_s",
    "response_lag_error_s",
    "response_lag_warning_count",
    "response_lag_error_count",
    "response_unmatched_warning_count",
    "response_unmatched_error_count",
    "pose_translation_step_warning_m",
    "pose_translation_step_error_m",
    "pose_translation_speed_warning_m_s",
    "pose_translation_speed_error_m_s",
    "pose_rotation_step_warning_rad",
    "pose_rotation_step_error_rad",
    "pose_rotation_speed_warning_rad_s",
    "pose_rotation_speed_error_rad_s",
)

COUNT_THRESHOLDS = {
    "motion_error_count",
    "response_lag_error_count",
    "response_unmatched_warning_count",
    "response_unmatched_error_count",
}


def load_action_semantics_config(path: str | Path) -> dict[str, Any]:
    """Load and return an action-semantics configuration mapping.

    Configuration file errors are raised to the caller.  There is no weaker
    runtime configuration or inferred limit fallback.
    """

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"action semantics config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"action semantics config must be a JSON object: {config_path}")
    return config


class ActionSemanticsManager:
    """Small main-chain adapter around the independent validation kernel."""

    def __init__(self, config: Mapping[str, Any] | str | Path):
        if isinstance(config, (str, Path)):
            self.config = load_action_semantics_config(config)
        elif isinstance(config, Mapping):
            self.config = copy.deepcopy(dict(config))
        else:
            raise TypeError("config must be a mapping or a JSON configuration path")

    def validate(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Validate already parsed episode items and return a JSON report."""

        return validate_action_semantics(items, self.config)


def _issue(code: str, severity: str, message: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    result.update(details)
    return result


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _status_from_issues(issues: Sequence[Mapping[str, Any]], default: str = "ok") -> str:
    status = default
    for item in issues:
        severity = str(item.get("severity", "error"))
        if STATUS_ORDER.get(severity, STATUS_ORDER["error"]) > STATUS_ORDER.get(status, 0):
            status = severity
    return status


def _merge_status(*statuses: str) -> str:
    return max(statuses, key=lambda value: STATUS_ORDER.get(value, STATUS_ORDER["error"]))


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * weight


def _statistics(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    return {
        "count": len(clean),
        "peak": max(clean) if clean else None,
        "median": _percentile(clean, 0.5),
        "p95": _percentile(clean, 0.95),
    }


def _empty_section(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "reason": reason, "issues": []}


def _config_errors(config: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(config, Mapping):
        return [_issue("invalid_config", "error", "config must be a mapping")]
    if config.get("robot") != "G1_29":
        issues.append(_issue("invalid_config_robot", "error", "config.robot must be G1_29"))
    if config.get("gripper_quality_policy") != "metrics_only":
        issues.append(
            _issue(
                "invalid_gripper_quality_policy",
                "error",
                "config.gripper_quality_policy must be metrics_only until force-hold diagnostics are recorded",
            )
        )

    joint_order = config.get("joint_order")
    if not isinstance(joint_order, list) or len(joint_order) != 14:
        issues.append(_issue("invalid_joint_order", "error", "joint_order must contain 14 joint names"))

    arms = config.get("arms")
    if not isinstance(arms, Mapping):
        issues.append(_issue("missing_config_arms", "error", "config.arms is required"))
    else:
        all_joint_names: list[str] = []
        for side in ARM_SIDES:
            arm = arms.get(side)
            if not isinstance(arm, Mapping):
                issues.append(_issue("missing_config_arm", "error", f"config.arms.{side} is required"))
                continue
            joints = arm.get("joints")
            if not isinstance(joints, list) or len(joints) != 7:
                issues.append(_issue("invalid_arm_joint_count", "error", f"config.arms.{side}.joints must contain 7 joints"))
                continue
            for joint_index, joint in enumerate(joints):
                if not isinstance(joint, Mapping):
                    issues.append(_issue("invalid_joint_limit", "error", f"config.arms.{side}.joints[{joint_index}] must be an object"))
                    continue
                name = joint.get("name")
                lower = joint.get("lower")
                upper = joint.get("upper")
                if not isinstance(name, str) or not name:
                    issues.append(_issue("invalid_joint_name", "error", f"config.arms.{side}.joints[{joint_index}].name is required"))
                else:
                    all_joint_names.append(name)
                if not _is_finite_number(lower) or not _is_finite_number(upper) or float(lower) >= float(upper):
                    issues.append(_issue("invalid_joint_limit", "error", f"config.arms.{side}.joints[{joint_index}] requires finite lower < upper"))
        if isinstance(joint_order, list) and joint_order != all_joint_names:
            issues.append(_issue("joint_order_mismatch", "error", "joint_order must equal left then right configured joint names"))

    grippers = config.get("grippers")
    if not isinstance(grippers, Mapping):
        issues.append(_issue("missing_config_grippers", "error", "config.grippers is required"))
    else:
        for side in ARM_SIDES:
            gripper = grippers.get(side)
            if not isinstance(gripper, Mapping) or not isinstance(gripper.get("group"), str):
                issues.append(_issue("invalid_config_gripper", "error", f"config.grippers.{side}.group is required"))
            if not isinstance(gripper, Mapping) or not isinstance(gripper.get("index"), int):
                issues.append(_issue("invalid_config_gripper", "error", f"config.grippers.{side}.index is required"))

    thresholds = config.get("thresholds")
    if not isinstance(thresholds, Mapping):
        issues.append(_issue("missing_config_thresholds", "error", "config.thresholds is required"))
    else:
        for key in REQUIRED_THRESHOLDS:
            value = thresholds.get(key)
            if not _is_finite_number(value) or float(value) < 0:
                issues.append(_issue("missing_or_invalid_threshold", "error", f"config.thresholds.{key} must be a finite non-negative number", threshold=key))
            elif key in COUNT_THRESHOLDS and int(value) != float(value):
                issues.append(_issue("invalid_count_threshold", "error", f"config.thresholds.{key} must be an integer", threshold=key))
    return issues


def _read_qpos(
    item: Mapping[str, Any],
    source_key: str,
    group_key: str,
    expected_len: int,
    frame_index: int,
    issues: list[dict[str, Any]],
) -> tuple[list[float] | None, bool]:
    source = item.get(source_key)
    if not isinstance(source, Mapping):
        return None, False
    group = source.get(group_key)
    if not isinstance(group, Mapping) or "qpos" not in group:
        return None, False
    raw = group["qpos"]
    label = f"frame {frame_index} {source_key}.{group_key}.qpos"
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        issues.append(_issue("dimension_mismatch", "error", f"{label} must be a sequence of length {expected_len}", frame_index=frame_index, path=f"{source_key}.{group_key}.qpos"))
        return None, True
    if len(raw) != expected_len:
        issues.append(_issue("dimension_mismatch", "error", f"{label} expected length {expected_len}, got {len(raw)}", frame_index=frame_index, path=f"{source_key}.{group_key}.qpos"))
        return None, True
    values: list[float] = []
    finite = True
    for joint_index, value in enumerate(raw):
        if not _is_finite_number(value):
            finite = False
            issues.append(_issue("non_finite", "error", f"{label}[{joint_index}] is NaN or Inf", frame_index=frame_index, path=f"{source_key}.{group_key}.qpos[{joint_index}]"))
        else:
            values.append(float(value))
    return (values if finite else None), True


def _read_pose(
    item: Mapping[str, Any],
    source_key: str,
    group_key: str,
    frame_index: int,
    issues: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bool]:
    source = item.get(source_key)
    if not isinstance(source, Mapping):
        return None, False
    group = source.get(group_key)
    if not isinstance(group, Mapping) or "pose" not in group:
        return None, False
    pose = group["pose"]
    label = f"frame {frame_index} {source_key}.{group_key}.pose"
    if not isinstance(pose, Mapping):
        issues.append(_issue("invalid_pose", "error", f"{label} must be an object", frame_index=frame_index, path=f"{source_key}.{group_key}.pose"))
        return None, True

    if "matrix4x4" in pose:
        matrix = pose["matrix4x4"]
        if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)) or len(matrix) != 4:
            issues.append(_issue("invalid_pose", "error", f"{label}.matrix4x4 must be 4x4", frame_index=frame_index))
            return None, True
        rows: list[list[float]] = []
        valid = True
        for row in matrix:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 4:
                valid = False
                break
            if not all(_is_finite_number(value) for value in row):
                valid = False
                break
            rows.append([float(value) for value in row])
        if not valid:
            issues.append(_issue("invalid_pose", "error", f"{label}.matrix4x4 contains invalid values", frame_index=frame_index))
            return None, True
        return {"position": [rows[0][3], rows[1][3], rows[2][3]], "rotation": [row[:3] for row in rows[:3]]}, True

    position = pose.get("position")
    if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) != 3 or not all(_is_finite_number(value) for value in position):
        issues.append(_issue("invalid_pose", "error", f"{label}.position must contain three finite values", frame_index=frame_index))
        return None, True
    position_values = [float(value) for value in position]

    if "rotation_matrix" in pose:
        rotation = pose["rotation_matrix"]
        if not isinstance(rotation, Sequence) or isinstance(rotation, (str, bytes)) or len(rotation) != 3:
            issues.append(_issue("invalid_pose", "error", f"{label}.rotation_matrix must be 3x3", frame_index=frame_index))
            return None, True
        rotation_values: list[list[float]] = []
        for row in rotation:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 3 or not all(_is_finite_number(value) for value in row):
                issues.append(_issue("invalid_pose", "error", f"{label}.rotation_matrix contains invalid values", frame_index=frame_index))
                return None, True
            rotation_values.append([float(value) for value in row])
        return {"position": position_values, "rotation": rotation_values}, True

    rpy = pose.get("rpy")
    if not isinstance(rpy, Sequence) or isinstance(rpy, (str, bytes)) or len(rpy) != 3 or not all(_is_finite_number(value) for value in rpy):
        issues.append(_issue("invalid_pose", "error", f"{label}.rpy or rotation_matrix is required", frame_index=frame_index))
        return None, True
    return {"position": position_values, "rotation": _rpy_to_rotation([float(value) for value in rpy])}, True


def _rpy_to_rotation(rpy: Sequence[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _valid_rows(rows: Sequence[list[float] | None], times: Sequence[float | None]) -> list[tuple[float, list[float]]]:
    return [(time, row) for time, row in zip(times, rows) if time is not None and row is not None]


def _motion_issue(
    metric_name: str,
    values: Sequence[float],
    warning_threshold: float,
    error_threshold: float,
    error_count: int,
    unit: str,
) -> list[dict[str, Any]]:
    warning_count = sum(value > warning_threshold for value in values)
    error_value_count = sum(value > error_threshold for value in values)
    if error_value_count or warning_count >= error_count:
        severity = "error"
    elif warning_count:
        severity = "warning"
    else:
        return []
    return [_issue(
        f"excessive_{metric_name}",
        severity,
        f"{metric_name} exceeds configured threshold",
        count=warning_count,
        error_value_count=error_value_count,
        warning_threshold=warning_threshold,
        error_threshold=error_threshold,
        unit=unit,
    )]


def _limit_report(
    rows: Sequence[tuple[float, list[float]]],
    joint: Mapping[str, Any],
    margin: float,
) -> dict[str, Any]:
    lower = float(joint["lower"])
    upper = float(joint["upper"])
    action_near = sum(value <= lower + margin or value >= upper - margin for _, row in rows for value in row[:1])
    values = [value for _, row in rows for value in row[:1]]
    return {
        "sample_count": len(values),
        "near_sample_count": action_near,
        "near_ratio": action_near / len(values) if values else None,
        "lower": lower,
        "upper": upper,
        "margin_rad": margin,
    }


def _joint_motion(
    action_rows: Sequence[tuple[float, list[float]]],
    joint_index: int,
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steps: list[float] = []
    velocities: list[tuple[float, float]] = []
    valid = [(time, row[joint_index]) for time, row in action_rows]
    for (previous_time, previous_value), (time, value) in zip(valid, valid[1:]):
        dt = (time - previous_time) / 1_000_000_000.0
        if dt <= 0:
            continue
        delta = value - previous_value
        steps.append(abs(delta))
        velocities.append((time, delta / dt))
    velocity_values = [abs(value) for _, value in velocities]
    accelerations: list[float] = []
    for (previous_time, previous_velocity), (time, velocity) in zip(velocities, velocities[1:]):
        dt = (time - previous_time) / 1_000_000_000.0
        if dt > 0:
            accelerations.append(abs((velocity - previous_velocity) / dt))

    issues = []
    issues.extend(_motion_issue("action_step", steps, float(thresholds["action_step_warning_rad"]), float(thresholds["action_step_error_rad"]), int(thresholds["motion_error_count"]), "rad"))
    issues.extend(_motion_issue("action_velocity", velocity_values, float(thresholds["action_velocity_warning_rad_s"]), float(thresholds["action_velocity_error_rad_s"]), int(thresholds["motion_error_count"]), "rad/s"))
    issues.extend(_motion_issue("action_acceleration", accelerations, float(thresholds["action_acceleration_warning_rad_s2"]), float(thresholds["action_acceleration_error_rad_s2"]), int(thresholds["motion_error_count"]), "rad/s^2"))
    return {
        "action_step_rad": _statistics(steps),
        "action_velocity_rad_s": _statistics(velocity_values),
        "action_acceleration_rad_s2": _statistics(accelerations),
    }, issues


def _check_hard_limits(
    rows: Sequence[tuple[float, list[float]]],
    side: str,
    source_name: str,
    joints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for sample_index, (_, row) in enumerate(rows):
        for joint_index, joint in enumerate(joints):
            value = row[joint_index]
            if value < float(joint["lower"]) or value > float(joint["upper"]):
                issues.append(_issue(
                    "hard_limit_exceeded",
                    "error",
                    f"{source_name} {side} {joint['name']} exceeds configured hard limit",
                    source=source_name,
                    side=side,
                    joint=joint["name"],
                    sample_index=sample_index,
                    value=value,
                    lower=float(joint["lower"]),
                    upper=float(joint["upper"]),
                ))
    return issues


def _near_limit_for_joint(
    rows: Sequence[tuple[float, list[float]]],
    joint_index: int,
    joint: Mapping[str, Any],
    margin: float,
) -> dict[str, Any]:
    values = [row[joint_index] for _, row in rows]
    lower = float(joint["lower"])
    upper = float(joint["upper"])
    near_count = sum(value <= lower + margin or value >= upper - margin for value in values)
    return {
        "sample_count": len(values),
        "near_sample_count": near_count,
        "near_ratio": near_count / len(values) if values else None,
        "lower": lower,
        "upper": upper,
        "margin_rad": margin,
    }


def _read_items_timestamps(items: Sequence[Mapping[str, Any]], issues: list[dict[str, Any]]) -> list[float | None]:
    times: list[float | None] = []
    previous: float | None = None
    for frame_index, item in enumerate(items):
        if not isinstance(item, Mapping):
            issues.append(_issue("invalid_item", "error", f"items[{frame_index}] must be an object", frame_index=frame_index))
            times.append(None)
            continue
        timestamps = item.get("timestamps")
        if not isinstance(timestamps, Mapping) or "sample_monotonic_ns" not in timestamps:
            issues.append(_issue("missing_timestamp", "error", "host monotonic timestamp is required", frame_index=frame_index, path="timestamps.sample_monotonic_ns"))
            times.append(None)
            continue
        value = timestamps["sample_monotonic_ns"]
        if not _is_finite_number(value) or float(value) < 0:
            issues.append(_issue("invalid_timestamp", "error", "timestamps.sample_monotonic_ns must be finite and non-negative", frame_index=frame_index, path="timestamps.sample_monotonic_ns"))
            times.append(None)
            continue
        current = float(value)
        if previous is not None and current <= previous:
            issues.append(_issue("non_monotonic_timestamp", "error", "host monotonic timestamps must be strictly increasing", frame_index=frame_index, previous=previous, current=current))
        previous = current
        times.append(current)
    return times


def _read_host_timestamps(
    items: Sequence[Mapping[str, Any]],
    source_name: str,
    issues: list[dict[str, Any]],
) -> list[float | None]:
    path = f"timestamps.{source_name}.host_monotonic_ns"
    times: list[float | None] = []
    previous: float | None = None
    edge_hold_started = False
    normal_seen = False
    for frame_index, item in enumerate(items):
        if not isinstance(item, Mapping):
            times.append(None)
            continue
        timestamps = item.get("timestamps")
        source = timestamps.get(source_name) if isinstance(timestamps, Mapping) else None
        if not isinstance(source, Mapping) or "host_monotonic_ns" not in source:
            issues.append(_issue("missing_host_timestamp", "error", f"{path} is required for {source_name} signals", frame_index=frame_index, path=path))
            times.append(None)
            continue
        value = source["host_monotonic_ns"]
        if not _is_finite_number(value) or float(value) < 0:
            issues.append(_issue("invalid_host_timestamp", "error", f"{path} must be finite and non-negative", frame_index=frame_index, path=path))
            times.append(None)
            continue
        current = float(value)
        interpolation_mode = source.get("interpolation_mode")
        is_terminal_edge_hold = interpolation_mode == "edge_hold_last"
        is_leading_edge_future = interpolation_mode == "edge_nearest_future" and not normal_seen and not edge_hold_started
        if not is_leading_edge_future and not is_terminal_edge_hold:
            normal_seen = True
        if edge_hold_started and not is_terminal_edge_hold:
            issues.append(
                _issue(
                    "nonterminal_edge_hold_timestamp",
                    "error",
                    f"{path} resumes after terminal edge_hold_last support",
                    frame_index=frame_index,
                    path=path,
                    interpolation_mode=interpolation_mode,
                )
            )
        if is_terminal_edge_hold:
            edge_hold_started = True
        if previous is not None and (current < previous or (current == previous and not (is_terminal_edge_hold or is_leading_edge_future))):
            issues.append(_issue("non_monotonic_host_timestamp", "error", f"{path} must be strictly increasing", frame_index=frame_index, previous=previous, current=current, path=path))
        previous = current
        times.append(current)
    return times


def _collect_qpos(
    items: Sequence[Mapping[str, Any]],
    times: Sequence[float | None],
    source_key: str,
    group_key: str,
    expected_len: int,
    issues: list[dict[str, Any]],
) -> tuple[list[list[float] | None], list[bool]]:
    rows: list[list[float] | None] = []
    present: list[bool] = []
    for frame_index, item in enumerate(items):
        if isinstance(item, Mapping):
            row, was_present = _read_qpos(item, source_key, group_key, expected_len, frame_index, issues)
        else:
            row, was_present = None, False
        rows.append(row)
        present.append(was_present)
    return rows, present


def _collect_pose(
    items: Sequence[Mapping[str, Any]],
    source_key: str,
    group_key: str,
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any] | None], list[bool]]:
    rows: list[dict[str, Any] | None] = []
    present: list[bool] = []
    for frame_index, item in enumerate(items):
        if isinstance(item, Mapping):
            row, was_present = _read_pose(item, source_key, group_key, frame_index, issues)
        else:
            row, was_present = None, False
        rows.append(row)
        present.append(was_present)
    return rows, present


def _missing_present_issue(present: Sequence[bool], code: str, message: str) -> list[dict[str, Any]]:
    if any(present) and not all(present):
        return [_issue(code, "error", message, missing_count=sum(not value for value in present))]
    return []


def _gripper_report(
    rows: Sequence[list[float] | None],
    present: Sequence[bool],
    times: Sequence[float | None],
    thresholds: Mapping[str, Any],
    side: str,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if not any(present):
        return _empty_section("not_applicable", "gripper action qpos is absent")
    local_issues = _missing_present_issue(present, "missing_gripper_qpos", f"{side} gripper action qpos is missing in some samples")
    valid = [(time, row[0]) for time, row in zip(times, rows) if time is not None and row is not None]
    steps: list[float] = []
    for (previous_time, previous_value), (time, value) in zip(valid, valid[1:]):
        if time > previous_time:
            steps.append(abs(value - previous_value))
    jump_threshold = float(thresholds["gripper_jump_diagnostic_rad"])
    jumps = [value for value in steps if value > jump_threshold]
    return {
        "status": "metrics_only",
        "reason": "raw target and force-hold diagnostics are not recorded; step magnitude alone cannot distinguish normal closure from a fault",
        "issues": local_issues,
        "action_step_rad": _statistics(steps),
        "diagnostic_threshold_rad": jump_threshold,
        "jump_count": len(jumps),
    }


def _joint_space_report(
    items: Sequence[Mapping[str, Any]],
    action_times: Sequence[float | None],
    state_times: Sequence[float | None],
    config: Mapping[str, Any],
    data_issues: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    thresholds = config["thresholds"]
    arms = config["arms"]
    arm_rows: dict[str, list[list[float] | None]] = {}
    arm_present: dict[str, list[bool]] = {}
    state_rows: dict[str, list[list[float] | None]] = {}
    state_present: dict[str, list[bool]] = {}
    joint_action_exists = False
    for side in ARM_SIDES:
        group = arms[side]["group"]
        arm_rows[side], arm_present[side] = _collect_qpos(items, action_times, "actions", group, 7, list(data_issues))
        state_rows[side], state_present[side] = _collect_qpos(items, state_times, "states", group, 7, list(data_issues))
        joint_action_exists = joint_action_exists or any(arm_present[side])

    if not joint_action_exists:
        arms_report = {side: _empty_section("not_applicable", "joint-space action is absent") for side in ARM_SIDES}
        grippers = {
            side: _gripper_report([], [], action_times, thresholds, side, [])
            for side in ARM_SIDES
        }
        return {
            "status": "not_applicable",
            "reason": "joint-space action is absent",
            "issues": [],
            "arms": arms_report,
            "grippers": grippers,
        }

    section_issues: list[dict[str, Any]] = list(data_issues)
    arms_report: dict[str, Any] = {}
    margin = float(thresholds["near_limit_margin_rad"])
    for side in ARM_SIDES:
        local_issues: list[dict[str, Any]] = []
        if not any(arm_present[side]):
            local_issues.append(_issue("missing_action_qpos", "error", f"{side} arm action qpos is absent while joint validation is active"))
            arms_report[side] = {"status": "error", "issues": local_issues, "reason": "joint-space action is absent"}
            section_issues.extend(local_issues)
            continue
        local_issues.extend(_missing_present_issue(arm_present[side], "missing_action_qpos", f"{side} arm action qpos is missing in some samples"))
        if not any(state_present[side]):
            local_issues.append(_issue("missing_state_qpos", "error", f"{side} arm state qpos is absent while action qpos is present"))
        else:
            local_issues.extend(_missing_present_issue(state_present[side], "missing_state_qpos", f"{side} arm state qpos is missing in some samples"))

        action_valid = _valid_rows(arm_rows[side], action_times)
        state_valid = _valid_rows(state_rows[side], state_times)
        action_value_rows = [(0.0, row) for row in arm_rows[side] if row is not None]
        state_value_rows = [(0.0, row) for row in state_rows[side] if row is not None]
        joints_report: dict[str, Any] = {}
        for joint_index, joint in enumerate(arms[side]["joints"]):
            motion, motion_issues = _joint_motion(action_valid, joint_index, thresholds)
            action_limit_issues = _check_hard_limits(action_value_rows, side, "action", [joint if index == joint_index else {"lower": -math.inf, "upper": math.inf, "name": "unused"} for index, joint in enumerate(arms[side]["joints"])])
            state_limit_issues = _check_hard_limits(state_value_rows, side, "state", [joint if index == joint_index else {"lower": -math.inf, "upper": math.inf, "name": "unused"} for index, joint in enumerate(arms[side]["joints"])])
            near_action = _near_limit_for_joint(action_value_rows, joint_index, joint, margin)
            near_state = _near_limit_for_joint(state_value_rows, joint_index, joint, margin)
            near_issues: list[dict[str, Any]] = []
            for source_name, near in (("action", near_action), ("state", near_state)):
                ratio = near["near_ratio"]
                if ratio is not None and ratio >= float(thresholds["near_limit_error_ratio"]):
                    near_issues.append(_issue("large_near_limit_ratio", "error", f"{source_name} {side} {joint['name']} is frequently near a hard limit", source=source_name, ratio=ratio))
                elif ratio is not None and ratio >= float(thresholds["near_limit_warning_ratio"]):
                    near_issues.append(_issue("near_limit_ratio", "warning", f"{source_name} {side} {joint['name']} is near a hard limit", source=source_name, ratio=ratio))
            joint_issues = motion_issues + near_issues + action_limit_issues + state_limit_issues
            joints_report[joint["name"]] = {
                "status": _status_from_issues(joint_issues),
                "issues": joint_issues,
                **motion,
                "near_hard_limit": {"action": near_action, "state": near_state},
            }
            local_issues.extend(joint_issues)
        section_issues.extend(local_issues)
        arms_report[side] = {
            "status": _status_from_issues(local_issues),
            "issues": local_issues,
            "joints": joints_report,
        }

    gripper_issues: list[dict[str, Any]] = []
    grippers_report: dict[str, Any] = {}
    for side in ARM_SIDES:
        group = config["grippers"][side]["group"]
        gripper_rows, gripper_present = _collect_qpos(items, action_times, "actions", group, 1, gripper_issues)
        gripper_report = _gripper_report(gripper_rows, gripper_present, action_times, thresholds, side, gripper_issues)
        grippers_report[side] = gripper_report
        gripper_issues.extend(gripper_report.get("issues", []))
    section_issues.extend(gripper_issues)
    return {
        "status": _status_from_issues(section_issues),
        "issues": section_issues,
        "arms": arms_report,
        "grippers": grippers_report,
    }


def _rotation_angle(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> float:
    relative = [
        [sum(first[row][index] * second[row][index] for row in range(3)) for index in range(3)]
        for row in range(3)
    ]
    trace = relative[0][0] + relative[1][1] + relative[2][2]
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.acos(cosine)


def _vector_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def _pose_motion(
    rows: Sequence[tuple[float, dict[str, Any]]],
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    translation_steps: list[float] = []
    translation_speeds: list[float] = []
    rotation_steps: list[float] = []
    rotation_speeds: list[float] = []
    for (previous_time, previous_pose), (time, pose) in zip(rows, rows[1:]):
        dt = (time - previous_time) / 1_000_000_000.0
        if dt <= 0:
            continue
        translation_step = _vector_distance(previous_pose["position"], pose["position"])
        rotation_step = _rotation_angle(previous_pose["rotation"], pose["rotation"])
        translation_steps.append(translation_step)
        translation_speeds.append(translation_step / dt)
        rotation_steps.append(rotation_step)
        rotation_speeds.append(rotation_step / dt)
    issues: list[dict[str, Any]] = []
    error_count = int(thresholds["motion_error_count"])
    issues.extend(_motion_issue("pose_translation_step", translation_steps, float(thresholds["pose_translation_step_warning_m"]), float(thresholds["pose_translation_step_error_m"]), error_count, "m"))
    issues.extend(_motion_issue("pose_translation_speed", translation_speeds, float(thresholds["pose_translation_speed_warning_m_s"]), float(thresholds["pose_translation_speed_error_m_s"]), error_count, "m/s"))
    issues.extend(_motion_issue("pose_rotation_step", rotation_steps, float(thresholds["pose_rotation_step_warning_rad"]), float(thresholds["pose_rotation_step_error_rad"]), error_count, "rad"))
    issues.extend(_motion_issue("pose_rotation_speed", rotation_speeds, float(thresholds["pose_rotation_speed_warning_rad_s"]), float(thresholds["pose_rotation_speed_error_rad_s"]), error_count, "rad/s"))
    return {
        "translation_step_m": _statistics(translation_steps),
        "translation_speed_m_s": _statistics(translation_speeds),
        "rotation_step_rad": _statistics(rotation_steps),
        "rotation_speed_rad_s": _statistics(rotation_speeds),
    }, issues


def _pose_space_report(
    items: Sequence[Mapping[str, Any]],
    action_times: Sequence[float | None],
    config: Mapping[str, Any],
    data_issues: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    arms = config["arms"]
    thresholds = config["thresholds"]
    arms_report: dict[str, Any] = {}
    any_pose = False
    for side in ARM_SIDES:
        group = arms[side]["group"]
        rows, present = _collect_pose(items, "actions", group, list(data_issues))
        any_pose = any_pose or any(present)
        if not any(present):
            arms_report[side] = _empty_section("not_applicable", "pose action is absent")
            continue
        local_issues = _missing_present_issue(present, "missing_action_pose", f"{side} arm pose action is missing in some samples")
        valid = [(time, row) for time, row in zip(action_times, rows) if time is not None and row is not None]
        motion, motion_issues = _pose_motion(valid, thresholds)
        local_issues.extend(motion_issues)
        arms_report[side] = {
            "status": _status_from_issues(local_issues),
            "issues": local_issues,
            **motion,
        }
    if not any_pose:
        return {
            "status": "not_applicable",
            "reason": "pose action is absent",
            "issues": [],
            "arms": arms_report,
        }
    section_issues: list[dict[str, Any]] = list(data_issues)
    for arm in arms_report.values():
        section_issues.extend(arm.get("issues", []))
    return {"status": _status_from_issues(section_issues), "issues": section_issues, "arms": arms_report}


def _event_list(
    rows: Sequence[list[float] | None],
    times: Sequence[float | None],
    joint_index: int,
    minimum_step: float,
) -> list[dict[str, Any]]:
    valid = [(index, time, row[joint_index]) for index, (time, row) in enumerate(zip(times, rows)) if time is not None and row is not None]
    events: list[dict[str, Any]] = []
    for previous, current in zip(valid, valid[1:]):
        _, previous_time, previous_value = previous
        sample_index, current_time, current_value = current
        delta = current_value - previous_value
        if current_time > previous_time and abs(delta) >= minimum_step:
            events.append({"sample_index": sample_index, "time_ns": current_time, "direction": 1 if delta > 0 else -1, "step": abs(delta)})
    return events


def _match_response_events(
    commands: Sequence[Mapping[str, Any]],
    feedback: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> tuple[list[float], int]:
    window_ns = float(thresholds["response_window_s"]) * 1_000_000_000.0
    used_feedback: set[int] = set()
    lags: list[float] = []
    unmatched = 0
    for command in commands:
        candidates = [
            (index, event)
            for index, event in enumerate(feedback)
            if index not in used_feedback
            and event["time_ns"] > command["time_ns"]
            and event["time_ns"] - command["time_ns"] <= window_ns
            and event["direction"] == command["direction"]
        ]
        if not candidates:
            unmatched += 1
            continue
        feedback_index, event = min(candidates, key=lambda pair: pair[1]["time_ns"])
        used_feedback.add(feedback_index)
        lags.append((event["time_ns"] - command["time_ns"]) / 1_000_000_000.0)
    return lags, unmatched


def _response_lag_joint(
    action_rows: Sequence[list[float] | None],
    state_rows: Sequence[list[float] | None],
    action_times: Sequence[float | None],
    state_times: Sequence[float | None],
    joint_index: int,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    commands = _event_list(action_rows, action_times, joint_index, float(thresholds["action_event_min_step_rad"]))
    feedback = _event_list(state_rows, state_times, joint_index, float(thresholds["feedback_event_min_step_rad"]))
    lags, unmatched = _match_response_events(commands, feedback, thresholds)
    lag_warning_count = sum(value > float(thresholds["response_lag_warning_s"]) for value in lags)
    lag_error_value_count = sum(value > float(thresholds["response_lag_error_s"]) for value in lags)
    local_issues: list[dict[str, Any]] = []
    if lag_error_value_count >= int(thresholds["response_lag_error_count"]):
        local_issues.append(_issue("excessive_response_lag", "error", "response lag exceeds configured severity threshold", count=lag_warning_count, error_value_count=lag_error_value_count))
    elif lag_warning_count >= int(thresholds["response_lag_warning_count"]):
        local_issues.append(_issue("excessive_response_lag", "warning", "response lag exceeds configured warning threshold", count=lag_warning_count))
    if unmatched >= int(thresholds["response_unmatched_error_count"]):
        local_issues.append(_issue("unmatched_command_events", "error", "many command events have no matching later feedback event", count=unmatched))
    elif unmatched >= int(thresholds["response_unmatched_warning_count"]):
        local_issues.append(_issue("unmatched_command_events", "warning", "a command event has no matching later feedback event", count=unmatched))
    statistics = _statistics(lags)
    return {
        "status": _status_from_issues(local_issues),
        "issues": local_issues,
        "command_event_count": len(commands),
        "feedback_event_count": len(feedback),
        "sample_count": len(lags),
        "median_s": statistics["median"],
        "p95_s": statistics["p95"],
        "max_s": statistics["peak"],
        "unmatched_command_event_count": unmatched,
        "window_s": float(thresholds["response_window_s"]),
    }


def _response_lag_report(
    action_rows: Mapping[str, Sequence[list[float] | None]],
    action_present: Mapping[str, Sequence[bool]],
    state_rows: Mapping[str, Sequence[list[float] | None]],
    state_present: Mapping[str, Sequence[bool]],
    action_times: Sequence[float | None],
    state_times: Sequence[float | None],
    config: Mapping[str, Any],
    data_issues: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    thresholds = config["thresholds"]
    arms_config = config["arms"]
    arms_report: dict[str, Any] = {}
    section_issues: list[dict[str, Any]] = list(data_issues)
    any_action = any(any(present) for present in action_present.values())
    if not any_action:
        return {"status": "not_applicable", "reason": "joint-space action is absent", "issues": [], "arms": {side: _empty_section("not_applicable", "joint-space action is absent") for side in ARM_SIDES}}
    for side in ARM_SIDES:
        local_issues: list[dict[str, Any]] = []
        if not any(action_present[side]):
            local_issues.append(_issue("missing_action_qpos", "error", f"{side} arm action qpos is absent while response validation is active"))
            arms_report[side] = {"status": "error", "issues": local_issues, "reason": "joint-space action is absent"}
            section_issues.extend(local_issues)
            continue
        if not any(state_present[side]):
            local_issues.append(_issue("missing_state_qpos", "error", f"{side} arm state qpos is absent; response lag cannot be measured"))
            arms_report[side] = {"status": "error", "issues": local_issues, "reason": "state feedback qpos is absent"}
            section_issues.extend(local_issues)
            continue
        local_issues.extend(_missing_present_issue(action_present[side], "missing_action_qpos", f"{side} arm action qpos is missing in some samples"))
        local_issues.extend(_missing_present_issue(state_present[side], "missing_state_qpos", f"{side} arm state qpos is missing in some samples"))
        joints_report: dict[str, Any] = {}
        for joint_index, joint in enumerate(arms_config[side]["joints"]):
            joint_report = _response_lag_joint(action_rows[side], state_rows[side], action_times, state_times, joint_index, thresholds)
            joints_report[joint["name"]] = joint_report
            local_issues.extend(joint_report["issues"])
        section_issues.extend(local_issues)
        arms_report[side] = {"status": _status_from_issues(local_issues), "issues": local_issues, "joints": joints_report}
    return {
        "status": _status_from_issues(section_issues),
        "issues": section_issues,
        "window_s": float(thresholds["response_window_s"]),
        "arms": arms_report,
    }


def validate_action_semantics(
    items: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate already parsed episode items and return a JSON-serializable report."""

    if config_path is not None:
        config = load_action_semantics_config(config_path)
    config_snapshot = copy.deepcopy(dict(config)) if isinstance(config, Mapping) else {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "error",
        "timestamp_source": "timestamps.sample_monotonic_ns",
        "sample_count": len(items) if isinstance(items, Sequence) and not isinstance(items, (str, bytes)) else 0,
        "config": config_snapshot,
        "issues": [],
        "joint_space": _empty_section("not_applicable", "config is not validated"),
        "pose_space": _empty_section("not_applicable", "config is not validated"),
        "response_lag": _empty_section("not_applicable", "config is not validated"),
    }
    config_issues = _config_errors(config)
    if config_issues:
        report["issues"] = config_issues
        report["status"] = "error"
        return report
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        report["issues"] = [_issue("invalid_items", "error", "items must be a sequence of parsed episode objects")]
        return report
    if not items:
        report["issues"] = [_issue("empty_items", "error", "episode items must not be empty")]
        return report

    data_issues: list[dict[str, Any]] = []
    _read_items_timestamps(items, data_issues)
    action_times = _read_host_timestamps(items, "action", data_issues)
    state_times = _read_host_timestamps(items, "state", data_issues)
    arms_config = config["arms"]
    action_rows: dict[str, list[list[float] | None]] = {}
    action_present: dict[str, list[bool]] = {}
    state_rows: dict[str, list[list[float] | None]] = {}
    state_present: dict[str, list[bool]] = {}
    for side in ARM_SIDES:
        group = arms_config[side]["group"]
        action_rows[side], action_present[side] = _collect_qpos(items, action_times, "actions", group, 7, data_issues)
        state_rows[side], state_present[side] = _collect_qpos(items, state_times, "states", group, 7, data_issues)

    joint_space = _joint_space_report(items, action_times, state_times, config, data_issues)
    pose_space = _pose_space_report(items, action_times, config, data_issues)
    response_lag = _response_lag_report(
        action_rows,
        action_present,
        state_rows,
        state_present,
        action_times,
        state_times,
        config,
        data_issues,
    )
    report["joint_space"] = joint_space
    report["pose_space"] = pose_space
    report["response_lag"] = response_lag
    report["issues"] = (
        data_issues
        + list(joint_space.get("issues", []))
        + list(pose_space.get("issues", []))
        + list(response_lag.get("issues", []))
    )
    report["status"] = _merge_status(
        _status_from_issues(report["issues"]),
        joint_space["status"],
        pose_space["status"],
        response_lag["status"],
    )
    report["errors"] = [
        issue for issue in report["issues"] if issue.get("severity") == "error"
    ]
    report["warnings"] = [
        issue for issue in report["issues"] if issue.get("severity") == "warning"
    ]
    return report


__all__ = [
    "ActionSemanticsManager",
    "load_action_semantics_config",
    "validate_action_semantics",
]
