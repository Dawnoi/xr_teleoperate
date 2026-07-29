#!/usr/bin/env python3
"""离线重采样移动操作 episode 的底盘状态、SLAM 位姿和底盘动作。

原始 ``data.json`` 从不修改。输出 episode 使用软链接复用原始图像和
``data.json``，并将时间重采样结果写入 ``data.mobile_aligned.json``。

旧 episode 必须从图像时刻两侧的真实源样本严格插值。新 episode 若已经由
实时采集链路严格插值，则直接复用已记录的值和其前后支撑时间戳。
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping


Json = dict[str, Any]

_DEFAULT_BASE_STATE_MAX_DELTA_MS = 131.578947
_DEFAULT_BASE_HEIGHT_MAX_DELTA_MS = 150.0
_DEFAULT_SLAM_TF_MAX_DELTA_MS = 125.0
_DEFAULT_BASE_ACTION_MAX_DELTA_MS = 83.333333


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object, got {type(value).__name__}")
    return value


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer timestamp, got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be a positive timestamp, got {value!r}")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _item_base(item: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    return _mapping(_mapping(item.get("states"), f"{field}.states").get("base"), f"{field}.states.base")


def _item_action_base(item: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    return _mapping(_mapping(item.get("actions"), f"{field}.actions").get("base"), f"{field}.actions.base")


def _item_timestamps(item: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    return _mapping(item.get("timestamps"), f"{field}.timestamps")


def _frame_index(item: Mapping[str, Any], episode_name: str) -> int:
    value = item.get("idx")
    if isinstance(value, bool):
        raise ValueError(f"episode={episode_name} item.idx must be a non-negative integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"episode={episode_name} item.idx must be a non-negative integer, got {value!r}")
    return result


def _payload_base_state(item: Mapping[str, Any], field: str) -> Json:
    base = _item_base(item, field)
    return {
        "world_pose": copy.deepcopy(_mapping(base.get("world_pose"), f"{field}.states.base.world_pose")),
        "velocity": copy.deepcopy(_mapping(base.get("velocity"), f"{field}.states.base.velocity")),
    }


def _payload_base_height(item: Mapping[str, Any], field: str) -> Json:
    base = _item_base(item, field)
    return {"height": copy.deepcopy(_mapping(base.get("height"), f"{field}.states.base.height"))}


def _payload_slam_tf(item: Mapping[str, Any], field: str) -> Json:
    base = _item_base(item, field)
    return {"slam_map_pose": copy.deepcopy(_mapping(base.get("slam_map_pose"), f"{field}.states.base.slam_map_pose"))}


def _payload_base_action(item: Mapping[str, Any], field: str) -> Json:
    payload = copy.deepcopy(dict(_item_action_base(item, field)))
    # waist_yaw_target belongs to the arm action alignment at the image timestamp,
    # not to the independently timestamped base-action command history.
    payload.pop("waist_yaw_target", None)
    return payload


def _stream_stamp(item: Mapping[str, Any], key: str, field: str) -> Mapping[str, Any]:
    return _mapping(_item_timestamps(item, field).get(key), f"{field}.timestamps.{key}")


def _source_series(
    items: list[Mapping[str, Any]],
    *,
    episode_name: str,
    timestamp_key: str,
    payload: Callable[[Mapping[str, Any], str], Json],
    include: Callable[[Mapping[str, Any]], bool] | None = None,
    allow_empty: bool = False,
) -> list[tuple[int, Json]]:
    samples: dict[int, Json] = {}
    for item in items:
        idx = _frame_index(item, episode_name)
        field = f"episode={episode_name} frame={idx}"
        stamp = _stream_stamp(item, timestamp_key, field)
        if include is not None and not include(stamp):
            continue
        source_t_ns = _integer(stamp.get("support_source_t_ns"), f"{field}.timestamps.{timestamp_key}.support_source_t_ns")
        candidate = payload(item, field)
        existing = samples.get(source_t_ns)
        if existing is not None and _canonical_json(existing) != _canonical_json(candidate):
            raise RuntimeError(
                f"episode={episode_name} timestamps.{timestamp_key}.support_source_t_ns={source_t_ns} "
                "maps to inconsistent source payloads"
            )
        samples[source_t_ns] = candidate
    if not samples and not allow_empty:
        raise RuntimeError(f"episode={episode_name} has no source samples for timestamps.{timestamp_key}")
    return sorted(samples.items())


def _normalized_quaternion(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field} must be a length-4 quaternion")
    quaternion = [_finite(component, f"{field}[{index}]") for index, component in enumerate(value)]
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm < 1e-12:
        raise ValueError(f"{field} has zero norm")
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"{field} must have unit norm, got {norm:.9f}")
    return [component / norm for component in quaternion]


def _slerp_xyzw(first: list[float], second: list[float], alpha: float) -> list[float]:
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


def _yaw_from_quaternion_xyzw(quaternion: list[float]) -> float:
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _same_metadata(first: Mapping[str, Any], second: Mapping[str, Any], keys: tuple[str, ...], field: str) -> Json:
    result: Json = {}
    for key in keys:
        first_value = first.get(key)
        second_value = second.get(key)
        if first_value != second_value:
            raise RuntimeError(f"{field}.{key} changes between interpolation supports: {first_value!r} != {second_value!r}")
        result[key] = copy.deepcopy(first_value)
    return result


def _interpolate_pose(first: Mapping[str, Any], second: Mapping[str, Any], alpha: float, field: str) -> Json:
    metadata = _same_metadata(first, second, ("frame_id", "source_topic"), field)
    if "child_frame_id" in first or "child_frame_id" in second:
        metadata.update(_same_metadata(first, second, ("child_frame_id",), field))
    quaternion = _slerp_xyzw(
        _normalized_quaternion(first.get("quat_xyzw"), f"{field}.prev.quat_xyzw"),
        _normalized_quaternion(second.get("quat_xyzw"), f"{field}.next.quat_xyzw"),
        alpha,
    )
    for key in ("yaw",):
        _finite(first.get(key), f"{field}.prev.{key}")
        _finite(second.get(key), f"{field}.next.{key}")
    result: Json = {
        "x": _finite(first.get("x"), f"{field}.prev.x") + alpha * (_finite(second.get("x"), f"{field}.next.x") - _finite(first.get("x"), f"{field}.prev.x")),
        "y": _finite(first.get("y"), f"{field}.prev.y") + alpha * (_finite(second.get("y"), f"{field}.next.y") - _finite(first.get("y"), f"{field}.prev.y")),
        "z": _finite(first.get("z"), f"{field}.prev.z") + alpha * (_finite(second.get("z"), f"{field}.next.z") - _finite(first.get("z"), f"{field}.prev.z")),
        "yaw": _yaw_from_quaternion_xyzw(quaternion),
        "quat_xyzw": quaternion,
    }
    result.update(metadata)
    return result


def _interpolate_velocity(first: Mapping[str, Any], second: Mapping[str, Any], alpha: float, field: str) -> Json:
    metadata = _same_metadata(first, second, ("frame_id", "linear_unit", "angular_unit", "source_topic"), field)
    result = {
        key: _finite(first.get(key), f"{field}.prev.{key}")
        + alpha * (_finite(second.get(key), f"{field}.next.{key}") - _finite(first.get(key), f"{field}.prev.{key}"))
        for key in ("vx", "vy", "vz", "wz")
    }
    result.update(metadata)
    return result


def _interpolate_base_state(first: Json, second: Json, alpha: float) -> Json:
    return {
        "world_pose": _interpolate_pose(first["world_pose"], second["world_pose"], alpha, "base_state.world_pose"),
        "velocity": _interpolate_velocity(first["velocity"], second["velocity"], alpha, "base_state.velocity"),
    }


def _interpolate_height(first: Json, second: Json, alpha: float) -> Json:
    first_height = _mapping(first.get("height"), "base_height.prev.height")
    second_height = _mapping(second.get("height"), "base_height.next.height")
    metadata = _same_metadata(first_height, second_height, ("source_topic",), "base_height.height")
    result = {
        "z": _finite(first_height.get("z"), "base_height.prev.height.z")
        + alpha * (_finite(second_height.get("z"), "base_height.next.height.z") - _finite(first_height.get("z"), "base_height.prev.height.z")),
    }
    result.update(metadata)
    return {"height": result}


def _interpolate_slam_pose(first: Json, second: Json, alpha: float) -> Json:
    first_pose = _mapping(first.get("slam_map_pose"), "slam_tf.prev.slam_map_pose")
    second_pose = _mapping(second.get("slam_map_pose"), "slam_tf.next.slam_map_pose")
    result = _interpolate_pose(first_pose, second_pose, alpha, "slam_tf.slam_map_pose")
    semantic_keys = ("source_child_frame_id", "source_to_base_link_identity_assumed")
    result.update(_same_metadata(first_pose, second_pose, semantic_keys, "slam_tf.slam_map_pose"))
    result["interpolation_support"] = {
        "prev_tf_header_stamp_ns": _integer(first_pose.get("tf_header_stamp_ns"), "slam_tf.prev.tf_header_stamp_ns"),
        "next_tf_header_stamp_ns": _integer(second_pose.get("tf_header_stamp_ns"), "slam_tf.next.tf_header_stamp_ns"),
        "prev_tf_lookup_monotonic_ns": _integer(first_pose.get("tf_lookup_monotonic_ns"), "slam_tf.prev.tf_lookup_monotonic_ns"),
        "next_tf_lookup_monotonic_ns": _integer(second_pose.get("tf_lookup_monotonic_ns"), "slam_tf.next.tf_lookup_monotonic_ns"),
        "prev_source_chain": copy.deepcopy(first_pose.get("source_chain")),
        "next_source_chain": copy.deepcopy(second_pose.get("source_chain")),
    }
    return {"slam_map_pose": result}


def _validate_action(action: Mapping[str, Any], field: str) -> Json:
    result = copy.deepcopy(dict(action))
    for key in ("vx_cmd", "vy_cmd", "wz_cmd", "z_cmd"):
        result[key] = _finite(action.get(key), f"{field}.{key}")
    if "waist_yaw_target" in action:
        result["waist_yaw_target"] = _finite(action.get("waist_yaw_target"), f"{field}.waist_yaw_target")
    return result


def _append_image_aligned_waist_target(action: Json, item: Mapping[str, Any], field: str) -> None:
    raw_action = _item_action_base(item, field)
    if "waist_yaw_target" in raw_action:
        action["waist_yaw_target"] = _finite(raw_action.get("waist_yaw_target"), f"{field}.actions.base.waist_yaw_target")
        action["waist_yaw_target_origin"] = "recorded_arm_action_at_sample"


def _interpolate_action(first: Json, second: Json, alpha: float) -> Json:
    first = _validate_action(first, "base_action.prev")
    second = _validate_action(second, "base_action.next")
    numeric_keys = ["vx_cmd", "vy_cmd", "wz_cmd", "z_cmd"]
    if ("waist_yaw_target" in first) != ("waist_yaw_target" in second):
        raise RuntimeError("base_action.waist_yaw_target is present on only one interpolation support")
    if "waist_yaw_target" in first:
        numeric_keys.append("waist_yaw_target")
    result: Json = {
        key: first[key] + alpha * (second[key] - first[key])
        for key in numeric_keys
    }
    metadata_keys = tuple(sorted((set(first) | set(second)) - set(numeric_keys)))
    result.update(_same_metadata(first, second, metadata_keys, "base_action"))
    return result


def _alignment_stamp(
    *,
    target_t_ns: int,
    mode: str,
    prev_t_ns: int | None,
    next_t_ns: int | None,
    fallback_reason: str | None = None,
    recorded_online_linear: bool = False,
) -> Json:
    result: Json = {
        "interpolation_mode": mode,
        "target_monotonic_ns": target_t_ns,
    }
    if mode == "exact":
        if prev_t_ns is None:
            raise RuntimeError("exact alignment requires a support timestamp")
        result.update(
            {
                "support_source_count": 1,
                "support_source_t_ns": prev_t_ns,
                "delta_to_sample_ns": prev_t_ns - target_t_ns,
            }
        )
        return result
    if mode == "nearest_boundary_fallback":
        if prev_t_ns is None or fallback_reason is None:
            raise RuntimeError("nearest boundary fallback requires support timestamp and reason")
        result.update(
            {
                "support_source_count": 1,
                "support_source_t_ns": prev_t_ns,
                "delta_to_sample_ns": prev_t_ns - target_t_ns,
                "fallback_reason": fallback_reason,
            }
        )
        return result
    if prev_t_ns is None or next_t_ns is None or next_t_ns <= prev_t_ns:
        raise RuntimeError(f"{mode} alignment requires strictly increasing previous and next support timestamps")
    alpha = (target_t_ns - prev_t_ns) / (next_t_ns - prev_t_ns)
    result.update(
        {
            "support_source_count": 2,
            "interp_prev_t_ns": prev_t_ns,
            "interp_next_t_ns": next_t_ns,
            "interp_alpha": alpha,
            "support_prev_delta_to_sample_ns": prev_t_ns - target_t_ns,
            "support_next_delta_to_sample_ns": next_t_ns - target_t_ns,
        }
    )
    if recorded_online_linear:
        result["value_origin"] = "recorded_online_linear"
    return result


def _align_from_series(
    series: list[tuple[int, Json]],
    *,
    target_t_ns: int,
    interpolate: Callable[[Json, Json, float], Json],
    stream_name: str,
    max_support_delta_ns: int,
) -> tuple[Json, Json]:
    times = [entry[0] for entry in series]
    index = bisect.bisect_left(times, target_t_ns)
    if index < len(series) and series[index][0] == target_t_ns:
        source_t_ns, payload = series[index]
        return copy.deepcopy(payload), _alignment_stamp(target_t_ns=target_t_ns, mode="exact", prev_t_ns=source_t_ns, next_t_ns=None)
    if index == 0:
        raise RuntimeError(
            f"{stream_name} has no previous source support for target_monotonic_ns={target_t_ns}"
        )
    if index == len(series):
        raise RuntimeError(
            f"{stream_name} has no next source support for target_monotonic_ns={target_t_ns}"
        )
    prev_t_ns, previous = series[index - 1]
    next_t_ns, following = series[index]
    if next_t_ns <= prev_t_ns:
        raise RuntimeError(f"{stream_name} source timestamps are not strictly increasing")
    previous_delta_ns = target_t_ns - prev_t_ns
    next_delta_ns = next_t_ns - target_t_ns
    if previous_delta_ns > max_support_delta_ns or next_delta_ns > max_support_delta_ns:
        raise RuntimeError(
            f"{stream_name} support exceeds max delta at target_monotonic_ns={target_t_ns}: "
            f"prev_delta_ns={previous_delta_ns} next_delta_ns={next_delta_ns} "
            f"limit_ns={max_support_delta_ns}"
        )
    alpha = (target_t_ns - prev_t_ns) / (next_t_ns - prev_t_ns)
    return interpolate(previous, following, alpha), _alignment_stamp(
        target_t_ns=target_t_ns,
        mode="linear",
        prev_t_ns=prev_t_ns,
        next_t_ns=next_t_ns,
    )


def _align_recorded_linear_payload(
    item: Mapping[str, Any],
    *,
    episode_name: str,
    timestamp_key: str,
    payload: Callable[[Mapping[str, Any], str], Json],
    max_support_delta_ns: int,
) -> tuple[Json, Json]:
    idx = _frame_index(item, episode_name)
    field = f"episode={episode_name} frame={idx}"
    stamp = _stream_stamp(item, timestamp_key, field)
    target_t_ns = _integer(_item_timestamps(item, field).get("sample_monotonic_ns"), f"{field}.timestamps.sample_monotonic_ns")
    previous_t_ns = _integer(stamp.get("support_prev_t_ns"), f"{field}.timestamps.{timestamp_key}.support_prev_t_ns")
    next_t_ns = _integer(stamp.get("support_next_t_ns"), f"{field}.timestamps.{timestamp_key}.support_next_t_ns")
    if previous_t_ns > target_t_ns or next_t_ns < target_t_ns or next_t_ns <= previous_t_ns:
        raise RuntimeError(f"{field}.timestamps.{timestamp_key} has invalid recorded linear support")
    previous_delta_ns = target_t_ns - previous_t_ns
    next_delta_ns = next_t_ns - target_t_ns
    if previous_delta_ns > max_support_delta_ns or next_delta_ns > max_support_delta_ns:
        raise RuntimeError(
            f"{field}.timestamps.{timestamp_key} support exceeds max delta: "
            f"prev_delta_ns={previous_delta_ns} next_delta_ns={next_delta_ns} "
            f"limit_ns={max_support_delta_ns}"
        )
    return copy.deepcopy(payload(item, field)), _alignment_stamp(
        target_t_ns=target_t_ns,
        mode="linear",
        prev_t_ns=previous_t_ns,
        next_t_ns=next_t_ns,
        recorded_online_linear=True,
    )


def _align_recorded_linear_action(
    item: Mapping[str, Any], *, episode_name: str, max_support_delta_ns: int
) -> tuple[Json, Json]:
    return _align_recorded_linear_payload(
        item,
        episode_name=episode_name,
        timestamp_key="base_action",
        payload=lambda source_item, field: _validate_action(
            _payload_base_action(source_item, field),
            f"{field}.actions.base",
        ),
        max_support_delta_ns=max_support_delta_ns,
    )


def _align_episode(
    source_episode_dir: Path,
    output_episode_dir: Path,
    *,
    base_state_max_delta_ns: int,
    base_height_max_delta_ns: int,
    slam_tf_max_delta_ns: int,
    base_action_max_delta_ns: int,
) -> Json:
    source_data_path = source_episode_dir / "data.json"
    document = json.loads(source_data_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"episode={source_episode_dir.name} data.json root must be an object")
    raw_items = document.get("data")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"episode={source_episode_dir.name} data.json.data must be a non-empty list")
    items = [_mapping(item, f"episode={source_episode_dir.name} data[{position}]") for position, item in enumerate(raw_items)]
    episode_name = source_episode_dir.name
    state_series = _source_series(
        items,
        episode_name=episode_name,
        timestamp_key="base_state",
        payload=_payload_base_state,
        include=lambda stamp: stamp.get("interpolation_mode") != "linear",
        allow_empty=True,
    )
    height_series = _source_series(
        items,
        episode_name=episode_name,
        timestamp_key="base_height",
        payload=_payload_base_height,
        include=lambda stamp: stamp.get("interpolation_mode") != "linear",
        allow_empty=True,
    )
    slam_series = _source_series(
        items,
        episode_name=episode_name,
        timestamp_key="slam_tf",
        payload=_payload_slam_tf,
        include=lambda stamp: stamp.get("interpolation_mode") != "linear",
        allow_empty=True,
    )
    action_series = _source_series(
        items,
        episode_name=episode_name,
        timestamp_key="base_action",
        payload=_payload_base_action,
        include=lambda stamp: stamp.get("interpolation_mode") != "linear",
        allow_empty=True,
    )

    aligned_items: list[Json] = []
    mode_counts: dict[str, Counter[str]] = {
        "base_state": Counter(),
        "base_height": Counter(),
        "slam_tf": Counter(),
        "base_action": Counter(),
    }
    for item in items:
        idx = _frame_index(item, episode_name)
        field = f"episode={episode_name} frame={idx}"
        target_t_ns = _integer(_item_timestamps(item, field).get("sample_monotonic_ns"), f"{field}.timestamps.sample_monotonic_ns")
        raw_base_state_stamp = _stream_stamp(item, "base_state", field)
        if raw_base_state_stamp.get("interpolation_mode") == "linear":
            base_state, base_state_stamp = _align_recorded_linear_payload(
                item,
                episode_name=episode_name,
                timestamp_key="base_state",
                payload=_payload_base_state,
                max_support_delta_ns=base_state_max_delta_ns,
            )
        else:
            if not state_series:
                raise RuntimeError(f"{field} has no reconstructable base_state source samples")
            base_state, base_state_stamp = _align_from_series(
                state_series,
                target_t_ns=target_t_ns,
                interpolate=_interpolate_base_state,
                stream_name="base_state",
                max_support_delta_ns=base_state_max_delta_ns,
            )
        raw_height_stamp = _stream_stamp(item, "base_height", field)
        if raw_height_stamp.get("interpolation_mode") == "linear":
            base_height, base_height_stamp = _align_recorded_linear_payload(
                item,
                episode_name=episode_name,
                timestamp_key="base_height",
                payload=_payload_base_height,
                max_support_delta_ns=base_height_max_delta_ns,
            )
        else:
            if not height_series:
                raise RuntimeError(f"{field} has no reconstructable base_height source samples")
            base_height, base_height_stamp = _align_from_series(
                height_series,
                target_t_ns=target_t_ns,
                interpolate=_interpolate_height,
                stream_name="base_height",
                max_support_delta_ns=base_height_max_delta_ns,
            )
        raw_slam_stamp = _stream_stamp(item, "slam_tf", field)
        if raw_slam_stamp.get("interpolation_mode") == "linear":
            slam_tf, slam_tf_stamp = _align_recorded_linear_payload(
                item,
                episode_name=episode_name,
                timestamp_key="slam_tf",
                payload=_payload_slam_tf,
                max_support_delta_ns=slam_tf_max_delta_ns,
            )
        else:
            if not slam_series:
                raise RuntimeError(f"{field} has no reconstructable slam_tf source samples")
            slam_tf, slam_tf_stamp = _align_from_series(
                slam_series,
                target_t_ns=target_t_ns,
                interpolate=_interpolate_slam_pose,
                stream_name="slam_tf",
                max_support_delta_ns=slam_tf_max_delta_ns,
            )
        raw_action_stamp = _stream_stamp(item, "base_action", field)
        if raw_action_stamp.get("interpolation_mode") == "linear":
            base_action, base_action_stamp = _align_recorded_linear_action(
                item, episode_name=episode_name, max_support_delta_ns=base_action_max_delta_ns
            )
        else:
            if not action_series:
                raise RuntimeError(f"{field} has no reconstructable base_action source samples")
            base_action, base_action_stamp = _align_from_series(
                action_series,
                target_t_ns=target_t_ns,
                interpolate=_interpolate_action,
                stream_name="base_action",
                max_support_delta_ns=base_action_max_delta_ns,
            )
        _append_image_aligned_waist_target(base_action, item, field)

        aligned_item = copy.deepcopy(dict(item))
        aligned_base = _mapping(_mapping(aligned_item["states"], f"{field}.states")["base"], f"{field}.states.base")
        aligned_base["world_pose_interpolated"] = base_state["world_pose"]
        aligned_base["velocity_interpolated"] = base_state["velocity"]
        aligned_base["height_interpolated"] = base_height["height"]
        aligned_base["slam_map_pose_interpolated"] = slam_tf["slam_map_pose"]
        aligned_action = _mapping(_mapping(aligned_item["actions"], f"{field}.actions")["base"], f"{field}.actions.base")
        aligned_action["interpolated"] = base_action
        aligned_timestamps = _mapping(aligned_item["timestamps"], f"{field}.timestamps")
        aligned_timestamps["base_state_interpolated"] = base_state_stamp
        aligned_timestamps["base_height_interpolated"] = base_height_stamp
        aligned_timestamps["slam_tf_interpolated"] = slam_tf_stamp
        aligned_timestamps["base_action_interpolated"] = base_action_stamp
        for name, stamp in zip(mode_counts, (base_state_stamp, base_height_stamp, slam_tf_stamp, base_action_stamp)):
            mode_counts[name][str(stamp["interpolation_mode"])] += 1
        aligned_items.append(aligned_item)

    output_episode_dir.mkdir(parents=True)
    for name in ("colors", "depths", "audios", "rerun.rrd", "validation.json"):
        source_path = source_episode_dir / name
        if source_path.exists() or source_path.is_symlink():
            relative_source = os.path.relpath(source_path, start=output_episode_dir)
            (output_episode_dir / name).symlink_to(relative_source, target_is_directory=source_path.is_dir())
    (output_episode_dir / "data.json").symlink_to(os.path.relpath(source_data_path, start=output_episode_dir))
    aligned_document = copy.deepcopy(dict(document))
    info = _mapping(aligned_document.get("info"), f"episode={episode_name}.info")
    info["mobile_offline_alignment"] = {
        "schema_version": 2,
        "source_data_json": str(source_data_path),
        "policy": "strict_linear_or_recorded_online_linear",
        "source_sample_constraints": "no nearest fallback and no extrapolation",
        "max_support_delta_ms": {
            "base_state": base_state_max_delta_ns / 1e6,
            "base_height": base_height_max_delta_ns / 1e6,
            "slam_tf": slam_tf_max_delta_ns / 1e6,
            "base_action": base_action_max_delta_ns / 1e6,
        },
        "frame_count": len(aligned_items),
        "mode_counts": {name: dict(counter) for name, counter in mode_counts.items()},
    }
    aligned_document["data"] = aligned_items
    (output_episode_dir / "data.mobile_aligned.json").write_text(
        json.dumps(aligned_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "episode": episode_name,
        "source_episode_dir": str(source_episode_dir),
        "frame_count": len(aligned_items),
        "mode_counts": {name: dict(counter) for name, counter in mode_counts.items()},
    }


def align_task_dir(
    task_dir: Path,
    output_task_dir: Path,
    *,
    overwrite: bool = False,
    base_state_max_delta_ms: float = _DEFAULT_BASE_STATE_MAX_DELTA_MS,
    base_height_max_delta_ms: float = _DEFAULT_BASE_HEIGHT_MAX_DELTA_MS,
    slam_tf_max_delta_ms: float = _DEFAULT_SLAM_TF_MAX_DELTA_MS,
    base_action_max_delta_ms: float = _DEFAULT_BASE_ACTION_MAX_DELTA_MS,
) -> Json:
    task_dir = task_dir.resolve()
    output_task_dir = output_task_dir.resolve()
    if not task_dir.is_dir():
        raise ValueError(f"--task-dir is not a directory: {task_dir}")
    if task_dir == output_task_dir:
        raise ValueError("--output-task-dir must differ from --task-dir")
    thresholds_ms = {
        "base_state": float(base_state_max_delta_ms),
        "base_height": float(base_height_max_delta_ms),
        "slam_tf": float(slam_tf_max_delta_ms),
        "base_action": float(base_action_max_delta_ms),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds_ms.values()):
        raise ValueError(f"all max support deltas must be finite and positive, got {thresholds_ms}")
    thresholds_ns = {name: int(value * 1e6) for name, value in thresholds_ms.items()}
    episodes = sorted(path for path in task_dir.glob("episode_*") if (path / "data.json").is_file())
    if not episodes:
        raise RuntimeError(f"no episode_xxxx/data.json found under: {task_dir}")
    if output_task_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output task directory already exists: {output_task_dir}; rerun with --overwrite")
        shutil.rmtree(output_task_dir)
    output_task_dir.mkdir(parents=True)
    records = [
        _align_episode(
            episode.resolve(),
            output_task_dir / episode.name,
            base_state_max_delta_ns=thresholds_ns["base_state"],
            base_height_max_delta_ns=thresholds_ns["base_height"],
            slam_tf_max_delta_ns=thresholds_ns["slam_tf"],
            base_action_max_delta_ns=thresholds_ns["base_action"],
        )
        for episode in episodes
    ]
    report = {
        "schema_version": 2,
        "source_task_dir": str(task_dir),
        "output_task_dir": str(output_task_dir),
        "policy": "strict_linear_or_recorded_online_linear",
        "max_support_delta_ms": thresholds_ms,
        "episode_count": len(records),
        "frame_count": sum(record["frame_count"] for record in records),
        "episodes": records,
    }
    (output_task_dir / "alignment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[ALIGN_MOBILE_EPISODES] episodes={report['episode_count']} frames={report['frame_count']} "
        f"output={output_task_dir}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, type=Path, help="输入 episode 目录，可为清洗后的任务目录")
    parser.add_argument("--output-task-dir", required=True, type=Path, help="输出对齐任务目录")
    parser.add_argument("--base-state-max-delta-ms", type=float, default=_DEFAULT_BASE_STATE_MAX_DELTA_MS)
    parser.add_argument("--base-height-max-delta-ms", type=float, default=_DEFAULT_BASE_HEIGHT_MAX_DELTA_MS)
    parser.add_argument("--slam-tf-max-delta-ms", type=float, default=_DEFAULT_SLAM_TF_MAX_DELTA_MS)
    parser.add_argument("--base-action-max-delta-ms", type=float, default=_DEFAULT_BASE_ACTION_MAX_DELTA_MS)
    parser.add_argument("--overwrite", action="store_true", help="明确允许删除同名输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    align_task_dir(
        args.task_dir,
        args.output_task_dir,
        overwrite=args.overwrite,
        base_state_max_delta_ms=args.base_state_max_delta_ms,
        base_height_max_delta_ms=args.base_height_max_delta_ms,
        slam_tf_max_delta_ms=args.slam_tf_max_delta_ms,
        base_action_max_delta_ms=args.base_action_max_delta_ms,
    )


if __name__ == "__main__":
    main()
