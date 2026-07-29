"""Helpers for recording aligned mobile-base state/action data."""

from __future__ import annotations

import copy
import math


def _finite_float(value, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return result


def _finite_float_list(values, field_name: str, expected_len: int) -> list[float]:
    if values is None:
        raise ValueError(f"{field_name} is required")
    if len(values) != int(expected_len):
        raise ValueError(f"{field_name} must have length {expected_len}, got {len(values)}")
    return [_finite_float(value, f"{field_name}[{idx}]") for idx, value in enumerate(values)]


def build_base_state_record(aligned_base_state: dict, aligned_base_height: dict | None):
    if not isinstance(aligned_base_state, dict):
        raise TypeError("aligned_base_state must be a dict")
    world_pose = aligned_base_state.get("world_pose")
    velocity = aligned_base_state.get("velocity")
    if not isinstance(world_pose, dict):
        raise ValueError("aligned_base_state.world_pose is required")
    if not isinstance(velocity, dict):
        raise ValueError("aligned_base_state.velocity is required")

    record = {
        "world_pose": {
            "x": _finite_float(world_pose.get("x"), "world_pose.x"),
            "y": _finite_float(world_pose.get("y"), "world_pose.y"),
            "z": _finite_float(world_pose.get("z"), "world_pose.z"),
            "yaw": _finite_float(world_pose.get("yaw"), "world_pose.yaw"),
            "quat_xyzw": _finite_float_list(world_pose.get("quat_xyzw"), "world_pose.quat_xyzw", 4),
            "frame_id": str(world_pose.get("frame_id") or ""),
            "source_topic": str(world_pose.get("source_topic") or ""),
        },
        "velocity": {
            "vx": _finite_float(velocity.get("vx"), "velocity.vx"),
            "vy": _finite_float(velocity.get("vy"), "velocity.vy"),
            "vz": _finite_float(velocity.get("vz"), "velocity.vz"),
            "wz": _finite_float(velocity.get("wz"), "velocity.wz"),
            "frame_id": str(velocity.get("frame_id") or ""),
            "linear_unit": str(velocity.get("linear_unit") or ""),
            "angular_unit": str(velocity.get("angular_unit") or ""),
            "source_topic": str(velocity.get("source_topic") or ""),
        },
    }
    slam_map_pose = aligned_base_state.get("slam_map_pose")
    if slam_map_pose is not None:
        if not isinstance(slam_map_pose, dict):
            raise ValueError("aligned_base_state.slam_map_pose must be a dict when present")
        record["slam_map_pose"] = {
            "x": _finite_float(slam_map_pose.get("x"), "slam_map_pose.x"),
            "y": _finite_float(slam_map_pose.get("y"), "slam_map_pose.y"),
            "z": _finite_float(slam_map_pose.get("z"), "slam_map_pose.z"),
            "yaw": _finite_float(slam_map_pose.get("yaw"), "slam_map_pose.yaw"),
            "quat_xyzw": _finite_float_list(slam_map_pose.get("quat_xyzw"), "slam_map_pose.quat_xyzw", 4),
            "frame_id": str(slam_map_pose.get("frame_id") or ""),
            "child_frame_id": str(slam_map_pose.get("child_frame_id") or ""),
            "source_child_frame_id": str(slam_map_pose.get("source_child_frame_id") or ""),
            "source_to_base_link_identity_assumed": bool(
                slam_map_pose.get("source_to_base_link_identity_assumed", False)
            ),
            "source_topic": str(slam_map_pose.get("source_topic") or ""),
            "tf_age_ms": _finite_float(slam_map_pose.get("tf_age_ms"), "slam_map_pose.tf_age_ms"),
        }
        interpolation_support = slam_map_pose.get("interpolation_support")
        if interpolation_support is None:
            record["slam_map_pose"].update(
                {
                    "tf_header_stamp_ns": int(slam_map_pose.get("tf_header_stamp_ns")),
                    "tf_lookup_wall_time_ns": int(slam_map_pose.get("tf_lookup_wall_time_ns")),
                    "tf_lookup_monotonic_ns": int(slam_map_pose.get("tf_lookup_monotonic_ns")),
                }
            )
            source_chain = slam_map_pose.get("source_chain")
            if source_chain is not None:
                if not isinstance(source_chain, dict):
                    raise ValueError("slam_map_pose.source_chain must be a dict when present")
                record["slam_map_pose"]["source_chain"] = source_chain
        else:
            if not isinstance(interpolation_support, dict):
                raise ValueError("slam_map_pose.interpolation_support must be a dict when present")
            semantics = str(slam_map_pose.get("tf_age_ms_semantics") or "")
            if semantics != "max_support_age_ms":
                raise ValueError(
                    "interpolated slam_map_pose.tf_age_ms must declare tf_age_ms_semantics='max_support_age_ms'"
                )
            record["slam_map_pose"]["tf_age_ms_semantics"] = semantics
            record["slam_map_pose"]["interpolation_support"] = copy.deepcopy(interpolation_support)
    if aligned_base_height is not None:
        height = aligned_base_height.get("height")
        if not isinstance(height, dict):
            raise ValueError("aligned_base_height.height is required")
        record["height"] = {
            "z": _finite_float(height.get("z"), "height.z"),
            "source_topic": str(height.get("source_topic") or ""),
        }
    return record


def build_base_action_record(aligned_base_action: dict):
    if not isinstance(aligned_base_action, dict):
        raise TypeError("aligned_base_action must be a dict")
    return {
        "vx_cmd": _finite_float(aligned_base_action.get("vx_cmd"), "base_action.vx_cmd"),
        "vy_cmd": _finite_float(aligned_base_action.get("vy_cmd"), "base_action.vy_cmd"),
        "wz_cmd": _finite_float(aligned_base_action.get("wz_cmd"), "base_action.wz_cmd"),
        "z_cmd": _finite_float(aligned_base_action.get("z_cmd"), "base_action.z_cmd"),
        "frame_id": str(aligned_base_action.get("frame_id") or ""),
        "linear_unit": "m/s",
        "angular_unit": "rad/s",
        "z_cmd_unit": "normalized",
        "source": str(aligned_base_action.get("source") or ""),
    }


__all__ = [
    "build_base_action_record",
    "build_base_state_record",
]
