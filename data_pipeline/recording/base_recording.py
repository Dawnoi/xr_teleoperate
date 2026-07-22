"""Helpers for recording aligned mobile-base state/action data."""

from __future__ import annotations

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
            "source_topic": str(velocity.get("source_topic") or ""),
        },
    }
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
        "source": str(aligned_base_action.get("source") or ""),
    }


__all__ = [
    "build_base_action_record",
    "build_base_state_record",
]
