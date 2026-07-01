from __future__ import annotations

"""pi0.5 协议纯数据转换层。

这个模块只处理：
- observation payload 打包
- action response 归一化 / 校验
- 20D 双臂 action 的 observation-delta 重映射

不依赖主循环、不依赖 DDS、不依赖控制器。
"""

from typing import Any

import numpy as np

from inference.transforms.pose_transform import matrix_to_pose7_xyzw, matrix_to_pose9_rot6d, pose9_rot6d_to_matrix

PI05_IMAGE_ROLES = ("third_front", "head_fpv", "left_hand", "right_hand")


def _finite_vector(values: Any, expected_len: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != int(expected_len):
        raise ValueError(f"{label} expected length {expected_len}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or Inf")
    return arr.copy()


def _pose9_to_list(pose: Any) -> list[float]:
    arr = _finite_vector(pose, 9, "pose9")
    return [float(v) for v in arr.tolist()]


def _image_roles(images: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {role: "" for role in PI05_IMAGE_ROLES}
    for role, value in (images or {}).items():
        role_str = str(role)
        if not role_str:
            raise ValueError("pi05 observation image role must be non-empty")
        if role_str not in PI05_IMAGE_ROLES:
            raise ValueError(
                f"unsupported pi05 image role: {role_str!r}; "
                f"expected one of {', '.join(PI05_IMAGE_ROLES)}"
            )
        if value is None:
            continue
        payload[role_str] = value
    return payload


def build_pi05_observation_payload(
    *,
    images: dict[str, Any],
    poses_left: list,
    grippers_left: list,
    poses_right: list,
    grippers_right: list,
    prompt: str = "",
    upstream_timestamp_start: float | None = None,
    **extra,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "observation",
        "images": _image_roles(images),
        "poses_left": [_pose9_to_list(pose) for pose in poses_left],
        "grippers_left": [[float(np.asarray(item, dtype=float).reshape(-1)[0])] for item in grippers_left],
        "poses_right": [_pose9_to_list(pose) for pose in poses_right],
        "grippers_right": [[float(np.asarray(item, dtype=float).reshape(-1)[0])] for item in grippers_right],
    }
    if str(prompt).strip():
        payload["prompt"] = str(prompt)
    if upstream_timestamp_start is not None:
        payload["upstream_timestamp_start"] = float(upstream_timestamp_start)
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def validate_pi05_action_sequence(resp: dict, expected_dim: int) -> np.ndarray:
    resp = normalize_pi05_action_response(resp)
    if not isinstance(resp, dict):
        raise TypeError("action response must be an object")
    if resp.get("type") != "action_sequence":
        raise RuntimeError(f"unexpected response type: {resp.get('type')}")
    raw_actions = resp.get("actions")
    if raw_actions is None:
        raw_actions = resp.get("action_sequence", [])
    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] <= 0 or actions.shape[1] != int(expected_dim):
        raise ValueError(f"invalid action shape: {actions.shape}, expected (*, {expected_dim})")
    if not np.all(np.isfinite(actions)):
        raise ValueError("action sequence contains NaN or Inf")
    return actions


def normalize_pi05_action_response(resp: dict) -> dict:
    if not isinstance(resp, dict) or "actions" in resp or "action_sequence" in resp:
        return resp
    if resp.get("type") != "action":
        return resp
    left = resp.get("action_l") or []
    right = resp.get("action_r") or []
    n = min(len(left), len(right))
    if n <= 0:
        return resp
    actions = []
    for idx in range(n):
        left_step = _finite_vector(left[idx], 10, f"action_l[{idx}]")
        right_step = _finite_vector(right[idx], 10, f"action_r[{idx}]")
        actions.append(np.concatenate([left_step, right_step]).astype(np.float32))
    out = dict(resp)
    out["type"] = "action_sequence"
    out["actions"] = np.asarray(actions, dtype=np.float32).tolist()
    return out


def remap_pi05_action_sequence_observation_delta(
    actions: np.ndarray,
    *,
    observation_anchor20: np.ndarray,
    robot_anchor20: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 20:
        raise ValueError(f"invalid actions shape: {arr.shape}, expected (*, 20)")
    obs_anchor = np.asarray(observation_anchor20, dtype=np.float32).reshape(20)
    robot_anchor = np.asarray(robot_anchor20, dtype=np.float32).reshape(20)
    if not np.all(np.isfinite(arr)):
        raise ValueError("actions must contain only finite values")
    if not np.all(np.isfinite(obs_anchor)):
        raise ValueError("observation_anchor20 must contain only finite values")
    if not np.all(np.isfinite(robot_anchor)):
        raise ValueError("robot_anchor20 must contain only finite values")

    def _remap_arm(action10: np.ndarray, obs10: np.ndarray, robot10: np.ndarray) -> np.ndarray:
        target = np.asarray(action10, dtype=np.float32).reshape(10).copy()
        obs = np.asarray(obs10, dtype=np.float32).reshape(10)
        robot = np.asarray(robot10, dtype=np.float32).reshape(10)

        target_obs_matrix = np.asarray(pose9_rot6d_to_matrix(obs[:9]), dtype=np.float32)
        target_action_matrix = np.asarray(pose9_rot6d_to_matrix(target[:9]), dtype=np.float32)
        target_robot_matrix = np.asarray(pose9_rot6d_to_matrix(robot[:9]), dtype=np.float32)

        remapped_matrix = np.eye(4, dtype=np.float32)
        remapped_matrix[:3, 3] = (
            target_robot_matrix[:3, 3] + (target_action_matrix[:3, 3] - target_obs_matrix[:3, 3])
        )
        remapped_matrix[:3, :3] = (
            target_action_matrix[:3, :3] @ target_obs_matrix[:3, :3].T @ target_robot_matrix[:3, :3]
        )
        target[:9] = np.asarray(matrix_to_pose9_rot6d(remapped_matrix), dtype=np.float32)
        target[9] = float(np.clip(robot[9] + (target[9] - obs[9]), 0.0, 0.1))
        return target

    out = np.zeros_like(arr, dtype=np.float32)
    for idx, step in enumerate(arr):
        out[idx, :10] = _remap_arm(step[:10], obs_anchor[:10], robot_anchor[:10])
        out[idx, 10:20] = _remap_arm(step[10:20], obs_anchor[10:20], robot_anchor[10:20])
    return out


def pi05_action_sequence_to_pose7_chunks(actions: Any, arm_side: str) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None]:
    if arm_side not in {"left", "right", "both"}:
        raise ValueError(f"unsupported arm_side: {arm_side!r}")
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] != 20:
        raise ValueError(f"invalid actions shape: {arr.shape}, expected (*, 20)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("actions must contain only finite values")

    def _arm10_to_pose8(step10: Any, label: str) -> np.ndarray:
        step = _finite_vector(step10, 10, label)
        matrix = pose9_rot6d_to_matrix(step[:9])
        pose7 = matrix_to_pose7_xyzw(matrix)
        return np.asarray([*pose7, float(step[9])], dtype=np.float64)

    left_steps = None
    right_steps = None
    if arm_side in {"left", "both"}:
        left_steps = [_arm10_to_pose8(step[:10], f"actions[{idx}].left") for idx, step in enumerate(arr)]
    if arm_side in {"right", "both"}:
        right_steps = [_arm10_to_pose8(step[10:20], f"actions[{idx}].right") for idx, step in enumerate(arr)]
    return left_steps, right_steps


def pose9_to_matrix(pose9: Any) -> np.ndarray:
    return np.asarray(pose9_rot6d_to_matrix(pose9), dtype=np.float32)


def matrix_to_pose9(matrix: Any) -> list[float]:
    return [float(v) for v in matrix_to_pose9_rot6d(matrix)]
