from __future__ import annotations

from typing import Any

from inference.pose_transform import matrix_to_pose9_rot6d


def wrist_pose_to_debug_sample(pose: Any) -> dict[str, list[float]]:
    pose9 = matrix_to_pose9_rot6d(pose)
    return {
        "xyz": [float(value) for value in pose9[:3]],
        "rot6d": [float(value) for value in pose9[3:]],
    }
