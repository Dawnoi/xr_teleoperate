import numpy as np


def project_point_to_front_hemisphere(point, center, radius):
    """
    Project a 3D point onto the closed front-hemisphere workspace:

        { p | ||p - center|| <= radius, (p - center)[0] >= 0 }

    The +x axis is treated as the robot-forward direction in the arm IK frame.

    Returns:
        projected_point: np.ndarray shape (3,)
        was_clamped: bool
    """
    point = np.asarray(point, dtype=float).reshape(3)
    center = np.asarray(center, dtype=float).reshape(3)
    radius = float(radius)

    if radius <= 0.0:
        return center.copy(), True

    rel = point - center
    if rel[0] >= 0.0 and np.linalg.norm(rel) <= radius:
        return point.copy(), False

    projected_rel = rel.copy()
    if projected_rel[0] < 0.0:
        projected_rel[0] = 0.0

    norm = np.linalg.norm(projected_rel)
    if norm > radius and norm > 1e-12:
        projected_rel = projected_rel / norm * radius

    return center + projected_rel, True


def clamp_wrist_pose_to_front_hemisphere(pose, center, radius):
    """
    Clamp only the translation component of a 4x4 wrist pose to the front
    hemisphere workspace while preserving orientation.
    """
    clamped_pose = np.asarray(pose, dtype=float).copy()
    projected_point, was_clamped = project_point_to_front_hemisphere(
        clamped_pose[:3, 3],
        center,
        radius,
    )
    clamped_pose[:3, 3] = projected_point
    return clamped_pose, was_clamped


def clamp_dual_wrist_poses_to_front_hemisphere(left_pose, right_pose, center, radius):
    left_clamped_pose, left_clamped = clamp_wrist_pose_to_front_hemisphere(
        left_pose,
        center,
        radius,
    )
    right_clamped_pose, right_clamped = clamp_wrist_pose_to_front_hemisphere(
        right_pose,
        center,
        radius,
    )
    return left_clamped_pose, right_clamped_pose, (left_clamped or right_clamped)


def clamp_point_to_aabb(point, min_bound, max_bound):
    """
    Clamp a 3D point into an axis-aligned box.

    Returns:
        clamped_point: np.ndarray shape (3,)
        was_clamped: bool
    """
    point = np.asarray(point, dtype=float).reshape(3)
    min_bound = np.asarray(min_bound, dtype=float).reshape(3)
    max_bound = np.asarray(max_bound, dtype=float).reshape(3)
    clamped = np.minimum(np.maximum(point, min_bound), max_bound)
    return clamped, bool(np.any(np.abs(clamped - point) > 1e-12))


def clamp_wrist_pose_to_box(pose, min_bound, max_bound):
    """
    Clamp only the translation component of a 4x4 wrist pose into an
    axis-aligned box while preserving orientation.
    """
    clamped_pose = np.asarray(pose, dtype=float).copy()
    clamped_point, was_clamped = clamp_point_to_aabb(
        clamped_pose[:3, 3],
        min_bound,
        max_bound,
    )
    clamped_pose[:3, 3] = clamped_point
    return clamped_pose, was_clamped


def clamp_dual_wrist_poses_to_box(left_pose, right_pose, min_bound, max_bound):
    left_clamped_pose, left_clamped = clamp_wrist_pose_to_box(
        left_pose,
        min_bound,
        max_bound,
    )
    right_clamped_pose, right_clamped = clamp_wrist_pose_to_box(
        right_pose,
        min_bound,
        max_bound,
    )
    return left_clamped_pose, right_clamped_pose, (left_clamped or right_clamped)
