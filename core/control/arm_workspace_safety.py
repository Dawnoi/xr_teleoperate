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


def _lerp(a, b, t):
    return float(a) + (float(b) - float(a)) * float(t)


def clamp_point_to_tapered_workspace(
    point,
    z_min,
    z_max,
    x_min,
    x_max_low,
    x_max_high,
    y_max_low,
    y_max_high,
):
    """
    Clamp a 3D point into a z-dependent forward workspace:

      z in [z_min, z_max]
      x in [x_min, x_max(z)]
      y in [-y_max(z), y_max(z)]

    where x_max(z) and y_max(z) expand linearly from the low-z slice to the
    high-z slice, forming an "inverted trapezoid" / tapered prism.
    """
    point = np.asarray(point, dtype=float).reshape(3)

    if z_max <= z_min:
        z = float(z_min)
        x_max = float(x_max_low)
        y_max = float(y_max_low)
    else:
        z = float(np.clip(point[2], z_min, z_max))
        t = (z - float(z_min)) / (float(z_max) - float(z_min))
        x_max = _lerp(x_max_low, x_max_high, t)
        y_max = _lerp(y_max_low, y_max_high, t)

    clamped = np.array(
        [
            np.clip(float(point[0]), float(x_min), x_max),
            np.clip(float(point[1]), -y_max, y_max),
            z,
        ],
        dtype=float,
    )
    return clamped, bool(np.any(np.abs(clamped - point) > 1e-12))


def clamp_point_to_asymmetric_tapered_workspace(
    point,
    z_min,
    z_max,
    x_min,
    x_max_low,
    x_max_high,
    y_min_low,
    y_min_high,
    y_max_low,
    y_max_high,
):
    """Clamp into a tapered prism with independently configurable y bounds.

    ``y_min`` and ``y_max`` are interpolated over z independently.  This is
    required for a cross-body workspace: for example, the left wrist may have
    more room towards negative y than towards positive y.
    """
    point = np.asarray(point, dtype=float).reshape(3)
    values = np.array(
        [z_min, z_max, x_min, x_max_low, x_max_high, y_min_low, y_min_high, y_max_low, y_max_high],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("asymmetric tapered workspace parameters must be finite")
    if z_max <= z_min:
        raise ValueError("asymmetric tapered workspace requires z_max > z_min")

    z = float(np.clip(point[2], z_min, z_max))
    t = (z - float(z_min)) / (float(z_max) - float(z_min))
    x_max = _lerp(x_max_low, x_max_high, t)
    y_min = _lerp(y_min_low, y_min_high, t)
    y_max = _lerp(y_max_low, y_max_high, t)
    if x_max < float(x_min):
        raise ValueError("asymmetric tapered workspace has x_max below x_min")
    if y_max < y_min:
        raise ValueError("asymmetric tapered workspace has y_max below y_min")

    clamped = np.array(
        [
            np.clip(float(point[0]), float(x_min), x_max),
            np.clip(float(point[1]), y_min, y_max),
            z,
        ],
        dtype=float,
    )
    return clamped, bool(np.any(np.abs(clamped - point) > 1e-12))


def clamp_wrist_pose_to_tapered_workspace(
    pose,
    z_min,
    z_max,
    x_min,
    x_max_low,
    x_max_high,
    y_max_low,
    y_max_high,
):
    clamped_pose = np.asarray(pose, dtype=float).copy()
    clamped_point, was_clamped = clamp_point_to_tapered_workspace(
        clamped_pose[:3, 3],
        z_min,
        z_max,
        x_min,
        x_max_low,
        x_max_high,
        y_max_low,
        y_max_high,
    )
    clamped_pose[:3, 3] = clamped_point
    return clamped_pose, was_clamped


def clamp_wrist_pose_to_asymmetric_tapered_workspace(
    pose,
    z_min,
    z_max,
    x_min,
    x_max_low,
    x_max_high,
    y_min_low,
    y_min_high,
    y_max_low,
    y_max_high,
):
    clamped_pose = np.asarray(pose, dtype=float).copy()
    if clamped_pose.shape != (4, 4) or not np.all(np.isfinite(clamped_pose)):
        raise ValueError("workspace pose must be a finite 4x4 matrix")
    clamped_point, was_clamped = clamp_point_to_asymmetric_tapered_workspace(
        clamped_pose[:3, 3],
        z_min,
        z_max,
        x_min,
        x_max_low,
        x_max_high,
        y_min_low,
        y_min_high,
        y_max_low,
        y_max_high,
    )
    clamped_pose[:3, 3] = clamped_point
    return clamped_pose, was_clamped


def clamp_wrist_pose_to_workspace(pose, workspace):
    """Clamp one wrist pose using one explicit workspace specification.

    Required schema:
      ``{mode, workspace_min, workspace_max, tapered}``

    Tapered workspaces require independent ``y_min_low/high`` and
    ``y_max_low/high`` values.  Shared legacy workspaces are normalized to
    this schema before reaching this function.
    """
    if not isinstance(workspace, dict):
        raise ValueError("workspace specification must be a dict")
    mode = str(workspace.get("mode", ""))
    if mode == "box":
        if "workspace_min" not in workspace or "workspace_max" not in workspace:
            raise ValueError("box workspace requires workspace_min and workspace_max")
        return clamp_wrist_pose_to_box(pose, workspace["workspace_min"], workspace["workspace_max"])
    if mode != "tapered":
        raise ValueError("workspace mode must be box or tapered")
    tapered = workspace.get("tapered")
    if not isinstance(tapered, dict):
        raise ValueError("tapered workspace requires a tapered parameter dict")
    names = (
        "z_min", "z_max", "x_min", "x_max_low", "x_max_high",
        "y_min_low", "y_min_high", "y_max_low", "y_max_high",
    )
    missing = [name for name in names if name not in tapered]
    if missing:
        raise ValueError(f"tapered workspace is missing parameters: {missing}")
    return clamp_wrist_pose_to_asymmetric_tapered_workspace(
        pose,
        *(float(tapered[name]) for name in names),
    )


def clamp_dual_wrist_poses_to_side_workspaces(left_pose, right_pose, side_workspaces):
    """Clamp each wrist with its own explicit workspace definition."""
    if not isinstance(side_workspaces, dict) or set(side_workspaces) != {"left", "right"}:
        raise ValueError("side workspaces must contain exactly left and right")
    left_clamped_pose, left_clamped = clamp_wrist_pose_to_workspace(left_pose, side_workspaces["left"])
    right_clamped_pose, right_clamped = clamp_wrist_pose_to_workspace(right_pose, side_workspaces["right"])
    return left_clamped_pose, right_clamped_pose, (left_clamped or right_clamped)


def clamp_dual_wrist_poses_to_tapered_workspace(
    left_pose,
    right_pose,
    z_min,
    z_max,
    x_min,
    x_max_low,
    x_max_high,
    y_max_low,
    y_max_high,
):
    left_clamped_pose, left_clamped = clamp_wrist_pose_to_tapered_workspace(
        left_pose,
        z_min,
        z_max,
        x_min,
        x_max_low,
        x_max_high,
        y_max_low,
        y_max_high,
    )
    right_clamped_pose, right_clamped = clamp_wrist_pose_to_tapered_workspace(
        right_pose,
        z_min,
        z_max,
        x_min,
        x_max_low,
        x_max_high,
        y_max_low,
        y_max_high,
    )
    return left_clamped_pose, right_clamped_pose, (left_clamped or right_clamped)
