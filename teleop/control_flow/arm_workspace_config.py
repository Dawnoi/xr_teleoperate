"""Build and validate explicit arm workspace specifications."""

from __future__ import annotations

import numpy as np


def build_arm_side_workspaces(args, *, workspace_mode, workspace_min, workspace_max, tapered_workspace_params):
    """Build the explicit left/right workspace specs shared by IK and QP."""
    if args.arm_workspace_layout == "shared":
        tapered = dict(tapered_workspace_params)
        tapered["y_min_low"] = -float(tapered["y_max_low"])
        tapered["y_min_high"] = -float(tapered["y_max_high"])
        shared = {
            "mode": workspace_mode,
            "workspace_min": np.asarray(workspace_min, dtype=float).copy(),
            "workspace_max": np.asarray(workspace_max, dtype=float).copy(),
            "tapered": tapered,
        }
        return {
            "left": shared,
            "right": {
                **shared,
                "workspace_min": shared["workspace_min"].copy(),
                "workspace_max": shared["workspace_max"].copy(),
                "tapered": dict(tapered),
            },
        }

    if workspace_mode == "box":
        raw_specs = {
            "left": (args.left_arm_workspace_min, args.left_arm_workspace_max),
            "right": (args.right_arm_workspace_min, args.right_arm_workspace_max),
        }
        result = {}
        for side, (minimum, maximum) in raw_specs.items():
            if minimum is None or maximum is None:
                raise ValueError(f"per_arm box workspace requires --{side}-arm-workspace-min and --{side}-arm-workspace-max")
            minimum = np.asarray(minimum, dtype=float)
            maximum = np.asarray(maximum, dtype=float)
            if minimum.shape != (3,) or maximum.shape != (3,) or not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
                raise ValueError(f"{side} box workspace bounds must be three finite values")
            if np.any(maximum <= minimum):
                raise ValueError(f"{side} box workspace maximum must exceed minimum on every axis")
            result[side] = {"mode": "box", "workspace_min": minimum, "workspace_max": maximum, "tapered": {}}
        return result

    names = (
        "z_min", "z_max", "x_min", "x_max_low", "x_max_high",
        "y_min_low", "y_min_high", "y_max_low", "y_max_high",
    )
    result = {}
    for side, values in (("left", args.left_arm_workspace_tapered), ("right", args.right_arm_workspace_tapered)):
        if values is None:
            raise ValueError(f"per_arm tapered workspace requires --{side}-arm-workspace-tapered with nine values")
        values = np.asarray(values, dtype=float)
        if values.shape != (9,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{side} tapered workspace must contain nine finite values")
        tapered = dict(zip(names, values.tolist()))
        if tapered["z_max"] <= tapered["z_min"]:
            raise ValueError(f"{side} tapered workspace requires z_max > z_min")
        if tapered["x_max_low"] < tapered["x_min"] or tapered["x_max_high"] < tapered["x_min"]:
            raise ValueError(f"{side} tapered workspace x_max must be at least x_min")
        if tapered["y_max_low"] <= tapered["y_min_low"] or tapered["y_max_high"] <= tapered["y_min_high"]:
            raise ValueError(f"{side} tapered workspace y_max must exceed y_min")
        result[side] = {
            "mode": "tapered",
            "workspace_min": np.zeros(3, dtype=float),
            "workspace_max": np.zeros(3, dtype=float),
            "tapered": tapered,
        }
    return result
