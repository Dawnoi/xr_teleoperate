"""Pure 4-DoF workspace governor for mobile manipulation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

import numpy as np

from core.control.arm_workspace_safety import (
    clamp_wrist_pose_to_box,
    clamp_wrist_pose_to_tapered_workspace,
)


@dataclass(frozen=True)
class WorkspaceGovernorConfig:
    mode: str
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    tapered: Mapping[str, float]
    comfort_margin: float = 0.04
    hysteresis: float = 0.02
    prediction_horizon: float = 0.4
    max_forward_speed: float = 0.20
    max_base_yaw_rate: float = 0.60
    max_column_command: float = 1.0
    column_speed_mps: float = 0.10
    max_torso_yaw_rate: float = 0.50
    column_minimum: float = 0.0
    column_maximum: float = 0.42
    torso_minimum: float = -2.60
    torso_maximum: float = 2.60


@dataclass(frozen=True)
class WorkspaceGovernorResult:
    command: np.ndarray
    active: bool
    feasible: bool
    recovery_delta_m: float


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def solve_box_qp(hessian, gradient, lower, upper) -> np.ndarray:
    """Solve a strictly convex box QP by exact active-set enumeration."""
    gradient = _finite_vector(gradient, len(np.asarray(gradient).reshape(-1)), "QP gradient")
    size = gradient.size
    hessian = np.asarray(hessian, dtype=float)
    lower = _finite_vector(lower, size, "QP lower bound")
    upper = _finite_vector(upper, size, "QP upper bound")
    if hessian.shape != (size, size) or not np.all(np.isfinite(hessian)):
        raise ValueError("QP Hessian has an invalid shape or non-finite value")
    if np.any(lower > upper):
        raise ValueError("QP lower bounds exceed upper bounds")
    hessian = 0.5 * (hessian + hessian.T)
    if float(np.min(np.linalg.eigvalsh(hessian))) <= 1e-10:
        raise ValueError("QP Hessian must be strictly positive definite")

    best = None
    best_cost = np.inf
    tolerance = 1e-9
    for status in product((-1, 0, 1), repeat=size):
        fixed = np.asarray([i for i, value in enumerate(status) if value], dtype=int)
        free = np.asarray([i for i, value in enumerate(status) if not value], dtype=int)
        candidate = np.empty(size, dtype=float)
        for index, value in enumerate(status):
            if value == -1:
                candidate[index] = lower[index]
            elif value == 1:
                candidate[index] = upper[index]
        if free.size:
            reduced = hessian[np.ix_(free, free)]
            if float(np.min(np.linalg.eigvalsh(reduced))) <= 1e-10:
                raise ValueError("QP active-set Hessian is not strictly positive definite")
            rhs = -gradient[free]
            if fixed.size:
                rhs -= hessian[np.ix_(free, fixed)] @ candidate[fixed]
            candidate[free] = np.linalg.solve(reduced, rhs)
        if np.any(candidate < lower - tolerance) or np.any(candidate > upper + tolerance):
            continue
        candidate = np.clip(candidate, lower, upper)
        cost = float(0.5 * candidate @ hessian @ candidate + gradient @ candidate)
        if cost < best_cost:
            best = candidate.copy()
            best_cost = cost
    if best is None:
        raise RuntimeError("box QP has no feasible finite solution")
    return best


def workspace_command_delta(command, *, dt: float, column_speed_mps: float) -> np.ndarray:
    """Integrate a manual [vx, wz, column, torso] command in IK coordinates."""
    vx, wz, column, torso = _finite_vector(command, 4, "body command")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("body command dt must be positive and finite")
    if not np.isfinite(column_speed_mps) or column_speed_mps <= 0.0:
        raise ValueError("column speed must be positive and finite")
    yaw = float((wz + torso) * dt)
    cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
    result = np.eye(4)
    result[:3, :3] = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    if abs(float(wz)) > 1e-9:
        result[0, 3] = float(vx / wz * np.sin(wz * dt))
        result[1, 3] = float(vx / wz * (1.0 - np.cos(wz * dt)))
    else:
        result[0, 3] = float(vx * dt)
    result[2, 3] = float(column * column_speed_mps * dt)
    return result


class WorkspaceGovernor:
    """Keep enabled wrist targets inside a shared comfortable IK workspace."""

    def __init__(self, config: WorkspaceGovernorConfig) -> None:
        self.config = config
        if config.mode not in ("box", "tapered"):
            raise ValueError("workspace mode must be box or tapered")
        self._active = False
        self._last_command = np.zeros(4, dtype=float)
        self._validate_config()

    def _validate_config(self) -> None:
        for name in (
            "comfort_margin", "hysteresis", "prediction_horizon", "max_forward_speed",
            "max_base_yaw_rate", "max_column_command", "column_speed_mps", "max_torso_yaw_rate",
        ):
            value = float(getattr(self.config, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.config.column_minimum >= self.config.column_maximum:
            raise ValueError("column limits must be ordered")
        if self.config.torso_minimum >= self.config.torso_maximum:
            raise ValueError("torso limits must be ordered")

    def reset(self) -> None:
        self._active = False
        self._last_command[:] = 0.0

    def _clamp_position(self, position: np.ndarray, margin: float) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = _finite_vector(position, 3, "workspace target")
        if self.config.mode == "box":
            minimum = _finite_vector(self.config.workspace_min, 3, "workspace minimum") + margin
            maximum = _finite_vector(self.config.workspace_max, 3, "workspace maximum") - margin
            if np.any(maximum <= minimum):
                raise ValueError("workspace margin collapses the box workspace")
            return clamp_wrist_pose_to_box(pose, minimum, maximum)[0][:3, 3]
        parameters = {name: float(value) for name, value in self.config.tapered.items()}
        for name in ("z_min", "x_min"):
            parameters[name] += margin
        for name in ("z_max", "x_max_low", "x_max_high", "y_max_low", "y_max_high"):
            parameters[name] -= margin
        return clamp_wrist_pose_to_tapered_workspace(pose, **parameters)[0][:3, 3]

    def step(self, targets, enabled, nominal_command, *, column_position: float, torso_yaw: float, dt: float) -> WorkspaceGovernorResult:
        nominal = _finite_vector(nominal_command, 4, "nominal body command")
        if not all(np.isfinite(value) for value in (column_position, torso_yaw, dt)) or dt <= 0.0:
            raise ValueError("mobile state must be finite and dt positive")
        active = {
            side: _finite_vector(np.asarray(target, dtype=float)[:3, 3], 3, f"{side} target")
            for side, target in targets.items() if bool(enabled.get(side, False))
        }
        if not active:
            self.reset()
            self._last_command[:] = nominal
            return WorkspaceGovernorResult(nominal.copy(), False, True, 0.0)
        deltas = [self._clamp_position(point, self.config.comfort_margin) - point for point in active.values()]
        if not self._active and any(np.linalg.norm(delta) > 1e-9 for delta in deltas):
            self._active = True
        recovery_margin = self.config.comfort_margin + self.config.hysteresis
        recovery = [self._clamp_position(point, recovery_margin) - point for point in active.values()]
        if self._active and not any(np.linalg.norm(delta) > 1e-9 for delta in recovery):
            self._active = False
        if not self._active:
            self._last_command[:] = nominal
            return WorkspaceGovernorResult(nominal.copy(), False, True, 0.0)
        nominal_weight = np.diag([7.0, 7.0, 3.0, 1.0])
        smooth_weight = np.diag([2.0, 2.0, 1.0, 1.5])
        hessian = nominal_weight + smooth_weight + np.eye(4) * 1e-8
        gradient = -(nominal_weight @ nominal + smooth_weight @ self._last_command)
        recovery_weight = np.diag([20.0, 20.0, 10.0])
        for point, delta in zip(active.values(), recovery):
            x, y, _ = point
            mapping = np.array([[-1.0, y, 0.0, y], [0.0, -x, 0.0, -x], [0.0, 0.0, -self.config.column_speed_mps, 0.0]])
            desired = delta / self.config.prediction_horizon + mapping @ nominal
            hessian += mapping.T @ recovery_weight @ mapping
            gradient -= mapping.T @ recovery_weight @ desired
        hessian[3, 3] += 4.0
        gradient[3] -= 4.0 * (-1.5 * float(torso_yaw))
        lower = np.array([-self.config.max_forward_speed, -self.config.max_base_yaw_rate, -self.config.max_column_command, -self.config.max_torso_yaw_rate])
        upper = -lower
        column_scale = self.config.column_speed_mps * self.config.prediction_horizon
        lower[2] = max(lower[2], (self.config.column_minimum - float(column_position)) / column_scale)
        upper[2] = min(upper[2], (self.config.column_maximum - float(column_position)) / column_scale)
        lower[3] = max(lower[3], (self.config.torso_minimum - float(torso_yaw)) / self.config.prediction_horizon)
        upper[3] = min(upper[3], (self.config.torso_maximum - float(torso_yaw)) / self.config.prediction_horizon)
        command = solve_box_qp(hessian, gradient, lower, upper)
        self._last_command[:] = command
        return WorkspaceGovernorResult(command, True, True, max(float(np.linalg.norm(delta)) for delta in recovery))
