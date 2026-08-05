"""Base-link kinematic queries shared by recording and mobile inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class MobileBaseKinematicsProvider:
    """Evaluate base-link wrist and TCP poses from measured G1-D joints."""

    def __init__(
        self,
        *,
        g1d_kinematics: Any,
        legacy_wrist_pose_solver: Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]],
        dex1_tcp_fk: Any = None,
    ) -> None:
        if g1d_kinematics is None:
            raise ValueError("mobile base kinematics requires G1D kinematics")
        if not callable(legacy_wrist_pose_solver):
            raise ValueError("mobile base kinematics requires a legacy wrist pose solver")
        self._g1d_kinematics = g1d_kinematics
        self._legacy_wrist_pose_solver = legacy_wrist_pose_solver
        self._dex1_tcp_fk = dex1_tcp_fk

    @property
    def dex1_tcp_available(self) -> bool:
        return self._dex1_tcp_fk is not None

    def base_link_from_ik(self, *, column_height_m: float, waist_yaw_rad: float) -> np.ndarray:
        """Return base_link <- legacy IK frame under the active G1-D convention."""
        result = self._g1d_kinematics.agv_from_ik(
            column_position=float(column_height_m),
            torso_yaw=float(waist_yaw_rad),
        )
        return self._matrix4x4(result, "base_link<-legacy IK")

    def wrist_poses_base_link(
        self,
        arm_ik: Any,
        arm_q: np.ndarray,
        column_height_m: float,
        waist_yaw_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return base_link <- left/right legacy wrist poses."""
        base_link_from_ik = self.base_link_from_ik(
            column_height_m=column_height_m,
            waist_yaw_rad=waist_yaw_rad,
        )
        left_ik_pose, right_ik_pose = self._legacy_wrist_pose_solver(arm_ik, arm_q)
        return (
            self._matrix4x4(base_link_from_ik @ left_ik_pose, "base_link<-left wrist"),
            self._matrix4x4(base_link_from_ik @ right_ik_pose, "base_link<-right wrist"),
        )

    def dex1_tcp_poses_base_link(
        self,
        arm_q: np.ndarray,
        column_height_m: float,
        waist_yaw_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return base_link <- left/right calibrated Dex1 TCP poses."""
        return self._require_dex1_tcp_fk().compute_tcp_poses(
            arm_q,
            column_height_m,
            waist_yaw_rad,
        )

    def legacy_ik_ee_pose_from_tcp_target(
        self,
        side: str,
        tcp_target: np.ndarray,
        *,
        column_height_m: float,
        waist_yaw_rad: float,
    ) -> np.ndarray:
        """Convert a base-link TCP target into the legacy G1_29 IK EEF frame."""
        legacy_ee_base_link = self._require_dex1_tcp_fk().legacy_g1_29_ik_ee_pose_from_tcp_target(
            side,
            tcp_target,
        )
        ik_from_base_link = np.linalg.inv(
            self.base_link_from_ik(
                column_height_m=column_height_m,
                waist_yaw_rad=waist_yaw_rad,
            )
        )
        return self._matrix4x4(
            ik_from_base_link @ legacy_ee_base_link,
            f"legacy G1_29 IK EEF target: side={side}",
        )

    def _require_dex1_tcp_fk(self):
        if self._dex1_tcp_fk is None:
            raise RuntimeError("Dex1 TCP FK is not initialized")
        return self._dex1_tcp_fk

    @staticmethod
    def _matrix4x4(value: np.ndarray, label: str) -> np.ndarray:
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"{label} must be a finite 4x4 transform")
        return matrix.copy()
