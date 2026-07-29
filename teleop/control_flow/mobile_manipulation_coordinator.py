"""Coordinate 4-DoF mobile workspace recovery before the legacy arm IK."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.control.workspace_governor import WorkspaceGovernor, WorkspaceGovernorResult, workspace_command_delta
from core.input.base import MotionIntent


@dataclass(frozen=True)
class MobileStateSample:
    global_from_ik: np.ndarray
    monotonic_ns: int
    column_position: float
    torso_yaw: float


@dataclass(frozen=True)
class CoordinatorResult:
    motion_intent_for_ik: MotionIntent
    final_body_command: np.ndarray
    waist_yaw_target: float
    governor_active: bool
    workspace_feasible: bool
    recovery_delta_m: float


class G1DIkFrameKinematics:
    """Evaluate the G1D AGV-to-torso chain from the repository URDF."""

    def __init__(self, *, torso_from_ik: np.ndarray | None = None) -> None:
        import pinocchio as pin

        repo_root = Path(__file__).resolve().parents[2]
        urdf_path = repo_root / "assets" / "g1_d" / "g1_d.urdf"
        if not urdf_path.is_file():
            raise RuntimeError(f"missing G1D kinematics model: {urdf_path}")
        self._pin = pin
        self._model = pin.buildModelFromUrdf(str(urdf_path))
        self._data = self._model.createData()
        self._agv_frame = self._model.getFrameId("AGV_link")
        self._torso_frame = self._model.getFrameId("torso_link")
        self._joint_ids = {name: self._model.getJointId(name) for name in ("LZ_mt_Joint", "LZ_it_Joint", "torso_Joint")}
        self._torso_from_ik = np.eye(4) if torso_from_ik is None else np.asarray(torso_from_ik, dtype=float).copy()
        if self._torso_from_ik.shape != (4, 4) or not np.all(np.isfinite(self._torso_from_ik)):
            raise ValueError("torso_from_ik must be a finite 4x4 transform")

    def agv_from_ik(self, *, column_position: float, torso_yaw: float) -> np.ndarray:
        """Return AGV_link from the legacy G1_29 IK pelvis frame."""
        if not np.isfinite(column_position) or not np.isfinite(torso_yaw):
            raise ValueError("column position and torso yaw must be finite")
        q = np.zeros(self._model.nq)
        for name in ("LZ_mt_Joint", "LZ_it_Joint"):
            q[int(self._model.idx_qs[self._joint_ids[name]])] = float(column_position) * 0.5
        q[int(self._model.idx_qs[self._joint_ids["torso_Joint"]])] = float(torso_yaw)
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacements(self._model, self._data)
        agv_from_torso = self._data.oMf[self._agv_frame].inverse() * self._data.oMf[self._torso_frame]
        agv_from_ik = np.eye(4)
        agv_from_ik[:3, :3] = agv_from_torso.rotation
        agv_from_ik[:3, 3] = agv_from_torso.translation
        return agv_from_ik @ self._torso_from_ik

    def global_from_ik(self, *, odom_world_from_agv: np.ndarray, column_position: float, torso_yaw: float) -> np.ndarray:
        odom_world_from_agv = np.asarray(odom_world_from_agv, dtype=float)
        if odom_world_from_agv.shape != (4, 4) or not np.all(np.isfinite(odom_world_from_agv)):
            raise ValueError("odom_world_from_agv must be a finite 4x4 transform")
        return odom_world_from_agv @ self.agv_from_ik(
            column_position=column_position,
            torso_yaw=torso_yaw,
        )

def legacy_g1_29_torso_from_ik_urdf() -> np.ndarray:
    """Read the fixed ``torso_link <- pelvis`` registration used by G1_29 IK."""
    import pinocchio as pin

    repo_root = Path(__file__).resolve().parents[2]
    legacy_urdf_path = repo_root / "assets" / "g1" / "g1_body29_hand14.urdf"
    if not legacy_urdf_path.is_file():
        raise RuntimeError(f"missing legacy G1_29 IK model: {legacy_urdf_path}")
    legacy_model = pin.buildModelFromUrdf(str(legacy_urdf_path))
    legacy_data = legacy_model.createData()
    pin.framesForwardKinematics(legacy_model, legacy_data, pin.neutral(legacy_model))
    pin.updateFramePlacements(legacy_model, legacy_data)
    pelvis_frame = _required_frame_id(legacy_model, "pelvis", "legacy G1_29 IK")
    torso_frame = _required_frame_id(legacy_model, "torso_link", "legacy G1_29 IK")
    pelvis_from_torso = legacy_data.oMf[pelvis_frame].inverse() * legacy_data.oMf[torso_frame]
    torso_from_pelvis = pelvis_from_torso.inverse()
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(torso_from_pelvis.rotation, dtype=float)
    transform[:3, 3] = np.asarray(torso_from_pelvis.translation, dtype=float)
    return _validated_rigid_transform(transform, "legacy G1_29 torso_link<-pelvis")


def _required_frame_id(model, frame_name: str, model_label: str) -> int:
    frame_id = int(model.getFrameId(frame_name))
    if frame_id >= int(model.nframes) or model.frames[frame_id].name != frame_name:
        raise RuntimeError(f"{model_label} is missing required frame: {frame_name}")
    return frame_id


def _validated_rigid_transform(value: np.ndarray, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-12, rtol=0.0):
        raise ValueError(f"{label} has invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f"{label} rotation determinant must be +1")
    return transform.copy()


def column_position_from_raw_height(
    *,
    raw_height: float,
    raw_minimum: float,
    raw_maximum: float,
    column_travel_m: float,
) -> float:
    """Map the calibrated DDS height reading to physical column travel in metres."""
    values = {
        "raw_height": raw_height,
        "raw_minimum": raw_minimum,
        "raw_maximum": raw_maximum,
        "column_travel_m": column_travel_m,
    }
    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if raw_maximum <= raw_minimum:
        raise ValueError("raw_maximum must be greater than raw_minimum")
    if column_travel_m <= 0.0:
        raise ValueError("column_travel_m must be positive")
    if raw_height < raw_minimum or raw_height > raw_maximum:
        raise RuntimeError("MOBILE_COLUMN_STATE_OUT_OF_RANGE")
    return float(column_travel_m) * (float(raw_height) - float(raw_minimum)) / (
        float(raw_maximum) - float(raw_minimum)
    )


class MobileManipulationCoordinator:
    def __init__(self, governor: WorkspaceGovernor, *, state_timeout_sec: float) -> None:
        if not np.isfinite(state_timeout_sec) or state_timeout_sec <= 0.0:
            raise ValueError("state_timeout_sec must be positive and finite")
        self._governor = governor
        self._state_timeout_ns = int(state_timeout_sec * 1e9)
        self._anchors_global_from_ik: dict[str, np.ndarray] = {}
        self._targets_global: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._governor.reset()
        self._anchors_global_from_ik.clear()
        self._targets_global.clear()

    def step(self, *, motion_intent: MotionIntent, enabled: dict[str, bool], rising: dict[str, bool], nominal_body_command, mobile_state: MobileStateSample, now_monotonic_ns: int, dt: float, home_active: bool, stop_active: bool) -> CoordinatorResult:
        if motion_intent.kind != "pose":
            raise ValueError("mobile_ik_qp requires a pose MotionIntent")
        global_from_ik = np.asarray(mobile_state.global_from_ik, dtype=float)
        if global_from_ik.shape != (4, 4) or not np.all(np.isfinite(global_from_ik)):
            raise ValueError("mobile state global_from_ik must be a finite 4x4 transform")
        if int(now_monotonic_ns) - int(mobile_state.monotonic_ns) > self._state_timeout_ns:
            raise RuntimeError("MOBILE_STATE_STALE")
        if home_active or stop_active:
            self.reset()
            return self._result(motion_intent, np.zeros(4), mobile_state, WorkspaceGovernorResult(np.zeros(4), False, True, 0.0), dt)
        for side in ("left", "right"):
            if not enabled.get(side, False):
                self._anchors_global_from_ik.pop(side, None)
                self._targets_global.pop(side, None)
        ik_from_global = np.linalg.inv(global_from_ik)
        nominal = np.asarray(nominal_body_command, dtype=float).reshape(4)
        manual_delta_global = global_from_ik @ workspace_command_delta(nominal, dt=dt, column_speed_mps=self._governor.config.column_speed_mps) @ ik_from_global
        target_mode = str(motion_intent.metadata.get("mobile_target_frame", "relative_ik" if motion_intent.source == "xr" else "ik"))
        raw_global = {}
        for side, pose in (("left", motion_intent.left_wrist_pose), ("right", motion_intent.right_wrist_pose)):
            if rising.get(side, False):
                self._anchors_global_from_ik[side] = global_from_ik.copy()
            if target_mode == "global":
                raw_global[side] = np.asarray(pose, dtype=float).copy()
            elif target_mode == "ik":
                raw_global[side] = global_from_ik @ np.asarray(pose, dtype=float)
            elif target_mode == "relative_ik":
                anchor = self._anchors_global_from_ik.get(side)
                if enabled.get(side, False) and anchor is None:
                    raise RuntimeError(f"missing {side} Grip takeover anchor")
                if anchor is not None:
                    if enabled.get(side, False):
                        anchor = manual_delta_global @ anchor
                        self._anchors_global_from_ik[side] = anchor
                    raw_global[side] = anchor @ np.asarray(pose, dtype=float)
            else:
                raise ValueError("mobile_target_frame must be relative_ik, ik, or global")
        self._targets_global.update(raw_global)
        local_targets = {side: ik_from_global @ target for side, target in self._targets_global.items()}
        governor_result = self._governor.step(local_targets, enabled, nominal, column_position=mobile_state.column_position, torso_yaw=mobile_state.torso_yaw, dt=dt)
        left = local_targets.get("left", np.asarray(motion_intent.left_wrist_pose, dtype=float))
        right = local_targets.get("right", np.asarray(motion_intent.right_wrist_pose, dtype=float))
        local_intent = MotionIntent(kind="pose", left_wrist_pose=left, right_wrist_pose=right, gripper_q=motion_intent.gripper_q, timestamp=motion_intent.timestamp, frame_index=motion_intent.frame_index, source=motion_intent.source, metadata=motion_intent.metadata)
        return self._result(local_intent, governor_result.command, mobile_state, governor_result, dt)

    @staticmethod
    def _result(intent, command, state, governor_result, dt: float) -> CoordinatorResult:
        command = np.asarray(command, dtype=float).copy()
        return CoordinatorResult(intent, command, float(state.torso_yaw + command[3] * float(dt)), governor_result.active, governor_result.feasible, governor_result.recovery_delta_m)
