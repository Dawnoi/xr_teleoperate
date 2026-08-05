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
        data = self._model.createData()
        self._pin.forwardKinematics(self._model, data, q)
        self._pin.updateFramePlacements(self._model, data)
        agv_from_torso = data.oMf[self._agv_frame].inverse() * data.oMf[self._torso_frame]
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
        raise RuntimeError(
            "MOBILE_COLUMN_STATE_OUT_OF_RANGE "
            f"raw_height_y={raw_height:.9f} "
            f"raw_minimum_y={raw_minimum:.9f} "
            f"raw_maximum_y={raw_maximum:.9f}"
        )
    return float(column_travel_m) * (float(raw_height) - float(raw_minimum)) / (
        float(raw_maximum) - float(raw_minimum)
    )


class LegacyWorkspaceGovernorCoordinator:
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


@dataclass(frozen=True)
class WholeBodyCoordinatorResult:
    arm_q_target: np.ndarray
    final_body_command: np.ndarray
    qp_velocity: np.ndarray
    active: bool


class MobileManipulationCoordinator:
    """Measured-state closed-loop G1-D whole-body velocity-QP coordinator.

    ``mobile_ik_qp`` used to invoke a workspace governor and then the legacy
    G1_29 inverse kinematics.  It now owns the complete kinematic allocation:
    two Dex1 TCP tasks, both column joints and planar base motion. The torso
    yaw joint is deliberately outside this controller: it remains an
    independently commanded hardware joint and is not part of the WBC state,
    optimization, or output.
    The legacy IK frame is used only to interpret the existing XR input pose.
    """

    def __new__(cls, *args, **kwargs):
        if not args:
            return super().__new__(cls)
        if len(args) != 1 or not isinstance(args[0], WorkspaceGovernor):
            raise TypeError(
                "MobileManipulationCoordinator accepts either the legacy "
                "WorkspaceGovernor positional constructor or the keyword-only WBC constructor"
            )
        if set(kwargs) != {"state_timeout_sec"}:
            raise TypeError(
                "legacy MobileManipulationCoordinator requires exactly "
                "WorkspaceGovernor and state_timeout_sec"
            )
        return LegacyWorkspaceGovernorCoordinator(
            args[0],
            state_timeout_sec=kwargs["state_timeout_sec"],
        )

    _ARM_JOINT_NAMES = (
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    )
    _TCP_FROM_LEGACY_IK_EE = np.array(
        [[1.0, 0.0, 0.0, 0.0701], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )

    def __init__(
        self,
        *,
        state_timeout_sec: float,
        column_travel_m: float,
        command_horizon_sec: float = 0.1,
        max_position_lead_rad: float = 0.12,
        enable_collision_avoidance: bool = True,
    ) -> None:
        if not np.isfinite(state_timeout_sec) or state_timeout_sec <= 0.0:
            raise ValueError("mobile WBC state_timeout_sec must be positive and finite")
        if not np.isfinite(column_travel_m) or column_travel_m <= 0.0:
            raise ValueError("mobile WBC column_travel_m must be positive and finite")
        if not np.isfinite(command_horizon_sec) or command_horizon_sec <= 0.0:
            raise ValueError("mobile WBC command_horizon_sec must be positive and finite")
        if not np.isfinite(max_position_lead_rad) or max_position_lead_rad <= 0.0:
            raise ValueError("mobile WBC max_position_lead_rad must be positive and finite")
        from core.control.g1d_kinematic_wbc import G1DKinematicWBC

        repo_root = Path(__file__).resolve().parents[2]
        model_path = repo_root / "assets" / "g1_d" / "g1_d_wbc.xml"
        if not model_path.is_file():
            raise RuntimeError(f"mobile WBC model is missing: {model_path}")
        self._wbc = G1DKinematicWBC(
            model_path=model_path,
            control_frequency=30.0,
            enable_collision_avoidance=bool(enable_collision_avoidance),
        )
        self._state_timeout_ns = int(state_timeout_sec * 1e9)
        self._column_travel_m = float(column_travel_m)
        self._command_horizon_sec = float(command_horizon_sec)
        self._max_position_lead_rad = float(max_position_lead_rad)
        self._ik_kinematics = G1DIkFrameKinematics(
            torso_from_ik=legacy_g1_29_torso_from_ik_urdf(),
        )
        self._anchors_world_from_ik: dict[str, np.ndarray] = {}
        self._takeover_alignment_world: dict[str, np.ndarray] = {}
        self._targets_world: dict[str, np.ndarray] = {}
        self._validate_tcp_sites()

    def _validate_tcp_sites(self) -> None:
        import mujoco

        data = mujoco.MjData(self._wbc.model)
        mujoco.mj_forward(self._wbc.model, data)
        for side in ("left", "right"):
            wrist = self._wbc.model.body(f"{side}_wrist_yaw_link").id
            tcp = self._wbc.model.site(f"{side}_gripper_tip_site").id
            wrist_rotation = data.xmat[wrist].reshape(3, 3)
            local_translation = wrist_rotation.T @ (data.site_xpos[tcp] - data.xpos[wrist])
            local_rotation = wrist_rotation.T @ data.site_xmat[tcp].reshape(3, 3)
            if not np.allclose(local_translation, np.array([0.1201, 0.0, 0.0]), atol=1e-6, rtol=0.0):
                raise RuntimeError(f"mobile WBC {side} TCP calibration differs from wrist local +X 0.1201 m: {local_translation}")
            if not np.allclose(local_rotation, np.eye(3), atol=1e-6, rtol=0.0):
                raise RuntimeError(f"mobile WBC {side} TCP rotation is not parallel to wrist flange")

    def reset(self) -> None:
        self._anchors_world_from_ik.clear()
        self._takeover_alignment_world.clear()
        self._targets_world.clear()

    def step(
        self,
        *,
        motion_intent: MotionIntent,
        enabled: dict[str, bool],
        rising: dict[str, bool],
        settling: dict[str, bool],
        nominal_body_command: np.ndarray,
        mobile_state: MobileStateSample,
        arm_q: np.ndarray,
        now_monotonic_ns: int,
        dt: float,
        home_active: bool,
        stop_active: bool,
    ) -> WholeBodyCoordinatorResult:
        if motion_intent.kind != "pose":
            raise ValueError("mobile_ik_qp WBC requires a pose MotionIntent")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("mobile WBC dt must be positive and finite")
        if int(now_monotonic_ns) - int(mobile_state.monotonic_ns) > self._state_timeout_ns:
            raise RuntimeError("MOBILE_STATE_STALE")
        measured_qpos = self._measured_qpos(mobile_state, arm_q)
        if home_active or stop_active:
            self.reset()
            return self._hold_result(arm_q, mobile_state, enabled)
        if any(bool(settling.get(side, False)) for side in ("left", "right")):
            # The legacy arm path waits two frames after a Grip rising edge before
            # using XR deltas.  WBC must obey the same barrier: otherwise it can
            # freeze a pre-rebase XR pose as a world target and jump both arms.
            self.reset()
            return self._hold_result(arm_q, mobile_state, enabled)

        world_from_ik = np.asarray(mobile_state.global_from_ik, dtype=float)
        if world_from_ik.shape != (4, 4) or not np.all(np.isfinite(world_from_ik)):
            raise ValueError("mobile WBC global_from_ik must be a finite 4x4 transform")
        world_from_base = self._world_from_base(mobile_state, world_from_ik)
        rebased = self._update_targets(
            motion_intent,
            enabled,
            rising,
            world_from_ik,
            world_from_base,
            measured_qpos,
        )
        if rebased:
            return self._hold_result(arm_q, mobile_state, enabled)
        target_matrices = {
            "left_gripper_tip_site": self._targets_world["left"],
            "right_gripper_tip_site": self._targets_world["right"],
        }
        _, velocity = self._wbc.solve(measured_qpos, target_matrices, dt=dt)
        arm_target = self._lookahead_arm_target(measured_qpos, velocity)
        final_body_command = self._resolve_base_command(velocity, nominal_body_command)
        return WholeBodyCoordinatorResult(arm_target, final_body_command, velocity.copy(), bool(any(enabled.values())))

    def _hold_result(
        self,
        arm_q: np.ndarray,
        mobile_state: MobileStateSample,
        enabled: dict[str, bool],
    ) -> WholeBodyCoordinatorResult:
        arm_q = np.asarray(arm_q, dtype=float).reshape(-1)
        if arm_q.shape != (14,) or not np.all(np.isfinite(arm_q)):
            raise ValueError("mobile WBC hold arm_q must be a finite 14D vector")
        return WholeBodyCoordinatorResult(
            arm_q.copy(),
            np.zeros(3),
            np.zeros(self._wbc.model.nv),
            bool(any(enabled.values())),
        )

    def _measured_qpos(self, state: MobileStateSample, arm_q: np.ndarray) -> np.ndarray:
        arm_q = np.asarray(arm_q, dtype=float).reshape(-1)
        if arm_q.shape != (14,) or not np.all(np.isfinite(arm_q)):
            raise ValueError("mobile WBC arm_q must be a finite 14D measured vector")
        qpos = self._wbc.home_qpos
        # The physical torso is held by the independent arm controller. Its
        # measured angle must not enter WBC, otherwise the QP is still reading
        # and allocating the waist yaw despite having no output authority.
        base_from_ik = self._ik_kinematics.agv_from_ik(column_position=float(state.column_position), torso_yaw=0.0)
        world_from_base = np.asarray(state.global_from_ik, dtype=float) @ np.linalg.inv(base_from_ik)
        yaw = float(np.arctan2(world_from_base[1, 0], world_from_base[0, 0]))
        for name, value in (("joint_x", world_from_base[0, 3]), ("joint_y", world_from_base[1, 3]), ("joint_th", yaw), ("LZ_mt_Joint", state.column_position * 0.5), ("LZ_it_Joint", state.column_position * 0.5)):
            qpos[int(self._wbc.model.joint(name).qposadr[0])] = float(value)
        for name, value in zip(self._ARM_JOINT_NAMES, arm_q):
            qpos[int(self._wbc.model.joint(name).qposadr[0])] = float(value)
        if not np.all(np.isfinite(qpos)):
            raise RuntimeError("mobile WBC measured qpos contains non-finite values")
        return qpos

    def _world_from_base(self, state: MobileStateSample, world_from_ik: np.ndarray) -> np.ndarray:
        base_from_ik = self._ik_kinematics.agv_from_ik(column_position=float(state.column_position), torso_yaw=0.0)
        return world_from_ik @ np.linalg.inv(base_from_ik)

    def _update_targets(self, intent, enabled, rising, world_from_ik, world_from_base, measured_qpos) -> bool:
        target_mode = str(intent.metadata.get("mobile_target_frame", "relative_ik" if intent.source == "xr" else "ik"))
        site_targets = self._site_world_poses(measured_qpos)
        rebased = False
        for side, pose in (("left", intent.left_wrist_pose), ("right", intent.right_wrist_pose)):
            if not enabled.get(side, False):
                self._anchors_world_from_ik.pop(side, None)
                self._takeover_alignment_world.pop(side, None)
                self._targets_world[side] = site_targets[side]
                continue
            pose = np.asarray(pose, dtype=float)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                raise ValueError(f"mobile WBC {side} wrist target must be a finite 4x4 transform")
            if target_mode == "global":
                target_legacy_ee_world = pose
            elif target_mode == "ik":
                target_legacy_ee_world = world_from_ik @ pose
            elif target_mode == "relative_ik":
                if bool(rising.get(side, False)):
                    self._anchors_world_from_ik.pop(side, None)
                    self._takeover_alignment_world.pop(side, None)
                anchor = self._anchors_world_from_ik.get(side)
                alignment = self._takeover_alignment_world.get(side)
                if anchor is None or alignment is None:
                    # The first post-settle XR pose establishes an input origin.
                    # Align it to the measured TCP; do not interpret it as a
                    # world command before the XR reference has been rebased.
                    anchor = world_from_ik.copy()
                    raw_target_world = anchor @ pose @ self._TCP_FROM_LEGACY_IK_EE
                    self._anchors_world_from_ik[side] = anchor
                    self._takeover_alignment_world[side] = (
                        site_targets[side] @ np.linalg.inv(raw_target_world)
                    )
                    self._targets_world[side] = site_targets[side]
                    rebased = True
                    continue
                target_legacy_ee_world = alignment @ anchor @ pose
            else:
                raise ValueError("mobile_target_frame must be relative_ik, ik, or global")
            self._targets_world[side] = target_legacy_ee_world @ self._TCP_FROM_LEGACY_IK_EE
        return rebased

    def _site_world_poses(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        import mujoco

        data = mujoco.MjData(self._wbc.model)
        data.qpos[:] = qpos
        mujoco.mj_forward(self._wbc.model, data)
        poses = {}
        for side in ("left", "right"):
            site = self._wbc.model.site(f"{side}_gripper_tip_site").id
            matrix = np.eye(4)
            matrix[:3, :3] = data.site_xmat[site].reshape(3, 3)
            matrix[:3, 3] = data.site_xpos[site]
            poses[side] = matrix
        return poses

    def _lookahead_arm_target(self, qpos: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        return np.asarray([self._lookahead_joint_target(qpos, velocity, name) for name in self._ARM_JOINT_NAMES], dtype=float)

    def _lookahead_joint_target(self, qpos: np.ndarray, velocity: np.ndarray, name: str) -> float:
        joint = self._wbc.model.joint(name)
        qpos_address, dof_address = int(joint.qposadr[0]), int(joint.dofadr[0])
        measured = float(qpos[qpos_address])
        lead = float(np.clip(float(velocity[dof_address]) * self._command_horizon_sec, -self._max_position_lead_rad, self._max_position_lead_rad))
        target = measured + lead
        if self._wbc.model.jnt_limited[joint.id]:
            lower, upper = self._wbc.model.jnt_range[joint.id]
            target = float(np.clip(target, lower, upper))
        return float(target)

    def _resolve_base_command(self, velocity: np.ndarray, nominal: np.ndarray) -> np.ndarray:
        nominal = np.asarray(nominal, dtype=float).reshape(-1)
        if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
            raise ValueError("mobile WBC nominal body command must be a finite [vx, wz, z]")
        if np.any(np.abs(nominal[:3]) > 1e-9):
            return nominal.copy()
        base = self._wbc.base_velocity_command
        left_lift = float(velocity[int(self._wbc.model.joint("LZ_mt_Joint").dofadr[0])])
        right_lift = float(velocity[int(self._wbc.model.joint("LZ_it_Joint").dofadr[0])])
        column_normalized = float(np.clip((left_lift + right_lift) / 0.10, -1.0, 1.0))
        return np.array([base[0], base[1], column_normalized], dtype=float)
