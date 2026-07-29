"""Strict G1D full-body FK for the calibrated Dex1.1 TCP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin


G1D_URDF_ROOT_FRAME = "AGV_link"
BASE_LINK_FRAME = "base_link"
LEFT_WRIST_FRAME = "left_wrist_yaw_link"
RIGHT_WRIST_FRAME = "right_wrist_yaw_link"
LEFT_TCP_FRAME = "left_dex1_tcp"
RIGHT_TCP_FRAME = "right_dex1_tcp"
LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINT_NAMES = tuple(name.replace("left_", "right_", 1) for name in LEFT_ARM_JOINT_NAMES)
COLUMN_JOINT_NAMES = ("LZ_mt_Joint", "LZ_it_Joint")
WAIST_YAW_JOINT_NAME = "torso_Joint"
TCP_TRANSLATION_M = np.array([0.1201, 0.0, 0.0], dtype=float)
TCP_RPY_RAD = np.zeros(3, dtype=float)


class Dex1TcpFkProvider:
    """Compute ``base_link -> Dex1 TCP`` from measured or target G1D joints.

    The G1D URDF root is ``AGV_link``. The active mobile collection calibration
    explicitly defines ``base_link == AGV_link`` with an identity transform; this
    provider records output under the externally defined ``base_link`` semantic.
    """

    def __init__(self, robot_fk_urdf: str | Path, eef_model_urdf: str | Path) -> None:
        self.robot_fk_urdf = _required_file(robot_fk_urdf, "G1D FK URDF")
        self.eef_model_urdf = _required_file(eef_model_urdf, "Dex1.1 model URDF")
        self._wrist_from_tcp = _validated_wrist_to_tcp_transform()
        self.model = pin.buildModelFromUrdf(str(self.robot_fk_urdf), pin.JointModelFreeFlyer())
        self._neutral_q = pin.neutral(self.model)
        self._root_frame_id = self._required_frame_id(G1D_URDF_ROOT_FRAME)
        self._left_wrist_frame_id = self._required_frame_id(LEFT_WRIST_FRAME)
        self._right_wrist_frame_id = self._required_frame_id(RIGHT_WRIST_FRAME)
        self._arm_q_indices = self._required_joint_q_indices(LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES)
        self._column_q_indices = self._required_joint_q_indices(COLUMN_JOINT_NAMES)
        self._waist_yaw_q_index = self._required_joint_q_indices((WAIST_YAW_JOINT_NAME,))[0]

    def compute_wrist_poses(
        self,
        arm_qpos: np.ndarray,
        column_height_m: float,
        waist_yaw_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        data = self._forward(arm_qpos, column_height_m, waist_yaw_rad)
        return (
            _validated_transform(data.oMf[self._left_wrist_frame_id], LEFT_WRIST_FRAME),
            _validated_transform(data.oMf[self._right_wrist_frame_id], RIGHT_WRIST_FRAME),
        )

    def compute_tcp_poses(
        self,
        arm_qpos: np.ndarray,
        column_height_m: float,
        waist_yaw_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        left_wrist, right_wrist = self.compute_wrist_poses(arm_qpos, column_height_m, waist_yaw_rad)
        left_tcp = left_wrist @ self._wrist_from_tcp
        right_tcp = right_wrist @ self._wrist_from_tcp
        return (
            _validated_matrix4x4(left_tcp, "left base_link->Dex1 TCP"),
            _validated_matrix4x4(right_tcp, "right base_link->Dex1 TCP"),
        )

    def metadata(self) -> dict[str, object]:
        return {
            "eef_pose_frame": "dex1_tcp",
            "eef_pose_parent_frame": BASE_LINK_FRAME,
            "eef_pose_unit": "m/rad",
            "robot_fk_urdf": str(self.robot_fk_urdf),
            "eef_model_urdf": str(self.eef_model_urdf),
            "left_wrist_to_tcp_xyz_m": TCP_TRANSLATION_M.tolist(),
            "left_wrist_to_tcp_rpy_rad": TCP_RPY_RAD.tolist(),
            "right_wrist_to_tcp_xyz_m": TCP_TRANSLATION_M.tolist(),
            "right_wrist_to_tcp_rpy_rad": TCP_RPY_RAD.tolist(),
        }

    def _forward(self, arm_qpos: np.ndarray, column_height_m: float, waist_yaw_rad: float):
        arm_qpos = np.asarray(arm_qpos, dtype=float).reshape(-1)
        if arm_qpos.shape != (14,):
            raise ValueError(f"Dex1 TCP FK arm_qpos expected length 14, got {arm_qpos.shape[0]}")
        if not np.all(np.isfinite(arm_qpos)):
            raise ValueError("Dex1 TCP FK arm_qpos contains NaN or Inf")
        column_height_m = _finite_scalar(column_height_m, "column_height_m")
        waist_yaw_rad = _finite_scalar(waist_yaw_rad, "waist_yaw_rad")
        if column_height_m < 0.0 or column_height_m > 0.42:
            raise ValueError(f"column_height_m must be within [0.0, 0.42], got {column_height_m}")

        q = self._neutral_q.copy()
        q[list(self._arm_q_indices)] = arm_qpos
        # G1D has two serial 0.21 m prismatic lift joints. The recorded 0.42 m
        # column height is their physical sum, so each receives exactly half.
        q[list(self._column_q_indices)] = column_height_m * 0.5
        q[self._waist_yaw_q_index] = waist_yaw_rad
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        root_pose = _validated_transform(data.oMf[self._root_frame_id], G1D_URDF_ROOT_FRAME)
        if not np.allclose(root_pose, np.eye(4), atol=1e-12, rtol=0.0):
            raise RuntimeError("G1D FK URDF root AGV_link is not identity; base_link binding is invalid")
        return data

    def _required_joint_q_indices(self, joint_names: tuple[str, ...]) -> tuple[int, ...]:
        indices = []
        for joint_name in joint_names:
            joint_id = self.model.getJointId(joint_name)
            if joint_id == 0 or joint_id >= self.model.njoints or self.model.names[joint_id] != joint_name:
                raise ValueError(f"G1D URDF is missing required TCP FK joint: {joint_name}")
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"G1D TCP FK joint must have nq=1: {joint_name}, got nq={joint.nq}")
            indices.append(int(joint.idx_q))
        if len(set(indices)) != len(indices):
            raise ValueError("G1D TCP FK joint q indices are not unique")
        return tuple(indices)

    def _required_frame_id(self, frame_name: str) -> int:
        frame_id = self.model.getFrameId(frame_name)
        if frame_id >= self.model.nframes or self.model.frames[frame_id].name != frame_name:
            raise ValueError(f"G1D URDF is missing required TCP FK frame: {frame_name}")
        return int(frame_id)


def _required_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _validated_wrist_to_tcp_transform() -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = TCP_TRANSLATION_M
    return _validated_matrix4x4(transform, "wrist->Dex1 TCP calibration")


def _validated_transform(se3: pin.SE3, label: str) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(se3.rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(se3.translation, dtype=float).reshape(3)
    return _validated_matrix4x4(transform, label)


def _validated_matrix4x4(value: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{label} must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} contains NaN or Inf")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-12, rtol=0.0):
        raise ValueError(f"{label} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f"{label} rotation determinant must be +1")
    return matrix.copy()


def _finite_scalar(value: float, label: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return scalar
