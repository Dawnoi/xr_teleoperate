"""Strict forward kinematics for the recorded fourteen G1D arm joints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin


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

LEFT_ARM_FRAME_NAMES = (
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
)
RIGHT_ARM_FRAME_NAMES = tuple(name.replace("left_", "right_", 1) for name in LEFT_ARM_FRAME_NAMES)
FLANGE_FRAME_NAMES = {
    "left.gripper_flange": "left_hand_palm_link",
    "right.gripper_flange": "right_hand_palm_link",
}


class G1DArmFkProvider:
    """Compute G1D arm FK relative to the URDF neutral body configuration."""

    def __init__(self, urdf_path: str | Path) -> None:
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"G1D URDF not found: {self.urdf_path}")

        self.model = pin.buildModelFromUrdf(str(self.urdf_path), pin.JointModelFreeFlyer())
        self._neutral_q = pin.neutral(self.model)
        self._arm_q_indices = self._resolve_arm_q_indices()
        self._frame_ids = self._resolve_frame_ids()

    def compute(self, arm_qpos: np.ndarray) -> dict[str, list[float]]:
        arm_qpos = np.asarray(arm_qpos, dtype=float).reshape(-1)
        if arm_qpos.shape != (14,):
            raise ValueError(f"G1D arm qpos expected length 14, got {arm_qpos.shape[0]}")
        if not np.all(np.isfinite(arm_qpos)):
            raise ValueError("G1D arm qpos contains NaN or Inf")

        # q starts at neutral so unrecorded chassis/lift/torso DoFs have a fixed reference value.
        q = self._neutral_q.copy()
        q[list(self._arm_q_indices)] = arm_qpos
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        return {
            output_key: _xyz_rpy(data.oMf[frame_id])
            for output_key, frame_id in self._frame_ids.items()
        }

    def _resolve_arm_q_indices(self) -> tuple[int, ...]:
        joint_names = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
        q_indices = []
        for joint_name in joint_names:
            joint_id = self.model.getJointId(joint_name)
            if joint_id == 0 or joint_id >= self.model.njoints or self.model.names[joint_id] != joint_name:
                raise ValueError(f"G1D URDF is missing required arm joint: {joint_name}")
            joint = self.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"G1D arm joint must have nq=1: {joint_name}, got nq={joint.nq}")
            q_indices.append(int(joint.idx_q))
        if len(set(q_indices)) != len(q_indices):
            raise ValueError("G1D arm joint q indices are not unique")
        return tuple(q_indices)

    def _resolve_frame_ids(self) -> dict[str, int]:
        frame_names = {
            **{
                f"left.joint{index}": frame_name
                for index, frame_name in enumerate(LEFT_ARM_FRAME_NAMES, start=1)
            },
            **{
                f"right.joint{index}": frame_name
                for index, frame_name in enumerate(RIGHT_ARM_FRAME_NAMES, start=1)
            },
            **FLANGE_FRAME_NAMES,
        }
        frame_ids = {}
        for output_key, frame_name in frame_names.items():
            frame_id = self.model.getFrameId(frame_name)
            if frame_id >= self.model.nframes or self.model.frames[frame_id].name != frame_name:
                raise ValueError(f"G1D URDF is missing required FK frame: {frame_name}")
            frame_ids[output_key] = int(frame_id)
        return frame_ids


def _xyz_rpy(se3: pin.SE3) -> list[float]:
    translation = np.asarray(se3.translation, dtype=float).reshape(3)
    rotation = np.asarray(se3.rotation, dtype=float).reshape(3, 3)
    rpy = np.asarray(pin.rpy.matrixToRpy(rotation), dtype=float).reshape(3)
    return [float(value) for value in np.concatenate((translation, rpy))]
