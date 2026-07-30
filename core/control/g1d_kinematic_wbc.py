"""Model-name driven kinematic whole-body controller for a G1-D model.

The controller does not encode guessed G1-D joint or frame names.  Those names
must come from the exact MJCF/URDF revision, which makes model mismatches fail at
startup instead of silently commanding the wrong joints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np
import numpy.typing as npt
import qpsolvers

from . import wbc_mink as mink


@dataclass(frozen=True)
class FrameRef:
    name: str
    frame_type: str = "site"


@dataclass(frozen=True)
class CollisionGeometryState:
    """Pose and dimensions of one collision proxy."""

    name: str
    geom_type: str
    size: np.ndarray
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class CollisionPairStatus:
    """Signed distance and closest points for one monitored proxy pair."""

    geom1: str
    geom2: str
    distance: float
    point1: np.ndarray
    point2: np.ndarray


class LateralBaseVelocityTask(mink.Task):
    """Penalize velocity perpendicular to a planar base's heading."""

    def __init__(self, model: mujoco.MjModel, cost: float = 1e3) -> None:
        super().__init__(cost=np.array([cost]))
        self._x_dof = int(model.joint("joint_x").dofadr[0])
        self._y_dof = int(model.joint("joint_y").dofadr[0])
        self._yaw_qpos = int(model.joint("joint_th").qposadr[0])
        self._nv = model.nv

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        del configuration
        return np.zeros(1)

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        yaw = configuration.q[self._yaw_qpos]
        jacobian = np.zeros((1, self._nv))
        # n = (-sin(yaw), cos(yaw)) is the body lateral direction in world axes.
        jacobian[0, self._x_dof] = -np.sin(yaw)
        jacobian[0, self._y_dof] = np.cos(yaw)
        return jacobian


class HandInFrontOfTorsoTask(mink.Task):
    """Soft one-sided task keeping a hand in front of the torso frame."""

    def __init__(
        self,
        model: mujoco.MjModel,
        hand_site: str,
        minimum_forward_distance: float,
        cost: float,
    ) -> None:
        if minimum_forward_distance < 0.0:
            raise ValueError("minimum_forward_distance must be nonnegative")
        super().__init__(cost=np.array([cost]))
        self._hand_site_id = model.site(hand_site).id
        self._torso_body_id = model.body("torso_link").id
        self._minimum_forward_distance = float(minimum_forward_distance)
        self._nv = model.nv

    def _forward_distance(self, configuration: mink.Configuration) -> float:
        data = configuration.data
        torso_rotation = data.xmat[self._torso_body_id].reshape(3, 3)
        hand_offset_world = (
            data.site_xpos[self._hand_site_id] - data.xpos[self._torso_body_id]
        )
        return float((torso_rotation.T @ hand_offset_world)[0])

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        distance = self._forward_distance(configuration)
        # A hinge loss: inactive once the hand is at least this far forward.
        return np.array([min(0.0, distance - self._minimum_forward_distance)])

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        if self._forward_distance(configuration) >= self._minimum_forward_distance:
            return np.zeros((1, self._nv))

        model, data = configuration.model, configuration.data
        hand_jacobian_position = np.zeros((3, self._nv))
        hand_jacobian_rotation = np.zeros((3, self._nv))
        torso_jacobian_position = np.zeros((3, self._nv))
        torso_jacobian_rotation = np.zeros((3, self._nv))
        mujoco.mj_jacSite(
            model,
            data,
            hand_jacobian_position,
            hand_jacobian_rotation,
            self._hand_site_id,
        )
        mujoco.mj_jacBody(
            model,
            data,
            torso_jacobian_position,
            torso_jacobian_rotation,
            self._torso_body_id,
        )

        torso_rotation = data.xmat[self._torso_body_id].reshape(3, 3)
        hand_offset_world = (
            data.site_xpos[self._hand_site_id] - data.xpos[self._torso_body_id]
        )
        skew_offset = np.array(
            [
                [0.0, -hand_offset_world[2], hand_offset_world[1]],
                [hand_offset_world[2], 0.0, -hand_offset_world[0]],
                [-hand_offset_world[1], hand_offset_world[0], 0.0],
            ]
        )
        relative_position_jacobian = torso_rotation.T @ (
            hand_jacobian_position
            - torso_jacobian_position
            + skew_offset @ torso_jacobian_rotation
        )
        return relative_position_jacobian[[0], :]


class HandsMidpointOnTorsoSagittalPlaneTask(mink.Task):
    """Soft task placing the two-hand midpoint on the torso x-z plane."""

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: float,
        left_hand_site: str = "left_gripper_tip_site",
        right_hand_site: str = "right_gripper_tip_site",
    ) -> None:
        if cost < 0.0:
            raise ValueError("hand midpoint lateral cost must be nonnegative")
        super().__init__(cost=np.array([cost]))
        self._hand_site_ids = (
            model.site(left_hand_site).id,
            model.site(right_hand_site).id,
        )
        self._torso_body_id = model.body("torso_link").id
        self._nv = model.nv

    def _midpoint_offset_world(
        self, configuration: mink.Configuration
    ) -> np.ndarray:
        data = configuration.data
        midpoint_world = 0.5 * (
            data.site_xpos[self._hand_site_ids[0]]
            + data.site_xpos[self._hand_site_ids[1]]
        )
        return midpoint_world - data.xpos[self._torso_body_id]

    def compute_error(self, configuration: mink.Configuration) -> np.ndarray:
        data = configuration.data
        torso_rotation = data.xmat[self._torso_body_id].reshape(3, 3)
        midpoint_torso = torso_rotation.T @ self._midpoint_offset_world(
            configuration
        )
        # y=0 is the torso sagittal (x-z) plane containing its forward x axis.
        return midpoint_torso[[1]]

    def compute_jacobian(self, configuration: mink.Configuration) -> np.ndarray:
        model, data = configuration.model, configuration.data
        midpoint_jacobian_position = np.zeros((3, self._nv))
        site_jacobian_position = np.zeros((3, self._nv))
        site_jacobian_rotation = np.zeros((3, self._nv))
        for site_id in self._hand_site_ids:
            mujoco.mj_jacSite(
                model,
                data,
                site_jacobian_position,
                site_jacobian_rotation,
                site_id,
            )
            midpoint_jacobian_position += 0.5 * site_jacobian_position

        torso_jacobian_position = np.zeros((3, self._nv))
        torso_jacobian_rotation = np.zeros((3, self._nv))
        mujoco.mj_jacBody(
            model,
            data,
            torso_jacobian_position,
            torso_jacobian_rotation,
            self._torso_body_id,
        )

        midpoint_offset_world = self._midpoint_offset_world(configuration)
        skew_offset = np.array(
            [
                [0.0, -midpoint_offset_world[2], midpoint_offset_world[1]],
                [midpoint_offset_world[2], 0.0, -midpoint_offset_world[0]],
                [-midpoint_offset_world[1], midpoint_offset_world[0], 0.0],
            ]
        )
        torso_rotation = data.xmat[self._torso_body_id].reshape(3, 3)
        relative_midpoint_jacobian = torso_rotation.T @ (
            midpoint_jacobian_position
            - torso_jacobian_position
            + skew_offset @ torso_jacobian_rotation
        )
        return relative_midpoint_jacobian[[1], :]


class KinematicWholeBodyController:
    """Velocity-QP WBC with support, end-effector, CoM, and posture tasks.

    Support frames are held at their pose from :meth:`reset`.  This is a
    kinematic contact approximation; it is not a force/torque controller.
    """

    def __init__(
        self,
        model_path: str | Path,
        controlled_frames: Sequence[FrameRef],
        support_frames: Sequence[FrameRef] = (),
        velocity_limits: Mapping[str, npt.ArrayLike] | None = None,
        *,
        solver: str = "quadprog",
        posture_cost: npt.ArrayLike = 1e-2,
        controlled_position_cost: float = 1.0,
        controlled_orientation_cost: float = 1.0,
        controlled_gain: float = 1.0,
        controlled_lm_damping: float = 1.0,
        support_cost: float = 100.0,
        com_cost: npt.ArrayLike | None = None,
        qp_damping: float = 1e-6,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.configuration = mink.Configuration(self.model)
        # Keep dynamics evaluation separate from the kinematic QP state.  The
        # real controller uses this buffer to compute static gravity feed-forward
        # from measured qpos without mutating the configuration being integrated
        # by differential IK.
        self._gravity_data = mujoco.MjData(self.model)
        self.solver = solver
        self.qp_damping = qp_damping

        self._validate_frames([*controlled_frames, *support_frames])
        self.controlled_tasks = {
            frame.name: mink.FrameTask(
                frame_name=frame.name,
                frame_type=frame.frame_type,
                position_cost=controlled_position_cost,
                orientation_cost=controlled_orientation_cost,
                gain=controlled_gain,
                lm_damping=controlled_lm_damping,
            )
            for frame in controlled_frames
        }
        self.support_tasks = {
            frame.name: mink.FrameTask(
                frame_name=frame.name,
                frame_type=frame.frame_type,
                position_cost=support_cost,
                orientation_cost=support_cost,
                lm_damping=1.0,
            )
            for frame in support_frames
        }

        self.com_task = mink.ComTask(cost=com_cost, lm_damping=1.0) if com_cost is not None else None
        self.posture_task = mink.PostureTask(self.model, cost=posture_cost)
        self.extra_tasks: list[mink.Task] = []
        self.limits = [mink.ConfigurationLimit(self.model)]
        if velocity_limits:
            self.limits.append(mink.VelocityLimit(self.model, velocity_limits))

        self.reset(self.model.qpos0)

    def _validate_frames(self, frames: Sequence[FrameRef]) -> None:
        enums = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "site": mujoco.mjtObj.mjOBJ_SITE,
        }
        seen = set()
        for frame in frames:
            if frame.frame_type not in enums:
                raise ValueError(f"unsupported frame type: {frame.frame_type}")
            key = (frame.name, frame.frame_type)
            if key in seen:
                raise ValueError(f"duplicate frame: {frame.frame_type}:{frame.name}")
            seen.add(key)
            if mujoco.mj_name2id(self.model, enums[frame.frame_type], frame.name) == -1:
                raise ValueError(f"model has no {frame.frame_type} named {frame.name!r}")

    def reset(self, qpos: npt.ArrayLike) -> None:
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.model.nq,):
            raise ValueError(f"qpos must have shape ({self.model.nq},), got {qpos.shape}")
        self.configuration.update(qpos)
        for task in [*self.controlled_tasks.values(), *self.support_tasks.values()]:
            task.set_target_from_configuration(self.configuration)
        if self.com_task is not None:
            self.com_task.set_target_from_configuration(self.configuration)
        self.posture_task.set_target_from_configuration(self.configuration)

    def solve(
        self,
        qpos: npt.ArrayLike,
        frame_targets: Mapping[str, npt.ArrayLike],
        dt: float,
        *,
        com_target: npt.ArrayLike | None = None,
        iterations: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(next_qpos, last_velocity)`` for the supplied task targets.

        Frame targets are 4x4 world transforms.  Repeating IK iterations is useful
        for waypoint planning; use one iteration in a real-time servo loop.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if iterations < 1:
            raise ValueError("iterations must be at least one")

        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.model.nq,):
            raise ValueError(f"qpos must have shape ({self.model.nq},), got {qpos.shape}")
        self.configuration.update(qpos)

        unknown = set(frame_targets) - set(self.controlled_tasks)
        if unknown:
            raise ValueError(f"targets supplied for uncontrolled frames: {sorted(unknown)}")
        for name, matrix in frame_targets.items():
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (4, 4):
                raise ValueError(f"target for {name!r} must be 4x4, got {matrix.shape}")
            self.controlled_tasks[name].set_target(mink.SE3.from_matrix(matrix))
        if com_target is not None:
            if self.com_task is None:
                raise ValueError("com_target supplied but the CoM task is disabled")
            self.com_task.set_target(com_target)

        tasks = [
            *self.support_tasks.values(),
            *self.controlled_tasks.values(),
            *([self.com_task] if self.com_task is not None else []),
            *self.extra_tasks,
            self.posture_task,
        ]
        velocity = np.zeros(self.model.nv)
        for _ in range(iterations):
            velocity = mink.solve_ik(
                self.configuration,
                tasks,
                dt,
                solver=self.solver,
                damping=self.qp_damping,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(velocity, dt)
        return self.configuration.q.copy(), velocity

    def gravity_compensation(self, qpos: npt.ArrayLike) -> np.ndarray:
        """Return generalized forces that balance gravity at ``qpos``.

        MuJoCo's ``qfrc_bias`` contains Coriolis and gravity forces.  Evaluating
        it with zero velocity removes the Coriolis contribution, leaving the
        static feed-forward torque/force required by the model equations.
        """
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.model.nq,):
            raise ValueError(
                f"qpos must have shape ({self.model.nq},), got {qpos.shape}"
            )
        if not np.all(np.isfinite(qpos)):
            raise ValueError("gravity compensation qpos must be finite")

        self._gravity_data.qpos[:] = qpos
        self._gravity_data.qvel[:] = 0.0
        self._gravity_data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self._gravity_data)
        gravity = np.asarray(self._gravity_data.qfrc_bias, dtype=float).copy()
        if gravity.shape != (self.model.nv,) or not np.all(np.isfinite(gravity)):
            raise RuntimeError("MuJoCo produced invalid gravity compensation")
        return gravity

    def gravity_compensation_for_joints(
        self,
        qpos: npt.ArrayLike,
        joint_names: Sequence[str],
    ) -> np.ndarray:
        """Return gravity feed-forward for scalar joints in ``joint_names``."""
        generalized_gravity = self.gravity_compensation(qpos)
        result = []
        for name in joint_names:
            joint = self.model.joint(name)
            joint_type = int(self.model.jnt_type[joint.id])
            scalar_types = {
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            }
            if joint_type not in scalar_types:
                raise ValueError(
                    f"gravity feed-forward requires a scalar joint, got {name!r}"
                )
            result.append(generalized_gravity[int(joint.dofadr[0])])
        return np.asarray(result, dtype=float)


G1D_CONTROL_FREQUENCY = 100.0
G1D_MAX_FORWARD_SPEED = 0.2
G1D_MAX_YAW_RATE = 0.5
G1D_MAX_WHEEL_SPEED = 10.0
G1D_WHEEL_RADIUS = 0.08484
G1D_WHEEL_TRACK = 0.4062
G1D_GRIPPER_OPEN = -0.015
G1D_GRIPPER_CLOSED = 0.0245

G1D_VELOCITY_LIMITS = {
    "joint_x": G1D_MAX_FORWARD_SPEED,
    "joint_y": G1D_MAX_FORWARD_SPEED,
    "joint_th": G1D_MAX_YAW_RATE,
    "Left_Wheel_Joint": G1D_MAX_WHEEL_SPEED,
    "Right_Wheel_Joint": G1D_MAX_WHEEL_SPEED,
    "LZ_mt_Joint": 0.05,
    "LZ_it_Joint": 0.05,
    # Despite its vendor name, this is the waist pitch joint (Y axis).  Keep
    # the measured startup angle but remove it from WBC motion allocation.
    "Yaw_Joint": 0.0,
    "torso_Joint": 1.0,
    **{
        f"{side}_{joint}_joint": 0.5
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    },
    **{
        f"{side}_gripper_finger_{finger}_joint": 0.2
        for side in ("left", "right")
        for finger in (1, 2)
    },
}


G1D_POSTURE_COST_BY_GROUP = {
    "base": 1e-5,
    "wheel": 1e-5,
    "lift": 1e-1,
    "torso": 1e0,
    "upper_arm": 0.1,
    "forearm": 0.1,
    "wrist": 0.1,
    "gripper": 1e-5,
}

# Soft per-DoF velocity penalties.  These complement (but do not replace) the
# hard limits in ``G1D_VELOCITY_LIMITS``: increasing a group's value makes the
# QP prefer moving that group less when another kinematic solution is available.
G1D_VELOCITY_COST_BY_GROUP = {
    # Planar translation (joint_x/joint_y) and yaw (joint_th) are separated
    # so WBC motion allocation can tune linear and angular base motion
    # independently.
    "base_linear": 40.0,
    "base_angular": 100.0,
    "wheel": 1e-6,
    "lift": 100.0,
    "torso": 20.0,
    "upper_arm": 0.5,
    "forearm": 0.5,
    "wrist": 0.5,
    "gripper": 1e-6,
}

# Runtime-only overrides applied on top of the XML ``home`` keyframe. With the
# vendor link-frame offsets, these elbow values make the shoulder-to-elbow and
# elbow-to-wrist segments geometrically perpendicular (90 degrees).
G1D_RUNTIME_HOME_JOINT_POSITIONS = {
    "left_elbow_joint": -0.293216,
    "right_elbow_joint": -0.293216,
}

G1D_COLLISION_GEOM_PAIRS = [
    (
        ("col_left_forearm", "col_left_hand"),
        ("col_head", "col_lift"),
    ),
    (
        ("col_right_forearm", "col_right_hand"),
        ("col_head", "col_lift"),
    ),
    (
        ("col_left_upper_arm", "col_left_forearm", "col_left_hand"),
        ("col_right_upper_arm", "col_right_forearm", "col_right_hand"),
    ),
    (("col_left_upper_arm",), ("col_left_hand",)),
    (("col_right_upper_arm",), ("col_right_hand",)),
]


def make_g1d_group_cost(
    model: mujoco.MjModel,
    cost_by_group: Mapping[str, float],
    *,
    cost_name: str,
    base_groups: Mapping[str, Sequence[str]] | None = None,
) -> np.ndarray:
    """Build a per-DoF cost vector from the G1-D joint groups."""
    costs = np.zeros(model.nv)
    assigned = np.zeros(model.nv, dtype=bool)
    groups = {
        **(
            base_groups
            if base_groups is not None
            else {"base": ("joint_x", "joint_y", "joint_th")}
        ),
        "wheel": ("Left_Wheel_Joint", "Right_Wheel_Joint"),
        "lift": ("LZ_mt_Joint", "LZ_it_Joint"),
        "torso": ("Yaw_Joint", "torso_Joint"),
        # The shoulder joints place the upper arm; the elbow joint controls
        # the forearm.  Keeping them separate allows independent Home and
        # velocity regularization without mixing the wrist group.
        "upper_arm": tuple(
            f"{side}_{joint}_joint"
            for side in ("left", "right")
            for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw")
        ),
        "forearm": tuple(
            f"{side}_elbow_joint" for side in ("left", "right")
        ),
        "wrist": tuple(
            f"{side}_wrist_{axis}_joint"
            for side in ("left", "right")
            for axis in ("roll", "pitch", "yaw")
        ),
        "gripper": tuple(
            f"{side}_gripper_finger_{finger}_joint"
            for side in ("left", "right")
            for finger in (1, 2)
        ),
    }
    for group, joint_names in groups.items():
        if group not in cost_by_group:
            raise ValueError(f"G1-D {cost_name} cost is missing group {group!r}")
        for joint_name in joint_names:
            joint_id = model.joint(joint_name).id
            dof_address = model.jnt_dofadr[joint_id]
            costs[dof_address] = cost_by_group[group]
            assigned[dof_address] = True
    if not np.all(assigned):
        missing = np.flatnonzero(~assigned).tolist()
        raise ValueError(f"G1-D {cost_name} cost is missing DoF indices: {missing}")
    return costs


def make_g1d_posture_cost(model: mujoco.MjModel) -> np.ndarray:
    """Build a per-DoF Home posture cost vector for the G1-D topology."""
    return make_g1d_group_cost(
        model,
        G1D_POSTURE_COST_BY_GROUP,
        cost_name="posture",
    )


def make_g1d_velocity_cost(model: mujoco.MjModel) -> np.ndarray:
    """Build a per-DoF soft joint-velocity cost vector for the G1-D topology."""
    return make_g1d_group_cost(
        model,
        G1D_VELOCITY_COST_BY_GROUP,
        cost_name="velocity",
        base_groups={
            "base_linear": ("joint_x", "joint_y"),
            "base_angular": ("joint_th",),
        },
    )


def make_g1d_runtime_home_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Return XML Home with the configured runtime joint overrides applied."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id == -1:
        raise ValueError("G1-D model has no 'home' keyframe")
    qpos = model.key_qpos[key_id].copy()
    for joint_name, value in G1D_RUNTIME_HOME_JOINT_POSITIONS.items():
        joint_id = model.joint(joint_name).id
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            if not lower <= value <= upper:
                raise ValueError(
                    f"runtime Home value {value} for {joint_name!r} is outside "
                    f"[{lower}, {upper}]"
                )
        qpos[model.jnt_qposadr[joint_id]] = value
    return qpos


class G1DKinematicWBC(KinematicWholeBodyController):
    """Ready-to-use dual-hand WBC configuration for the supplied G1-D MJCF."""

    def __init__(
        self,
        model_path: str | Path = "mj_assets/unitree_g1d/g1_d.xml",
        *,
        control_frequency: float = G1D_CONTROL_FREQUENCY,
        hand_position_cost: float = 10.0,
        hand_orientation_cost: float = 4.0,
        hand_gain: float = 0.1,
        hand_lm_damping: float = 1.0,
        lateral_velocity_cost: float = 100.0,
        hand_forward_minimum: float = 0.25,
        hand_forward_cost: float = 1.0,
        hand_midpoint_lateral_cost: float = 0.0,
        enable_collision_avoidance: bool = True,
        collision_minimum_distance: float = 0.03,
        collision_detection_distance: float = 0.08,
        collision_linearization_margin: float = 0.001,
        collision_gain: float = 0.5,
        collision_bound_relaxation: float = 0.0,
        collision_backtracking_steps: int = 8,
        collision_backtracking_tolerance: float = 1e-6,
        enable_collision_recovery: bool = False,
        collision_recovery_gain: float = 0.2,
        collision_recovery_exit_tolerance: float = 1e-4,
        collision_recovery_hand_cost_scale: float = 10.0,
        collision_slack_cost: float = 1e6,
        collision_pair_worsening_limit: float = 1e-4,
        collision_recovery_energy_tolerance: float = 1e-12,
        enable_safety_qp: bool = True,
        qp_damping: float = 1e-6,
    ) -> None:
        if control_frequency <= 0.0:
            raise ValueError("control_frequency must be positive")
        self.control_frequency = float(control_frequency)
        self.control_dt = 1.0 / self.control_frequency
        super().__init__(
            model_path=model_path,
            controlled_frames=[
                FrameRef("left_gripper_tip_site"),
                FrameRef("right_gripper_tip_site"),
            ],
            velocity_limits=G1D_VELOCITY_LIMITS,
            posture_cost=1e-2,  # Replaced with a per-DoF vector after model loading.
            controlled_position_cost=hand_position_cost,
            controlled_orientation_cost=hand_orientation_cost,
            controlled_gain=hand_gain,
            controlled_lm_damping=hand_lm_damping,
            qp_damping=qp_damping,
        )
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id == -1:
            raise ValueError("G1-D model has no 'home' keyframe")
        self.posture_task.set_cost(make_g1d_posture_cost(self.model))
        self.velocity_damping_task = mink.DampingTask(
            self.model,
            cost=make_g1d_velocity_cost(self.model),
        )
        self.extra_tasks.append(self.velocity_damping_task)
        self.extra_tasks.append(
            LateralBaseVelocityTask(self.model, cost=lateral_velocity_cost)
        )
        self.hand_forward_tasks = {
            hand_site: HandInFrontOfTorsoTask(
                self.model,
                hand_site=hand_site,
                minimum_forward_distance=hand_forward_minimum,
                cost=hand_forward_cost,
            )
            for hand_site in (
                "left_gripper_tip_site", "right_gripper_tip_site"
            )
        }
        self.extra_tasks.extend(self.hand_forward_tasks.values())
        self.hand_midpoint_task = HandsMidpointOnTorsoSagittalPlaneTask(
            self.model, cost=hand_midpoint_lateral_cost
        )
        self.extra_tasks.append(self.hand_midpoint_task)
        if collision_minimum_distance < 0.0:
            raise ValueError("collision_minimum_distance must be nonnegative")
        if collision_detection_distance < collision_minimum_distance:
            raise ValueError(
                "collision_detection_distance must be at least the minimum distance"
            )
        if collision_backtracking_steps < 1:
            raise ValueError("collision_backtracking_steps must be at least one")
        if collision_backtracking_tolerance < 0.0:
            raise ValueError("collision_backtracking_tolerance must be nonnegative")
        if enable_collision_recovery:
            if not 0.0 < collision_recovery_gain <= 1.0:
                raise ValueError("collision_recovery_gain must be in (0, 1]")
            if not (
                0.0
                <= collision_recovery_exit_tolerance
                <= collision_minimum_distance
            ):
                raise ValueError(
                    "collision_recovery_exit_tolerance must be between zero and "
                    "collision_minimum_distance"
                )
            if not 0.0 <= collision_recovery_hand_cost_scale <= 1.0:
                raise ValueError(
                    "collision_recovery_hand_cost_scale must be in [0, 1]"
                )
        if collision_slack_cost <= 0.0:
            raise ValueError("collision_slack_cost must be positive")
        if collision_pair_worsening_limit < 0.0:
            raise ValueError("collision_pair_worsening_limit must be nonnegative")
        if collision_recovery_energy_tolerance < 0.0:
            raise ValueError(
                "collision_recovery_energy_tolerance must be nonnegative"
            )
        if collision_linearization_margin < 0.0:
            raise ValueError("collision_linearization_margin must be nonnegative")
        if collision_detection_distance < (
            collision_minimum_distance + collision_linearization_margin
        ):
            raise ValueError(
                "collision_detection_distance must include the linearization margin"
            )
        self.collision_minimum_distance = float(collision_minimum_distance)
        self.collision_qp_distance = float(
            collision_minimum_distance + collision_linearization_margin
        )
        self.collision_detection_distance = float(collision_detection_distance)
        self.collision_backtracking_steps = collision_backtracking_steps
        self.collision_backtracking_tolerance = float(
            collision_backtracking_tolerance
        )
        self.enable_collision_recovery = bool(enable_collision_recovery)
        self.collision_recovery_gain = float(collision_recovery_gain)
        self.collision_recovery_exit_tolerance = float(
            collision_recovery_exit_tolerance
        )
        self.collision_recovery_distance = (
            self.collision_minimum_distance
            - self.collision_recovery_exit_tolerance
        )
        self.collision_recovery_hand_cost_scale = float(
            collision_recovery_hand_cost_scale
        )
        self.collision_slack_cost = float(collision_slack_cost)
        self.collision_pair_worsening_limit = float(
            collision_pair_worsening_limit
        )
        self.collision_recovery_energy_tolerance = float(
            collision_recovery_energy_tolerance
        )
        self.enable_safety_qp = enable_safety_qp
        self.collision_avoidance_limit = None
        self._collision_data = None
        if enable_collision_avoidance:
            self.collision_avoidance_limit = mink.CollisionAvoidanceLimit(
                model=self.model,
                geom_pairs=G1D_COLLISION_GEOM_PAIRS,
                gain=collision_gain,
                minimum_distance_from_collisions=self.collision_qp_distance,
                collision_detection_distance=collision_detection_distance,
                bound_relaxation=collision_bound_relaxation,
            )
            # Collision rows are handled by the augmented safety QP below.  They
            # are deliberately not added to the nominal IK QP, otherwise an
            # already penetrating configuration can make IK fail before the
            # recovery slack variables are available.
            if not enable_safety_qp:
                self.limits.append(self.collision_avoidance_limit)
            self._collision_data = mujoco.MjData(self.model)
        self._base_qpos_addresses = {
            name: int(self.model.joint(name).qposadr[0])
            for name in ("joint_x", "joint_y", "joint_th")
        }
        self._base_dof_addresses = {
            name: int(self.model.joint(name).dofadr[0])
            for name in ("joint_x", "joint_y", "joint_th")
        }
        self._last_base_command = np.zeros(2)
        self.last_collision_scale = 1.0
        self.in_collision_recovery = False
        self.last_collision_energy = 0.0
        self.last_collision_slacks: dict[tuple[str, str], float] = {}
        self.last_max_collision_slack = 0.0
        self.last_safety_qp_correction_norm = 0.0
        self.last_safety_qp_succeeded = True
        self._nominal_hand_task_costs = {
            name: task.cost.copy() for name, task in self.controlled_tasks.items()
        }
        self._home_qpos = make_g1d_runtime_home_qpos(self.model)
        self._gripper_targets = {"left": 0.0, "right": 0.0}
        self.reset(self._home_qpos)
        if self.collision_avoidance_limit is not None:
            home_distances = self._collision_distances(self._home_qpos)
            if np.any(home_distances < self.collision_qp_distance):
                raise ValueError(
                    "runtime Home violates collision proxy minimum distance: "
                    f"minimum={home_distances.min():.6f} m"
                )

    @property
    def home_qpos(self) -> np.ndarray:
        """Current WBC Home configuration."""
        return self._home_qpos.copy()

    def set_home_qpos(self, qpos: npt.ArrayLike) -> None:
        """Store a measured configuration as Home and reset all task targets."""
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.model.nq,):
            raise ValueError(f"qpos must have shape ({self.model.nq},), got {qpos.shape}")
        if not np.all(np.isfinite(qpos)):
            raise ValueError("Home qpos must contain only finite values")
        self._home_qpos = qpos.copy()
        self.reset(self._home_qpos)

    @property
    def gripper_targets(self) -> dict[str, float]:
        """Normalized gripper commands, where 0 is open and 1 is closed."""
        return self._gripper_targets.copy()

    def reset(self, qpos: npt.ArrayLike) -> None:
        """Reset WBC tasks and synchronize normalized gripper target state."""
        super().reset(qpos)
        if not hasattr(self, "_gripper_targets"):
            return
        target_q = self.posture_task.target_q
        assert target_q is not None
        travel = G1D_GRIPPER_CLOSED - G1D_GRIPPER_OPEN
        for side in ("left", "right"):
            positions = [
                target_q[
                    int(
                        self.model.joint(
                            f"{side}_gripper_finger_{finger}_joint"
                        ).qposadr[0]
                    )
                ]
                for finger in (1, 2)
            ]
            self._gripper_targets[side] = float(
                np.clip((np.mean(positions) - G1D_GRIPPER_OPEN) / travel, 0.0, 1.0)
            )

    def set_gripper_target(
        self, *, left: float | None = None, right: float | None = None
    ) -> None:
        """Set either gripper's symmetric opening target in normalized units.

        The command updates the two finger entries in the posture target. It
        therefore shares the WBC velocity and configuration limits instead of
        bypassing the QP with a direct actuator command.
        """
        if self.posture_task.target_q is None:
            raise RuntimeError("posture target has not been initialized")
        target_q = self.posture_task.target_q.copy()
        for side, command in (("left", left), ("right", right)):
            if command is None:
                continue
            command = float(command)
            if not np.isfinite(command) or not 0.0 <= command <= 1.0:
                raise ValueError(f"{side} gripper target must be in [0, 1]")
            joint_position = G1D_GRIPPER_OPEN + command * (
                G1D_GRIPPER_CLOSED - G1D_GRIPPER_OPEN
            )
            for finger in (1, 2):
                joint = self.model.joint(
                    f"{side}_gripper_finger_{finger}_joint"
                )
                target_q[int(joint.qposadr[0])] = joint_position
            self._gripper_targets[side] = command
        self.posture_task.set_target(target_q)

    def actuator_ctrl(self, qpos: npt.ArrayLike) -> np.ndarray:
        """Map a complete WBC configuration to the model's position actuators."""
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.model.nq,):
            raise ValueError(f"qpos must have shape ({self.model.nq},), got {qpos.shape}")
        ctrl = np.empty(self.model.nu)
        for actuator_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            ctrl[actuator_id] = qpos[self.model.jnt_qposadr[joint_id]]
        return ctrl

    @property
    def base_velocity_command(self) -> np.ndarray:
        """Last nonholonomic base command ``[forward_velocity, yaw_rate]``."""
        return self._last_base_command.copy()

    def wheel_velocity_command(
        self, forward_velocity: float | None = None, yaw_rate: float | None = None
    ) -> np.ndarray:
        """Convert ``(v, omega)`` to the model's left/right joint velocities."""
        if forward_velocity is None:
            forward_velocity = float(self._last_base_command[0])
        if yaw_rate is None:
            yaw_rate = float(self._last_base_command[1])
        left_linear = forward_velocity - yaw_rate * G1D_WHEEL_TRACK / 2.0
        right_linear = forward_velocity + yaw_rate * G1D_WHEEL_TRACK / 2.0
        # The URDF defines opposite Y axes for the left and right wheel joints.
        return np.array(
            [-left_linear / G1D_WHEEL_RADIUS, right_linear / G1D_WHEEL_RADIUS]
        )

    def _project_base_velocity(
        self, qpos: np.ndarray, velocity: np.ndarray
    ) -> np.ndarray:
        velocity = velocity.copy()
        yaw = qpos[self._base_qpos_addresses["joint_th"]]
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        xy_dofs = [
            self._base_dof_addresses["joint_x"],
            self._base_dof_addresses["joint_y"],
        ]
        forward_velocity = float(heading @ velocity[xy_dofs])
        forward_velocity = float(
            np.clip(forward_velocity, -G1D_MAX_FORWARD_SPEED, G1D_MAX_FORWARD_SPEED)
        )
        yaw_dof = self._base_dof_addresses["joint_th"]
        yaw_rate = float(np.clip(velocity[yaw_dof], -G1D_MAX_YAW_RATE, G1D_MAX_YAW_RATE))
        velocity[xy_dofs] = heading * forward_velocity
        velocity[yaw_dof] = yaw_rate
        self._last_base_command[:] = (forward_velocity, yaw_rate)
        return velocity

    def _collision_distances(self, qpos: np.ndarray) -> np.ndarray:
        if self.collision_avoidance_limit is None or self._collision_data is None:
            return np.empty(0)
        self._update_collision_data(qpos)
        distances = np.empty(len(self.collision_avoidance_limit.geom_id_pairs))
        fromto = np.empty(6)
        for index, (geom1, geom2) in enumerate(
            self.collision_avoidance_limit.geom_id_pairs
        ):
            distances[index] = mujoco.mj_geomDistance(
                self.model,
                self._collision_data,
                geom1,
                geom2,
                self.collision_detection_distance,
                fromto,
            )
        return distances

    def _update_collision_data(self, qpos: npt.ArrayLike) -> None:
        """Update the MuJoCo data shared by collision queries."""
        if self._collision_data is None:
            return
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (self.model.nq,):
            raise ValueError(f"qpos must have shape ({self.model.nq},), got {qpos.shape}")
        self._collision_data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self._collision_data)

    def collision_geometry_states(
        self,
        qpos: npt.ArrayLike | None = None,
        *,
        relative_to_body: str | None = None,
    ) -> list[CollisionGeometryState]:
        """Return the collision proxies used by the QP for visualization."""
        if self.collision_avoidance_limit is None or self._collision_data is None:
            return []
        if qpos is None:
            qpos = self.configuration.q
        self._update_collision_data(qpos)
        data = self._collision_data

        reference_position = np.zeros(3)
        reference_rotation = np.eye(3)
        if relative_to_body is not None:
            body_id = self.model.body(relative_to_body).id
            reference_position = data.xpos[body_id]
            reference_rotation = data.xmat[body_id].reshape(3, 3)

        geom_ids = sorted(
            {
                geom_id
                for pair in self.collision_avoidance_limit.geom_id_pairs
                for geom_id in pair
            }
        )
        states = []
        for geom_id in geom_ids:
            world_position = data.geom_xpos[geom_id]
            world_rotation = data.geom_xmat[geom_id].reshape(3, 3)
            states.append(
                CollisionGeometryState(
                    name=self.model.geom(geom_id).name,
                    geom_type=mujoco.mjtGeom(int(self.model.geom_type[geom_id]))
                    .name.removeprefix("mjGEOM_")
                    .lower(),
                    size=self.model.geom_size[geom_id].copy(),
                    position=reference_rotation.T
                    @ (world_position - reference_position),
                    rotation=reference_rotation.T @ world_rotation,
                )
            )
        return states

    def collision_pair_statuses(
        self,
        qpos: npt.ArrayLike | None = None,
        *,
        relative_to_body: str | None = None,
    ) -> list[CollisionPairStatus]:
        """Return signed distances for every pair monitored by collision QP."""
        if self.collision_avoidance_limit is None or self._collision_data is None:
            return []
        if qpos is None:
            qpos = self.configuration.q
        self._update_collision_data(qpos)
        data = self._collision_data

        reference_position = np.zeros(3)
        reference_rotation = np.eye(3)
        if relative_to_body is not None:
            body_id = self.model.body(relative_to_body).id
            reference_position = data.xpos[body_id]
            reference_rotation = data.xmat[body_id].reshape(3, 3)

        statuses = []
        for geom1, geom2 in self.collision_avoidance_limit.geom_id_pairs:
            fromto = np.zeros(6)
            distance = mujoco.mj_geomDistance(
                self.model,
                data,
                geom1,
                geom2,
                self.collision_detection_distance,
                fromto,
            )
            point1 = reference_rotation.T @ (fromto[:3] - reference_position)
            point2 = reference_rotation.T @ (fromto[3:] - reference_position)
            statuses.append(
                CollisionPairStatus(
                    geom1=self.model.geom(geom1).name,
                    geom2=self.model.geom(geom2).name,
                    distance=float(distance),
                    point1=point1,
                    point2=point2,
                )
            )
        return statuses

    def _solve_safety_qp(
        self, qpos: np.ndarray, desired_velocity: np.ndarray, dt: float
    ) -> np.ndarray:
        """Project desired velocity through hard limits and soft collision rows.

        Each active collision pair gets its own nonnegative slack variable.  If
        the robot is already inside the safe distance, the corresponding row
        requests positive separating motion.  The high slack cost makes recovery
        the priority while preserving QP feasibility when several contacts or
        joint limits are mutually incompatible.
        """
        self.last_safety_qp_correction_norm = 0.0
        self.last_safety_qp_succeeded = True
        self.last_collision_slacks = {}
        self.last_max_collision_slack = 0.0
        if not self.enable_safety_qp:
            return desired_velocity

        # Mink limits are expressed in configuration displacement dq.
        self.configuration.update(qpos)
        desired_dq = desired_velocity * dt
        hard_inequality_matrices = []
        hard_inequality_bounds = []
        for limit in self.limits:
            constraint = limit.compute_qp_inequalities(self.configuration, dt)
            if not constraint.inactive:
                assert constraint.G is not None and constraint.h is not None
                hard_inequality_matrices.append(constraint.G)
                hard_inequality_bounds.append(constraint.h)

        active_pair_indices = np.empty(0, dtype=int)
        collision_G = np.empty((0, self.model.nv))
        collision_h = np.empty(0)
        if self.collision_avoidance_limit is not None:
            distances = self._collision_distances(qpos)
            if self.enable_collision_recovery:
                gaps = np.maximum(
                    0.0, self.collision_recovery_distance - distances
                )
                self.last_collision_energy = float(gaps @ gaps)
                self.in_collision_recovery = bool(np.any(gaps > 0.0))
            else:
                self.last_collision_energy = 0.0
                self.in_collision_recovery = False

            collision_constraint = (
                self.collision_avoidance_limit.compute_qp_inequalities(
                    self.configuration, dt
                )
            )
            assert (
                collision_constraint.G is not None
                and collision_constraint.h is not None
            )
            active_pair_indices = np.flatnonzero(
                np.isfinite(collision_constraint.h)
            )
            collision_G = collision_constraint.G[active_pair_indices].copy()
            collision_h = collision_constraint.h[active_pair_indices].copy()

            # MuJoCo's closest-point segment reverses its distance-gradient
            # direction after actual geometric penetration (signed distance <
            # zero).  Flip those rows so -G*dq remains the predicted signed
            # distance increase on both sides of contact.
            geometrically_penetrating = (
                distances[active_pair_indices] < 0.0
            )
            collision_G[geometrically_penetrating] *= -1.0

            if self.enable_collision_recovery:
                recovering = (
                    distances[active_pair_indices]
                    < self.collision_minimum_distance
                )
                recovery_gaps = (
                    self.collision_minimum_distance
                    - distances[active_pair_indices][recovering]
                )
                collision_h[recovering] = (
                    -self.collision_recovery_gain * recovery_gaps
                    + self.collision_avoidance_limit.bound_relaxation
                )
        else:
            self.last_collision_energy = 0.0
            self.in_collision_recovery = False

        num_slacks = len(active_pair_indices)
        num_variables = self.model.nv + num_slacks
        inequality_matrices = []
        inequality_bounds = []
        if hard_inequality_matrices:
            hard_G = np.vstack(hard_inequality_matrices)
            inequality_matrices.append(
                np.hstack([hard_G, np.zeros((hard_G.shape[0], num_slacks))])
            )
            inequality_bounds.append(np.hstack(hard_inequality_bounds))
        if num_slacks:
            # G_collision * dq - slack <= h_collision, slack >= 0.
            inequality_matrices.append(
                np.hstack([collision_G, -np.eye(num_slacks)])
            )
            inequality_bounds.append(collision_h)
            inequality_matrices.append(
                np.hstack(
                    [np.zeros((num_slacks, self.model.nv)), -np.eye(num_slacks)]
                )
            )
            inequality_bounds.append(np.zeros(num_slacks))
        G = np.vstack(inequality_matrices) if inequality_matrices else None
        h = np.hstack(inequality_bounds) if inequality_bounds else None

        # Exact nonholonomic equality in body coordinates: lateral dq is zero.
        yaw = qpos[self._base_qpos_addresses["joint_th"]]
        A = np.zeros((1, num_variables))
        A[0, self._base_dof_addresses["joint_x"]] = -np.sin(yaw)
        A[0, self._base_dof_addresses["joint_y"]] = np.cos(yaw)
        b = np.zeros(1)

        P = np.eye(num_variables)
        if num_slacks:
            P[self.model.nv :, self.model.nv :] *= self.collision_slack_cost
        q = np.zeros(num_variables)
        q[: self.model.nv] = -desired_dq

        problem = qpsolvers.Problem(
            P=P,
            q=q,
            G=G,
            h=h,
            A=A,
            b=b,
        )
        result = qpsolvers.solve_problem(problem, solver=self.solver)
        if result.x is None:
            self.last_safety_qp_succeeded = False
            raise RuntimeError("safety projection QP returned no solution")

        solution = np.asarray(result.x)
        safe_velocity = solution[: self.model.nv] / dt
        if num_slacks:
            slack_values = np.maximum(0.0, solution[self.model.nv :])
            pairs = self.collision_avoidance_limit.geom_id_pairs
            self.last_collision_slacks = {
                (
                    self.model.geom(pairs[pair_index][0]).name,
                    self.model.geom(pairs[pair_index][1]).name,
                ): float(slack)
                for pair_index, slack in zip(active_pair_indices, slack_values)
            }
            self.last_max_collision_slack = float(slack_values.max())
        self.last_safety_qp_correction_norm = float(
            np.linalg.norm(safe_velocity - desired_velocity)
        )
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        xy_dofs = [
            self._base_dof_addresses["joint_x"],
            self._base_dof_addresses["joint_y"],
        ]
        self._last_base_command[:] = (
            heading @ safe_velocity[xy_dofs],
            safe_velocity[self._base_dof_addresses["joint_th"]],
        )
        return safe_velocity

    @property
    def minimum_collision_distance(self) -> float:
        """Minimum monitored proxy distance at the current WBC configuration."""
        distances = self._collision_distances(self.configuration.q)
        return float(distances.min()) if distances.size else np.inf

    def _collision_safe_step(
        self, qpos: np.ndarray, velocity: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.collision_avoidance_limit is None:
            self.last_collision_scale = 1.0
            self.last_collision_energy = 0.0
            self.in_collision_recovery = False
            next_qpos = qpos.copy()
            mujoco.mj_integratePos(self.model, next_qpos, velocity, dt)
            return next_qpos, velocity

        current_distances = self._collision_distances(qpos)
        collision_threshold = (
            self.collision_recovery_distance
            if self.enable_collision_recovery
            else self.collision_minimum_distance
        )
        current_gaps = np.maximum(
            0.0, collision_threshold - current_distances
        )
        current_energy = float(current_gaps @ current_gaps)
        penetrating_pairs = current_gaps > 0.0
        for attempt in range(self.collision_backtracking_steps):
            scale = 0.5**attempt
            scaled_velocity = velocity * scale
            candidate_qpos = qpos.copy()
            mujoco.mj_integratePos(
                self.model, candidate_qpos, scaled_velocity, dt
            )
            candidate_distances = self._collision_distances(candidate_qpos)
            non_recovery_lower_bound = np.minimum(
                current_distances[~penetrating_pairs],
                self.collision_minimum_distance,
            ) - self.collision_backtracking_tolerance
            remains_safe = candidate_distances[~penetrating_pairs] >= (
                non_recovery_lower_bound
            )
            candidate_gaps = np.maximum(
                0.0, collision_threshold - candidate_distances
            )
            candidate_energy = float(candidate_gaps @ candidate_gaps)

            accepted = bool(np.all(remains_safe))
            if np.any(penetrating_pairs):
                if self.enable_collision_recovery:
                    pair_worsening_is_bounded = candidate_distances[
                        penetrating_pairs
                    ] >= (
                        current_distances[penetrating_pairs]
                        - self.collision_pair_worsening_limit
                    )
                    recovery_progresses = candidate_energy <= (
                        current_energy
                        - self.collision_recovery_energy_tolerance
                    )
                    accepted = (
                        accepted
                        and bool(np.all(pair_worsening_is_bounded))
                        and recovery_progresses
                    )
                else:
                    does_not_worsen = candidate_distances[
                        penetrating_pairs
                    ] >= (
                        current_distances[penetrating_pairs]
                        - self.collision_backtracking_tolerance
                    )
                    accepted = accepted and bool(np.all(does_not_worsen))

            if accepted:
                self.last_collision_scale = scale
                self.last_collision_energy = (
                    candidate_energy if self.enable_collision_recovery else 0.0
                )
                self.in_collision_recovery = bool(
                    self.enable_collision_recovery and candidate_energy > 0.0
                )
                self._last_base_command *= scale
                return candidate_qpos, scaled_velocity

        self.last_collision_scale = 0.0
        self.last_collision_energy = (
            current_energy if self.enable_collision_recovery else 0.0
        )
        self.in_collision_recovery = bool(
            self.enable_collision_recovery and current_energy > 0.0
        )
        self._last_base_command[:] = 0.0
        return qpos.copy(), np.zeros_like(velocity)

    def solve(
        self,
        qpos: npt.ArrayLike,
        frame_targets: Mapping[str, npt.ArrayLike],
        dt: float | None = None,
        *,
        com_target: npt.ArrayLike | None = None,
        iterations: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve WBC while enforcing zero body-frame lateral base velocity."""
        if dt is None:
            dt = self.control_dt
        current_qpos = np.asarray(qpos, dtype=float).copy()
        if current_qpos.shape != (self.model.nq,):
            raise ValueError(
                f"qpos must have shape ({self.model.nq},), got {current_qpos.shape}"
            )
        if iterations < 1:
            raise ValueError("iterations must be at least one")

        velocity = np.zeros(self.model.nv)
        for _ in range(iterations):
            distances = self._collision_distances(current_qpos)
            recovering = bool(
                self.enable_collision_recovery
                and distances.size
                and np.any(distances < self.collision_recovery_distance)
            )
            hand_cost_scale = (
                self.collision_recovery_hand_cost_scale if recovering else 1.0
            )
            for name, task in self.controlled_tasks.items():
                task.cost[:] = (
                    self._nominal_hand_task_costs[name] * hand_cost_scale
                )
            try:
                _, velocity = super().solve(
                    current_qpos,
                    frame_targets,
                    dt,
                    com_target=com_target,
                    iterations=1,
                )
            finally:
                for name, task in self.controlled_tasks.items():
                    task.cost[:] = self._nominal_hand_task_costs[name]
            velocity = self._project_base_velocity(current_qpos, velocity)
            velocity = self._solve_safety_qp(current_qpos, velocity, dt)
            next_qpos, velocity = self._collision_safe_step(
                current_qpos, velocity, dt
            )
            current_qpos = next_qpos

        self.configuration.update(current_qpos)
        return current_qpos, velocity
