from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.control.arm_workspace_safety import clamp_dual_wrist_poses_to_side_workspaces
from core.control.workspace_governor import WorkspaceGovernor, WorkspaceGovernorConfig
from core.input.base import MotionIntent
from teleop.control_flow.arm_workspace_config import build_arm_side_workspaces
from teleop.control_flow.dex1_tcp_fk import Dex1TcpFkProvider
from teleop.control_flow.mobile_manipulation_coordinator import (
    G1DIkFrameKinematics,
    MobileManipulationCoordinator,
    MobileStateSample,
)
from teleop.real.args import build_arg_parser


def _governor() -> WorkspaceGovernor:
    return WorkspaceGovernor(
        WorkspaceGovernorConfig(
            mode="box",
            workspace_min=np.array([0.1, -0.3, -0.1]),
            workspace_max=np.array([0.5, 0.3, 0.4]),
            tapered={},
            max_forward_speed=0.2,
            max_base_yaw_rate=0.6,
            max_column_command=1.0,
            column_speed_mps=0.1,
            max_torso_yaw_rate=0.5,
        )
    )


def _pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = [x, y, z]
    return pose


def _per_arm_box_workspaces():
    return {
        "left": {
            "mode": "box",
            "workspace_min": np.array([0.1, -0.60, -0.1]),
            "workspace_max": np.array([0.5, 0.20, 0.4]),
            "tapered": {},
        },
        "right": {
            "mode": "box",
            "workspace_min": np.array([0.1, -0.20, -0.1]),
            "workspace_max": np.array([0.5, 0.60, 0.4]),
            "tapered": {},
        },
    }


def test_governor_preserves_nominal_command_inside_comfort_zone():
    governor = _governor()
    nominal = np.array([0.05, -0.1, 0.2, 0.1])

    result = governor.step(
        {"left": _pose(0.3, 0.0, 0.1)},
        {"left": True},
        nominal,
        column_position=0.2,
        torso_yaw=0.0,
        dt=0.03,
    )

    assert result.active is False
    assert result.feasible is True
    assert np.allclose(result.command, nominal)


def test_governor_recovers_boundary_target_without_exceeding_limits():
    governor = _governor()

    result = governor.step(
        {"left": _pose(0.7, 0.0, 0.1)},
        {"left": True},
        np.zeros(4),
        column_position=0.2,
        torso_yaw=0.0,
        dt=0.03,
    )

    assert result.active is True
    assert result.feasible is True
    assert result.command[0] > 0.0
    assert abs(result.command[0]) <= 0.2
    assert abs(result.command[1]) <= 0.6
    assert abs(result.command[2]) <= 1.0
    assert abs(result.command[3]) <= 0.5


def test_coordinator_reexpresses_fixed_global_target_after_mobile_motion():
    coordinator = MobileManipulationCoordinator(_governor(), state_timeout_sec=0.1)
    intent = MotionIntent(
        kind="pose",
        left_wrist_pose=_pose(0.4, 0.0, 0.0),
        right_wrist_pose=_pose(0.3, 0.0, 0.0),
        source="xr",
    )
    initial = MobileStateSample(np.eye(4), 1_000_000_000, 0.2, 0.0)

    coordinator.step(
        motion_intent=intent,
        enabled={"left": True, "right": False},
        rising={"left": True, "right": False},
        nominal_body_command=np.zeros(4),
        mobile_state=initial,
        now_monotonic_ns=1_010_000_000,
        dt=0.03,
        home_active=False,
        stop_active=False,
    )
    moved_global_from_ik = np.eye(4)
    moved_global_from_ik[0, 3] = 0.1
    moved = MobileStateSample(moved_global_from_ik, 1_020_000_000, 0.2, 0.0)
    result = coordinator.step(
        motion_intent=intent,
        enabled={"left": True, "right": False},
        rising={"left": False, "right": False},
        nominal_body_command=np.zeros(4),
        mobile_state=moved,
        now_monotonic_ns=1_030_000_000,
        dt=0.03,
        home_active=False,
        stop_active=False,
    )

    assert np.isclose(result.motion_intent_for_ik.left_wrist_pose[0, 3], 0.3)


def test_per_arm_workspace_allows_left_cross_body_without_expanding_right_workspace():
    left, right, clamped = clamp_dual_wrist_poses_to_side_workspaces(
        _pose(0.3, -0.45, 0.1),
        _pose(0.3, -0.45, 0.1),
        _per_arm_box_workspaces(),
    )

    assert clamped is True
    assert np.isclose(left[1, 3], -0.45)
    assert np.isclose(right[1, 3], -0.20)


def test_per_arm_qp_uses_only_the_gripped_hand_workspace():
    governor = WorkspaceGovernor(
        WorkspaceGovernorConfig(
            mode="box",
            workspace_min=np.array([0.1, -0.3, -0.1]),
            workspace_max=np.array([0.5, 0.3, 0.4]),
            tapered={},
            side_workspaces=_per_arm_box_workspaces(),
        )
    )

    # Right is outside its workspace but is not gripped, so it must not move the base.
    idle_right = governor.step(
        {"left": _pose(0.3, -0.45, 0.1), "right": _pose(0.3, -0.45, 0.1)},
        {"left": True, "right": False},
        np.zeros(4),
        column_position=0.2,
        torso_yaw=0.0,
        dt=0.03,
    )
    assert idle_right.active is False
    assert np.allclose(idle_right.command, 0.0)

    # The same right target must activate recovery as soon as right grip is held.
    active_right = governor.step(
        {"right": _pose(0.3, -0.45, 0.1)},
        {"right": True},
        np.zeros(4),
        column_position=0.2,
        torso_yaw=0.0,
        dt=0.03,
    )
    assert active_right.active is True


def test_single_hand_qp_starts_four_centimeters_before_body_midline():
    workspaces = _per_arm_box_workspaces()
    workspaces["left"]["workspace_min"][1] = 0.0
    governor = WorkspaceGovernor(
        WorkspaceGovernorConfig(
            mode="box",
            workspace_min=np.array([0.1, -0.3, -0.1]),
            workspace_max=np.array([0.5, 0.3, 0.4]),
            tapered={},
            side_workspaces=workspaces,
            comfort_margin=0.04,
        )
    )

    before_margin = governor.step(
        {"left": _pose(0.3, 0.041, 0.1)},
        {"left": True},
        np.zeros(4),
        column_position=0.2,
        torso_yaw=0.0,
        dt=0.03,
    )
    assert before_margin.active is False

    beyond_margin = governor.step(
        {"left": _pose(0.3, 0.039, 0.1)},
        {"left": True},
        np.zeros(4),
        column_position=0.2,
        torso_yaw=0.0,
        dt=0.03,
    )
    assert beyond_margin.active is True


def test_per_arm_tapered_arguments_are_explicit_and_asymmetric():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--arm-workspace-layout", "per_arm",
        "--left-arm-workspace-tapered", "-0.055", "0.245", "0.154", "0.38", "0.52", "0.00", "0.00", "0.20", "0.28",
        "--right-arm-workspace-tapered", "-0.055", "0.245", "0.154", "0.38", "0.52", "-0.20", "-0.28", "0.00", "0.00",
    ])
    workspaces = build_arm_side_workspaces(
        args,
        workspace_mode="tapered",
        workspace_min=np.array([0.1, -0.3, -0.1]),
        workspace_max=np.array([0.5, 0.3, 0.4]),
        tapered_workspace_params={},
    )

    assert workspaces["left"]["tapered"]["y_min_low"] == 0.0
    assert workspaces["right"]["tapered"]["y_max_low"] == 0.0
    assert workspaces["left"]["tapered"]["z_min"] == -0.055
    assert workspaces["left"]["tapered"]["z_max"] == 0.245
    assert workspaces["left"]["tapered"]["x_min"] == 0.154
    assert workspaces["left"]["tapered"]["y_max_low"] == 0.20
    assert workspaces["left"]["tapered"]["y_max_high"] == 0.28
    assert workspaces["right"]["tapered"]["y_min_low"] == -0.20
    assert workspaces["right"]["tapered"]["y_min_high"] == -0.28

    incomplete = parser.parse_args(["--arm-workspace-layout", "per_arm"])
    with pytest.raises(ValueError, match="left-arm-workspace-tapered"):
        build_arm_side_workspaces(
            incomplete,
            workspace_mode="tapered",
            workspace_min=np.array([0.1, -0.3, -0.1]),
            workspace_max=np.array([0.5, 0.3, 0.4]),
            tapered_workspace_params={},
        )


def test_mobile_workspace_height_is_zero_pose_dex1_tcp_plus_minus_15_cm():
    repo_root = Path(__file__).resolve().parents[1]
    tcp_fk = Dex1TcpFkProvider(
        repo_root / "assets/g1_d/g1_d.urdf",
        repo_root / "assets/dex1_1/dex1_1.urdf",
    )
    base_from_ik = G1DIkFrameKinematics(
        torso_from_ik=np.array([
            [1.0, 0.0, 0.0, 0.00396],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.044],
            [0.0, 0.0, 0.0, 1.0],
        ])
    ).agv_from_ik(column_position=0.0, torso_yaw=0.0)
    left_tcp, right_tcp = tcp_fk.compute_tcp_poses(
        np.zeros(14),
        column_height_m=0.0,
        waist_yaw_rad=0.0,
    )
    left_zero_tcp_z = float((np.linalg.inv(base_from_ik) @ left_tcp)[2, 3])
    right_zero_tcp_z = float((np.linalg.inv(base_from_ik) @ right_tcp)[2, 3])

    assert np.isclose(left_zero_tcp_z, right_zero_tcp_z, atol=1e-9)
    assert np.isclose(left_zero_tcp_z, 0.095226232, atol=1e-9)
    assert np.isclose(round(left_zero_tcp_z - 0.15, 3), -0.055)
    assert np.isclose(round(left_zero_tcp_z + 0.15, 3), 0.245)


def test_mobile_workspace_body_side_boundary_is_zero_pose_j6_center():
    repo_root = Path(__file__).resolve().parents[1]
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(repo_root / "assets/g1_d/g1_d.urdf"), pin.JointModelFreeFlyer())
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    base_from_ik = G1DIkFrameKinematics(
        torso_from_ik=np.array([
            [1.0, 0.0, 0.0, 0.00396],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.044],
            [0.0, 0.0, 0.0, 1.0],
        ])
    ).agv_from_ik(column_position=0.0, torso_yaw=0.0)
    ik_from_base = np.linalg.inv(base_from_ik)
    for side in ("left", "right"):
        joint_id = model.getJointId(f"{side}_wrist_pitch_joint")
        base_from_j6 = np.eye(4)
        base_from_j6[:3, :3] = data.oMi[joint_id].rotation
        base_from_j6[:3, 3] = data.oMi[joint_id].translation
        assert np.isclose(round(float((ik_from_base @ base_from_j6)[0, 3]), 3), 0.154)
