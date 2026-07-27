from __future__ import annotations

import numpy as np

from core.control.workspace_governor import WorkspaceGovernor, WorkspaceGovernorConfig
from core.input.base import MotionIntent
from teleop.control_flow.mobile_manipulation_coordinator import (
    MobileManipulationCoordinator,
    MobileStateSample,
)


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
