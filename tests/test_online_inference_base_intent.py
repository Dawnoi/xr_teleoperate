from types import SimpleNamespace

import numpy as np
import pytest

from core.input.base import BaseCommandIntent
from core.input.online_inference_provider import OnlineInferenceInputProvider
from inference.online_session import OnlineInferenceConfig, OnlineInferenceSession


def test_online_inference_provider_converts_base_action_metadata_to_base_intent():
    class FakeSession:
        max_camera_delta_ns = 100_000_000

        def tick(self, *, state_sample, camera_samples):
            return SimpleNamespace(
                left_pose=np.eye(4),
                right_pose=np.eye(4),
                left_gripper_width=0.01,
                right_gripper_width=0.02,
                enabled_arms=["left", "right"],
                status="executing_chunk",
                metadata={
                    "online_chunk_index": 3,
                    "online_base_action": [0.1, -0.2, 0.3, -0.4],
                },
            )

    provider = OnlineInferenceInputProvider(
        session=FakeSession(),
        arm_side="both",
        ee="dex1",
        no_gripper=False,
    )

    sample = provider.get_sample(
        current_left_robot_wrist_pose=np.eye(4),
        current_right_robot_wrist_pose=np.eye(4),
        current_state_host_monotonic_ns=1_000_000_000,
        current_left_gripper_width=0.01,
        current_right_gripper_width=0.02,
        camera_samples=[],
    )

    assert isinstance(sample.base_intent, BaseCommandIntent)
    assert sample.base_intent.vx == 0.1
    assert sample.base_intent.vy == -0.2
    assert sample.base_intent.wz == 0.3
    assert sample.base_intent.z == -0.4
    assert sample.base_intent.source == "online_inference"
    assert sample.base_intent.frame_index == 0


def test_online_inference_provider_emits_zero_base_intent_while_waiting_for_action():
    class FakeSession:
        max_camera_delta_ns = 100_000_000

        def tick(self, *, state_sample, camera_samples):
            return SimpleNamespace(
                left_pose=np.eye(4),
                right_pose=np.eye(4),
                left_gripper_width=0.01,
                right_gripper_width=0.02,
                enabled_arms=[],
                status="waiting_action",
                metadata={},
            )

    provider = OnlineInferenceInputProvider(
        session=FakeSession(),
        arm_side="both",
        ee="dex1",
        no_gripper=False,
    )

    sample = provider.get_sample(
        current_left_robot_wrist_pose=np.eye(4),
        current_right_robot_wrist_pose=np.eye(4),
        current_state_host_monotonic_ns=1_000_000_000,
        current_left_gripper_width=0.01,
        current_right_gripper_width=0.02,
        camera_samples=[],
    )

    assert isinstance(sample.base_intent, BaseCommandIntent)
    assert sample.base_intent.vx == 0.0
    assert sample.base_intent.vy == 0.0
    assert sample.base_intent.wz == 0.0
    assert sample.base_intent.z == 0.0
    assert sample.base_intent.source == "online_inference:missing_base_action"
    assert sample.base_intent.metadata["online_inference_status"] == "waiting_action"
    assert sample.base_intent.metadata["online_base_action_available"] is False


def test_online_inference_provider_emits_zero_base_intent_when_chunk_has_no_base_action():
    class FakeSession:
        max_camera_delta_ns = 100_000_000

        def tick(self, *, state_sample, camera_samples):
            return SimpleNamespace(
                left_pose=np.eye(4),
                right_pose=np.eye(4),
                left_gripper_width=0.01,
                right_gripper_width=0.02,
                enabled_arms=["left", "right"],
                status="executing_chunk",
                metadata={"online_chunk_index": 2},
            )

    provider = OnlineInferenceInputProvider(
        session=FakeSession(),
        arm_side="both",
        ee="dex1",
        no_gripper=False,
    )

    sample = provider.get_sample(
        current_left_robot_wrist_pose=np.eye(4),
        current_right_robot_wrist_pose=np.eye(4),
        current_state_host_monotonic_ns=1_000_000_000,
        current_left_gripper_width=0.01,
        current_right_gripper_width=0.02,
        camera_samples=[],
    )

    assert isinstance(sample.base_intent, BaseCommandIntent)
    assert (sample.base_intent.vx, sample.base_intent.vy, sample.base_intent.wz, sample.base_intent.z) == (0.0, 0.0, 0.0, 0.0)
    assert sample.base_intent.source == "online_inference:missing_base_action"
    assert sample.base_intent.metadata["online_inference_status"] == "executing_chunk"
    assert sample.base_intent.metadata["online_base_action_available"] is False


def test_mobile_pelvis_planar22_builds_legacy_pelvis_eef_state_in_contract_order():
    class FakeSession:
        config = SimpleNamespace(protocol_profile="mobile_pelvis_planar22")
        max_camera_delta_ns = 100_000_000

        def tick(self, *, state_sample, camera_samples):
            self.state_sample = state_sample
            return SimpleNamespace(
                left_pose=np.eye(4),
                right_pose=np.eye(4),
                left_gripper_width=0.01,
                right_gripper_width=0.02,
                enabled_arms=[],
                status="waiting_action",
                metadata={},
            )

    session = FakeSession()
    provider = OnlineInferenceInputProvider(
        session=session,
        arm_side="both",
        ee="dex1",
        no_gripper=False,
    )
    left_pelvis_eef = np.eye(4)
    left_pelvis_eef[:3, 3] = [0.1, 0.2, 0.3]
    right_pelvis_eef = np.eye(4)
    right_pelvis_eef[:3, 3] = [-0.1, -0.2, 0.4]

    provider.get_sample(
        current_left_robot_wrist_pose=left_pelvis_eef,
        current_right_robot_wrist_pose=right_pelvis_eef,
        current_left_gripper_width=0.011,
        current_right_gripper_width=0.022,
        current_map_base_pose=[1.0, 2.0, 0.5],
        current_base_velocity_base_link=[0.12, -0.34],
        camera_samples=[],
    )

    state = session.state_sample.mobile_pelvis_state25
    assert state is not None
    assert state.shape == (25,)
    np.testing.assert_allclose(state[:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(state[9], 0.011)
    np.testing.assert_allclose(state[10:13], [-0.1, -0.2, 0.4])
    np.testing.assert_allclose(state[19:], [0.022, 1.0, 2.0, 0.5, 0.12, -0.34])

    with pytest.raises(ValueError, match="mobile_pelvis_planar22 current Dex1 gripper q"):
        provider.get_sample(
            current_left_robot_wrist_pose=left_pelvis_eef,
            current_right_robot_wrist_pose=right_pelvis_eef,
            current_left_gripper_width=5.5,
            current_right_gripper_width=0.022,
            current_map_base_pose=[1.0, 2.0, 0.5],
            current_base_velocity_base_link=[0.12, -0.34],
            camera_samples=[],
        )


def test_mobile_pelvis_planar22_requires_its_wire_contract_and_keeps_pelvis_target_frame():
    session = OnlineInferenceSession(
        config=OnlineInferenceConfig(
            arm_side="both",
            protocol_profile="mobile_pelvis_planar22",
            enable_motion=False,
        ),
        transport=object(),
    )
    payload = {
        "type": "action_sequence",
        "wire_format": "mobile_pelvis_planar22_pose20_base4",
        "model_action_dim": 22,
        "wire_arm_action_dim": 20,
        "base_action_dim": 4,
        "actions": [[0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.03] * 2],
        "base_action": [[0.12, 0.0, -0.34, 0.0]],
    }
    session._load_action_payload(payload)

    assert session._current_chunk_left is not None
    left_target = session._action_pose_to_unitree("left", session._current_chunk_left[0][:7])
    np.testing.assert_allclose(left_target[:3, 3], [0.4, 0.5, 0.6])
    np.testing.assert_allclose(session._current_chunk_base[0], [0.12, 0.0, -0.34, 0.0])

    invalid = dict(payload)
    invalid["wire_format"] = "mobile_tcp23_pose20_base4"
    with pytest.raises(ValueError, match="mobile_pelvis_planar22 response requires wire_format"):
        session._load_action_payload(invalid)

    invalid = dict(payload)
    invalid["actions"] = [[0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.5] * 2]
    with pytest.raises(ValueError, match="mobile_pelvis_planar22 response gripper q"):
        session._load_action_payload(invalid)
