from types import SimpleNamespace

import numpy as np

from core.input.base import BaseCommandIntent
from core.input.online_inference_provider import OnlineInferenceInputProvider


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
