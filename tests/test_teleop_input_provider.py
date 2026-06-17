import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_lerobot_episode(dataset_root: pathlib.Path, include_fk: bool = False):
    data_dir = dataset_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    columns = {
        "timestamp": pa.array([0.0, 0.1], type=pa.float64()),
        "frame_index": pa.array([0, 1], type=pa.int64()),
        "episode_index": pa.array([0, 0], type=pa.int64()),
        "observation.state": pa.array(
            [
                [float(i + 20) for i in range(16)],
                [float(i + 40) for i in range(16)],
            ],
            type=pa.list_(pa.float64()),
        ),
        "action": pa.array(
            [
                [float(i) for i in range(16)],
                [float(i + 100) for i in range(16)],
            ],
            type=pa.list_(pa.float64()),
        ),
    }
    if include_fk:
        columns["observation.fk.cmd.left.gripper_flange"] = pa.array(
            [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0], [0.2, 0.3, 0.4, 0.1, 0.0, 0.0]],
            type=pa.list_(pa.float64()),
        )
        columns["observation.fk.cmd.right.gripper_flange"] = pa.array(
            [[0.4, 0.5, 0.6, 0.0, 0.1, 0.0], [0.5, 0.6, 0.7, 0.0, 0.1, 0.1]],
            type=pa.list_(pa.float64()),
        )
    pq.write_table(pa.table(columns), data_dir / "episode_000000.parquet")


class TeleopInputProviderTest(unittest.TestCase):
    def test_lerobot_action_source_emits_joint_position_intent(self):
        from teleop.utils.teleop_input_provider import LeRobotOfflineInputProvider

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_lerobot_episode(dataset_root)
            provider = LeRobotOfflineInputProvider(dataset_root, 0, arm_source="action", speed_scale=0.0)
            sample = provider.get_sample()

        self.assertEqual(sample.motion_intent.kind, "joint_position")
        self.assertEqual(sample.motion_intent.arm_q.tolist(), [float(i) for i in range(7)] + [float(i) for i in range(8, 15)])
        self.assertEqual(sample.motion_intent.gripper_q.tolist(), [7.0, 15.0])
        self.assertTrue(sample.tele_data.left_ctrl_squeeze)
        self.assertTrue(sample.tele_data.right_ctrl_squeeze)

    def test_lerobot_state_source_emits_joint_position_intent(self):
        from teleop.utils.teleop_input_provider import LeRobotOfflineInputProvider

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_lerobot_episode(dataset_root)
            provider = LeRobotOfflineInputProvider(dataset_root, 0, arm_source="state", speed_scale=0.0)
            sample = provider.get_sample()

        self.assertEqual(sample.motion_intent.kind, "joint_position")
        self.assertEqual(sample.motion_intent.arm_q.tolist(), [float(i + 20) for i in range(7)] + [float(i + 28) for i in range(7)])

    def test_lerobot_fk_cmd_pose_source_emits_pose_intent(self):
        from teleop.utils.teleop_input_provider import LeRobotOfflineInputProvider

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_lerobot_episode(dataset_root, include_fk=True)
            provider = LeRobotOfflineInputProvider(dataset_root, 0, arm_source="fk_cmd_pose", speed_scale=0.0)
            sample = provider.get_sample()

        self.assertEqual(sample.motion_intent.kind, "pose")
        self.assertTrue(np.allclose(sample.motion_intent.left_wrist_pose[:3, 3], [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(sample.motion_intent.right_wrist_pose[:3, 3], [0.4, 0.5, 0.6]))

    def test_gripper_q_maps_to_existing_dex1_trigger_value_range(self):
        from teleop.utils.teleop_input_provider import dex1_q_to_trigger_value

        self.assertAlmostEqual(dex1_q_to_trigger_value(0.0), 5.0)
        self.assertAlmostEqual(dex1_q_to_trigger_value(2.7), 6.0)
        self.assertAlmostEqual(dex1_q_to_trigger_value(5.4), 7.0)

    def test_lerobot_provider_marks_done_after_final_frame(self):
        from teleop.utils.teleop_input_provider import LeRobotOfflineInputProvider

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_lerobot_episode(dataset_root)
            provider = LeRobotOfflineInputProvider(dataset_root, 0, arm_source="action", speed_scale=0.0)
            self.assertFalse(provider.done)
            provider.get_sample()
            self.assertFalse(provider.done)
            provider.get_sample()
            self.assertTrue(provider.done)
            self.assertIsNone(provider.get_sample())

    def test_xr_provider_delegates_to_wrapped_xr_robotics_wrapper(self):
        from teleop.utils.teleop_input_provider import XRTeleopInputProvider
        from teleop.utils.xr_input_types import TeleData

        tele_data = TeleData(head_pose=np.eye(4), left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4))
        wrapped = mock.Mock()
        wrapped.get_tele_data.return_value = tele_data
        provider = XRTeleopInputProvider(wrapped)
        sample = provider.get_sample(current_left_robot_wrist_pose=np.eye(4))

        wrapped.get_tele_data.assert_called_once()
        self.assertIs(sample.tele_data, tele_data)
        self.assertEqual(sample.motion_intent.kind, "pose")

    def test_xr_provider_ignores_offline_only_kwargs(self):
        from teleop.utils.teleop_input_provider import XRTeleopInputProvider
        from teleop.utils.xr_input_types import TeleData

        class StrictXRWrapper:
            def get_tele_data(self, current_left_robot_wrist_pose=None, current_right_robot_wrist_pose=None):
                self.left_pose = current_left_robot_wrist_pose
                self.right_pose = current_right_robot_wrist_pose
                return TeleData(head_pose=np.eye(4), left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4))

        wrapped = StrictXRWrapper()
        provider = XRTeleopInputProvider(wrapped)
        sample = provider.get_sample(
            current_left_robot_wrist_pose=np.eye(4),
            current_right_robot_wrist_pose=np.eye(4),
            current_arm_q=np.zeros(14),
            current_arm_dq=np.zeros(14),
            dt=1.0 / 30.0,
        )

        self.assertIsNotNone(sample)
        self.assertTrue(np.allclose(wrapped.left_pose, np.eye(4)))
        self.assertTrue(np.allclose(wrapped.right_pose, np.eye(4)))

    def test_create_provider_rejects_lerobot_without_dataset(self):
        from teleop.utils.teleop_input_provider import create_teleop_input_provider

        args = SimpleNamespace(input_provider="lerobot_offline", offline_replay_dataset_root="", offline_replay_episode_index=0)
        with self.assertRaises(ValueError):
            create_teleop_input_provider(args)


if __name__ == "__main__":
    unittest.main()
