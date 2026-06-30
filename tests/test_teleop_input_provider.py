import pathlib
import json
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


def _pose_matrix(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def _write_raw_episode(task_root: pathlib.Path, sample_overrides=None):
    episode_dir = task_root / "episode_0001"
    (episode_dir / "colors" / "head").mkdir(parents=True, exist_ok=True)
    (episode_dir / "colors" / "wrist_left").mkdir(parents=True, exist_ok=True)
    (episode_dir / "colors" / "wrist_right").mkdir(parents=True, exist_ok=True)

    image = np.full((8, 10, 3), 32, dtype=np.uint8)
    for idx in range(2):
        for camera_name in ("head", "wrist_left", "wrist_right"):
            camera_dir = episode_dir / "colors" / camera_name
            file_name = f"{idx:06d}_{camera_name}.jpg"
            image_path = camera_dir / file_name
            ok = __import__("cv2").imwrite(str(image_path), image)
            if not ok:
                raise RuntimeError(f"failed to write test image: {image_path}")

    samples = []
    for idx in range(2):
        sample = {
            "idx": idx,
            "colors": {
                "head": f"colors/head/{idx:06d}_head.jpg",
                "left_wrist": f"colors/wrist_left/{idx:06d}_wrist_left.jpg",
                "right_wrist": f"colors/wrist_right/{idx:06d}_wrist_right.jpg",
            },
            "states": {
                "left_arm": {"qpos": [float(idx + i + 20) for i in range(7)]},
                "right_arm": {"qpos": [float(idx + i + 40) for i in range(7)]},
                "left_ee": {"qpos": [float(idx + 1.5)]},
                "right_ee": {"qpos": [float(idx + 2.5)]},
            },
            "actions": {
                "left_arm": {"qpos": [float(idx + i) for i in range(7)]},
                "right_arm": {"qpos": [float(idx + i + 10) for i in range(7)]},
                "left_ee": {"qpos": [float(idx + 3.5)]},
                "right_ee": {"qpos": [float(idx + 4.5)]},
            },
            "timestamps": {
                "sample_monotonic_ns": 1_000_000_000 + idx * 33_333_333,
            },
        }
        if sample_overrides and idx in sample_overrides:
            override = sample_overrides[idx]
            for key, value in override.items():
                sample[key] = value
        samples.append(sample)

    payload = {"info": {}, "text": {}, "data": samples}
    (episode_dir / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    return episode_dir


class FakeOnlineSession:
    def __init__(self, steps):
        self.steps = list(steps)
        self.tick_calls = []
        self.feedback = []
        self.max_camera_delta_ns = 100_000_000
        self.required_camera_names = []

    def tick(self, state_sample, camera_samples):
        self.tick_calls.append((state_sample, list(camera_samples)))
        return self.steps.pop(0)

    def set_required_camera_names(self, names):
        self.required_camera_names = list(names)

    def report_control_feedback(self, feedback):
        self.feedback.append(feedback)


class FakeCameraSource:
    def __init__(self, frame, host_monotonic_ns: int):
        self.frame = frame
        self.host_monotonic_ns = int(host_monotonic_ns)

    def get_latest(self, copy=True):
        frame = self.frame.copy() if copy else self.frame
        return frame, {"host_recv_monotonic_ns": self.host_monotonic_ns}


class TeleopInputProviderTest(unittest.TestCase):
    def test_raw_action_source_emits_joint_position_intent(self):
        from teleop.input.teleop_input_provider import create_teleop_input_provider

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root)
            args = SimpleNamespace(
                input_provider="lerobot_offline",
                offline_replay_dataset_root=str(task_root),
                offline_replay_episode_index=1,
                offline_replay_arm_source="action",
                offline_replay_speed_scale=0.0,
            )
            provider = create_teleop_input_provider(args)
            sample = provider.get_sample()

        self.assertEqual(sample.motion_intent.kind, "joint_position")
        self.assertEqual(sample.motion_intent.arm_q.tolist(), [float(i) for i in range(7)] + [float(i + 10) for i in range(7)])
        self.assertEqual(sample.motion_intent.gripper_q.tolist(), [3.5, 4.5])

    def test_raw_state_source_emits_joint_position_intent(self):
        from teleop.input.teleop_input_provider import create_teleop_input_provider

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root)
            args = SimpleNamespace(
                input_provider="lerobot_offline",
                offline_replay_dataset_root=str(task_root),
                offline_replay_episode_index=1,
                offline_replay_arm_source="state",
                offline_replay_speed_scale=0.0,
            )
            provider = create_teleop_input_provider(args)
            sample = provider.get_sample()

        self.assertEqual(sample.motion_intent.kind, "joint_position")
        self.assertEqual(sample.motion_intent.arm_q.tolist(), [float(i + 20) for i in range(7)] + [float(i + 40) for i in range(7)])

    def test_raw_fk_cmd_pose_source_uses_action_gripper_qpos(self):
        from teleop.input.teleop_input_provider import create_teleop_input_provider

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(
                task_root,
                sample_overrides={
                    0: {
                        "states": {
                            "left_arm": {"qpos": [float(i + 20) for i in range(7)]},
                            "right_arm": {"qpos": [float(i + 40) for i in range(7)]},
                            "left_ee": {"qpos": [4.1]},
                            "right_ee": {"qpos": [4.2]},
                        },
                        "actions": {
                            "left_arm": {
                                "qpos": [float(i) for i in range(7)],
                                "pose": {"matrix4x4": _pose_matrix(0.1, 0.2, 0.3).tolist()},
                            },
                            "right_arm": {
                                "qpos": [float(i + 10) for i in range(7)],
                                "pose": {"matrix4x4": _pose_matrix(0.4, 0.5, 0.6).tolist()},
                            },
                            "left_ee": {"qpos": [2.1]},
                            "right_ee": {"qpos": [2.3]},
                        },
                    },
                    1: {
                        "states": {
                            "left_arm": {"qpos": [float(i + 21) for i in range(7)]},
                            "right_arm": {"qpos": [float(i + 41) for i in range(7)]},
                            "left_ee": {"qpos": [4.3]},
                            "right_ee": {"qpos": [4.4]},
                        },
                        "actions": {
                            "left_arm": {
                                "qpos": [float(i + 1) for i in range(7)],
                                "pose": {"matrix4x4": _pose_matrix(0.2, 0.3, 0.4).tolist()},
                            },
                            "right_arm": {
                                "qpos": [float(i + 11) for i in range(7)],
                                "pose": {"matrix4x4": _pose_matrix(0.5, 0.6, 0.7).tolist()},
                            },
                            "left_ee": {"qpos": [2.2]},
                            "right_ee": {"qpos": [2.4]},
                        },
                    }
                },
            )
            args = SimpleNamespace(
                input_provider="lerobot_offline",
                offline_replay_dataset_root=str(task_root),
                offline_replay_episode_index=1,
                offline_replay_arm_source="fk_cmd_pose",
                offline_replay_speed_scale=0.0,
            )
            provider = create_teleop_input_provider(args)
            sample = provider.get_sample()

        self.assertEqual(sample.motion_intent.kind, "pose")
        self.assertEqual(sample.motion_intent.gripper_q.tolist(), [2.1, 2.3])
        self.assertAlmostEqual(sample.tele_data.left_ctrl_triggerValue, 5.777777777777778)
        self.assertAlmostEqual(sample.tele_data.right_ctrl_triggerValue, 5.851851851851852)

    def test_raw_episode_missing_qpos_raises(self):
        from teleop.input.teleop_input_provider import create_teleop_input_provider

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(
                task_root,
                sample_overrides={
                    0: {
                        "actions": {
                            "left_arm": {},
                            "right_arm": {"qpos": [float(i + 10) for i in range(7)]},
                            "left_ee": {"qpos": [3.5]},
                            "right_ee": {"qpos": [4.5]},
                        }
                    }
                },
            )
            args = SimpleNamespace(
                input_provider="lerobot_offline",
                offline_replay_dataset_root=str(task_root),
                offline_replay_episode_index=1,
                offline_replay_arm_source="action",
                offline_replay_speed_scale=0.0,
            )
            with self.assertRaises(KeyError):
                create_teleop_input_provider(args)

    def test_raw_episode_non_monotonic_timestamp_raises(self):
        from teleop.input.teleop_input_provider import create_teleop_input_provider

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(
                task_root,
                sample_overrides={
                    1: {
                        "timestamps": {
                            "sample_monotonic_ns": 1_000_000_000,
                        }
                    }
                },
            )
            args = SimpleNamespace(
                input_provider="lerobot_offline",
                offline_replay_dataset_root=str(task_root),
                offline_replay_episode_index=1,
                offline_replay_arm_source="action",
                offline_replay_speed_scale=0.0,
            )
            with self.assertRaises(ValueError):
                create_teleop_input_provider(args)

    def test_lerobot_action_source_emits_joint_position_intent(self):
        from teleop.input.teleop_input_provider import LeRobotOfflineInputProvider

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
        from teleop.input.teleop_input_provider import LeRobotOfflineInputProvider

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_lerobot_episode(dataset_root)
            provider = LeRobotOfflineInputProvider(dataset_root, 0, arm_source="state", speed_scale=0.0)
            sample = provider.get_sample()

        self.assertEqual(sample.motion_intent.kind, "joint_position")
        self.assertEqual(sample.motion_intent.arm_q.tolist(), [float(i + 20) for i in range(7)] + [float(i + 28) for i in range(7)])

    def test_lerobot_fk_cmd_pose_source_emits_pose_intent(self):
        from teleop.input.teleop_input_provider import LeRobotOfflineInputProvider

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_lerobot_episode(dataset_root, include_fk=True)
            provider = LeRobotOfflineInputProvider(dataset_root, 0, arm_source="fk_cmd_pose", speed_scale=0.0)
            sample = provider.get_sample()

        self.assertEqual(sample.motion_intent.kind, "pose")
        self.assertTrue(np.allclose(sample.motion_intent.left_wrist_pose[:3, 3], [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(sample.motion_intent.right_wrist_pose[:3, 3], [0.4, 0.5, 0.6]))

    def test_gripper_q_maps_to_existing_dex1_trigger_value_range(self):
        from teleop.input.teleop_input_provider import dex1_q_to_trigger_value

        self.assertAlmostEqual(dex1_q_to_trigger_value(0.0), 5.0)
        self.assertAlmostEqual(dex1_q_to_trigger_value(2.7), 6.0)
        self.assertAlmostEqual(dex1_q_to_trigger_value(5.4), 7.0)

    def test_lerobot_provider_marks_done_after_final_frame(self):
        from teleop.input.teleop_input_provider import LeRobotOfflineInputProvider

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
        from teleop.input.teleop_input_provider import XRTeleopInputProvider
        from teleop.input.xr_input_types import TeleData

        tele_data = TeleData(head_pose=np.eye(4), left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4))
        wrapped = mock.Mock()
        wrapped.get_tele_data.return_value = tele_data
        provider = XRTeleopInputProvider(wrapped)
        sample = provider.get_sample(current_left_robot_wrist_pose=np.eye(4))

        wrapped.get_tele_data.assert_called_once()
        self.assertIs(sample.tele_data, tele_data)
        self.assertEqual(sample.motion_intent.kind, "pose")

    def test_xr_provider_ignores_offline_only_kwargs(self):
        from teleop.input.teleop_input_provider import XRTeleopInputProvider
        from teleop.input.xr_input_types import TeleData

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

    def test_create_xr_provider_logs_provider_selection(self):
        import teleop.input.teleop_input_provider as input_provider_module

        class FakeXRWrapper:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        args = SimpleNamespace(input_provider="xr")
        with mock.patch.object(input_provider_module, "_load_xr_robotics_wrapper", return_value=FakeXRWrapper):
            with mock.patch.object(input_provider_module, "logger_mp", create=True) as logger:
                input_provider_module.create_teleop_input_provider(args)

        logger.info.assert_any_call("Using XR-Robotics as the teleop input provider.")

    def test_create_lerobot_provider_logs_provider_selection(self):
        import teleop.input.teleop_input_provider as input_provider_module

        args = SimpleNamespace(
            input_provider="lerobot_offline",
            offline_replay_dataset_root="/tmp/dataset",
            offline_replay_episode_index=7,
            offline_replay_arm_source="action",
            offline_replay_speed_scale=0.5,
        )
        with mock.patch.object(
            input_provider_module,
            "validate_lerobot_offline_episode",
            return_value={"dataset_kind": "lerobot_parquet"},
        ):
            with mock.patch.object(input_provider_module, "LeRobotOfflineInputProvider", return_value=mock.Mock()):
                with mock.patch.object(input_provider_module, "logger_mp", create=True) as logger:
                    input_provider_module.create_teleop_input_provider(args)

        logger.info.assert_any_call(
            "Using offline teleop input provider (%s), dataset_root=%s, episode_index=%d.",
            "lerobot_offline",
            "/tmp/dataset",
            7,
        )

    def test_create_provider_rejects_lerobot_without_dataset(self):
        from teleop.input.teleop_input_provider import create_teleop_input_provider

        args = SimpleNamespace(input_provider="lerobot_offline", offline_replay_dataset_root="", offline_replay_episode_index=0)
        with self.assertRaises(ValueError):
            create_teleop_input_provider(args)

    def test_online_provider_left_arm_emits_pose_intent_and_dex1_qpos_trigger(self):
        from teleop.inference.online_session import CameraSample, OnlineInferenceStep
        from teleop.input.teleop_input_provider import OnlineInferenceInputProvider

        current_left = _pose_matrix(1.0, 0.0, 0.0)
        current_right = _pose_matrix(2.0, 0.0, 0.0)
        target_left = _pose_matrix(3.0, 0.0, 0.0)
        session = FakeOnlineSession(
            [
                OnlineInferenceStep(
                    left_pose=target_left,
                    right_pose=current_right,
                    left_gripper_width=5.4,
                    right_gripper_width=2.7,
                    enabled_arms=["left"],
                    status="executing_chunk",
                )
            ]
        )
        provider = OnlineInferenceInputProvider(
            session=session,
            arm_side="left",
            ee="dex1",
            no_gripper=False,
        )

        sample = provider.get_sample(
            current_state_host_monotonic_ns=123,
            current_left_robot_wrist_pose=current_left,
            current_right_robot_wrist_pose=current_right,
            current_left_gripper_width=2.7,
            current_right_gripper_width=3.6,
            camera_samples=[
                CameraSample(name="head", frame=np.zeros((4, 4, 3), dtype=np.uint8), host_monotonic_ns=123)
            ],
        )

        self.assertEqual(sample.motion_intent.source, "online_inference")
        self.assertEqual(sample.motion_intent.metadata["enabled_arms"], ["left"])
        self.assertTrue(np.allclose(sample.motion_intent.left_wrist_pose, target_left))
        self.assertTrue(np.allclose(sample.motion_intent.right_wrist_pose, current_right))
        self.assertTrue(sample.tele_data.left_ctrl_squeeze)
        self.assertFalse(sample.tele_data.right_ctrl_squeeze)
        self.assertAlmostEqual(sample.tele_data.left_ctrl_triggerValue, 7.0)
        state_sample, camera_samples = session.tick_calls[0]
        self.assertEqual(state_sample.host_monotonic_ns, 123)
        self.assertAlmostEqual(state_sample.left_gripper_width, 2.7)
        self.assertEqual(camera_samples[0].name, "head")

    def test_online_provider_right_arm_holds_left_side(self):
        from teleop.inference.online_session import OnlineInferenceStep
        from teleop.input.teleop_input_provider import OnlineInferenceInputProvider

        current_left = _pose_matrix(1.0, 0.0, 0.0)
        current_right = _pose_matrix(2.0, 0.0, 0.0)
        target_right = _pose_matrix(4.0, 0.0, 0.0)
        session = FakeOnlineSession(
            [
                OnlineInferenceStep(
                    left_pose=current_left,
                    right_pose=target_right,
                    left_gripper_width=2.7,
                    right_gripper_width=5.4,
                    enabled_arms=["right"],
                    status="executing_chunk",
                )
            ]
        )
        provider = OnlineInferenceInputProvider(session=session, arm_side="right", ee="dex1", no_gripper=False)

        sample = provider.get_sample(
            current_state_host_monotonic_ns=456,
            current_left_robot_wrist_pose=current_left,
            current_right_robot_wrist_pose=current_right,
            current_left_gripper_width=2.7,
            current_right_gripper_width=3.6,
            camera_samples=[],
        )

        self.assertEqual(sample.motion_intent.metadata["enabled_arms"], ["right"])
        self.assertTrue(np.allclose(sample.motion_intent.left_wrist_pose, current_left))
        self.assertTrue(np.allclose(sample.motion_intent.right_wrist_pose, target_right))
        self.assertFalse(sample.tele_data.left_ctrl_squeeze)
        self.assertTrue(sample.tele_data.right_ctrl_squeeze)
        self.assertAlmostEqual(sample.tele_data.right_ctrl_triggerValue, 7.0)

    def test_online_provider_both_arms_enabled(self):
        from teleop.inference.online_session import OnlineInferenceStep
        from teleop.input.teleop_input_provider import OnlineInferenceInputProvider

        session = FakeOnlineSession(
            [
                OnlineInferenceStep(
                    left_pose=_pose_matrix(3.0, 0.0, 0.0),
                    right_pose=_pose_matrix(4.0, 0.0, 0.0),
                    left_gripper_width=0.000,
                    right_gripper_width=5.4,
                    enabled_arms=["left", "right"],
                    status="executing_chunk",
                )
            ]
        )
        provider = OnlineInferenceInputProvider(session=session, arm_side="both", ee="dex1", no_gripper=False)

        sample = provider.get_sample(
            current_state_host_monotonic_ns=789,
            current_left_robot_wrist_pose=_pose_matrix(1.0, 0.0, 0.0),
            current_right_robot_wrist_pose=_pose_matrix(2.0, 0.0, 0.0),
            current_left_gripper_width=0.0,
            current_right_gripper_width=5.4,
            camera_samples=[],
        )

        self.assertEqual(sample.motion_intent.metadata["enabled_arms"], ["left", "right"])
        self.assertTrue(sample.tele_data.left_ctrl_squeeze)
        self.assertTrue(sample.tele_data.right_ctrl_squeeze)
        self.assertAlmostEqual(sample.tele_data.left_ctrl_triggerValue, 5.0)
        self.assertAlmostEqual(sample.tele_data.right_ctrl_triggerValue, 7.0)

    def test_online_provider_dry_run_status_does_not_enable_motion(self):
        from teleop.inference.online_session import OnlineInferenceStep
        from teleop.input.teleop_input_provider import OnlineInferenceInputProvider

        current_left = _pose_matrix(1.0, 0.0, 0.0)
        current_right = _pose_matrix(2.0, 0.0, 0.0)
        session = FakeOnlineSession(
            [
                OnlineInferenceStep(
                    left_pose=_pose_matrix(3.0, 0.0, 0.0),
                    right_pose=current_right,
                    left_gripper_width=0.054,
                    right_gripper_width=0.020,
                    enabled_arms=[],
                    status="executing_chunk",
                    metadata={"dry_run": True},
                )
            ]
        )
        provider = OnlineInferenceInputProvider(session=session, arm_side="left", ee="dex1", no_gripper=False)

        sample = provider.get_sample(
            current_state_host_monotonic_ns=123,
            current_left_robot_wrist_pose=current_left,
            current_right_robot_wrist_pose=current_right,
            current_left_gripper_width=0.010,
            current_right_gripper_width=0.020,
            camera_samples=[],
        )

        self.assertEqual(sample.motion_intent.metadata["enabled_arms"], [])
        self.assertFalse(sample.tele_data.left_ctrl_squeeze)
        self.assertFalse(sample.tele_data.right_ctrl_squeeze)

    def test_online_provider_failed_session_returns_done(self):
        from teleop.inference.online_session import OnlineInferenceStep
        from teleop.input.teleop_input_provider import OnlineInferenceInputProvider

        session = FakeOnlineSession(
            [
                OnlineInferenceStep(
                    left_pose=_pose_matrix(1.0, 0.0, 0.0),
                    right_pose=_pose_matrix(2.0, 0.0, 0.0),
                    left_gripper_width=0.010,
                    right_gripper_width=0.020,
                    enabled_arms=[],
                    status="failed",
                    metadata={"error": "timeout"},
                )
            ]
        )
        provider = OnlineInferenceInputProvider(session=session, arm_side="left", ee="dex1", no_gripper=False)

        sample = provider.get_sample(
            current_state_host_monotonic_ns=123,
            current_left_robot_wrist_pose=_pose_matrix(1.0, 0.0, 0.0),
            current_right_robot_wrist_pose=_pose_matrix(2.0, 0.0, 0.0),
            current_left_gripper_width=0.010,
            current_right_gripper_width=0.020,
            camera_samples=[],
        )

        self.assertTrue(sample.done)
        self.assertEqual(sample.motion_intent.metadata["online_inference_status"], "failed")

    def test_online_provider_declares_non_empty_camera_sources_as_required(self):
        from teleop.inference.online_session import OnlineInferenceStep
        from teleop.input.teleop_input_provider import OnlineInferenceInputProvider

        session = FakeOnlineSession(
            [
                OnlineInferenceStep(
                    left_pose=_pose_matrix(1.0, 0.0, 0.0),
                    right_pose=_pose_matrix(2.0, 0.0, 0.0),
                    left_gripper_width=0.010,
                    right_gripper_width=0.020,
                    enabled_arms=[],
                    status="collecting_observation",
                )
            ]
        )
        provider = OnlineInferenceInputProvider(session=session, arm_side="left", ee="dex1", no_gripper=False)

        provider.get_sample(
            current_state_host_monotonic_ns=123,
            current_left_robot_wrist_pose=_pose_matrix(1.0, 0.0, 0.0),
            current_right_robot_wrist_pose=_pose_matrix(2.0, 0.0, 0.0),
            current_left_gripper_width=0.010,
            current_right_gripper_width=0.020,
            camera_sources={
                "head": FakeCameraSource(np.zeros((4, 4, 3), dtype=np.uint8), 123),
                "left_wrist": FakeCameraSource(np.zeros((4, 4, 3), dtype=np.uint8), 123),
                "right_wrist": None,
            },
        )

        self.assertEqual(session.required_camera_names, ["head", "left_wrist"])
        _, camera_samples = session.tick_calls[0]
        self.assertEqual([sample.name for sample in camera_samples], ["head", "left_wrist"])

    def test_online_provider_rejects_unsupported_gripper_without_no_gripper(self):
        from teleop.input.teleop_input_provider import OnlineInferenceInputProvider

        with self.assertRaises(ValueError):
            OnlineInferenceInputProvider(
                session=mock.Mock(),
                arm_side="left",
                ee="dex3",
                no_gripper=False,
            )

    def test_create_online_provider_requires_transform_config_for_real_motion_before_connecting(self):
        import teleop.input.teleop_input_provider as input_provider_module

        args = SimpleNamespace(
            input_provider="online_inference",
            online_inference_host="127.0.0.1",
            online_inference_port=5555,
            online_inference_arm_side="left",
            online_inference_n_obs_steps=2,
            online_inference_camera_freq=30.0,
            online_inference_jpeg_quality=85,
            online_inference_action_step_sec=0.10,
            online_inference_interp_sec=0.01,
            online_inference_post_action_delay_ms=75,
            online_inference_response_timeout_sec=2.0,
            online_inference_transform_config="",
            online_inference_enable_motion=True,
            online_inference_dry_run=False,
            ee="dex1",
            no_gripper=False,
        )

        with mock.patch.object(input_provider_module, "TcpJsonTransport") as transport_cls:
            with self.assertRaises(ValueError):
                input_provider_module.create_teleop_input_provider(args)

        transport_cls.connect.assert_not_called()

    def test_create_online_provider_passes_chunk_step_mode(self):
        import teleop.input.teleop_input_provider as input_provider_module

        args = SimpleNamespace(
            input_provider="online_inference",
            online_inference_host="127.0.0.1",
            online_inference_port=5555,
            online_inference_arm_side="left",
            online_inference_n_obs_steps=2,
            online_inference_camera_freq=30.0,
            online_inference_jpeg_quality=85,
            online_inference_action_step_sec=0.10,
            online_inference_chunk_step_mode="per_tick",
            online_inference_interp_sec=0.01,
            online_inference_post_action_delay_ms=75,
            online_inference_response_timeout_sec=2.0,
            online_inference_transform_config="",
            online_inference_enable_motion=False,
            online_inference_dry_run=False,
            ee="dex1",
            no_gripper=False,
        )

        with mock.patch.object(input_provider_module.TcpJsonTransport, "connect", return_value=mock.Mock()):
            provider = input_provider_module.create_teleop_input_provider(args)

        self.assertEqual(provider.session.config.chunk_step_mode, "per_tick")

    def test_create_online_provider_pi05_uses_http_transport_profile_and_prompt(self):
        import teleop.input.teleop_input_provider as input_provider_module

        args = SimpleNamespace(
            input_provider="online_inference",
            online_inference_transport="http",
            online_inference_base_url="http://115.190.134.186:8017",
            online_inference_http_handshake_path="/handshake",
            online_inference_http_infer_path="/infer",
            online_inference_protocol_profile="pi05_dual_arm_20d",
            online_inference_prompt="pick up the cube",
            online_inference_host="115.190.134.186",
            online_inference_port=8017,
            online_inference_arm_side="both",
            online_inference_n_obs_steps=2,
            online_inference_camera_freq=30.0,
            online_inference_jpeg_quality=85,
            online_inference_action_step_sec=0.10,
            online_inference_chunk_step_mode="per_tick",
            online_inference_interp_sec=0.01,
            online_inference_post_action_delay_ms=75,
            online_inference_response_timeout_sec=2.0,
            online_inference_transform_config="",
            online_inference_enable_motion=False,
            online_inference_dry_run=True,
            ee="dex1",
            no_gripper=False,
        )

        fake_transport = mock.Mock()
        with mock.patch.object(input_provider_module.HttpJsonInferenceTransport, "connect", return_value=fake_transport) as connect:
            provider = input_provider_module.create_teleop_input_provider(args)

        connect.assert_called_once_with(
            base_url="http://115.190.134.186:8017",
            handshake_path="/handshake",
            infer_path="/infer",
            timeout_sec=2.0,
            handshake_payload={
                "action_dim": 20,
                "action_space": "pose20",
                "robot": "nero_dual_arm",
                "transport": "http",
            },
        )
        self.assertIs(provider.session.transport, fake_transport)
        self.assertEqual(provider.session.config.protocol_profile, "pi05_dual_arm_20d")
        self.assertEqual(provider.session.config.task_prompt, "pick up the cube")


if __name__ == "__main__":
    unittest.main()
