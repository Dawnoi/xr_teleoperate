import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class _DummyArmCtrl:
    def __init__(self):
        self.calls = []
        self.arm_velocity_limit = None

    def ctrl_dual_arm_go_home(self):
        self.calls.append(("home",))

    def ctrl_dual_arm(self, q, tauff):
        self.calls.append(("arm", list(q), list(tauff)))


class _DummyArmIk:
    pass


def _fake_initialize_robot_runtime(_network_interface, _motion):
    return _DummyArmIk(), _DummyArmCtrl()


def _fake_compute_gravity_tauff(_arm_ik, arm_q):
    return np.asarray(arm_q, dtype=float) * 0.0 + 1.0


from teleop import replay_lerobot_real


def _write_episode(dataset_root: pathlib.Path, episode_index: int = 0):
    data_dir = dataset_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    table = pa.table(
        {
            "timestamp": pa.array([0.0, 0.1], type=pa.float64()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "episode_index": pa.array([episode_index, episode_index], type=pa.int64()),
            "action": pa.array(
                [
                    [float(i) for i in range(16)],
                    [float(i + 100) for i in range(16)],
                ],
                type=pa.list_(pa.float64()),
            ),
        }
    )
    pq.write_table(table, data_dir / f"episode_{episode_index:06d}.parquet")


def _write_sidecar(dataset_root: pathlib.Path, episode_index: int = 0):
    sidecar_dir = dataset_root / "extras" / "control" / "chunk-000"
    sidecar_dir.mkdir(parents=True)
    rows = [
        {
            "schema_version": 1,
            "episode_index": episode_index,
            "frame_index": 0,
            "timestamp": 0.0,
            "sample_monotonic_ns": 1000,
            "arm_tauff": [float(i) for i in range(14)],
            "source": "teleop_runtime",
        },
        {
            "schema_version": 1,
            "episode_index": episode_index,
            "frame_index": 1,
            "timestamp": 0.1,
            "sample_monotonic_ns": 2000,
            "arm_tauff": [float("nan")] + [0.0] * 13,
            "source": "teleop_runtime",
        },
    ]
    with open(sidecar_dir / f"episode_{episode_index:06d}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def _write_empty_sidecar(dataset_root: pathlib.Path, episode_index: int = 0):
    sidecar_dir = dataset_root / "extras" / "control" / "chunk-000"
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / f"episode_{episode_index:06d}.jsonl").write_text("", encoding="utf-8")


class ReplayLeRobotRealTest(unittest.TestCase):
    def test_split_action_uses_lerobot_joint_order(self):
        split = replay_lerobot_real.split_action([float(i) for i in range(16)])
        self.assertEqual(split.left_arm7.tolist(), [float(i) for i in range(7)])
        self.assertEqual(split.left_gripper, 7.0)
        self.assertEqual(split.right_arm7.tolist(), [float(i) for i in range(8, 15)])
        self.assertEqual(split.right_gripper, 15.0)
        self.assertEqual(split.arm_q.tolist(), [float(i) for i in range(7)] + [float(i) for i in range(8, 15)])

    def test_load_episode_prefers_valid_recorded_tauff_and_falls_back_for_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root)
            _write_sidecar(dataset_root)

            with mock.patch.object(replay_lerobot_real, "warn") as warn_mock:
                frames = replay_lerobot_real.load_replay_frames(
                    dataset_root=dataset_root,
                    episode_index=0,
                    use_recorded_tauff=True,
                )

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].recorded_tauff.tolist(), [float(i) for i in range(14)])
        self.assertIsNone(frames[1].recorded_tauff)
        warn_mock.assert_called()
        self.assertTrue(any("invalid recorded tauff" in call.args[0] for call in warn_mock.call_args_list))

    def test_load_episode_warns_once_when_recorded_tauff_sidecar_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root)

            with mock.patch.object(replay_lerobot_real, "warn") as warn_mock:
                frames = replay_lerobot_real.load_replay_frames(
                    dataset_root=dataset_root,
                    episode_index=0,
                    use_recorded_tauff=True,
                )

        self.assertEqual(len(frames), 2)
        self.assertTrue(all(frame.recorded_tauff is None for frame in frames))
        warn_mock.assert_called_once()
        self.assertIn("control sidecar missing", warn_mock.call_args.args[0])

    def test_load_episode_warns_when_sidecar_has_no_matching_tauff_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root)
            _write_empty_sidecar(dataset_root)

            with mock.patch.object(replay_lerobot_real, "warn") as warn_mock:
                frames = replay_lerobot_real.load_replay_frames(
                    dataset_root=dataset_root,
                    episode_index=0,
                    use_recorded_tauff=True,
                )

        self.assertEqual(len(frames), 2)
        self.assertTrue(all(frame.recorded_tauff is None for frame in frames))
        self.assertTrue(any("recorded tauff missing for 2 frame(s)" in call.args[0] for call in warn_mock.call_args_list))

    def test_dry_run_does_not_initialize_robot_runtime(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root)

            with mock.patch.object(replay_lerobot_real, "initialize_robot_runtime") as init_mock:
                exit_code = replay_lerobot_real.main([
                    "--dataset-root",
                    str(dataset_root),
                    "--episode-index",
                    "0",
                    "--dry-run",
                ])

        self.assertEqual(exit_code, 0)
        init_mock.assert_not_called()

    def test_main_uses_recorded_tauff_when_available(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root)
            _write_sidecar(dataset_root)
            with mock.patch.object(replay_lerobot_real, "initialize_robot_runtime", _fake_initialize_robot_runtime):
                with mock.patch.object(replay_lerobot_real, "compute_gravity_tauff", _fake_compute_gravity_tauff):
                    exit_code = replay_lerobot_real.main([
                        "--dataset-root",
                        str(dataset_root),
                        "--episode-index",
                        "0",
                        "--use-recorded-tauff",
                        "--no-gripper",
                    ])

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
