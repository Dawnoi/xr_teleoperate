import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


class OfflineDDSReplayScriptTest(unittest.TestCase):
    def test_cli_prints_joint_targets_in_dry_run(self):
        from tests.manual import offline_dds_replay_probe

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root)

            with mock.patch("builtins.print") as print_mock:
                exit_code = offline_dds_replay_probe.main(
                    [
                        "--dataset-root",
                        str(dataset_root),
                        "--episode-index",
                        "0",
                    ]
                )

        self.assertEqual(exit_code, 0)
        printed = " ".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list)
        self.assertIn("frame=0", printed)
        self.assertIn("arm_q=", printed)
        self.assertIn("gripper_q=", printed)

    def test_cli_builds_dds_smoke_path(self):
        from tests.manual import offline_dds_replay_probe

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root)

            with mock.patch.object(offline_dds_replay_probe, "_initialize_robot_runtime") as init_mock:
                with mock.patch.object(offline_dds_replay_probe, "_publish_episode") as publish_mock:
                    exit_code = offline_dds_replay_probe.main(
                        [
                            "--dataset-root",
                            str(dataset_root),
                            "--episode-index",
                            "0",
                            "--dds-smoke",
                            "--max-frames",
                            "1",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        init_mock.assert_called_once()
        publish_mock.assert_called_once()

    def test_dds_smoke_routes_joint_position_samples_to_ctrl_dual_arm(self):
        from tests.manual import offline_dds_replay_probe

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root)

            arm_ctrl = mock.Mock()
            samples = list(offline_dds_replay_probe._iter_samples(
                mock.Mock(
                    dataset_root=str(dataset_root),
                    episode_index=0,
                    arm_source="action",
                    max_frames=1,
                    speed_scale=0.0,
                    no_timing=True,
                )
            ))
            with mock.patch.object(offline_dds_replay_probe.time, "sleep") as sleep_mock:
                offline_dds_replay_probe._publish_episode(
                    mock.Mock(publish_frequency=30.0, motion=False, max_arm_joint_speed=0.5, settle_time=0.0),
                    arm_ctrl,
                    samples,
                )

        arm_ctrl.ctrl_dual_arm.assert_called_once()
        sleep_mock.assert_called()


if __name__ == "__main__":
    unittest.main()
