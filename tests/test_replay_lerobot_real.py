import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import replay_lerobot_real


def _write_episode(
    dataset_root: pathlib.Path,
    episode_index: int = 0,
    include_state: bool = False,
    include_fk: bool = False,
):
    data_dir = dataset_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    columns = {
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
    if include_state:
        columns["observation.state"] = pa.array(
            [
                [float(i + 20) for i in range(16)],
                [float(i + 40) for i in range(16)],
            ],
            type=pa.list_(pa.float64()),
        )
    if include_fk:
        columns["observation.fk.cmd.left.gripper_flange"] = pa.array(
            [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0], [0.2, 0.3, 0.4, 0.1, 0.0, 0.0]],
            type=pa.list_(pa.float64()),
        )
        columns["observation.fk.cmd.right.gripper_flange"] = pa.array(
            [[0.4, 0.5, 0.6, 0.0, 0.1, 0.0], [0.5, 0.6, 0.7, 0.0, 0.1, 0.1]],
            type=pa.list_(pa.float64()),
        )
    table = pa.table(columns)
    pq.write_table(table, data_dir / f"episode_{episode_index:06d}.parquet")


def _write_sidecar(dataset_root: pathlib.Path, episode_index: int = 0):
    sidecar_dir = dataset_root / "extras" / "control" / "chunk-000"
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / f"episode_{episode_index:06d}.jsonl").write_text(
        '{"frame_index": 0, "arm_tauff": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}\n',
        encoding="utf-8",
    )


class ReplayLeRobotRealTest(unittest.TestCase):
    def test_split_action_uses_lerobot_joint_order(self):
        split = replay_lerobot_real.split_action([float(i) for i in range(16)])
        self.assertEqual(split.left_arm7.tolist(), [float(i) for i in range(7)])
        self.assertEqual(split.left_gripper, 7.0)
        self.assertEqual(split.right_arm7.tolist(), [float(i) for i in range(8, 15)])
        self.assertEqual(split.right_gripper, 15.0)
        self.assertEqual(
            split.arm_q.tolist(),
            [float(i) for i in range(7)] + [float(i) for i in range(8, 15)],
        )

    def test_load_dry_run_summary_reports_frame_range_shape_and_sidecar_presence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root, include_state=True)
            _write_sidecar(dataset_root)

            summary = replay_lerobot_real.load_dry_run_summary(
                dataset_root=dataset_root,
                episode_index=0,
                arm_source="state",
            )

        self.assertEqual(summary["frame_count"], 2)
        self.assertEqual(summary["timestamp_range"], (0.0, 0.1))
        self.assertEqual(summary["action_shape"], (16,))
        self.assertEqual(summary["sidecar"], "present")
        self.assertEqual(summary["arm_source"], "state")

    def test_dry_run_prints_summary_and_does_not_call_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root, include_fk=True)

            with mock.patch.object(replay_lerobot_real.subprocess, "call") as call_mock:
                with mock.patch("builtins.print") as print_mock:
                    exit_code = replay_lerobot_real.main(
                        [
                            "--dataset-root",
                            str(dataset_root),
                            "--episode-index",
                            "0",
                            "--dry-run",
                            "--arm-source",
                            "fk_cmd_pose",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        call_mock.assert_not_called()
        printed = " ".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list)
        self.assertIn("frames=2", printed)
        self.assertIn("timestamp=0.000000->0.100000", printed)
        self.assertIn("action_shape=(16,)", printed)
        self.assertIn("sidecar=missing", printed)
        self.assertIn("arm_source=fk_cmd_pose", printed)

    def test_dry_run_rejects_missing_state_column_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root)

            with mock.patch.object(replay_lerobot_real.subprocess, "call") as call_mock:
                with self.assertRaises(KeyError):
                    replay_lerobot_real.main(
                        [
                            "--dataset-root",
                            str(dataset_root),
                            "--episode-index",
                            "0",
                            "--dry-run",
                            "--arm-source",
                            "state",
                        ]
                    )

        call_mock.assert_not_called()

    def test_dry_run_rejects_missing_fk_columns_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root)

            with mock.patch.object(replay_lerobot_real.subprocess, "call") as call_mock:
                with self.assertRaises(KeyError):
                    replay_lerobot_real.main(
                        [
                            "--dataset-root",
                            str(dataset_root),
                            "--episode-index",
                            "0",
                            "--dry-run",
                            "--arm-source",
                            "fk_cmd_pose",
                        ]
                    )

        call_mock.assert_not_called()

    def test_non_dry_run_builds_expected_subprocess_command(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root, episode_index=3, include_state=True)

            with mock.patch.object(replay_lerobot_real.subprocess, "call", return_value=7) as call_mock:
                exit_code = replay_lerobot_real.main(
                    [
                        "--dataset-root",
                        str(dataset_root),
                        "--episode-index",
                        "3",
                        "--network-interface",
                        "eth0",
                        "--max-arm-joint-speed",
                        "0.7",
                        "--speed-scale",
                        "1.5",
                        "--arm-source",
                        "state",
                        "--end-action",
                        "hold",
                        "--motion",
                    ]
                )

        self.assertEqual(exit_code, 7)
        call_mock.assert_called_once()
        cmd = call_mock.call_args.args[0]
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1], str((REPO_ROOT / "teleop" / "teleop_hand_and_arm.py").resolve()))
        self.assertIn("--input-provider", cmd)
        self.assertIn("lerobot_offline", cmd)
        self.assertIn("--offline-replay-dataset-root", cmd)
        self.assertIn(str(dataset_root), cmd)
        self.assertIn("--offline-replay-episode-index", cmd)
        self.assertIn("3", cmd)
        self.assertIn("--offline-replay-arm-source", cmd)
        self.assertIn("state", cmd)
        self.assertIn("--offline-replay-speed-scale", cmd)
        self.assertIn("1.5", cmd)
        self.assertIn("--offline-replay-end-action", cmd)
        self.assertIn("hold", cmd)
        self.assertIn("--network-interface", cmd)
        self.assertIn("eth0", cmd)
        self.assertIn("--max-arm-joint-speed", cmd)
        self.assertIn("0.7", cmd)
        self.assertIn("--controller-deadman", cmd)
        self.assertIn("grip", cmd)
        self.assertIn("--head-reference-mode", cmd)
        self.assertIn("fixed_per_grip", cmd)
        self.assertIn("--controller-mapping-mode", cmd)
        self.assertIn("anchored_safe", cmd)
        self.assertIn("--controller-orientation-mode", cmd)
        self.assertIn("relative", cmd)
        self.assertIn("--base-controller", cmd)
        self.assertIn("none", cmd)
        self.assertIn("--headless", cmd)
        self.assertIn("--auto-start", cmd)
        self.assertIn("--motion", cmd)

    def test_non_dry_run_forwards_no_gripper_to_teleop_entry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = pathlib.Path(tmp_dir)
            _write_episode(dataset_root)

            with mock.patch.object(replay_lerobot_real.subprocess, "call", return_value=0) as call_mock:
                with mock.patch("builtins.print") as print_mock:
                    exit_code = replay_lerobot_real.main(
                        [
                            "--dataset-root",
                            str(dataset_root),
                            "--episode-index",
                            "0",
                            "--no-gripper",
                            "--use-recorded-tauff",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        cmd = call_mock.call_args.args[0]
        ee_indices = [i for i, token in enumerate(cmd) if token == "--ee"]
        self.assertEqual(len(ee_indices), 1)
        self.assertEqual(cmd[ee_indices[0] + 1], "dex1")
        self.assertIn("--no-gripper", cmd)
        printed = " ".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list)
        self.assertIn("use_recorded_tauff", printed)
        self.assertIn("runtime tauff", printed)

    def test_build_subprocess_command_omits_optional_network_interface(self):
        cmd = replay_lerobot_real.build_subprocess_command(
            dataset_root="/tmp/dataset",
            episode_index=5,
            network_interface=None,
            max_arm_joint_speed=0.9,
            speed_scale=2.0,
            motion=False,
            arm_source="action",
            end_action="home",
        )

        self.assertNotIn("--network-interface", cmd)
        self.assertIn("--offline-replay-episode-index", cmd)
        self.assertIn("5", cmd)
        self.assertIn("--offline-replay-speed-scale", cmd)
        self.assertIn("2.0", cmd)
        self.assertNotIn("--motion", cmd)


if __name__ == "__main__":
    unittest.main()
