import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from unittest import mock

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "replay_raw_episode_real.py"
SPEC = importlib.util.spec_from_file_location("replay_raw_episode_real", SCRIPT_PATH)
replay_raw_episode_real = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay_raw_episode_real)


def _write_raw_episode(task_root: pathlib.Path, episode_index: int = 1):
    episode_dir = task_root / f"episode_{episode_index:04d}"
    (episode_dir / "colors" / "head").mkdir(parents=True, exist_ok=True)
    (episode_dir / "colors" / "wrist_left").mkdir(parents=True, exist_ok=True)
    (episode_dir / "colors" / "wrist_right").mkdir(parents=True, exist_ok=True)
    image = np.full((8, 10, 3), 64, dtype=np.uint8)
    for frame_index in range(2):
        for camera_name in ("head", "wrist_left", "wrist_right"):
            image_path = episode_dir / "colors" / camera_name / f"{frame_index:06d}_{camera_name}.jpg"
            ok = cv2.imwrite(str(image_path), image)
            if not ok:
                raise RuntimeError(f"failed to write test image: {image_path}")

    payload = {
        "info": {},
        "text": {},
        "data": [
            {
                "idx": 0,
                "colors": {
                    "head": "colors/head/000000_head.jpg",
                    "left_wrist": "colors/wrist_left/000000_wrist_left.jpg",
                    "right_wrist": "colors/wrist_right/000000_wrist_right.jpg",
                },
                "states": {
                    "left_arm": {"qpos": [20.0 + i for i in range(7)]},
                    "right_arm": {"qpos": [40.0 + i for i in range(7)]},
                    "left_ee": {"qpos": [1.5]},
                    "right_ee": {"qpos": [2.5]},
                },
                "actions": {
                    "left_arm": {"qpos": [float(i) for i in range(7)]},
                    "right_arm": {"qpos": [10.0 + i for i in range(7)]},
                    "left_ee": {"qpos": [3.5]},
                    "right_ee": {"qpos": [4.5]},
                },
                "timestamps": {"sample_monotonic_ns": 1_000_000_000},
            },
            {
                "idx": 1,
                "colors": {
                    "head": "colors/head/000001_head.jpg",
                    "left_wrist": "colors/wrist_left/000001_wrist_left.jpg",
                    "right_wrist": "colors/wrist_right/000001_wrist_right.jpg",
                },
                "states": {
                    "left_arm": {"qpos": [21.0 + i for i in range(7)]},
                    "right_arm": {"qpos": [41.0 + i for i in range(7)]},
                    "left_ee": {"qpos": [2.5]},
                    "right_ee": {"qpos": [3.5]},
                },
                "actions": {
                    "left_arm": {"qpos": [1.0 + i for i in range(7)]},
                    "right_arm": {"qpos": [11.0 + i for i in range(7)]},
                    "left_ee": {"qpos": [4.5]},
                    "right_ee": {"qpos": [5.5]},
                },
                "timestamps": {"sample_monotonic_ns": 1_033_333_333},
            },
        ],
    }
    (episode_dir / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    return episode_dir


class ReplayRawEpisodeRealTest(unittest.TestCase):
    def test_load_dry_run_summary_reports_frame_range_and_replay_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root, episode_index=1)

            summary = replay_raw_episode_real.load_dry_run_summary(
                dataset_root=task_root,
                episode_index=1,
                arm_source="action",
            )

        self.assertEqual(summary["frame_count"], 2)
        self.assertEqual(summary["sample_monotonic_ns_range"], (1_000_000_000, 1_033_333_333))
        self.assertEqual(summary["motion_repr"], "qpos")
        self.assertEqual(summary["arm_source"], "action")

    def test_dry_run_prints_summary_and_does_not_call_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root, episode_index=1)

            with mock.patch.object(replay_raw_episode_real.subprocess, "call") as call_mock:
                with mock.patch("builtins.print") as print_mock:
                    exit_code = replay_raw_episode_real.main(
                        [
                            "--dataset-root",
                            str(task_root),
                            "--episode-index",
                            "1",
                            "--dry-run",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        call_mock.assert_not_called()
        printed = " ".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list)
        self.assertIn("frames=2", printed)
        self.assertIn("sample_monotonic_ns=1000000000->1033333333", printed)
        self.assertIn("motion_repr=qpos", printed)

    def test_non_dry_run_builds_expected_subprocess_command(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root, episode_index=1)

            with mock.patch.object(replay_raw_episode_real.subprocess, "call", return_value=9) as call_mock:
                exit_code = replay_raw_episode_real.main(
                    [
                        "--dataset-root",
                        str(task_root),
                        "--episode-index",
                        "1",
                        "--network-interface",
                        "eth0",
                        "--max-arm-joint-speed",
                        "0.8",
                        "--speed-scale",
                        "1.2",
                        "--arm-source",
                        "state",
                        "--end-action",
                        "hold",
                    ]
                )

        self.assertEqual(exit_code, 9)
        call_mock.assert_called_once()
        cmd = call_mock.call_args.args[0]
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1], str((REPO_ROOT / "teleop" / "teleop_hand_and_arm.py").resolve()))
        self.assertIn("--input-provider", cmd)
        self.assertIn("lerobot_offline", cmd)
        self.assertIn("--offline-replay-dataset-root", cmd)
        self.assertIn(str(task_root), cmd)
        self.assertIn("--offline-replay-episode-index", cmd)
        self.assertIn("1", cmd)
        self.assertIn("--offline-replay-arm-source", cmd)
        self.assertIn("state", cmd)
        self.assertIn("--offline-replay-speed-scale", cmd)
        self.assertIn("1.2", cmd)
        self.assertIn("--offline-replay-end-action", cmd)
        self.assertIn("hold", cmd)
        self.assertIn("--network-interface", cmd)
        self.assertIn("eth0", cmd)
        self.assertIn("--max-arm-joint-speed", cmd)
        self.assertIn("0.8", cmd)


if __name__ == "__main__":
    unittest.main()
