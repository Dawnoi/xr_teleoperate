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

from teleop.operator.gripper_state_ui import format_gripper_state_panel, render_gripper_state_panel_image

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
        self.assertEqual(cmd[1], "-u")
        self.assertEqual(cmd[2], str((REPO_ROOT / "teleop" / "teleop_hand_and_arm.py").resolve()))
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
        self.assertNotIn("--show-state-ui", cmd)

    def test_show_state_ui_starts_sidecar_without_passing_flag_to_teleop(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root, episode_index=1)

            def fake_run_subprocess(cmd, start_event, env=None):
                del env
                start_event.set()
                return 0

            with mock.patch.object(replay_raw_episode_real, "run_subprocess_with_start_signal", side_effect=fake_run_subprocess) as run_mock:
                with mock.patch.object(replay_raw_episode_real, "run_state_ui") as ui_mock:
                    exit_code = replay_raw_episode_real.main(
                        [
                            "--dataset-root",
                            str(task_root),
                            "--episode-index",
                            "1",
                            "--show-state-ui",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        ui_mock.assert_called_once()
        cmd = run_mock.call_args.args[0]
        self.assertNotIn("--show-state-ui", cmd)

    def test_force_hold_cli_sets_replay_subprocess_environment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root, episode_index=1)

            with mock.patch.object(replay_raw_episode_real.subprocess, "call", return_value=0) as call_mock:
                exit_code = replay_raw_episode_real.main(
                    [
                        "--dataset-root",
                        str(task_root),
                        "--episode-index",
                        "1",
                        "--gripper-force-hold-tau-thresh",
                        "1.2",
                        "--gripper-force-hold-grace-sec",
                        "0.2",
                    ]
                )

        self.assertEqual(exit_code, 0)
        env = call_mock.call_args.kwargs["env"]
        self.assertEqual(env["DEX1_FORCE_HOLD_TAU_ENGAGE_THRESH"], "1.2")
        self.assertEqual(env["DEX1_FORCE_HOLD_ENGAGE_GRACE_SEC"], "0.2")

    def test_disable_force_hold_sets_high_replay_subprocess_threshold(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root, episode_index=1)

            with mock.patch.object(replay_raw_episode_real.subprocess, "call", return_value=0) as call_mock:
                exit_code = replay_raw_episode_real.main(
                    [
                        "--dataset-root",
                        str(task_root),
                        "--episode-index",
                        "1",
                        "--disable-gripper-force-hold",
                    ]
                )

        self.assertEqual(exit_code, 0)
        env = call_mock.call_args.kwargs["env"]
        self.assertEqual(env["DEX1_FORCE_HOLD_TAU_ENGAGE_THRESH"], "999")

    def test_disable_force_hold_conflicts_with_explicit_threshold(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root, episode_index=1)

            with self.assertRaises(ValueError):
                replay_raw_episode_real.main(
                    [
                        "--dataset-root",
                        str(task_root),
                        "--episode-index",
                        "1",
                        "--disable-gripper-force-hold",
                        "--gripper-force-hold-tau-thresh",
                        "1.0",
                    ]
                )

    def test_raw_episode_timeline_does_not_advance_before_start_signal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root, episode_index=1)

            with mock.patch.object(replay_raw_episode_real.time, "time", return_value=100.0):
                timeline = replay_raw_episode_real.RawEpisodeTimeline(
                    dataset_root=task_root,
                    episode_index=1,
                    arm_source="action",
                    speed_scale=1.0,
                )
            with mock.patch.object(replay_raw_episode_real.time, "time", return_value=101.0):
                before_start = timeline.current()
            with mock.patch.object(replay_raw_episode_real.time, "time", return_value=200.0):
                timeline.start()
            with mock.patch.object(replay_raw_episode_real.time, "time", return_value=200.04):
                after_start = timeline.current()

        self.assertEqual(before_start.frame_index, 0)
        self.assertEqual(after_start.frame_index, 1)

    def test_subprocess_start_signal_is_set_from_tracking_log(self):
        process = mock.Mock()
        process.stdout = iter(["booting\\n", "---------------------🚀start Tracking🚀-------------------------\\n"])
        process.wait.return_value = 7

        start_event = replay_raw_episode_real.threading.Event()
        with mock.patch.object(replay_raw_episode_real.subprocess, "Popen", return_value=process):
            exit_code = replay_raw_episode_real.run_subprocess_with_start_signal(["python"], start_event)

        self.assertEqual(exit_code, 7)
        self.assertTrue(start_event.is_set())

    def test_raw_episode_timeline_reports_current_gripper_qpos(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root, episode_index=1)

            timeline = replay_raw_episode_real.RawEpisodeTimeline(
                dataset_root=task_root,
                episode_index=1,
                arm_source="action",
                speed_scale=0.0,
            )
            sample = timeline.current()

        self.assertEqual(sample.frame_index, 0)
        self.assertEqual(sample.gripper_q, [3.5, 4.5])
        self.assertEqual(sample.source, "raw_episode:actions:qpos")

    def test_state_ui_snapshot_reports_clip_prediction_and_elapsed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_root = pathlib.Path(tmp_dir)
            _write_raw_episode(task_root, episode_index=1)

            with mock.patch.object(replay_raw_episode_real.time, "time", return_value=10.0):
                timeline = replay_raw_episode_real.RawEpisodeTimeline(
                    dataset_root=task_root,
                    episode_index=1,
                    arm_source="action",
                    speed_scale=1.0,
                )
                timeline.start()
            with mock.patch.object(replay_raw_episode_real.time, "time", return_value=10.04):
                snapshot = replay_raw_episode_real._state_ui_snapshot(
                    timeline=timeline,
                    probe_snapshot={
                        "dds_command_q": [3.1, 2.8],
                        "feedback_state": [3.2, 5.0],
                        "tau_est": [0.2, 0.7],
                    },
                    episode_index=1,
                    speed_scale=1.0,
                    status="RUNNING",
                    start_source="dds_cmd_fallback",
                    force_hold_tau_thresh=0.55,
                )

        self.assertEqual(snapshot["frame_index"], 1)
        self.assertEqual(snapshot["clip_predicted_q"], [4.0, 5.5])
        self.assertEqual(snapshot["tau_high"], [False, True])
        self.assertAlmostEqual(snapshot["ui_elapsed_sec"], 0.04)
        self.assertEqual(snapshot["cmd_minus_raw"], [-1.4, -2.7])
        self.assertEqual(snapshot["cmd_minus_clip_pred"], [-0.8999999999999999, -2.7])

    def test_gripper_state_panel_contains_raw_qpos_command_action_and_feedback_state(self):
        snapshot = {
            "episode_index": 130,
            "frame_index": 96,
            "source": "raw_episode:action:qpos",
            "raw_gripper_qpos": [3.4, 1.6],
            "trigger_value": [6.259259, 5.592593],
            "clip_predicted_q": [3.1, 2.1],
            "dds_command_q": [3.1, 2.8],
            "cmd_minus_raw": [-0.3, 1.2],
            "cmd_minus_clip_pred": [0.0, 0.7],
            "feedback_state": [3.2, 2.9],
            "tau_est": [0.1, 0.7],
            "tau_high": [False, True],
            "tau_high_threshold": 0.55,
            "left_enabled": True,
            "right_enabled": True,
            "ui_start_source": "dds_cmd_fallback",
            "ui_elapsed_sec": 1.2,
            "speed_scale": 1.0,
        }
        text = format_gripper_state_panel(snapshot)

        self.assertIn("episode=0130 frame=96", text)
        self.assertIn("ui_start=dds_cmd_fallback", text)
        self.assertIn("raw_qpos", text)
        self.assertIn("trigger", text)
        self.assertIn("clip_pred", text)
        self.assertIn("dds_cmd", text)
        self.assertIn("cmd-raw", text)
        self.assertIn("cmd-clip", text)
        self.assertIn("state", text)
        self.assertIn("tau_est", text)
        self.assertIn("tau_high", text)
        self.assertIn("thresh=0.550", text)
        self.assertIn("force_hold", text)
        self.assertIn("1.6000", text)
        self.assertIn("2.9000", text)
        self.assertIn("2.8000", text)
        self.assertIn("True", text)

        image = render_gripper_state_panel_image(snapshot)
        self.assertEqual(image.shape, (560, 980, 3))
        self.assertGreater(int(np.max(image)), 30)


if __name__ == "__main__":
    unittest.main()
