import json
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.recording import lerobot_v2_writer
from tests.test_lerobot_v2_writer_rerun import _FakeArmIk, _fake_compute_fk, _fake_video_writer, _sample


class LeRobotV2WriterStatsTest(unittest.TestCase):
    def test_episode_stats_include_runtime_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_compute_fk_for_qpos", _fake_compute_fk):
                with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_write_episode_artifacts") as write_mock:
                    write_mock.side_effect = lambda samples: {
                        "validation_passed": True,
                        "episode_index": int(samples[0]["frame_index"]),
                        "task_index": 0,
                        "num_samples": len(samples),
                        "parquet": {"path": "data/chunk-000/episode_000000.parquet", "row_count": len(samples)},
                        "videos": {},
                    }
                    with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_video_writer", _fake_video_writer):
                        writer = lerobot_v2_writer.LeRobotV2Writer(
                            task_dir=tmp_dir,
                            arm_ik=_FakeArmIk(),
                            rerun_log=False,
                        )
                        try:
                            self.assertTrue(writer.create_episode())
                            writer.add_item(**_sample())
                            writer._item_queue.join()
                            writer.save_episode()
                            writer._item_queue.join()
                            while not writer.is_ready():
                                time.sleep(0.01)
                        finally:
                            writer.close()

            stats_path = pathlib.Path(tmp_dir) / "meta" / "episodes_stats.jsonl"
            rows = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
            self.assertTrue(rows)
            stats = rows[0]["stats"]
            self.assertIn("runtime", stats)
            self.assertIn("accepted_samples", stats["runtime"])
            self.assertIn("queue", stats["runtime"])
            self.assertIn("processing", stats["runtime"])

    def test_cancel_episode_reuses_episode_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_compute_fk_for_qpos", _fake_compute_fk):
                with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_video_writer", _fake_video_writer):
                    writer = lerobot_v2_writer.LeRobotV2Writer(
                        task_dir=tmp_dir,
                        arm_ik=_FakeArmIk(),
                        rerun_log=False,
                    )
                    try:
                        self.assertTrue(writer.create_episode())
                        self.assertEqual(writer._current_episode_index, 0)
                        self.assertEqual(writer._next_episode_index, 1)

                        writer.cancel_episode()
                        self.assertTrue(writer.is_ready())
                        self.assertEqual(writer._next_episode_index, 0)

                        self.assertTrue(writer.create_episode())
                        self.assertEqual(writer._current_episode_index, 0)
                    finally:
                        writer.cancel_episode()
                        writer.close()


if __name__ == "__main__":
    unittest.main()
