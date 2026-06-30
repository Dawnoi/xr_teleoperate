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
from tests.test_lerobot_v2_writer_alignment_sidecar import (
    _FakeVideoCapture,
    _FakeVideoWriter,
    _fake_video_writer,
)
from tests.test_lerobot_v2_writer_rerun import _FakeArmIk, _fake_compute_fk, _sample


class LeRobotV2WriterControlSidecarTest(unittest.TestCase):
    def test_save_episode_writes_control_sidecar_without_changing_parquet_schema(self):
        _FakeVideoWriter.by_path.clear()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_compute_fk_for_qpos", _fake_compute_fk):
                with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_video_writer", _fake_video_writer):
                    with mock.patch.object(lerobot_v2_writer.cv2, "VideoCapture", _FakeVideoCapture):
                        writer = lerobot_v2_writer.LeRobotV2Writer(
                            task_dir=tmp_dir,
                            arm_ik=_FakeArmIk(),
                            rerun_log=False,
                        )
                        try:
                            self.assertTrue(writer.create_episode())
                            writer.add_item(
                                **_sample(),
                                control_extras={"arm_tauff": [float(i) for i in range(14)]},
                            )
                            writer.add_item(
                                **_sample(),
                                control_extras={"arm_tauff": [float(i + 10) for i in range(14)]},
                            )
                            writer._item_queue.join()
                            writer.save_episode()
                            writer._item_queue.join()
                            while not writer.is_ready():
                                time.sleep(0.01)
                        finally:
                            writer.close()

            sidecar_path = pathlib.Path(tmp_dir) / "extras" / "control" / "chunk-000" / "episode_000000.jsonl"
            self.assertTrue(sidecar_path.exists())
            rows = [json.loads(line) for line in sidecar_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["schema_version"], 1)
            self.assertEqual(rows[0]["episode_index"], 0)
            self.assertEqual(rows[0]["frame_index"], 0)
            self.assertEqual(rows[0]["source"], "teleop_runtime")
            self.assertEqual(rows[0]["arm_tauff"], [float(i) for i in range(14)])
            self.assertEqual(len(rows[1]["arm_tauff"]), 14)

            info_path = pathlib.Path(tmp_dir) / "meta" / "info.json"
            info = json.loads(info_path.read_text())
            self.assertNotIn("action.tauff", info["features"])
            self.assertNotIn("control_extras", info["features"])

    def test_invalid_control_tauff_fails_episode_before_meta_update(self):
        self._assert_invalid_control_tauff_fails([0.0] * 13)

    def test_non_finite_control_tauff_fails_episode_before_meta_update(self):
        self._assert_invalid_control_tauff_fails([float("nan")] + [0.0] * 13)

    def _assert_invalid_control_tauff_fails(self, arm_tauff):
        _FakeVideoWriter.by_path.clear()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_compute_fk_for_qpos", _fake_compute_fk):
                with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_video_writer", _fake_video_writer):
                    with mock.patch.object(lerobot_v2_writer.cv2, "VideoCapture", _FakeVideoCapture):
                        writer = lerobot_v2_writer.LeRobotV2Writer(
                            task_dir=tmp_dir,
                            arm_ik=_FakeArmIk(),
                            rerun_log=False,
                        )
                        try:
                            self.assertTrue(writer.create_episode())
                            writer.add_item(
                                **_sample(),
                                control_extras={"arm_tauff": arm_tauff},
                            )
                            writer._item_queue.join()
                            writer.save_episode()
                            writer._item_queue.join()
                            while not writer.is_ready():
                                time.sleep(0.01)
                        finally:
                            writer.close()

            sidecar_path = pathlib.Path(tmp_dir) / "extras" / "control" / "chunk-000" / "episode_000000.jsonl"
            self.assertFalse(sidecar_path.exists())
            episodes_path = pathlib.Path(tmp_dir) / "meta" / "episodes.jsonl"
            rows = [line for line in episodes_path.read_text().splitlines() if line.strip()]
            self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
