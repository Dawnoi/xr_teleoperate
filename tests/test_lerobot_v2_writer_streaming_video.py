import pathlib
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.recording import lerobot_v2_writer
from tests.test_lerobot_v2_writer_rerun import _FakeArmIk, _fake_compute_fk, _sample


class _FakeVideoWriter:
    instances = []

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        self.frames = []
        self.released = False
        self.__class__.instances.append(self)

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True
        self.path.write_bytes(b"fake-video")


def _fake_video_writer(self, path, size, fps):
    return _FakeVideoWriter(path), "FAKE"


class LeRobotV2WriterStreamingVideoTest(unittest.TestCase):
    def test_accepted_samples_do_not_retain_raw_image_frames(self):
        _FakeVideoWriter.instances.clear()
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
                        writer.add_item(**_sample())
                        writer._item_queue.join()

                        self.assertEqual(len(_FakeVideoWriter.instances), 3)
                        self.assertTrue(all(len(instance.frames) == 1 for instance in _FakeVideoWriter.instances))

                        self.assertEqual(len(writer._current_samples), 1)
                        sample = writer._current_samples[0]
                        for slot in lerobot_v2_writer.CAMERA_SLOTS:
                            self.assertNotIn("frame", sample["images"][slot])
                            self.assertIsInstance(sample["images"][slot]["path"], str)
                            self.assertIsInstance(sample["images"][slot]["timestamp_s"], float)
                        writer._reset_episode_state()
                    finally:
                        writer.close()


if __name__ == "__main__":
    unittest.main()
