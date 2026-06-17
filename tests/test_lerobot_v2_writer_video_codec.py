import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.utils import lerobot_v2_writer
from tests.test_lerobot_v2_writer_rerun import _FakeArmIk


class LeRobotV2WriterVideoCodecTest(unittest.TestCase):
    def test_video_writer_defaults_to_opencv_mjpeg_mp4(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = lerobot_v2_writer.LeRobotV2Writer(
                task_dir=tmp_dir,
                arm_ik=_FakeArmIk(),
                rerun_log=False,
            )
            try:
                config = writer._video_encoder_config()
                self.assertEqual(config["backend"], "opencv")
                self.assertEqual(config["codec"], "MJPG")
                self.assertEqual(config["container"], "mp4")
                self.assertEqual(config["fourccs"][0], "MJPG")
            finally:
                writer.close()

    def test_ffmpeg_backend_remains_available_when_explicitly_requested(self):
        with mock.patch.dict(os.environ, {"LEROBOT_VIDEO_BACKEND": "ffmpeg"}):
            config = lerobot_v2_writer._video_encoder_settings()
        self.assertEqual(config["backend"], "ffmpeg")
        self.assertEqual(config["codec"], "libx264")
        self.assertEqual(config["pix_fmt"], "yuv420p")
        self.assertEqual(config["profile"], "baseline")
        self.assertEqual(config["level"], "3.0")

    def test_ffmpeg_command_includes_baseline_constraints(self):
        with mock.patch.object(lerobot_v2_writer._FFmpegMP4Writer, "_start_process"):
            writer = lerobot_v2_writer._FFmpegMP4Writer(
                path="/tmp/out.mp4",
                size=(640, 480),
                fps=30.0,
                ffmpeg_bin="ffmpeg",
                preset="veryfast",
                crf=18,
                profile="baseline",
                level="3.0",
            )
        try:
            cmd = writer._command()
            self.assertIn("-profile:v", cmd)
            self.assertIn("baseline", cmd)
            self.assertIn("-level:v", cmd)
            self.assertIn("3.0", cmd)
            self.assertIn("-bf", cmd)
            self.assertIn("0", cmd)
        finally:
            writer.release()


if __name__ == "__main__":
    unittest.main()
