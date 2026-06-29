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

from teleop.utils import lerobot_v2_writer
from tests.test_lerobot_v2_writer_rerun import _FakeArmIk, _fake_compute_fk, _sample


class _FakeVideoWriter:
    by_path = {}

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        self.frames = []
        self.released = False
        self.__class__.by_path[str(self.path)] = self

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True
        self.path.write_bytes(b"fake-video")


def _fake_video_writer(self, path, size, fps):
    return _FakeVideoWriter(path), "FAKE"


class _FakeVideoCapture:
    def __init__(self, path):
        writer = _FakeVideoWriter.by_path.get(str(pathlib.Path(path)))
        self.frames = list(writer.frames) if writer is not None else []
        self.index = 0

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        pass


def _sample_with_alignment(sample_ns, seq_base):
    sample = _sample()
    sample["timestamps"] = {
        "sample_wall_time_ns": 1_700_000_000_000_000_000 + int(sample_ns),
        "sample_monotonic_ns": int(sample_ns),
        "teleop_input_perf_counter_ns": int(sample_ns) - 2_000_000,
        "primary_camera_name": "head",
        "state": {
            "host_monotonic_ns": int(sample_ns) - 1_000_000,
            "delta_to_sample_ns": -1_000_000,
            "interpolation_mode": "linear",
            "support_span_ns": 2_000_000,
            "support_max_abs_delta_ns": 1_000_000,
        },
        "action": {
            "host_monotonic_ns": int(sample_ns) + 500_000,
            "delta_to_sample_ns": 500_000,
            "interpolation_mode": "nearest_fallback",
            "support_span_ns": 0,
            "support_max_abs_delta_ns": 500_000,
        },
        "camera": {
            "head": {
                "frame_seq": seq_base,
                "host_recv_monotonic_ns": int(sample_ns) - 3_000_000,
                "source_monotonic_ns": 10_000 + seq_base,
                "source_wall_time_ns": 20_000 + seq_base,
                "delta_to_sample_ns": -3_000_000,
                "shape": [10, 8, 3],
                "protocol": "xraw_v1",
            },
            "left_wrist": {
                "frame_seq": seq_base + 100,
                "host_recv_monotonic_ns": int(sample_ns) + 2_000_000,
                "source_monotonic_ns": 30_000 + seq_base,
                "source_wall_time_ns": 40_000 + seq_base,
                "delta_to_sample_ns": 2_000_000,
                "shape": [10, 8, 3],
                "protocol": "xraw_v1",
            },
            "right_wrist": {
                "frame_seq": seq_base + 200,
                "host_recv_monotonic_ns": int(sample_ns) + 5_000_000,
                "source_monotonic_ns": 50_000 + seq_base,
                "source_wall_time_ns": 60_000 + seq_base,
                "delta_to_sample_ns": 5_000_000,
                "shape": [10, 8, 3],
                "protocol": "xraw_v1",
            },
        },
    }
    return sample


class LeRobotV2WriterAlignmentSidecarTest(unittest.TestCase):
    def test_save_episode_writes_alignment_sidecar_without_stats_registration(self):
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
                            writer.add_item(**_sample_with_alignment(1_000_000_000, 10))
                            writer.add_item(**_sample_with_alignment(1_033_333_333, 11))
                            writer._item_queue.join()
                            writer.save_episode()
                            writer._item_queue.join()
                            while not writer.is_ready():
                                time.sleep(0.01)
                        finally:
                            writer.close()

            sidecar_path = pathlib.Path(tmp_dir) / "meta" / "alignment" / "episode_000000.jsonl"
            self.assertTrue(sidecar_path.exists())
            rows = [json.loads(line) for line in sidecar_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["episode_index"], 0)
            self.assertEqual(rows[0]["frame_index"], 0)
            self.assertEqual(rows[0]["primary_camera_name"], "head")
            self.assertEqual(rows[0]["cameras"]["cam0"]["source_name"], "head")
            self.assertEqual(rows[0]["cameras"]["cam0"]["frame_seq"], 10)
            self.assertEqual(rows[0]["cameras"]["cam1"]["source_name"], "left_wrist")
            self.assertEqual(rows[0]["cameras"]["cam1"]["delta_to_sample_ns"], 2_000_000)
            self.assertEqual(rows[0]["cameras"]["cam2"]["source_name"], "right_wrist")
            self.assertEqual(rows[0]["cameras"]["cam2"]["abs_delta_to_sample_ms"], 5.0)
            self.assertEqual(rows[0]["state"]["interpolation_mode"], "linear")
            self.assertEqual(rows[0]["action"]["interpolation_mode"], "nearest_fallback")

            stats_path = pathlib.Path(tmp_dir) / "meta" / "episodes_stats.jsonl"
            stats_rows = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
            self.assertNotIn("alignment", stats_rows[0]["stats"])


if __name__ == "__main__":
    unittest.main()
