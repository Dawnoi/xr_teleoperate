import tempfile
import time
import unittest
import pathlib
import sys
from unittest import mock

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.recording import lerobot_v2_writer


class _FakeVideoWriter:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.path.write_bytes(b"fake-video")


def _fake_video_writer(self, path, size, fps):
    return _FakeVideoWriter(path), "FAKE"


class _FakeFrame:
    def __init__(self, name):
        self.name = name


class _FakeModel:
    def __init__(self):
        names = (
            lerobot_v2_writer.LEFT_ARM_LINK_NAMES
            + ["L_ee"]
            + lerobot_v2_writer.RIGHT_ARM_LINK_NAMES
            + ["R_ee"]
        )
        self.frames = [_FakeFrame(name) for name in names]
        self.nframes = len(self.frames)
        self._name_to_id = {name: idx for idx, name in enumerate(names)}

    def createData(self):
        return object()

    def getFrameId(self, frame_name):
        return self._name_to_id.get(frame_name, self.nframes)


class _FakeReducedRobot:
    def __init__(self):
        self.model = _FakeModel()


class _FakeArmIk:
    def __init__(self):
        self.reduced_robot = _FakeReducedRobot()


class _FakeRerunLogger:
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logged_items = []
        self.closed = False
        self.__class__.created.append(self)

    def log_item_data(self, item_data):
        self.logged_items.append(item_data)

    def close(self):
        self.closed = True


def _fake_compute_fk(self, arm_qpos, group):
    out = {}
    for key in self._fk_frame_ids:
        if key.startswith(f"observation.fk.{group}."):
            out[key] = [0.0] * 6
    return out


def _patch_fk():
    return mock.patch.object(
        lerobot_v2_writer.LeRobotV2Writer,
        "_compute_fk_for_qpos",
        _fake_compute_fk,
    )


def _sample():
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    q7 = [0.0] * 7
    q1 = [0.0]
    state_or_action = {
        "left_arm": {"qpos": q7},
        "right_arm": {"qpos": q7},
        "left_ee": {"qpos": q1},
        "right_ee": {"qpos": q1},
    }
    return {
        "colors": {
            "head": image,
            "left_wrist": image,
            "right_wrist": image,
        },
        "states": state_or_action,
        "actions": state_or_action,
        "timestamps": {"sample_monotonic_ns": time.monotonic_ns()},
    }


def _fake_write_episode_artifacts(self, samples):
    return {
        "validation_passed": True,
        "episode_index": int(self._current_episode_index),
        "task_index": int(self._current_task_index),
        "num_samples": len(samples),
        "parquet": {"path": "data/chunk-000/episode_000000.parquet", "row_count": len(samples)},
        "videos": {},
    }


class LeRobotV2WriterRerunTest(unittest.TestCase):
    def test_rerun_log_true_creates_live_logger_and_logs_samples(self):
        _FakeRerunLogger.created.clear()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(lerobot_v2_writer, "RerunLogger", _FakeRerunLogger):
                with _patch_fk():
                    with mock.patch.object(
                        lerobot_v2_writer.LeRobotV2Writer,
                        "_write_episode_artifacts",
                        _fake_write_episode_artifacts,
                    ):
                        with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_video_writer", _fake_video_writer):
                            writer = lerobot_v2_writer.LeRobotV2Writer(
                                task_dir=tmp_dir,
                                arm_ik=_FakeArmIk(),
                                rerun_log=True,
                            )
                            try:
                                self.assertTrue(writer.create_episode())
                                self.assertEqual(len(_FakeRerunLogger.created), 0)

                                writer.add_item(**_sample())
                                writer._item_queue.join()

                                self.assertEqual(len(_FakeRerunLogger.created), 1)
                                logger = _FakeRerunLogger.created[0]
                                self.assertEqual(logger.kwargs["prefix"], "online/")
                                self.assertTrue(logger.kwargs["spawn_viewer"])
                                self.assertIsNone(logger.kwargs["rrd_path"])

                                self.assertEqual(len(logger.logged_items), 1)
                                logged = logger.logged_items[0]
                                self.assertEqual(logged["idx"], 0)
                                self.assertEqual(sorted(logged["colors"]), ["head", "left_wrist", "right_wrist"])

                                writer.save_episode()
                                writer._item_queue.join()
                                while not writer.is_ready():
                                    time.sleep(0.01)

                                self.assertTrue(logger.closed)
                            finally:
                                writer.close()


if __name__ == "__main__":
    unittest.main()
