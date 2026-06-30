import json
import pathlib
import sys
import tempfile
import types
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _DummyLogger:
    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None

        return _noop


logging_mp_stub = types.ModuleType("logging_mp")
logging_mp_stub.getLogger = lambda *_args, **_kwargs: _DummyLogger()
sys.modules.setdefault("logging_mp", logging_mp_stub)
sys.modules.setdefault("zmq", types.ModuleType("zmq"))

from teleop.recording import lerobot_v2_writer
from tests.test_lerobot_v2_writer_alignment_sidecar import _FakeVideoWriter, _fake_video_writer
from tests.test_lerobot_v2_writer_rerun import _FakeArmIk, _fake_compute_fk


class _FinalPathAwareFakeVideoCapture:
    def __init__(self, path):
        path = pathlib.Path(path)
        writer = _FakeVideoWriter.by_path.get(str(path))
        if writer is None:
            partial_path = path.with_name(path.stem + ".partial" + path.suffix)
            writer = _FakeVideoWriter.by_path.get(str(partial_path))
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


def _write_image(path: pathlib.Path, value: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 10, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _pose(x: float):
    return {
        "position": [x, x + 0.1, x + 0.2],
        "rpy": [x + 0.3, x + 0.4, x + 0.5],
        "rotation_matrix": np.eye(3, dtype=float).tolist(),
        "matrix4x4": np.eye(4, dtype=float).tolist(),
    }


def _sample(raw_episode_dir: pathlib.Path, idx: int, include_pose: bool = True):
    colors = {}
    for name in ("head", "left_wrist", "right_wrist"):
        image_path = raw_episode_dir / "colors" / name / f"{idx:06d}_{name}.jpg"
        _write_image(image_path, value=idx + 1)
        colors[name] = str(image_path.relative_to(raw_episode_dir))

    states = {
        "left_arm": {"qpos": [0.1 + idx] * 7},
        "right_arm": {"qpos": [0.2 + idx] * 7},
        "left_ee": {"qpos": [0.3 + idx]},
        "right_ee": {"qpos": [0.4 + idx]},
    }
    actions = {
        "left_arm": {"qpos": [1.1 + idx] * 7},
        "right_arm": {"qpos": [1.2 + idx] * 7},
        "left_ee": {"qpos": [1.3 + idx]},
        "right_ee": {"qpos": [1.4 + idx]},
    }
    if include_pose:
        states["left_arm"]["pose"] = _pose(0.01 + idx)
        states["right_arm"]["pose"] = _pose(0.02 + idx)
        actions["left_arm"]["pose"] = _pose(0.03 + idx)
        actions["right_arm"]["pose"] = _pose(0.04 + idx)

    return {
        "idx": idx,
        "colors": colors,
        "states": states,
        "actions": actions,
        "timestamps": {
            "sample_wall_time_ns": 1_700_000_000_000_000_000 + idx,
            "sample_monotonic_ns": 1_000_000_000 + idx * 33_333_333,
            "primary_camera_name": "head",
            "camera": {
                "head": {"frame_seq": idx, "host_recv_monotonic_ns": 1_000_000_000 + idx * 33_333_333},
                "left_wrist": {"frame_seq": idx, "host_recv_monotonic_ns": 1_000_000_000 + idx * 33_333_333},
                "right_wrist": {"frame_seq": idx, "host_recv_monotonic_ns": 1_000_000_000 + idx * 33_333_333},
            },
        },
    }


def _write_raw_episode(task_dir: pathlib.Path, episode_name: str, samples):
    episode_dir = task_dir / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "info": {},
        "text": {},
        "data": samples,
    }
    (episode_dir / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    return episode_dir


class ExportRawDataToLeRobotV2Test(unittest.TestCase):
    def test_exports_complete_pose_episode_with_raw_pose_sidecar(self):
        _FakeVideoWriter.by_path.clear()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            raw_task_dir = root / "raw"
            raw_episode_dir = raw_task_dir / "episode_0001"
            samples = [_sample(raw_episode_dir, idx, include_pose=True) for idx in range(3)]
            _write_raw_episode(raw_task_dir, "episode_0001", samples)

            from tools import export_raw_data_to_lerobot_v2 as exporter

            with mock.patch.object(exporter, "create_arm_ik", return_value=_FakeArmIk()):
                with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_compute_fk_for_qpos", _fake_compute_fk):
                    with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_video_writer", _fake_video_writer):
                        with mock.patch.object(lerobot_v2_writer.cv2, "VideoCapture", _FinalPathAwareFakeVideoCapture):
                            with mock.patch.object(exporter.cv2, "VideoCapture", _FinalPathAwareFakeVideoCapture):
                                summary = exporter.export_raw_task_dir(
                                    input_task_dir=raw_task_dir,
                                    output_root=root / "lerobot",
                                    task="pick",
                                    fps=30.0,
                                    overwrite=False,
                                )

            self.assertEqual(summary["episodes_exported"], 1)
            pose_path = root / "lerobot" / "extras" / "raw_pose" / "chunk-000" / "episode_000000.jsonl"
            self.assertTrue(pose_path.exists())
            rows = [json.loads(line) for line in pose_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["frame_index"], 0)
            self.assertEqual(rows[0]["fb"]["left"]["gripper_flange"], [0.01, 0.11, 0.21000000000000002, 0.31, 0.41000000000000003, 0.51])
            self.assertEqual(rows[0]["cmd"]["right"]["gripper_flange"], [0.04, 0.14, 0.24000000000000002, 0.33999999999999997, 0.44, 0.54])

    def test_exports_without_pose_sidecar_when_pose_is_absent(self):
        _FakeVideoWriter.by_path.clear()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            raw_task_dir = root / "raw"
            raw_episode_dir = raw_task_dir / "episode_0001"
            samples = [_sample(raw_episode_dir, idx, include_pose=False) for idx in range(2)]
            _write_raw_episode(raw_task_dir, "episode_0001", samples)

            from tools import export_raw_data_to_lerobot_v2 as exporter

            with mock.patch.object(exporter, "create_arm_ik", return_value=_FakeArmIk()):
                with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_compute_fk_for_qpos", _fake_compute_fk):
                    with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_video_writer", _fake_video_writer):
                        with mock.patch.object(lerobot_v2_writer.cv2, "VideoCapture", _FinalPathAwareFakeVideoCapture):
                            with mock.patch.object(exporter.cv2, "VideoCapture", _FinalPathAwareFakeVideoCapture):
                                summary = exporter.export_raw_task_dir(
                                    input_task_dir=raw_task_dir,
                                    output_root=root / "lerobot",
                                    task="pick",
                                    fps=30.0,
                                    overwrite=False,
                                )

            pose_path = root / "lerobot" / "extras" / "raw_pose" / "chunk-000" / "episode_000000.jsonl"
            self.assertFalse(pose_path.exists())
            self.assertFalse(summary["episodes"][0]["pose_sidecar"])
            self.assertEqual(summary["episodes"][0]["pose_missing_frame_count"], 2)

    def test_partial_pose_episode_exports_main_data_without_pose_sidecar(self):
        _FakeVideoWriter.by_path.clear()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            raw_task_dir = root / "raw"
            raw_episode_dir = raw_task_dir / "episode_0001"
            samples = [
                _sample(raw_episode_dir, 0, include_pose=True),
                _sample(raw_episode_dir, 1, include_pose=False),
                _sample(raw_episode_dir, 2, include_pose=True),
            ]
            _write_raw_episode(raw_task_dir, "episode_0001", samples)

            from tools import export_raw_data_to_lerobot_v2 as exporter

            with mock.patch.object(exporter, "create_arm_ik", return_value=_FakeArmIk()):
                with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_compute_fk_for_qpos", _fake_compute_fk):
                    with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_video_writer", _fake_video_writer):
                        with mock.patch.object(lerobot_v2_writer.cv2, "VideoCapture", _FinalPathAwareFakeVideoCapture):
                            with mock.patch.object(exporter.cv2, "VideoCapture", _FinalPathAwareFakeVideoCapture):
                                summary = exporter.export_raw_task_dir(
                                    input_task_dir=raw_task_dir,
                                    output_root=root / "lerobot",
                                    task="pick",
                                    fps=30.0,
                                    overwrite=False,
                                )

            pose_path = root / "lerobot" / "extras" / "raw_pose" / "chunk-000" / "episode_000000.jsonl"
            self.assertFalse(pose_path.exists())
            self.assertFalse(summary["episodes"][0]["pose_sidecar"])
            self.assertEqual(summary["episodes"][0]["pose_missing_frame_count"], 1)

    def test_missing_qpos_fails_instead_of_inventing_joint_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            raw_task_dir = root / "raw"
            raw_episode_dir = raw_task_dir / "episode_0001"
            sample = _sample(raw_episode_dir, 0, include_pose=True)
            del sample["states"]["left_arm"]["qpos"]
            _write_raw_episode(raw_task_dir, "episode_0001", [sample])

            from tools import export_raw_data_to_lerobot_v2 as exporter

            with mock.patch.object(exporter, "create_arm_ik", return_value=_FakeArmIk()):
                with self.assertRaisesRegex(KeyError, "states.left_arm.qpos"):
                    exporter.export_raw_task_dir(
                        input_task_dir=raw_task_dir,
                        output_root=root / "lerobot",
                        task="pick",
                        fps=30.0,
                        overwrite=False,
                    )

    def test_non_monotonic_timestamp_fails_before_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            raw_task_dir = root / "raw"
            raw_episode_dir = raw_task_dir / "episode_0001"
            samples = [_sample(raw_episode_dir, idx, include_pose=True) for idx in range(2)]
            samples[1]["timestamps"]["sample_monotonic_ns"] = samples[0]["timestamps"]["sample_monotonic_ns"]
            _write_raw_episode(raw_task_dir, "episode_0001", samples)

            from tools import export_raw_data_to_lerobot_v2 as exporter

            with mock.patch.object(exporter, "create_arm_ik", return_value=_FakeArmIk()):
                with self.assertRaisesRegex(ValueError, "not strictly increasing"):
                    exporter.export_raw_task_dir(
                        input_task_dir=raw_task_dir,
                        output_root=root / "lerobot",
                        task="pick",
                        fps=30.0,
                        overwrite=False,
                    )

    def test_cli_main_writes_export_summary(self):
        _FakeVideoWriter.by_path.clear()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            raw_task_dir = root / "raw"
            raw_episode_dir = raw_task_dir / "episode_0001"
            samples = [_sample(raw_episode_dir, idx, include_pose=True) for idx in range(1)]
            _write_raw_episode(raw_task_dir, "episode_0001", samples)

            from tools import export_raw_data_to_lerobot_v2 as exporter

            output_root = root / "lerobot"
            with mock.patch.object(exporter, "create_arm_ik", return_value=_FakeArmIk()):
                with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_compute_fk_for_qpos", _fake_compute_fk):
                    with mock.patch.object(lerobot_v2_writer.LeRobotV2Writer, "_video_writer", _fake_video_writer):
                        with mock.patch.object(lerobot_v2_writer.cv2, "VideoCapture", _FinalPathAwareFakeVideoCapture):
                            with mock.patch.object(exporter.cv2, "VideoCapture", _FinalPathAwareFakeVideoCapture):
                                stdout = StringIO()
                                with redirect_stdout(stdout):
                                    exit_code = exporter.main(
                                        [
                                            "--input-task-dir",
                                            str(raw_task_dir),
                                            "--output-root",
                                            str(output_root),
                                            "--task",
                                            "pick",
                                            "--fps",
                                            "30",
                                        ]
                                    )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_root / "export_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["episodes_exported"], 1)
            self.assertEqual(summary["frames_total"], 1)


if __name__ == "__main__":
    unittest.main()
