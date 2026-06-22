import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import export_lerobot_to_umi_dp as exporter


class ExportLeRobotToUmiDpTest(unittest.TestCase):
    def test_wall_time_filename_uses_milliseconds_and_frame_index(self):
        self.assertEqual(
            exporter.format_wall_time_ns(1_700_000_000_123_456_789, timezone_name="utc"),
            "20231114_221320_123",
        )
        self.assertEqual(
            exporter.png_name(frame_index=42, wall_time_ns=1_700_000_000_123_456_789, timezone_name="utc"),
            "000042_20231114_221320_123.png",
        )

    def test_pose6_to_xyz_quat_xyzw_converts_rpy(self):
        pose = exporter.pose6_to_xyz_quat_xyzw([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        self.assertEqual(pose["xyz"], [0.1, 0.2, 0.3])
        self.assertEqual(pose["quat_xyzw"], [0.0, 0.0, 0.0, 1.0])

    def test_export_episode_keeps_row_frame_alignment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            lerobot_root = pathlib.Path(tmp_dir) / "lerobot"
            output_root = pathlib.Path(tmp_dir) / "umi"
            episode_index = 0
            episode = exporter.EpisodeData(
                episode_index=episode_index,
                timestamps=[0.0, 0.033],
                frame_indices=[0, 1],
                left_tcp=[
                    [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                    [1.1, 1.2, 1.3, 0.0, 0.0, 0.0],
                ],
                right_tcp=[
                    [0.4, 0.5, 0.6, 0.0, 0.0, 0.0],
                    [1.4, 1.5, 1.6, 0.0, 0.0, 0.0],
                ],
                left_gripper=[7.0, 17.0],
                right_gripper=[15.0, 25.0],
            )
            alignment_rows = [
                {
                    "frame_index": 0,
                    "sample_wall_time_ns": 1_700_000_000_123_000_000,
                    "cameras": {
                        "cam0": {"host_recv_wall_time_ns": 1_700_000_000_123_100_000},
                        "cam1": {"host_recv_wall_time_ns": 1_700_000_000_123_200_000},
                        "cam2": {"host_recv_wall_time_ns": 1_700_000_000_123_300_000},
                    },
                },
                {
                    "frame_index": 1,
                    "sample_wall_time_ns": 1_700_000_000_156_000_000,
                    "cameras": {
                        "cam0": {"host_recv_wall_time_ns": 1_700_000_000_156_100_000},
                        "cam1": {"host_recv_wall_time_ns": 1_700_000_000_156_200_000},
                        "cam2": {"host_recv_wall_time_ns": 1_700_000_000_156_300_000},
                    },
                },
            ]
            frames_by_slot = {
                "cam0": ["cam0-frame0", "cam0-frame1"],
                "cam1": ["cam1-frame0", "cam1-frame1"],
                "cam2": ["cam2-frame0", "cam2-frame1"],
            }
            written_images = []

            def fake_read_episode(root, ep_index, tcp_source, gripper_source):
                self.assertEqual(pathlib.Path(root), lerobot_root)
                self.assertEqual(ep_index, episode_index)
                self.assertEqual(tcp_source, "cmd")
                self.assertEqual(gripper_source, "action")
                return episode

            def fake_read_alignment(root, ep_index, expected_len):
                self.assertEqual(pathlib.Path(root), lerobot_root)
                self.assertEqual(ep_index, episode_index)
                self.assertEqual(expected_len, 2)
                return alignment_rows

            def fake_decode_video_frames(path, expected_frames):
                slot = pathlib.Path(path).parent.name.split(".")[-1]
                self.assertEqual(expected_frames, 2)
                return list(frames_by_slot[slot])

            def fake_write_png(path, frame):
                written_images.append((path.relative_to(output_root).as_posix(), frame))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(frame), encoding="utf-8")

            with mock.patch.object(exporter, "read_episode_data", fake_read_episode):
                with mock.patch.object(exporter, "read_alignment_rows", fake_read_alignment):
                    with mock.patch.object(exporter, "decode_video_frames", fake_decode_video_frames):
                        with mock.patch.object(exporter, "write_png", fake_write_png):
                            summary = exporter.export_episode(
                                lerobot_root=lerobot_root,
                                output_root=output_root,
                                episode_index=episode_index,
                                tcp_source="cmd",
                                gripper_source="action",
                                timestamp_source="sample",
                                timezone_name="utc",
                                overwrite=False,
                            )

            self.assertEqual(summary["frames"], 2)
            self.assertEqual(
                written_images,
                [
                    ("episode_000000/camera/cam0/000000_20231114_221320_123.png", "cam0-frame0"),
                    ("episode_000000/camera/cam0/000001_20231114_221320_156.png", "cam0-frame1"),
                    ("episode_000000/camera/cam1/000000_20231114_221320_123.png", "cam1-frame0"),
                    ("episode_000000/camera/cam1/000001_20231114_221320_156.png", "cam1-frame1"),
                    ("episode_000000/camera/cam2/000000_20231114_221320_123.png", "cam2-frame0"),
                    ("episode_000000/camera/cam2/000001_20231114_221320_156.png", "cam2-frame1"),
                ],
            )

            left_tcp = json.loads((output_root / "episode_000000" / "tcp" / "left.json").read_text())
            left_gripper = json.loads((output_root / "episode_000000" / "gripper" / "left.json").read_text())
            self.assertEqual(left_tcp["items"][1]["frame_index"], 1)
            self.assertEqual(left_tcp["items"][1]["timestamp_name"], "20231114_221320_156")
            self.assertEqual(left_tcp["items"][1]["xyz"], [1.1, 1.2, 1.3])
            self.assertEqual(left_gripper["items"][1]["position"], 17.0)


if __name__ == "__main__":
    unittest.main()
