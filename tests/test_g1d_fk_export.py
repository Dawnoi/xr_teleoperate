import pathlib
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.export.g1d_fk import G1DArmFkProvider
from data_pipeline.recording import lerobot_v2_writer


class G1DArmFkProviderTest(unittest.TestCase):
    def test_g1d_fk_returns_finite_xyz_rpy_for_all_arm_frames(self):
        provider = G1DArmFkProvider(REPO_ROOT / "assets/g1_d/g1_d.urdf")

        poses = provider.compute(np.zeros(14, dtype=float))

        expected_keys = {
            f"{side}.joint{joint_index}"
            for side in ("left", "right")
            for joint_index in range(1, 8)
        }
        expected_keys.update({"left.gripper_flange", "right.gripper_flange"})
        self.assertEqual(set(poses), expected_keys)
        for key, value in poses.items():
            vector = np.asarray(value, dtype=float)
            self.assertEqual(vector.shape, (6,), key)
            self.assertTrue(np.all(np.isfinite(vector)), key)

    def test_g1d_fk_rejects_urdf_without_required_arm_joint(self):
        source_urdf = (REPO_ROOT / "assets/g1_d/g1_d.urdf").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp_dir:
            malformed_urdf = pathlib.Path(tmp_dir) / "missing_left_elbow_joint.urdf"
            malformed_urdf.write_text(source_urdf.replace("left_elbow_joint", "renamed_joint", 1), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "left_elbow_joint"):
                G1DArmFkProvider(malformed_urdf)

    def test_writer_without_fk_provider_has_no_fk_feature_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = lerobot_v2_writer.LeRobotV2Writer(
                task_dir=tmp_dir,
                task_goal="pick",
                rerun_log=False,
            )
            try:
                self.assertFalse(any(key.startswith("observation.fk.") for key in writer._feature_spec))
            finally:
                writer.close()

    def test_writer_with_g1d_provider_has_xyz_rpy_fk_feature_columns(self):
        provider = G1DArmFkProvider(REPO_ROOT / "assets/g1_d/g1_d.urdf")
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = lerobot_v2_writer.LeRobotV2Writer(
                task_dir=tmp_dir,
                fk_provider=provider,
                task_goal="pick",
                rerun_log=False,
            )
            try:
                self.assertEqual(writer._feature_spec["observation.fk.fb.left.joint1"]["length"], 6)
            finally:
                writer.close()


if __name__ == "__main__":
    unittest.main()
