import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.debug.inference_pose_debug import wrist_pose_to_debug_sample


class InferencePoseDebugTest(unittest.TestCase):
    def test_wrist_pose_debug_sample_uses_xyz_and_column_major_rot6d(self):
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = [0.1, -0.2, 0.3]
        pose[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )

        sample = wrist_pose_to_debug_sample(pose)

        self.assertEqual(sample["xyz"], [0.1, -0.2, 0.3])
        self.assertEqual(sample["rot6d"], [0.0, 1.0, 0.0, -1.0, 0.0, 0.0])

    def test_wrist_pose_debug_sample_rejects_nonfinite_matrix(self):
        pose = np.eye(4, dtype=float)
        pose[0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            wrist_pose_to_debug_sample(pose)


if __name__ == "__main__":
    unittest.main()
