import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Pi05ProtocolTest(unittest.TestCase):
    def test_build_observation_payload_packs_reference_image_roles_prompt_and_pose9(self):
        from teleop.inference.pi05_protocol import build_pi05_observation_payload
        from teleop.inference.pose_transform import matrix_to_pose9_rot6d

        pose = np.eye(4, dtype=float)
        pose[0, 3] = 0.25
        pose[1, 3] = -0.5
        pose[2, 3] = 1.2
        pose9 = matrix_to_pose9_rot6d(pose)

        payload = build_pi05_observation_payload(
            images={"head_fpv": b"head_jpeg", "right_hand": b"right_jpeg"},
            poses_left=[pose9],
            grippers_left=[[0.01]],
            poses_right=[pose9],
            grippers_right=[[0.02]],
            prompt="pick up the cube",
        )

        self.assertEqual(payload["type"], "observation")
        self.assertEqual(payload["prompt"], "pick up the cube")
        self.assertEqual(
            list(payload["images"].keys()),
            ["third_front", "head_fpv", "left_hand", "right_hand"],
        )
        self.assertEqual(payload["images"]["third_front"], "")
        self.assertEqual(payload["images"]["head_fpv"], b"head_jpeg")
        self.assertEqual(payload["images"]["left_hand"], "")
        self.assertEqual(payload["images"]["right_hand"], b"right_jpeg")
        self.assertNotIn("right_wrist", payload["images"])
        self.assertEqual(payload["poses_left"], [pose9])
        self.assertEqual(payload["poses_right"], [pose9])
        self.assertEqual(payload["grippers_left"], [[0.01]])
        self.assertEqual(payload["grippers_right"], [[0.02]])

    def test_build_observation_payload_rejects_unknown_image_roles(self):
        from teleop.inference.pi05_protocol import build_pi05_observation_payload

        with self.assertRaises(ValueError):
            build_pi05_observation_payload(
                images={"right_wrist": b"not_a_reference_role"},
                poses_left=[[0.0] * 9],
                grippers_left=[[0.01]],
                poses_right=[[0.0] * 9],
                grippers_right=[[0.02]],
            )

    def test_normalize_action_response_combines_dual_arm_action_payload(self):
        from teleop.inference.pi05_protocol import normalize_pi05_action_response

        payload = normalize_pi05_action_response(
            {
                "type": "action",
                "action_l": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
                "action_r": [[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]],
            }
        )

        self.assertEqual(payload["type"], "action_sequence")
        self.assertEqual(payload["actions"], [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])

    def test_validate_action_sequence_requires_expected_dim(self):
        from teleop.inference.pi05_protocol import validate_pi05_action_sequence

        actions = validate_pi05_action_sequence(
            {"type": "action_sequence", "actions": [[1.0] * 20]},
            expected_dim=20,
        )

        self.assertEqual(actions.shape, (1, 20))
        with self.assertRaises(ValueError):
            validate_pi05_action_sequence({"type": "action_sequence", "actions": [[1.0] * 10]}, expected_dim=20)

    def test_validate_action_sequence_accepts_action_sequence_alias(self):
        from teleop.inference.pi05_protocol import validate_pi05_action_sequence

        actions = validate_pi05_action_sequence(
            {"type": "action_sequence", "action_sequence": [[1.0] * 20]},
            expected_dim=20,
        )

        self.assertEqual(actions.shape, (1, 20))

    def test_action_sequence_to_pose7_chunks_splits_dual_arm_rot6d(self):
        from teleop.inference.pi05_protocol import pi05_action_sequence_to_pose7_chunks

        actions = np.asarray(
            [
                [
                    1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.01,
                    4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.02,
                ]
            ],
            dtype=np.float32,
        )

        left, right = pi05_action_sequence_to_pose7_chunks(actions, arm_side="both")

        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 1)
        self.assertTrue(np.allclose(left[0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.01]))
        self.assertTrue(np.allclose(right[0], [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0, 0.02]))

    def test_remap_action_sequence_applies_observation_delta(self):
        from teleop.inference.pi05_protocol import remap_pi05_action_sequence_observation_delta

        obs_anchor = np.asarray([
            1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.01, 0.0,
            4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.02, 0.0,
        ], dtype=np.float32)
        robot_anchor = np.asarray([
            10.0, 20.0, 30.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.05, 0.0,
            40.0, 50.0, 60.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.06, 0.0,
        ], dtype=np.float32)
        actions = np.asarray([
            [
                1.5, 2.5, 3.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.03, 0.0,
                4.5, 5.5, 6.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.04, 0.0,
            ]
        ], dtype=np.float32)

        remapped = remap_pi05_action_sequence_observation_delta(
            actions,
            observation_anchor20=obs_anchor,
            robot_anchor20=robot_anchor,
        )

        self.assertEqual(remapped.shape, (1, 20))
        self.assertAlmostEqual(remapped[0, 0], 10.5)
        self.assertAlmostEqual(remapped[0, 1], 20.5)
        self.assertAlmostEqual(remapped[0, 2], 30.5)
        self.assertAlmostEqual(remapped[0, 10], 40.5)
        self.assertAlmostEqual(remapped[0, 11], 50.5)
        self.assertAlmostEqual(remapped[0, 12], 60.5)


if __name__ == "__main__":
    unittest.main()
