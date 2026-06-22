import math
import unittest


class UmiRolloutToolTest(unittest.TestCase):
    def test_rollout_frame_indices_respect_obs_and_action_bounds(self):
        from tools.rollout_umi_online_inference_dataset import rollout_frame_indices

        indices = rollout_frame_indices(
            episode_len=10,
            n_obs_steps=2,
            chunk_size=3,
            start_frame=None,
            end_frame=None,
            stride=None,
        )

        self.assertEqual(indices, [1, 4, 7])

    def test_rollout_frame_indices_reject_empty_range(self):
        from tools.rollout_umi_online_inference_dataset import rollout_frame_indices

        with self.assertRaisesRegex(ValueError, "no rollout frames selected"):
            rollout_frame_indices(
                episode_len=3,
                n_obs_steps=2,
                chunk_size=3,
                start_frame=None,
                end_frame=None,
                stride=None,
            )

    def test_summarize_rollout_samples_reports_expanded_xyz_and_gripper_errors(self):
        from tools.rollout_umi_online_inference_dataset import summarize_rollout_samples

        samples = [
            {
                "dataset_poses": [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                ],
                "dataset_grippers": [5.4, 3.0],
                "expanded_actions": [
                    [0.003, 0.004, 0.0, 0.0, 0.0, 0.0, 1.0, 0.1],
                    [1.0, 2.0, 3.012, 0.0, 0.0, 0.0, 1.0, 0.2],
                ],
            },
        ]

        summary = summarize_rollout_samples(samples)

        self.assertEqual(summary["num_samples"], 1)
        self.assertEqual(summary["num_expanded_points"], 2)
        self.assertTrue(math.isclose(summary["mean_xyz_error_m"], 0.0085))
        self.assertTrue(math.isclose(summary["max_xyz_error_m"], 0.012))
        self.assertTrue(math.isclose(summary["mean_abs_gripper_error"], 4.05))


if __name__ == "__main__":
    unittest.main()
