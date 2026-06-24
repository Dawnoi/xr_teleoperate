import json
import math
import sys
import unittest
from unittest.mock import patch


class UmiRolloutToolTest(unittest.TestCase):
    def test_request_action_sends_reset_before_observation_even_without_reset_flag(self):
        from tools.probe_umi_online_inference_dataset import request_action

        class FakeSocket:
            def __init__(self) -> None:
                self.sent_messages: list[bytes] = []
                self.recv_queue = [
                    b'{"type":"reset_ack","timestamp":1}\n',
                    b'{"type":"action","action":[[0,0,0,0,0,0,1,0.02]]}\n',
                ]
                self.timeout = None

            def settimeout(self, value) -> None:
                self.timeout = value

            def sendall(self, data: bytes) -> None:
                self.sent_messages.append(data)

            def recv(self, _buffer_size: int) -> bytes:
                if self.recv_queue:
                    return self.recv_queue.pop(0)
                return b""

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_socket = FakeSocket()

        with patch("tools.probe_umi_online_inference_dataset.socket.create_connection", return_value=fake_socket):
            response = request_action(
                host="127.0.0.1",
                port=8007,
                timeout_sec=1.0,
                observation={"type": "observation", "arm_l": {"images": [], "poses": [], "grippers": []}},
            )

        sent_payloads = [json.loads(message.decode("utf-8")) for message in fake_socket.sent_messages]
        self.assertEqual(sent_payloads[0], {"type": "reset"})
        self.assertEqual([payload["type"] for payload in sent_payloads], ["reset", "observation"])
        self.assertEqual(response["type"], "action")

    def test_probe_parse_args_rejects_no_reset_flag(self):
        from tools import probe_umi_online_inference_dataset as probe_module

        with patch.object(sys, "argv", ["probe_umi_online_inference_dataset.py", "--no-reset"]):
            with self.assertRaises(SystemExit) as exc_info:
                probe_module.parse_args()

        self.assertEqual(exc_info.exception.code, 2)

    def test_rollout_parse_args_rejects_no_reset_flag(self):
        from tools import rollout_umi_online_inference_dataset as rollout_module

        with patch.object(sys, "argv", ["rollout_umi_online_inference_dataset.py", "--no-reset"]):
            with self.assertRaises(SystemExit) as exc_info:
                rollout_module.parse_args()

        self.assertEqual(exc_info.exception.code, 2)

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
