import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeClock:
    def __init__(self, start_ns: int = 10_000_000_000) -> None:
        self.now_ns = int(start_ns)

    def __call__(self) -> int:
        return self.now_ns

    def advance_ms(self, value: float) -> None:
        self.now_ns += int(value * 1_000_000)


class DualFakeClock:
    def __init__(self, control_start_ns: int = 10_000_000_000, trace_start_ns: int = 100_000_000_000) -> None:
        self.control_ns = int(control_start_ns)
        self.trace_ns = int(trace_start_ns)

    def control(self) -> int:
        return self.control_ns

    def trace(self) -> int:
        return self.trace_ns

    def advance_ms(self, value: float) -> None:
        delta_ns = int(value * 1_000_000)
        self.control_ns += delta_ns
        self.trace_ns += delta_ns


class FakeTransport:
    def __init__(self, connected: bool = True) -> None:
        self.connected = connected
        self.sent_messages = []
        self.recv_queue = []
        self.reset_calls = 0

    def send_json(self, payload: dict) -> None:
        self.sent_messages.append(payload)

    def recv_json_nonblocking(self):
        if self.recv_queue:
            return self.recv_queue.pop(0)
        return None

    def is_connected(self) -> bool:
        return self.connected

    def reset(self) -> None:
        self.reset_calls += 1

    def queue_recv(self, payload: dict) -> None:
        self.recv_queue.append(payload)


class ResetCaptureTransport(FakeTransport):
    def __init__(self, connected: bool = True) -> None:
        super().__init__(connected=connected)
        self.reset_calls = []

    def reset(self, *args, **kwargs) -> None:
        self.reset_calls.append((args, kwargs))


class AdvancingSendTransport(FakeTransport):
    def __init__(self, clock: DualFakeClock, send_ms: float) -> None:
        super().__init__()
        self.clock = clock
        self.send_ms = float(send_ms)

    def send_json(self, payload: dict) -> None:
        self.clock.advance_ms(self.send_ms)
        super().send_json(payload)


class FakePoseTransformer:
    def __init__(self) -> None:
        self.observation_calls = []
        self.action_calls = []

    def observation_to_server(self, side: str, pose) -> list[float]:
        matrix = np.asarray(pose, dtype=np.float64)
        self.observation_calls.append((side, matrix.copy()))
        offset = 100.0 if side == "left" else 200.0
        return [
            float(matrix[0, 3] + offset),
            float(matrix[1, 3] + offset),
            float(matrix[2, 3] + offset),
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    def action_to_unitree(self, side: str, pose7) -> np.ndarray:
        pose7_array = np.asarray(pose7, dtype=np.float64)
        self.action_calls.append((side, pose7_array.copy()))
        offset = 10.0 if side == "left" else 20.0
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = pose7_array[0] + offset
        matrix[1, 3] = pose7_array[1] + offset
        matrix[2, 3] = pose7_array[2] + offset
        return matrix


def _pose_matrix(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose


def _image(fill: int) -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = fill
    return image


class OnlineInferenceSessionTest(unittest.TestCase):
    def test_speed_limit_delta_returns_none_for_small_clip(self):
        from teleop.utils.online_inference import online_inference_speed_limit_delta

        delta = online_inference_speed_limit_delta(
            target_q=np.array([0.01] * 14, dtype=float),
            limited_q=np.array([0.0] * 14, dtype=float),
            max_arm_joint_speed=0.1,
            frequency=30.0,
        )

        self.assertIsNone(delta)

    def test_speed_limit_delta_reports_large_clip(self):
        from teleop.utils.online_inference import online_inference_speed_limit_delta

        target = np.zeros(14, dtype=float)
        limited = np.zeros(14, dtype=float)
        target[9] = 0.50
        limited[9] = 0.01

        delta = online_inference_speed_limit_delta(
            target_q=target,
            limited_q=limited,
            max_arm_joint_speed=0.1,
            frequency=30.0,
        )

        self.assertAlmostEqual(delta, 0.49)

    def _make_state(self, host_monotonic_ns: int):
        from teleop.utils.online_inference import RobotStateSample

        return RobotStateSample(
            host_monotonic_ns=host_monotonic_ns,
            left_pose=_pose_matrix(1.0, 2.0, 3.0),
            right_pose=_pose_matrix(-1.0, -2.0, -3.0),
            left_gripper_width=0.012,
            right_gripper_width=0.034,
        )

    def _make_cameras(self, host_monotonic_ns: int):
        from teleop.utils.online_inference import CameraSample

        return [
            CameraSample(name="head", frame=_image(40), host_monotonic_ns=host_monotonic_ns),
            CameraSample(name="left_wrist", frame=_image(80), host_monotonic_ns=host_monotonic_ns),
        ]

    def _make_pi05_cameras(self, host_monotonic_ns: int):
        from teleop.utils.online_inference import CameraSample

        return [
            CameraSample(name="head", frame=_image(40), host_monotonic_ns=host_monotonic_ns),
            CameraSample(name="right_wrist", frame=_image(120), host_monotonic_ns=host_monotonic_ns),
        ]

    def _make_config(self, **overrides):
        from teleop.utils.online_inference import OnlineInferenceConfig

        payload = {
            "arm_side": "left",
            "n_obs_steps": 2,
            "camera_freq": 30.0,
            "action_step_sec": 0.10,
            "interpolation_interval_sec": 0.01,
            "post_action_delay_ms": 75,
            "response_timeout_sec": 2.0,
            "jpeg_quality": 85,
            "enable_motion": False,
            "dry_run": False,
        }
        payload.update(overrides)
        return OnlineInferenceConfig(**payload)

    def test_tick_history_insufficient_returns_hold_without_send(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(),
            transport=transport,
            clock_ns=clock,
        )

        step = session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )

        self.assertEqual(step.status, "collecting_observation")
        self.assertEqual(step.enabled_arms, [])
        self.assertTrue(np.allclose(step.left_pose, _pose_matrix(1.0, 2.0, 3.0)))
        self.assertTrue(np.allclose(step.right_pose, _pose_matrix(-1.0, -2.0, -3.0)))
        self.assertEqual(step.left_gripper_width, 0.012)
        self.assertEqual(step.right_gripper_width, 0.034)
        self.assertEqual(step.metadata["history_len"], 1)
        self.assertEqual(transport.sent_messages, [])

    def test_ready_observation_sends_pika_message_and_waits_nonblocking(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        transformer = FakePoseTransformer()
        session = OnlineInferenceSession(
            config=self._make_config(),
            transport=transport,
            pose_transformer=transformer,
            clock_ns=clock,
        )

        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        clock.advance_ms(40)
        step = session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )

        self.assertEqual(step.status, "waiting_action")
        self.assertEqual(step.enabled_arms, [])
        self.assertEqual(len(transport.sent_messages), 1)
        message = transport.sent_messages[0]
        self.assertEqual(message["type"], "observation")
        self.assertIn("arm_l", message)
        self.assertNotIn("arm_r", message)
        self.assertNotIn("state", message)
        self.assertNotIn("history", message)
        self.assertNotIn("cameras", message)

        arm_l = message["arm_l"]
        self.assertEqual(set(arm_l.keys()), {"images", "poses", "grippers", "init_pose", "arm_current_pose"})
        self.assertEqual(len(arm_l["images"]), 4)
        self.assertEqual(len(arm_l["poses"]), 2)
        self.assertEqual(len(arm_l["grippers"]), 2)
        self.assertEqual(arm_l["poses"][-1], [101.0, 102.0, 103.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(arm_l["init_pose"], [101.0, 102.0, 103.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(arm_l["arm_current_pose"], [101.0, 102.0, 103.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(arm_l["grippers"], [0.012, 0.012])
        for encoded_image in arm_l["images"]:
            self.assertIsInstance(encoded_image, str)
            self.assertGreater(len(encoded_image), 10)
        self.assertEqual(len(transformer.observation_calls), 2)

    def test_observation_history_uses_camera_freq_target_window(self):
        from teleop.utils.online_inference import CameraSample, OnlineInferenceSession, RobotStateSample

        clock = FakeClock()
        transport = FakeTransport()
        transformer = FakePoseTransformer()
        session = OnlineInferenceSession(
            config=self._make_config(n_obs_steps=2, camera_freq=10.0),
            transport=transport,
            pose_transformer=transformer,
            clock_ns=clock,
        )

        for offset_ms, x in ((0, 1.0), (50, 2.0), (100, 3.0)):
            t_ns = clock() + offset_ms * 1_000_000
            state = RobotStateSample(
                host_monotonic_ns=t_ns,
                left_pose=_pose_matrix(x, 2.0, 3.0),
                right_pose=_pose_matrix(-x, -2.0, -3.0),
                left_gripper_width=x / 100.0,
                right_gripper_width=0.034,
            )
            cameras = [
                CameraSample(name="head", frame=_image(int(20 + x)), host_monotonic_ns=t_ns),
                CameraSample(name="left_wrist", frame=_image(int(40 + x)), host_monotonic_ns=t_ns),
            ]
            step = session.tick(state_sample=state, camera_samples=cameras)

        self.assertEqual(step.status, "waiting_action")
        arm_l = transport.sent_messages[0]["arm_l"]
        self.assertEqual(
            arm_l["poses"],
            [
                [101.0, 102.0, 103.0, 0.0, 0.0, 0.0, 1.0],
                [103.0, 102.0, 103.0, 0.0, 0.0, 0.0, 1.0],
            ],
        )
        self.assertEqual(arm_l["grippers"], [0.01, 0.03])

    def test_both_arm_observation_sends_arm_l_and_arm_r(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        transformer = FakePoseTransformer()
        session = OnlineInferenceSession(
            config=self._make_config(arm_side="both"),
            transport=transport,
            pose_transformer=transformer,
            clock_ns=clock,
        )

        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        clock.advance_ms(40)
        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )

        message = transport.sent_messages[0]
        self.assertIn("arm_l", message)
        self.assertIn("arm_r", message)
        self.assertEqual(message["arm_l"]["poses"][-1], [101.0, 102.0, 103.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(message["arm_r"]["poses"][-1], [199.0, 198.0, 197.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(message["arm_r"]["grippers"], [0.034, 0.034])
        self.assertEqual(len(transformer.observation_calls), 4)

    def test_pi05_profile_sends_dual_arm_pose9_observation_with_prompt(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(
                arm_side="both",
                protocol_profile="pi05_dual_arm_20d",
                task_prompt="pick up the cube",
            ),
            transport=transport,
            clock_ns=clock,
        )

        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_pi05_cameras(clock()))
        clock.advance_ms(40)
        step = session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_pi05_cameras(clock()))

        self.assertEqual(step.status, "waiting_action")
        self.assertEqual(len(transport.sent_messages), 1)
        message = transport.sent_messages[0]
        self.assertEqual(message["type"], "observation")
        self.assertEqual(message["prompt"], "pick up the cube")
        self.assertIn("images", message)
        self.assertEqual(
            list(message["images"].keys()),
            ["third_front", "head_fpv", "left_hand", "right_hand"],
        )
        self.assertIsInstance(message["images"]["third_front"], bytes)
        self.assertGreater(len(message["images"]["third_front"]), 0)
        self.assertEqual(message["images"]["third_front"], message["images"]["right_hand"])
        self.assertIsInstance(message["images"]["head_fpv"], bytes)
        self.assertEqual(message["images"]["left_hand"], "")
        self.assertIsInstance(message["images"]["right_hand"], bytes)
        self.assertNotIn("right_wrist", message["images"])
        self.assertEqual(len(message["poses_left"]), 2)
        self.assertEqual(len(message["poses_right"]), 2)
        self.assertEqual(len(message["poses_left"][-1]), 9)
        self.assertEqual(len(message["poses_right"][-1]), 9)
        self.assertEqual(message["grippers_left"], [[0.012], [0.012]])
        self.assertEqual(message["grippers_right"], [[0.034], [0.034]])
        snapshot = session.get_debug_snapshot()
        self.assertEqual(snapshot["protocol_profile"], "pi05_dual_arm_20d")
        self.assertEqual(snapshot["last_observation"]["profile"], "pi05_dual_arm_20d")
        self.assertEqual(snapshot["last_observation"]["prompt"], "pick up the cube")

    def test_pi05_profile_maps_local_camera_names_to_nero_reference_roles(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(
                arm_side="both",
                protocol_profile="pi05_dual_arm_20d",
                task_prompt="pick up the cube",
            ),
            transport=transport,
            clock_ns=clock,
        )

        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_pi05_cameras(clock()))
        clock.advance_ms(40)
        step = session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_pi05_cameras(clock()))

        self.assertEqual(step.status, "waiting_action")
        message = transport.sent_messages[0]
        self.assertEqual(
            list(message["images"].keys()),
            ["third_front", "head_fpv", "left_hand", "right_hand"],
        )
        self.assertIsInstance(message["images"]["third_front"], bytes)
        self.assertGreater(len(message["images"]["third_front"]), 0)
        self.assertEqual(message["images"]["third_front"], message["images"]["right_hand"])
        self.assertIsInstance(message["images"]["head_fpv"], bytes)
        self.assertGreater(len(message["images"]["head_fpv"]), 0)
        self.assertEqual(message["images"]["left_hand"], "")
        self.assertIsInstance(message["images"]["right_hand"], bytes)
        self.assertGreater(len(message["images"]["right_hand"]), 0)
        self.assertNotIn("right_wrist", message["images"])
        self.assertEqual(
            session.get_debug_snapshot()["last_observation"]["image_roles"],
            ["head_fpv", "right_hand", "third_front"],
        )

    def test_pi05_profile_fails_closed_when_required_head_or_right_hand_image_missing(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(
                arm_side="both",
                protocol_profile="pi05_dual_arm_20d",
            ),
            transport=transport,
            clock_ns=clock,
        )

        cameras_without_right = [self._make_pi05_cameras(clock())[0]]
        session.tick(state_sample=self._make_state(clock()), camera_samples=cameras_without_right)
        clock.advance_ms(40)
        cameras_without_right = [self._make_pi05_cameras(clock())[0]]
        step = session.tick(state_sample=self._make_state(clock()), camera_samples=cameras_without_right)

        self.assertEqual(step.status, "failed")
        self.assertIn("pi05 observation requires non-empty image role: right_hand", session.error)
        self.assertEqual(transport.sent_messages, [])

    def test_reconnect_rearms_reset(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(),
            transport=transport,
            clock_ns=clock,
        )

        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        clock.advance_ms(40)
        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        self.assertEqual(transport.reset_calls, 1)

        transport.connected = False
        step = session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        self.assertEqual(step.status, "failed")
        transport.connected = True
        transport.recv_queue = []
        transport.sent_messages = []
        session.status = "collecting_observation"
        session.error = None
        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        clock.advance_ms(40)
        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        self.assertEqual(transport.reset_calls, 2)

    def test_transport_reset_is_called_without_reason(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = ResetCaptureTransport()
        session = OnlineInferenceSession(
            config=self._make_config(),
            transport=transport,
            clock_ns=clock,
        )

        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        clock.advance_ms(40)
        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))

        self.assertEqual(transport.reset_calls, [((), {})])

    def test_real_motion_requires_pose_transformer(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        with self.assertRaises(ValueError):
            OnlineInferenceSession(
                config=self._make_config(enable_motion=True, dry_run=False),
                transport=FakeTransport(),
                pose_transformer=None,
                clock_ns=FakeClock(),
            )

    def test_wait_action_timeout_fails_closed(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(response_timeout_sec=0.05),
            transport=transport,
            clock_ns=clock,
        )

        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        clock.advance_ms(40)
        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        clock.advance_ms(60)
        step = session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )

        self.assertEqual(step.status, "failed")
        self.assertEqual(step.enabled_arms, [])
        self.assertIn("timeout", session.error.lower())
        self.assertEqual(session.status, "failed")

    def test_wait_action_ignores_reset_ack_control_response(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(response_timeout_sec=0.50),
            transport=transport,
            clock_ns=clock,
        )

        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        clock.advance_ms(40)
        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        transport.queue_recv({"type": "reset_ack", "timestamp": 123.0})

        step = session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )

        self.assertEqual(step.status, "waiting_action")
        self.assertEqual(step.enabled_arms, [])
        self.assertIsNone(session.error)
        self.assertEqual(session.status, "waiting_action")

    def test_debug_snapshot_tracks_latest_right_observation_and_action(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(arm_side="right"),
            transport=transport,
            clock_ns=clock,
        )

        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        clock.advance_ms(40)
        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        transport.queue_recv(
            {
                "type": "action",
                "action_r": [[-0.5, -1.0, -2.5, 0.0, 0.0, 0.0, 1.0, 0.02]],
            }
        )

        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        snapshot = session.get_debug_snapshot()

        self.assertEqual(snapshot["arm_side"], "right")
        self.assertEqual(snapshot["last_observation"]["right"]["arm_current_pose"][:3], [-1.0, -2.0, -3.0])
        self.assertEqual(snapshot["last_action"]["right"]["first_step"][:3], [-0.5, -1.0, -2.5])
        self.assertEqual(snapshot["last_action"]["right"]["chunk_size"], 1)
        self.assertAlmostEqual(snapshot["last_action"]["right"]["delta_xyz_m"], 1.224744871, places=6)

    def test_post_action_delay_timeout_fails_closed(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(post_action_delay_ms=75),
            transport=transport,
            clock_ns=clock,
        )

        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        clock.advance_ms(40)
        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        transport.queue_recv({"type": "action", "action": [[0.5, 0.25, -0.75, 0.0, 0.0, 0.0, 1.0, 0.066]]})
        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))

        clock.advance_ms(5000)
        step = session.tick(
            state_sample=self._make_state(clock() - 1000),
            camera_samples=self._make_cameras(clock() - 1000),
        )

        self.assertEqual(step.status, "failed")
        self.assertIn("post_action_delay", session.error.lower())

    def test_valid_left_action_returns_enabled_left_only_when_motion_enabled(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        transformer = FakePoseTransformer()
        session = OnlineInferenceSession(
            config=self._make_config(enable_motion=True),
            transport=transport,
            pose_transformer=transformer,
            clock_ns=clock,
        )

        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        clock.advance_ms(40)
        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        transport.queue_recv(
            {
                "type": "action",
                "action": [[0.5, 0.25, -0.75, 0.0, 0.0, 0.0, 1.0, 0.066]],
            }
        )

        step = session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )

        self.assertEqual(step.status, "executing_chunk")
        self.assertEqual(step.enabled_arms, ["left"])
        self.assertTrue(np.allclose(step.left_pose, _pose_matrix(10.5, 10.25, 9.25)))
        self.assertTrue(np.allclose(step.right_pose, _pose_matrix(-1.0, -2.0, -3.0)))
        self.assertAlmostEqual(step.left_gripper_width, 0.066)
        self.assertAlmostEqual(step.right_gripper_width, 0.034)
        self.assertEqual(len(transformer.action_calls), 1)
        self.assertEqual(transformer.action_calls[0][0], "left")

    def test_pi05_profile_loads_action_sequence20_as_dual_pose7_chunks(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        transformer = FakePoseTransformer()
        session = OnlineInferenceSession(
            config=self._make_config(
                arm_side="both",
                protocol_profile="pi05_dual_arm_20d",
                enable_motion=True,
            ),
            transport=transport,
            pose_transformer=transformer,
            clock_ns=clock,
        )

        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_pi05_cameras(clock()))
        clock.advance_ms(40)
        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_pi05_cameras(clock()))
        transport.queue_recv(
            {
                "type": "action_sequence",
                "actions": [
                    [
                        1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.01,
                        4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.02,
                    ]
                ],
            }
        )

        step = session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_pi05_cameras(clock()))

        self.assertEqual(step.status, "executing_chunk")
        self.assertEqual(step.enabled_arms, ["left", "right"])
        self.assertTrue(np.allclose(step.left_pose, _pose_matrix(11.0, 12.0, 13.0)))
        self.assertTrue(np.allclose(step.right_pose, _pose_matrix(24.0, 25.0, 26.0)))
        self.assertAlmostEqual(step.left_gripper_width, 0.01)
        self.assertAlmostEqual(step.right_gripper_width, 0.02)
        self.assertEqual(session.get_debug_snapshot()["last_action"]["profile"], "pi05_dual_arm_20d")

    def test_dry_run_advances_action_but_disables_enabled_arms(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(enable_motion=True, dry_run=True),
            transport=transport,
            clock_ns=clock,
        )

        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        clock.advance_ms(40)
        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        transport.queue_recv(
            {
                "type": "action",
                "action": [[0.5, 0.25, -0.75, 0.0, 0.0, 0.0, 1.0, 0.066]],
            }
        )

        executing_step = session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        clock.advance_ms(110)
        hold_step = session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )

        self.assertEqual(executing_step.status, "executing_chunk")
        self.assertEqual(executing_step.enabled_arms, [])
        self.assertEqual(session.status, "post_action_delay")
        self.assertEqual(hold_step.status, "post_action_delay")
        self.assertEqual(hold_step.enabled_arms, [])

    def test_post_action_delay_waits_for_new_state_and_camera_coverage_before_next_send(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(post_action_delay_ms=75),
            transport=transport,
            clock_ns=clock,
        )

        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        clock.advance_ms(40)
        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        transport.queue_recv(
            {
                "type": "action",
                "action": [[0.5, 0.25, -0.75, 0.0, 0.0, 0.0, 1.0, 0.066]],
            }
        )
        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )

        clock.advance_ms(110)
        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )

        self.assertEqual(session.status, "post_action_delay")
        self.assertEqual(len(transport.sent_messages), 1)

        clock.advance_ms(50)
        early_step = session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        self.assertEqual(early_step.status, "post_action_delay")
        self.assertEqual(len(transport.sent_messages), 1)

        clock.advance_ms(30)
        stale_step = session.tick(
            state_sample=self._make_state(clock() - 40_000_000),
            camera_samples=self._make_cameras(clock() - 40_000_000),
        )
        self.assertEqual(stale_step.status, "post_action_delay")
        self.assertEqual(len(transport.sent_messages), 1)

        fresh_step = session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        self.assertEqual(fresh_step.status, "waiting_action")
        self.assertEqual(len(transport.sent_messages), 2)

    def test_action_chunk_cadence_respects_action_step_sec(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(action_step_sec=0.10, enable_motion=True),
            transport=transport,
            pose_transformer=FakePoseTransformer(),
            clock_ns=clock,
        )

        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        clock.advance_ms(40)
        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        transport.queue_recv(
            {
                "type": "action",
                "action": [
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.010],
                    [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.020],
                ],
            }
        )

        first = session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        clock.advance_ms(50)
        still_first = session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        clock.advance_ms(60)
        second = session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))

        self.assertEqual(first.metadata["chunk_index"], 0)
        self.assertEqual(still_first.metadata["chunk_index"], 0)
        self.assertEqual(second.metadata["chunk_index"], 1)
        self.assertTrue(np.allclose(first.left_pose, _pose_matrix(10.5, 10.0, 10.0)))
        self.assertTrue(np.allclose(second.left_pose, _pose_matrix(10.7, 10.0, 10.0)))

    def test_per_tick_chunk_step_mode_consumes_one_step_each_tick(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(chunk_step_mode="per_tick", action_step_sec=10.0, enable_motion=True),
            transport=transport,
            pose_transformer=FakePoseTransformer(),
            clock_ns=clock,
        )

        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        clock.advance_ms(40)
        session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        transport.queue_recv(
            {
                "type": "action",
                "action": [
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.010],
                    [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.020],
                    [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.030],
                ],
            }
        )

        first = session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        second = session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))
        third = session.tick(state_sample=self._make_state(clock()), camera_samples=self._make_cameras(clock()))

        self.assertEqual(first.metadata["chunk_index"], 0)
        self.assertEqual(second.metadata["chunk_index"], 1)
        self.assertEqual(third.metadata["chunk_index"], 2)
        self.assertTrue(np.allclose(first.left_pose, _pose_matrix(10.5, 10.0, 10.0)))
        self.assertTrue(np.allclose(second.left_pose, _pose_matrix(10.7, 10.0, 10.0)))
        self.assertTrue(np.allclose(third.left_pose, _pose_matrix(10.9, 10.0, 10.0)))

    def test_action_step_metadata_tracks_control_and_perf_timestamps_separately(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = DualFakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(chunk_step_mode="per_tick", action_step_sec=10.0, enable_motion=True),
            transport=transport,
            pose_transformer=FakePoseTransformer(),
            clock_ns=clock.control,
            trace_clock_ns=clock.trace,
        )

        session.tick(state_sample=self._make_state(clock.control()), camera_samples=self._make_cameras(clock.control()))
        clock.advance_ms(40)
        session.tick(state_sample=self._make_state(clock.control()), camera_samples=self._make_cameras(clock.control()))
        obs_send_ns = clock.control()
        obs_send_perf_ns = clock.trace()
        clock.advance_ms(17)
        action_recv_ns = clock.control()
        action_recv_perf_ns = clock.trace()
        transport.queue_recv(
            {
                "type": "action",
                "action": [
                    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.010],
                    [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.020],
                ],
            }
        )

        first = session.tick(state_sample=self._make_state(clock.control()), camera_samples=self._make_cameras(clock.control()))
        clock.advance_ms(5)
        second = session.tick(state_sample=self._make_state(clock.control()), camera_samples=self._make_cameras(clock.control()))

        self.assertEqual(first.metadata["online_observation_seq"], 1)
        self.assertEqual(first.metadata["online_chunk_seq"], 1)
        self.assertEqual(first.metadata["online_chunk_size"], 2)
        self.assertEqual(first.metadata["online_chunk_index"], 0)
        self.assertEqual(first.metadata["online_obs_send_ns"], obs_send_ns)
        self.assertEqual(first.metadata["online_action_recv_ns"], action_recv_ns)
        self.assertEqual(first.metadata["online_step_output_ns"], action_recv_ns)
        self.assertEqual(first.metadata["online_obs_send_perf_ns"], obs_send_perf_ns)
        self.assertEqual(first.metadata["online_action_recv_perf_ns"], action_recv_perf_ns)
        self.assertEqual(first.metadata["online_step_output_perf_ns"], action_recv_perf_ns)
        self.assertNotEqual(first.metadata["online_obs_send_perf_ns"], first.metadata["online_obs_send_ns"])
        self.assertAlmostEqual(first.metadata["online_obs_send_to_action_recv_ms"], 17.0)
        self.assertEqual(second.metadata["online_observation_seq"], 1)
        self.assertEqual(second.metadata["online_chunk_seq"], 1)
        self.assertEqual(second.metadata["online_chunk_index"], 1)
        self.assertEqual(second.metadata["online_step_output_ns"], action_recv_ns + 5_000_000)
        self.assertEqual(second.metadata["online_step_output_perf_ns"], action_recv_perf_ns + 5_000_000)

    def test_observation_send_perf_timestamp_is_before_transport_send(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = DualFakeClock()
        transport = AdvancingSendTransport(clock=clock, send_ms=3.0)
        session = OnlineInferenceSession(
            config=self._make_config(chunk_step_mode="per_tick", action_step_sec=10.0, enable_motion=True),
            transport=transport,
            pose_transformer=FakePoseTransformer(),
            clock_ns=clock.control,
            trace_clock_ns=clock.trace,
        )

        session.tick(state_sample=self._make_state(clock.control()), camera_samples=self._make_cameras(clock.control()))
        clock.advance_ms(40)
        obs_send_start_ns = clock.control()
        obs_send_start_perf_ns = clock.trace()
        session.tick(state_sample=self._make_state(clock.control()), camera_samples=self._make_cameras(clock.control()))

        clock.advance_ms(17)
        action_recv_ns = clock.control()
        action_recv_perf_ns = clock.trace()
        transport.queue_recv(
            {
                "type": "action",
                "action": [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.010]],
            }
        )

        step = session.tick(state_sample=self._make_state(clock.control()), camera_samples=self._make_cameras(clock.control()))

        self.assertEqual(step.metadata["online_obs_send_ns"], obs_send_start_ns)
        self.assertEqual(step.metadata["online_obs_send_perf_ns"], obs_send_start_perf_ns)
        self.assertEqual(step.metadata["online_action_recv_ns"], action_recv_ns)
        self.assertEqual(step.metadata["online_action_recv_perf_ns"], action_recv_perf_ns)
        self.assertAlmostEqual(step.metadata["online_obs_send_to_action_recv_ms"], 20.0)

    def test_control_feedback_fatal_aborts_chunk(self):
        from teleop.utils.online_inference import OnlineInferenceSession

        clock = FakeClock()
        transport = FakeTransport()
        session = OnlineInferenceSession(
            config=self._make_config(enable_motion=True),
            transport=transport,
            pose_transformer=FakePoseTransformer(),
            clock_ns=clock,
        )

        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        clock.advance_ms(40)
        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )
        transport.queue_recv(
            {
                "action": [
                    [0.5, 0.25, -0.75, 0.0, 0.0, 0.0, 1.0, 0.066],
                    [0.6, 0.35, -0.55, 0.0, 0.0, 0.0, 1.0, 0.044],
                ],
            }
        )
        session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )

        session.report_control_feedback({"fatal": True, "reason": "ik failed"})
        step = session.tick(
            state_sample=self._make_state(clock()),
            camera_samples=self._make_cameras(clock()),
        )

        self.assertEqual(step.status, "failed")
        self.assertEqual(step.enabled_arms, [])
        self.assertEqual(session.status, "failed")
        self.assertIn("ik failed", session.error)


if __name__ == "__main__":
    unittest.main()
