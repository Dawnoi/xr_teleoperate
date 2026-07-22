import pathlib
import sys
import time
from threading import Event, Lock, Timer
import unittest
from types import SimpleNamespace

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.input.base import MotionIntent, TeleopInputSample, _build_offline_tele_data
from teleop.control_flow.end_effector_command import read_dual_gripper_snapshot
from teleop.control_flow import operator_state
from teleop.runtime.provider_switch import ActiveProviderKind, TeleopProviderRuntime
from teleop.ui.command_bus import UiCommandBus, UiCommandName
from teleop.ui.integration import (
    build_runtime_camera_status,
    build_runtime_recording_status,
    build_runtime_web_payload,
    dispatch_ui_commands,
)
from teleop.ui.payload import build_camera_status, build_web_payload
from teleop.ui.state_store import UiStateStore


class UiCommandBusTest(unittest.TestCase):
    def test_submit_rejects_unknown_command_names(self):
        bus = UiCommandBus()

        with self.assertRaises(ValueError):
            bus.submit("set_joint")

    def test_drain_returns_commands_in_submit_order_and_clears_queue(self):
        bus = UiCommandBus()

        first = bus.submit(UiCommandName.START)
        second = bus.submit(UiCommandName.RECORD_TOGGLE)

        drained = bus.drain()
        self.assertEqual([command.name for command in drained], [UiCommandName.START, UiCommandName.RECORD_TOGGLE])
        self.assertLess(first.created_monotonic_ns, second.created_monotonic_ns)
        self.assertEqual(bus.drain(), [])

    def test_online_inference_stop_request_stays_visible_until_drained(self):
        bus = UiCommandBus()

        bus.submit(UiCommandName.STOP_ONLINE_INFERENCE)

        self.assertTrue(bus.online_inference_stop_requested())
        self.assertEqual(bus.drain()[0].name, UiCommandName.STOP_ONLINE_INFERENCE)
        self.assertFalse(bus.online_inference_stop_requested())

    def test_raw_replay_stop_request_stays_visible_until_drained(self):
        bus = UiCommandBus()

        bus.submit(UiCommandName.STOP_RAW_REPLAY)

        self.assertTrue(bus.raw_replay_stop_requested())
        self.assertEqual(bus.drain()[0].name, UiCommandName.STOP_RAW_REPLAY)
        self.assertFalse(bus.raw_replay_stop_requested())


class RawReplayStopInterruptTest(unittest.TestCase):
    def test_raw_replay_frame_wait_is_interrupted_by_stop_request(self):
        from core.input.raw_offline import RawEpisodeInputProvider

        provider = object.__new__(RawEpisodeInputProvider)
        provider.speed_scale = 1.0
        provider._items = [
            {"timestamps": {"sample_monotonic_ns": 0}},
            {"timestamps": {"sample_monotonic_ns": 500_000_000}},
        ]
        provider._cursor = 1
        provider._first_sample_ns = 0
        provider._start_wall_time = time.time()
        provider._done = False
        provider._stop_interrupted = False
        stop_requested = Event()
        Timer(0.02, stop_requested.set).start()

        started = time.monotonic()
        sample = provider.get_sample(raw_replay_stop_requested=stop_requested.is_set)
        elapsed = time.monotonic() - started

        self.assertIsNone(sample)
        self.assertTrue(provider.stop_interrupted)
        self.assertLess(elapsed, 0.15)


class DualGripperSnapshotTest(unittest.TestCase):
    def test_reads_feedback_and_processed_command_under_one_lock(self):
        args = SimpleNamespace(no_gripper=False, ee="dex1")

        snapshot = read_dual_gripper_snapshot(
            args=args,
            dual_gripper_data_lock=Lock(),
            dual_gripper_state_array=[0.25, 0.75],
            dual_gripper_action_array=[0.50, 1.00],
        )

        self.assertEqual(snapshot, (0.25, 0.75, 0.50, 1.00))

    def test_disabled_gripper_has_no_ui_curve_values(self):
        args = SimpleNamespace(no_gripper=True, ee="dex1")

        snapshot = read_dual_gripper_snapshot(
            args=args,
            dual_gripper_data_lock=None,
            dual_gripper_state_array=None,
            dual_gripper_action_array=None,
        )

        self.assertEqual(snapshot, (None, None, None, None))


class XrProviderResumeTest(unittest.TestCase):
    def test_rebases_xr_and_ik_to_current_robot_pose_after_provider_switch(self):
        calls = []

        class Wrapper:
            def sync_reference_to_current_live_pose(self, *, require_live):
                calls.append(("xr", require_live))

        def reset_ik(arm_ik, arm_q):
            calls.append(("ik", arm_ik, arm_q))

        arm_ik = object()
        arm_q = [0.0] * 14
        operator_state.rebase_xr_takeover_after_provider_switch(
            tv_wrapper=Wrapper(),
            arm_ik=arm_ik,
            current_arm_q=arm_q,
            reset_arm_ik_state=reset_ik,
        )

        self.assertEqual(calls, [("xr", False), ("ik", arm_ik, arm_q)])


class ProviderSwitchTest(unittest.TestCase):
    def test_runtime_switches_xr_hold_raw_replay_hold(self):
        live_provider = object()
        created = []

        class FakeRawProvider:
            done = False

            def __init__(self, **kwargs):
                self.kwargs = dict(kwargs)
                created.append(self)

        runtime = TeleopProviderRuntime(
            live_provider=live_provider,
            replay_provider_factory=FakeRawProvider,
        )

        self.assertEqual(runtime.active_provider_kind, ActiveProviderKind.XR_LIVE)
        self.assertIs(runtime.active_provider(), live_provider)
        self.assertEqual(runtime.active_input_provider_name(), "xr")

        runtime.set_hold(reason="tab:playback")
        self.assertEqual(runtime.active_provider_kind, ActiveProviderKind.HOLD)
        self.assertIsNone(runtime.active_provider())
        self.assertEqual(runtime.active_input_provider_name(), "hold")

        runtime.start_raw_replay(
            dataset_root="/tmp/raw_root",
            episode_index=245,
            episode_name="episode_0245",
            arm_source="action",
            speed_scale=0.5,
        )
        self.assertEqual(runtime.active_provider_kind, ActiveProviderKind.RAW_REPLAY)
        self.assertIs(runtime.active_provider(), created[0])
        self.assertEqual(runtime.active_input_provider_name(), "lerobot_offline")
        self.assertEqual(created[0].kwargs["dataset_root"], "/tmp/raw_root")
        self.assertEqual(created[0].kwargs["episode_index"], 245)
        self.assertEqual(created[0].kwargs["arm_source"], "action")
        self.assertEqual(created[0].kwargs["speed_scale"], 0.5)

        motion_intent = MotionIntent(
            kind="joint_position",
            arm_q=np.zeros(14),
            frame_index=7,
            source="raw_episode:action:qpos",
        )
        sample = TeleopInputSample(
            tele_data=_build_offline_tele_data(np.eye(4), np.eye(4), np.zeros(2)),
            motion_intent=motion_intent,
            done=True,
        )
        runtime.note_sample(sample)
        self.assertEqual(runtime.status()["real_replay"]["frame_index"], 7)
        runtime.note_raw_replay_execution_trace(
            {"current": None, "latest": {"seq": 3, "status": "completed", "recv_to_pub_ms": 7.5}}
        )
        self.assertEqual(
            runtime.status()["real_replay"]["runtime_debug"]["execution_trace"]["latest"]["seq"],
            3,
        )

        runtime.finish_raw_replay(reason="provider_done")
        self.assertEqual(runtime.active_provider_kind, ActiveProviderKind.HOLD)
        self.assertIsNone(runtime.active_provider())
        self.assertEqual(runtime.status()["real_replay"]["state"], "finished")
        self.assertEqual(runtime.status()["real_replay"]["episode_name"], "episode_0245")

    def test_raw_replay_gripper_history_pairs_feedback_with_same_frame_recorded_state(self):
        runtime = TeleopProviderRuntime(live_provider=object(), replay_provider_factory=lambda **_: object())
        runtime.set_hold(reason="tab:playback")
        runtime.start_raw_replay(
            dataset_root="/tmp/raw_root",
            episode_index=1,
            episode_name="episode_0001",
            arm_source="action",
        )

        runtime.note_raw_replay_gripper_state_pair(
            frame_index=8,
            left_feedback_q=0.11,
            right_feedback_q=0.22,
            left_recorded_state_q=0.31,
            right_recorded_state_q=0.42,
        )
        runtime.note_raw_replay_gripper_state_pair(
            frame_index=8,
            left_feedback_q=0.12,
            right_feedback_q=0.23,
            left_recorded_state_q=0.31,
            right_recorded_state_q=0.42,
        )
        initial_debug = runtime.status()["real_replay"]["runtime_debug"]
        self.assertEqual(
            initial_debug["gripper_state_history"],
            [
                {
                    "frame_index": 8,
                    "left_feedback_q": 0.12,
                    "right_feedback_q": 0.23,
                    "left_recorded_state_q": 0.31,
                    "right_recorded_state_q": 0.42,
                }
            ],
        )
        for frame_index in range(9, 910):
            runtime.note_raw_replay_gripper_state_pair(
                frame_index=frame_index,
                left_feedback_q=float(frame_index),
                right_feedback_q=float(frame_index) + 0.1,
                left_recorded_state_q=float(frame_index) + 0.2,
                right_recorded_state_q=float(frame_index) + 0.3,
            )

        debug = runtime.status()["real_replay"]["runtime_debug"]
        history = debug["gripper_state_history"]
        self.assertEqual(len(history), 900)
        self.assertEqual(history[0]["frame_index"], 10)
        self.assertEqual(history[-1]["frame_index"], 909)
        self.assertEqual(debug["gripper_state_current"], history[-1])

    def test_runtime_rejects_raw_replay_start_unless_hold_is_active(self):
        runtime = TeleopProviderRuntime(live_provider=object(), replay_provider_factory=lambda **_: object())

        with self.assertRaises(RuntimeError):
            runtime.start_raw_replay(
                dataset_root="/tmp/raw_root",
                episode_index=1,
                episode_name="episode_0001",
                arm_source="action",
                speed_scale=1.0,
            )

    def test_runtime_switches_hold_online_inference_hold(self):
        class FakeOnlineProvider:
            def __init__(self):
                self.closed = False

            def get_debug_snapshot(self):
                return {"status": "collecting_observation", "error": None}

            def close(self):
                self.closed = True

        created = []

        def create_online_provider(*, prompt):
            self.assertEqual(prompt, "pick up the cube")
            provider = FakeOnlineProvider()
            created.append(provider)
            return provider

        runtime = TeleopProviderRuntime(
            live_provider=object(),
            online_provider_factory=create_online_provider,
        )

        runtime.set_hold(reason="ui_inference_start")
        runtime.start_online_inference(prompt="pick up the cube")

        self.assertEqual(runtime.active_provider_kind, ActiveProviderKind.ONLINE_INFERENCE)
        self.assertIs(runtime.active_provider(), created[0])
        self.assertEqual(runtime.active_input_provider_name(), "online_inference")
        self.assertEqual(runtime.status()["online_inference"]["state"], "running")
        self.assertEqual(runtime.status()["online_inference"]["prompt"], "pick up the cube")
        self.assertEqual(
            runtime.status()["online_inference"]["debug"],
            {"status": "collecting_observation", "error": None},
        )

        runtime.stop_online_inference(reason="ui_inference_stop")

        self.assertEqual(runtime.active_provider_kind, ActiveProviderKind.HOLD)
        self.assertTrue(created[0].closed)
        self.assertEqual(runtime.status()["online_inference"]["state"], "stopped")

    def test_runtime_fails_online_inference_closed_to_hold(self):
        provider = type("Provider", (), {"close": lambda self: setattr(self, "closed", True)})()
        provider.closed = False
        runtime = TeleopProviderRuntime(
            live_provider=object(),
            online_provider_factory=lambda **_kwargs: provider,
        )

        runtime.set_hold(reason="ui_inference_start")
        runtime.start_online_inference(prompt="pick up the cube")
        runtime.fail_online_inference("action response timeout")

        self.assertEqual(runtime.active_provider_kind, ActiveProviderKind.HOLD)
        self.assertTrue(provider.closed)
        self.assertEqual(runtime.status()["online_inference"]["state"], "error")
        self.assertEqual(runtime.status()["online_inference"]["error"], "action response timeout")

    def test_runtime_exposes_latest_online_inference_execution_debug(self):
        runtime = TeleopProviderRuntime(
            live_provider=object(),
            online_provider_factory=lambda **_kwargs: type("Provider", (), {"close": lambda self: None})(),
        )

        runtime.set_hold(reason="ui_inference_start")
        runtime.start_online_inference(prompt="pick up the cube")
        runtime.note_online_inference_runtime_debug(
            {
                "http_roundtrip_ms": 112.5,
                "ik_ms": 4.0,
                "trajectory": {
                    "left_target_xyz": [0.1, 0.2, 0.3],
                    "right_feedback_xyz": [0.4, 0.5, 0.6],
                },
            }
        )

        self.assertEqual(
            runtime.status()["online_inference"]["runtime_debug"],
            {
                "http_roundtrip_ms": 112.5,
                "ik_ms": 4.0,
                "trajectory": {
                    "left_target_xyz": [0.1, 0.2, 0.3],
                    "right_feedback_xyz": [0.4, 0.5, 0.6],
                },
            },
        )

    def test_runtime_keeps_execution_trace_when_action_debug_is_refreshed(self):
        runtime = TeleopProviderRuntime(
            live_provider=object(),
            online_provider_factory=lambda **_kwargs: type("Provider", (), {"close": lambda self: None})(),
        )

        runtime.set_hold(reason="ui_inference_start")
        runtime.start_online_inference(prompt="pick up the cube")
        runtime.note_online_inference_runtime_debug({"latency": {"ik_ms": 4.0}})
        runtime.note_online_inference_execution_trace(
            {"current": None, "latest": {"seq": 7, "status": "completed", "pub_to_exec_thread_ms": 18.0}}
        )
        runtime.note_online_inference_runtime_debug({"latency": {"ik_ms": 5.0}})

        debug = runtime.status()["online_inference"]["runtime_debug"]
        self.assertEqual(debug["latency"], {"ik_ms": 5.0})
        self.assertEqual(debug["execution_trace"]["latest"]["seq"], 7)


class UiStateStoreTest(unittest.TestCase):
    def test_snapshot_returns_copy_with_incrementing_version(self):
        store = UiStateStore()
        source = {"recording": {"active": False}}

        version = store.update(source)
        source["recording"]["active"] = True

        snapshot_version, snapshot = store.snapshot()
        self.assertEqual(snapshot_version, version)
        self.assertEqual(snapshot, {"recording": {"active": False}})

        next_version = store.update({"recording": {"active": True}})
        self.assertGreater(next_version, version)


class UiPayloadTest(unittest.TestCase):
    def test_build_web_payload_keeps_data_collector_recording_field_names(self):
        payload = build_web_payload(
            left_q_fb=[1.0, 2.0],
            right_q_fb=[3.0, 4.0],
            recording_status={
                "is_recording": True,
                "session_dir": "/tmp/session",
                "root_dir": "/tmp/root",
                "fps": 30.0,
                "frame_index": 12,
                "last_alignment": {"state_delta_ms": 1.0},
                "last_alert": {"level": "warning"},
                "alert_seq": 7,
                "last_validation": {"level": "ok"},
            },
            active_root_dir="/tmp/root",
            playback_status={"state": "idle"},
            convert_status={"state": "idle"},
            provider_status={
                "active_provider": "raw_replay",
                "real_replay": {"state": "running", "episode_name": "episode_0001", "frame_index": 4},
            },
            updated_mono=123.0,
        )

        self.assertEqual(payload["schema"], "data_collector/v1")
        self.assertEqual(payload["left"]["q_fb"], [1.0, 2.0])
        self.assertEqual(payload["right"]["q_fb"], [3.0, 4.0])
        self.assertTrue(payload["recording_active"])
        self.assertEqual(payload["recording"]["active"], True)
        self.assertEqual(payload["recording"]["session_dir"], "/tmp/session")
        self.assertEqual(payload["recording"]["active_root_dir"], "/tmp/root")
        self.assertEqual(payload["recording"]["frame_index"], 12)
        self.assertEqual(payload["recording"]["last_alert"], {"level": "warning"})
        self.assertEqual(payload["playback"], {"state": "idle"})
        self.assertEqual(payload["convert"], {"state": "idle"})
        self.assertEqual(payload["provider"]["active_provider"], "raw_replay")
        self.assertEqual(payload["provider"]["real_replay"]["frame_index"], 4)
        self.assertEqual(payload["updated_mono"], 123.0)

    def test_build_camera_status_keeps_data_collector_camera_field_names(self):
        status = build_camera_status(
            {
                "head": {
                    "frame_seq": 10,
                    "host_monotonic_ns": 1_000_000_000,
                    "shape": [640, 480, 3],
                    "transport": "local",
                },
                "left_wrist": None,
            },
            now_monotonic_ns=1_050_000_000,
        )

        self.assertEqual(status["active_camera_ids"], [0])
        self.assertEqual(len(status["streams"]), 1)
        stream = status["streams"][0]
        self.assertEqual(stream["camera_id"], 0)
        self.assertEqual(stream["camera_name"], "head")
        self.assertEqual(stream["camera_role"], "head")
        self.assertEqual(stream["camera_mode"], "rgb")
        self.assertEqual(stream["url"], "/camera/frame?camera_id=0")
        self.assertEqual(stream["shared_seq"], 10)
        self.assertEqual(stream["shared_timestamp_ns"], 1_000_000_000)
        self.assertEqual(stream["shared_age_ms"], 50)
        self.assertEqual(stream["actual_capture"]["frame_width"], 640)
        self.assertEqual(stream["actual_capture"]["frame_height"], 480)


class UiIntegrationTest(unittest.TestCase):
    def test_dispatch_ui_commands_reuses_existing_keyboard_keys(self):
        bus = UiCommandBus()
        bus.submit(UiCommandName.START)
        bus.submit(UiCommandName.HOME)
        bus.submit(UiCommandName.RECORD_TOGGLE)
        keys = []

        pressed = dispatch_ui_commands(bus.drain(), keys.append)

        self.assertEqual(pressed, ["r", "h", "s"])
        self.assertEqual(keys, ["r", "h", "s"])

    def test_build_runtime_recording_status_keeps_reference_field_names(self):
        args = SimpleNamespace(task_dir="/tmp/data", task_name="pick_cube", frequency=30.0)
        recorder = SimpleNamespace(item_id=4, episode_dir="/tmp/data/pick_cube/episode_0001")
        flow_state = SimpleNamespace(
            waiting_for_first_frame=True,
            record_start_monotonic_ns=123456,
            pending_samples=[1, 2],
        )
        recording_flow = SimpleNamespace(state=flow_state)

        status = build_runtime_recording_status(
            args=args,
            recorder=recorder,
            recording_flow=recording_flow,
            record_running=False,
        )

        self.assertTrue(status["is_recording"])
        self.assertTrue(status["active"])
        self.assertFalse(status["enabled"])
        self.assertEqual(status["phase"], "armed")
        self.assertEqual(status["root_dir"], "/tmp/data/pick_cube")
        self.assertEqual(status["active_root_dir"], "/tmp/data/pick_cube")
        self.assertEqual(status["session_dir"], "/tmp/data/pick_cube/episode_0001")
        self.assertEqual(status["fps"], 30.0)
        self.assertEqual(status["frame_index"], 5)
        self.assertEqual(status["last_alignment"]["pending_samples"], 2)
        self.assertTrue(status["last_alignment"]["waiting_for_first_frame"])

    def test_build_runtime_recording_status_reports_base_receiver_status(self):
        args = SimpleNamespace(
            task_dir="/tmp/data",
            task_name="pick_cube",
            frequency=30.0,
            record=True,
            record_base=True,
            base_odom_topic="rt/agv/odom",
            base_height_topic="rt/hispeed_state",
            base_state_max_age_ms=150.0,
            base_action_max_age_ms=500.0,
        )
        recorder = SimpleNamespace(item_id=0, episode_dir="/tmp/data/pick_cube/episode_0001")
        recording_flow = SimpleNamespace(state=SimpleNamespace(waiting_for_first_frame=False, pending_samples=[]))
        base_state_receiver = SimpleNamespace(is_alive=lambda: True)

        status = build_runtime_recording_status(
            args=args,
            recorder=recorder,
            recording_flow=recording_flow,
            record_running=True,
            base_state_receiver=base_state_receiver,
        )

        self.assertEqual(
            status["base"],
            {
                "enabled": True,
                "receiver_alive": True,
                "odom_topic": "rt/agv/odom",
                "height_topic": "rt/hispeed_state",
                "state_max_age_ms": 150.0,
                "action_max_age_ms": 500.0,
            },
        )

    def test_build_web_payload_preserves_recording_base_status(self):
        payload = build_web_payload(
            recording_status={
                "is_recording": False,
                "base": {
                    "enabled": True,
                    "receiver_alive": True,
                    "odom_topic": "rt/agv/odom",
                    "height_topic": "rt/hispeed_state",
                },
            },
            updated_mono=123.0,
        )

        self.assertEqual(
            payload["recording"]["base"],
            {
                "enabled": True,
                "receiver_alive": True,
                "odom_topic": "rt/agv/odom",
                "height_topic": "rt/hispeed_state",
            },
        )

    def test_build_runtime_camera_status_reads_latest_meta_without_owning_camera(self):
        class Source:
            def __init__(self, meta):
                self.meta = meta

            def get_latest(self, copy=True):
                return object(), dict(self.meta)

        cameras = SimpleNamespace(
            sources=lambda: {
                "head": Source(
                    {
                        "frame_seq": 3,
                        "host_monotonic_ns": 2_000_000_000,
                        "shape": [640, 480, 3],
                    }
                ),
                "left_wrist": None,
                "right_wrist": None,
            }
        )

        status = build_runtime_camera_status(cameras, now_monotonic_ns=2_010_000_000)

        self.assertEqual(status["active_camera_ids"], [0])
        self.assertEqual(status["streams"][0]["shared_seq"], 3)
        self.assertEqual(status["streams"][0]["shared_age_ms"], 10)

    def test_build_runtime_web_payload_appends_gripper_feedback_to_dual_arm_state(self):
        args = SimpleNamespace(task_dir="/tmp/data", task_name="pick_cube", frequency=20.0)
        recorder = SimpleNamespace(item_id=-1, episode_dir="")
        recording_flow = SimpleNamespace(state=SimpleNamespace(waiting_for_first_frame=False, pending_samples=[]))

        payload = build_runtime_web_payload(
            args=args,
            recorder=recorder,
            recording_flow=recording_flow,
            record_running=False,
            current_lr_arm_q=list(range(14)),
            current_left_gripper_q=0.25,
            current_right_gripper_q=0.75,
            current_left_gripper_cmd=0.50,
            current_right_gripper_cmd=1.00,
            current_state_sample_ns=111,
            started=True,
            ready=True,
            stopping=False,
            provider_status={"active_provider": "hold", "real_replay": {"state": "idle"}},
        )

        self.assertEqual(payload["left"]["q_fb"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25])
        self.assertEqual(payload["right"]["q_fb"], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 0.75])
        self.assertEqual(payload["left"]["gripper_q_fb"], 0.25)
        self.assertEqual(payload["right"]["gripper_q_fb"], 0.75)
        self.assertEqual(payload["left"]["gripper_q_cmd"], 0.50)
        self.assertEqual(payload["right"]["gripper_q_cmd"], 1.00)
        self.assertEqual(payload["left"]["stamp_fb_ns"], 111)
        self.assertEqual(payload["right"]["stamp_fb_ns"], 111)
        self.assertEqual(payload["recording"]["phase"], "idle")
        self.assertFalse(payload["recording"]["enabled"])
        self.assertTrue(payload["teleop"]["started"])
        self.assertTrue(payload["teleop"]["ready"])
        self.assertEqual(payload["provider"]["active_provider"], "hold")

    def test_build_runtime_web_payload_rejects_bad_arm_state_shape(self):
        args = SimpleNamespace(task_dir="/tmp/data", task_name="pick_cube", frequency=20.0)

        with self.assertRaises(ValueError):
            build_runtime_web_payload(
                args=args,
                recorder=None,
                recording_flow=None,
                record_running=False,
                current_lr_arm_q=[1, 2, 3],
            )


if __name__ == "__main__":
    unittest.main()
