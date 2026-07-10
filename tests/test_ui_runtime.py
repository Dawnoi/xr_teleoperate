import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.input.base import MotionIntent, TeleopInputSample, _build_offline_tele_data
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

        runtime.finish_raw_replay(reason="provider_done")
        self.assertEqual(runtime.active_provider_kind, ActiveProviderKind.HOLD)
        self.assertIsNone(runtime.active_provider())
        self.assertEqual(runtime.status()["real_replay"]["state"], "finished")
        self.assertEqual(runtime.status()["real_replay"]["episode_name"], "episode_0245")

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

    def test_build_runtime_web_payload_splits_current_dual_arm_state(self):
        args = SimpleNamespace(task_dir="/tmp/data", task_name="pick_cube", frequency=20.0)
        recorder = SimpleNamespace(item_id=-1, episode_dir="")
        recording_flow = SimpleNamespace(state=SimpleNamespace(waiting_for_first_frame=False, pending_samples=[]))

        payload = build_runtime_web_payload(
            args=args,
            recorder=recorder,
            recording_flow=recording_flow,
            record_running=False,
            current_lr_arm_q=list(range(14)),
            current_state_sample_ns=111,
            started=True,
            ready=True,
            stopping=False,
            provider_status={"active_provider": "hold", "real_replay": {"state": "idle"}},
        )

        self.assertEqual(payload["left"]["q_fb"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(payload["right"]["q_fb"], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
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
