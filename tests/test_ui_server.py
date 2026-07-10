import http.client
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.ui.command_bus import UiCommandBus, UiCommandName
from teleop.ui.server import TeleopUiServer
from teleop.ui.state_store import UiStateStore


class TeleopUiServerTest(unittest.TestCase):
    def test_snapshot_and_camera_status_use_reference_routes(self):
        state_store = UiStateStore({"recording": {"active": False, "frame_index": 0}})
        command_bus = UiCommandBus()
        server = TeleopUiServer(
            command_bus=command_bus,
            state_store=state_store,
            camera_status_getter=lambda: {"active_camera_ids": [0], "streams": []},
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        snapshot = self._get_json(server, "/snapshot")
        camera_status = self._get_json(server, "/camera/status")

        self.assertEqual(snapshot["recording"]["active"], False)
        self.assertEqual(camera_status["active_camera_ids"], [0])

    def test_reference_static_assets_are_served(self):
        server = TeleopUiServer(
            command_bus=UiCommandBus(),
            state_store=UiStateStore({"recording": {"active": False}}),
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        index_status, index_body = self._get(server, "/")
        css_status, css_body = self._get(server, "/assets/app.css")
        js_status, js_body = self._get(server, "/assets/app.js")

        self.assertEqual(index_status, 200)
        self.assertIn("app.css", index_body)
        self.assertEqual(css_status, 200)
        self.assertIn(":root", css_body)
        self.assertEqual(js_status, 200)
        self.assertIn("function renderRecord", js_body)

    def test_reference_frontend_empty_data_routes_exist(self):
        server = TeleopUiServer(
            command_bus=UiCommandBus(),
            state_store=UiStateStore({"recording": {"active": False}}),
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        self.assertEqual(self._get_json(server, "/camera/realsense_list"), {"devices": []})
        self.assertEqual(self._get_json(server, "/recording/episodes"), {"ok": True, "episodes": [], "root_dir": "."})
        convert_status = self._get_json(server, "/convert/status")
        self.assertEqual(convert_status["state"], "disabled")
        self.assertIn("not implemented", convert_status["error"])

    def test_recording_episodes_and_playback_read_raw_episode_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            episode_dir = root / "episode_0001"
            colors_dir = episode_dir / "colors" / "head"
            colors_dir.mkdir(parents=True)
            image_path = colors_dir / "000000_head_1000.jpg"
            image_path_2 = colors_dir / "000001_head_34334333.jpg"
            image_path.write_bytes(b"\xff\xd8\xff\xd9")
            image_path_2.write_bytes(b"\xff\xd8\xff\xd9")
            (episode_dir / "data.json").write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "idx": 0,
                                "colors": {"head": "colors/head/000001_head_34334333.jpg"},
                                "states": {"left": {"qpos": [1, 2, 3, 4, 5, 6, 7]}, "right": {"qpos": [8, 9, 10, 11, 12, 13, 14]}},
                                "actions": {"left": {"qpos": [1, 2, 3, 4, 5, 6, 7]}, "right": {"qpos": [8, 9, 10, 11, 12, 13, 14]}},
                                "timestamps": {"sample_monotonic_ns": 1000},
                            },
                            {
                                "idx": 1,
                                "colors": {"head": "colors/head/000000_head_1000.jpg"},
                                "states": {"left": {"qpos": [1, 2, 3, 4, 5, 6, 7]}, "right": {"qpos": [8, 9, 10, 11, 12, 13, 14]}},
                                "actions": {"left": {"qpos": [1, 2, 3, 4, 5, 6, 7]}, "right": {"qpos": [8, 9, 10, 11, 12, 13, 14]}},
                                "timestamps": {"sample_monotonic_ns": 34_334_333},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            server = TeleopUiServer(
                command_bus=UiCommandBus(),
                state_store=UiStateStore({"recording": {"active": False}}),
                host="127.0.0.1",
                port=0,
            )
            server.start()
            self.addCleanup(server.stop)

            episodes = self._get_json(server, f"/recording/episodes?root_dir={root}")
            self.assertEqual([item["name"] for item in episodes["episodes"]], ["episode_0001"])
            self.assertEqual(episodes["episodes"][0]["frame_count"], 2)
            self.assertEqual(episodes["episodes"][0]["validation_level"], "ok")

            load_result = self._get_json(server, f"/playback/load?root_dir={root}&episode=episode_0001")
            self.assertEqual(load_result["ok"], True)
            status = self._get_json(server, "/playback/status")
            self.assertEqual(status["episode_name"], "episode_0001")
            self.assertEqual(status["total_frames"], 2)
            self.assertEqual(status["cameras"][0]["camera_name"], "head")

            image_status, headers, body = self._get_bytes(server, "/playback/image?camera_id=0&frame=0")
            self.assertEqual(image_status, 200)
            self.assertEqual(headers["content-type"], "image/jpeg")
            self.assertEqual(body, b"\xff\xd8\xff\xd9")

    def test_recording_status_reports_incomplete_latest_episode_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            episode_dir = root / "episode_0001"
            episode_dir.mkdir(parents=True)
            (episode_dir / "data.json").write_text(
                '{"info": {}, "text": {}, "data": [\n'
                '{"idx": 0, "colors": {}, "states": {}, "actions": {}, "timestamps": {}}\n',
                encoding="utf-8",
            )
            state_store = UiStateStore(
                {
                    "recording": {
                        "active": False,
                        "root_dir": str(root),
                        "active_root_dir": str(root),
                    }
                }
            )
            server = TeleopUiServer(
                command_bus=UiCommandBus(),
                state_store=state_store,
                host="127.0.0.1",
                port=0,
            )
            server.start()
            self.addCleanup(server.stop)

            status = self._get_json(server, "/recording/status")

            validation = status["last_validation"]
            self.assertEqual(validation["episode_name"], "episode_0001")
            self.assertEqual(validation["level"], "warning")
            self.assertTrue(any("not finalized" in item for item in validation["warnings"]))

    def test_recording_start_and_stop_routes_queue_toggle_only_when_state_matches(self):
        state_store = UiStateStore({"recording": {"active": False, "frame_index": 0}})
        command_bus = UiCommandBus()
        server = TeleopUiServer(
            command_bus=command_bus,
            state_store=state_store,
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        start_result = self._get_json(server, "/recording/start")
        self.assertEqual(start_result["ok"], True)
        self.assertEqual([command.name for command in command_bus.drain()], [UiCommandName.RECORD_TOGGLE])

        state_store.update({"recording": {"active": True, "frame_index": 1}})
        active_start_result = self._get_json(server, "/recording/start")
        self.assertEqual(active_start_result["ok"], False)
        self.assertEqual(command_bus.drain(), [])

        stop_result = self._get_json(server, "/recording/stop")
        self.assertEqual(stop_result["ok"], True)
        self.assertEqual([command.name for command in command_bus.drain()], [UiCommandName.RECORD_TOGGLE])

        state_store.update({"recording": {"active": False, "frame_index": 1}})
        inactive_stop_result = self._get_json(server, "/recording/stop")
        self.assertEqual(inactive_stop_result["ok"], False)
        self.assertEqual(command_bus.drain(), [])

    def test_recording_start_route_rejects_when_recording_is_disabled(self):
        state_store = UiStateStore({"recording": {"active": False, "enabled": False, "frame_index": 0}})
        command_bus = UiCommandBus()
        server = TeleopUiServer(
            command_bus=command_bus,
            state_store=state_store,
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        start_result = self._get_json(server, "/recording/start")

        self.assertEqual(start_result["ok"], False)
        self.assertIn("recording is disabled", start_result["error"])
        self.assertEqual(command_bus.drain(), [])

    def test_command_routes_queue_existing_keyboard_equivalent_commands(self):
        command_bus = UiCommandBus()
        server = TeleopUiServer(
            command_bus=command_bus,
            state_store=UiStateStore({"recording": {"active": False}}),
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        self._get_json(server, "/command/start")
        self._get_json(server, "/command/home")
        self._get_json(server, "/command/recenter")
        self._get_json(server, "/recording/toggle")
        self._get_json(server, "/recording/cancel")
        self._get_json(server, "/command/stop")

        self.assertEqual(
            [command.name for command in command_bus.drain()],
            [
                UiCommandName.START,
                UiCommandName.HOME,
                UiCommandName.RECENTER,
                UiCommandName.RECORD_TOGGLE,
                UiCommandName.RECORD_CANCEL,
                UiCommandName.STOP,
            ],
        )

    def test_real_replay_routes_queue_payload_commands_when_recording_idle(self):
        command_bus = UiCommandBus()
        state_store = UiStateStore(
            {
                "recording": {
                    "active": False,
                    "phase": "idle",
                    "last_alignment": {"waiting_for_first_frame": False},
                },
                "playback": {"episode_name": "episode_0245"},
                "provider": {"active_provider": "hold"},
                "teleop": {"started": True},
            }
        )
        server = TeleopUiServer(
            command_bus=command_bus,
            state_store=state_store,
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        start_result = self._get_json(
            server,
            "/replay/real/start?root_dir=/tmp/raw_root&episode=episode_0245&arm_source=action&speed_scale=0.5",
        )
        stop_result = self._get_json(server, "/replay/real/stop")
        commands = command_bus.drain()

        self.assertEqual(start_result["ok"], True)
        self.assertEqual(stop_result["ok"], True)
        self.assertEqual([command.name for command in commands], [UiCommandName.START_RAW_REPLAY, UiCommandName.STOP_RAW_REPLAY])
        self.assertEqual(commands[0].payload["dataset_root"], "/tmp/raw_root")
        self.assertEqual(commands[0].payload["episode_index"], 245)
        self.assertEqual(commands[0].payload["episode_name"], "episode_0245")
        self.assertEqual(commands[0].payload["arm_source"], "action")
        self.assertEqual(commands[0].payload["speed_scale"], 0.5)

    def test_real_replay_start_rejects_when_recording_active_or_armed(self):
        for recording in (
            {"active": True, "phase": "recording", "last_alignment": {"waiting_for_first_frame": False}},
            {"active": False, "phase": "armed", "last_alignment": {"waiting_for_first_frame": True}},
        ):
            command_bus = UiCommandBus()
            server = TeleopUiServer(
                command_bus=command_bus,
                state_store=UiStateStore({"recording": recording, "teleop": {"started": True}}),
                host="127.0.0.1",
                port=0,
            )
            server.start()
            self.addCleanup(server.stop)

            result = self._get_json(server, "/replay/real/start?root_dir=/tmp/raw_root&episode=episode_0001")

            self.assertEqual(result["ok"], False)
            self.assertIn("recording", result["error"])
            self.assertEqual(command_bus.drain(), [])

    def test_camera_frame_route_serves_latest_jpeg_from_getter(self):
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        frame[:, :, 1] = 255
        server = TeleopUiServer(
            command_bus=UiCommandBus(),
            state_store=UiStateStore({"recording": {"active": False}}),
            camera_frame_getter=lambda camera_id: (frame, {"frame_seq": 1}) if camera_id == 0 else (None, None),
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        status, headers, body = self._get_bytes(server, "/camera/frame?camera_id=0")

        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "image/jpeg")
        self.assertTrue(body.startswith(b"\xff\xd8"))

    def test_unsupported_reference_routes_fail_explicitly(self):
        server = TeleopUiServer(
            command_bus=UiCommandBus(),
            state_store=UiStateStore({"recording": {"active": False}}),
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        for path in ("/camera/start", "/camera/stop", "/convert/start"):
            result = self._get_json(server, path)
            self.assertEqual(result["ok"], False)
            self.assertIn("not implemented", result["error"])

    def test_events_route_returns_when_client_disconnects(self):
        class BrokenPipeWriter:
            def write(self, payload):
                raise BrokenPipeError("client disconnected")

            def flush(self):
                raise AssertionError("flush must not run after write disconnect")

        class Handler:
            wfile = BrokenPipeWriter()

            def __init__(self):
                self.responses = []
                self.headers = []

            def send_response(self, status):
                self.responses.append(status)

            def send_header(self, name, value):
                self.headers.append((name, value))

            def end_headers(self):
                return None

        server = TeleopUiServer(
            command_bus=UiCommandBus(),
            state_store=UiStateStore({"recording": {"active": False}}),
            host="127.0.0.1",
            port=0,
            publish_rate_hz=1000.0,
        )
        server._running.set()
        handler = Handler()

        server._serve_events(handler)

        self.assertEqual(handler.responses, [200])

    @staticmethod
    def _get_json(server: TeleopUiServer, path: str) -> dict:
        status, body = TeleopUiServerTest._get(server, path)
        if status >= 400:
            return json.loads(body)
        return json.loads(body)

    @staticmethod
    def _get(server: TeleopUiServer, path: str) -> tuple[int, str]:
        conn = http.client.HTTPConnection(server.host, server.port, timeout=2)
        conn.request("GET", path)
        response = conn.getresponse()
        status = int(response.status)
        body = response.read().decode("utf-8")
        conn.close()
        return status, body

    @staticmethod
    def _get_bytes(server: TeleopUiServer, path: str) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection(server.host, server.port, timeout=2)
        conn.request("GET", path)
        response = conn.getresponse()
        status = int(response.status)
        headers = {k.lower(): v for k, v in response.getheaders()}
        body = response.read()
        conn.close()
        return status, headers, body


if __name__ == "__main__":
    unittest.main()
