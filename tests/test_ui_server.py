import http.client
import json
import pathlib
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.ui.command_bus import UiCommandBus, UiCommandName
from teleop.ui.episode_store import load_persisted_validation
from teleop.ui.exporter import DEFAULT_UI_URDF_PATH, UiExportManager, UiExportRequest
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
        three_status, _three_headers, three_body = self._get_bytes(server, "/assets/vendor/three.module.js")
        three_core_status, _three_core_headers, three_core_body = self._get_bytes(server, "/assets/vendor/three.core.min.js")

        self.assertEqual(index_status, 200)
        self.assertIn("app.css", index_body)
        self.assertEqual(css_status, 200)
        self.assertIn(":root", css_body)
        self.assertEqual(js_status, 200)
        self.assertEqual(three_status, 200)
        self.assertGreater(len(three_body), 300_000)
        self.assertIn("./three.core.min.js", three_body.decode("utf-8"))
        self.assertEqual(three_core_status, 200)
        self.assertGreater(len(three_core_body), 300_000)
        self.assertIn("function renderRecord", js_body)
        self.assertIn("renderInference", js_body)
        self.assertIn("/inference/start", js_body)
        self.assertIn("/inference/stop", js_body)
        self.assertIn("/inference/status", js_body)
        self.assertIn("实际发送 prompt", js_body)
        self.assertIn("latestObservation.prompt", js_body)
        self.assertIn("changedInference", js_body)
        self.assertIn("推理运行中，请先停止并 HOLD", js_body)
        self.assertIn("推理闭环延迟", js_body)
        self.assertIn("安全与下发", js_body)
        self.assertIn("执行 trace", js_body)
        self.assertIn("data-inference-camera", js_body)
        self.assertIn("inferenceDualTcpCanvas", js_body)
        self.assertIn("inferenceLeftRot6dCanvas", js_body)
        self.assertIn("WebGL", js_body)
        self.assertIn("allowApplicationError", js_body)
        self.assertIn('api("/convert/status", { allowApplicationError: true })', js_body)
        self.assertIn("episode_length_warnings", js_body)
        self.assertIn("displayedPlayback", js_body)
        self.assertIn("真机回放执行 trace", js_body)
        self.assertIn("gripper_q_cmd", js_body)
        self.assertIn("已下发目标", js_body)
        self.assertIn("playback-top-grid", js_body)
        self.assertIn("quickPlaybackToggle", js_body)
        self.assertIn("quickPlaybackFilter", js_body)
        self.assertIn("selectQuickPlaybackEpisode", js_body)
        self.assertIn("changedPlaybackRuntime", js_body)
        self.assertIn("recorded state q", js_body)
        self.assertIn("相机帧复用", js_body)
        self.assertIn("连续复用", js_body)
        self.assertIn("动作插值支撑质量", js_body)
        self.assertIn("最近邻备用", js_body)
        self.assertIn("playback-top-grid", css_body)
        self.assertIn("const trace = latestReplayTrace || pendingReplayTrace || {};", js_body)
        self.assertIn("function refreshRealReplayStatus()", js_body)
        self.assertIn("function reconcileRealReplayStatus", js_body)
        self.assertIn("realReplayCommandPending", js_body)
        self.assertIn("changedPlaybackProviderState", js_body)
        self.assertIn("function syncPlaybackTrace()", js_body)
        self.assertIn("playbackTraceBadge", js_body)
        self.assertLess(
            js_body.index('["export", "导出"'),
            js_body.index('["playback", "回放"'),
        )
        self.assertIn(DEFAULT_UI_URDF_PATH, js_body)
        self.assertIn("/assets/g1_d/g1_d.urdf", DEFAULT_UI_URDF_PATH)
        self.assertNotIn("/home/luodongxu/agx_arm_ws/src/nero-dual-arm", js_body)
        self.assertNotIn("/home/luopengcheng/Programs/nero-dual-arm", js_body)

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
        self.assertEqual(convert_status["state"], "idle")
        self.assertNotIn("error", convert_status)
        self.assertIn("ready", convert_status["message"])

    def test_inference_routes_queue_commands_and_report_provider_status(self):
        command_bus = UiCommandBus()
        server = TeleopUiServer(
            command_bus=command_bus,
            state_store=UiStateStore(
                {
                    "recording": {"active": False, "phase": "idle"},
                    "provider": {
                        "active_provider": "xr_live",
                        "online_inference": {"state": "idle", "prompt": "", "error": ""},
                    },
                }
            ),
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        start = self._get_json(server, "/inference/start?prompt=pick%20up%20the%20cube")
        status = self._get_json(server, "/inference/status")
        stop = self._get_json(server, "/inference/stop")

        self.assertEqual(start["command"], UiCommandName.START_ONLINE_INFERENCE.value)
        self.assertEqual(status["online_inference"]["state"], "idle")
        self.assertEqual(stop["command"], UiCommandName.STOP_ONLINE_INFERENCE.value)
        commands = command_bus.drain()
        self.assertEqual(
            [command.name for command in commands],
            [UiCommandName.START_ONLINE_INFERENCE, UiCommandName.STOP_ONLINE_INFERENCE],
        )
        self.assertEqual(commands[0].payload, {"prompt": "pick up the cube"})

    def test_inference_start_rejects_active_recording_and_raw_replay(self):
        command_bus = UiCommandBus()
        server = TeleopUiServer(
            command_bus=command_bus,
            state_store=UiStateStore(
                {
                    "recording": {"active": True, "phase": "recording"},
                    "provider": {"active_provider": "raw_replay"},
                }
            ),
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        status, result = self._get(server, "/inference/start?prompt=pick%20up%20the%20cube")

        self.assertEqual(status, 400)
        self.assertIn("recording", result)
        self.assertEqual(command_bus.drain(), [])

    def test_inference_start_rejects_active_raw_replay(self):
        command_bus = UiCommandBus()
        server = TeleopUiServer(
            command_bus=command_bus,
            state_store=UiStateStore(
                {
                    "recording": {"active": False, "phase": "idle"},
                    "provider": {"active_provider": "raw_replay"},
                }
            ),
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        status, result = self._get(server, "/inference/start?prompt=pick%20up%20the%20cube")

        self.assertEqual(status, 400)
        self.assertIn("raw_replay", result)
        self.assertEqual(command_bus.drain(), [])

    def test_inference_start_rejects_blank_prompt(self):
        command_bus = UiCommandBus()
        server = TeleopUiServer(
            command_bus=command_bus,
            state_store=UiStateStore(
                {
                    "recording": {"active": False, "phase": "idle"},
                    "provider": {"active_provider": "xr_live"},
                }
            ),
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        status, result = self._get(server, "/inference/start?prompt=%20%20")

        self.assertEqual(status, 400)
        self.assertIn("prompt", result)
        self.assertEqual(command_bus.drain(), [])

    def test_convert_start_route_passes_reference_query_to_manager(self):
        class FakeConvertManager:
            def __init__(self):
                self.requests = []

            def status(self):
                return {"ok": True, "state": "idle", "phase": "idle", "running": False, "message": "ready"}

            def start(self, request):
                self.requests.append(request)
                return {"ok": True, "state": "running", "phase": "exporting", "running": True, "message": "started"}

        manager = FakeConvertManager()
        server = TeleopUiServer(
            command_bus=UiCommandBus(),
            state_store=UiStateStore({"recording": {"active": False, "fps": 22.0}}),
            convert_manager=manager,
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        result = self._get_json(
            server,
            "/convert/start?"
            "source_root=/tmp/raw_task&output_root=/tmp/export&dataset_name=nero_ui"
            "&default_task=pick%20cube&format_version=v2&export_mode=replace"
            "&export_video=1&export_fk=0&export_verify=1"
            "&episode=episode_0001&episode=episode_0002",
        )

        self.assertEqual(result["ok"], True)
        self.assertEqual(result["phase"], "exporting")
        self.assertEqual(len(manager.requests), 1)
        request = manager.requests[0]
        self.assertEqual(str(request.source_root), "/tmp/raw_task")
        self.assertEqual(str(request.output_root), "/tmp/export")
        self.assertEqual(request.dataset_name, "nero_ui")
        self.assertEqual(request.task, "pick cube")
        self.assertEqual(request.fps, 22.0)
        self.assertEqual(request.format_version, "v2")
        self.assertEqual(request.export_mode, "replace")
        self.assertEqual(request.export_video, True)
        self.assertEqual(request.export_fk, False)
        self.assertEqual(request.export_verify, True)
        self.assertEqual(request.selected_episodes, ("episode_0001", "episode_0002"))

    def test_ui_export_manager_exports_selected_episodes_with_raw_v2_exporter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source_root = root / "raw_task"
            output_root = root / "exports"
            self._write_raw_episode_stub(source_root / "episode_0001", frame_count=2)
            self._write_raw_episode_stub(source_root / "episode_0002", frame_count=3)
            calls = []

            def fake_export_raw_task_dir(**kwargs):
                input_task_dir = pathlib.Path(kwargs["input_task_dir"])
                episode_names = sorted(path.name for path in input_task_dir.iterdir() if path.name.startswith("episode_"))
                calls.append({**kwargs, "episode_names": episode_names})
                return {
                    "source_root": str(input_task_dir),
                    "output_root": str(kwargs["output_root"]),
                    "task": kwargs["task"],
                    "fps": kwargs["fps"],
                    "verify_export": kwargs["verify_export"],
                    "verify_video_frames": kwargs["verify_video_frames"],
                    "episodes_total": 1,
                    "episodes_exported": 1,
                    "frames_total": 3,
                    "episodes": [{"source_episode": "episode_0002", "episode_index": 0, "frame_count": 3}],
                }

            manager = UiExportManager(export_raw_task_dir=fake_export_raw_task_dir)
            start_status = manager.start(
                UiExportRequest(
                    source_root=source_root,
                    output_root=output_root,
                    dataset_name="nero_dataset",
                    task="pick cube",
                    fps=30.0,
                    format_version="v2",
                    export_mode="replace",
                    export_video=True,
                    export_fk=True,
                    export_verify=True,
                    selected_episodes=("episode_0002",),
                    urdf_path=DEFAULT_UI_URDF_PATH,
                )
            )

            final_status = self._wait_convert_done(manager)

            self.assertEqual(start_status["ok"], True)
            self.assertEqual(final_status["ok"], True)
            self.assertEqual(final_status["state"], "done")
            self.assertEqual(final_status["phase"], "done")
            self.assertEqual(final_status["processed_frames"], 3)
            self.assertEqual(final_status["total_frames"], 3)
            self.assertEqual(final_status["output_total_frames"], 3)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["episode_names"], ["episode_0002"])
            self.assertEqual(pathlib.Path(calls[0]["output_root"]), output_root / "nero_dataset")
            self.assertEqual(calls[0]["task"], "pick cube")
            self.assertEqual(calls[0]["fps"], 30.0)
            self.assertEqual(calls[0]["overwrite"], True)
            self.assertEqual(calls[0]["verify_export"], True)
            self.assertEqual(calls[0]["verify_video_frames"], True)
            self.assertEqual(calls[0]["export_fk"], True)
            self.assertEqual(pathlib.Path(calls[0]["urdf_path"]), pathlib.Path(DEFAULT_UI_URDF_PATH))

    def test_ui_export_manager_rejects_missing_fk_urdf(self):
        manager = UiExportManager(export_raw_task_dir=lambda **_kwargs: {})

        result = manager.start(
            UiExportRequest(
                source_root=pathlib.Path("/tmp/raw_task"),
                output_root=pathlib.Path("/tmp/export"),
                dataset_name="nero_dataset",
                task="pick cube",
                fps=30.0,
                format_version="v2",
                export_mode="new",
                export_video=False,
                export_fk=True,
                export_verify=False,
                selected_episodes=(),
                urdf_path="/tmp/xr_teleoperate_missing_g1d.urdf",
            )
        )

        self.assertEqual(result["ok"], False)
        self.assertEqual(result["state"], "error")
        self.assertIn("urdf_path", result["error"])

    def test_ui_export_manager_rejects_selected_empty_episode_before_starting_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source_root = root / "raw_task"
            output_root = root / "exports"
            self._write_raw_episode_stub(source_root / "episode_0001", frame_count=2)
            self._write_raw_episode_stub(source_root / "episode_0002", frame_count=0)
            calls = []
            manager = UiExportManager(export_raw_task_dir=lambda **kwargs: calls.append(kwargs) or {})

            result = manager.start(
                UiExportRequest(
                    source_root=source_root,
                    output_root=output_root,
                    dataset_name="nero_dataset",
                    task="pick cube",
                    fps=30.0,
                    selected_episodes=("episode_0001", "episode_0002"),
                    urdf_path=DEFAULT_UI_URDF_PATH,
                )
            )

            self.assertEqual(result["ok"], False)
            self.assertEqual(result["state"], "error")
            self.assertIn("episode_0002", result["error"])
            self.assertIn("no samples", result["error"])
            self.assertEqual(calls, [])

    def test_ui_export_manager_warns_for_selected_episode_length_outliers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source_root = root / "raw_task"
            output_root = root / "exports"
            frame_counts = {
                "episode_0001": 100,
                "episode_0002": 100,
                "episode_0003": 100,
                "episode_0004": 1000,
            }
            for episode_name, frame_count in frame_counts.items():
                self._write_raw_episode_stub(source_root / episode_name, frame_count=frame_count)
            release_export = threading.Event()

            def slow_export_raw_task_dir(**_kwargs):
                release_export.wait(timeout=2.0)
                return {"episodes_exported": 4, "frames_total": sum(frame_counts.values())}

            manager = UiExportManager(export_raw_task_dir=slow_export_raw_task_dir)
            result = manager.start(
                UiExportRequest(
                    source_root=source_root,
                    output_root=output_root,
                    dataset_name="nero_dataset",
                    task="pick cube",
                    fps=30.0,
                    selected_episodes=tuple(frame_counts),
                    urdf_path=DEFAULT_UI_URDF_PATH,
                )
            )
            release_export.set()
            self._wait_convert_done(manager)

            self.assertEqual(result["ok"], True)
            self.assertEqual(result["episode_length_reference"]["method"], "iqr")
            self.assertEqual(result["episode_length_reference"]["sample_count"], 4)
            self.assertEqual(
                result["episode_length_warnings"],
                [{"kind": "long", "episode": "episode_0004", "frame_count": 1000}],
            )

    def test_ui_export_manager_reports_bad_source_root_as_status_error(self):
        manager = UiExportManager(export_raw_task_dir=lambda **_kwargs: {})

        result = manager.start(
            UiExportRequest(
                source_root=pathlib.Path("/tmp/xr_teleoperate_missing_raw_task"),
                output_root=pathlib.Path("/tmp/export"),
                dataset_name="nero_dataset",
                task="pick cube",
                fps=30.0,
                format_version="v2",
                export_mode="new",
                export_video=False,
                export_fk=False,
                export_verify=False,
                selected_episodes=("episode_0001",),
                urdf_path=DEFAULT_UI_URDF_PATH,
            )
        )

        self.assertEqual(result["ok"], False)
        self.assertEqual(result["state"], "error")
        self.assertIn("source_root", result["error"])

    def test_ui_export_manager_rejects_dataset_name_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source_root = root / "raw_task"
            output_root = root / "exports"
            self._write_raw_episode_stub(source_root / "episode_0001", frame_count=1)
            calls = []
            manager = UiExportManager(export_raw_task_dir=lambda **kwargs: calls.append(kwargs) or {})

            result = manager.start(
                UiExportRequest(
                    source_root=source_root,
                    output_root=output_root,
                    dataset_name="..",
                    task="pick cube",
                    fps=30.0,
                    format_version="v2",
                    export_mode="replace",
                    export_video=False,
                    export_fk=False,
                    export_verify=False,
                    selected_episodes=("episode_0001",),
                    urdf_path=DEFAULT_UI_URDF_PATH,
                )
            )

            self.assertEqual(result["ok"], False)
            self.assertEqual(result["state"], "error")
            self.assertIn("dataset_name", result["error"])
            self.assertEqual(calls, [])

    def test_ui_export_manager_rejects_output_root_overlapping_source_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source_root = root / "raw_task"
            self._write_raw_episode_stub(source_root / "episode_0001", frame_count=1)
            self._write_raw_episode_stub(source_root / "episode_0002", frame_count=1)
            calls = []
            manager = UiExportManager(export_raw_task_dir=lambda **kwargs: calls.append(kwargs) or {})

            equal_source_result = manager.start(
                UiExportRequest(
                    source_root=source_root,
                    output_root=root,
                    dataset_name="raw_task",
                    task="pick cube",
                    fps=30.0,
                    format_version="v2",
                    export_mode="replace",
                    export_video=False,
                    export_fk=False,
                    export_verify=False,
                    selected_episodes=("episode_0002",),
                    urdf_path=DEFAULT_UI_URDF_PATH,
                )
            )
            ancestor_result = manager.start(
                UiExportRequest(
                    source_root=source_root,
                    output_root=pathlib.Path(tmp).parent,
                    dataset_name=pathlib.Path(tmp).name,
                    task="pick cube",
                    fps=30.0,
                    format_version="v2",
                    export_mode="replace",
                    export_video=False,
                    export_fk=False,
                    export_verify=False,
                    selected_episodes=("episode_0002",),
                    urdf_path=DEFAULT_UI_URDF_PATH,
                )
            )

            self.assertEqual(equal_source_result["ok"], False)
            self.assertIn("overlap", equal_source_result["error"])
            self.assertEqual(ancestor_result["ok"], False)
            self.assertIn("overlap", ancestor_result["error"])
            self.assertEqual(calls, [])

    def test_convert_start_route_rejects_missing_output_root_before_path_coercion(self):
        class FakeConvertManager:
            def __init__(self):
                self.requests = []

            def status(self):
                return {"ok": True, "state": "idle", "phase": "idle", "running": False, "message": "ready"}

            def start(self, request):
                self.requests.append(request)
                return {"ok": True, "state": "running", "phase": "exporting", "running": True, "message": "started"}

        manager = FakeConvertManager()
        server = TeleopUiServer(
            command_bus=UiCommandBus(),
            state_store=UiStateStore({"recording": {"active": False, "fps": 22.0}}),
            convert_manager=manager,
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        result = self._get_json(
            server,
            "/convert/start?source_root=/tmp/raw_task&dataset_name=nero_ui&default_task=pick%20cube",
        )

        self.assertEqual(result["ok"], False)
        self.assertIn("output_root", result["error"])
        self.assertEqual(manager.requests, [])

    def test_ui_export_manager_rejects_concurrent_start_without_clobbering_running_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source_root = root / "raw_task"
            output_root = root / "exports"
            self._write_raw_episode_stub(source_root / "episode_0001", frame_count=1)
            release_export = threading.Event()

            def slow_export_raw_task_dir(**kwargs):
                release_export.wait(timeout=2.0)
                return {
                    "source_root": str(kwargs["input_task_dir"]),
                    "output_root": str(kwargs["output_root"]),
                    "task": kwargs["task"],
                    "fps": kwargs["fps"],
                    "episodes_exported": 1,
                    "frames_total": 1,
                    "episodes": [{"source_episode": "episode_0001", "episode_index": 0, "frame_count": 1}],
                }

            manager = UiExportManager(export_raw_task_dir=slow_export_raw_task_dir)
            request = UiExportRequest(
                source_root=source_root,
                output_root=output_root,
                dataset_name="nero_dataset",
                task="pick cube",
                fps=30.0,
                format_version="v2",
                export_mode="new",
                export_video=False,
                export_fk=False,
                export_verify=False,
                selected_episodes=("episode_0001",),
                urdf_path=DEFAULT_UI_URDF_PATH,
            )
            first = manager.start(request)
            second = manager.start(request)
            running_status = manager.status()
            release_export.set()
            final_status = self._wait_convert_done(manager)

            self.assertEqual(first["ok"], True)
            self.assertEqual(second["ok"], False)
            self.assertIn("already running", second["error"])
            self.assertEqual(running_status["state"], "running")
            self.assertEqual(running_status["running"], True)
            self.assertNotIn("error", running_status)
            self.assertEqual(final_status["state"], "done")

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
                                "states": {"left": {"qpos": [1, 2, 3, 4, 5, 6, 7]}, "right": {"qpos": [8, 9, 10, 11, 12, 13, 14]}, "left_ee": {"qpos": [0.2]}, "right_ee": {"qpos": [0.3]}},
                                "actions": {"left": {"qpos": [1, 2, 3, 4, 5, 6, 7]}, "right": {"qpos": [8, 9, 10, 11, 12, 13, 14]}},
                                "timestamps": {"sample_monotonic_ns": 1000},
                            },
                            {
                                "idx": 1,
                                "colors": {"head": "colors/head/000000_head_1000.jpg"},
                                "states": {"left": {"qpos": [1, 2, 3, 4, 5, 6, 7]}, "right": {"qpos": [8, 9, 10, 11, 12, 13, 14]}, "left_ee": {"qpos": [0.4]}, "right_ee": {"qpos": [0.5]}},
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
            self.assertEqual(episodes["episodes"][0]["validation_level"], "error")
            self.assertEqual(episodes["episodes"][0]["validation"]["status"], "missing")

            load_result = self._get_json(server, f"/playback/load?root_dir={root}&episode=episode_0001")
            self.assertEqual(load_result["ok"], True)
            status = self._get_json(server, "/playback/status")
            self.assertEqual(status["episode_name"], "episode_0001")
            self.assertEqual(status["total_frames"], 2)
            self.assertEqual(status["cameras"][0]["camera_name"], "head")

            curves = self._get_json(server, "/playback/curves?max_points=900")
            self.assertEqual(curves["left"]["gripper_state"], [0.2, 0.4])
            self.assertEqual(curves["right"]["gripper_state"], [0.3, 0.5])

            image_status, headers, body = self._get_bytes(server, "/playback/image?camera_id=0&frame=0")
            self.assertEqual(image_status, 200)
            self.assertEqual(headers["content-type"], "image/jpeg")
            self.assertEqual(body, b"\xff\xd8\xff\xd9")

    def test_recording_episodes_returns_all_episodes_unless_limit_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for episode_index in range(81):
                self._write_raw_episode_stub(root / f"episode_{episode_index:04d}", frame_count=1)

            server = TeleopUiServer(
                command_bus=UiCommandBus(),
                state_store=UiStateStore({"recording": {"active": False}}),
                host="127.0.0.1",
                port=0,
            )
            server.start()
            self.addCleanup(server.stop)

            all_episodes = self._get_json(server, f"/recording/episodes?root_dir={root}")
            limited_episodes = self._get_json(server, f"/recording/episodes?root_dir={root}&limit=7")
            invalid_status, invalid_body = self._get(server, f"/recording/episodes?root_dir={root}&limit=all")

            self.assertEqual(len(all_episodes["episodes"]), 81)
            self.assertEqual(all_episodes["episodes"][0]["name"], "episode_0080")
            self.assertEqual(all_episodes["episodes"][-1]["name"], "episode_0000")
            self.assertEqual(len(limited_episodes["episodes"]), 7)
            self.assertEqual(invalid_status, 400)
            self.assertIn("episode limit must be a non-negative integer", invalid_body)

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
            self.assertEqual(validation["level"], "error")
            self.assertEqual(validation["status"], "incomplete")
            self.assertTrue(any("not finalized" in item for item in validation["errors"]))

    def test_persisted_validation_is_stale_after_data_json_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode_dir = pathlib.Path(tmp) / "episode_0001"
            episode_dir.mkdir()
            data_path = episode_dir / "data.json"
            data_path.write_text('{"data": []}\n', encoding="utf-8")
            source_stat = data_path.stat()
            (episode_dir / "validation.json").write_text(
                json.dumps(
                    {
                        "checked_at_ns": 1,
                        "level": "ok",
                        "source_data": {
                            "size_bytes": source_stat.st_size,
                            "mtime_ns": source_stat.st_mtime_ns,
                        },
                        "errors": [],
                        "warnings": [],
                    }
                ),
                encoding="utf-8",
            )
            data_path.write_text('{"data": [1]}\n', encoding="utf-8")

            validation = load_persisted_validation(episode_dir)

            self.assertEqual(validation["level"], "error")
            self.assertEqual(validation["status"], "stale")
            self.assertTrue(any("stale" in item for item in validation["errors"]))

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

        for path in ("/camera/start", "/camera/stop"):
            result = self._get_json(server, path)
            self.assertEqual(result["ok"], False)
            self.assertIn("not implemented", result["error"])

    def test_recording_root_change_queues_control_loop_command(self):
        command_bus = UiCommandBus()
        server = TeleopUiServer(
            command_bus=command_bus,
            state_store=UiStateStore({"recording": {"active": False, "enabled": True}}),
            host="127.0.0.1",
            port=0,
        )
        server.start()
        self.addCleanup(server.stop)

        result = self._get_json(server, "/recording/set_root_dir?root_dir=/tmp/new_raw_task")

        self.assertEqual(result["ok"], True)
        command = command_bus.drain()
        self.assertEqual([item.name for item in command], [UiCommandName.SET_RECORD_ROOT])
        self.assertEqual(command[0].payload, {"root_dir": "/tmp/new_raw_task"})

    def test_delete_episodes_removes_only_requested_episode_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            first = root / "episode_0001"
            second = root / "episode_0002"
            first.mkdir()
            second.mkdir()
            (first / "data.json").write_text("{}", encoding="utf-8")
            (second / "data.json").write_text("{}", encoding="utf-8")
            server = TeleopUiServer(
                command_bus=UiCommandBus(),
                state_store=UiStateStore({"recording": {"active": False, "enabled": True}}),
                host="127.0.0.1",
                port=0,
            )
            server.start()
            self.addCleanup(server.stop)
            server.playback.load(str(root), "episode_0001")

            result = self._get_json(server, f"/recording/delete_episodes?root_dir={root}&episode=episode_0001")

            self.assertEqual(result["deleted"], ["episode_0001"])
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual(server.playback.state, "disabled")

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

    @staticmethod
    def _write_raw_episode_stub(episode_dir: pathlib.Path, frame_count: int) -> None:
        episode_dir.mkdir(parents=True)
        rows = []
        for frame_index in range(frame_count):
            rows.append(
                {
                    "idx": frame_index,
                    "colors": {},
                    "states": {},
                    "actions": {},
                    "timestamps": {"sample_monotonic_ns": frame_index + 1},
                }
            )
        (episode_dir / "data.json").write_text(json.dumps({"data": rows}), encoding="utf-8")

    @staticmethod
    def _wait_convert_done(manager: UiExportManager) -> dict:
        deadline = time.monotonic() + 2.0
        status = manager.status()
        while status.get("running") and time.monotonic() < deadline:
            time.sleep(0.01)
            status = manager.status()
        return status


if __name__ == "__main__":
    unittest.main()
