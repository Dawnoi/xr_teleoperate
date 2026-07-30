from __future__ import annotations

import json
import mimetypes
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import cv2

from teleop.ui.command_bus import UiCommandBus, UiCommandName
from teleop.ui.episode_store import PlaybackSession, latest_episode_summary, list_episodes
from teleop.ui.exporter import DEFAULT_UI_URDF_PATH, UiExportManager, UiExportRequest
from teleop.ui.state_store import UiStateStore


EPISODE_NAME_RE = re.compile(r"episode_\d+$")


class TeleopUiServer:
    """Small web server matching the reference data_collector route shape."""

    def __init__(
        self,
        *,
        command_bus: UiCommandBus,
        state_store: UiStateStore,
        camera_status_getter: Callable[[], dict[str, Any]] | None = None,
        camera_frame_getter: Callable[[int], tuple[Any, dict[str, Any] | None]] | None = None,
        inference_profiles_getter: Callable[[], list[dict[str, Any]]] | None = None,
        web_root: str | Path | None = None,
        convert_manager: UiExportManager | None = None,
        host: str = "127.0.0.1",
        port: int = 8085,
        publish_rate_hz: float = 5.0,
    ) -> None:
        self.command_bus = command_bus
        self.state_store = state_store
        self.camera_status_getter = camera_status_getter if camera_status_getter is not None else self._empty_camera_status
        self.camera_frame_getter = camera_frame_getter if camera_frame_getter is not None else self._empty_camera_frame
        self.inference_profiles_getter = (
            inference_profiles_getter if inference_profiles_getter is not None else self._default_inference_profiles
        )
        self.web_root = Path(web_root) if web_root is not None else Path(__file__).resolve().parent / "web"
        self.convert_manager = convert_manager if convert_manager is not None else UiExportManager()
        self.host = str(host)
        self.port = int(port)
        self.publish_rate_hz = float(publish_rate_hz)
        self.playback = PlaybackSession()
        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self) -> None:
        handler_cls = self._build_handler()
        self._running.set()
        self._http_server = ThreadingHTTPServer((self.host, self.port), handler_cls)
        self.port = int(self._http_server.server_address[1])
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
        self._http_server = None
        self._http_thread = None

    def _build_handler(self):
        owner = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path

                if path == "/" or path == "/index.html":
                    owner._serve_index(self)
                    return
                if path.startswith("/assets/"):
                    owner._serve_static(self, path)
                    return
                if path == "/events":
                    owner._serve_events(self)
                    return
                if path == "/snapshot":
                    _, snapshot = owner.state_store.snapshot()
                    owner._json_ok(self, snapshot)
                    return
                if path == "/camera/status":
                    owner._json_ok(self, owner.camera_status_getter())
                    return
                if path == "/camera/frame":
                    owner._serve_camera_frame(self, parsed.query)
                    return
                if path == "/camera/stream":
                    owner._serve_camera_frame(self, parsed.query)
                    return
                if path == "/camera/realsense_list":
                    owner._json_ok(self, {"devices": []})
                    return
                if path == "/recording/status":
                    owner._json_ok(self, owner._recording_status_with_validation())
                    return
                if path == "/recording/episodes":
                    params = parse_qs(parsed.query)
                    limit_text = owner._first_query_value(params, "limit")
                    if limit_text and not limit_text.isdigit():
                        owner._json_ok(self, {"ok": False, "error": "episode limit must be a non-negative integer"}, status=400)
                        return
                    limit = int(limit_text) if limit_text else None
                    owner._json_ok(
                        self,
                        list_episodes(
                            owner._first_query_value(params, "root_dir"),
                            owner._default_episode_root(),
                            limit=limit,
                        ),
                    )
                    return
                if path == "/playback/load":
                    params = parse_qs(parsed.query)
                    root_dir = owner._first_query_value(params, "root_dir") or owner._default_episode_root()
                    episode = owner._first_query_value(params, "episode")
                    if not episode:
                        owner._json_ok(self, {"ok": False, "error": "missing playback episode"}, status=400)
                        return
                    owner._json_ok(self, owner.playback.load(root_dir, episode))
                    return
                if path == "/playback/status":
                    owner._json_ok(self, owner.playback.status())
                    return
                if path == "/playback/start":
                    owner._json_ok(self, owner.playback.start())
                    return
                if path == "/playback/pause":
                    owner._json_ok(self, owner.playback.pause())
                    return
                if path == "/playback/stop":
                    owner._json_ok(self, owner.playback.stop())
                    return
                if path == "/playback/seek":
                    params = parse_qs(parsed.query)
                    frame_text = owner._first_query_value(params, "frame") or "0"
                    frame = int(frame_text) if frame_text.lstrip("-").isdigit() else 0
                    owner._json_ok(self, owner.playback.seek(frame))
                    return
                if path == "/playback/curves":
                    params = parse_qs(parsed.query)
                    max_points_text = owner._first_query_value(params, "max_points") or "900"
                    max_points = int(max_points_text) if max_points_text.isdigit() else 900
                    owner._json_ok(self, owner.playback.curves(max_points=max_points))
                    return
                if path == "/playback/image":
                    owner._serve_playback_image(self, parsed.query)
                    return
                if path == "/ui/provider/hold":
                    hold_rejection = owner._provider_hold_rejection()
                    if hold_rejection:
                        owner._json_ok(self, {"ok": False, "error": hold_rejection}, status=400)
                        return
                    owner.command_bus.submit(UiCommandName.SET_PROVIDER_HOLD, payload={"reason": "tab:playback"})
                    owner._json_ok(self, {"ok": True, "queued": True, "command": UiCommandName.SET_PROVIDER_HOLD.value})
                    return
                if path == "/ui/provider/xr":
                    owner.command_bus.submit(UiCommandName.SET_PROVIDER_XR, payload={"reason": "tab:record"})
                    owner._json_ok(self, {"ok": True, "queued": True, "command": UiCommandName.SET_PROVIDER_XR.value})
                    return
                if path == "/replay/real/status":
                    owner._json_ok(self, owner._real_replay_status())
                    return
                if path == "/replay/real/start":
                    owner._handle_real_replay_start(self, parsed.query)
                    return
                if path == "/replay/real/stop":
                    owner.command_bus.submit(UiCommandName.STOP_RAW_REPLAY, payload={"reason": "ui_stop"})
                    owner._json_ok(self, {"ok": True, "queued": True, "command": UiCommandName.STOP_RAW_REPLAY.value})
                    return
                if path == "/inference/status":
                    owner._json_ok(self, owner._inference_status())
                    return
                if path == "/inference/profiles":
                    owner._json_ok(self, {"ok": True, "profiles": owner._inference_profiles()})
                    return
                if path == "/inference/start":
                    owner._handle_inference_start(self, parsed.query)
                    return
                if path == "/inference/stop":
                    command = owner.command_bus.submit(
                        UiCommandName.STOP_ONLINE_INFERENCE,
                        payload={"reason": "ui_stop"},
                    )
                    owner._json_ok(self, {"ok": True, "queued": True, "command": command.name.value})
                    return
                if path == "/recording/set_root_dir":
                    owner._handle_recording_root_change(self, parsed.query)
                    return
                if path == "/recording/delete_episodes":
                    owner._handle_episode_delete(self, parsed.query)
                    return
                    return
                if path == "/recording/set_fps":
                    owner._json_ok(self, {"ok": True, "fps": 0.0})
                    return
                if path == "/convert/status":
                    owner._json_ok(self, owner.convert_manager.status())
                    return
                if path == "/convert/start":
                    owner._handle_convert_start(self, parsed.query)
                    return
                if owner._unsupported_reference_route(path):
                    owner._json_ok(
                        self,
                        {
                            "ok": False,
                            "error": f"{path} is not implemented in xr_teleoperate UI",
                        },
                        status=501,
                    )
                    return

                command_result = owner._command_for_path(path)
                if command_result is not None:
                    command, rejected_error = command_result
                    if rejected_error:
                        owner._json_ok(self, {"ok": False, "error": rejected_error}, status=400)
                        return
                    owner.command_bus.submit(command)
                    owner._json_ok(self, {"ok": True, "queued": True, "command": str(command.value)})
                    return

                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        return _Handler

    def _serve_camera_frame(self, handler: BaseHTTPRequestHandler, query: str) -> None:
        params = parse_qs(query)
        camera_id = int((params.get("camera_id") or ["0"])[0])
        frame, _ = self.camera_frame_getter(camera_id)
        if frame is None:
            self._json_ok(handler, {"ok": False, "error": f"camera_id={camera_id} has no frame"}, status=404)
            return
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError(f"failed to encode camera_id={camera_id} preview frame as jpeg")
        payload = encoded.tobytes()
        handler.send_response(200)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _serve_playback_image(self, handler: BaseHTTPRequestHandler, query: str) -> None:
        params = parse_qs(query)
        camera_text = self._first_query_value(params, "camera_id") or "0"
        frame_text = self._first_query_value(params, "frame") or "0"
        camera_id = int(camera_text) if camera_text.lstrip("-").isdigit() else 0
        frame = int(frame_text) if frame_text.lstrip("-").isdigit() else 0
        image_path = self.playback.image_path(camera_id, frame)
        payload = image_path.read_bytes()
        ctype, _ = mimetypes.guess_type(str(image_path))
        handler.send_response(200)
        handler.send_header("Content-Type", ctype or "application/octet-stream")
        handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _serve_index(self, handler: BaseHTTPRequestHandler) -> None:
        html_path = self.web_root / "index.html"
        if html_path.exists():
            payload = html_path.read_bytes()
        else:
            payload = _DEFAULT_HTML
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _serve_static(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        rel_path = path.removeprefix("/").strip("/")
        target = (self.web_root / rel_path).resolve()
        web_root = self.web_root.resolve()
        if web_root not in target.parents or not target.is_file():
            handler.send_response(404)
            handler.end_headers()
            return
        payload = target.read_bytes()
        ctype, _ = mimetypes.guess_type(str(target))
        handler.send_response(200)
        handler.send_header("Content-Type", ctype or "application/octet-stream")
        handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _serve_events(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        min_dt = 1.0 / max(float(self.publish_rate_hz or 1.0), 1e-6)
        last_version = -1
        while self._running.is_set():
            version, snapshot = self.state_store.snapshot()
            if version != last_version:
                payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                try:
                    handler.wfile.write(b"data: " + payload + b"\n\n")
                    handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                last_version = version
            time.sleep(min_dt)

    def _command_for_path(self, path: str) -> tuple[UiCommandName, str] | None:
        if path == "/command/start":
            return UiCommandName.START, ""
        if path == "/command/stop":
            return UiCommandName.STOP, ""
        if path == "/command/home":
            return UiCommandName.HOME, ""
        if path == "/command/recenter":
            return UiCommandName.RECENTER, ""
        if path == "/recording/toggle":
            return UiCommandName.RECORD_TOGGLE, ""
        if path == "/recording/cancel":
            return UiCommandName.RECORD_CANCEL, ""
        if path == "/recording/start":
            if not self._recording_enabled():
                return UiCommandName.RECORD_TOGGLE, "recording is disabled"
            if self._recording_validation_pending():
                return UiCommandName.RECORD_TOGGLE, "episode validation is still running; wait for validation.json before starting another recording"
            return UiCommandName.RECORD_TOGGLE, "recording already active" if self._recording_active() else ""
        if path == "/recording/stop":
            return UiCommandName.RECORD_TOGGLE, "" if self._recording_active() else "recording is not active"
        return None

    def _recording_validation_pending(self) -> bool:
        _, snapshot = self.state_store.snapshot()
        recording = snapshot.get("recording", {})
        return bool(recording.get("validation_pending", False)) if isinstance(recording, dict) else False

    def _handle_real_replay_start(self, handler: BaseHTTPRequestHandler, query: str) -> None:
        params = parse_qs(query)
        rejection = self._real_replay_start_rejection()
        if rejection:
            self._json_ok(handler, {"ok": False, "error": rejection}, status=400)
            return

        root_dir = self._first_query_value(params, "root_dir") or self._default_episode_root()
        episode_name = self._first_query_value(params, "episode")
        episode_index, episode_error = self._episode_index_from_name(episode_name)
        if episode_error:
            self._json_ok(handler, {"ok": False, "error": episode_error}, status=400)
            return

        arm_source = self._first_query_value(params, "arm_source") or "action"
        if arm_source not in {"action", "state", "fk_cmd_pose"}:
            self._json_ok(handler, {"ok": False, "error": f"unsupported arm_source: {arm_source}"}, status=400)
            return

        base_source = self._first_query_value(params, "base_source")
        if not base_source:
            self._json_ok(
                handler,
                {"ok": False, "error": "base_source is required; expected one of: none, action"},
                status=400,
            )
            return
        if base_source not in {"none", "action"}:
            self._json_ok(
                handler,
                {"ok": False, "error": f"unsupported base_source: {base_source}; expected one of: none, action"},
                status=400,
            )
            return

        speed_scale, speed_error = self._positive_float_query(params, "speed_scale", 1.0)
        if speed_error:
            self._json_ok(handler, {"ok": False, "error": speed_error}, status=400)
            return

        command = self.command_bus.submit(
            UiCommandName.START_RAW_REPLAY,
            payload={
                "dataset_root": str(root_dir),
                "episode_index": int(episode_index),
                "episode_name": str(episode_name),
                "arm_source": str(arm_source),
                "base_source": str(base_source),
                "speed_scale": float(speed_scale),
            },
        )
        self._json_ok(handler, {"ok": True, "queued": True, "command": command.name.value})

    def _recording_mutation_rejection(self) -> str:
        if self._recording_active():
            return "recording is active; stop/save or cancel before changing episode files"
        if self._recording_validation_pending():
            return "episode validation is pending; wait for validation.json before changing episode files"
        _, snapshot = self.state_store.snapshot()
        provider = snapshot.get("provider", {})
        if isinstance(provider, dict) and provider.get("active_provider") == "raw_replay":
            return "raw replay is active; stop real replay before changing episode files"
        return ""

    def _handle_recording_root_change(self, handler: BaseHTTPRequestHandler, query: str) -> None:
        if not self._recording_enabled():
            self._json_ok(handler, {"ok": False, "error": "recording is disabled; restart with --record before changing the recording root"}, status=400)
            return
        rejection = self._recording_mutation_rejection()
        if rejection:
            self._json_ok(handler, {"ok": False, "error": rejection}, status=400)
            return
        params = parse_qs(query)
        root_dir = str(self._first_query_value(params, "root_dir") or "").strip()
        if not root_dir:
            self._json_ok(handler, {"ok": False, "error": "root_dir is required"}, status=400)
            return
        command = self.command_bus.submit(
            UiCommandName.SET_RECORD_ROOT,
            payload={"root_dir": root_dir},
        )
        self._json_ok(
            handler,
            {"ok": True, "queued": True, "command": command.name.value, "root_dir": root_dir},
        )

    def _handle_episode_delete(self, handler: BaseHTTPRequestHandler, query: str) -> None:
        rejection = self._recording_mutation_rejection()
        if rejection:
            self._json_ok(handler, {"ok": False, "error": rejection}, status=400)
            return
        params = parse_qs(query)
        root = Path(self._first_query_value(params, "root_dir") or self._default_episode_root()).expanduser().resolve()
        if not root.is_dir():
            self._json_ok(handler, {"ok": False, "error": f"episode root does not exist: {root}"}, status=400)
            return
        names = list(dict.fromkeys(str(name).strip() for name in params.get("episode", []) if str(name).strip()))
        if not names:
            self._json_ok(handler, {"ok": False, "error": "at least one episode is required"}, status=400)
            return
        if any(EPISODE_NAME_RE.fullmatch(name) is None for name in names):
            self._json_ok(handler, {"ok": False, "error": "episode names must use episode_NNNN format"}, status=400)
            return
        playback_dir = self.playback.episode_dir.resolve() if self.playback.episode_dir is not None else None
        targets = [(root / name).resolve() for name in names]
        if any(target.parent != root or not target.is_dir() for target in targets):
            self._json_ok(handler, {"ok": False, "error": "one or more requested episode directories do not exist"}, status=404)
            return
        if playback_dir is not None and playback_dir in targets:
            if self.playback.state == "playing":
                self._json_ok(handler, {"ok": False, "error": "stop the selected playback episode before deleting it"}, status=400)
                return
            self.playback.unload()
        for target in targets:
            shutil.rmtree(target)
        self._json_ok(handler, {"ok": True, "deleted": names, "root_dir": str(root)})

    def _handle_inference_start(self, handler: BaseHTTPRequestHandler, query: str) -> None:
        rejection = self._inference_start_rejection()
        if rejection:
            self._json_ok(handler, {"ok": False, "error": rejection}, status=400)
            return
        params = parse_qs(query)
        prompt = str(self._first_query_value(params, "prompt") or "").strip()
        if not prompt:
            self._json_ok(handler, {"ok": False, "error": "inference prompt is required"}, status=400)
            return
        protocol_profile = str(self._first_query_value(params, "protocol_profile") or "").strip()
        if not protocol_profile:
            self._json_ok(handler, {"ok": False, "error": "inference protocol_profile is required"}, status=400)
            return
        profiles = {str(profile["id"]): profile for profile in self._inference_profiles()}
        profile = profiles.get(protocol_profile)
        if profile is None:
            self._json_ok(handler, {"ok": False, "error": f"unknown inference protocol_profile: {protocol_profile}"}, status=400)
            return
        if not bool(profile["available"]):
            self._json_ok(
                handler,
                {"ok": False, "error": str(profile["reason"])},
                status=400,
            )
            return
        command = self.command_bus.submit(
            UiCommandName.START_ONLINE_INFERENCE,
            payload={"prompt": prompt, "protocol_profile": protocol_profile},
        )
        self._json_ok(handler, {"ok": True, "queued": True, "command": command.name.value})

    def _handle_convert_start(self, handler: BaseHTTPRequestHandler, query: str) -> None:
        params = parse_qs(query)
        export_video, video_error = self._bool_query(params, "export_video", False)
        export_fk, fk_error = self._bool_query(params, "export_fk", False)
        export_verify, verify_error = self._bool_query(params, "export_verify", True)
        bool_error = video_error or fk_error or verify_error
        if bool_error:
            self._json_ok(handler, {"ok": False, "error": bool_error}, status=400)
            return

        fps, fps_error = self._positive_float_query(params, "fps", self._default_fps())
        if fps_error:
            self._json_ok(handler, {"ok": False, "error": fps_error}, status=400)
            return

        source_root_text = self._first_query_value(params, "source_root") or self._default_episode_root()
        output_root_text = self._first_query_value(params, "output_root")
        dataset_name = self._first_query_value(params, "dataset_name")
        task_text = self._first_query_value(params, "default_task") or self._first_query_value(params, "task")
        if not str(output_root_text).strip():
            self._json_ok(handler, {"ok": False, "error": "output_root is required"}, status=400)
            return
        if not str(dataset_name).strip():
            self._json_ok(handler, {"ok": False, "error": "dataset_name is required"}, status=400)
            return
        if not str(task_text).strip():
            self._json_ok(handler, {"ok": False, "error": "default_task is required"}, status=400)
            return

        request = UiExportRequest(
            source_root=Path(source_root_text),
            output_root=Path(output_root_text),
            dataset_name=dataset_name,
            task=task_text,
            fps=fps,
            format_version=self._first_query_value(params, "format_version") or "v2",
            export_mode=self._first_query_value(params, "export_mode") or "new",
            export_video=export_video,
            export_fk=export_fk,
            export_verify=export_verify,
            selected_episodes=tuple(str(value).strip() for value in params.get("episode", []) if str(value).strip()),
            urdf_path=self._first_query_value(params, "urdf_path") or DEFAULT_UI_URDF_PATH,
        )
        result = self.convert_manager.start(request)
        self._json_ok(handler, result, status=200 if result.get("ok") else 400)

    def _real_replay_start_rejection(self) -> str:
        recording_rejection = self._provider_hold_rejection()
        if recording_rejection:
            return recording_rejection
        _, snapshot = self.state_store.snapshot()
        teleop = snapshot.get("teleop", {})
        if isinstance(teleop, dict) and not bool(teleop.get("started", False)):
            return "teleop is not started; start teleop before real replay"
        provider = snapshot.get("provider", {})
        if isinstance(provider, dict) and str(provider.get("active_provider", "")) not in {"hold", "raw_replay"}:
            return "real replay requires active provider hold"
        return ""

    def _inference_start_rejection(self) -> str:
        recording_rejection = self._provider_hold_rejection()
        if recording_rejection:
            return recording_rejection
        _, snapshot = self.state_store.snapshot()
        provider = snapshot.get("provider", {})
        if isinstance(provider, dict):
            active_provider = str(provider.get("active_provider", ""))
            if active_provider == "raw_replay":
                return "online inference cannot start while raw_replay is active"
            if active_provider == "online_inference":
                return "online inference is already active"
            if active_provider not in {"hold", "xr_live"}:
                return f"online inference cannot start from provider={active_provider or 'unknown'}"
        return ""

    def _provider_hold_rejection(self) -> str:
        _, snapshot = self.state_store.snapshot()
        recording = snapshot.get("recording", {})
        if not isinstance(recording, dict):
            return ""
        alignment = recording.get("last_alignment", {})
        armed = bool(alignment.get("waiting_for_first_frame", False)) if isinstance(alignment, dict) else False
        if bool(recording.get("active", False)) or armed or str(recording.get("phase", "")) == "armed":
            return "recording is active or armed; stop/cancel recording before switching provider"
        return ""

    def _real_replay_status(self) -> dict[str, Any]:
        _, snapshot = self.state_store.snapshot()
        provider = snapshot.get("provider", {})
        if not isinstance(provider, dict):
            provider = {"active_provider": "unknown", "real_replay": {"state": "disabled"}}
        return {"ok": True, "provider": provider}

    def _inference_status(self) -> dict[str, Any]:
        _, snapshot = self.state_store.snapshot()
        provider = snapshot.get("provider", {})
        if not isinstance(provider, dict):
            provider = {"active_provider": "unknown"}
        online_inference = provider.get("online_inference", {})
        if not isinstance(online_inference, dict):
            online_inference = {"state": "disabled", "error": "missing online inference status"}
        return {"ok": True, "provider": provider, "online_inference": online_inference}

    def _inference_profiles(self) -> list[dict[str, Any]]:
        profiles = self.inference_profiles_getter()
        if not isinstance(profiles, list):
            raise RuntimeError("inference_profiles_getter must return a list")
        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, profile in enumerate(profiles):
            if not isinstance(profile, dict):
                raise RuntimeError(f"inference profile at index {index} must be an object")
            profile_id = str(profile.get("id", "")).strip()
            label = str(profile.get("label", "")).strip()
            available = profile.get("available")
            reason = str(profile.get("reason", "")).strip()
            if not profile_id or not label or not isinstance(available, bool):
                raise RuntimeError(f"invalid inference profile at index {index}")
            if profile_id in seen_ids:
                raise RuntimeError(f"duplicate inference profile id: {profile_id}")
            if not available and not reason:
                raise RuntimeError(f"unavailable inference profile requires reason: {profile_id}")
            seen_ids.add(profile_id)
            normalized.append(
                {
                    "id": profile_id,
                    "label": label,
                    "available": available,
                    "reason": reason,
                }
            )
        if not normalized:
            raise RuntimeError("inference_profiles_getter returned no profiles")
        return normalized

    @staticmethod
    def _default_inference_profiles() -> list[dict[str, Any]]:
        return [
            {
                "id": "pi05_dual_arm_20d",
                "label": "pi0.5 双臂 20D",
                "available": True,
                "reason": "",
            },
            {
                "id": "mobile_tcp23",
                "label": "移动操作 TCP23",
                "available": False,
                "reason": "mobile_tcp23 runtime is not configured",
            },
            {
                "id": "mobile_joint_base",
                "label": "移动操作 Joint19",
                "available": False,
                "reason": "mobile_joint_base runtime is not configured",
            },
        ]

    @staticmethod
    def _episode_index_from_name(episode_name: str) -> tuple[int, str]:
        name = str(episode_name or "").strip()
        if not name:
            return -1, "missing replay episode"
        if name.isdigit():
            return int(name), ""
        tail = name.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail), ""
        return -1, f"episode name must contain numeric index: {name}"

    def _positive_float_query(self, params: dict[str, list[str]], name: str, default: float) -> tuple[float, str]:
        text = self._first_query_value(params, name)
        if not text:
            return float(default), ""
        value_text = str(text).strip()
        numeric_text = value_text[1:] if value_text.startswith("+") else value_text
        if numeric_text.count(".") > 1 or not numeric_text.replace(".", "", 1).isdigit():
            return float(default), f"{name} must be a positive finite number"
        value = float(value_text)
        if value <= 0.0:
            return float(default), f"{name} must be positive"
        return value, ""

    @staticmethod
    def _unsupported_reference_route(path: str) -> bool:
        return (
            path in {"/camera/start", "/camera/stop"}
        )

    def _recording_active(self) -> bool:
        _, snapshot = self.state_store.snapshot()
        recording = snapshot.get("recording", {})
        return bool(recording.get("active", False)) if isinstance(recording, dict) else False

    def _recording_enabled(self) -> bool:
        _, snapshot = self.state_store.snapshot()
        recording = snapshot.get("recording", {})
        return bool(recording.get("enabled", True)) if isinstance(recording, dict) else True

    def _default_fps(self) -> float:
        _, snapshot = self.state_store.snapshot()
        recording = snapshot.get("recording", {})
        if not isinstance(recording, dict):
            return 30.0
        fps = recording.get("fps", 0.0)
        if isinstance(fps, (int, float)) and float(fps) > 0.0:
            return float(fps)
        return 30.0

    def _bool_query(self, params: dict[str, list[str]], name: str, default: bool) -> tuple[bool, str]:
        text = self._first_query_value(params, name)
        if not text:
            return bool(default), ""
        value = str(text).strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True, ""
        if value in {"0", "false", "no", "n", "off"}:
            return False, ""
        return bool(default), f"{name} must be boolean 0/1/true/false/yes/no/on/off"

    @staticmethod
    def _first_query_value(params: dict[str, list[str]], name: str) -> str:
        values = params.get(name) or []
        return str(values[0]) if values else ""

    def _default_episode_root(self) -> str:
        _, snapshot = self.state_store.snapshot()
        recording = snapshot.get("recording", {})
        if not isinstance(recording, dict):
            return "."
        return str(recording.get("active_root_dir") or recording.get("root_dir") or ".")

    def _recording_status_with_validation(self) -> dict[str, Any]:
        _, snapshot = self.state_store.snapshot()
        recording = dict(snapshot.get("recording", {}))
        root_dir = self._default_episode_root()
        latest = latest_episode_summary(root_dir)
        if latest is not None:
            recording["last_validation"] = dict(latest.get("validation", {}))
        return recording

    @staticmethod
    def _json_ok(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
        payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        handler.send_header("Content-Length", str(len(payload_bytes)))
        handler.end_headers()
        handler.wfile.write(payload_bytes)

    @staticmethod
    def _empty_camera_status() -> dict[str, Any]:
        return {
            "url": "",
            "managed_process_alive": False,
            "managed_pid": None,
            "camera_id": None,
            "camera_name": "",
            "camera_mode": "",
            "shared_seq": -1,
            "shared_timestamp_ns": 0,
            "shared_age_ms": -1,
            "requested_capture": None,
            "actual_capture": None,
            "active_camera_ids": [],
            "streams": [],
            "camera_name_map": {},
            "name_presets": [],
            "last_error": "",
        }

    @staticmethod
    def _empty_camera_frame(camera_id: int) -> tuple[Any, dict[str, Any] | None]:
        return None, None


_DEFAULT_HTML = b"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>xr_teleoperate UI</title></head>
<body><h1>xr_teleoperate UI</h1><p>UI backend is running.</p></body>
</html>
"""
