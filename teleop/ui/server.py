from __future__ import annotations

import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import cv2

from teleop.ui.command_bus import UiCommandBus, UiCommandName
from teleop.ui.episode_store import PlaybackSession, latest_episode_summary, list_episodes
from teleop.ui.state_store import UiStateStore


class TeleopUiServer:
    """Small web server matching the reference data_collector route shape."""

    def __init__(
        self,
        *,
        command_bus: UiCommandBus,
        state_store: UiStateStore,
        camera_status_getter: Callable[[], dict[str, Any]] | None = None,
        camera_frame_getter: Callable[[int], tuple[Any, dict[str, Any] | None]] | None = None,
        web_root: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8085,
        publish_rate_hz: float = 5.0,
    ) -> None:
        self.command_bus = command_bus
        self.state_store = state_store
        self.camera_status_getter = camera_status_getter if camera_status_getter is not None else self._empty_camera_status
        self.camera_frame_getter = camera_frame_getter if camera_frame_getter is not None else self._empty_camera_frame
        self.web_root = Path(web_root) if web_root is not None else Path(__file__).resolve().parent / "web"
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
                    limit_text = owner._first_query_value(params, "limit") or "80"
                    limit = int(limit_text) if limit_text.isdigit() else 80
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
                if path == "/recording/set_root_dir":
                    owner._json_ok(self, {"ok": True, "root_dir": ""})
                    return
                if path == "/recording/set_fps":
                    owner._json_ok(self, {"ok": True, "fps": 0.0})
                    return
                if path == "/convert/status":
                    owner._json_ok(self, {"state": "disabled", "error": "convert is not implemented in xr_teleoperate UI"})
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
            return UiCommandName.RECORD_TOGGLE, "recording already active" if self._recording_active() else ""
        if path == "/recording/stop":
            return UiCommandName.RECORD_TOGGLE, "" if self._recording_active() else "recording is not active"
        return None

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
                "speed_scale": float(speed_scale),
            },
        )
        self._json_ok(handler, {"ok": True, "queued": True, "command": command.name.value})

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
            path in {"/camera/start", "/camera/stop", "/convert/start"}
            or path in {"/recording/delete_episodes"}
        )

    def _recording_active(self) -> bool:
        _, snapshot = self.state_store.snapshot()
        recording = snapshot.get("recording", {})
        return bool(recording.get("active", False)) if isinstance(recording, dict) else False

    def _recording_enabled(self) -> bool:
        _, snapshot = self.state_store.snapshot()
        recording = snapshot.get("recording", {})
        return bool(recording.get("enabled", True)) if isinstance(recording, dict) else True

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
