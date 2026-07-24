from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from robot_ui_platform.backend import UiBackend
from robot_ui_platform.command_store import UiCommandStore
from robot_ui_platform.contracts import CommandOutcome, UiIntent, UiSnapshot
from robot_ui_platform.state_store import UiStateStore


class UiHost:
    """Shared HTTP/SSE control-plane host.

    This host intentionally never invokes ``backend.consume_intent`` from an
    HTTP thread. The robot runtime drains and consumes intents on its existing
    control thread.
    """

    def __init__(
        self,
        *,
        backend: UiBackend,
        initial_snapshot: UiSnapshot,
        host: str = "127.0.0.1",
        port: int = 0,
        publish_rate_hz: float = 5.0,
        web_root: str | Path | None = None,
    ) -> None:
        if float(publish_rate_hz) <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        self.backend = backend
        self.command_store = UiCommandStore()
        self.state_store = UiStateStore(initial_snapshot)
        self.host = str(host)
        self.port = int(port)
        self.publish_rate_hz = float(publish_rate_hz)
        self.web_root = Path(web_root) if web_root is not None else Path(__file__).resolve().parent / "web"
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("UI host is already running")
        self._running.set()
        self._server = ThreadingHTTPServer((self.host, self.port), self._handler_type())
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def drain_intents(self, *, max_commands: int = 64) -> list[UiIntent]:
        return self.command_store.drain(max_commands=max_commands)

    def complete_intent(self, outcome: CommandOutcome) -> CommandOutcome:
        return self.command_store.complete(outcome)

    def publish_snapshot(self, snapshot: UiSnapshot) -> UiSnapshot:
        return self.state_store.publish(snapshot)

    def _handler_type(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    raise ValueError("intent request must contain a JSON body")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise TypeError("intent request JSON must be an object")
                return payload

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path in {"/", "/index.html"}:
                    owner._serve_index(self)
                    return
                if path.startswith("/assets/"):
                    owner._serve_static(self, path)
                    return
                if path == "/api/v1/capabilities":
                    self._send_json(owner.backend.capabilities().as_dict())
                    return
                if path == "/api/v1/snapshot":
                    self._send_json(owner.state_store.snapshot().as_dict())
                    return
                if path.startswith("/api/v1/commands/"):
                    command_id = path.removeprefix("/api/v1/commands/").strip()
                    outcome = owner.command_store.outcome(command_id)
                    if outcome is None:
                        self._send_json({"error": f"unknown command id: {command_id}"}, status=404)
                        return
                    self._send_json(outcome.as_dict())
                    return
                if path.startswith("/api/v1/previews/"):
                    stream_id = path.removeprefix("/api/v1/previews/").strip()
                    preview = owner.backend.get_preview(stream_id)
                    if preview is None:
                        self._send_json({"error": f"preview stream is unavailable: {stream_id}"}, status=404)
                        return
                    body, content_type = preview
                    if not body:
                        raise RuntimeError(f"backend returned an empty preview payload for stream {stream_id!r}")
                    self.send_response(200)
                    self.send_header("Content-Type", str(content_type))
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/api/v1/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    version = -1
                    interval_sec = 1.0 / owner.publish_rate_hz
                    while owner._running.is_set():
                        snapshot = owner.state_store.wait_for_update(version, interval_sec)
                        if snapshot.version != version:
                            body = json.dumps(snapshot.as_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                            self.wfile.write(b"data: " + body + b"\n\n")
                            self.wfile.flush()
                            version = snapshot.version
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path != "/api/v1/intents":
                    self.send_response(404)
                    self.end_headers()
                    return
                intent = UiIntent.from_dict(self._read_json())
                expected_version = intent.expected_snapshot_version
                actual_version = owner.state_store.snapshot().version
                if expected_version is not None and int(expected_version) != actual_version:
                    self._send_json(
                        {
                            "command_id": intent.id,
                            "status": "rejected",
                            "code": "snapshot_version_conflict",
                            "message": f"expected snapshot version {expected_version}, current version is {actual_version}",
                        },
                        status=409,
                    )
                    return
                outcome = owner.command_store.submit(intent)
                self._send_json(outcome.as_dict(), status=202)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        return Handler

    def _serve_index(self, handler: BaseHTTPRequestHandler) -> None:
        index = self.web_root / "index.html"
        if not index.is_file():
            raise RuntimeError(f"robot UI web entrypoint is missing: {index}")
        body = index.read_bytes()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _serve_static(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        relative_path = path.removeprefix("/").strip("/")
        target = (self.web_root / relative_path).resolve()
        root = self.web_root.resolve()
        if root not in target.parents or not target.is_file():
            handler.send_response(404)
            handler.end_headers()
            return
        body = target.read_bytes()
        content_type, _ = mimetypes.guess_type(str(target))
        handler.send_response(200)
        handler.send_header("Content-Type", content_type or "application/octet-stream")
        handler.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
