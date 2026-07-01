import base64
import http.client
import json
import select
import socket
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import cv2
import numpy as np


@dataclass
class ParsedActionChunk:
    left: Optional[List[np.ndarray]]
    right: Optional[List[np.ndarray]]


class NewlineJsonCodec:
    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> List[Dict[str, Any]]:
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError("chunk must be bytes-like")
        self._buffer.extend(chunk)
        messages: List[Dict[str, Any]] = []
        while True:
            newline_index = self._buffer.find(b"\n")
            if newline_index < 0:
                break
            line = bytes(self._buffer[:newline_index])
            del self._buffer[: newline_index + 1]
            if not line:
                continue
            messages.append(
                json.loads(
                    line.decode("utf-8"),
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"invalid JSON constant: {value}")
                    ),
                )
            )
        return messages

    def clear(self) -> None:
        self._buffer.clear()


def encode_json_line(message: Dict[str, Any]) -> bytes:
    return json.dumps(
        message,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def encode_jpeg_base64(image: np.ndarray, quality: int = 95) -> str:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("failed to encode JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def make_reset_message(reason: Optional[str] = None) -> Dict[str, Any]:
    message: Dict[str, Any] = {"type": "reset"}
    if reason is not None:
        message["reason"] = reason
    return message


class TcpJsonTransport:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        *,
        sock: Optional[socket.socket] = None,
        connect_timeout_sec: float = 2.0,
    ) -> None:
        if sock is None:
            if host is None or port is None:
                raise ValueError("host and port are required when sock is not provided")
            sock = socket.create_connection((str(host), int(port)), timeout=float(connect_timeout_sec))
        sock.setblocking(False)
        self._socket = sock
        self._codec = NewlineJsonCodec()
        self._rx_messages: List[Dict[str, Any]] = []
        self._tx_buffer = bytearray()
        self._connected = True
        self._closed = False

    @classmethod
    def from_socket(cls, sock: socket.socket) -> "TcpJsonTransport":
        return cls(sock=sock)

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        connect_timeout_sec: float = 2.0,
    ) -> "TcpJsonTransport":
        return cls(host=host, port=port, connect_timeout_sec=connect_timeout_sec)

    def is_connected(self) -> bool:
        return bool(self._connected and not self._closed and self._socket is not None)

    def _mark_disconnected(self) -> None:
        self._connected = False

    def _flush_tx_buffer(self) -> None:
        while self._tx_buffer and self.is_connected():
            try:
                _, writable, _ = select.select([], [self._socket], [], 0.0)
            except Exception:
                writable = [self._socket]
            if not writable:
                return
            try:
                sent = self._socket.send(self._tx_buffer)
            except (BlockingIOError, InterruptedError):
                return
            except OSError as exc:
                self._mark_disconnected()
                raise ConnectionError(f"tcp send failed: {exc}") from exc
            if sent <= 0:
                return
            del self._tx_buffer[:sent]

    def send_json(self, payload: Dict[str, Any]) -> None:
        if not self.is_connected():
            raise ConnectionError("transport disconnected")
        self._tx_buffer.extend(encode_json_line(payload))
        self._flush_tx_buffer()

    def recv_json_nonblocking(self):
        if not self.is_connected():
            return None
        if self._tx_buffer:
            self._flush_tx_buffer()
        if self._rx_messages:
            return self._rx_messages.pop(0)

        try:
            readable, _, _ = select.select([self._socket], [], [], 0.0)
        except Exception:
            readable = [self._socket]
        if not readable:
            return None

        try:
            chunk = self._socket.recv(65536)
        except (BlockingIOError, InterruptedError):
            return None
        except OSError as exc:
            self._mark_disconnected()
            raise ConnectionError(f"tcp recv failed: {exc}") from exc

        if not chunk:
            self._mark_disconnected()
            return None

        messages = self._codec.feed(chunk)
        self._rx_messages.extend(messages)
        if self._rx_messages:
            return self._rx_messages.pop(0)
        return None

    def clear_pending_rx(self) -> None:
        self._rx_messages.clear()
        self._codec.clear()
        if not self.is_connected():
            return
        while True:
            try:
                readable, _, _ = select.select([self._socket], [], [], 0.0)
            except Exception:
                readable = [self._socket]
            if not readable:
                return
            try:
                chunk = self._socket.recv(65536)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                self._mark_disconnected()
                return
            if not chunk:
                self._mark_disconnected()
                return

    def reset(self, reason: Optional[str] = None) -> None:
        self.send_json(make_reset_message(reason))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connected = False
        try:
            self._socket.close()
        except Exception:
            pass


class HttpJsonInferenceTransport:
    def __init__(
        self,
        base_url: str,
        *,
        handshake_path: str = "/handshake",
        infer_path: str = "/infer",
        timeout_sec: float = 30.0,
        handshake_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        parsed = urlparse(str(base_url).strip())
        if parsed.scheme not in {"http", ""}:
            raise ValueError("only http:// inference transport is supported")
        if not parsed.hostname:
            raise ValueError("HTTP inference base_url host is required")
        self.host = str(parsed.hostname)
        self.port = int(parsed.port or 80)
        self.base_path = str(parsed.path or "").rstrip("/")
        self.handshake_path = self._normalize_path(handshake_path)
        self.infer_path = self._normalize_path(infer_path)
        self.timeout_sec = float(timeout_sec)
        if self.timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be positive")
        self.handshake_payload = dict(handshake_payload or {"transport": "http"})
        self._conn: http.client.HTTPConnection | None = None
        self._closed = False
        self._connected = True
        self._pending_rx: List[Dict[str, Any]] = []
        self.last_debug: Dict[str, Any] = {}

    @classmethod
    def connect(
        cls,
        *,
        base_url: str,
        handshake_path: str = "/handshake",
        infer_path: str = "/infer",
        timeout_sec: float = 30.0,
        handshake_payload: Optional[Dict[str, Any]] = None,
    ) -> "HttpJsonInferenceTransport":
        return cls(
            base_url=base_url,
            handshake_path=handshake_path,
            infer_path=infer_path,
            timeout_sec=timeout_sec,
            handshake_payload=handshake_payload,
        )

    def is_connected(self) -> bool:
        return bool(self._connected and not self._closed)

    def clear_pending_rx(self) -> None:
        self._pending_rx.clear()

    def reset(self, reason: Optional[str] = None) -> None:
        payload = {"type": "handshake", **self.handshake_payload}
        if reason is not None:
            payload["reason"] = reason
        response = self._post_json(self.handshake_path, payload)
        self.last_debug = {
            "ok": True,
            "path": self.handshake_path,
            "response_keys": sorted(str(key) for key in response.keys()),
        }

    def send_json(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("payload must be an object")
        response = self._post_infer(payload)
        self._pending_rx.append(response)

    def recv_json_nonblocking(self):
        if self._pending_rx:
            return self._pending_rx.pop(0)
        return None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._closed = True
        self._connected = False

    @staticmethod
    def _normalize_path(path: str) -> str:
        path_s = str(path or "").strip()
        if not path_s:
            raise ValueError("HTTP path is required")
        return path_s if path_s.startswith("/") else f"/{path_s}"

    def _path(self, path: str) -> str:
        return f"{self.base_path}{path}" if self.base_path else path

    def _connect(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_sec)
        return self._conn

    def _post_infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        images = payload.get("images")
        if isinstance(images, dict) and any(isinstance(value, (bytes, bytearray)) and value for value in images.values()):
            return self._post_multipart(self.infer_path, payload)
        return self._post_json(self.infer_path, payload)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return self._post_body(path, body, "application/json; charset=utf-8", {"multipart": False})

    def _post_multipart(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        boundary = f"xrteleop-{uuid.uuid4().hex}"
        images = payload.get("images") if isinstance(payload.get("images"), dict) else {}
        attached_images = {
            str(role): bytes(raw)
            for role, raw in images.items()
            if isinstance(raw, (bytes, bytearray)) and raw
        }
        meta = dict(payload)
        meta["images"] = {str(role): "" for role in images.keys()}
        parts: List[bytes] = []

        def add_part(name: str, content: bytes, content_type: str, filename: Optional[str] = None) -> None:
            disposition = f'form-data; name="{name}"'
            if filename:
                disposition += f'; filename="{filename}"'
            header = (
                f"--{boundary}\r\n"
                f"Content-Disposition: {disposition}\r\n"
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
            parts.append(header + content + b"\r\n")

        add_part(
            "meta",
            json.dumps(meta, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"),
            "application/json",
        )
        image_bytes = 0
        for role_str, content in attached_images.items():
            image_bytes += len(content)
            add_part(f"image_{role_str}", content, "image/jpeg", f"{role_str}.jpg")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode("utf-8")
        return self._post_body(
            path,
            body,
            f"multipart/form-data; boundary={boundary}",
            {"multipart": True, "image_bytes": int(image_bytes)},
        )

    def _post_body(
        self,
        path: str,
        body: bytes,
        content_type: str,
        debug_extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        req_path = self._path(path)
        headers = {
            "Accept": "application/json",
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "Connection": "keep-alive",
        }
        self.last_debug = {
            "path": path,
            "request_bytes": len(body),
            "reused_connection": self._conn is not None,
            **debug_extra,
        }
        conn = self._connect()
        conn.request("POST", req_path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()

        status = int(resp.status)
        reason = str(resp.reason or "")
        text = raw.decode("utf-8", errors="replace") if raw else "{}"
        if status < 200 or status >= 300:
            self.last_debug.update({"ok": False, "status": status, "error": text or reason})
            raise RuntimeError(f"HTTP POST {path} failed: status={status}; reason={reason}; detail={text}")
        decoded = json.loads(text)
        if not isinstance(decoded, dict):
            self.last_debug.update({"ok": False, "status": status, "error": "non-object JSON response"})
            raise RuntimeError(f"HTTP POST {path} returned non-object JSON: {type(decoded).__name__}")
        self.last_debug.update({"ok": True, "status": status, "response_keys": sorted(str(key) for key in decoded.keys())})
        return decoded

def parse_action_chunk(payload: Dict[str, Any], arm_side: str) -> ParsedActionChunk:
    if arm_side not in ("left", "right", "both"):
        raise ValueError(f"unsupported arm_side: {arm_side!r}")

    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    if payload.get("type") != "action":
        raise ValueError("action payload must have type='action'")

    has_single = "action" in payload
    has_left = "action_l" in payload
    has_right = "action_r" in payload

    if has_single and (has_left or has_right):
        raise ValueError("action payload must not mix action with action_l/action_r")

    if arm_side == "both":
        if has_single or not has_left or not has_right:
            raise ValueError("arm_side='both' requires both action_l and action_r")
        return ParsedActionChunk(
            left=_parse_action_steps(payload["action_l"]),
            right=_parse_action_steps(payload["action_r"]),
        )

    if arm_side == "left":
        if has_left:
            return ParsedActionChunk(left=_parse_action_steps(payload["action_l"]), right=None)
        if has_single:
            return ParsedActionChunk(left=_parse_action_steps(payload["action"]), right=None)
        raise ValueError("arm_side='left' requires action_l or action")

    if has_right:
        return ParsedActionChunk(left=None, right=_parse_action_steps(payload["action_r"]))
    if has_single:
        return ParsedActionChunk(left=None, right=_parse_action_steps(payload["action"]))
    raise ValueError("arm_side='right' requires action_r or action")


def _parse_action_steps(steps: Any) -> List[np.ndarray]:
    if not isinstance(steps, (list, tuple)) or len(steps) == 0:
        raise ValueError("action chunk must be a non-empty sequence")

    parsed_steps: List[np.ndarray] = []
    for step in steps:
        array = np.asarray(step, dtype=np.float64)
        if array.shape != (8,):
            raise ValueError("action step must have length 8")
        if not np.all(np.isfinite(array)):
            raise ValueError("action step must contain only finite values")

        quat = array[3:7]
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm < 1e-6:
            raise ValueError("quaternion norm too small")
        array = array.copy()
        array[3:7] = quat / quat_norm
        parsed_steps.append(array)
    return parsed_steps
