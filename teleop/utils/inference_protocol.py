import base64
import json
import select
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
