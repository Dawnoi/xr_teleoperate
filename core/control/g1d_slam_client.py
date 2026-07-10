from __future__ import annotations

import http.client
import json
import math
import sys
import time
from typing import Any, Callable
from urllib.parse import urlparse


class G1DSlamError(RuntimeError):
    def __init__(self, message: str, **context: Any) -> None:
        self.message = str(message)
        self.context = dict(context)
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if not self.context:
            return self.message
        context_text = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        return f"{self.message} ({context_text})"


class G1DSlamHTTPError(G1DSlamError):
    def __init__(self, message: str, *, method: str, path: str, status: int, reason: str, body: str) -> None:
        self.method = str(method)
        self.path = str(path)
        self.status = int(status)
        self.reason = str(reason)
        self.body = str(body)
        super().__init__(
            message,
            method=self.method,
            path=self.path,
            status=self.status,
            reason=self.reason,
            body=self.body,
        )


class G1DSlamActionError(G1DSlamError):
    def __init__(self, message: str, *, action_id: str, state: dict[str, Any], reason: Any = "") -> None:
        self.action_id = str(action_id)
        self.state = dict(state)
        self.reason = str(reason)
        super().__init__(message, action_id=self.action_id, state=self.state, reason=self.reason)


class G1DSlamClient:
    MOVE_BY_FORWARD = 0
    MOVE_BY_BACKWARD = 1
    MOVE_BY_TURN_RIGHT = 2
    MOVE_BY_TURN_LEFT = 3

    ACTIONS_PATH = "/api/core/motion/v1/actions"
    ROBOT_INFO_PATH = "/api/core/system/v1/robot/info"
    CAPABILITIES_PATH = "/api/core/system/v1/capabilities"
    HEALTH_PATH = "/api/core/system/v1/robot/health"
    CURRENT_ACTION_PATH = f"{ACTIONS_PATH}/:current"
    POIS_PATH = "/api/core/artifact/v1/pois"

    def __init__(
        self,
        robot_ip: str | None = None,
        *,
        base_url: str | None = None,
        port: int = 1448,
        timeout_sec: float = 5.0,
        connection_factory: Callable[[str, int, float], Any] | None = None,
        debug_http: bool = False,
        debug_stream: Any | None = None,
    ) -> None:
        if base_url is None:
            if not robot_ip:
                raise ValueError("robot_ip or base_url is required")
            base_url = f"http://{robot_ip}:{int(port)}"
        parsed = urlparse(str(base_url).strip())
        if parsed.scheme not in {"", "http"}:
            raise ValueError("only http:// G1D SLAM API endpoints are supported")
        host = parsed.hostname
        if not host:
            raise ValueError("G1D SLAM base_url host is required")
        self.host = str(host)
        self.port = int(parsed.port or port)
        self.base_path = str(parsed.path or "").rstrip("/")
        self.timeout_sec = self._finite_positive_float(timeout_sec, "timeout_sec")
        self._connection_factory = connection_factory
        self._conn = None
        self.debug_http = bool(debug_http)
        self.debug_stream = debug_stream if debug_stream is not None else sys.stderr
        self.last_debug: dict[str, Any] = {}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def robot_info(self) -> dict[str, Any]:
        return self._get_json(self.ROBOT_INFO_PATH)

    def capabilities(self) -> list[Any]:
        value = self._request_json("GET", self.CAPABILITIES_PATH, require_object=False)
        if not isinstance(value, list):
            raise G1DSlamError(f"capabilities response must be a list, got {type(value).__name__}")
        return value

    def health(self) -> dict[str, Any]:
        return self._get_json(self.HEALTH_PATH)

    def current_action(self) -> dict[str, Any]:
        return self._get_json(self.CURRENT_ACTION_PATH)

    def abort_current_action(self) -> dict[str, Any]:
        return self._request_json("DELETE", self.CURRENT_ACTION_PATH)

    def pois(self) -> list[Any]:
        value = self._request_json("GET", self.POIS_PATH, require_object=False)
        if not isinstance(value, list):
            raise G1DSlamError(f"pois response must be a list, got {type(value).__name__}")
        return value

    def create_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("action payload must be an object")
        if not isinstance(payload.get("action_name"), str) or not payload.get("action_name"):
            raise ValueError("action payload requires non-empty action_name")
        response = self._post_json(self.ACTIONS_PATH, payload)
        if "action_id" not in response:
            raise G1DSlamError(f"action creation response missing action_id: payload={payload!r}, response={response!r}")
        return response

    def create_action_by_name(self, action_name: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        name = str(action_name or "").strip()
        if not name:
            raise ValueError("action_name is required")
        payload: dict[str, Any] = {"action_name": name, "options": dict(options or {})}
        return self.create_action(payload)

    def get_action(self, action_id: str) -> dict[str, Any]:
        action_id_s = self._action_id(action_id)
        return self._get_json(f"{self.ACTIONS_PATH}/{action_id_s}")

    def wait_action(
        self,
        action_id: str,
        *,
        timeout_sec: float = 120.0,
        poll_interval_sec: float = 1.0,
    ) -> dict[str, Any]:
        action_id_s = self._action_id(action_id)
        timeout = self._finite_nonnegative_float(timeout_sec, "timeout_sec")
        poll_interval = self._finite_nonnegative_float(poll_interval_sec, "poll_interval_sec")
        deadline = time.monotonic() + timeout
        last_state = None
        while True:
            response = self.get_action(action_id_s)
            state = response.get("state")
            if not isinstance(state, dict):
                raise G1DSlamError(f"action state response missing object state: action_id={action_id_s!r}, response={response!r}")
            last_state = state
            status = state.get("status")
            if status == 4:
                result = state.get("result")
                if result == 0:
                    return state
                reason = state.get("reason", "")
                raise G1DSlamActionError(
                    "G1D action failed",
                    action_id=action_id_s,
                    state=state,
                    reason=reason,
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"G1D action wait timed out: action_id={action_id_s!r}, "
                    f"timeout_sec={timeout}, last_state={last_state!r}"
                )
            if poll_interval > 0.0:
                time.sleep(poll_interval)

    def move_to(
        self,
        *,
        x: float,
        y: float,
        z: float = 0.0,
        yaw: float | None = None,
        precise: bool = False,
        speed_ratio: float = 0.5,
        wait: bool = False,
    ) -> dict[str, Any]:
        target = self._point(x=x, y=y, z=z)
        move_options = self._move_options(speed_ratio=speed_ratio, precise=precise, yaw=yaw)
        return self._create_action_and_maybe_wait(
            {
                "action_name": "agent.actions.MoveToAction",
                "options": {"target": target, "move_options": move_options},
            },
            wait=wait,
        )

    def series_move_to(
        self,
        *,
        points: list[dict[str, Any]],
        speed_ratio: float = 0.5,
        wait: bool = False,
    ) -> dict[str, Any]:
        targets = self._points(points, "points")
        return self._create_action_and_maybe_wait(
            {
                "action_name": "agent.actions.SeriesMoveToAction",
                "options": {
                    "targets": targets,
                    "move_options": {"mode": 0, "speed_ratio": self._finite_float(speed_ratio, "speed_ratio")},
                },
            },
            wait=wait,
        )

    def follow_path_points(
        self,
        *,
        points: list[dict[str, Any]],
        precise: bool = False,
        speed_ratio: float = 0.5,
        wait: bool = False,
    ) -> dict[str, Any]:
        path_points = self._points(points, "points")
        flags = ["precise"] if precise else []
        return self._create_action_and_maybe_wait(
            {
                "action_name": "agent.actions.FollowPathPointsAction",
                "options": {
                    "path_points": path_points,
                    "move_options": {
                        "mode": 0,
                        "flags": flags,
                        "speed_ratio": self._finite_float(speed_ratio, "speed_ratio"),
                    },
                },
            },
            wait=wait,
        )

    def rotate(self, angle: float, *, wait: bool = False) -> dict[str, Any]:
        return self._create_action_and_maybe_wait(
            {"action_name": "agent.actions.RotateAction", "options": {"angle": self._finite_float(angle, "angle")}},
            wait=wait,
        )

    def rotate_to(self, yaw: float, *, wait: bool = False) -> dict[str, Any]:
        return self._create_action_and_maybe_wait(
            {"action_name": "agent.actions.RotateToAction", "options": {"angle": self._finite_float(yaw, "yaw")}},
            wait=wait,
        )

    def go_home(self, *, dock: bool = True, wait: bool = False) -> dict[str, Any]:
        flags = "dock" if bool(dock) else "no_dock"
        return self._create_action_and_maybe_wait(
            {
                "action_name": "agent.actions.GoHomeAction",
                "options": {
                    "gohome_options": {
                        "flags": flags,
                        "back_to_landing": True,
                        "charging_retry_count": 3,
                        "move_options": {"mode": 0},
                    }
                },
            },
            wait=wait,
        )

    def recover_localization(self, *, max_recover_time_ms: int = 30000, wait: bool = False) -> dict[str, Any]:
        max_recover_time = self._positive_int(max_recover_time_ms, "max_recover_time_ms")
        return self._create_action_and_maybe_wait(
            {
                "action_name": "agent.actions.RecoverLocalizationAction",
                "options": {
                    "relocalization_options": {
                        "max_recover_time": max_recover_time,
                        "recover_movement_type": "RotateOnly",
                    }
                },
            },
            wait=wait,
        )

    def return_to_parking(
        self,
        *,
        wait_for_parking: bool = True,
        max_wait_time_ms: int = 60000,
        wait: bool = False,
    ) -> dict[str, Any]:
        max_wait_time = self._positive_int(max_wait_time_ms, "max_wait_time_ms")
        return self._create_action_and_maybe_wait(
            {
                "action_name": "agent.actions.ReturnToParkingAction",
                "options": {
                    "wait_for_parking": bool(wait_for_parking),
                    "max_wait_time": max_wait_time,
                    "move_options": {"mode": 0},
                },
            },
            wait=wait,
        )

    def move_by(self, direction: int, *, duration_ms: int = 300, wait: bool = False) -> dict[str, Any]:
        """Send a short MoveByAction pulse. This action has no autonomous obstacle avoidance."""

        direction_i = self._strict_int(direction, "direction")
        if direction_i not in {
            self.MOVE_BY_FORWARD,
            self.MOVE_BY_BACKWARD,
            self.MOVE_BY_TURN_RIGHT,
            self.MOVE_BY_TURN_LEFT,
        }:
            raise ValueError("direction must be one of 0(forward), 1(backward), 2(right turn), 3(left turn)")
        duration = self._positive_int(duration_ms, "duration_ms")
        if duration > 500:
            raise ValueError("duration_ms must be <= 500 because MoveByAction must be refreshed within 500 ms")
        return self._create_action_and_maybe_wait(
            {
                "action_name": "agent.actions.MoveByAction",
                "options": {"direction": direction_i, "duration": duration},
            },
            wait=wait,
        )

    def _create_action_and_maybe_wait(self, payload: dict[str, Any], *, wait: bool) -> dict[str, Any]:
        response = self.create_action(payload)
        if wait:
            return self.wait_action(str(response["action_id"]))
        return response

    def _connect(self):
        if self._conn is None:
            if self._connection_factory is not None:
                self._conn = self._connection_factory(self.host, self.port, self.timeout_sec)
            else:
                self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_sec)
        return self._conn

    def _get_json(self, path: str) -> dict[str, Any]:
        return self._request_json("GET", path)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("POST", path, payload)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        require_object: bool = True,
    ) -> Any:
        method_s = str(method).upper()
        req_path = self._path(path)
        body = None
        headers = {"Accept": "application/json", "Connection": "keep-alive"}
        if payload is not None:
            if not isinstance(payload, dict):
                raise TypeError("payload must be an object")
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
            headers["Content-Length"] = str(len(body))
        self.last_debug = {
            "method": method_s,
            "path": req_path,
            "request_bytes": len(body) if body is not None else 0,
            "reused_connection": self._conn is not None,
        }
        conn = self._connect()
        self._debug_http(f"-> {method_s} {req_path}")
        if body is not None:
            self._debug_http(f"-> body {body.decode('utf-8', errors='replace')}")
        conn.request(method_s, req_path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        status = int(response.status)
        reason = str(response.reason or "")
        text = raw.decode("utf-8", errors="replace") if raw else "{}"
        if not text.strip():
            text = "{}"
        if status < 200 or status >= 300:
            self._debug_http(f"<- {status} {reason}")
            self._debug_http(f"<- body {text}")
            self.last_debug.update({"ok": False, "status": status, "error": text or reason})
            raise G1DSlamHTTPError(
                "G1D HTTP request failed",
                method=method_s,
                path=req_path,
                status=status,
                reason=reason,
                body=text,
            )
        decoded = json.loads(text)
        self._debug_http(f"<- {status} {reason}")
        self._debug_http(f"<- body {json.dumps(decoded, ensure_ascii=False, separators=(',', ':'), allow_nan=False)}")
        if require_object and not isinstance(decoded, dict):
            self.last_debug.update({"ok": False, "status": status, "error": "non-object JSON response"})
            raise G1DSlamError(f"HTTP {method_s} {req_path} returned non-object JSON: {type(decoded).__name__}")
        response_keys = sorted(str(key) for key in decoded) if isinstance(decoded, dict) else []
        self.last_debug.update({"ok": True, "status": status, "response_keys": response_keys})
        return decoded

    def _debug_http(self, message: str) -> None:
        if self.debug_http:
            print(f"[G1D_HTTP] {message}", file=self.debug_stream, flush=True)

    def _path(self, path: str) -> str:
        path_s = str(path or "").strip()
        if not path_s:
            raise ValueError("HTTP path is required")
        if not path_s.startswith("/"):
            path_s = f"/{path_s}"
        return f"{self.base_path}{path_s}" if self.base_path else path_s

    @staticmethod
    def _action_id(action_id: str) -> str:
        value = str(action_id or "").strip()
        if not value:
            raise ValueError("action_id is required")
        return value

    @classmethod
    def _point(cls, *, x: Any, y: Any, z: Any = 0.0) -> dict[str, float]:
        return {
            "x": cls._finite_float(x, "x"),
            "y": cls._finite_float(y, "y"),
            "z": cls._finite_float(z, "z"),
        }

    @classmethod
    def _points(cls, points: list[dict[str, Any]], label: str) -> list[dict[str, float]]:
        if not isinstance(points, (list, tuple)) or not points:
            raise ValueError(f"{label} must be a non-empty sequence")
        out = []
        for index, item in enumerate(points):
            if not isinstance(item, dict):
                raise ValueError(f"{label}[{index}] must be an object")
            out.append(cls._point(x=item.get("x"), y=item.get("y"), z=item.get("z", 0.0)))
        return out

    @classmethod
    def _move_options(
        cls,
        *,
        speed_ratio: float,
        precise: bool,
        yaw: float | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"mode": 0, "speed_ratio": cls._finite_float(speed_ratio, "speed_ratio")}
        if precise or yaw is not None:
            if yaw is None:
                raise ValueError("yaw is required when precise=True")
            options.update(
                {
                    "flags": ["precise", "with_yaw"],
                    "yaw": cls._finite_float(yaw, "yaw"),
                    "acceptable_precision": 100,
                }
            )
        return options

    @staticmethod
    def _finite_float(value: Any, label: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        return number

    @classmethod
    def _finite_positive_float(cls, value: Any, label: str) -> float:
        number = cls._finite_float(value, label)
        if number <= 0.0:
            raise ValueError(f"{label} must be positive")
        return number

    @classmethod
    def _finite_nonnegative_float(cls, value: Any, label: str) -> float:
        number = cls._finite_float(value, label)
        if number < 0.0:
            raise ValueError(f"{label} must be non-negative")
        return number

    @staticmethod
    def _positive_int(value: Any, label: str) -> int:
        number = G1DSlamClient._strict_int(value, label)
        if number <= 0:
            raise ValueError(f"{label} must be positive")
        return number

    @staticmethod
    def _strict_int(value: Any, label: str) -> int:
        if type(value) is not int:
            raise TypeError(f"{label} must be int, got {type(value).__name__}")
        return value


__all__ = [
    "G1DSlamActionError",
    "G1DSlamClient",
    "G1DSlamError",
    "G1DSlamHTTPError",
]
