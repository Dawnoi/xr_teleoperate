"""Project the existing XR UI state into Nero v1 snapshots."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from teleop.nero_console.contracts import RevisionTracker, build_console_config, build_snapshot


_NERO_AUTHORITY_BY_XR_PROVIDER = {
    "xr_live": "teleop",
    "online_inference": "vla",
    "hold": "hold",
    # Real replay is exposed by the Collector console.  It has no equivalent
    # VLA authority state, so VLA controls are explicitly unavailable here.
    "raw_replay": "disabled",
}
_CAMERA_ID_BY_NAME = {"head": 0, "left_wrist": 1, "right_wrist": 2}
_CAMERA_NAME_BY_ID = {camera_id: name for name, camera_id in _CAMERA_ID_BY_NAME.items()}
_CAMERA_STALE_AFTER_MS = 1_500.0
_XR_TELEOPERATION_EXTENSION_ID = "xr.teleoperation"


def _head_reference_mode(args: Any) -> str:
    """Return the canonical XR head-reference mode exposed to the extension."""
    configured = str(getattr(args, "head_reference_mode", "")).strip()
    if configured == "calibrated":
        return "head_coupled"
    if configured in {"live", "head_decoupled_live"}:
        return "live_head_reference"
    return configured


def _xr_teleoperation_extension_state(
    *,
    args: Any,
    provider: dict[str, Any],
    teleop: dict[str, Any],
    inference_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project only browser-visible XR-specific state for the extension."""
    head_reference_mode = _head_reference_mode(args)
    return {
        "authority": str(provider.get("active_provider", "unknown")),
        "head_reference": {
            "mode": head_reference_mode,
            "recenter_supported": head_reference_mode in {"head_coupled", "hybrid"},
        },
        "teleop": {
            "started": bool(teleop.get("started")),
            "ready": bool(teleop.get("ready")),
            "stopping": bool(teleop.get("stopping")),
        },
        "inference": {
            "state": str((provider.get("online_inference") or {}).get("state", "idle")),
            "prompt": str((provider.get("online_inference") or {}).get("prompt", "")),
            "protocol_profile": str((provider.get("online_inference") or {}).get("protocol_profile", "")),
            "profiles": [dict(profile) for profile in inference_profiles],
        },
    }


def _nero_authority(provider: dict[str, Any]) -> tuple[dict[str, str], str]:
    """Translate XR input-provider ownership into the Nero VLA vocabulary."""
    active_provider = str(provider.get("active_provider", ""))
    authority_state = _NERO_AUTHORITY_BY_XR_PROVIDER.get(active_provider)
    if authority_state is None:
        return (
            {
                "state": "unknown",
                "reason": f"unsupported_xr_provider:{active_provider or 'missing'}",
            },
            f"unsupported XR active_provider for Nero VLA authority: {active_provider or 'missing'}",
        )
    return (
        {
            "state": authority_state,
            "reason": str(provider.get("last_reason", "")),
        },
        "",
    )


def _camera_stream_health(camera_status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return explicit per-camera freshness states from XR host-side metadata."""
    streams = camera_status.get("streams", [])
    if not isinstance(streams, list):
        raise TypeError("XR camera status streams must be a list")
    stream_by_name: dict[str, dict[str, Any]] = {}
    for stream in streams:
        if not isinstance(stream, dict):
            raise TypeError("XR camera status stream must be an object")
        name = str(stream.get("camera_name", "")).strip()
        if name in _CAMERA_ID_BY_NAME:
            stream_by_name[name] = stream

    result: dict[str, dict[str, Any]] = {}
    for name in _CAMERA_ID_BY_NAME:
        stream = stream_by_name.get(name)
        if stream is None:
            result[name] = {"state": "missing", "age_ms": None, "message": "no XR frame metadata"}
            continue
        age_ms = stream.get("shared_age_ms")
        if isinstance(age_ms, bool) or not isinstance(age_ms, (int, float)) or float(age_ms) < 0.0:
            result[name] = {
                "state": "error",
                "age_ms": None,
                "message": "shared_age_ms is missing or invalid",
            }
            continue
        normalized_age_ms = float(age_ms)
        if normalized_age_ms > _CAMERA_STALE_AFTER_MS:
            result[name] = {
                "state": "stale",
                "age_ms": normalized_age_ms,
                "message": f"last XR frame is older than {_CAMERA_STALE_AFTER_MS:.0f} ms",
            }
            continue
        result[name] = {"state": "running", "age_ms": normalized_age_ms, "message": ""}
    return result


class NeroProjectionCache:
    """Thread-safe cache populated by the XR control thread.

    The ROS executor must only read this cache.  In particular, it must never
    access camera devices, encode images, or scan an episode directory.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {}
        self._camera_status: dict[str, Any] = {"streams": []}
        self._episodes: dict[str, Any] = {"ok": True, "root_dir": "", "episodes": []}
        self._offline_replay: dict[str, Any] = {"enabled": False, "state": "disabled"}
        self._online_replay: dict[str, Any] = {"state": "disabled", "loaded": False}
        self._export: dict[str, Any] = {"ok": False, "state": "disabled"}
        self._inference_profiles: list[dict[str, Any]] = []
        self._jpeg_by_camera_id: dict[int, bytes] = {}
        self._updated_monotonic_ns: dict[str, int] = {}

    def update_state(self, state: dict[str, Any]) -> None:
        self._update_object("state", state)

    def update_camera_status(self, status: dict[str, Any]) -> None:
        self._update_object("camera_status", status)

    def update_episodes(self, episodes: dict[str, Any]) -> None:
        self._update_object("episodes", episodes)

    def update_offline_replay(self, status: dict[str, Any]) -> None:
        self._update_object("offline_replay", status)

    def update_online_replay(self, status: dict[str, Any]) -> None:
        self._update_object("online_replay", status)

    def update_export(self, status: dict[str, Any]) -> None:
        self._update_object("export", status)

    def update_inference_profiles(self, profiles: list[dict[str, Any]]) -> None:
        if not isinstance(profiles, list):
            raise TypeError("XR inference profiles must be a list")
        normalized: list[dict[str, Any]] = []
        for profile in profiles:
            if not isinstance(profile, dict):
                raise TypeError("XR inference profile must be an object")
            profile_id = str(profile.get("id", "")).strip()
            label = str(profile.get("label", "")).strip()
            available = profile.get("available")
            reason = str(profile.get("reason", ""))
            if not profile_id or not label or not isinstance(available, bool):
                raise ValueError("XR inference profile requires id, label, and boolean available")
            normalized.append(
                {
                    "id": profile_id,
                    "label": label,
                    "available": available,
                    "reason": reason,
                }
            )
        with self._lock:
            self._inference_profiles = normalized
            self._updated_monotonic_ns["inference_profiles"] = time.monotonic_ns()

    def update_jpeg(self, camera_id: int, jpeg: bytes) -> None:
        if isinstance(camera_id, bool) or not isinstance(camera_id, int) or camera_id < 0:
            raise ValueError("XR camera id must be a non-negative integer")
        if not isinstance(jpeg, bytes):
            raise TypeError("XR JPEG cache entry must be bytes")
        with self._lock:
            self._jpeg_by_camera_id[camera_id] = bytes(jpeg)
            self._updated_monotonic_ns[f"jpeg:{camera_id}"] = time.monotonic_ns()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "camera_status": self._camera_status,
                "episodes": self._episodes,
                "offline_replay": self._offline_replay,
                "online_replay": self._online_replay,
                "export": self._export,
                "inference_profiles": [dict(profile) for profile in self._inference_profiles],
                "updated_monotonic_ns": dict(self._updated_monotonic_ns),
            }

    def jpeg(self, camera_id: int) -> bytes | None:
        with self._lock:
            jpeg = self._jpeg_by_camera_id.get(camera_id)
            return None if jpeg is None else bytes(jpeg)

    def _update_object(self, name: str, value: dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise TypeError(f"XR {name} cache value must be a dict")
        with self._lock:
            # Control code transfers ownership of a freshly built status object.
            # The provider only reads it while serializing a Nero message.
            setattr(self, f"_{name}", value)
            self._updated_monotonic_ns[name] = time.monotonic_ns()


class XrConsoleProjection:
    """Read-only projection; all mutation remains in the XR control loop."""

    def __init__(
        self,
        *,
        state_store: Any,
        args: Any,
        camera_status_getter: Callable[[], dict[str, Any]],
        camera_frame_getter: Callable[[int], tuple[Any, dict[str, Any] | None]],
        offline_replay_status_getter: Callable[[], dict[str, Any]] | None = None,
        online_replay_status_getter: Callable[[], dict[str, Any]] | None = None,
        export_status_getter: Callable[[], dict[str, Any]] | None = None,
        inference_profiles_getter: Callable[[], list[dict[str, Any]]] | None = None,
        cache: NeroProjectionCache | None = None,
    ) -> None:
        self._state_store = state_store
        self._args = args
        # These are retained solely for the control-thread refresh hook.  The
        # provider timer must not invoke them.
        self._camera_status_getter = camera_status_getter
        self._camera_frame_getter = camera_frame_getter
        self._offline_replay_status_getter = offline_replay_status_getter
        self._online_replay_status_getter = online_replay_status_getter
        self._export_status_getter = export_status_getter
        self._inference_profiles_getter = inference_profiles_getter
        self._cache = cache if cache is not None else NeroProjectionCache()
        self._worker_health_sources: dict[str, Callable[[], dict[str, Any]]] = {}
        self._revisions = {
            "collector": RevisionTracker("xr_collector"),
            "vla": RevisionTracker("xr_vla"),
        }
        self._next_auxiliary_refresh_monotonic_ns = 0

    def _cache_missing_entries(self, cache_snapshot: dict[str, Any]) -> list[str]:
        required = ("state", "camera_status")
        updated = cache_snapshot["updated_monotonic_ns"]
        return [name for name in required if name not in updated]

    @property
    def cache(self) -> NeroProjectionCache:
        """Return the explicit cache that the XR control thread must refresh."""
        return self._cache

    def register_worker_health_source(
        self,
        name: str,
        source: Callable[[], dict[str, Any]],
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Nero cache worker name must be a non-empty string")
        self._worker_health_sources[name] = source

    def _worker_health(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, source in self._worker_health_sources.items():
            status = source()
            if not isinstance(status, dict):
                raise TypeError(f"Nero cache worker health source {name} must return a dict")
            result[name] = status
        return result

    def refresh_control_thread_cache(self, state: dict[str, Any]) -> None:
        """Refresh non-image runtime data from the XR control thread only.

        This method intentionally does not obtain or encode camera frames.  A
        caller must encode a bounded, latest-only preview itself and call
        :meth:`NeroProjectionCache.update_jpeg`.
        """
        if not isinstance(state, dict):
            raise TypeError("XR UI state store must contain a dict")
        self._cache.update_state(state)
        now_ns = time.monotonic_ns()
        if now_ns < self._next_auxiliary_refresh_monotonic_ns:
            return
        self._next_auxiliary_refresh_monotonic_ns = now_ns + 200_000_000
        camera_status = self._camera_status_getter()
        if not isinstance(camera_status, dict):
            raise TypeError("XR camera status getter must return a dict")
        self._cache.update_camera_status(camera_status)
        if self._offline_replay_status_getter is not None:
            offline_replay = self._offline_replay_status_getter()
            self._cache.update_offline_replay(offline_replay)
        if self._online_replay_status_getter is not None:
            online_replay = self._online_replay_status_getter()
            self._cache.update_online_replay(online_replay)
        if self._export_status_getter is not None:
            export = self._export_status_getter()
            self._cache.update_export(export)
        if self._inference_profiles_getter is not None:
            self._cache.update_inference_profiles(self._inference_profiles_getter())

    def _streams(self, camera_status: dict[str, Any]) -> list[dict[str, Any]]:
        streams = camera_status.get("streams", [])
        if not isinstance(streams, list):
            raise TypeError("XR camera status streams must be a list")
        stream_by_name: dict[str, dict[str, Any]] = {}
        for stream in streams:
            if not isinstance(stream, dict):
                raise TypeError("XR camera status stream must be an object")
            name = str(stream.get("camera_name", "")).strip()
            if name in _CAMERA_ID_BY_NAME:
                stream_by_name[name] = stream

        configured_endpoints = {
            "head": str(getattr(self._args, "head_zmq_endpoint", "") or "").strip(),
            "left_wrist": str(getattr(self._args, "left_zmq_endpoint", "") or "").strip(),
            "right_wrist": str(getattr(self._args, "right_zmq_endpoint", "") or "").strip(),
        }
        result: list[dict[str, Any]] = []
        for name, camera_id in _CAMERA_ID_BY_NAME.items():
            stream = stream_by_name.get(name)
            endpoint = configured_endpoints[name]
            if stream is None and not endpoint:
                continue
            actual_capture = dict((stream or {}).get("actual_capture") or {})
            transport = str(actual_capture.get("transport", "") or "").strip()
            if not transport and endpoint:
                transport = "zmq_raw"
            source: dict[str, Any] = {"kind": transport, "read_only": True}
            source_endpoint = str(actual_capture.get("endpoint", "") or endpoint).strip()
            if source_endpoint:
                source["endpoint"] = source_endpoint
            protocol = str(actual_capture.get("protocol", "") or "").strip()
            if protocol:
                source["protocol"] = protocol
            camera_stream = {"stream_id": f"camera_{camera_id}", "role": name, "camera_name": name}
            if transport:
                camera_stream["source"] = source
            result.append(camera_stream)
            vla_stream = {"head": "head_fpv_rgb", "left_wrist": "left_hand_rgb", "right_wrist": "right_hand_rgb"}[name]
            vla_stream_config = {"stream_id": vla_stream, "role": name, "camera_name": name}
            if transport:
                vla_stream_config["source"] = source
            result.append(vla_stream_config)
        return result

    def config(self, mode: str) -> dict[str, Any]:
        if mode not in {"collector", "vla"}:
            raise ValueError(f"unsupported Nero projection mode: {mode}")
        capabilities = {
            "collector": [
                "collector.status",
                "collector.recording",
                "collector.episodes",
                "collector.offline_replay",
                "collector.online_replay",
                "collector.export",
                "camera.preview",
            ],
            "vla": [
                "runtime.start_stop",
                "inference.status",
                "diagnostics.latency",
                "diagnostics.trajectory",
                "camera.preview",
            ],
        }[mode]
        cached = self._cache.snapshot()
        config = build_console_config(
            mode=mode,
            instance={"id": f"xr_{mode}", "display_name": f"XR {mode.title()}", "robot_id": "xr_teleoperate"},
            config_revision="v2",
            capabilities=capabilities,
            robot={"arms": [
                {"arm_id": "left", "display_name": "Left Arm", "dof": 7},
                {"arm_id": "right", "display_name": "Right Arm", "dof": 7},
            ]},
            streams=self._streams(cached["camera_status"]),
        )
        # ``extensions`` is an optional v1 field consumed only by the Nero
        # Extension Host.  It never carries a remote URL or executable code.
        config["extensions"] = [{"id": _XR_TELEOPERATION_EXTENSION_ID}]
        return config

    def snapshot(self, mode: str) -> dict[str, Any]:
        cached = self._cache.snapshot()
        worker_health = self._worker_health()
        state = cached["state"]
        recording = dict(state.get("recording") or {})
        provider = dict(state.get("provider") or {})
        teleop = dict(state.get("teleop") or {})
        extension_state = _xr_teleoperation_extension_state(
            args=self._args,
            provider=provider,
            teleop=teleop,
            inference_profiles=cached["inference_profiles"],
        )
        camera = cached["camera_status"]
        camera_health = _camera_stream_health(camera)
        if mode == "collector":
            root = str(recording.get("active_root_dir") or recording.get("root_dir") or "")
            data = {
                "global": {
                    "config": {"record_root": root, "sampling_frequency_hz": float(recording.get("fps", 0.0) or 0.0)},
                    "episodes": cached["episodes"],
                },
                "collect": {
                    "status": recording,
                    "recording": recording,
                    "session": {"active": bool(recording.get("active")), "directory": str(recording.get("session_dir", "")), "frame_index": int(recording.get("frame_index", 0) or 0)},
                    "config": {"collection_mode": "expert", "effector_mode": "gripper"},
                    "feedback": {"arms": [
                        {"arm_id": "left", "q_fb": list(state.get("left", {}).get("q_fb") or []), "stamp_fb_ns": state.get("left", {}).get("stamp_fb_ns")},
                        {"arm_id": "right", "q_fb": list(state.get("right", {}).get("q_fb") or []), "stamp_fb_ns": state.get("right", {}).get("stamp_fb_ns")},
                    ]},
                    "cameras": camera,
                    "camera_devices": camera,
                },
                "offline_replay": cached["offline_replay"],
                "online_replay": cached["online_replay"],
                "export": cached["export"],
                "extensions": {_XR_TELEOPERATION_EXTENSION_ID: extension_state},
            }
            lifecycle_state = "running" if bool(recording.get("active")) else "idle"
            revision_state = {
                "recording": {
                    "active": bool(recording.get("active")),
                    "phase": str(recording.get("phase", "idle")),
                    "error": str(recording.get("error", "")),
                },
                "provider": dict(provider.get("real_replay") or {}),
                "global_config": data["global"]["config"],
                "offline_replay": cached["offline_replay"],
                "online_replay": cached["online_replay"],
                "export": {"state": str(data["export"].get("state", "idle"))},
                "episodes": cached["episodes"],
                "extensions": extension_state,
            }
        elif mode == "vla":
            online = dict(provider.get("online_inference") or {})
            authority, authority_error = _nero_authority(provider)
            protocol_profile = str(online.get("protocol_profile") or getattr(self._args, "online_inference_protocol_profile", ""))
            schema_by_profile = {
                "pi05_dual_arm_20d": {"arm_format": "rot6d", "effector_format": "nero_gripper1", "action_dim": 20},
                "mobile_tcp23": {"arm_format": "rot6d", "effector_format": "nero_gripper1", "action_dim": 20},
                "mobile_pelvis_planar22": {"arm_format": "rot6d", "effector_format": "nero_gripper1", "action_dim": 20},
                "mobile_joint_base": {"arm_format": "joint", "effector_format": "nero_gripper1", "action_dim": 16},
            }
            io_schema = dict(schema_by_profile.get(protocol_profile) or {})
            if io_schema:
                io_schema["wire_format"] = f"{io_schema['arm_format']}+{io_schema['effector_format']}"
            else:
                io_schema = {"arm_format": "unknown", "effector_format": "unknown", "action_dim": 0, "wire_format": "unknown"}
            runtime = {
                "state": "running" if str(provider.get("active_provider", "")) == "online_inference" else "idle",
                "dry_run": bool(getattr(self._args, "online_inference_dry_run", False)),
                "estop": False,
                "phase": str(online.get("state", "idle")),
                "authority": authority,
                "io_schema": io_schema,
                "io_schema_options": {"supported_pairs": []},
            }
            vla_roles = {
                "third_front": ("head", "head_fpv_rgb"),
                "head_fpv": ("head", "head_fpv_rgb"),
                "left_hand": ("left_wrist", "left_hand_rgb"),
                "right_hand": ("right_wrist", "right_hand_rgb"),
            }
            source_streams = {
                str(item.get("camera_name", "")): item
                for item in camera.get("streams", [])
                if isinstance(item, dict)
            }
            mapping = {}
            devices = []
            vla_streams = []
            emitted_device_serials: set[str] = set()
            emitted_vla_stream_ids: set[str] = set()
            for role, (camera_name, stream_id) in vla_roles.items():
                source = source_streams.get(camera_name)
                stream_health = camera_health[camera_name]
                enabled = source is not None
                mapping[role] = {
                    "serial": f"xr_{camera_name}" if enabled else "",
                    "stream": "rgb",
                    "width": int((source or {}).get("actual_capture", {}).get("frame_width", 0) or 0),
                    "height": int((source or {}).get("actual_capture", {}).get("frame_height", 0) or 0),
                    "fps": 0.0,
                    "enabled": enabled,
                    "read_only": True,
                }
                serial = f"xr_{camera_name}"
                if enabled and serial not in emitted_device_serials:
                    devices.append({"serial": serial, "name": camera_name, "role": role, "read_only": True})
                    emitted_device_serials.add(serial)
                if stream_id not in emitted_vla_stream_ids:
                    vla_streams.append(
                        {
                            "stream_id": stream_id,
                            "state": str(stream_health["state"]),
                            "camera_name": camera_name,
                            "role": role,
                            "age_ms": stream_health["age_ms"],
                            "message": stream_health["message"],
                        }
                    )
                    emitted_vla_stream_ids.add(stream_id)
            data = {
                "runtime": runtime,
                "inference": online,
                "robot": {"arms": [
                    {"arm_id": "left", "enabled": True, "q_fb": list(state.get("left", {}).get("q_fb") or [])},
                    {"arm_id": "right", "enabled": True, "q_fb": list(state.get("right", {}).get("q_fb") or [])},
                ]},
                "camera": {"devices": {"devices": devices, "config": {"mapping": mapping}, "read_only": True}, "streams": vla_streams},
                "diagnostics": {
                    "latency": state.get("teleop_latency", {}),
                    "execution": state.get("teleop_timing", {}),
                    "trajectory": dict((online.get("debug") or {}).get("trajectory") or {}),
                },
                "extensions": {_XR_TELEOPERATION_EXTENSION_ID: extension_state},
            }
            lifecycle_state = "running" if bool(teleop.get("started")) else "idle"
            revision_state = {
                "runtime": runtime,
                "inference": {"state": str(online.get("state", "idle")), "error": str(online.get("error", ""))},
                "extensions": extension_state,
            }
        else:
            raise ValueError(f"unsupported Nero projection mode: {mode}")
        revision = self._revisions[mode].revision_for(revision_state)
        expected_streams = set(_CAMERA_ID_BY_NAME)
        missing_streams = sorted(name for name, details in camera_health.items() if details["state"] == "missing")
        stale_streams = sorted(name for name, details in camera_health.items() if details["state"] == "stale")
        invalid_streams = sorted(name for name, details in camera_health.items() if details["state"] == "error")
        available_streams = expected_streams - set(missing_streams)
        runtime_errors = [str(value) for value in (recording.get("error"), provider.get("last_error")) if value]
        if mode == "vla" and authority_error:
            runtime_errors.append(authority_error)
        cache_missing = self._cache_missing_entries(cached)
        if cache_missing:
            runtime_errors.append("projection cache has not been refreshed: " + ", ".join(cache_missing))
        if missing_streams:
            runtime_errors.append("missing camera streams: " + ", ".join(missing_streams))
        if stale_streams:
            runtime_errors.append("stale camera streams: " + ", ".join(stale_streams))
        if invalid_streams:
            runtime_errors.append("invalid camera streams: " + ", ".join(invalid_streams))
        preview_demand = dict(worker_health.get("preview", {}).get("demand_by_camera") or {})
        missing_preview_caches = sorted(
            name
            for name in expected_streams & available_streams
            if camera_health[name]["state"] == "running"
            and int(preview_demand.get(_CAMERA_ID_BY_NAME[name], 0) or 0) > 0
            and f"jpeg:{_CAMERA_ID_BY_NAME[name]}" not in cached["updated_monotonic_ns"]
        )
        if missing_preview_caches:
            runtime_errors.append("projection JPEG cache has not been refreshed: " + ", ".join(missing_preview_caches))
        failed_workers = sorted(
            name for name, status in worker_health.items() if status.get("state") == "failed"
        )
        if failed_workers:
            runtime_errors.append("projection cache worker failed: " + ", ".join(failed_workers))
        health_level = "error" if runtime_errors else "ok"
        health_message = "; ".join(runtime_errors)
        return build_snapshot(
            mode=mode,
            instance_id=f"xr_{mode}",
            state_revision=revision,
            updated_monotonic_ms=int(time.monotonic() * 1000),
            lifecycle={"state": lifecycle_state, "message": health_message},
            health={"level": health_level, "code": "" if health_level == "ok" else "xr_runtime_error", "message": health_message, "camera_streams": camera_health, "missing_streams": missing_streams, "stale_streams": stale_streams, "invalid_streams": invalid_streams, "missing_preview_caches": missing_preview_caches, "cache_workers": worker_health},
            data=data,
        )

    def stream_jpeg(self, stream_id: str) -> bytes | None:
        stream_text = str(stream_id)
        camera_id = {"head": 0, "left_wrist": 1, "right_wrist": 2}.get(stream_text)
        if camera_id is None and stream_text.startswith("camera_"):
            suffix = stream_text.removeprefix("camera_")
            if suffix.isdigit():
                camera_id = int(suffix)
        if camera_id is None:
            camera_id = {"head_fpv_rgb": 0, "left_hand_rgb": 1, "right_hand_rgb": 2}.get(stream_text)
        if camera_id is None:
            raise ValueError(f"unknown XR camera stream: {stream_id}")
        camera_name = _CAMERA_NAME_BY_ID.get(camera_id)
        if camera_name is None:
            raise ValueError(f"unknown XR camera id for stream: {stream_id}")
        camera_health = _camera_stream_health(self._cache.snapshot()["camera_status"])
        if camera_health[camera_name]["state"] != "running":
            return None
        return self._cache.jpeg(camera_id)
