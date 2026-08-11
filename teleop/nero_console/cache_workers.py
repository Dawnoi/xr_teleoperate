"""Asynchronous producers for the XR Nero projection cache."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import cv2

from teleop.nero_console.projection import NeroProjectionCache
from teleop.ui.episode_store import list_episodes


_CAMERA_ID_BY_STREAM = {
    "head": 0,
    "camera_0": 0,
    "head_fpv_rgb": 0,
    "left_wrist": 1,
    "camera_1": 1,
    "left_hand_rgb": 1,
    "right_wrist": 2,
    "camera_2": 2,
    "right_hand_rgb": 2,
}


class PreviewCache:
    """Encode only demanded physical camera previews on an isolated thread."""

    def __init__(
        self,
        *,
        cache: NeroProjectionCache,
        camera_frame_getter: Callable[[int], tuple[Any, dict[str, Any] | None]],
        max_fps: float = 5.0,
    ) -> None:
        if float(max_fps) <= 0.0:
            raise ValueError("Nero preview cache max_fps must be positive")
        self._cache = cache
        self._camera_frame_getter = camera_frame_getter
        self._interval_sec = 1.0 / float(max_fps)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._demand_by_source: dict[str, tuple[int, int]] = {}
        self._last_encode_monotonic_ns: dict[int, int] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Nero preview cache already started")
        self._thread = threading.Thread(target=self._run, name="nero_preview_cache", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._wake.set()
        self._thread.join()

    def set_stream_demand(self, provider_id: str, stream_id: str, subscriber_count: int) -> None:
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError("Nero preview demand provider_id must be non-empty")
        camera_id = _CAMERA_ID_BY_STREAM.get(str(stream_id))
        if camera_id is None:
            raise ValueError(f"unknown XR preview stream: {stream_id}")
        if isinstance(subscriber_count, bool) or not isinstance(subscriber_count, int) or subscriber_count < 0:
            raise ValueError("Nero preview subscriber count must be a non-negative integer")
        with self._lock:
            self._demand_by_source[f"{provider_id}:{stream_id}"] = (camera_id, subscriber_count)
        self._wake.set()

    def status(self) -> dict[str, Any]:
        thread = self._thread
        alive = thread is not None and thread.is_alive()
        with self._lock:
            demand_by_camera = {
                camera_id: sum(count for item_camera_id, count in self._demand_by_source.values() if item_camera_id == camera_id)
                for camera_id in range(3)
            }
            last_encode = dict(self._last_encode_monotonic_ns)
        if thread is None:
            state = "new"
        elif self._stop.is_set():
            state = "stopped"
        elif alive:
            state = "running"
        else:
            state = "failed"
        return {
            "state": state,
            "demand_by_camera": demand_by_camera,
            "last_encode_monotonic_ns": last_encode,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self._interval_sec)
            self._wake.clear()
            now_ns = time.monotonic_ns()
            for camera_id in self._demanded_camera_ids():
                with self._lock:
                    last_ns = self._last_encode_monotonic_ns.get(camera_id, 0)
                if now_ns - last_ns < int(self._interval_sec * 1_000_000_000):
                    continue
                frame, _ = self._camera_frame_getter(camera_id)
                if frame is None:
                    continue
                encoded_ok, encoded = cv2.imencode(".jpg", frame)
                if not encoded_ok:
                    raise RuntimeError(f"failed to encode XR camera_id={camera_id} preview as JPEG")
                self._cache.update_jpeg(camera_id, encoded.tobytes())
                with self._lock:
                    self._last_encode_monotonic_ns[camera_id] = now_ns

    def _demanded_camera_ids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted({camera_id for camera_id, count in self._demand_by_source.values() if count > 0}))


class EpisodeIndexCache:
    """Refresh the episode index asynchronously with latest-request semantics."""

    def __init__(self, *, cache: NeroProjectionCache) -> None:
        self._cache = cache
        self._condition = threading.Condition()
        self._requested_root: str | None = None
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._last_completed_root = ""

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Nero episode index cache already started")
        self._thread = threading.Thread(target=self._run, name="nero_episode_index", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        self._thread.join()

    def request_refresh(self, root_dir: str) -> None:
        if not isinstance(root_dir, str) or not root_dir.strip():
            raise ValueError("Nero episode index root_dir must be a non-empty string")
        with self._condition:
            if self._stop_requested:
                raise RuntimeError("Nero episode index cache is stopped")
            self._requested_root = root_dir
            self._condition.notify()

    def status(self) -> dict[str, Any]:
        thread = self._thread
        alive = thread is not None and thread.is_alive()
        with self._condition:
            requested_root = self._requested_root or ""
            stopped = self._stop_requested
            last_completed_root = self._last_completed_root
        if thread is None:
            state = "new"
        elif stopped:
            state = "stopped"
        elif alive:
            state = "running"
        else:
            state = "failed"
        return {
            "state": state,
            "requested_root": requested_root,
            "last_completed_root": last_completed_root,
        }

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._requested_root is None and not self._stop_requested:
                    self._condition.wait()
                if self._stop_requested:
                    return
                root_dir = self._requested_root
                self._requested_root = None
            episodes = list_episodes(root_dir)
            self._cache.update_episodes(episodes)
            with self._condition:
                self._last_completed_root = root_dir
