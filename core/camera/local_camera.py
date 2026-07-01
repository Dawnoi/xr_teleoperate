import threading
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any

import cv2
import logging_mp

logger_mp = logging_mp.getLogger(__name__)


class LocalCameraStream:
    """Background reader for a local V4L2/OpenCV camera.

    The reader thread continuously grabs frames and keeps only the newest one.
    Each stored frame is paired with host-side timestamps measured around the
    blocking `cap.read()` call.
    """

    def __init__(
        self,
        name: str,
        camera_id: int,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fourcc: str = "MJPG",
        buffer_size: int = 1,
    ):
        self.name = name
        self.camera_id = int(camera_id)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.fourcc = str(fourcc or "MJPG")
        self.buffer_size = int(max(1, buffer_size))

        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"[LocalCameraStream:{self.name}] failed to open camera id={self.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        if len(self.fourcc) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

        self._lock = threading.Lock()
        self._running = True
        self._frame = None
        self._meta = None
        self._history = deque(maxlen=max(16, min(240, self.fps * 4)))
        self._frame_seq = -1
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

        logger_mp.info(
            f"[LocalCameraStream:{self.name}] opened camera id={self.camera_id}, "
            f"request={self.width}x{self.height}@{self.fps}, fourcc={self.fourcc}, buffer_size={self.buffer_size}"
        )

    def _reader_loop(self):
        while self._running:
            t0_wall_ns = time.time_ns()
            t0_mono_ns = time.monotonic_ns()
            ok, frame = self.cap.read()
            t1_mono_ns = time.monotonic_ns()
            t1_wall_ns = time.time_ns()

            if not ok or frame is None:
                time.sleep(0.005)
                continue

            self._frame_seq += 1
            meta = {
                "camera_name": self.name,
                "camera_id": self.camera_id,
                "frame_seq": int(self._frame_seq),
                "host_wall_time_ns": int((t0_wall_ns + t1_wall_ns) // 2),
                "host_monotonic_ns": int((t0_mono_ns + t1_mono_ns) // 2),
                "read_latency_ms": float((t1_mono_ns - t0_mono_ns) / 1e6),
                "shape": [int(frame.shape[1]), int(frame.shape[0]), int(frame.shape[2]) if frame.ndim == 3 else 1],
            }

            with self._lock:
                self._frame = frame
                self._meta = meta
                self._history.append((frame, dict(meta)))

    @staticmethod
    def _meta_time_ns(meta: Dict[str, Any]) -> Optional[int]:
        value = meta.get("host_monotonic_ns")
        return int(value) if value is not None else None

    def get_latest(self, copy: bool = True) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        with self._lock:
            if self._frame is None or self._meta is None:
                return None, None
            frame = self._frame.copy() if copy else self._frame
            meta = dict(self._meta)
        return frame, meta

    def get_nearest(
        self,
        target_monotonic_ns: int,
        max_delta_ns: Optional[int] = None,
        min_monotonic_ns: Optional[int] = None,
        copy: bool = True,
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        target_monotonic_ns = int(target_monotonic_ns)
        with self._lock:
            best_frame = None
            best_meta = None
            best_abs_delta = None
            for frame, meta in reversed(self._history):
                meta_ts = self._meta_time_ns(meta)
                if meta_ts is None:
                    continue
                if min_monotonic_ns is not None and meta_ts < int(min_monotonic_ns):
                    continue
                abs_delta = abs(meta_ts - target_monotonic_ns)
                if max_delta_ns is not None and abs_delta > int(max_delta_ns):
                    continue
                if best_abs_delta is None or abs_delta < best_abs_delta:
                    best_frame = frame
                    best_meta = meta
                    best_abs_delta = abs_delta
            if best_frame is None or best_meta is None:
                return None, None
            frame_out = best_frame.copy() if copy else best_frame
            meta_out = dict(best_meta)
            meta_out["align_target_monotonic_ns"] = target_monotonic_ns
            meta_out["delta_to_sample_ns"] = int(self._meta_time_ns(best_meta) - target_monotonic_ns)
        return frame_out, meta_out

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception as e:
            logger_mp.warning(f"[LocalCameraStream:{self.name}] release failed: {e}")
