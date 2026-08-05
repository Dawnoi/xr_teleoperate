import os
import cv2
import json
import datetime
import numpy as np
import time
import struct
import shutil
from collections import deque
from .rerun_visualizer import RerunLogger
from threading import Thread, Condition
import logging_mp
logger_mp = logging_mp.getLogger(__name__)


CAMERA_HISTORY_SIZE = 60
WRITER_QUEUE_SIZE = 60


class _WriterQueueView:
    """Compatibility view for queue depth and drain waits."""

    def __init__(self, writer, maxsize: int):
        self._writer = writer
        self.maxsize = int(maxsize)

    def qsize(self) -> int:
        with self._writer._writer_condition:
            return len(self._writer._pending_item_data)

    def empty(self) -> bool:
        return self.qsize() == 0

    def join(self) -> None:
        self._writer._wait_for_item_drain()


def canonical_color_key(color_key: str) -> str:
    key = str(color_key or "").strip()
    mapping = {
        "left_wrist": "wrist_left",
        "right_wrist": "wrist_right",
        "wrist_left": "wrist_left",
        "wrist_right": "wrist_right",
        "head": "head",
    }
    return mapping.get(key, key)


_CAMERA_MANIFEST_ORDER = ("head", "left_wrist", "right_wrist")
_CAMERA_MANIFEST_ALIASES = {
    "head": "head",
    "left_wrist": "left_wrist",
    "wrist_left": "left_wrist",
    "right_wrist": "right_wrist",
    "wrist_right": "right_wrist",
}


def normalize_enabled_cameras(enabled_cameras):
    if not isinstance(enabled_cameras, (list, tuple)):
        raise TypeError("enabled_cameras must be an explicit list or tuple of camera names")

    normalized = []
    for camera_name in enabled_cameras:
        if not isinstance(camera_name, str) or not camera_name.strip():
            raise ValueError("enabled_cameras entries must be non-empty strings")
        canonical_name = _CAMERA_MANIFEST_ALIASES.get(camera_name.strip())
        if canonical_name not in _CAMERA_MANIFEST_ORDER:
            raise ValueError(f"unsupported enabled camera: {camera_name}")
        if canonical_name in normalized:
            raise ValueError(f"duplicate enabled camera: {canonical_name}")
        normalized.append(canonical_name)

    return sorted(normalized, key=_CAMERA_MANIFEST_ORDER.index)

class ZMQRawCameraReceiver:
    """Receive latest raw image frame from a remote ZMQ PUB endpoint.

    Supported packet formats:

    1) New protocol:
       [4B magic='XRAW'][4B version=1]
       [4B width][4B height][4B channels]
       [8B frame_seq][8B source_wall_time_ns][8B source_monotonic_ns]
       [raw image bytes]

    2) Legacy protocol:
       [4B width][4B height][4B channels][raw image bytes]
    """

    MAGIC = b"XRAW"
    VERSION = 1
    HEADER_FMT = ">4sIIIIQQQ"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)
    LEGACY_HEADER_FMT = "<iii"
    LEGACY_HEADER_SIZE = struct.calcsize(LEGACY_HEADER_FMT)

    def __init__(self, endpoint: str, name: str = "camera"):
        import zmq as zmq_module

        self.endpoint = endpoint
        self.name = name
        self._zmq = zmq_module
        self._running = True
        self._frame = None
        self._meta = None
        self._history = deque(maxlen=CAMERA_HISTORY_SIZE)
        self._frame_seq_local = -1
        self._receiver_frame_seq = -1
        self._condition = Condition()

        self._ctx = self._zmq.Context.instance()
        self._socket = self._ctx.socket(self._zmq.SUB)
        self._socket.setsockopt(self._zmq.RCVHWM, 1)
        self._socket.setsockopt(self._zmq.LINGER, 0)
        self._socket.connect(self.endpoint)
        self._socket.setsockopt(self._zmq.SUBSCRIBE, b"")

        self._thread = Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger_mp.info(f"[ZMQRawCameraReceiver:{self.name}] connected to {self.endpoint}")

    def _recv_loop(self):
        zmq_module = self._zmq
        poller = zmq_module.Poller()
        poller.register(self._socket, zmq_module.POLLIN)
        while self._running:
            try:
                socks = dict(poller.poll(timeout=100))
                if self._socket not in socks:
                    continue
                recv_wall_ns = time.time_ns()
                recv_mono_ns = time.monotonic_ns()
                message = self._socket.recv(flags=zmq_module.NOBLOCK)
                frame, meta = self._decode_message(message, recv_wall_ns, recv_mono_ns)
                if frame is None:
                    continue
                with self._condition:
                    self._receiver_frame_seq += 1
                    meta["receiver_frame_seq"] = int(self._receiver_frame_seq)
                    self._frame = frame
                    self._meta = meta
                    self._history.append((frame, dict(meta)))
                    self._condition.notify_all()
            except zmq_module.Again:
                continue
            except Exception as e:
                logger_mp.warning(f"[ZMQRawCameraReceiver:{self.name}] recv loop error: {e}")

    def _decode_message(self, message: bytes, recv_wall_ns: int, recv_mono_ns: int):
        if len(message) >= self.HEADER_SIZE and message[:4] == self.MAGIC:
            try:
                magic, version, width, height, channels, frame_seq, src_wall_ns, src_mono_ns = struct.unpack(
                    self.HEADER_FMT, message[:self.HEADER_SIZE]
                )
                if magic != self.MAGIC or version != self.VERSION:
                    return None, None
                payload = message[self.HEADER_SIZE:]
                expected_size = width * height * channels
                if len(payload) != expected_size:
                    logger_mp.warning(
                        f"[ZMQRawCameraReceiver:{self.name}] payload size mismatch: "
                        f"expected={expected_size}, got={len(payload)}"
                    )
                    return None, None
                frame = self._payload_to_frame(payload, width, height, channels)
                if frame is None:
                    return None, None
                meta = {
                    "camera_name": self.name,
                    "transport": "zmq_raw",
                    "endpoint": self.endpoint,
                    "frame_seq": int(frame_seq),
                    "source_wall_time_ns": int(src_wall_ns),
                    "source_monotonic_ns": int(src_mono_ns),
                    "host_recv_wall_time_ns": int(recv_wall_ns),
                    "host_recv_monotonic_ns": int(recv_mono_ns),
                    "shape": [int(width), int(height), int(channels)],
                    "protocol": "xraw_v1",
                }
                return frame, meta
            except Exception as e:
                logger_mp.warning(f"[ZMQRawCameraReceiver:{self.name}] failed to decode xraw packet: {e}")
                return None, None

        if len(message) >= self.LEGACY_HEADER_SIZE:
            try:
                width, height, channels = struct.unpack(self.LEGACY_HEADER_FMT, message[:self.LEGACY_HEADER_SIZE])
                payload = message[self.LEGACY_HEADER_SIZE:]
                expected_size = width * height * channels
                if len(payload) != expected_size:
                    return None, None
                frame = self._payload_to_frame(payload, width, height, channels)
                if frame is None:
                    return None, None
                self._frame_seq_local += 1
                meta = {
                    "camera_name": self.name,
                    "transport": "zmq_raw",
                    "endpoint": self.endpoint,
                    "frame_seq": int(self._frame_seq_local),
                    "host_recv_wall_time_ns": int(recv_wall_ns),
                    "host_recv_monotonic_ns": int(recv_mono_ns),
                    "shape": [int(width), int(height), int(channels)],
                    "protocol": "legacy_raw_v0",
                }
                return frame, meta
            except Exception:
                return None, None
        return None, None

    def _payload_to_frame(self, payload: bytes, width: int, height: int, channels: int):
        if channels not in (3, 4):
            logger_mp.warning(f"[ZMQRawCameraReceiver:{self.name}] unsupported channels={channels}")
            return None
        frame = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, channels))
        if channels == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def get_latest(self, copy: bool = True):
        with self._condition:
            if self._frame is None or self._meta is None:
                return None, None
            frame = self._frame.copy() if copy else self._frame
            meta = dict(self._meta)
        return frame, meta

    @staticmethod
    def _meta_time_ns(meta):
        value = meta.get("host_recv_monotonic_ns")
        return int(value) if value is not None else None

    def get_nearest(self, target_monotonic_ns: int, max_delta_ns=None, min_monotonic_ns=None, copy: bool = True):
        target_monotonic_ns = int(target_monotonic_ns)
        with self._condition:
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

    def wait_for_next_frame(
        self,
        *,
        after_receiver_frame_seq: int | None,
        min_monotonic_ns: int | None = None,
        timeout_sec: float = 0.05,
        copy: bool = False,
    ):
        """Return the next locally received packet without consuming ``get_latest``."""
        if timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be positive")
        min_monotonic_ns = None if min_monotonic_ns is None else int(min_monotonic_ns)
        deadline_ns = time.monotonic_ns() + int(float(timeout_sec) * 1e9)
        with self._condition:
            while self._running:
                if self._history:
                    oldest_meta = self._history[0][1]
                    oldest_seq = int(oldest_meta["receiver_frame_seq"])
                    if (
                        after_receiver_frame_seq is not None
                        and oldest_seq > int(after_receiver_frame_seq) + 1
                    ):
                        raise RuntimeError(
                            f"[ZMQRawCameraReceiver:{self.name}] recorder camera cursor overrun: "
                            f"after_receiver_frame_seq={after_receiver_frame_seq} "
                            f"oldest_available_receiver_frame_seq={oldest_seq}"
                        )
                    for frame, meta in self._history:
                        receiver_seq = int(meta["receiver_frame_seq"])
                        if after_receiver_frame_seq is not None and receiver_seq <= int(after_receiver_frame_seq):
                            continue
                        frame_time_ns = self._meta_time_ns(meta)
                        if frame_time_ns is None:
                            raise RuntimeError(
                                f"[ZMQRawCameraReceiver:{self.name}] frame metadata missing host monotonic timestamp"
                            )
                        if min_monotonic_ns is not None and frame_time_ns < min_monotonic_ns:
                            continue
                        return (frame.copy() if copy else frame), dict(meta)

                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    return None, None
                self._condition.wait(timeout=remaining_ns / 1e9)
        return None, None

    def close(self):
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._socket.close()
        except Exception:
            pass

class EpisodeWriter():
    def __init__(
        self,
        task_dir,
        task_goal=None,
        task_desc=None,
        task_steps=None,
        frequency=30,
        image_size=[640, 480],
        rerun_log=True,
        episode_finalized_callback=None,
    ):
        """
        image_size: [width, height]
        """
        logger_mp.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "task description",
            "steps":"step1: do this; step2: do that; ...",
        }
        if task_goal is not None:
            self.text['goal'] = task_goal
        if task_desc is not None:
            self.text['desc'] = task_desc
        if task_steps is not None:
            self.text['steps'] = task_steps

        self.frequency = frequency
        self.image_size = image_size
        self.episode_finalized_callback = episode_finalized_callback

        self.rerun_log = rerun_log
        self.online_logger = None
        self.rerun_logger = None
        if self.rerun_log:
            logger_mp.info("==> Rerun live logging enabled; logger will be created per episode.\n")
        else:
            self.rerun_logger = None

        self.item_id = -1
        self.episode_id = -1
        self._data_file = None
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir and not episode_dir.endswith('.zip')]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info(f"==> episode directory does not exist, now create one.\n")
        self.data_info()

        self.is_available = True  # Indicates whether the class is available for new operations
        # The writer thread is the only owner of frame writes, file finalization,
        # and episode deletion. Producers only append data or lifecycle requests.
        self._writer_condition = Condition()
        self._pending_item_data = deque()
        self._unfinished_item_count = 0
        self._cancel_requested = False
        self._worker_status = "idle"
        self.item_data_queue = _WriterQueueView(self, WRITER_QUEUE_SIZE)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.\n")
    
    def is_ready(self):
        self._assert_worker_healthy()
        with self._writer_condition:
            return self.is_available

    def queue_status(self) -> dict[str, object]:
        self._assert_worker_healthy()
        with self._writer_condition:
            return {
                "depth": len(self._pending_item_data),
                "capacity": int(self.item_data_queue.maxsize),
                "unfinished": int(self._unfinished_item_count),
                "worker_status": str(self._worker_status),
                "cancel_requested": bool(self._cancel_requested),
            }

    def _assert_worker_healthy(self) -> None:
        if not self.worker_thread.is_alive() and not self.stop_worker:
            raise RuntimeError(
                "episode writer thread exited unexpectedly; inspect the writer traceback above"
            )

    def _wait_for_item_drain(self) -> None:
        with self._writer_condition:
            while self._unfinished_item_count > 0:
                self._writer_condition.wait(timeout=0.1)
                self._assert_worker_healthy()

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "unitree" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   [],
                    "left_ee":  [],
                    "right_arm":  [],
                    "right_ee": [],
                },

                "tactile_names": {
                    "left_ee": [],
                    "right_ee": [],
                }, 
                "sim_state": ""
            }

    def update_episode_info(self, metadata):
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("episode metadata must be a non-empty dict")
        for key, value in metadata.items():
            if not isinstance(key, str) or not key:
                raise ValueError("episode metadata keys must be non-empty strings")
            if key in self.info:
                raise ValueError(f"episode metadata must not overwrite existing key: {key}")
            self.info[key] = value

 
    def create_episode(self, *, enabled_cameras):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        self._assert_worker_healthy()
        with self._writer_condition:
            if not self.is_available:
                logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
                return False  # Return False if the class is unavailable
            if self._pending_item_data or self._unfinished_item_count:
                raise RuntimeError(
                    "episode writer is marked available while queued items still exist"
                )

        normalized_enabled_cameras = normalize_enabled_cameras(enabled_cameras)
        self.info = {**self.info, "enabled_cameras": normalized_enabled_cameras}

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.color_head_dir = os.path.join(self.color_dir, 'head')
        self.color_wrist_left_dir = os.path.join(self.color_dir, 'wrist_left')
        self.color_wrist_right_dir = os.path.join(self.color_dir, 'wrist_right')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.color_head_dir, exist_ok=True)
        os.makedirs(self.color_wrist_left_dir, exist_ok=True)
        os.makedirs(self.color_wrist_right_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        self._data_file = open(self.json_path, "w", encoding="utf-8", buffering=1024 * 1024)
        self._data_file.write('{\n')
        self._data_file.write('"info": ' + json.dumps(self.info, ensure_ascii=False, indent=4) + ',\n')
        self._data_file.write('"text": ' + json.dumps(self.text, ensure_ascii=False, indent=4) + ',\n')
        self._data_file.write('"data": [\n')
        self._data_file.flush()
        self.first_item = True   # Flag to handle commas in JSON array

        if self.rerun_log:
            try:
                rrd_path = os.path.join(self.episode_dir, "rerun.rrd")
                self.online_logger = RerunLogger(
                    prefix="online/",
                    IdxRangeBoundary=60,
                    memory_limit="300MB",
                    rrd_path=rrd_path,
                    spawn_viewer=True,
                )
            except Exception as e:
                self.rerun_log = False
                self.online_logger = None
                logger_mp.warning(f"==> Failed to create episode RerunLogger, disable live viewer logging: {e}")
        else:
            self.online_logger = None

        with self._writer_condition:
            self.need_save = False
            self._cancel_requested = False
            self.is_available = False
            self._worker_status = "recording"
            self._writer_condition.notify_all()
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, sim_state=None, timestamps=None, control_extras=None):
        self._assert_worker_healthy()
        with self._writer_condition:
            if self.is_available or self.need_save or self._cancel_requested:
                raise RuntimeError(
                    "episode writer rejected item because no writable episode is active"
                )
            if len(self._pending_item_data) >= self.item_data_queue.maxsize:
                raise RuntimeError(
                    "episode writer queue is full: "
                    f"depth={len(self._pending_item_data)} capacity={self.item_data_queue.maxsize}"
                )
            self.item_id += 1
            item_data = {
                'idx': self.item_id,
                'colors': colors,
                'depths': depths,
                'states': states,
                'actions': actions,
                'tactiles': tactiles,
                'audios': audios,
                'sim_state': sim_state,
                'timestamps': timestamps,
            }
            # `control_extras` is accepted for compatibility but intentionally not serialized.
            self._pending_item_data.append(item_data)
            self._unfinished_item_count += 1
            self._writer_condition.notify_all()

    def process_queue(self):
        while True:
            with self._writer_condition:
                while (
                    not self._cancel_requested
                    and not self._pending_item_data
                    and not self.need_save
                    and not self.stop_worker
                ):
                    self._worker_status = "idle"
                    self._writer_condition.wait()

                if self._cancel_requested:
                    dropped_count = len(self._pending_item_data)
                    self._pending_item_data.clear()
                    self._unfinished_item_count -= dropped_count
                    self._worker_status = "canceling"
                    action = "cancel"
                    item_data = None
                elif self._pending_item_data:
                    item_data = self._pending_item_data.popleft()
                    self._worker_status = "writing"
                    action = "write"
                elif self.need_save:
                    self.need_save = False
                    self._worker_status = "saving"
                    action = "save"
                    item_data = None
                elif self.stop_worker:
                    self._worker_status = "stopped"
                    return
                else:
                    raise RuntimeError("episode writer reached an invalid worker state")

            if action == "write":
                self._process_item_data(item_data)
                with self._writer_condition:
                    self._unfinished_item_count -= 1
                    self._writer_condition.notify_all()
                continue

            if action == "cancel":
                self._cancel_episode()
                with self._writer_condition:
                    self._cancel_requested = False
                    self.is_available = True
                    self._worker_status = "idle"
                    self._writer_condition.notify_all()
                continue

            self._save_episode()
            with self._writer_condition:
                if not self._cancel_requested:
                    self.is_available = True
                    self._worker_status = "idle"
                self._writer_condition.notify_all()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})
        timestamps = item_data.get('timestamps', {}) or {}
        rerun_colors = {}
        rerun_depths = {}
        rerun_enabled = self.online_logger is not None

        if rerun_enabled and colors:
            for color_key, color in colors.items():
                rerun_colors[color_key] = color.copy() if hasattr(color, "copy") else color
        if rerun_enabled and depths:
            for depth_key, depth in depths.items():
                rerun_depths[depth_key] = depth.copy() if hasattr(depth, "copy") else depth

        # Save images
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                canonical_key = canonical_color_key(color_key)
                camera_meta = ((timestamps.get('camera', {}) or {}).get(color_key, {}) or {})
                color_time_ns = (
                    camera_meta.get("host_recv_monotonic_ns")
                    or camera_meta.get("host_monotonic_ns")
                    or timestamps.get("sample_monotonic_ns")
                )
                color_name = (
                    f'{str(idx).zfill(6)}_{canonical_key}_{int(color_time_ns)}.jpg'
                    if color_time_ns is not None else
                    f'{str(idx).zfill(6)}_{canonical_key}.jpg'
                )
                target_dir = os.path.join(self.color_dir, canonical_key)
                color_path = os.path.join(target_dir, color_name)
                if not cv2.imwrite(color_path, color):
                    raise RuntimeError(f"failed to save color image: {color_path}")
                item_data['colors'][color_key] = os.path.join('colors', canonical_key, color_name)

        # Save depths
        if depths:
            for idx_depth, (depth_key, depth) in enumerate(depths.items()):
                depth_name = f'{str(idx).zfill(6)}_{depth_key}.jpg'
                depth_path = os.path.join(self.depth_dir, depth_name)
                if not cv2.imwrite(depth_path, depth):
                    raise RuntimeError(f"failed to save depth image: {depth_path}")
                item_data['depths'][depth_key] = os.path.join('depths', depth_name)

        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # Update episode data
        if self._data_file is None:
            raise RuntimeError("episode data file is not open while processing an item")
        if not self.first_item:
            self._data_file.write(",\n")
        self._data_file.write(json.dumps(item_data, ensure_ascii=False, separators=(",", ":")))
        self.first_item = False

        # Log data if necessary
        if rerun_enabled:
            rerun_item_data = dict(item_data)
            rerun_item_data['colors'] = rerun_colors
            rerun_item_data['depths'] = rerun_depths
            self.online_logger.log_item_data(rerun_item_data)

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self._assert_worker_healthy()
        with self._writer_condition:
            if self.is_available:
                raise RuntimeError("episode save requested without an active episode")
            if self._cancel_requested:
                raise RuntimeError("episode save requested while cancellation is pending")
            self.need_save = True
            self._writer_condition.notify_all()
        logger_mp.info(f"==> Episode saved start...")

    def cancel_episode(self):
        """
        Drop the active episode and reuse its episode index for the next recording.
        """
        self._assert_worker_healthy()
        with self._writer_condition:
            if self.is_available:
                return
            self.need_save = False
            self._cancel_requested = True
            self._worker_status = "cancel_requested"
            self._writer_condition.notify_all()
        logger_mp.info("==> Episode cancel requested; writer thread will discard it asynchronously.")

    def _cancel_episode(self):
        if self.online_logger is not None:
            self.online_logger.close()
            self.online_logger = None
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
        episode_dir = getattr(self, "episode_dir", None)
        if episode_dir and os.path.isdir(episode_dir):
            shutil.rmtree(episode_dir)
        self.item_id = -1
        self.episode_id = self.episode_id - 1
        self.first_item = True
        logger_mp.info("==> Episode canceled; next recording will reuse this episode index.")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        if self._data_file is None:
            raise RuntimeError("episode data file is not open while finalizing episode")
        self._data_file.write("\n]\n}")
        self._data_file.flush()
        self._data_file.close()
        self._data_file = None

        if self.online_logger is not None:
            self.online_logger.close()
            logger_mp.info(f"==> Rerun recording finalized at {os.path.join(self.episode_dir, 'rerun.rrd')}.")
            self.online_logger = None

        if self.episode_finalized_callback is not None:
            self.episode_finalized_callback(self.episode_dir)

        logger_mp.info(f"==> Episode saved successfully to {self.json_path}.")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self._assert_worker_healthy()
        with self._writer_condition:
            if not self.is_available and not self._cancel_requested:
                self.need_save = True
                self._writer_condition.notify_all()
            while not self.is_available:
                self._writer_condition.wait(timeout=0.1)
                self._assert_worker_healthy()
            self.stop_worker = True
            self._writer_condition.notify_all()
        self.worker_thread.join()
        if self.online_logger is not None:
            self.online_logger.close()
            self.online_logger = None
        if self.rerun_logger is not None:
            self.rerun_logger.close()
