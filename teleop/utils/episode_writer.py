import os
import cv2
import json
import datetime
import numpy as np
import time
import zmq
import struct
import shutil
from collections import deque
from .rerun_visualizer import RerunLogger
from queue import Queue, Empty
from threading import Thread, Lock
import logging_mp
logger_mp = logging_mp.getLogger(__name__)


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
        self.endpoint = endpoint
        self.name = name
        self._running = True
        self._frame = None
        self._meta = None
        self._history = deque(maxlen=240)
        self._frame_seq_local = -1
        self._lock = Lock()

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.endpoint)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")

        self._thread = Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger_mp.info(f"[ZMQRawCameraReceiver:{self.name}] connected to {self.endpoint}")

    def _recv_loop(self):
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            try:
                socks = dict(poller.poll(timeout=100))
                if self._socket not in socks:
                    continue
                recv_wall_ns = time.time_ns()
                recv_mono_ns = time.monotonic_ns()
                message = self._socket.recv(flags=zmq.NOBLOCK)
                frame, meta = self._decode_message(message, recv_wall_ns, recv_mono_ns)
                if frame is None:
                    continue
                with self._lock:
                    self._frame = frame
                    self._meta = meta
                    self._history.append((frame, dict(meta)))
            except zmq.Again:
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
        with self._lock:
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
            self._socket.close()
        except Exception:
            pass

class EpisodeWriter():
    def __init__(self, task_dir, task_goal=None, task_desc = None, task_steps = None, frequency=30, image_size=[640, 480], rerun_log = True):
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

        self.rerun_log = rerun_log
        self.rerun_logger = None
        if self.rerun_log:
            logger_mp.info("==> Rerun live logging enabled; logger will be created per episode.\n")
        else:
            self.rerun_logger = None

        self.item_id = -1
        self.episode_id = -1
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
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.\n")
    
    def is_ready(self):
        return self.is_available

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

 
    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

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
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write('{\n')
            f.write('"info": ' + json.dumps(self.info, ensure_ascii=False, indent=4) + ',\n')
            f.write('"text": ' + json.dumps(self.text, ensure_ascii=False, indent=4) + ',\n')
            f.write('"data": [\n')
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

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, sim_state=None, timestamps=None, control_extras=None):
        # Increment the item ID
        self.item_id += 1
        # Create the item data dictionary
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
        # Enqueue the item data
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    logger_mp.info(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            # Check if save_episode was triggered
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})
        timestamps = item_data.get('timestamps', {}) or {}
        rerun_colors = {}
        rerun_depths = {}

        if colors:
            for color_key, color in colors.items():
                rerun_colors[color_key] = color.copy() if hasattr(color, "copy") else color
        if depths:
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
                os.makedirs(target_dir, exist_ok=True)
                if not cv2.imwrite(os.path.join(target_dir, color_name), color):
                    logger_mp.info(f"Failed to save color image.")
                item_data['colors'][color_key] = os.path.join('colors', canonical_key, color_name)

        # Save depths
        if depths:
            for idx_depth, (depth_key, depth) in enumerate(depths.items()):
                depth_name = f'{str(idx).zfill(6)}_{depth_key}.jpg'
                if not cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth):
                    logger_mp.info(f"Failed to save depth image.")
                item_data['depths'][depth_key] = os.path.join('depths', depth_name)

        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # Update episode data
        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self.first_item:
                f.write(",\n")
            f.write(json.dumps(item_data, ensure_ascii=False, indent=4))
            self.first_item = False

        # Log data if necessary
        if self.rerun_log:
            curent_record_time = time.time()
            logger_mp.info(f"==> episode_id:{self.episode_id}  item_id:{idx}  current_time:{curent_record_time}")
            rerun_item_data = dict(item_data)
            rerun_item_data['colors'] = rerun_colors
            rerun_item_data['depths'] = rerun_depths
            if self.online_logger is not None:
                self.online_logger.log_item_data(rerun_item_data)

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self.need_save = True  # Set the save flag
        logger_mp.info(f"==> Episode saved start...")

    def cancel_episode(self):
        """
        Drop the active episode and reuse its episode index for the next recording.
        """
        if self.is_available:
            return
        self.need_save = False
        while True:
            try:
                self.item_data_queue.get_nowait()
                self.item_data_queue.task_done()
            except Empty:
                break
        if self.rerun_log and self.online_logger is not None:
            try:
                self.online_logger.close()
            except Exception:
                pass
            self.online_logger = None
        episode_dir = getattr(self, "episode_dir", None)
        if episode_dir and os.path.isdir(episode_dir):
            shutil.rmtree(episode_dir, ignore_errors=True)
        self.item_id = -1
        self.episode_id = self.episode_id - 1
        self.first_item = True
        self.is_available = True
        logger_mp.info("==> Episode canceled; next recording will reuse this episode index.")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write("\n]\n}")      # Close the JSON array and object

        if self.rerun_log and self.online_logger is not None:
            try:
                self.online_logger.close()
                logger_mp.info(f"==> Rerun recording finalized at {os.path.join(self.episode_dir, 'rerun.rrd')}.")
            except Exception:
                pass
            self.online_logger = None

        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        logger_mp.info(f"==> Episode saved successfully to {self.json_path}.")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()
        if self.online_logger is not None:
            try:
                self.online_logger.close()
            except Exception:
                pass
            self.online_logger = None
        if self.rerun_logger is not None:
            try:
                self.rerun_logger.close()
            except Exception:
                pass
