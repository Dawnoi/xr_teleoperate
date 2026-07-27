"""Background Unitree DDS receiver for G1D mobile-base recording state."""

from __future__ import annotations

import importlib
import importlib.util
import math
import threading
import time
from collections import deque

from data_pipeline.recording.alignment import append_timed_sample


def _require_module(module_name: str) -> None:
    top_level_module = str(module_name).split(".", 1)[0]
    if importlib.util.find_spec(top_level_module) is None:
        raise RuntimeError(
            f"--record-base requires Unitree SDK2 Python module {module_name!r}; "
            "run in an environment that can import unitree_sdk2py or disable --record-base"
        )


def _field(obj, name: str):
    value = getattr(obj, name)
    return value() if callable(value) else value


def _finite_float(value, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return result


def _yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _stamp_to_ns(stamp) -> int:
    return int(_field(stamp, "sec")) * 1_000_000_000 + int(_field(stamp, "nanosec"))


class BaseStateReceiver:
    """Subscribe to base odometry/height topics and expose timestamped snapshots."""

    def __init__(
        self,
        *,
        odom_topic: str = "rt/agv/odom",
        height_topic: str = "rt/hispeed_state",
        history_size: int = 512,
        network_interface: str | None = None,
        domain_id: int = 0,
        node_name: str = "xr_teleoperate_base_state_receiver",
    ):
        odom_topic = str(odom_topic or "").strip()
        height_topic = str(height_topic or "").strip()
        if not odom_topic:
            raise ValueError("odom_topic must not be empty")
        if int(history_size) <= 0:
            raise ValueError("history_size must be positive")

        self.odom_topic = odom_topic
        self.height_topic = height_topic
        self.history_size = int(history_size)
        self.network_interface = str(network_interface).strip() if network_interface is not None else None
        self.domain_id = int(domain_id)
        self.node_name = str(node_name)
        self._lock = threading.Lock()
        self._base_state_history = deque(maxlen=self.history_size)
        self._base_height_history = deque(maxlen=self.history_size)
        self._odom_ready = threading.Event()
        self._height_ready = threading.Event()
        self._odom_subscriber = None
        self._height_subscriber = None
        self._started = False

    @property
    def height_enabled(self) -> bool:
        return bool(self.height_topic)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("BaseStateReceiver.start() called more than once")
        _require_module("unitree_sdk2py")

        channel_mod = importlib.import_module("unitree_sdk2py.core.channel")
        nav_msgs = importlib.import_module("unitree_sdk2py.idl.nav_msgs.msg.dds_")
        geometry_msgs = importlib.import_module("unitree_sdk2py.idl.geometry_msgs.msg.dds_") if self.height_enabled else None

        channel_mod.ChannelFactoryInitialize(self.domain_id, networkInterface=self.network_interface)
        self._odom_subscriber = channel_mod.ChannelSubscriber(self.odom_topic, nav_msgs.Odometry_)
        self._odom_subscriber.Init(self._handle_odom, 10)
        if self.height_enabled:
            self._height_subscriber = channel_mod.ChannelSubscriber(self.height_topic, geometry_msgs.Point32_)
            self._height_subscriber.Init(self._handle_height, 10)
        self._started = True

    def close(self) -> None:
        if self._height_subscriber is not None:
            self._height_subscriber.Close()
            self._height_subscriber = None
        if self._odom_subscriber is not None:
            self._odom_subscriber.Close()
            self._odom_subscriber = None
        self._started = False

    def wait_until_ready(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            if self.is_ready():
                return True
            time.sleep(0.02)
        return self.is_ready()

    def is_ready(self) -> bool:
        if not self._odom_ready.is_set():
            return False
        if self.height_enabled and not self._height_ready.is_set():
            return False
        return True

    def is_alive(self) -> bool:
        return self._started

    def snapshot_histories(self):
        with self._lock:
            base_state_history = deque(self._base_state_history, maxlen=self.history_size)
            base_height_history = deque(self._base_height_history, maxlen=self.history_size)
        return base_state_history, base_height_history

    def snapshot_latest(self):
        """Return the newest measured odometry and column samples without interpolation."""
        with self._lock:
            odom = dict(self._base_state_history[-1]) if self._base_state_history else None
            height = dict(self._base_height_history[-1]) if self._base_height_history else None
        return odom, height

    def _handle_odom(self, msg) -> None:
        host_monotonic_ns = int(time.monotonic_ns())
        header = _field(msg, "header")
        pose = _field(_field(msg, "pose"), "pose")
        twist = _field(_field(msg, "twist"), "twist")
        position = _field(pose, "position")
        orientation = _field(pose, "orientation")
        linear = _field(twist, "linear")
        angular = _field(twist, "angular")
        qx = _finite_float(_field(orientation, "x"), "odom.orientation.x")
        qy = _finite_float(_field(orientation, "y"), "odom.orientation.y")
        qz = _finite_float(_field(orientation, "z"), "odom.orientation.z")
        qw = _finite_float(_field(orientation, "w"), "odom.orientation.w")
        world_pose = {
            "x": _finite_float(_field(position, "x"), "odom.position.x"),
            "y": _finite_float(_field(position, "y"), "odom.position.y"),
            "z": _finite_float(_field(position, "z"), "odom.position.z"),
            "yaw": _yaw_from_quat_xyzw(qx, qy, qz, qw),
            "quat_xyzw": [qx, qy, qz, qw],
            "frame_id": str(_field(header, "frame_id") or ""),
            "source_topic": self.odom_topic,
        }
        velocity = {
            "vx": _finite_float(_field(linear, "x"), "odom.twist.linear.x"),
            "vy": _finite_float(_field(linear, "y"), "odom.twist.linear.y"),
            "vz": _finite_float(_field(linear, "z"), "odom.twist.linear.z"),
            "wz": _finite_float(_field(angular, "z"), "odom.twist.angular.z"),
            "source_topic": self.odom_topic,
        }
        with self._lock:
            append_timed_sample(
                self._base_state_history,
                host_monotonic_ns,
                world_pose=world_pose,
                velocity=velocity,
                source_topic=self.odom_topic,
                source_stamp_ns=_stamp_to_ns(_field(header, "stamp")),
            )
        self._odom_ready.set()

    def _handle_height(self, msg) -> None:
        host_monotonic_ns = int(time.monotonic_ns())
        with self._lock:
            append_timed_sample(
                self._base_height_history,
                host_monotonic_ns,
                height={
                    "z": _finite_float(_field(msg, "y"), "hispeed_state.y"),
                    "source_topic": self.height_topic,
                },
                source_topic=self.height_topic,
            )
        self._height_ready.set()


__all__ = ["BaseStateReceiver"]
