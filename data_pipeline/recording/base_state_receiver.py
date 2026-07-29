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


def _normalize_quaternion(quaternion: list[float], field_name: str) -> list[float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if abs(norm - 1.0) > 0.01:
        raise ValueError(f"{field_name} quaternion norm must be near 1.0, got {norm:.6f}")
    return [value / norm for value in quaternion]


def _rotate_vector_by_quaternion(quaternion: list[float], vector: list[float]) -> list[float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _multiply_quaternions(first: list[float], second: list[float]) -> list[float]:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return _normalize_quaternion(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        "composed SLAM TF",
    )


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
        record_slam_map_pose: bool = False,
        slam_map_frame: str = "slamware_map",
        base_link_frame: str = "base_link",
        slam_pose_source_frame: str = "laser",
        base_velocity_frame: str | None = None,
        slam_chain_max_skew_ms: float = 75.0,
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
        self.record_slam_map_pose = bool(record_slam_map_pose)
        self.slam_map_frame = str(slam_map_frame).strip()
        self.base_link_frame = str(base_link_frame).strip()
        self.slam_pose_source_frame = str(slam_pose_source_frame).strip()
        self.base_velocity_frame = (
            None if base_velocity_frame is None else str(base_velocity_frame).strip()
        )
        self.slam_chain_max_skew_ns = int(float(slam_chain_max_skew_ms) * 1e6)
        if self.record_slam_map_pose and (
            not self.slam_map_frame or not self.base_link_frame or not self.slam_pose_source_frame
        ):
            raise ValueError(
                "slam_map_frame, base_link_frame, and slam_pose_source_frame must not be empty "
                "when recording SLAM map pose"
            )
        if self.base_velocity_frame not in {None, "base_link", "world"}:
            raise ValueError("base_velocity_frame must be None, 'base_link', or 'world'")
        if self.slam_chain_max_skew_ns <= 0:
            raise ValueError("slam_chain_max_skew_ms must be positive")
        self._lock = threading.Lock()
        self._base_state_history = deque(maxlen=self.history_size)
        self._base_height_history = deque(maxlen=self.history_size)
        self._slam_tf_history = deque(maxlen=self.history_size * 5)
        self._odom_ready = threading.Event()
        self._height_ready = threading.Event()
        self._slam_map_ready = threading.Event()
        self._odom_subscriber = None
        self._height_subscriber = None
        self._started = False
        self._ros_shutdown = threading.Event()
        self._ros_thread = None
        self._ros_node = None
        self._tf_subscription = None
        self._rclpy = None
        self._map_to_odom = None
        self._odom_to_base_link = None
        self._last_composed_chain_stamps = None

    @property
    def height_enabled(self) -> bool:
        return bool(self.height_topic)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("BaseStateReceiver.start() called more than once")
        _require_module("unitree_sdk2py")

        # Initialize every ROS/TF import before DDS reader threads can execute Python callbacks.
        if self.record_slam_map_pose:
            self._start_ros_tf_listener()

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
        if self._ros_thread is not None:
            self._ros_shutdown.set()
            self._ros_thread.join()
            self._ros_thread = None
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            self._ros_node = None
        if self._rclpy is not None:
            self._rclpy.shutdown()
            self._rclpy = None
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
        if self.record_slam_map_pose and not self._slam_map_ready.is_set():
            return False
        return True

    def _start_ros_tf_listener(self) -> None:
        rclpy = importlib.import_module("rclpy")
        node_mod = importlib.import_module("rclpy.node")
        tf2_msgs = importlib.import_module("tf2_msgs.msg")
        rclpy.init()
        self._rclpy = rclpy
        self._ros_node = node_mod.Node(f"{self.node_name}_tf")
        self._tf_subscription = self._ros_node.create_subscription(
            tf2_msgs.TFMessage,
            "/tf",
            self._handle_tf_message,
            100,
        )
        self._ros_thread = threading.Thread(target=self._spin_ros_tf, name="slam_map_tf", daemon=True)
        self._ros_thread.start()

    def _spin_ros_tf(self) -> None:
        while not self._ros_shutdown.is_set():
            self._rclpy.spin_once(self._ros_node, timeout_sec=0.05)

    def is_alive(self) -> bool:
        return self._started

    def snapshot_histories(self):
        with self._lock:
            base_state_history = deque(self._base_state_history, maxlen=self.history_size)
            base_height_history = deque(self._base_height_history, maxlen=self.history_size)
        return base_state_history, base_height_history

    def snapshot_slam_tf_history(self):
        if not self.record_slam_map_pose:
            return None
        with self._lock:
            return deque(self._slam_tf_history, maxlen=self._slam_tf_history.maxlen)

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
            "frame_id": self.base_velocity_frame or "",
            "linear_unit": "m/s",
            "angular_unit": "rad/s",
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

    def _handle_tf_message(self, message) -> None:
        if not self.record_slam_map_pose:
            raise RuntimeError("received /tf while SLAM map recording is disabled")
        for transform in message.transforms:
            parent_frame_id = str(transform.header.frame_id)
            child_frame_id = str(transform.child_frame_id)
            if self.slam_pose_source_frame == "odom":
                if (parent_frame_id, child_frame_id) == (self.slam_map_frame, "odom"):
                    self._map_to_odom = self._raw_tf_record(transform)
                    self._append_composed_map_to_base_if_ready()
                elif (parent_frame_id, child_frame_id) == ("odom", self.base_link_frame):
                    self._odom_to_base_link = self._raw_tf_record(transform)
                    self._append_composed_map_to_base_if_ready()
                continue
            if (parent_frame_id, child_frame_id) == (self.slam_map_frame, self.slam_pose_source_frame):
                self._append_direct_slam_tf_transform(transform)

    def _raw_tf_record(self, transform) -> dict:
        lookup_monotonic_ns = int(time.monotonic_ns())
        lookup_wall_time_ns = int(time.time_ns())
        tf_header_stamp_ns = _stamp_to_ns(transform.header.stamp)
        if tf_header_stamp_ns <= 0:
            raise RuntimeError(
                f"TF {self.slam_map_frame}->{self.slam_pose_source_frame} has invalid header stamp "
                f"{tf_header_stamp_ns}"
            )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        quat_xyzw = _normalize_quaternion(
            [
                _finite_float(rotation.x, "slam_map_pose.rotation.x"),
                _finite_float(rotation.y, "slam_map_pose.rotation.y"),
                _finite_float(rotation.z, "slam_map_pose.rotation.z"),
                _finite_float(rotation.w, "slam_map_pose.rotation.w"),
            ],
            "raw SLAM TF",
        )
        return {
            "parent_frame_id": str(transform.header.frame_id),
            "child_frame_id": str(transform.child_frame_id),
            "x": _finite_float(translation.x, "slam_map_pose.translation.x"),
            "y": _finite_float(translation.y, "slam_map_pose.translation.y"),
            "z": _finite_float(translation.z, "slam_map_pose.translation.z"),
            "yaw": _yaw_from_quat_xyzw(*quat_xyzw),
            "quat_xyzw": quat_xyzw,
            "tf_header_stamp_ns": tf_header_stamp_ns,
            "tf_lookup_wall_time_ns": lookup_wall_time_ns,
            "tf_lookup_monotonic_ns": int(lookup_monotonic_ns),
            "tf_age_ms": (lookup_wall_time_ns - tf_header_stamp_ns) / 1e6,
        }

    def _append_direct_slam_tf_transform(self, transform) -> None:
        raw_tf = self._raw_tf_record(transform)
        slam_map_pose = {
            "x": raw_tf["x"], "y": raw_tf["y"], "z": raw_tf["z"], "yaw": raw_tf["yaw"],
            "quat_xyzw": raw_tf["quat_xyzw"],
            "frame_id": self.slam_map_frame,
            "child_frame_id": self.base_link_frame,
            "source_child_frame_id": self.slam_pose_source_frame,
            "source_to_base_link_identity_assumed": self.slam_pose_source_frame != self.base_link_frame,
            "source_topic": "/tf",
            **{key: raw_tf[key] for key in ("tf_header_stamp_ns", "tf_lookup_wall_time_ns", "tf_lookup_monotonic_ns", "tf_age_ms")},
        }
        self._append_slam_map_pose(slam_map_pose, raw_tf["tf_lookup_monotonic_ns"])

    def _append_composed_map_to_base_if_ready(self) -> None:
        if self._map_to_odom is None or self._odom_to_base_link is None:
            return
        map_to_odom = self._map_to_odom
        odom_to_base = self._odom_to_base_link
        chain_stamps = (map_to_odom["tf_header_stamp_ns"], odom_to_base["tf_header_stamp_ns"])
        if chain_stamps == self._last_composed_chain_stamps:
            return
        chain_skew_ns = abs(chain_stamps[0] - chain_stamps[1])
        if chain_skew_ns > self.slam_chain_max_skew_ns:
            return
        rotated_translation = _rotate_vector_by_quaternion(
            map_to_odom["quat_xyzw"],
            [odom_to_base["x"], odom_to_base["y"], odom_to_base["z"]],
        )
        quat_xyzw = _multiply_quaternions(map_to_odom["quat_xyzw"], odom_to_base["quat_xyzw"])
        lookup_monotonic_ns = max(map_to_odom["tf_lookup_monotonic_ns"], odom_to_base["tf_lookup_monotonic_ns"])
        lookup_wall_time_ns = max(map_to_odom["tf_lookup_wall_time_ns"], odom_to_base["tf_lookup_wall_time_ns"])
        oldest_header_stamp_ns = min(chain_stamps)
        slam_map_pose = {
            "x": map_to_odom["x"] + rotated_translation[0],
            "y": map_to_odom["y"] + rotated_translation[1],
            "z": map_to_odom["z"] + rotated_translation[2],
            "yaw": _yaw_from_quat_xyzw(*quat_xyzw),
            "quat_xyzw": quat_xyzw,
            "frame_id": self.slam_map_frame,
            "child_frame_id": self.base_link_frame,
            "source_child_frame_id": "odom",
            "source_to_base_link_identity_assumed": False,
            "source_topic": "/tf",
            "tf_header_stamp_ns": oldest_header_stamp_ns,
            "tf_lookup_wall_time_ns": lookup_wall_time_ns,
            "tf_lookup_monotonic_ns": lookup_monotonic_ns,
            "tf_age_ms": (lookup_wall_time_ns - oldest_header_stamp_ns) / 1e6,
            "source_chain": {
                "slamware_map_to_odom": map_to_odom,
                "odom_to_base_link": odom_to_base,
                "max_header_skew_ms": chain_skew_ns / 1e6,
            },
        }
        self._last_composed_chain_stamps = chain_stamps
        self._append_slam_map_pose(slam_map_pose, lookup_monotonic_ns)

    def _append_slam_map_pose(self, slam_map_pose: dict, timestamp_ns: int) -> None:
        with self._lock:
            append_timed_sample(
                self._slam_tf_history,
                timestamp_ns,
                **slam_map_pose,
            )
        self._slam_map_ready.set()

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
