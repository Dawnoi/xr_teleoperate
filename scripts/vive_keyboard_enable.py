#!/usr/bin/env python3
"""Hold-to-enable foot-pedal input for VIVE tracker teleoperation."""

import argparse
import fcntl
import glob
import os
import selectors
import struct

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


# Keep auto-detection portable; ambiguous device names still require an exact path.
_PEDAL_NAME_HINTS = ("pedal", "foot", "key08")


def _input_device_name(path):
    event_name = os.path.basename(os.path.realpath(path))
    try:
        with open(f"/sys/class/input/{event_name}/device/name", encoding="utf-8") as stream:
            return stream.read().strip()
    except OSError:
        return ""


def _resolve_input_paths(input_devices, input_device_glob):
    configured = [str(path).strip() for path in input_devices if str(path).strip()]
    if configured:
        if any(glob.has_magic(path) for path in configured):
            raise RuntimeError("--input-device entries must be exact device paths")
        candidates = configured
    else:
        candidates = sorted(glob.glob(input_device_glob))
        pedal_candidates = [
            path
            for path in candidates
            if any(
                hint in f"{os.path.basename(path)} {_input_device_name(path)}".lower()
                for hint in _PEDAL_NAME_HINTS
            )
        ]
        if pedal_candidates:
            candidates = pedal_candidates
        elif len(candidates) != 1:
            raise RuntimeError(
                "could not identify a unique pedal input device; "
                f"candidates={candidates}"
            )

    paths_by_target = {}
    for path in candidates:
        paths_by_target.setdefault(os.path.realpath(path), path)
    return list(paths_by_target.values()), bool(configured)


class HoldEnableState:
    """Translate Linux key events into left/right deadman state."""

    ALT_CODES = frozenset((56, 100))  # KEY_LEFTALT, KEY_RIGHTALT
    LEFT_CODE = 38  # KEY_L
    RIGHT_CODE = 19  # KEY_R

    def __init__(self, swap_sides=False):
        self._pressed = set()
        self._swap_sides = bool(swap_sides)

    def update(self, device_id, code, value):
        key = (device_id, code)
        if value == 1:
            self._pressed.add(key)
        elif value == 0:
            self._pressed.discard(key)
        return self.enabled

    def remove_device(self, device_id):
        self._pressed = {key for key in self._pressed if key[0] != device_id}
        return self.enabled

    def clear(self):
        self._pressed.clear()

    @property
    def enabled(self):
        codes = {code for _, code in self._pressed}
        alt = bool(codes & self.ALT_CODES)
        enabled = alt and self.LEFT_CODE in codes, alt and self.RIGHT_CODE in codes
        return enabled[::-1] if self._swap_sides else enabled


class ViveKeyboardEnable(Node):
    _INPUT_EVENT = struct.Struct("@llHHi")
    _EVIOCGRAB = 0x40044590
    _GRAB_ON = struct.pack("I", 1)
    _GRAB_OFF = struct.pack("I", 0)
    _EV_SYN = 0
    _EV_KEY = 1
    _SYN_REPORT = 0
    _SYN_DROPPED = 3

    def __init__(self, args):
        super().__init__("vive_keyboard_enable")
        self.left = False
        self.right = False
        self._input_devices = args.input_device
        self._input_device_glob = args.input_device_glob
        self._grab_input_devices = args.grab_input_devices
        self._state = HoldEnableState(swap_sides=args.swap_sides)
        self._selector = selectors.DefaultSelector()
        self._devices = {}
        self._grabbed_devices = set()
        self._dropped_devices = set()
        self._left_pub = self.create_publisher(Bool, args.left_topic, 10)
        self._right_pub = self.create_publisher(Bool, args.right_topic, 10)
        self.create_timer(1.0 / max(1.0, args.publish_rate_hz), self.publish)
        mapping = (
            "Alt+L -> right, Alt+R -> left"
            if args.swap_sides
            else "Alt+L -> left, Alt+R -> right"
        )
        self.get_logger().info(f"hold-to-enable pedal mapping: {mapping}")

    def open_input_devices(self):
        paths, explicit = _resolve_input_paths(
            self._input_devices,
            self._input_device_glob,
        )
        denied = []
        for path in paths:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError as exc:
                denied.append(f"{path}: {exc.strerror}")
                continue
            if self._grab_input_devices:
                try:
                    fcntl.ioctl(fd, self._EVIOCGRAB, self._GRAB_ON)
                except OSError as exc:
                    os.close(fd)
                    denied.append(f"{path}: exclusive grab failed: {exc.strerror}")
                    continue
                self._grabbed_devices.add(fd)
            self._selector.register(fd, selectors.EVENT_READ)
            self._devices[fd] = path

        if explicit and denied:
            for fd in list(self._devices):
                self._close_device(fd, warn=False)
            raise RuntimeError(
                "failed to open all configured input devices "
                f"({'; '.join(denied)})"
            )
        if not self._devices:
            detail = f" ({'; '.join(denied)})" if denied else ""
            raise RuntimeError(
                "no readable keyboard input device matches "
                f"{self._input_device_glob!r}{detail}"
            )
        mode = "configured" if explicit else "auto-selected"
        for path in self._devices.values():
            self.get_logger().info(f"{mode} pedal input: {path}")
        self.get_logger().info(
            f"listening on {len(self._devices)} input device(s); "
            f"exclusive_grab={self._grab_input_devices}"
        )

    def poll_input(self, timeout_sec=0.05):
        for selected, _ in self._selector.select(timeout_sec):
            fd = selected.fd
            try:
                data = os.read(fd, self._INPUT_EVENT.size * 64)
            except OSError:
                self._close_device(fd)
                continue
            if not data:
                self._close_device(fd)
                continue
            stop = len(data) - self._INPUT_EVENT.size + 1
            for offset in range(0, stop, self._INPUT_EVENT.size):
                _, _, event_type, code, value = self._INPUT_EVENT.unpack_from(data, offset)
                if event_type == self._EV_SYN:
                    if code == self._SYN_DROPPED:
                        self._dropped_devices.add(fd)
                        self._set_enabled(*self._state.remove_device(fd))
                    elif code == self._SYN_REPORT:
                        self._dropped_devices.discard(fd)
                elif fd not in self._dropped_devices and event_type == self._EV_KEY:
                    self._set_enabled(*self._state.update(fd, code, value))

    def _close_device(self, fd, warn=True):
        path = self._devices.pop(fd, str(fd))
        try:
            self._selector.unregister(fd)
        except Exception:
            pass
        if fd in self._grabbed_devices:
            try:
                fcntl.ioctl(fd, self._EVIOCGRAB, self._GRAB_OFF)
            except OSError:
                pass
            self._grabbed_devices.discard(fd)
        os.close(fd)
        self._dropped_devices.discard(fd)
        self._set_enabled(*self._state.remove_device(fd))
        if warn:
            self.get_logger().warning(f"input device disconnected: {path}")

    def close_input_devices(self):
        for fd in list(self._devices):
            self._close_device(fd, warn=False)
        self._selector.close()

    def _set_enabled(self, left, right):
        if (left, right) == (self.left, self.right):
            return
        self.left, self.right = left, right
        self.publish()
        self.get_logger().info(f"vive enable: left={self.left} right={self.right}")

    def publish(self):
        self._left_pub.publish(Bool(data=self.left))
        self._right_pub.publish(Bool(data=self.right))

    def disable(self):
        self._state.clear()
        self.left = self.right = False
        self.publish()
        self.get_logger().info("vive enable: left=False right=False")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-device",
        action="append",
        default=[],
        help="Exact /dev/input path; repeat for two pedals.",
    )
    parser.add_argument("--input-device-glob", default="/dev/input/by-path/*-event-kbd")
    parser.add_argument("--grab-input-devices", action="store_true")
    parser.add_argument(
        "--swap-sides",
        action="store_true",
        help="Map Alt+L to right and Alt+R to left.",
    )
    parser.add_argument("--left-topic", default="/vive/enable_left")
    parser.add_argument("--right-topic", default="/vive/enable_right")
    parser.add_argument("--publish-rate-hz", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rclpy.init(args=None)
    node = ViveKeyboardEnable(args)
    try:
        node.open_input_devices()
    except RuntimeError as exc:
        node.get_logger().error(str(exc))
        node.destroy_node()
        rclpy.shutdown()
        return 2

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.poll_input()
    finally:
        node.disable()
        node.close_input_devices()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
