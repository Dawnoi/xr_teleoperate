#!/usr/bin/env python3
"""VIVE-only keyboard enable publisher; other input providers never subscribe."""

import argparse
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class ViveKeyboardEnable(Node):
    def __init__(self, args):
        super().__init__("vive_keyboard_enable")
        self.left = False
        self.right = False
        self._left_pub = self.create_publisher(Bool, args.left_topic, 10)
        self._right_pub = self.create_publisher(Bool, args.right_topic, 10)
        self.create_timer(1.0 / max(1.0, args.publish_rate_hz), self.publish)
        self.get_logger().info("keys: space=toggle both, l=left, r=right, 0=off, q=quit")

    def publish(self):
        self._left_pub.publish(Bool(data=self.left))
        self._right_pub.publish(Bool(data=self.right))

    def toggle_both(self):
        self.left = self.right = not (self.left or self.right)
        self._log_state()

    def toggle_left(self):
        self.left = not self.left
        self._log_state()

    def toggle_right(self):
        self.right = not self.right
        self._log_state()

    def disable(self):
        self.left = self.right = False
        self._log_state()

    def _log_state(self):
        self.publish()
        self.get_logger().info(f"vive enable: left={self.left} right={self.right}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-topic", default="/vive/enable_left")
    parser.add_argument("--right-topic", default="/vive/enable_right")
    parser.add_argument("--publish-rate-hz", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rclpy.init(args=None)
    node = ViveKeyboardEnable(args)
    if not sys.stdin.isatty():
        node.get_logger().error("interactive terminal required")
        node.destroy_node()
        rclpy.shutdown()
        return 2

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if not select.select([sys.stdin], [], [], 0.0)[0]:
                continue
            key = sys.stdin.read(1).lower()
            if key == " ":
                node.toggle_both()
            elif key == "l":
                node.toggle_left()
            elif key == "r":
                node.toggle_right()
            elif key == "0":
                node.disable()
            elif key == "q":
                break
    finally:
        node.disable()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
