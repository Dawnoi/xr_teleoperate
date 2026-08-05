from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.input.vive_provider import ViveTrackerInputProvider, vive_config_from_args
from scripts.vive_axis_calibrator import solve_axis_calibration
from scripts.vive_keyboard_enable import HoldEnableState, _resolve_input_paths


IDENTITY_FLAT = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _provider() -> ViveTrackerInputProvider:
    provider = object.__new__(ViveTrackerInputProvider)
    provider._rclpy = SimpleNamespace(spin_once=lambda *_args, **_kwargs: None)
    provider._node = object()
    provider._timeout_sec = 0.25
    provider._enable_timeout_sec = 0.5
    provider._position_scale = 1.0
    provider._orientation_mode = "relative"
    provider._r_robot_vive = np.eye(3)
    provider._offset_xyz = np.zeros(3)
    provider._mount_rotation = {"left": np.eye(3), "right": np.eye(3)}
    provider._left_pose = np.eye(4)
    provider._right_pose = None
    provider._left_recv_time = time.monotonic()
    provider._right_recv_time = 0.0
    provider._left_tracker_anchor = None
    provider._right_tracker_anchor = None
    provider._left_robot_anchor = None
    provider._right_robot_anchor = None
    provider._left_enabled = False
    provider._right_enabled = False
    provider._left_enable_recv_time = 0.0
    provider._right_enable_recv_time = 0.0
    provider._frame_index = 0
    return provider


class ViveInputTest(unittest.TestCase):
    def test_pedal_chords_are_hold_to_enable(self):
        state = HoldEnableState()

        self.assertEqual(state.update("pedal", 56, 2), (False, False))
        self.assertEqual(state.update("pedal", 38, 1), (False, False))
        self.assertEqual(state.update("pedal", 56, 1), (True, False))
        self.assertEqual(state.update("pedal", 19, 1), (True, True))
        self.assertEqual(state.update("pedal", 38, 0), (False, True))
        self.assertEqual(state.update("pedal", 56, 0), (False, False))

    def test_pedal_sides_can_be_swapped(self):
        state = HoldEnableState(swap_sides=True)

        state.update("pedal", 56, 1)
        self.assertEqual(state.update("pedal", 38, 1), (False, True))
        self.assertEqual(state.update("pedal", 38, 0), (False, False))
        self.assertEqual(state.update("pedal", 19, 1), (True, False))

    def test_glob_rejects_multiple_devices(self):
        with patch(
            "scripts.vive_keyboard_enable.glob.glob",
            return_value=["/dev/input/event1", "/dev/input/event2"],
        ):
            with self.assertRaisesRegex(RuntimeError, "unique pedal"):
                _resolve_input_paths([], "/dev/input/event*")

    def test_common_pedal_name_is_auto_selected_from_by_path(self):
        with patch(
            "scripts.vive_keyboard_enable.glob.glob",
            return_value=[
                "/dev/input/by-path/pci-0000-usb-0:1-event-kbd",
                "/dev/input/by-path/pci-0000-usb-0:2-event-kbd",
            ],
        ), patch(
            "scripts.vive_keyboard_enable._input_device_name",
            side_effect=lambda path: "KM key08" if "0:2" in path else "Logitech Keyboard",
        ):
            paths, explicit = _resolve_input_paths([], "/dev/input/by-path/*-event-kbd")

        self.assertFalse(explicit)
        self.assertEqual(paths, ["/dev/input/by-path/pci-0000-usb-0:2-event-kbd"])

    def test_two_common_pedals_are_auto_selected_by_physical_path(self):
        candidates = [
            "/dev/input/by-path/pci-0000-usb-0:1-event-kbd",
            "/dev/input/by-path/pci-0000-usb-0:2-event-kbd",
        ]
        with patch("scripts.vive_keyboard_enable.glob.glob", return_value=candidates), patch(
            "scripts.vive_keyboard_enable._input_device_name",
            return_value="KM key08",
        ):
            paths, explicit = _resolve_input_paths([], "/dev/input/by-path/*-event-kbd")

        self.assertFalse(explicit)
        self.assertEqual(paths, candidates)

    def test_two_exact_pedal_paths_are_supported(self):
        paths, explicit = _resolve_input_paths(
            ["/dev/input/by-path/left-event-kbd", "/dev/input/by-path/right-event-kbd"],
            "/dev/input/event*",
        )

        self.assertTrue(explicit)
        self.assertEqual(
            paths,
            ["/dev/input/by-path/left-event-kbd", "/dev/input/by-path/right-event-kbd"],
        )

    def test_agx_axis_calibration_returns_rotation_and_origin_offset(self):
        rotation, offset = solve_axis_calibration(
            [2.0, 3.0, 4.0],
            [3.0, 3.0, 4.0],
            [2.0, 4.0, 4.0],
            robot_origin_xyz=[0.1, 0.2, 0.3],
        )

        np.testing.assert_allclose(rotation, np.eye(3))
        np.testing.assert_allclose(offset, [-1.9, -2.8, -3.7])

    def test_calibration_file_overrides_only_vive_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            calibration = Path(tmp) / "vive.json"
            calibration.write_text(
                json.dumps(
                    {
                        "rotation_robot_from_vive": IDENTITY_FLAT,
                        "offset_xyz": [0.1, 0.2, 0.3],
                        "position_scale": 0.8,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                vive_calibration_file=str(calibration),
                vive_rotation_robot_from_vive=[0.0] * 9,
                vive_offset_xyz=[0.0, 0.0, 0.0],
                vive_left_mount_rotation=IDENTITY_FLAT,
                vive_right_mount_rotation=IDENTITY_FLAT,
                vive_position_scale=1.0,
            )

            config = vive_config_from_args(args)

        self.assertEqual(config["rotation_robot_from_vive"], IDENTITY_FLAT)
        self.assertEqual(config["offset_xyz"], [0.1, 0.2, 0.3])
        self.assertEqual(config["position_scale"], 0.8)

    def test_default_calibration_file_uses_persistent_config_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            calibration = Path(tmp) / "xr_teleoperate" / "vive_calibration.json"
            calibration.parent.mkdir()
            calibration.write_text(json.dumps({"position_scale": 0.7}), encoding="utf-8")
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": tmp}, clear=False):
                config = vive_config_from_args(SimpleNamespace())

        self.assertEqual(config["position_scale"], 0.7)

    def test_pedal_enable_is_required_and_reanchors_on_each_enable(self):
        provider = _provider()
        current_left = np.eye(4)
        current_right = np.eye(4)

        disabled = provider.get_sample(
            current_left_robot_wrist_pose=current_left,
            current_right_robot_wrist_pose=current_right,
        )
        self.assertEqual(disabled.motion_intent.metadata["enabled_arms"], [])
        self.assertFalse(disabled.tele_data.left_ctrl_squeeze)

        provider._on_left_enable(SimpleNamespace(data=True))
        enabled = provider.get_sample(
            current_left_robot_wrist_pose=current_left,
            current_right_robot_wrist_pose=current_right,
        )
        self.assertEqual(enabled.motion_intent.metadata["enabled_arms"], ["left"])
        self.assertTrue(enabled.tele_data.left_ctrl_squeeze)
        self.assertIsNotNone(provider._left_tracker_anchor)

        provider._on_left_enable(SimpleNamespace(data=False))
        self.assertIsNone(provider._left_tracker_anchor)
        self.assertIsNone(provider._left_robot_anchor)

    def test_stale_pedal_heartbeat_disables_and_clears_anchor(self):
        provider = _provider()
        provider._on_left_enable(SimpleNamespace(data=True))
        provider.get_sample(
            current_left_robot_wrist_pose=np.eye(4),
            current_right_robot_wrist_pose=np.eye(4),
        )
        self.assertIsNotNone(provider._left_tracker_anchor)

        provider._left_enable_recv_time -= 1.0
        sample = provider.get_sample(
            current_left_robot_wrist_pose=np.eye(4),
            current_right_robot_wrist_pose=np.eye(4),
        )

        self.assertEqual(sample.motion_intent.metadata["enabled_arms"], [])
        self.assertIsNone(provider._left_tracker_anchor)

    def test_relative_orientation_matches_pico_anchor_semantics(self):
        provider = _provider()
        current_robot = np.eye(4)
        anchor = np.eye(4)
        provider._target_pose("left", anchor, current_robot)

        angle = np.pi / 2.0
        moved = np.eye(4)
        moved[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        moved[:3, 3] = [0.1, 0.2, 0.3]

        target = provider._target_pose("left", moved, current_robot)

        np.testing.assert_allclose(target[:3, :3], moved[:3, :3])
        np.testing.assert_allclose(target[:3, 3], moved[:3, 3])


if __name__ == "__main__":
    unittest.main()
