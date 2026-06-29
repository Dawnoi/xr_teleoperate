import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.utils.operator_keybinds import (
    resolve_operator_controller_buttons,
)


class OperatorKeybindsTest(unittest.TestCase):
    def test_controller_left_x_requests_record_toggle_on_rising_edge(self):
        command = resolve_operator_controller_buttons(
            left_primary_pressed=True,
            prev_left_primary_pressed=False,
            right_secondary_pressed=False,
            prev_right_secondary_pressed=False,
            started=True,
            record_enabled=True,
        )
        self.assertTrue(command.set_record_toggle)
        self.assertIsNone(command.log_message)

    def test_controller_right_b_requests_record_cancel_on_rising_edge(self):
        command = resolve_operator_controller_buttons(
            left_primary_pressed=False,
            prev_left_primary_pressed=False,
            right_secondary_pressed=True,
            prev_right_secondary_pressed=False,
            started=True,
            record_enabled=True,
        )
        self.assertTrue(command.set_record_cancel)
        self.assertIsNone(command.log_message)

    def test_controller_record_shortcuts_require_record_mode(self):
        command = resolve_operator_controller_buttons(
            left_primary_pressed=True,
            prev_left_primary_pressed=False,
            right_secondary_pressed=False,
            prev_right_secondary_pressed=False,
            started=True,
            record_enabled=False,
        )
        self.assertIsNone(command.set_record_toggle)
        self.assertEqual(command.log_level, "warning")
        self.assertIn("recording is disabled", command.log_message)

    def test_controller_conflicting_record_shortcuts_are_rejected_explicitly(self):
        command = resolve_operator_controller_buttons(
            left_primary_pressed=True,
            prev_left_primary_pressed=False,
            right_secondary_pressed=True,
            prev_right_secondary_pressed=False,
            started=True,
            record_enabled=True,
        )
        self.assertIsNone(command.set_record_toggle)
        self.assertIsNone(command.set_record_cancel)
        self.assertEqual(command.log_level, "warning")
        self.assertIn("simultaneous left X and right B presses", command.log_message)


if __name__ == "__main__":
    unittest.main()
