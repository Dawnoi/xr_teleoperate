import pathlib
import sys
import unittest
from types import SimpleNamespace

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.utils.operator_runtime import OperatorRuntime


class OperatorRuntimeTest(unittest.TestCase):
    def test_home_request_injects_one_frame_home_button_pulse(self):
        runtime = OperatorRuntime(record_enabled=True)
        tele_data = SimpleNamespace(
            left_ctrl_aButton=False,
            right_ctrl_bButton=False,
            left_ctrl_bButton=False,
        )

        runtime.request_home("test")
        runtime.apply_to_tele_data(tele_data, started=True, on_press=lambda key: None)
        self.assertTrue(tele_data.left_ctrl_bButton)

        tele_data.left_ctrl_bButton = False
        runtime.apply_to_tele_data(tele_data, started=True, on_press=lambda key: None)
        self.assertFalse(tele_data.left_ctrl_bButton)

    def test_controller_buttons_forward_to_existing_keyboard_shortcuts(self):
        runtime = OperatorRuntime(record_enabled=True)
        pressed_keys = []
        tele_data = SimpleNamespace(
            left_ctrl_aButton=True,
            right_ctrl_bButton=False,
            left_ctrl_bButton=False,
        )

        runtime.apply_to_tele_data(tele_data, started=True, on_press=pressed_keys.append)
        self.assertEqual(pressed_keys, ["s"])

        tele_data.left_ctrl_aButton = False
        tele_data.right_ctrl_bButton = True
        runtime.apply_to_tele_data(tele_data, started=True, on_press=pressed_keys.append)
        self.assertEqual(pressed_keys, ["s", "v"])

    def test_controller_button_hold_does_not_repeat_shortcut(self):
        runtime = OperatorRuntime(record_enabled=True)
        pressed_keys = []
        tele_data = SimpleNamespace(
            left_ctrl_aButton=True,
            right_ctrl_bButton=False,
            left_ctrl_bButton=False,
        )

        runtime.apply_to_tele_data(tele_data, started=True, on_press=pressed_keys.append)
        runtime.apply_to_tele_data(tele_data, started=True, on_press=pressed_keys.append)
        self.assertEqual(pressed_keys, ["s"])


if __name__ == "__main__":
    unittest.main()
