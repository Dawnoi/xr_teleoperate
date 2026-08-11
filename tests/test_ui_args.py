import pathlib
import sys
import types
import unittest
from unittest.mock import patch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.real.args import parse_args
from teleop.nero_console.runtime import require_rclpy
from teleop.ui.online_inference_runtime import OnlineInferenceUiRuntime


class UiArgsTest(unittest.TestCase):
    def test_ui_args_are_disabled_by_default(self):
        args = parse_args([])

        self.assertFalse(args.ui)
        self.assertEqual(args.ui_host, "127.0.0.1")
        self.assertEqual(args.ui_port, 8085)
        self.assertEqual(args.ui_preview_fps, 5.0)

    def test_ui_args_can_be_enabled(self):
        args = parse_args(["--ui", "--ui-host", "0.0.0.0", "--ui-port", "8090", "--ui-preview-fps", "10"])

        self.assertTrue(args.ui)
        self.assertEqual(args.ui_host, "0.0.0.0")
        self.assertEqual(args.ui_port, 8090)
        self.assertEqual(args.ui_preview_fps, 10.0)

    def test_ui_enables_headless_mode(self):
        args = parse_args(["--ui"])

        self.assertTrue(args.ui)
        self.assertTrue(args.headless)

    def test_nero_provider_is_disabled_by_default_and_enables_headless_mode(self):
        self.assertFalse(parse_args([]).nero_console_provider)

        args = parse_args(["--nero-console-provider"])

        self.assertTrue(args.nero_console_provider)
        self.assertTrue(args.headless)

    def test_nero_online_replay_requires_explicit_base_source_authorization(self):
        self.assertEqual(parse_args([]).nero_online_replay_base_source, "none")

        args = parse_args([
            "--nero-console-provider",
            "--nero-online-replay-arm-source", "state",
            "--nero-online-replay-base-source", "action",
        ])

        self.assertEqual(args.nero_online_replay_arm_source, "state")
        self.assertEqual(args.nero_online_replay_base_source, "action")

    def test_nero_provider_ros_preflight_imports_rclpy_only_on_demand(self):
        fake_rclpy = types.ModuleType("rclpy")

        with patch.dict(sys.modules, {"rclpy": fake_rclpy}):
            self.assertIs(require_rclpy(), fake_rclpy)

    def test_mobile_tcp23_profile_exposes_26d_state_and_23d_action_contract(self):
        class MobileInputs:
            @staticmethod
            def tcp23_runtime_error():
                return ""

            @staticmethod
            def pelvis_planar22_runtime_error():
                return ""

            @staticmethod
            def joint_base_runtime_error():
                return ""

        profiles = OnlineInferenceUiRuntime(args=types.SimpleNamespace(), mobile_inputs=MobileInputs()).profiles()
        tcp23_profile = next(profile for profile in profiles if profile["id"] == "mobile_tcp23")

        self.assertEqual(tcp23_profile["label"], "移动操作 BaseLink TCP 26D/23D")


if __name__ == "__main__":
    unittest.main()
