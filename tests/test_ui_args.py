import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.real.args import parse_args


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


if __name__ == "__main__":
    unittest.main()
