import pathlib
import sys
import unittest
from types import SimpleNamespace

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.debug.latency_setup import setup_latency_tracker


class LatencySetupTest(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "ui": True,
            "latency_trace": False,
            "latency_trace_path": "/tmp/latency.jsonl",
            "latency_summary_every": 1,
            "latency_timeout": 2.0,
            "latency_command_threshold": 0.01,
            "latency_exec_q_threshold": 0.01,
            "latency_exec_dq_threshold": 0.05,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_ui_mode_attaches_memory_only_latency_tracker(self):
        class Controller:
            def set_latency_tracker(self, tracker):
                self.tracker = tracker

            def set_latency_exec_thresholds(self, q_threshold, dq_threshold):
                self.thresholds = (q_threshold, dq_threshold)

        controller = Controller()
        tracker = setup_latency_tracker(self._args(), controller, log=SimpleNamespace(info=lambda *_args: None))

        self.assertIs(controller.tracker, tracker)
        self.assertIsNone(tracker.output_path)
        self.assertTrue(tracker.online_inference_only)
        self.assertEqual(controller.thresholds, (0.01, 0.05))

    def test_file_trace_keeps_existing_all_provider_scope(self):
        class Controller:
            def set_latency_tracker(self, tracker):
                self.tracker = tracker

            def set_latency_exec_thresholds(self, _q_threshold, _dq_threshold):
                return None

        controller = Controller()
        tracker = setup_latency_tracker(
            self._args(ui=False, latency_trace=True),
            controller,
            log=SimpleNamespace(info=lambda *_args: None),
        )

        self.assertFalse(tracker.online_inference_only)

    def test_ui_mode_rejects_controller_without_latency_hooks(self):
        with self.assertRaisesRegex(RuntimeError, "latency trace hooks"):
            setup_latency_tracker(self._args(), object(), log=SimpleNamespace(info=lambda *_args: None))


if __name__ == "__main__":
    unittest.main()
