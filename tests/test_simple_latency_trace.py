import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SimpleLatencyTraceTest(unittest.TestCase):
    def test_snapshot_exposes_pending_and_completed_online_execution_trace(self):
        from teleop.debug.latency_trace import SimpleLatencyTracker

        tracker = SimpleLatencyTracker(None, log_each_trace=False)
        seq = tracker.begin_trace(
            recv_ts_ns=1_100_000_000,
            recv_q=np.zeros(2, dtype=float),
            extra={
                "online_obs_send_perf_ns": 1_000_000_000,
                "online_action_recv_perf_ns": 1_075_000_000,
                "online_step_output_perf_ns": 1_095_000_000,
            },
        )
        tracker.mark_publish(seq, publish_ts_ns=1_112_000_000, dds_write_ms=0.4)

        pending = tracker.get_snapshot()
        self.assertEqual(pending["current"]["status"], "pending")
        self.assertAlmostEqual(pending["current"]["online_obs_send_to_publish_ms"], 112.0)
        self.assertAlmostEqual(pending["current"]["online_action_recv_to_publish_ms"], 37.0)

        tracker.maybe_mark_execute_thread(
            current_q=np.array([0.02, 0.0], dtype=float),
            current_dq=np.zeros(2, dtype=float),
            q_threshold=0.01,
            dq_threshold=0.05,
            detect_ts_ns=1_118_000_000,
        )
        tracker.maybe_mark_execute(
            current_q=np.array([0.03, 0.0], dtype=float),
            current_dq=np.zeros(2, dtype=float),
            q_threshold=0.01,
            dq_threshold=0.05,
            detect_ts_ns=1_125_000_000,
        )

        completed = tracker.get_snapshot()
        self.assertIsNone(completed["current"])
        self.assertEqual(completed["latest"]["status"], "completed")
        self.assertAlmostEqual(completed["latest"]["online_action_recv_to_exec_thread_ms"], 43.0)

    def test_online_inference_timestamps_derive_response_latency_fields(self):
        from teleop.debug.latency_trace import SimpleLatencyTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = pathlib.Path(tmpdir) / "latency.jsonl"
            tracker = SimpleLatencyTracker(str(trace_path), log_each_trace=False)
            seq = tracker.begin_trace(
                recv_ts_ns=1_100_000_000,
                recv_q=np.zeros(2, dtype=float),
                extra={
                    "online_obs_send_ns": 10_000_000_000,
                    "online_action_recv_ns": 10_075_000_000,
                    "online_step_output_ns": 10_100_000_000,
                    "online_obs_send_perf_ns": 1_000_000_000,
                    "online_action_recv_perf_ns": 1_075_000_000,
                    "online_step_output_perf_ns": 1_095_000_000,
                    "online_observation_seq": 3,
                    "online_chunk_seq": 4,
                    "online_chunk_index": 1,
                    "online_chunk_size": 2,
                },
            )

            tracker.mark_publish(seq, publish_ts_ns=1_112_000_000, dds_write_ms=0.4)
            tracker.maybe_mark_execute(
                current_q=np.array([0.02, 0.0], dtype=float),
                current_dq=np.zeros(2, dtype=float),
                q_threshold=0.01,
                dq_threshold=0.05,
            )

            records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["status"], "completed")
        self.assertAlmostEqual(record["online_obs_send_to_action_recv_ms"], 75.0)
        self.assertAlmostEqual(record["online_action_recv_to_step_output_ms"], 20.0)
        self.assertAlmostEqual(record["online_step_output_to_provider_return_ms"], 5.0)
        self.assertAlmostEqual(record["online_provider_return_to_pub_ms"], 12.0)
        self.assertAlmostEqual(record["online_step_output_to_pub_ms"], 17.0)
        self.assertGreaterEqual(record["online_step_output_to_exec_ms"], 17.0)
        self.assertGreaterEqual(record["online_obs_send_to_exec_ms"], 117.0)
        self.assertEqual(record["online_observation_seq"], 3)
        self.assertEqual(record["online_chunk_seq"], 4)
        self.assertEqual(record["online_chunk_index"], 1)
        self.assertEqual(record["online_chunk_size"], 2)

    def test_lowstate_thread_execute_marker_is_reported_separately(self):
        from teleop.debug.latency_trace import SimpleLatencyTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = pathlib.Path(tmpdir) / "latency.jsonl"
            tracker = SimpleLatencyTracker(str(trace_path), log_each_trace=False)
            seq = tracker.begin_trace(
                recv_ts_ns=1_000_000_000,
                recv_q=np.zeros(2, dtype=float),
                extra={
                    "online_step_output_perf_ns": 990_000_000,
                    "online_provider_output_perf_ns": 1_000_000_000,
                },
            )

            tracker.mark_publish(seq, publish_ts_ns=1_010_000_000, dds_write_ms=0.4)
            tracker.maybe_mark_execute_thread(
                current_q=np.array([0.02, 0.0], dtype=float),
                current_dq=np.zeros(2, dtype=float),
                q_threshold=0.01,
                dq_threshold=0.05,
                detect_ts_ns=1_018_000_000,
            )
            tracker.maybe_mark_execute(
                current_q=np.array([0.03, 0.0], dtype=float),
                current_dq=np.zeros(2, dtype=float),
                q_threshold=0.01,
                dq_threshold=0.05,
                detect_ts_ns=1_025_000_000,
            )

            record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["t_exec_thread_ns"], 1_018_000_000)
        self.assertEqual(record["t_exec_ns"], 1_025_000_000)
        self.assertAlmostEqual(record["pub_to_exec_thread_ms"], 8.0)
        self.assertAlmostEqual(record["recv_to_exec_thread_ms"], 18.0)
        self.assertAlmostEqual(record["online_step_output_to_exec_thread_ms"], 28.0)
        self.assertAlmostEqual(record["pub_to_exec_ms"], 15.0)
        self.assertAlmostEqual(record["q_delta_thread_trigger"], 0.02)
        self.assertAlmostEqual(record["dq_peak_thread_trigger"], 0.0)

    def test_non_online_trace_does_not_emit_online_latency_values(self):
        from teleop.debug.latency_trace import SimpleLatencyTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = pathlib.Path(tmpdir) / "latency.jsonl"
            tracker = SimpleLatencyTracker(str(trace_path), log_each_trace=False)
            seq = tracker.begin_trace(
                recv_ts_ns=1_100_000_000,
                recv_q=np.zeros(2, dtype=float),
                extra={"tele_fetch_ms": 3.0},
            )

            tracker.mark_publish(seq, publish_ts_ns=1_112_000_000, dds_write_ms=0.4)
            tracker.maybe_mark_execute(
                current_q=np.array([0.02, 0.0], dtype=float),
                current_dq=np.zeros(2, dtype=float),
                q_threshold=0.01,
                dq_threshold=0.05,
            )

            record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertIsNone(record["online_obs_send_to_action_recv_ms"])
        self.assertIsNone(record["online_action_recv_to_step_output_ms"])
        self.assertIsNone(record["online_step_output_trace_ns"])
        self.assertIsNone(record["online_step_output_to_pub_ms"])
        self.assertIsNone(record["online_step_output_to_exec_ms"])
        self.assertIsNone(record["online_obs_send_to_exec_ms"])
        self.assertIsNotNone(record["recv_to_exec_ms"])


if __name__ == "__main__":
    unittest.main()
