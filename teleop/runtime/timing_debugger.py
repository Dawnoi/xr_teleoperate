from collections import deque
import time

import numpy as np
import logging_mp


logger_mp = logging_mp.getLogger(__name__)


class TimingDebugger:
    def __init__(self, enabled: bool = False, interval_sec: float = 2.0, history_size: int = 400):
        self.enabled = enabled
        self.interval_sec = max(0.5, float(interval_sec))
        self.history_size = max(50, int(history_size))
        self._last_report_time = time.time()
        self._reset()

    def _make_window(self):
        return deque(maxlen=self.history_size)

    def _reset(self):
        self.loop_count = 0
        self.loop_total = 0.0
        self.loop_max = 0.0
        self.overrun_count = 0
        self.tele_fetch_total = 0.0
        self.tele_fetch_max = 0.0
        self.tele_none_count = 0
        self.ik_total = 0.0
        self.ik_max = 0.0
        self.ik_count = 0
        self.agv_total = 0.0
        self.agv_max = 0.0
        self.agv_count = 0
        self.loop_window = self._make_window()
        self.tele_window = self._make_window()
        self.ik_window = self._make_window()
        self.agv_window = self._make_window()

    def _p95_ms(self, values):
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=float), 95) * 1000.0)

    def add_loop(self, dt: float, overrun: bool):
        if not self.enabled:
            return
        self.loop_count += 1
        self.loop_total += dt
        self.loop_max = max(self.loop_max, dt)
        self.loop_window.append(float(dt))
        if overrun:
            self.overrun_count += 1

    def add_tele_fetch(self, dt: float, got_data: bool):
        if not self.enabled:
            return
        self.tele_fetch_total += dt
        self.tele_fetch_max = max(self.tele_fetch_max, dt)
        self.tele_window.append(float(dt))
        if not got_data:
            self.tele_none_count += 1

    def add_ik(self, dt: float):
        if not self.enabled:
            return
        self.ik_total += dt
        self.ik_max = max(self.ik_max, dt)
        self.ik_count += 1
        self.ik_window.append(float(dt))

    def add_agv(self, dt: float):
        if not self.enabled:
            return
        self.agv_total += dt
        self.agv_max = max(self.agv_max, dt)
        self.agv_count += 1
        self.agv_window.append(float(dt))

    def maybe_report(self, arm_ctrl=None, gripper_ctrl=None):
        if not self.enabled:
            return
        now = time.time()
        if (now - self._last_report_time) < self.interval_sec:
            return

        arm_age = None
        if arm_ctrl is not None and hasattr(arm_ctrl, "lowstate_buffer"):
            try:
                arm_age = arm_ctrl.lowstate_buffer.GetAge()
            except Exception:
                arm_age = None

        gripper_age = None
        if gripper_ctrl is not None and hasattr(gripper_ctrl, "get_state_age"):
            try:
                gripper_age = gripper_ctrl.get_state_age()
            except Exception:
                gripper_age = None

        loop_avg_ms = (self.loop_total / self.loop_count * 1000.0) if self.loop_count else 0.0
        tele_avg_ms = (self.tele_fetch_total / self.loop_count * 1000.0) if self.loop_count else 0.0
        ik_avg_ms = (self.ik_total / self.ik_count * 1000.0) if self.ik_count else 0.0
        agv_avg_ms = (self.agv_total / self.agv_count * 1000.0) if self.agv_count else 0.0

        timing_msg = (
            "[TIMING] "
            f"loop avg/p95/max={loop_avg_ms:.1f}/{self._p95_ms(self.loop_window):.1f}/{self.loop_max * 1000.0:.1f} ms, "
            f"tele avg/p95/max={tele_avg_ms:.1f}/{self._p95_ms(self.tele_window):.1f}/{self.tele_fetch_max * 1000.0:.1f} ms, "
            f"ik avg/p95/max={ik_avg_ms:.1f}/{self._p95_ms(self.ik_window):.1f}/{self.ik_max * 1000.0:.1f} ms ({self.ik_count} calls), "
            f"agv avg/p95/max={agv_avg_ms:.1f}/{self._p95_ms(self.agv_window):.1f}/{self.agv_max * 1000.0:.1f} ms ({self.agv_count} calls)"
        )
        if arm_age is not None:
            timing_msg += f", arm_state_age_ms={arm_age * 1000.0:.1f}"
        logger_mp.info(timing_msg)

        arm_timing = None
        if arm_ctrl is not None and hasattr(arm_ctrl, "get_timing_snapshot"):
            try:
                arm_timing = arm_ctrl.get_timing_snapshot()
            except Exception:
                arm_timing = None

        if arm_timing is not None:
            loop_stats = arm_timing.get("loop_stats") or {}
            write_stats = arm_timing.get("write_stats") or {}
            enqueue_stats = arm_timing.get("enqueue_stats") or {}
            logger_mp.info(
                "[TIMING_DDS] publish_hz=%.1f, ctrl_loop avg/p95/max=%.2f/%.2f/%.2f ms, dds_write avg/p95/max=%.3f/%.3f/%.3f ms, enqueue->publish avg/p95/max=%s/%s/%s ms",
                arm_timing.get("publish_hz", 0.0),
                loop_stats.get("avg_ms", 0.0),
                loop_stats.get("p95_ms", 0.0),
                loop_stats.get("max_ms", 0.0),
                write_stats.get("avg_ms", 0.0),
                write_stats.get("p95_ms", 0.0),
                write_stats.get("max_ms", 0.0),
                f"{enqueue_stats.get('avg_ms', 0.0):.2f}" if enqueue_stats else "N/A",
                f"{enqueue_stats.get('p95_ms', 0.0):.2f}" if enqueue_stats else "N/A",
                f"{enqueue_stats.get('max_ms', 0.0):.2f}" if enqueue_stats else "N/A",
            )

        state_parts = [f"tele_none={self.tele_none_count}", f"overrun={self.overrun_count}/{self.loop_count}"]
        if arm_age is not None:
            state_parts.insert(0, f"arm_state_age_ms={arm_age * 1000.0:.1f}")
        if gripper_age is not None:
            state_parts.insert(1 if arm_age is not None else 0, f"gripper_state_age_ms={gripper_age * 1000.0:.1f}")
        logger_mp.info("[TIMING_STATE] " + ", ".join(state_parts))

        self._last_report_time = now
        self._reset()
