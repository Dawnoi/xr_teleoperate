"""Measured G1-D state acquisition for the whole-body controller."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from teleop.control_flow.mobile_manipulation_coordinator import (
    MobileStateSample,
    column_position_from_raw_height,
)


class MobileStateProvider:
    """Build fresh WBC mobile state from timestamped DDS receiver snapshots.

    The caller owns the robot-specific fail-closed action. It is invoked once
    before the provider starts waiting for replacement DDS samples.
    """

    def __init__(
        self,
        *,
        receiver: Any,
        kinematics: Any,
        state_timeout_sec: float,
        retry_count: int,
        retry_interval_sec: float,
        raw_height_minimum: float,
        raw_height_maximum: float,
        column_travel_m: float,
        enter_fail_closed_hold: Callable[[], None],
        log: Any,
    ) -> None:
        if receiver is None:
            raise ValueError("mobile state provider requires a base-state receiver")
        if kinematics is None:
            raise ValueError("mobile state provider requires G1D kinematics")
        if not math.isfinite(state_timeout_sec) or state_timeout_sec <= 0.0:
            raise ValueError("mobile state timeout must be positive and finite")
        if int(retry_count) < 0:
            raise ValueError("mobile state retry count must be non-negative")
        if not math.isfinite(retry_interval_sec) or retry_interval_sec <= 0.0:
            raise ValueError("mobile state retry interval must be positive and finite")
        if not math.isfinite(raw_height_minimum) or not math.isfinite(raw_height_maximum):
            raise ValueError("mobile state raw height limits must be finite")
        if raw_height_maximum <= raw_height_minimum:
            raise ValueError("mobile state raw height limits must be ordered")
        if not math.isfinite(column_travel_m) or column_travel_m <= 0.0:
            raise ValueError("mobile state column travel must be positive and finite")

        self._receiver = receiver
        self._kinematics = kinematics
        self._state_timeout_ms = float(state_timeout_sec) * 1e3
        self._retry_count = int(retry_count)
        self._retry_interval_sec = float(retry_interval_sec)
        self._raw_height_minimum = float(raw_height_minimum)
        self._raw_height_maximum = float(raw_height_maximum)
        self._column_travel_m = float(column_travel_m)
        self._enter_fail_closed_hold = enter_fail_closed_hold
        self._log = log

    def current(self) -> MobileStateSample:
        """Return a fresh measured base state or raise an explicit safety error."""
        for retry_index in range(self._retry_count + 1):
            odom_sample, height_sample = self._receiver.snapshot_latest()
            failure, odom_age_ms, height_age_ms = self._freshness_failure(odom_sample, height_sample)
            if failure is None:
                break

            if retry_index >= self._retry_count:
                raise RuntimeError(
                    f"{failure} odom_age_ms={odom_age_ms:.1f} height_age_ms={height_age_ms:.1f} "
                    f"timeout_ms={self._state_timeout_ms:.1f} retries={self._retry_count} "
                    f"retry_interval_ms={self._retry_interval_sec * 1e3:.1f}"
                )

            if retry_index == 0:
                self._enter_fail_closed_hold()

            odom_after_t_ns = self._required_cursor(odom_sample, odom_age_ms)
            height_after_t_ns = self._required_cursor(height_sample, height_age_ms)
            self._log.warning(
                "[MOBILE_STATE_RETRY] attempt=%d/%d %s odom_age_ms=%.1f height_age_ms=%.1f; "
                "holding arm/base and waiting up to %.1fms for fresh feedback.",
                retry_index + 1,
                self._retry_count,
                failure,
                odom_age_ms,
                height_age_ms,
                self._retry_interval_sec * 1e3,
            )
            self._receiver.wait_for_new_samples(
                odom_after_t_ns=odom_after_t_ns,
                height_after_t_ns=height_after_t_ns,
                timeout_sec=self._retry_interval_sec,
            )
        else:
            raise RuntimeError("MOBILE_STATE_RETRY_LOOP_EXHAUSTED")

        if retry_index > 0:
            self._log.info(
                "[MOBILE_STATE_RETRY] recovered after %d/%d retries: odom_age_ms=%.1f height_age_ms=%.1f.",
                retry_index,
                self._retry_count,
                odom_age_ms,
                height_age_ms,
            )
        return self._build_sample(odom_sample, height_sample)

    def _freshness_failure(self, odom_sample, height_sample) -> tuple[str | None, float, float]:
        if odom_sample is None or height_sample is None:
            return (
                "MOBILE_STATE_MISSING",
                float("inf") if odom_sample is None else 0.0,
                float("inf") if height_sample is None else 0.0,
            )
        now_monotonic_ns = time.monotonic_ns()
        odom_age_ms = (now_monotonic_ns - int(odom_sample["t_ns"])) / 1e6
        height_age_ms = (now_monotonic_ns - int(height_sample["t_ns"])) / 1e6
        if odom_age_ms <= self._state_timeout_ms and height_age_ms <= self._state_timeout_ms:
            return None, odom_age_ms, height_age_ms
        return "MOBILE_STATE_STALE", odom_age_ms, height_age_ms

    def _required_cursor(self, sample, age_ms: float) -> int | None:
        if sample is None:
            # ``None`` means "already fresh" to BaseStateReceiver. A zero
            # cursor instead requires the first real DDS sample to arrive.
            return 0
        if age_ms > self._state_timeout_ms:
            return int(sample["t_ns"])
        return None

    def _build_sample(self, odom_sample: dict, height_sample: dict) -> MobileStateSample:
        pose = odom_sample["world_pose"]
        yaw = float(pose["yaw"])
        cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
        odom_world_from_agv = np.array([
            [cosine, -sine, 0.0, float(pose["x"])],
            [sine, cosine, 0.0, float(pose["y"])],
            [0.0, 0.0, 1.0, float(pose["z"])],
            [0.0, 0.0, 0.0, 1.0],
        ])
        column_position = column_position_from_raw_height(
            raw_height=float(height_sample["height"]["z"]),
            raw_minimum=self._raw_height_minimum,
            raw_maximum=self._raw_height_maximum,
            column_travel_m=self._column_travel_m,
        )
        global_from_ik = self._kinematics.global_from_ik(
            odom_world_from_agv=odom_world_from_agv,
            column_position=column_position,
            torso_yaw=0.0,
        )
        return MobileStateSample(
            global_from_ik=global_from_ik,
            monotonic_ns=min(int(odom_sample["t_ns"]), int(height_sample["t_ns"])),
            column_position=column_position,
            torso_yaw=0.0,
        )
