from __future__ import annotations

from typing import Any, Mapping
import time

import numpy as np

from inference.online_session import CameraSample, OnlineInferenceSession, RobotStateSample
from core.input.base import (
    BaseTeleopInputProvider,
    MotionIntent,
    TeleopInputSample,
    _finite_gripper_width,
    _finite_pose_matrix,
    _make_identity_pose,
    dex1_q_to_trigger_value,
)
from core.input.xr_input_types import TeleData


class OnlineInferenceInputProvider(BaseTeleopInputProvider):
    def __init__(
        self,
        session: OnlineInferenceSession,
        arm_side: str,
        ee: str | None = None,
        no_gripper: bool = False,
        dex1_max_width_m: float = 0.054,
    ):
        self.session = session
        self.arm_side = str(arm_side)
        if self.arm_side not in {"left", "right", "both"}:
            raise ValueError(f"unsupported online inference arm_side: {arm_side!r}")
        self.ee = ee
        self.no_gripper = bool(no_gripper)
        self.dex1_max_width_m = float(dex1_max_width_m)
        self._frame_index = 0

        if not self.no_gripper and self.ee != "dex1":
            raise ValueError("online inference gripper control only supports ee='dex1'; use --no-gripper to ignore gripper actions")

    def get_sample(self, *args, **kwargs) -> TeleopInputSample | None:
        del args
        left_pose = _finite_pose_matrix(kwargs.get("current_left_robot_wrist_pose"), "current_left_robot_wrist_pose")
        right_pose = _finite_pose_matrix(kwargs.get("current_right_robot_wrist_pose"), "current_right_robot_wrist_pose")
        host_monotonic_ns = kwargs.get("current_state_host_monotonic_ns")
        if host_monotonic_ns is None:
            host_monotonic_ns = time.monotonic_ns()

        state_sample = RobotStateSample(
            host_monotonic_ns=int(host_monotonic_ns),
            left_pose=left_pose,
            right_pose=right_pose,
            left_gripper_width=_finite_gripper_width(kwargs.get("current_left_gripper_width", 0.0), "current_left_gripper_width"),
            right_gripper_width=_finite_gripper_width(kwargs.get("current_right_gripper_width", 0.0), "current_right_gripper_width"),
        )
        camera_samples = self._coerce_camera_samples(kwargs)
        step = self.session.tick(state_sample=state_sample, camera_samples=camera_samples)

        enabled_arms = [str(side) for side in step.enabled_arms if str(side) in {"left", "right"}]
        trigger_values = [10.0, 10.0]
        trigger_pressed = [False, False]
        if not self.no_gripper:
            if "left" in enabled_arms:
                trigger_values[0] = dex1_q_to_trigger_value(step.left_gripper_width)
                trigger_pressed[0] = True
            if "right" in enabled_arms:
                trigger_values[1] = dex1_q_to_trigger_value(step.right_gripper_width)
                trigger_pressed[1] = True

        tele_data = TeleData(
            head_pose=_make_identity_pose(),
            left_wrist_pose=np.asarray(step.left_pose, dtype=float),
            right_wrist_pose=np.asarray(step.right_pose, dtype=float),
            left_ctrl_trigger=trigger_pressed[0],
            left_ctrl_triggerValue=float(trigger_values[0]),
            left_ctrl_squeeze="left" in enabled_arms,
            left_ctrl_squeezeValue=1.0 if "left" in enabled_arms else 0.0,
            left_ctrl_thumbstickValue=np.zeros(2, dtype=float),
            right_ctrl_trigger=trigger_pressed[1],
            right_ctrl_triggerValue=float(trigger_values[1]),
            right_ctrl_squeeze="right" in enabled_arms,
            right_ctrl_squeezeValue=1.0 if "right" in enabled_arms else 0.0,
            right_ctrl_thumbstickValue=np.zeros(2, dtype=float),
        )
        metadata = dict(step.metadata or {})
        metadata.update(
            {
                "enabled_arms": enabled_arms,
                "online_inference_status": step.status,
                "arm_side": self.arm_side,
            }
        )
        motion_intent = MotionIntent(
            kind="pose",
            left_wrist_pose=np.asarray(step.left_pose, dtype=float),
            right_wrist_pose=np.asarray(step.right_pose, dtype=float),
            timestamp=time.time(),
            frame_index=self._frame_index,
            source="online_inference",
            metadata=metadata,
        )
        self._frame_index += 1
        done = step.status == "failed"
        return TeleopInputSample(tele_data=tele_data, motion_intent=motion_intent, done=done)

    def _coerce_camera_samples(self, kwargs: dict[str, Any]) -> list[CameraSample]:
        explicit_samples = kwargs.get("camera_samples")
        if explicit_samples is not None:
            samples = list(explicit_samples)
            self._set_required_camera_names([getattr(sample, "name", "") for sample in samples])
            return samples

        camera_sources = kwargs.get("camera_sources") or {}
        target_monotonic_ns = kwargs.get("current_state_host_monotonic_ns")
        samples: list[CameraSample] = []
        required_names: list[str] = []
        iterable = camera_sources.items() if isinstance(camera_sources, Mapping) else camera_sources
        for entry in iterable:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("camera_sources entries must be (name, source) pairs")
            name, source = entry
            if source is None:
                continue
            name = str(name)
            required_names.append(name)
            frame = None
            meta = None
            get_nearest = getattr(source, "get_nearest", None)
            if callable(get_nearest) and target_monotonic_ns is not None:
                frame, meta = get_nearest(int(target_monotonic_ns), max_delta_ns=self.session.max_camera_delta_ns, copy=True)
            if frame is None or meta is None:
                get_latest = getattr(source, "get_latest", None)
                if not callable(get_latest):
                    raise TypeError(f"camera source {name!r} must provide get_latest(copy=True)")
                frame, meta = get_latest(copy=True)
            if frame is None or meta is None:
                continue
            host_ns = meta.get("host_recv_monotonic_ns", meta.get("host_monotonic_ns"))
            if host_ns is None:
                raise KeyError(f"camera source {name!r} metadata missing host monotonic timestamp")
            if target_monotonic_ns is not None and abs(int(host_ns) - int(target_monotonic_ns)) > self.session.max_camera_delta_ns:
                continue
            samples.append(CameraSample(name=name, frame=frame, host_monotonic_ns=int(host_ns)))
        self._set_required_camera_names(required_names)
        return samples

    def _set_required_camera_names(self, names: list[str]) -> None:
        unique_names = list(dict.fromkeys(str(name) for name in names if str(name)))
        if not unique_names:
            return
        set_required = getattr(self.session, "set_required_camera_names", None)
        if callable(set_required):
            set_required(unique_names)

    def report_control_feedback(self, feedback: dict) -> None:
        report = getattr(self.session, "report_control_feedback", None)
        if callable(report):
            report(feedback)

    def get_debug_snapshot(self) -> dict:
        snapshot = getattr(self.session, "get_debug_snapshot", None)
        if callable(snapshot):
            return snapshot()
        return {}

    def close(self) -> None:
        close_fn = getattr(self.session, "close", None)
        if callable(close_fn):
            close_fn()
            return
        transport = getattr(self.session, "transport", None)
        close_transport = getattr(transport, "close", None)
        if callable(close_transport):
            close_transport()

    def has_live_pose_data(self) -> bool:
        return True

    @property
    def done(self) -> bool:
        return False
