from __future__ import annotations

import time

from teleop.input.base import BaseTeleopInputProvider, MotionIntent, TeleopInputSample


class XRTeleopInputProvider(BaseTeleopInputProvider):
    def __init__(self, xr_wrapper):
        self._xr_wrapper = xr_wrapper
        self._frame_index = 0

    def get_sample(self, *args, **kwargs) -> TeleopInputSample | None:
        del args
        tele_data = self._xr_wrapper.get_tele_data(
            current_left_robot_wrist_pose=kwargs.get("current_left_robot_wrist_pose"),
            current_right_robot_wrist_pose=kwargs.get("current_right_robot_wrist_pose"),
        )
        if tele_data is None:
            return None
        sample = TeleopInputSample(
            tele_data=tele_data,
            motion_intent=MotionIntent(
                kind="pose",
                left_wrist_pose=tele_data.left_wrist_pose,
                right_wrist_pose=tele_data.right_wrist_pose,
                timestamp=time.time(),
                frame_index=self._frame_index,
                source="xr",
                metadata={},
            ),
            done=False,
        )
        self._frame_index += 1
        return sample

    def close(self) -> None:
        close_fn = getattr(self._xr_wrapper, "close", None)
        if callable(close_fn):
            close_fn()

    def calibrate_head_reference(self, *args, **kwargs):
        calibrate_fn = getattr(self._xr_wrapper, "calibrate_head_reference", None)
        if callable(calibrate_fn):
            return calibrate_fn(*args, **kwargs)
        return None

    def sync_reference_to_current_live_pose(self, *args, **kwargs):
        sync_fn = getattr(self._xr_wrapper, "sync_reference_to_current_live_pose", None)
        if callable(sync_fn):
            return sync_fn(*args, **kwargs)
        return None

    def has_live_pose_data(self) -> bool:
        has_live_pose_fn = getattr(self._xr_wrapper, "has_live_pose_data", None)
        if callable(has_live_pose_fn):
            return bool(has_live_pose_fn())
        return True

    @property
    def done(self) -> bool:
        return False
