from __future__ import annotations

import sys
import time
from typing import Any, TextIO

import cv2
import numpy as np


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _pair(values: Any) -> list[Any]:
    if values is None:
        return [None, None]
    result = list(values)
    if len(result) != 2:
        raise ValueError(f"expected 2 gripper values, got {len(result)}")
    return result


def _format_pair(values: Any) -> str:
    left, right = _pair(values)
    return f"L={_format_value(left)} R={_format_value(right)}"


def _format_bool_pair(values: Any) -> str:
    left, right = _pair(values)
    return f"L={bool(left)} R={bool(right)}"


def _format_optional_bool_pair(values: Any) -> str:
    if values is None:
        return "L=N/A R=N/A"
    left, right = _pair(values)
    return f"L={_format_optional_bool(left)} R={_format_optional_bool(right)}"


def _format_optional_bool(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(bool(value))


def format_gripper_state_panel(snapshot: dict[str, Any]) -> str:
    episode_index = int(snapshot.get("episode_index", -1))
    frame_index = int(snapshot.get("frame_index", -1))
    source = str(snapshot.get("source", "unknown"))
    state_age = snapshot.get("state_age_sec")
    command_age = snapshot.get("command_age_sec")
    state_age_text = "N/A" if state_age is None else f"{float(state_age) * 1000.0:.1f}ms"
    command_age_text = "N/A" if command_age is None else f"{float(command_age) * 1000.0:.1f}ms"
    elapsed = snapshot.get("ui_elapsed_sec")
    elapsed_text = "N/A" if elapsed is None else f"{float(elapsed):.3f}s"
    tau_thresh = snapshot.get("tau_high_threshold")
    tau_thresh_text = "N/A" if tau_thresh is None else f"{float(tau_thresh):.3f}"
    last_error = str(snapshot.get("last_error", ""))

    lines = [
        "[GRIPPER_STATE]",
        f"status={snapshot.get('status', 'running')}",
        f"episode={episode_index:04d} frame={frame_index} source={source}",
        f"ui_start={snapshot.get('ui_start_source', 'unknown')} elapsed={elapsed_text} speed={_format_value(snapshot.get('speed_scale'))}",
        f"enabled      {_format_bool_pair([snapshot.get('left_enabled'), snapshot.get('right_enabled')])}",
        f"raw_qpos     {_format_pair(snapshot.get('raw_gripper_qpos'))}",
        f"trigger      {_format_pair(snapshot.get('trigger_value'))}",
        f"clip_pred    {_format_pair(snapshot.get('clip_predicted_q'))}",
        f"dds_cmd      {_format_pair(snapshot.get('dds_command_q'))}",
        f"cmd-raw      {_format_pair(snapshot.get('cmd_minus_raw'))}",
        f"cmd-clip     {_format_pair(snapshot.get('cmd_minus_clip_pred'))}",
        f"state        {_format_pair(snapshot.get('feedback_state'))}",
        f"tau_est      {_format_pair(snapshot.get('tau_est'))}",
        f"tau_high     {_format_optional_bool_pair(snapshot.get('tau_high'))} thresh={tau_thresh_text}",
        f"force_hold   {_format_optional_bool_pair(snapshot.get('force_hold_active'))}",
        f"lost         {_format_pair(snapshot.get('lost'))}",
        f"age          cmd={command_age_text} state={state_age_text}",
    ]
    if last_error:
        lines.append(f"last_error   {last_error}")
    return "\n".join(lines)


def render_gripper_state_panel_image(snapshot: dict[str, Any], width: int = 980, height: int = 560) -> np.ndarray:
    image = np.full((int(height), int(width), 3), (24, 26, 30), dtype=np.uint8)
    lines = format_gripper_state_panel(snapshot).splitlines()
    y = 42
    for row_index, line in enumerate(lines):
        color = (230, 235, 240)
        scale = 0.72
        thickness = 1
        if row_index == 0:
            color = (120, 220, 255)
            scale = 0.9
            thickness = 2
        elif line.startswith("status="):
            color = (120, 255, 170) if "RUNNING" in line or "running" in line else (80, 210, 255)
        elif line.startswith("tau"):
            color = (120, 190, 255)
        elif line.startswith("cmd-raw") or line.startswith("cmd-clip"):
            color = (255, 170, 130)
        elif line.startswith("state") or line.startswith("dds_cmd") or line.startswith("raw_qpos") or line.startswith("clip_pred"):
            color = (245, 230, 160)
        cv2.putText(
            image,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += 34
    return image


class TerminalGripperStateUI:
    def __init__(
        self,
        enabled: bool,
        interval_sec: float = 0.2,
        stream: TextIO | None = None,
    ):
        self.enabled = bool(enabled)
        self.interval_sec = max(0.05, float(interval_sec))
        self.stream = stream if stream is not None else sys.stdout
        self._last_update_time = 0.0
        self._line_count = 0

    def update(self, snapshot: dict[str, Any]) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if (now - self._last_update_time) < self.interval_sec:
            return
        panel = format_gripper_state_panel(snapshot)
        if self._line_count > 0:
            self.stream.write(f"\x1b[{self._line_count}F")
        lines = panel.splitlines()
        for line in lines:
            self.stream.write("\x1b[2K" + line + "\n")
        self.stream.flush()
        self._line_count = len(lines)
        self._last_update_time = now

    def close(self) -> None:
        if not self.enabled or self._line_count <= 0:
            return
        self.stream.write("\n")
        self.stream.flush()
        self._line_count = 0


class OpenCVGripperStateUI:
    def __init__(self, enabled: bool, window_name: str = "Dex1 Gripper Replay State"):
        self.enabled = bool(enabled)
        self.window_name = str(window_name)
        if self.enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 980, 560)

    def update(self, snapshot: dict[str, Any]) -> None:
        if not self.enabled:
            return
        image = render_gripper_state_panel_image(snapshot)
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)

    def close(self) -> None:
        if not self.enabled:
            return
        cv2.destroyWindow(self.window_name)
