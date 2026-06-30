#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.input.raw_offline import load_raw_dry_run_summary
from teleop.input.raw_offline import load_raw_episode_items
from teleop.input.raw_offline import raw_gripper_source_key
from teleop.input.teleop_input_provider import dex1_q_to_trigger_value
from teleop.operator.gripper_state_ui import OpenCVGripperStateUI


DEX1_LEFT_CMD_TOPIC = "rt/dex1/left/cmd"
DEX1_RIGHT_CMD_TOPIC = "rt/dex1/right/cmd"
DEX1_LEFT_STATE_TOPIC = "rt/dex1/left/state"
DEX1_RIGHT_STATE_TOPIC = "rt/dex1/right/state"
TRACKING_START_MARKER = "start Tracking"
DEX1_DELTA_GRIPPER_CMD = 0.80
DEX1_FORCE_HOLD_TAU_THRESH = 0.55
DEX1_FORCE_HOLD_DISABLED_TAU_THRESH = 999.0


class RawGripperTimelineSample:
    def __init__(self, timestamp: float, frame_index: int, gripper_q: list[float], source: str):
        self.timestamp = float(timestamp)
        self.frame_index = int(frame_index)
        self.gripper_q = list(gripper_q)
        self.source = str(source)


def load_dry_run_summary(dataset_root: str | Path, episode_index: int, arm_source: str) -> dict:
    return load_raw_dry_run_summary(dataset_root, episode_index, arm_source)


def print_dry_run_summary(summary: dict) -> None:
    print(
        f"[RAW_REPLAY] frames={summary['frame_count']} "
        f"sample_monotonic_ns={summary['sample_monotonic_ns_range'][0]}->{summary['sample_monotonic_ns_range'][1]} "
        f"motion_repr={summary['motion_repr']} "
        f"arm_source={summary['arm_source']} "
        f"episode_dir={summary['episode_dir']}"
    )


def _finite_float(value: float | str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_float(value: float | str, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_float(value: float | str, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def resolve_force_hold_tau_threshold(args) -> float:
    if bool(getattr(args, "disable_gripper_force_hold", False)):
        if getattr(args, "gripper_force_hold_tau_thresh", None) is not None:
            raise ValueError("--disable-gripper-force-hold conflicts with --gripper-force-hold-tau-thresh")
        return DEX1_FORCE_HOLD_DISABLED_TAU_THRESH
    cli_value = getattr(args, "gripper_force_hold_tau_thresh", None)
    if cli_value is not None:
        return _positive_float(cli_value, "--gripper-force-hold-tau-thresh")
    env_value = os.environ.get("DEX1_FORCE_HOLD_TAU_ENGAGE_THRESH")
    if env_value is not None:
        return _positive_float(env_value, "DEX1_FORCE_HOLD_TAU_ENGAGE_THRESH")
    return DEX1_FORCE_HOLD_TAU_THRESH


def build_subprocess_env(args) -> dict[str, str]:
    env = os.environ.copy()
    tau_thresh = resolve_force_hold_tau_threshold(args)
    if bool(getattr(args, "disable_gripper_force_hold", False)) or getattr(args, "gripper_force_hold_tau_thresh", None) is not None:
        env["DEX1_FORCE_HOLD_TAU_ENGAGE_THRESH"] = f"{tau_thresh:.6g}"
    optional_env_args = [
        ("gripper_force_hold_close_margin", "DEX1_FORCE_HOLD_CLOSE_MARGIN", "--gripper-force-hold-close-margin", _positive_float),
        ("gripper_force_hold_release_margin", "DEX1_FORCE_HOLD_RELEASE_MARGIN", "--gripper-force-hold-release-margin", _positive_float),
        ("gripper_force_hold_grace_sec", "DEX1_FORCE_HOLD_ENGAGE_GRACE_SEC", "--gripper-force-hold-grace-sec", _nonnegative_float),
        ("gripper_force_hold_engage_error_margin", "DEX1_FORCE_HOLD_ENGAGE_ERROR_MARGIN", "--gripper-force-hold-engage-error-margin", _positive_float),
    ]
    for attr, env_name, label, validator in optional_env_args:
        value = getattr(args, attr, None)
        if value is not None:
            env[env_name] = f"{validator(value, label):.6g}"
    return env


def print_force_hold_config(args) -> None:
    tau_thresh = resolve_force_hold_tau_threshold(args)
    mode = "disabled" if bool(getattr(args, "disable_gripper_force_hold", False)) else "enabled"
    print(
        f"[RAW_REPLAY] gripper_force_hold={mode} "
        f"tau_thresh={tau_thresh:.6g} "
        f"close_margin={getattr(args, 'gripper_force_hold_close_margin', None)} "
        f"release_margin={getattr(args, 'gripper_force_hold_release_margin', None)} "
        f"grace_sec={getattr(args, 'gripper_force_hold_grace_sec', None)} "
        f"engage_error_margin={getattr(args, 'gripper_force_hold_engage_error_margin', None)}"
    )


def teleop_entry_path() -> Path:
    return (REPO_ROOT / "teleop" / "teleop_hand_and_arm.py").resolve()


def build_subprocess_command(
    *,
    dataset_root: str,
    episode_index: int,
    network_interface: Optional[str],
    max_arm_joint_speed: float,
    speed_scale: float,
    motion: bool,
    arm_source: str,
    end_action: str,
) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        str(teleop_entry_path()),
        "--input-mode",
        "controller",
        "--arm",
        "G1_29",
        "--ee",
        "dex1",
        "--input-provider",
        "lerobot_offline",
        "--offline-replay-dataset-root",
        str(dataset_root),
        "--offline-replay-episode-index",
        str(episode_index),
        "--offline-replay-arm-source",
        str(arm_source),
        "--offline-replay-speed-scale",
        str(speed_scale),
        "--offline-replay-end-action",
        str(end_action),
        "--max-arm-joint-speed",
        str(max_arm_joint_speed),
        "--controller-deadman",
        "grip",
        "--head-reference-mode",
        "fixed_per_grip",
        "--controller-mapping-mode",
        "anchored_safe",
        "--controller-orientation-mode",
        "relative",
        "--base-controller",
        "none",
        "--headless",
        "--auto-start",
    ]
    if network_interface:
        cmd.extend(["--network-interface", str(network_interface)])
    if motion:
        cmd.append("--motion")
    return cmd


class Dex1DDSStateProbe:
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

        self.left_cmd_sub = ChannelSubscriber(DEX1_LEFT_CMD_TOPIC, MotorCmds_)
        self.right_cmd_sub = ChannelSubscriber(DEX1_RIGHT_CMD_TOPIC, MotorCmds_)
        self.left_state_sub = ChannelSubscriber(DEX1_LEFT_STATE_TOPIC, MotorStates_)
        self.right_state_sub = ChannelSubscriber(DEX1_RIGHT_STATE_TOPIC, MotorStates_)
        self.left_cmd_sub.Init()
        self.right_cmd_sub.Init()
        self.left_state_sub.Init()
        self.right_state_sub.Init()

        self._lock = threading.Lock()
        self._left_cmd_q = None
        self._right_cmd_q = None
        self._left_state_q = None
        self._right_state_q = None
        self._left_tau_est = None
        self._right_tau_est = None
        self._left_lost = None
        self._right_lost = None
        self._last_cmd_time = None
        self._last_state_time = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _read_cmd_q(self, msg):
        if msg is None or not getattr(msg, "cmds", None):
            return None
        return float(msg.cmds[0].q)

    def _read_state_tuple(self, msg):
        if msg is None or not getattr(msg, "states", None):
            return None
        state = msg.states[0]
        return float(state.q), float(state.tau_est), int(state.lost)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            left_cmd = self.left_cmd_sub.Read()
            right_cmd = self.right_cmd_sub.Read()
            left_state = self.left_state_sub.Read()
            right_state = self.right_state_sub.Read()

            left_cmd_q = self._read_cmd_q(left_cmd)
            right_cmd_q = self._read_cmd_q(right_cmd)
            left_state_tuple = self._read_state_tuple(left_state)
            right_state_tuple = self._read_state_tuple(right_state)
            now = time.time()

            with self._lock:
                if left_cmd_q is not None:
                    self._left_cmd_q = left_cmd_q
                    self._last_cmd_time = now
                if right_cmd_q is not None:
                    self._right_cmd_q = right_cmd_q
                    self._last_cmd_time = now
                if left_state_tuple is not None:
                    self._left_state_q, self._left_tau_est, self._left_lost = left_state_tuple
                    self._last_state_time = now
                if right_state_tuple is not None:
                    self._right_state_q, self._right_tau_est, self._right_lost = right_state_tuple
                    self._last_state_time = now
            time.sleep(0.01)

    def snapshot(self) -> dict:
        with self._lock:
            command_age = None if self._last_cmd_time is None else time.time() - self._last_cmd_time
            state_age = None if self._last_state_time is None else time.time() - self._last_state_time
            return {
                "dds_command_q": [self._left_cmd_q, self._right_cmd_q],
                "feedback_state": [self._left_state_q, self._right_state_q],
                "tau_est": [self._left_tau_est, self._right_tau_est],
                "lost": [self._left_lost, self._right_lost],
                "command_age_sec": command_age,
                "state_age_sec": state_age,
                "command_seen": self._left_cmd_q is not None or self._right_cmd_q is not None,
            }


class RawEpisodeTimeline:
    def __init__(self, dataset_root: str | Path, episode_index: int, arm_source: str, speed_scale: float):
        if arm_source not in {"action", "state", "fk_cmd_pose"}:
            raise ValueError(f"unsupported arm_source: {arm_source}")
        source_key = raw_gripper_source_key(arm_source)
        self.speed_scale = float(speed_scale)
        items = load_raw_episode_items(dataset_root, episode_index)
        self._samples: list[RawGripperTimelineSample] = []
        for row_index, item in enumerate(items):
            timestamps = item.get("timestamps")
            if not isinstance(timestamps, dict) or "sample_monotonic_ns" not in timestamps:
                raise KeyError(f"frame {row_index} missing timestamps.sample_monotonic_ns")
            source = item.get(source_key)
            if not isinstance(source, dict):
                raise KeyError(f"frame {row_index} missing {source_key}")
            left_ee = source.get("left_ee")
            right_ee = source.get("right_ee")
            if not isinstance(left_ee, dict) or not isinstance(right_ee, dict):
                raise KeyError(f"frame {row_index} missing {source_key}.left_ee/right_ee")
            left_q = left_ee.get("qpos")
            right_q = right_ee.get("qpos")
            if not isinstance(left_q, list) or len(left_q) != 1:
                raise ValueError(f"frame {row_index} {source_key}.left_ee.qpos must have length 1")
            if not isinstance(right_q, list) or len(right_q) != 1:
                raise ValueError(f"frame {row_index} {source_key}.right_ee.qpos must have length 1")
            gripper_q = [float(left_q[0]), float(right_q[0])]
            if not all(math.isfinite(value) for value in gripper_q):
                raise ValueError(f"frame {row_index} gripper qpos contains non-finite value")
            sample_ns = int(timestamps["sample_monotonic_ns"])
            if sample_ns < 0:
                raise ValueError(f"frame {row_index} sample_monotonic_ns must be non-negative")
            self._samples.append(
                RawGripperTimelineSample(
                    timestamp=float(sample_ns) / 1e9,
                    frame_index=row_index,
                    gripper_q=gripper_q,
                    source=f"raw_episode:{source_key}:qpos",
                )
            )
        if not self._samples:
            raise ValueError("raw episode timeline has no samples")
        self._first_timestamp = float(self._samples[0].timestamp)
        self._start_wall_time = None

    def start(self) -> None:
        if self._start_wall_time is None:
            self._start_wall_time = time.time()

    @property
    def started(self) -> bool:
        return self._start_wall_time is not None

    def elapsed_episode(self) -> float:
        if self._start_wall_time is None or self.speed_scale <= 0.0:
            return 0.0
        return (time.time() - self._start_wall_time) * self.speed_scale

    def current(self):
        target_timestamp = self._first_timestamp + self.elapsed_episode()
        best = self._samples[0]
        for sample in self._samples:
            if float(sample.timestamp) > target_timestamp:
                break
            best = sample
        return best


def _pair_delta(left_values, right_values) -> list[float | None]:
    if left_values is None or right_values is None:
        return [None, None]
    left_pair = list(left_values)
    right_pair = list(right_values)
    if len(left_pair) != 2 or len(right_pair) != 2:
        raise ValueError("gripper delta requires two-value pairs")
    result = []
    for left, right in zip(left_pair, right_pair):
        if left is None or right is None:
            result.append(None)
        else:
            result.append(float(left) - float(right))
    return result


def _clip_prediction(raw_qpos: list[float], feedback_state) -> list[float | None]:
    if feedback_state is None:
        return [None, None]
    state_pair = list(feedback_state)
    if len(raw_qpos) != 2 or len(state_pair) != 2:
        raise ValueError("clip prediction requires two-value pairs")
    predicted = []
    for raw_value, state_value in zip(raw_qpos, state_pair):
        if state_value is None:
            predicted.append(None)
        else:
            lower = float(state_value) - DEX1_DELTA_GRIPPER_CMD
            upper = float(state_value) + DEX1_DELTA_GRIPPER_CMD
            predicted.append(min(max(float(raw_value), lower), upper))
    return predicted


def _tau_high_flags(tau_est, threshold: float) -> list[bool | None]:
    if tau_est is None:
        return [None, None]
    tau_threshold = _positive_float(threshold, "tau threshold")
    tau_pair = list(tau_est)
    if len(tau_pair) != 2:
        raise ValueError("tau flags require two-value pairs")
    result = []
    for value in tau_pair:
        if value is None:
            result.append(None)
        else:
            result.append(abs(float(value)) > tau_threshold)
    return result


def _state_ui_snapshot(
    *,
    timeline: RawEpisodeTimeline,
    probe_snapshot: dict,
    episode_index: int,
    speed_scale: float,
    status: str,
    start_source: str,
    force_hold_tau_thresh: float,
) -> dict:
    intent = timeline.current()
    raw_qpos = [float(intent.gripper_q[0]), float(intent.gripper_q[1])]
    clip_predicted_q = _clip_prediction(raw_qpos, probe_snapshot.get("feedback_state"))
    dds_command_q = probe_snapshot.get("dds_command_q")
    return {
        "status": status,
        "episode_index": episode_index,
        "frame_index": int(intent.frame_index),
        "source": str(intent.source),
        "left_enabled": True,
        "right_enabled": True,
        "raw_gripper_qpos": raw_qpos,
        "trigger_value": [
            dex1_q_to_trigger_value(raw_qpos[0]),
            dex1_q_to_trigger_value(raw_qpos[1]),
        ],
        "clip_predicted_q": clip_predicted_q,
        "cmd_minus_raw": _pair_delta(dds_command_q, raw_qpos),
        "cmd_minus_clip_pred": _pair_delta(dds_command_q, clip_predicted_q),
        "tau_high": _tau_high_flags(probe_snapshot.get("tau_est"), force_hold_tau_thresh),
        "tau_high_threshold": float(force_hold_tau_thresh),
        "ui_start_source": start_source,
        "ui_elapsed_sec": timeline.elapsed_episode(),
        "speed_scale": float(speed_scale),
        **probe_snapshot,
    }


def run_state_ui(
    *,
    dataset_root: str | Path,
    episode_index: int,
    arm_source: str,
    speed_scale: float,
    force_hold_tau_thresh: float,
    network_interface: str | None,
    stop_event: threading.Event,
) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(0, networkInterface=network_interface)
    timeline = RawEpisodeTimeline(dataset_root, episode_index, arm_source, speed_scale)
    probe = Dex1DDSStateProbe()
    ui = OpenCVGripperStateUI(enabled=True)
    probe.start()
    timeline_started = False
    start_source = "not_started"
    try:
        while not stop_event.is_set():
            start_event = getattr(stop_event, "start_event", None)
            probe_snapshot = probe.snapshot()
            if start_event is not None and start_event.is_set():
                timeline.start()
                timeline_started = True
                start_source = "tracking_log"
                break
            if start_event is None:
                timeline.start()
                timeline_started = True
                start_source = "no_start_event"
                break
            if bool(probe_snapshot.get("command_seen")):
                timeline.start()
                timeline_started = True
                start_source = "dds_cmd_fallback"
                break
            ui.update(
                _state_ui_snapshot(
                    timeline=timeline,
                    probe_snapshot=probe_snapshot,
                    episode_index=episode_index,
                    speed_scale=speed_scale,
                    status="WAITING_TRACKING_START",
                    start_source=start_source,
                    force_hold_tau_thresh=force_hold_tau_thresh,
                )
            )
            time.sleep(0.02)
        while not stop_event.is_set():
            probe_snapshot = probe.snapshot()
            ui.update(
                _state_ui_snapshot(
                    timeline=timeline,
                    probe_snapshot=probe_snapshot,
                    episode_index=episode_index,
                    speed_scale=speed_scale,
                    status="RUNNING" if timeline_started else "WAITING_TRACKING_START",
                    start_source=start_source,
                    force_hold_tau_thresh=force_hold_tau_thresh,
                )
            )
            time.sleep(0.05)
    finally:
        probe.stop()
        ui.close()


def run_subprocess_with_start_signal(cmd: list[str], start_event: threading.Event, env: dict[str, str] | None = None) -> int:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if TRACKING_START_MARKER in line:
            start_event.set()
    return int(process.wait())


class SubprocessReplayRunner:
    def __init__(self, cmd: list[str], start_event: threading.Event, stop_event: threading.Event, env: dict[str, str] | None = None):
        self.cmd = list(cmd)
        self.start_event = start_event
        self.stop_event = stop_event
        self.env = env
        self.exit_code = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        self.exit_code = run_subprocess_with_start_signal(self.cmd, self.start_event, env=self.env)
        self.stop_event.set()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)


def run_replay(args) -> int:
    summary = load_raw_dry_run_summary(args.dataset_root, args.episode_index, args.arm_source)
    print_dry_run_summary(summary)
    print_force_hold_config(args)
    if args.dry_run:
        return 0

    subprocess_env = build_subprocess_env(args)
    cmd = build_subprocess_command(
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        network_interface=args.network_interface,
        max_arm_joint_speed=args.max_arm_joint_speed,
        speed_scale=args.speed_scale,
        motion=args.motion,
        arm_source=args.arm_source,
        end_action=args.end_action,
    )
    if args.no_gripper:
        cmd.append("--no-gripper")

    if not args.show_state_ui or args.no_gripper:
        return int(subprocess.call(cmd, env=subprocess_env))

    stop_event = threading.Event()
    start_event = threading.Event()
    setattr(stop_event, "start_event", start_event)
    force_hold_tau_thresh = resolve_force_hold_tau_threshold(args)
    runner = SubprocessReplayRunner(cmd, start_event, stop_event, env=subprocess_env)
    runner.start()
    try:
        run_state_ui(
            dataset_root=args.dataset_root,
            episode_index=args.episode_index,
            arm_source=args.arm_source,
            speed_scale=args.speed_scale,
            force_hold_tau_thresh=force_hold_tau_thresh,
            network_interface=args.network_interface,
            stop_event=stop_event,
        )
    finally:
        stop_event.set()
        runner.join(timeout=2.0)
    if runner.exit_code is None:
        raise RuntimeError("replay subprocess did not report an exit code")
    return int(runner.exit_code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a raw episode_xxxx/data.json recording through teleop_hand_and_arm.py.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--max-arm-joint-speed", type=float, default=1.5)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--show-state-ui", action="store_true")
    parser.add_argument(
        "--gripper-force-hold-tau-thresh",
        type=float,
        default=None,
        help="Override DEX1_FORCE_HOLD_TAU_ENGAGE_THRESH for this replay subprocess.",
    )
    parser.add_argument(
        "--disable-gripper-force-hold",
        action="store_true",
        help="Diagnostic only: set DEX1_FORCE_HOLD_TAU_ENGAGE_THRESH very high for this replay subprocess.",
    )
    parser.add_argument("--gripper-force-hold-close-margin", type=float, default=None)
    parser.add_argument("--gripper-force-hold-release-margin", type=float, default=None)
    parser.add_argument("--gripper-force-hold-grace-sec", type=float, default=None)
    parser.add_argument("--gripper-force-hold-engage-error-margin", type=float, default=None)
    parser.add_argument(
        "--arm-source",
        choices=["action", "state", "fk_cmd_pose"],
        default="action",
    )
    parser.add_argument(
        "--end-action",
        choices=["home", "hold"],
        default="home",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
