#!/usr/bin/env python3
"""Passively record G1 arm data for static payload/center-of-mass fitting."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HgLowCmd
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HgLowState

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.robot_control.robot_arm import G1_29_JointArmIndex


POSES = (
    (
        "neutral_supported",
        "右臂靠近躯干、肘部弯曲、前臂受支撑，手腕中立。",
    ),
    (
        "wrist_pitch_up_supported",
        "保持肩肘不动，右手腕向上翻；保持前臂受支撑。",
    ),
    (
        "wrist_pitch_down_supported",
        "保持肩肘不动，右手腕向下翻；保持前臂受支撑。",
    ),
    (
        "wrist_roll_out_supported",
        "保持肩肘不动，右手腕向外滚转；保持前臂受支撑。",
    ),
    (
        "wrist_roll_in_supported",
        "保持肩肘不动，右手腕向内滚转；保持前臂受支撑。",
    ),
    (
        "forward_low_supported",
        "在有人或支架托住前臂时，小幅向前伸，手腕中立。不要到力矩极限。",
    ),
    (
        "forward_pitch_up_supported",
        "保持上述受支撑前伸，仅向上翻手腕。不要到力矩极限。",
    ),
)
FORWARD_POSE_LABELS = {"forward_low_supported", "forward_pitch_up_supported"}
PULL_RELEASE_SEGMENTS = (
    (
        "pull_baseline",
        "保持右侧 VIVE 使能和 tracker 静止；不要接触机械臂，记录 1 秒基线。",
        1.0,
    ),
    (
        "pull_down_hold",
        "保持 tracker 静止，轻向下拉右前臂或腕部；不要前伸或超过约 5 cm，保持 1 秒。",
        1.0,
    ),
    (
        "pull_release_observe",
        "松开机械臂但保持 tracker 静止，观察 2 秒；有人在旁防护，异常立即松开右侧使能。",
        2.0,
    ),
)


def _arm_sample(state_msg, command_msg):
    motors = []
    for joint in G1_29_JointArmIndex:
        state = state_msg.motor_state[joint.value]
        command = None if command_msg is None else command_msg.motor_cmd[joint.value]
        motors.append(
            {
                "index": joint.value,
                "name": joint.name,
                "q": float(state.q),
                "dq": float(state.dq),
                "ddq": float(state.ddq),
                "tau_est": float(state.tau_est),
                "temperature": [int(value) for value in state.temperature],
                "vol": float(state.vol),
                "motorstate": int(state.motorstate),
                "command": None
                if command is None
                else {
                    "mode": int(command.mode),
                    "q": float(command.q),
                    "dq": float(command.dq),
                    "tau": float(command.tau),
                    "kp": float(command.kp),
                    "kd": float(command.kd),
                },
            }
        )
    return motors


def _write_segment(*, label, state_sub, command_sub, output, seconds, sample_hz, temperature_warning):
    deadline = time.monotonic() + seconds
    next_sample = time.monotonic()
    last_command = None
    samples = 0
    max_temperature = None

    while time.monotonic() < deadline:
        command = command_sub.Read(timeout=0.0)
        if command is not None:
            last_command = command
        state = state_sub.Read(timeout=0.02)
        if state is None or time.monotonic() < next_sample:
            continue

        now_ns = time.monotonic_ns()
        motors = _arm_sample(state, last_command)
        temperatures = [value for motor in motors for value in motor["temperature"]]
        segment_temperature = max(temperatures) if temperatures else None
        max_temperature = segment_temperature if max_temperature is None else max(max_temperature, segment_temperature)
        row = {
            "host_monotonic_ns": now_ns,
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "segment": label,
            "lowstate_tick": int(state.tick),
            "mode_pr": int(state.mode_pr),
            "mode_machine": int(state.mode_machine),
            "command_received": last_command is not None,
            "motors": motors,
        }
        output.write(json.dumps(row, separators=(",", ":")) + "\n")
        output.flush()
        samples += 1
        if segment_temperature is not None and segment_temperature >= temperature_warning:
            break
        next_sample = now_ns / 1e9 + 1.0 / sample_hz

    print(
        f"[PAYLOAD] {label}: samples={samples}, max_temperature={max_temperature}, "
        f"command_seen={last_command is not None}",
        flush=True,
    )
    if max_temperature is not None and max_temperature >= temperature_warning:
        print(
            f"[PAYLOAD][WARNING] {label}: temperature reached {max_temperature}; "
            "release the teleop deadman and inspect the robot before continuing.",
            flush=True,
        )
    return {
        "label": label,
        "samples": samples,
        "max_temperature": max_temperature,
        "command_seen": last_command is not None,
        "temperature_warning_reached": max_temperature is not None and max_temperature >= temperature_warning,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Passive G1 static-payload data collector. It never publishes DDS commands."
    )
    parser.add_argument("--network-interface", required=True, help="DDS interface, for example eno1")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain for the real robot")
    parser.add_argument("--state-topic", default="rt/lowstate")
    parser.add_argument(
        "--command-topic",
        default="rt/lowcmd",
        help="Use rt/arm_sdk only when teleop was started with --motion.",
    )
    parser.add_argument("--segment-sec", type=float, default=2.0)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--temperature-warning", type=int, default=70)
    parser.add_argument("--output-dir", default="data/payload_diagnostics")
    parser.add_argument(
        "--include-forward",
        action="store_true",
        help="Also collect the two small forward-reach poses; use only with physical support.",
    )
    parser.add_argument(
        "--pose",
        action="append",
        choices=[label for label, _ in POSES],
        help="Collect only this pose. Repeat to collect a small selected set.",
    )
    parser.add_argument(
        "--pull-release",
        action="store_true",
        help="Run the fixed 1s baseline, 1s gentle pull-down, then 2s release observation diagnostic.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the pose sequence without DDS access")
    args = parser.parse_args()

    if args.segment_sec <= 0.0 or args.sample_hz <= 0.0:
        parser.error("--segment-sec and --sample-hz must be positive")

    if args.pull_release and (args.pose or args.include_forward):
        parser.error("--pull-release cannot be combined with --pose or --include-forward")

    pose_sequence = tuple(pose for pose in POSES if args.include_forward or pose[0] not in FORWARD_POSE_LABELS)
    if args.pose:
        requested_poses = set(args.pose)
        if not args.include_forward and requested_poses & FORWARD_POSE_LABELS:
            parser.error("forward poses require --include-forward")
        pose_sequence = tuple(pose for pose in pose_sequence if pose[0] in requested_poses)
    segment_sequence = (
        PULL_RELEASE_SEGMENTS
        if args.pull_release
        else tuple((label, instruction, args.segment_sec) for label, instruction in pose_sequence)
    )

    if args.dry_run:
        for index, (label, instruction, seconds) in enumerate(segment_sequence, start=1):
            print(f"{index}. {label} ({seconds:.1f}s): {instruction}")
        return 0

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    data_path = output_dir / "samples.jsonl"
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "network_interface": args.network_interface,
        "domain": args.domain,
        "state_topic": args.state_topic,
        "command_topic": args.command_topic,
        "segment_sec": args.segment_sec,
        "sample_hz": args.sample_hz,
        "pull_release": args.pull_release,
        "planned_segments": [
            {"label": label, "instruction": instruction, "seconds": seconds}
            for label, instruction, seconds in segment_sequence
        ],
        "include_forward": args.include_forward,
    }

    ChannelFactoryInitialize(args.domain, networkInterface=args.network_interface)
    state_sub = ChannelSubscriber(args.state_topic, HgLowState)
    command_sub = ChannelSubscriber(args.command_topic, HgLowCmd)
    state_sub.Init()
    command_sub.Init()
    print(f"[PAYLOAD] writing {data_path}", flush=True)
    print("[PAYLOAD] passive collector only: it does not send stop, hold, or motor commands.", flush=True)

    summaries = []
    with data_path.open("w", encoding="utf-8") as output:
        for label, instruction, seconds in segment_sequence:
            input(f"\n[{label}] {instruction}\nPress Enter to collect: ")
            summaries.append(
                _write_segment(
                    label=label,
                    state_sub=state_sub,
                    command_sub=command_sub,
                    output=output,
                    seconds=seconds,
                    sample_hz=args.sample_hz,
                    temperature_warning=args.temperature_warning,
                )
            )
            if summaries[-1]["temperature_warning_reached"]:
                print("[PAYLOAD] stopping remaining segments after temperature warning.", flush=True)
                break

    metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
    metadata["segments"] = summaries
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[PAYLOAD] complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
