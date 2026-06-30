#!/usr/bin/env python3
"""LeRobot v2 three-camera recording entrypoint.

This entrypoint reuses the existing teleop main loop with only the
recording stack swapped to the LeRobot v2 writer. The legacy
`teleop_hand_and_arm.py` file remains untouched.
"""

from __future__ import annotations

import pathlib


SOURCE_PATH = pathlib.Path(__file__).with_name("teleop_hand_and_arm.py")


def _rewrite_source(source: str) -> str:
    replacements = [
        (
            "from teleop.recording.episode_writer import EpisodeWriter, ZMQRawCameraReceiver",
            "from teleop.recording.lerobot_v2_writer import EpisodeWriter, ZMQRawCameraReceiver",
        ),
        (
            "parser.add_argument('--record-arm-repr', type=str, choices=['qpos', 'pose', 'both'], default='qpos',",
            "parser.add_argument('--record-arm-repr', type=str, choices=['qpos', 'pose', 'both'], default='both',",
        ),
        (
            "            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),\n"
            "                                     task_goal = args.task_goal,\n"
            "                                     task_desc = args.task_desc,\n"
            "                                     task_steps = args.task_steps,\n"
            "                                     frequency = args.frequency,\n"
            "                                     image_size = [args.camera_width, args.camera_height],\n"
            "                                     rerun_log = not args.headless)",
            "            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),\n"
            "                                     arm_ik = arm_ik,\n"
            "                                     task_goal = args.task_goal,\n"
            "                                     task_desc = args.task_desc,\n"
            "                                     task_steps = args.task_steps,\n"
            "                                     frequency = args.frequency,\n"
            "                                     image_size = [args.camera_width, args.camera_height],\n"
            "                                     rerun_log = not args.headless)",
        ),
    ]
    rewritten = source
    for old, new in replacements:
        if old not in rewritten:
            raise RuntimeError(f"Expected source pattern not found: {old[:80]}...")
        rewritten = rewritten.replace(old, new)
    return rewritten


def main() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    rewritten = _rewrite_source(source)
    exec_globals = {
        "__name__": "__main__",
        "__file__": str(SOURCE_PATH),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(rewritten, str(SOURCE_PATH), "exec"), exec_globals, exec_globals)


if __name__ == "__main__":
    main()
