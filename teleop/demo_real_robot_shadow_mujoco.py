import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_G1_XML = REPO_ROOT / "assets/g1/g1_body29_hand14.xml"
DEFAULT_DEX1_XML = REPO_ROOT / "assets/.generated/g1_29dof_mode_15_with_dex1_1_scene.xml"
DEFAULT_G1D_XML = REPO_ROOT / "assets/g1_d/g1_d_scene.xml"

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

DEX1_JOINT_NAMES = {
    "left": ["left_dex1_finger_joint_1", "left_dex1_finger_joint_2"],
    "right": ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"],
}

DEX1_OPEN_Q = 0.0245
DEX1_CLOSED_Q = 0.0
DEX1_REAL_MAX_Q = 5.40


def bootstrap():
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    extras = [
        "/home/dx/miniconda3/envs/tv/lib/python3.10/site-packages",
        "/home/dx/miniconda3/envs/tv/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages",
        "/home/dx/miniconda3/envs/gmr/lib/python3.10/site-packages",
    ]
    for p in reversed(extras):
        if os.path.isdir(p):
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)


bootstrap()

import mujoco as mj
import mujoco.viewer as mjv
import glfw

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_

from teleop.robot_control.robot_arm import G1_29_JointArmIndex, kTopicLowState
from teleop.robot_control.robot_hand_unitree import kTopicGripperLeftState, kTopicGripperRightState
from teleop.utils.g1d_mujoco_builder import prepare_g1d_mobile_scene


def parse_args():
    parser = argparse.ArgumentParser(
        description="Passive MuJoCo shadow viewer for the real robot arm + Dex1 gripper state.",
    )
    parser.add_argument("--frequency", type=float, default=60.0, help="Viewer refresh frequency.")
    parser.add_argument("--network-interface", type=str, default=None, help="DDS network interface, e.g. eno1/wlo1.")
    parser.add_argument("--xml", type=str, default=None, help="Optional MuJoCo XML path override.")
    parser.add_argument("--ee", type=str, choices=["none", "dex1"], default="dex1", help="Whether to visualize Dex1 gripper state.")
    parser.add_argument(
        "--viewer-robot",
        type=str,
        choices=["g1", "g1_d", "g1_d_mobile"],
        default="g1_d_mobile",
        help='Shadow viewer model. "g1_d" matches the fixed-base G1D viewer, "g1_d_mobile" matches the movable G1D MuJoCo scene used by pure simulation.',
    )
    return parser.parse_args()


def resolve_xml_path(args):
    if args.xml:
        return Path(args.xml).expanduser().resolve()
    if args.viewer_robot == "g1_d_mobile":
        return prepare_g1d_mobile_scene(use_dex1=(args.ee == "dex1"))
    if args.viewer_robot == "g1_d" and DEFAULT_G1D_XML.exists():
        return DEFAULT_G1D_XML
    if args.ee == "dex1" and DEFAULT_DEX1_XML.exists():
        return DEFAULT_DEX1_XML
    return DEFAULT_G1_XML.resolve()


def joint_qpos_indices(model, joint_names):
    qpos_indices = []
    for name in joint_names:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint not found in model: {name}")
        qpos_indices.append(model.jnt_qposadr[jid])
    return qpos_indices


def joint_qpos_indices_if_present(model, joint_names):
    qpos_indices = []
    for name in joint_names:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            return None
        qpos_indices.append(model.jnt_qposadr[jid])
    return qpos_indices


def gripper_state_to_mujoco_q(gripper_q: float) -> float:
    open_ratio = float(np.clip(float(gripper_q) / DEX1_REAL_MAX_Q, 0.0, 1.0))
    return DEX1_CLOSED_Q + (DEX1_OPEN_Q - DEX1_CLOSED_Q) * open_ratio


class RealArmStateSubscriber:
    def __init__(self):
        self._lock = threading.Lock()
        self._arm_q = None
        self._running = True
        self._subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self._subscriber.Init()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        arm_ids = [int(member.value) for member in G1_29_JointArmIndex]
        while self._running:
            msg = self._subscriber.Read()
            if msg is not None:
                arm_q = np.array([msg.motor_state[idx].q for idx in arm_ids], dtype=float)
                with self._lock:
                    self._arm_q = arm_q
            time.sleep(0.002)

    def get(self):
        with self._lock:
            return None if self._arm_q is None else self._arm_q.copy()

    def ready(self):
        with self._lock:
            return self._arm_q is not None

    def close(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class Dex1StateSubscriber:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = None
        self._running = True
        self._left_sub = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self._right_sub = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self._left_sub.Init()
        self._right_sub.Init()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            left_msg = self._left_sub.Read()
            right_msg = self._right_sub.Read()
            if left_msg is not None and right_msg is not None:
                state = np.array([left_msg.states[0].q, right_msg.states[0].q], dtype=float)
                with self._lock:
                    self._state = state
            time.sleep(0.002)

    def get(self):
        with self._lock:
            return None if self._state is None else self._state.copy()

    def ready(self):
        with self._lock:
            return self._state is not None

    def close(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def main():
    args = parse_args()
    ChannelFactoryInitialize(0, networkInterface=args.network_interface)

    arm_state = RealArmStateSubscriber()
    gripper_state = Dex1StateSubscriber() if args.ee == "dex1" else None

    xml_path = resolve_xml_path(args)
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    base_qpos = data.qpos.copy()

    arm_joint_qpos_indices = joint_qpos_indices(model, ARM_JOINT_NAMES)
    dex1_qpos_indices = None
    if args.ee == "dex1":
        dex1_qpos_indices = {
            side: joint_qpos_indices_if_present(model, names)
            for side, names in DEX1_JOINT_NAMES.items()
        }
        if dex1_qpos_indices["left"] is None or dex1_qpos_indices["right"] is None:
            dex1_qpos_indices = None
            print("[DEX1] warning: current XML has no Dex1 joints, gripper state will not be visualized.")

    stop_requested = False
    dt = 1.0 / args.frequency

    def key_callback(keycode):
        nonlocal stop_requested
        if keycode == glfw.KEY_Q:
            stop_requested = True
            print("[EXIT] keyboard Q pressed, closing real-robot shadow viewer.")

    print(
        f"[SHADOW] starting passive real-robot MuJoCo viewer: "
        f"viewer_robot={args.viewer_robot}, ee={args.ee}, xml={xml_path}"
    )
    print("[SHADOW] this script only subscribes to DDS state; it does not publish commands or affect teleop.")
    print("[SHADOW] launch the real robot teleop normally, and this viewer will follow the live arm/gripper state.")
    if args.viewer_robot == "g1_d" and args.ee == "dex1" and dex1_qpos_indices is None:
        print(
            "[G1D_VIEWER] fixed-base G1D XML has no Dex1 joints in the current asset path; "
            "arm state will mirror normally, but Dex1 gripper will not animate."
        )
    if args.viewer_robot == "g1_d_mobile":
        print(
            "[G1D_MOBILE] using the same movable G1D MuJoCo scene path as pure simulation. "
            "In shadow mode, only real arm/gripper state is mirrored; base stays at its ready pose."
        )

    try:
        with mjv.launch_passive(
            model,
            data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            printed_arm_wait = False
            printed_gripper_wait = False
            while viewer.is_running():
                if stop_requested:
                    break

                live_arm_q = arm_state.get()
                live_gripper_q = gripper_state.get() if gripper_state is not None else None

                if live_arm_q is None:
                    if not printed_arm_wait:
                        print("[SHADOW] waiting for real robot arm state on rt/lowstate ...")
                        printed_arm_wait = True
                    viewer.sync()
                    time.sleep(dt)
                    continue

                if args.ee == "dex1" and gripper_state is not None and live_gripper_q is None:
                    if not printed_gripper_wait:
                        print("[SHADOW] waiting for Dex1 gripper state on rt/dex1/left/right/state ...")
                        printed_gripper_wait = True
                data.qpos[:] = base_qpos
                for idx, qpos_idx in enumerate(arm_joint_qpos_indices):
                    data.qpos[qpos_idx] = live_arm_q[idx]

                if dex1_qpos_indices is not None and live_gripper_q is not None:
                    left_q = gripper_state_to_mujoco_q(live_gripper_q[0])
                    right_q = gripper_state_to_mujoco_q(live_gripper_q[1])
                    for qpos_idx in dex1_qpos_indices["left"]:
                        data.qpos[qpos_idx] = left_q
                    for qpos_idx in dex1_qpos_indices["right"]:
                        data.qpos[qpos_idx] = right_q

                if data.qvel is not None and data.qvel.size:
                    data.qvel[:] = 0.0
                if data.ctrl is not None and data.ctrl.size:
                    data.ctrl[:] = 0.0

                mj.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)
    finally:
        arm_state.close()
        if gripper_state is not None:
            gripper_state.close()


if __name__ == "__main__":
    main()
