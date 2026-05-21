import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_G1_XML = REPO_ROOT / "assets/g1/g1_body29_hand14.xml"
DEFAULT_DEX1_XML = Path("/home/dx/unitree_ws/src/TWIST2/assets/.generated/g1_29dof_mode_15_with_dex1_1_scene.xml")
DEFAULT_G1D_XML = Path("/home/dx/unitree_ws/src/TWIST2/assets/g1_d/g1_d_scene.xml")
DEX1_OPEN_Q = 0.0245
DEX1_CLOSED_Q = 0.0
G1D_MAX_VX = 1.5
G1D_MAX_WZ = 0.6
G1D_MAX_COLUMN_VZ = 0.0765


def bootstrap():
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    extras = [
        "/home/dx/miniconda3/envs/tv/lib/python3.10/site-packages",
        "/home/dx/miniconda3/envs/tv/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages",
        "/home/dx/miniconda3/envs/gmr/lib/python3.10/site-packages",
        "/home/dx/pico_ws/src/XRoboToolkit-PC-Service-Pybind/build/lib.linux-x86_64-cpython-310",
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
import pinocchio as pin

from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.utils.arm_target_safety import limit_arm_joint_target_velocity
from teleop.utils.g1d_mujoco_builder import prepare_g1d_mobile_scene
from teleop.utils.xr_robotics_wrapper import XRRoboticsWrapper


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--ee", type=str, choices=["none", "dex1"], default="dex1")
    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument(
        "--viewer-robot",
        type=str,
        choices=["g1", "g1_d", "g1_d_mobile"],
        default="g1_d_mobile",
        help='MuJoCo viewer model. "g1_d" is the fixed-base G1D viewer. "g1_d_mobile" builds a local movable G1D MuJoCo scene with a free base and wheel actuators. Both keep the current G1 arm IK/control chain unchanged.',
    )
    parser.add_argument("--base-max-vx", type=float, default=0.15, help="Maximum simulated G1D chassis forward velocity command in m/s.")
    parser.add_argument("--base-max-wz", type=float, default=0.40, help="Maximum simulated G1D chassis yaw velocity command in rad/s.")
    parser.add_argument("--base-max-z", type=float, default=1.0, help="Maximum normalized G1D column height velocity command.")
    parser.add_argument("--base-stick-deadzone", type=float, default=0.12, help="Deadzone for MuJoCo G1D chassis thumbstick axes.")
    parser.add_argument(
        "--controller-deadman",
        type=str,
        choices=["grip", "none"],
        default="grip",
        help='Match real-robot teleop logic. "grip" means each arm/ee only moves while the same-side grip is held.',
    )
    parser.add_argument(
        "--max-arm-joint-speed",
        type=float,
        default=1.5,
        help="Outer-loop arm target speed limit in rad/s. Lower values reduce sudden jumps from teleop/IK.",
    )
    parser.add_argument(
        "--home-return-speed",
        type=float,
        default=0.6,
        help="Dedicated arm joint speed limit in rad/s used only while returning to the ready/home pose via left Y.",
    )
    parser.add_argument(
        "--head-reference-mode",
        type=str,
        choices=["head_coupled", "fixed_per_grip", "live_head_reference", "head_decoupled_live", "hybrid", "calibrated", "live"],
        default="live_head_reference",
        help='Reference-frame policy. "head_coupled" = fixed once after calibration. "fixed_per_grip" = freeze the current head translation at each grip takeover, release it when grip is released. "live_head_reference" = current head translation is always the live reference. "hybrid" = live when idle, frozen while gripping. Legacy aliases: calibrated=head_coupled, live/live_head_reference=head_decoupled_live semantics.',
    )
    parser.add_argument(
        "--controller-orientation-mode",
        type=str,
        choices=["absolute", "relative", "neutral"],
        default="absolute",
        help='Wrist orientation control. "absolute" matches the original main-branch controller feel most closely (controller orientation directly drives wrist orientation). "relative" uses controller rotation delta from the current grip anchor. "neutral" fixes wrist orientation.',
    )
    parser.add_argument(
        "--controller-mapping-mode",
        type=str,
        choices=["legacy_main", "anchored_safe"],
        default="anchored_safe",
        help='"legacy_main" reproduces the original main-branch controller mapping semantics as closely as possible. "anchored_safe" uses the newer grip-anchor based takeover-safe mapping.',
    )
    parser.add_argument(
        "--calibration-mode",
        type=str,
        choices=["manual", "auto"],
        default="manual",
        help='Calibration trigger in head_coupled/hybrid mode. "manual" waits for keyboard C inside the MuJoCo window; "auto" calibrates as soon as live pose arrives. fixed_per_grip/live_head_reference do not require manual calibration.',
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


def actuator_id_if_present(model, actuator_name):
    actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    return actuator_id if actuator_id >= 0 else None


def joint_id_if_present(model, joint_name):
    joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    return joint_id if joint_id >= 0 else None


def apply_deadzone(value: float, deadzone: float) -> float:
    return 0.0 if abs(value) < deadzone else value


def quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def yaw_from_quat_wxyz(quat_wxyz) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def trigger_value_to_close_ratio(trigger_value):
    return float(np.clip((10.0 - float(trigger_value)) / 10.0, 0.0, 1.0))


def apply_dex1_to_qpos(data, left_trigger_value, right_trigger_value, dex1_qpos_indices):
    left_close_ratio = trigger_value_to_close_ratio(left_trigger_value)
    right_close_ratio = trigger_value_to_close_ratio(right_trigger_value)

    left_q = DEX1_OPEN_Q + (DEX1_CLOSED_Q - DEX1_OPEN_Q) * left_close_ratio
    right_q = DEX1_OPEN_Q + (DEX1_CLOSED_Q - DEX1_OPEN_Q) * right_close_ratio

    for qpos_idx in dex1_qpos_indices["left"]:
        data.qpos[qpos_idx] = left_q
    for qpos_idx in dex1_qpos_indices["right"]:
        data.qpos[qpos_idx] = right_q

    return left_q, right_q


def reset_arm_ik_state(arm_ik, arm_q):
    arm_q = np.asarray(arm_q, dtype=float).copy()
    if hasattr(arm_ik, "init_data"):
        arm_ik.init_data = arm_q.copy()
    smooth_filter = getattr(arm_ik, "smooth_filter", None)
    if smooth_filter is not None:
        try:
            smooth_filter._data_queue = [arm_q.copy() for _ in range(smooth_filter._window_size)]
            smooth_filter._filtered_data = arm_q.copy()
        except Exception:
            pass


def get_robot_wrist_poses(arm_ik, arm_q):
    q = np.asarray(arm_q, dtype=float).copy()
    model = arm_ik.reduced_robot.model
    data = arm_ik.reduced_robot.data
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    left_pose = np.eye(4, dtype=float)
    right_pose = np.eye(4, dtype=float)
    left_se3 = data.oMf[arm_ik.L_hand_id]
    right_se3 = data.oMf[arm_ik.R_hand_id]
    left_pose[:3, :3] = left_se3.rotation
    left_pose[:3, 3] = left_se3.translation
    right_pose[:3, :3] = right_se3.rotation
    right_pose[:3, 3] = right_se3.translation
    return left_pose, right_pose


def main():
    args = parse_args()
    normalized_head_mode = "head_coupled" if args.head_reference_mode == "calibrated" else (
        "live_head_reference" if args.head_reference_mode in {"live", "head_decoupled_live"} else args.head_reference_mode
    )
    xml_path = resolve_xml_path(args)
    xr_wrapper = XRRoboticsWrapper(
        use_hand_tracking=False,
        head_reference_mode=args.head_reference_mode,
        controller_orientation_mode=args.controller_orientation_mode,
        controller_mapping_mode=args.controller_mapping_mode,
    )

    # G1_29_ArmIK uses relative asset paths internally. Force cwd to this teleop
    # directory so its ../assets/... references resolve exactly like the original
    # teleop_hand_and_arm.py launch flow.
    os.chdir(Path(__file__).resolve().parent)
    arm_ik = G1_29_ArmIK()

    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    base_qpos = data.qpos.copy()

    arm_q = np.zeros(14, dtype=float)
    arm_dq = np.zeros(14, dtype=float)
    left_dex1_q = DEX1_OPEN_Q
    right_dex1_q = DEX1_OPEN_Q
    arm_joint_qpos_indices = joint_qpos_indices(model, ARM_JOINT_NAMES)
    left_wheel_actuator_id = actuator_id_if_present(model, "left_wheel_drive")
    right_wheel_actuator_id = actuator_id_if_present(model, "right_wheel_drive")
    root_freejoint_id = joint_id_if_present(model, "g1d_freejoint")
    lz_mt_joint_id = joint_id_if_present(model, "LZ_mt_Joint")
    lz_it_joint_id = joint_id_if_present(model, "LZ_it_Joint")
    yaw_joint_id = joint_id_if_present(model, "Yaw_Joint")
    torso_joint_id = joint_id_if_present(model, "torso_Joint")
    left_wheel_joint_id = joint_id_if_present(model, "Left_Wheel_Joint")
    right_wheel_joint_id = joint_id_if_present(model, "Right_Wheel_Joint")
    root_qpos_idx = model.jnt_qposadr[root_freejoint_id] if root_freejoint_id is not None else None
    root_qvel_idx = model.jnt_dofadr[root_freejoint_id] if root_freejoint_id is not None else None
    left_wheel_qpos_idx = model.jnt_qposadr[left_wheel_joint_id] if left_wheel_joint_id is not None else None
    right_wheel_qpos_idx = model.jnt_qposadr[right_wheel_joint_id] if right_wheel_joint_id is not None else None
    lz_mt_qpos_idx = model.jnt_qposadr[lz_mt_joint_id] if lz_mt_joint_id is not None else None
    lz_it_qpos_idx = model.jnt_qposadr[lz_it_joint_id] if lz_it_joint_id is not None else None
    yaw_qpos_idx = model.jnt_qposadr[yaw_joint_id] if yaw_joint_id is not None else None
    torso_qpos_idx = model.jnt_qposadr[torso_joint_id] if torso_joint_id is not None else None
    lz_mt_range = model.jnt_range[lz_mt_joint_id].copy() if lz_mt_joint_id is not None else None
    lz_it_range = model.jnt_range[lz_it_joint_id].copy() if lz_it_joint_id is not None else None
    column_mt_target = float(base_qpos[lz_mt_qpos_idx]) if lz_mt_qpos_idx is not None else 0.0
    column_it_target = float(base_qpos[lz_it_qpos_idx]) if lz_it_qpos_idx is not None else 0.0
    yaw_target = float(base_qpos[yaw_qpos_idx]) if yaw_qpos_idx is not None else 0.0
    torso_target = float(base_qpos[torso_qpos_idx]) if torso_qpos_idx is not None else 0.0
    mobile_base_kinematic = (
        args.viewer_robot == "g1_d_mobile"
        and root_qpos_idx is not None
        and root_qvel_idx is not None
    )
    base_root_x = float(base_qpos[root_qpos_idx]) if root_qpos_idx is not None else 0.0
    base_root_y = float(base_qpos[root_qpos_idx + 1]) if root_qpos_idx is not None else 0.0
    base_root_z = float(base_qpos[root_qpos_idx + 2]) if root_qpos_idx is not None else 0.0
    base_root_yaw = (
        yaw_from_quat_wxyz(base_qpos[root_qpos_idx + 3: root_qpos_idx + 7])
        if root_qpos_idx is not None
        else 0.0
    )
    dex1_qpos_indices = None
    if args.ee == "dex1":
        dex1_qpos_indices = {
            side: joint_qpos_indices_if_present(model, names) for side, names in DEX1_JOINT_NAMES.items()
        }
        if dex1_qpos_indices["left"] is None or dex1_qpos_indices["right"] is None:
            dex1_qpos_indices = None
    prev_left_arm_enabled = None
    prev_right_arm_enabled = None
    prev_home_button_pressed = False
    home_return_active = False
    home_target_q = np.zeros(14, dtype=float)
    home_wait_grip_release = False
    prev_any_grip_pressed = False
    post_home_takeover_armed = False
    takeover_settle_frames = 0
    TAKEOVER_SETTLE_FRAMES = 0 if args.controller_mapping_mode == "legacy_main" else 2
    start_tracking = False
    stop_requested = False
    physics_substeps = max(1, int(round((1.0 / args.frequency) / float(model.opt.timestep))))

    print(
        f"XR-Robotics MuJoCo demo started: ee={args.ee}, viewer_robot={args.viewer_robot}, xml={xml_path}, "
        f"controller_deadman={args.controller_deadman}, "
        f"head_reference_mode={normalized_head_mode}, "
        f"controller_mapping_mode={args.controller_mapping_mode}, "
        f"controller_orientation_mode={args.controller_orientation_mode}, "
        f"calibration_mode={args.calibration_mode}, physics_substeps={physics_substeps}"
    )
    if left_wheel_actuator_id is not None and right_wheel_actuator_id is not None:
        print(
            "[G1D_BASE] enabled: left stick Y -> x(vx), left stick X -> yaw(wz), "
            f"right stick Y -> z (max_vx={args.base_max_vx:.2f} m/s, max_wz={args.base_max_wz:.2f} rad/s, max_z={args.base_max_z:.2f})"
        )
    if args.viewer_robot == "g1_d" and args.ee == "dex1" and dex1_qpos_indices is None:
        print(
            "[G1D_VIEWER] switched to local G1D viewer asset. "
            "Arm motion is still driven by the same G1 IK/control chain, "
            "but this viewer model has no Dex1 joints, so trigger input will not animate a Dex1 gripper."
        )
    if args.viewer_robot == "g1_d_mobile":
        print(
            "[G1D_MOBILE] using a locally generated movable G1D MuJoCo scene "
            "(free base + wheel actuators). This is not an official Unitree MJCF asset."
        )
        if mobile_base_kinematic:
            print(
                "[G1D_MOBILE] enabling kinematic planar base hold for teleop stability "
                "(locks roll/pitch so the two-wheel viewer does not tip over at startup)."
            )
    calibration_required = normalized_head_mode in {"head_coupled", "hybrid"}
    calibrated = not calibration_required
    calibration_requested = calibration_required and args.calibration_mode == "auto"
    printed_wait_live = False
    dt = 1.0 / args.frequency

    def key_callback(keycode):
        nonlocal calibration_requested, calibrated, start_tracking, stop_requested
        if keycode == glfw.KEY_C:
            calibrated = False
            calibration_requested = True
            print("[HEAD_REF] calibration requested from keyboard (C).")
        elif keycode == glfw.KEY_R:
            start_tracking = True
            print("---------------------🚀start Tracking🚀-------------------------")
        elif keycode == glfw.KEY_Q:
            stop_requested = True
            print("[EXIT] keyboard Q pressed, closing MuJoCo teleop.")

    try:
        with mjv.launch_passive(
            model,
            data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            print("[HOME] MuJoCo viewer initialized at home/ready pose.")
            print("🟢  Press [R] in the MuJoCo window to start syncing the robot with your movements.")
            if calibration_required:
                if args.calibration_mode == "manual":
                    print("🟣  After [R], move to your ready pose and press [C] to calibrate.")
                else:
                    print("🟣  Calibration will start automatically once live headset/controller pose is available.")
                if normalized_head_mode == "hybrid":
                    print("🟣  After calibration: idle=no grip -> reference auto-follows, gripping -> freeze+operate.")
                print("🟣  After calibration, press [C] anytime to recenter the reference.")
            print("🔴  Press [Q] in the MuJoCo window to stop and exit.")
            if calibration_required and args.calibration_mode == "manual":
                print("[HEAD_REF] manual mode: after pressing R, move headset/controllers to your ready pose, then press C in the MuJoCo window.")
            if normalized_head_mode == "hybrid" and args.controller_deadman != "grip":
                print('[HEAD_REF] warning: hybrid mode is most natural with --controller-deadman grip.')
            while viewer.is_running():
                if stop_requested:
                    break

                if not start_tracking:
                    data.qpos[:] = base_qpos
                    if mobile_base_kinematic:
                        base_root_x = float(base_qpos[root_qpos_idx])
                        base_root_y = float(base_qpos[root_qpos_idx + 1])
                        base_root_z = float(base_qpos[root_qpos_idx + 2])
                        base_root_yaw = yaw_from_quat_wxyz(base_qpos[root_qpos_idx + 3: root_qpos_idx + 7])
                    if data.ctrl is not None and data.ctrl.size:
                        data.ctrl[:] = 0.0
                    for idx, qpos_idx in enumerate(arm_joint_qpos_indices):
                        data.qpos[qpos_idx] = home_target_q[idx]
                    if dex1_qpos_indices is not None:
                        for qpos_idx in dex1_qpos_indices["left"]:
                            data.qpos[qpos_idx] = left_dex1_q
                        for qpos_idx in dex1_qpos_indices["right"]:
                            data.qpos[qpos_idx] = right_dex1_q
                    mj.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(dt)
                    continue

                if not calibrated:
                    if calibration_requested:
                        head_ref = xr_wrapper.calibrate_head_reference(require_live=True)
                        if head_ref is None:
                            if not printed_wait_live:
                                print("[HEAD_REF] waiting for live headset/controller pose before calibration...")
                                printed_wait_live = True
                            viewer.sync()
                            time.sleep(dt)
                            continue
                        print(
                            "[HEAD_REF] calibrated reference translation = "
                            f"({head_ref[0]:.3f}, {head_ref[1]:.3f}, {head_ref[2]:.3f})"
                        )
                        if normalized_head_mode == "hybrid":
                            print("[HEAD_REF] hybrid mode armed: idle=recenter follow, grip=freeze+operate.")
                        calibrated = True
                        calibration_requested = False
                        printed_wait_live = False
                    else:
                        viewer.sync()
                        time.sleep(dt)
                        continue

                current_left_wrist_pose, current_right_wrist_pose = get_robot_wrist_poses(arm_ik, arm_q)
                tele_data = xr_wrapper.get_tele_data(
                    current_left_robot_wrist_pose=current_left_wrist_pose,
                    current_right_robot_wrist_pose=current_right_wrist_pose,
                )
                if tele_data is not None:
                    left_stick_x = apply_deadzone(float(tele_data.left_ctrl_thumbstickValue[0]), args.base_stick_deadzone)
                    left_stick_y = apply_deadzone(float(tele_data.left_ctrl_thumbstickValue[1]), args.base_stick_deadzone)
                    right_stick_y = apply_deadzone(float(tele_data.right_ctrl_thumbstickValue[1]), args.base_stick_deadzone)

                    home_button_pressed = bool(tele_data.left_ctrl_bButton)
                    if home_button_pressed and not prev_home_button_pressed:
                        home_return_active = True
                        home_wait_grip_release = True
                        arm_dq[:] = 0.0
                        print("[HOME] left Y pressed -> returning both arms to ready/calibration pose with speed limit.")
                    prev_home_button_pressed = home_button_pressed

                    if args.controller_deadman == "grip":
                        left_arm_enabled = bool(tele_data.left_ctrl_squeeze)
                        right_arm_enabled = bool(tele_data.right_ctrl_squeeze)
                    else:
                        left_arm_enabled = True
                        right_arm_enabled = True
                    if home_wait_grip_release:
                        if not bool(tele_data.left_ctrl_squeeze) and not bool(tele_data.right_ctrl_squeeze):
                            home_wait_grip_release = False
                            if normalized_head_mode in {"head_coupled", "hybrid"}:
                                xr_wrapper.sync_reference_to_current_live_pose(require_live=False)
                                print("[HOME] reference synced to current live pose after home return.")
                            reset_arm_ik_state(arm_ik, arm_q)
                            print("[HOME] IK state reset at current home pose.")
                            post_home_takeover_armed = True
                            print("[HOME] grip released -> teleop re-enabled.")
                        else:
                            left_arm_enabled = False
                            right_arm_enabled = False

                    if left_arm_enabled != prev_left_arm_enabled or right_arm_enabled != prev_right_arm_enabled:
                        print(
                            f"[DEADMAN] left_enabled={left_arm_enabled} "
                            f"right_enabled={right_arm_enabled} "
                            f"(controller_deadman={args.controller_deadman})"
                        )
                        prev_left_arm_enabled = left_arm_enabled
                        prev_right_arm_enabled = right_arm_enabled

                    any_grip_pressed = bool(tele_data.left_ctrl_squeeze) or bool(tele_data.right_ctrl_squeeze)
                    takeover_rising_edge = any_grip_pressed and (not prev_any_grip_pressed)
                    if takeover_rising_edge:
                        takeover_settle_frames = TAKEOVER_SETTLE_FRAMES
                    zero_takeover_this_frame = takeover_settle_frames > 0
                    prev_any_grip_pressed = any_grip_pressed

                    current_lr_arm_q = arm_q.copy()
                    if zero_takeover_this_frame:
                        if post_home_takeover_armed and normalized_head_mode in {"head_coupled", "hybrid"}:
                            xr_wrapper.sync_reference_to_current_live_pose(require_live=False)
                        reset_arm_ik_state(arm_ik, current_lr_arm_q)
                        sol_q = current_lr_arm_q.copy()
                        post_home_takeover_armed = False
                        if takeover_rising_edge:
                            print(f"[TAKEOVER] grip rising edge -> zero-delta hold for {TAKEOVER_SETTLE_FRAMES} frames.")
                        takeover_settle_frames -= 1
                    elif home_return_active:
                        sol_q = home_target_q.copy()
                    elif left_arm_enabled or right_arm_enabled:
                        sol_q, _ = arm_ik.solve_ik(
                            tele_data.left_wrist_pose,
                            tele_data.right_wrist_pose,
                            current_lr_arm_q,
                            arm_dq,
                        )
                    else:
                        sol_q = current_lr_arm_q.copy()

                    if not left_arm_enabled:
                        sol_q[:7] = current_lr_arm_q[:7]
                    if not right_arm_enabled:
                        sol_q[-7:] = current_lr_arm_q[-7:]
                    if home_return_active:
                        sol_q = home_target_q.copy()

                    arm_q = limit_arm_joint_target_velocity(
                        sol_q,
                        current_lr_arm_q,
                        max_joint_speed=(args.home_return_speed if home_return_active else args.max_arm_joint_speed),
                        control_frequency=args.frequency,
                    )
                    for idx, qpos_idx in enumerate(arm_joint_qpos_indices):
                        data.qpos[qpos_idx] = arm_q[idx]
                    if dex1_qpos_indices is not None:
                        if left_arm_enabled:
                            left_dex1_q, _ = apply_dex1_to_qpos(
                                data,
                                tele_data.left_ctrl_triggerValue,
                                10.0,
                                {"left": dex1_qpos_indices["left"], "right": []},
                            )
                        else:
                            for qpos_idx in dex1_qpos_indices["left"]:
                                data.qpos[qpos_idx] = left_dex1_q

                        if right_arm_enabled:
                            _, right_dex1_q = apply_dex1_to_qpos(
                                data,
                                10.0,
                                tele_data.right_ctrl_triggerValue,
                                {"left": [], "right": dex1_qpos_indices["right"]},
                            )
                        else:
                            for qpos_idx in dex1_qpos_indices["right"]:
                                data.qpos[qpos_idx] = right_dex1_q

                    if left_wheel_actuator_id is not None and right_wheel_actuator_id is not None:
                        if home_return_active:
                            base_vx = 0.0
                            base_wz = 0.0
                            base_z = 0.0
                        else:
                            # Match the operator-facing semantics used in RUN.md:
                            #   left stick up/down   -> forward/backward
                            #   left stick left/right-> turn left/right
                            # For the local MuJoCo mobile viewer, the raw OpenXR
                            # stick signs already align with the desired body-frame
                            # motion after the kinematic base projection below.
                            base_vx = left_stick_y * min(args.base_max_vx, G1D_MAX_VX)
                            base_wz = -left_stick_x * min(args.base_max_wz, G1D_MAX_WZ)
                            base_z = right_stick_y * np.clip(args.base_max_z, 0.0, 1.0)

                        left_wheel_speed = (base_vx - 0.5 * base_wz * 0.4062) / 0.0848
                        right_wheel_speed = (base_vx + 0.5 * base_wz * 0.4062) / 0.0848
                        if mobile_base_kinematic:
                            base_root_yaw += base_wz * dt
                            base_root_x += math.cos(base_root_yaw) * base_vx * dt
                            base_root_y += math.sin(base_root_yaw) * base_vx * dt
                            if data.ctrl is not None and data.ctrl.size:
                                data.ctrl[left_wheel_actuator_id] = 0.0
                                data.ctrl[right_wheel_actuator_id] = 0.0
                            if left_wheel_qpos_idx is not None:
                                data.qpos[left_wheel_qpos_idx] += left_wheel_speed * dt
                            if right_wheel_qpos_idx is not None:
                                data.qpos[right_wheel_qpos_idx] += right_wheel_speed * dt
                            if root_qvel_idx is not None:
                                data.qvel[root_qvel_idx: root_qvel_idx + 6] = 0.0
                        else:
                            data.ctrl[left_wheel_actuator_id] = left_wheel_speed
                            data.ctrl[right_wheel_actuator_id] = right_wheel_speed

                        if lz_mt_qpos_idx is not None and lz_it_qpos_idx is not None:
                            total_delta = base_z * G1D_MAX_COLUMN_VZ * dt
                            half_delta = 0.5 * total_delta
                            column_mt_target = np.clip(column_mt_target + half_delta, lz_mt_range[0], lz_mt_range[1])
                            column_it_target = np.clip(column_it_target + half_delta, lz_it_range[0], lz_it_range[1])
                    for _ in range(physics_substeps):
                        if mobile_base_kinematic and root_qpos_idx is not None:
                            data.qpos[root_qpos_idx] = base_root_x
                            data.qpos[root_qpos_idx + 1] = base_root_y
                            data.qpos[root_qpos_idx + 2] = base_root_z
                            data.qpos[root_qpos_idx + 3: root_qpos_idx + 7] = quat_wxyz_from_yaw(base_root_yaw)
                            if root_qvel_idx is not None:
                                data.qvel[root_qvel_idx: root_qvel_idx + 6] = 0.0
                        if lz_mt_qpos_idx is not None:
                            data.qpos[lz_mt_qpos_idx] = column_mt_target
                        if lz_it_qpos_idx is not None:
                            data.qpos[lz_it_qpos_idx] = column_it_target
                        if yaw_qpos_idx is not None:
                            data.qpos[yaw_qpos_idx] = yaw_target
                        if torso_qpos_idx is not None:
                            data.qpos[torso_qpos_idx] = torso_target
                        for idx, qpos_idx in enumerate(arm_joint_qpos_indices):
                            data.qpos[qpos_idx] = arm_q[idx]
                        if dex1_qpos_indices is not None:
                            for qpos_idx in dex1_qpos_indices["left"]:
                                data.qpos[qpos_idx] = left_dex1_q
                            for qpos_idx in dex1_qpos_indices["right"]:
                                data.qpos[qpos_idx] = right_dex1_q
                        mj.mj_forward(model, data)
                        mj.mj_step(model, data)
                    if home_return_active and np.all(np.abs(arm_q - home_target_q) < 0.03):
                        home_return_active = False
                        print("[HOME] reached ready/calibration pose. Waiting for both grips to release before teleop resumes.")
                else:
                    if left_wheel_actuator_id is not None and right_wheel_actuator_id is not None:
                        data.ctrl[left_wheel_actuator_id] = 0.0
                        data.ctrl[right_wheel_actuator_id] = 0.0

                viewer.sync()
                time.sleep(dt)
    finally:
        xr_wrapper.close()


if __name__ == "__main__":
    main()
