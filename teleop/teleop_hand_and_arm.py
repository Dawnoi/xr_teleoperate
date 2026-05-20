import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Prefer the conda-provided pinocchio/casadi stack over the ROS site-packages
# shadowing, otherwise `from pinocchio import casadi` may fail even inside the
# intended tv environment.
preferred_python_paths = [
    "/home/dx/miniconda3/envs/tv/lib/python3.10/site-packages",
    "/home/dx/miniconda3/envs/tv/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages",
]
for _p in reversed(preferred_python_paths):
    if os.path.isdir(_p):
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)

import pinocchio as pin

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.g1d_agv_bridge import G1DAgvBridge
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from teleop.utils.xr_robotics_wrapper import XRRoboticsWrapper
from teleop.utils.arm_target_safety import limit_arm_joint_target_velocity
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
RECENTER       = False  # Recalibrate fixed head reference for controller-space teleop
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE, RECENTER
    if key == 'r':
        START = True
    elif key == 'c':
        RECENTER = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


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

if __name__ == '__main__':
    arm_ctrl = None
    tv_wrapper = None
    listen_keyboard_thread = None
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    parser.add_argument('--controller-deadman', type=str, choices=['grip', 'none'], default='grip',
                        help='Controller safety enable logic. "grip" means each arm/ee only moves while the same-side grip is held.')
    parser.add_argument('--max-arm-joint-speed', type=float, default=1.5,
                        help='Outer-loop arm target speed limit in rad/s. Lower values reduce sudden jumps from teleop/IK.')
    parser.add_argument('--home-return-speed', type=float, default=0.6,
                        help='Dedicated arm joint speed limit in rad/s used only while returning to the ready/home pose via left Y.')
    parser.add_argument('--base-max-vx', type=float, default=0.3,
                        help='Maximum commanded chassis x velocity in m/s from the left thumbstick Y axis when --motion is enabled.')
    parser.add_argument('--base-max-vy', type=float, default=0.3,
                        help='Maximum commanded chassis y velocity in m/s from the left thumbstick X axis when --motion is enabled.')
    parser.add_argument('--base-max-wz', type=float, default=0.3,
                        help='Maximum commanded chassis angular z velocity in rad/s from the right thumbstick X axis when --motion is enabled.')
    parser.add_argument('--base-max-z', type=float, default=1.0,
                        help='Maximum normalized G1D AGV HeightAdjust command from the right thumbstick Y axis when --base-controller g1d_agv is enabled.')
    parser.add_argument('--base-stick-deadzone', type=float, default=0.08,
                        help='Thumbstick deadzone for chassis motion commands in motion mode.')
    parser.add_argument('--base-controller', type=str, choices=['none', 'g1d_agv'], default='none',
                        help='Optional chassis control path. Use g1d_agv for the official G1D AGV API while keeping arm control in debug mode.')
    parser.add_argument('--head-reference-mode', type=str, choices=['head_coupled', 'fixed_per_grip', 'live_head_reference', 'head_decoupled_live', 'hybrid', 'calibrated', 'live'], default='live_head_reference',
                        help='Controller wrist reference frame. "head_coupled" = fixed once after calibration. "fixed_per_grip" = freeze the current head translation at each grip takeover, release it when grip is released. "live_head_reference" = current head translation is always the live reference. "hybrid" = live when idle, frozen while gripping. Legacy aliases: calibrated=head_coupled, live/head_decoupled_live=live_head_reference semantics.')
    parser.add_argument('--controller-orientation-mode', type=str, choices=['absolute', 'relative', 'neutral'], default='absolute',
                        help='Wrist orientation control. "absolute" matches the original main-branch controller feel most closely (controller orientation directly drives wrist orientation). "relative" uses controller rotation delta from the current grip anchor. "neutral" fixes wrist orientation.')
    parser.add_argument('--controller-mapping-mode', type=str, choices=['legacy_main', 'anchored_safe'], default='anchored_safe',
                        help='"legacy_main" reproduces the original main-branch controller mapping semantics as closely as possible. "anchored_safe" uses the newer grip-anchor based takeover-safe mapping.')
    parser.add_argument('--calibration-mode', type=str, choices=['manual', 'auto'], default='manual',
                        help='Calibration trigger in head_coupled/hybrid mode. "manual" waits for key c after r; "auto" calibrates once live pose data is available. fixed_per_grip/live_head_reference do not require manual calibration.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.debug(f"args: {args}")
    normalized_head_mode = "head_coupled" if args.head_reference_mode == "calibrated" else (
        "live_head_reference" if args.head_reference_mode in {"live", "head_decoupled_live"} else args.head_reference_mode
    )
    if args.base_controller == "g1d_agv" and args.motion:
        raise ValueError("Do not combine --base-controller g1d_agv with --motion. G1D AGV base control should run with the arms kept in debug mode.")

    def compute_arm_gravity_tauff(arm_ik_obj, arm_q):
        try:
            model = arm_ik_obj.reduced_robot.model
            data = arm_ik_obj.reduced_robot.data
            q = np.asarray(arm_q, dtype=float).copy()
            dq = np.zeros(model.nv, dtype=float)
            ddq = np.zeros(model.nv, dtype=float)
            return pin.rnea(model, data, q, dq, ddq)
        except Exception as e:
            logger_mp.warning(f"[HOLD_TAUFF] failed to compute gravity compensation, fallback to zeros: {e}")
            return np.zeros_like(np.asarray(arm_q, dtype=float))

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        tv_wrapper = XRRoboticsWrapper(
            use_hand_tracking=args.input_mode == "hand",
            head_reference_mode=args.head_reference_mode,
            controller_orientation_mode=args.controller_orientation_mode,
            controller_mapping_mode=args.controller_mapping_mode,
        )
        logger_mp.info("Using XR-Robotics as the only Pico input source.")
        
        agv_bridge = None
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
                logger_mp.info(
                    "[BASE_CTRL] enabled: left stick Y -> x(vx), left stick X -> y(vy), "
                    "right stick X -> yaw(wz) "
                    f"(max_vx={args.base_max_vx:.2f}, max_vy={args.base_max_vy:.2f}, max_wz={args.base_max_wz:.2f})"
                )
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")
            if args.base_controller == "g1d_agv":
                agv_bridge = G1DAgvBridge(network_interface=args.network_interface, auto_build=True)
                logger_mp.info(
                    "[BASE_CTRL] enabled: official G1D AgvClient bridge "
                    f"(left stick Y -> x(vx), left stick X -> yaw(wz), right stick Y -> z, "
                    f"max_vx={args.base_max_vx:.2f}, "
                    f"max_wz={args.base_max_wz:.2f}, max_z={args.base_max_z:.2f})"
                )
                logger_mp.warning(
                    "[BASE_CTRL] G1D AgvClient limitation: official API currently ignores vy, "
                    "so this path maps left stick X to in-place yaw instead of lateral strafing."
                )

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        elif args.arm == "H2":
            arm_ik = H2_ArmIK()
            arm_ctrl = H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

        # end-effector
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("Move arms to home pose before entering teleop wait state...")
        arm_ctrl.ctrl_dual_arm_go_home()

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        calibration_required = normalized_head_mode in {"head_coupled", "hybrid"}
        if calibration_required:
            if args.calibration_mode == "manual":
                logger_mp.info("🟣  After [r], move to your ready pose and press [c] to calibrate.")
            else:
                logger_mp.info("🟣  Calibration will start automatically once live headset/controller pose is available.")
            if normalized_head_mode == "hybrid":
                logger_mp.info("🟣  After calibration: idle=no grip -> reference auto-follows, gripping -> reference freezes for operation.")
            logger_mp.info("🟣  After calibration, press [c] anytime to recenter the reference.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()
        current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
        current_hold_q = current_lr_arm_q.copy()
        current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
        arm_ctrl.ctrl_dual_arm(current_hold_q.copy(), current_hold_tauff.copy())
        calibration_hold_q = current_lr_arm_q.copy()
        calibration_hold_tauff = current_hold_tauff.copy()
        calibrated = not calibration_required
        calibration_requested = calibration_required and args.calibration_mode == "auto"
        printed_wait_live = False
        if calibration_required and args.calibration_mode == "manual":
            logger_mp.info("[HEAD_REF] waiting for manual calibration request. Press [c] when you are ready.")

        prev_left_arm_enabled = None
        prev_right_arm_enabled = None
        prev_home_button_pressed = False
        home_return_active = False
        home_target_q = np.zeros_like(current_hold_q)
        home_wait_grip_release = False
        prev_any_grip_pressed = False
        post_home_takeover_armed = False
        takeover_settle_frames = 0
        TAKEOVER_SETTLE_FRAMES = 0 if args.controller_mapping_mode == "legacy_main" else 2

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            if RECENTER and calibration_required:
                RECENTER = False
                calibrated = False
                calibration_requested = True
                calibration_hold_q = arm_ctrl.get_current_dual_arm_q().copy()
                calibration_hold_tauff = compute_arm_gravity_tauff(arm_ik, calibration_hold_q)
                logger_mp.info("[HEAD_REF] calibration requested from keyboard (c).")

            if calibration_required and not calibrated:
                if calibration_requested:
                    head_ref = tv_wrapper.calibrate_head_reference(require_live=True)
                    if head_ref is None:
                        if not printed_wait_live:
                            logger_mp.info("[HEAD_REF] waiting for live headset/controller pose before calibration...")
                            printed_wait_live = True
                        arm_ctrl.ctrl_dual_arm(calibration_hold_q.copy(), calibration_hold_tauff.copy())
                        time.sleep(0.01)
                        continue
                    logger_mp.info(
                        f"[HEAD_REF] calibrated reference translation = "
                        f"({head_ref[0]:.3f}, {head_ref[1]:.3f}, {head_ref[2]:.3f})"
                    )
                    if normalized_head_mode == "hybrid":
                        logger_mp.info("[HEAD_REF] hybrid mode armed: idle=no grip -> follow reference, gripping -> freeze+operate.")
                    calibrated = True
                    calibration_requested = False
                    printed_wait_live = False
                    current_hold_q = calibration_hold_q.copy()
                    current_hold_tauff = calibration_hold_tauff.copy()
                else:
                    arm_ctrl.ctrl_dual_arm(calibration_hold_q.copy(), calibration_hold_tauff.copy())
                    time.sleep(0.01)
                    continue

            # get current robot state data first so XR wrapper can anchor each new
            # grip takeover to the *current* robot wrist pose instead of a fixed home pose.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            current_left_wrist_pose, current_right_wrist_pose = get_robot_wrist_poses(arm_ik, current_lr_arm_q)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data(
                current_left_robot_wrist_pose=current_left_wrist_pose,
                current_right_robot_wrist_pose=current_right_wrist_pose,
            )
            if tele_data is None:
                time.sleep(0.01)
                continue

            home_button_pressed = bool(tele_data.left_ctrl_bButton)
            if home_button_pressed and not prev_home_button_pressed:
                logger_mp.info("[HOME] left Y pressed -> returning both arms to ready/calibration pose with speed limit.")
                home_return_active = True
                home_wait_grip_release = True
            prev_home_button_pressed = home_button_pressed

            if args.input_mode == "controller" and args.controller_deadman == "grip":
                left_arm_enabled = bool(tele_data.left_ctrl_squeeze)
                right_arm_enabled = bool(tele_data.right_ctrl_squeeze)
            else:
                left_arm_enabled = True
                right_arm_enabled = True
            if home_wait_grip_release:
                if not bool(tele_data.left_ctrl_squeeze) and not bool(tele_data.right_ctrl_squeeze):
                    home_wait_grip_release = False
                    if normalized_head_mode in {"head_coupled", "hybrid"}:
                        tv_wrapper.sync_reference_to_current_live_pose(require_live=False)
                        logger_mp.info("[HOME] reference synced to current live pose after home return.")
                    reset_arm_ik_state(arm_ik, current_hold_q)
                    logger_mp.info("[HOME] IK state reset at current home pose.")
                    post_home_takeover_armed = True
                    logger_mp.info("[HOME] grip released -> teleop re-enabled.")
                else:
                    left_arm_enabled = False
                    right_arm_enabled = False

            if left_arm_enabled != prev_left_arm_enabled or right_arm_enabled != prev_right_arm_enabled:
                logger_mp.info(
                    f"[DEADMAN] left_enabled={left_arm_enabled} right_enabled={right_arm_enabled} "
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

            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.input_mode == "controller":
                if left_arm_enabled:
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_ctrl_triggerValue
                if right_arm_enabled:
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                    time.sleep(0.05)
                    continue

                left_stick_x = float(tele_data.left_ctrl_thumbstickValue[0])
                left_stick_y = float(tele_data.left_ctrl_thumbstickValue[1])
                right_stick_x = float(tele_data.right_ctrl_thumbstickValue[0])

                def apply_deadzone(value: float, deadzone: float) -> float:
                    return 0.0 if abs(value) < deadzone else value

                if home_return_active:
                    base_vx = 0.0
                    base_vy = 0.0
                    base_wz = 0.0
                else:
                    left_stick_x = apply_deadzone(left_stick_x, args.base_stick_deadzone)
                    left_stick_y = apply_deadzone(left_stick_y, args.base_stick_deadzone)
                    right_stick_x = apply_deadzone(right_stick_x, args.base_stick_deadzone)

                    # Match Unitree loco semantics:
                    #   left stick  -> body-frame x/y velocity
                    #   right stick -> body-frame angular z velocity
                    base_vx = -left_stick_y * args.base_max_vx
                    base_vy = -left_stick_x * args.base_max_vy
                    base_wz = -right_stick_x * args.base_max_wz

                loco_wrapper.Move(base_vx, base_vy, base_wz)
            elif args.input_mode == "controller" and args.base_controller == "g1d_agv":
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True

                left_stick_x = float(tele_data.left_ctrl_thumbstickValue[0])
                left_stick_y = float(tele_data.left_ctrl_thumbstickValue[1])
                right_stick_x = float(tele_data.right_ctrl_thumbstickValue[0])
                right_stick_y = float(tele_data.right_ctrl_thumbstickValue[1])

                def apply_deadzone(value: float, deadzone: float) -> float:
                    return 0.0 if abs(value) < deadzone else value

                left_stick_x = apply_deadzone(left_stick_x, args.base_stick_deadzone)
                left_stick_y = apply_deadzone(left_stick_y, args.base_stick_deadzone)
                right_stick_x = apply_deadzone(right_stick_x, args.base_stick_deadzone)
                right_stick_y = apply_deadzone(right_stick_y, args.base_stick_deadzone)

                if home_return_active:
                    base_vx = 0.0
                    base_vy = 0.0
                    base_wz = 0.0
                    base_z = 0.0
                else:
                    # Backported from the official unitree_sdk2 G1D example path:
                    #   AgvClient.Move(vx, vy, vyaw)
                    #   AgvClient.HeightAdjust(vz)
                    # Note: the official G1D AGV header comments that vy is currently
                    # ignored by the AGV side. For practical teleop on G1D, we therefore
                    # map left stick X to yaw so it behaves like the official remote:
                    #   left stick up/down -> forward/backward
                    #   left stick left/right -> in-place turn
                    #   right stick up/down -> column height adjust
                    base_vx = left_stick_y * args.base_max_vx
                    base_vy = 0.0
                    base_wz = -left_stick_x * args.base_max_wz
                    base_z = right_stick_y * args.base_max_z

                if agv_bridge is not None:
                    agv_bridge.move(base_vx, base_vy, base_wz)
                    agv_bridge.height_adjust(base_z)

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            if zero_takeover_this_frame:
                if post_home_takeover_armed and normalized_head_mode in {"head_coupled", "hybrid"}:
                    tv_wrapper.sync_reference_to_current_live_pose(require_live=False)
                reset_arm_ik_state(arm_ik, current_lr_arm_q)
                sol_q = current_lr_arm_q.copy()
                sol_tauff = compute_arm_gravity_tauff(arm_ik, sol_q)
                post_home_takeover_armed = False
                if takeover_rising_edge:
                    logger_mp.info(f"[TAKEOVER] grip rising edge -> zero-delta hold for {TAKEOVER_SETTLE_FRAMES} frames.")
                takeover_settle_frames -= 1
            elif home_return_active:
                sol_q = home_target_q.copy()
                sol_tauff = compute_arm_gravity_tauff(arm_ik, sol_q)
            elif left_arm_enabled or right_arm_enabled:
                time_ik_start = time.time()
                sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
                time_ik_end = time.time()
                logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            else:
                sol_q = current_hold_q.copy()
                sol_tauff = current_hold_tauff.copy()

            if (not left_arm_enabled) and (not home_return_active):
                sol_q[:7] = current_hold_q[:7]
                sol_tauff[:7] = current_hold_tauff[:7]
            if (not right_arm_enabled) and (not home_return_active):
                sol_q[-7:] = current_hold_q[-7:]
                sol_tauff[-7:] = current_hold_tauff[-7:]

            sol_q = limit_arm_joint_target_velocity(
                sol_q,
                current_lr_arm_q,
                max_joint_speed=(args.home_return_speed if home_return_active else args.max_arm_joint_speed),
                control_frequency=args.frequency,
            )
            sol_tauff = compute_arm_gravity_tauff(arm_ik, sol_q)
            current_hold_q = sol_q.copy()
            current_hold_tauff = sol_tauff.copy()

            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            if home_return_active and np.all(np.abs(sol_q - home_target_q) < 0.05):
                home_return_active = False
                calibration_hold_q = current_hold_q.copy()
                calibration_hold_tauff = current_hold_tauff.copy()
                logger_mp.info("[HOME] reached ready/calibration pose. Waiting for both grips to release before teleop resumes.")

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                if listen_keyboard_thread is not None:
                    listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close XR wrapper: {e}")

        try:
            if 'agv_bridge' in locals() and agv_bridge is not None:
                agv_bridge.close()
        except Exception as e:
            logger_mp.error(f"Failed to close G1D AGV bridge: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
