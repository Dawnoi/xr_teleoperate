import time
from multiprocessing import Value, Array, Lock
import threading
from collections import deque
import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
teleop_dir = os.path.dirname(current_dir)
repo_root = os.path.dirname(teleop_dir)
sys.path.append(repo_root)

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

from teleop.real.args import parse_args
from data_pipeline.recording.alignment import (
    append_timed_sample,
    nearest_timed_sample,
    camera_meta_monotonic_ns,
    camera_frame_identity,
    build_alignment_timestamp_entry,
    timed_buffer_bounds,
    interpolate_timed_sample_strict,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
from data_pipeline.recording.episode_writer import EpisodeWriter, ZMQRawCameraReceiver
from core.control.g1d_agv_bridge import G1DAgvBridge
from core.control.motion_switcher import MotionSwitcher, LocoClientWrapper
from core.input.teleop_input_provider import create_teleop_input_provider, validate_lerobot_offline_episode
from core.camera.local_camera import LocalCameraStream
from teleop.debug.latency_trace import SimpleLatencyTracker
from teleop.debug.timing_debugger import TimingDebugger
from teleop.runtime.operator_runtime import OperatorRuntime
from teleop.control_flow.base_command import apply_base_command
from teleop.control_flow.arm_command_pipeline import build_arm_command
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
RECORD_CANCEL  = False  # Cancel active/armed recording without saving
RECENTER       = False  # Recalibrate fixed head reference for controller-space teleop
operator_runtime = None
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
    global STOP, START, RECORD_TOGGLE, RECORD_CANCEL, RECENTER, operator_runtime
    if key == 'r':
        START = True
    elif key == 'c':
        RECENTER = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        if operator_runtime is not None and not operator_runtime.record_enabled:
            operator_runtime.warn_record_disabled("keyboard shortcut [s]")
        else:
            RECORD_TOGGLE = True
    elif key == 'v' and START == True:
        if operator_runtime is not None and not operator_runtime.record_enabled:
            operator_runtime.warn_record_disabled("keyboard shortcut [v]")
        else:
            RECORD_CANCEL = True
    elif key == 'h' and START == True:
        if operator_runtime is None:
            logger_mp.warning("[HOME] ignored keyboard shortcut [h]: operator runtime is not initialized.")
        else:
            operator_runtime.request_home("keyboard")
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

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


def pose_matrix_to_record(pose_mat):
    pose = np.asarray(pose_mat, dtype=float)
    from scipy.spatial.transform import Rotation as R
    rpy = R.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=False)
    return {
        "position": pose[:3, 3].tolist(),
        "rpy": rpy.tolist(),
        "rotation_matrix": pose[:3, :3].tolist(),
        "matrix4x4": pose.tolist(),
    }


if __name__ == '__main__':
    arm_ctrl = None
    tv_wrapper = None
    listen_keyboard_thread = None
    gripper_ctrl = None
    loco_wrapper = None
    recorder = None
    sim_state_subscriber = None
    exit_go_home = True
    exit_home_hold_sec = 5.0
    head_camera = None
    left_camera = None
    right_camera = None
    head_remote_camera = None
    left_remote_camera = None
    right_remote_camera = None
    args = parse_args()
    logger_mp.debug(f"args: {args}")
    operator_runtime = OperatorRuntime(record_enabled=bool(args.record))
    normalized_head_mode = "head_coupled" if args.head_reference_mode == "calibrated" else (
        "live_head_reference" if args.head_reference_mode in {"live", "head_decoupled_live"} else args.head_reference_mode
    )
    workspace_limit_enabled = not args.disable_arm_workspace_limit
    workspace_mode = args.arm_workspace_mode
    workspace_min = np.asarray(args.arm_workspace_min, dtype=float)
    workspace_max = np.asarray(args.arm_workspace_max, dtype=float)
    tapered_workspace_params = {
        "z_min": float(args.arm_workspace_z_min),
        "z_max": float(args.arm_workspace_z_max),
        "x_min": float(args.arm_workspace_x_min),
        "x_max_low": float(args.arm_workspace_x_max_low),
        "x_max_high": float(args.arm_workspace_x_max_high),
        "y_max_low": float(args.arm_workspace_y_max_low),
        "y_max_high": float(args.arm_workspace_y_max_high),
    }
    timing_debugger = TimingDebugger(enabled=args.timing_debug, interval_sec=args.timing_debug_interval)
    state_history_size = max(32, int(args.frequency * 6))
    action_history_size = max(32, int(args.frequency * 6))
    state_history = deque(maxlen=state_history_size)
    action_history = deque(maxlen=action_history_size)
    pending_record_samples = deque()
    record_start_monotonic_ns = None
    camera_align_max_delta_ns = int(max(20_000_000, (1.5 / max(args.frequency, 1e-6)) * 1e9))
    state_align_max_delta_ns = int(max(20_000_000, (1.25 / max(args.frequency, 1e-6)) * 1e9))
    action_align_max_delta_ns = state_align_max_delta_ns
    state_nearest_fallback_max_delta_ns = int(max(80_000_000, (2.5 / max(args.frequency, 1e-6)) * 1e9))
    action_nearest_fallback_max_delta_ns = state_nearest_fallback_max_delta_ns
    record_future_wait_timeout_ns = int(max(120_000_000, (2.0 / max(args.frequency, 1e-6)) * 1e9))
    pending_sample_timeout_ns = int(1_000_000_000)
    control_dt = 1.0 / max(args.frequency, 1e-6)
    if args.base_controller == "g1d_agv" and args.motion:
        raise ValueError("Do not combine --base-controller g1d_agv with --motion. G1D AGV base control should run with the arms kept in debug mode.")
    if args.input_provider == "lerobot_offline":
        validate_lerobot_offline_episode(
            args.offline_replay_dataset_root,
            args.offline_replay_episode_index,
            args.offline_replay_arm_source,
        )

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

    def maybe_open_local_camera(name: str, camera_id: int):
        if int(camera_id) < 0:
            return None
        try:
            return LocalCameraStream(
                name=name,
                camera_id=int(camera_id),
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                fourcc=args.camera_fourcc,
                buffer_size=args.camera_buffer_size,
            )
        except Exception as e:
            logger_mp.warning(f"[CAM] failed to open local camera '{name}' (id={camera_id}): {e}")
            return None

    def maybe_open_remote_camera(name: str, endpoint: str):
        endpoint = str(endpoint or "").strip()
        if not endpoint:
            return None
        try:
            return ZMQRawCameraReceiver(endpoint=endpoint, name=name)
        except Exception as e:
            logger_mp.warning(f"[CAM] failed to connect remote camera '{name}' ({endpoint}): {e}")
            return None

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # keyboard communication mode
        listen_keyboard_thread = threading.Thread(target=listen_keyboard,
                                                  kwargs={"on_press": on_press, "until": None, "sequential": False,},
                                                  daemon=True)
        listen_keyboard_thread.start()

        if workspace_limit_enabled:
            if workspace_mode == "box":
                logger_mp.info(
                    "[ARM_WORKSPACE] enabled: forward box, min=(%.3f, %.3f, %.3f), max=(%.3f, %.3f, %.3f)",
                    workspace_min[0], workspace_min[1], workspace_min[2],
                    workspace_max[0], workspace_max[1], workspace_max[2],
                )
            else:
                logger_mp.info(
                    "[ARM_WORKSPACE] enabled: tapered prism, z=[%.3f, %.3f], x_min=%.3f, "
                    "x_max(low->high)=(%.3f -> %.3f), |y|max(low->high)=(%.3f -> %.3f)",
                    tapered_workspace_params["z_min"],
                    tapered_workspace_params["z_max"],
                    tapered_workspace_params["x_min"],
                    tapered_workspace_params["x_max_low"],
                    tapered_workspace_params["x_max_high"],
                    tapered_workspace_params["y_max_low"],
                    tapered_workspace_params["y_max_high"],
                )
            logger_mp.info("[ARM_WORKSPACE] +z is arm-up in the IK/base frame.")
        else:
            logger_mp.info("[ARM_WORKSPACE] disabled.")
        if args.timing_debug:
            logger_mp.info(f"[TIMING] debug enabled, report interval = {args.timing_debug_interval:.1f}s")
        
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
                    "[BASE_CTRL] enabled: official G1D AgvClient bridge (async target queue) "
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

        tv_wrapper = create_teleop_input_provider(args, arm_ik=arm_ik)

        # end-effector
        if args.no_gripper:
            logger_mp.info("[EE] --no-gripper enabled; end-effector controller is disabled.")
        elif args.ee == "dex3":
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
            from teleop.sim.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        head_remote_camera = None
        left_remote_camera = None
        right_remote_camera = None
        head_camera = None
        left_camera = None
        right_camera = None

        # record / online inference camera sources are shared so each camera is opened once.
        needs_camera = bool(args.record or args.input_provider == "online_inference")
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency,
                                     image_size = [args.camera_width, args.camera_height],
                                     rerun_log = not args.headless)
        if needs_camera:
            head_remote_camera = maybe_open_remote_camera("head", args.head_zmq_endpoint)
            left_remote_camera = maybe_open_remote_camera("left_wrist", args.left_zmq_endpoint)
            right_remote_camera = maybe_open_remote_camera("right_wrist", args.right_zmq_endpoint)
            head_camera = maybe_open_local_camera("head", args.head_camera_id)
            left_camera = maybe_open_local_camera("left_wrist", args.left_camera_id)
            right_camera = maybe_open_local_camera("right_wrist", args.right_camera_id)

        latency_tracker = None
        if args.latency_trace:
            latency_tracker = SimpleLatencyTracker(
                output_path=args.latency_trace_path,
                summary_every=args.latency_summary_every,
                log_each_trace=True,
                timeout_s=args.latency_timeout,
            )
            if hasattr(arm_ctrl, 'set_latency_tracker'):
                arm_ctrl.set_latency_tracker(latency_tracker)
            if hasattr(arm_ctrl, 'set_latency_exec_thresholds'):
                arm_ctrl.set_latency_exec_thresholds(
                    args.latency_exec_q_threshold,
                    args.latency_exec_dq_threshold,
                )
            logger_mp.info(
                "[LATENCY] tracing enabled: output=%s, command_threshold=%.4f rad, exec_q_threshold=%.4f rad, exec_dq_threshold=%.4f rad/s",
                args.latency_trace_path,
                args.latency_command_threshold,
                args.latency_exec_q_threshold,
                args.latency_exec_dq_threshold,
            )

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
            logger_mp.info("🟠  Press [v] to CANCEL the active recording without saving.")
            if args.input_mode == "controller":
                logger_mp.info("🟡  Controller [left X] mirrors [s] for START or SAVE recording.")
                logger_mp.info("🟠  Controller [right B] mirrors [v] for CANCEL recording without saving.")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🟤  Press [h] to return both arms to the ready/home pose.")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        if args.auto_start:
            START = True
            logger_mp.info("[AUTO_START] entering START state automatically.")
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
        post_home_takeover_armed = False
        prev_left_grip_pressed = False
        prev_right_grip_pressed = False
        left_takeover_settle_frames = 0
        right_takeover_settle_frames = 0
        TAKEOVER_SETTLE_FRAMES = 0 if (
            args.controller_mapping_mode == "legacy_main" or args.input_provider in {"lerobot_offline", "online_inference"}
        ) else 2
        recording_waiting_for_first_frame = False
        last_record_wait_log_ns = 0
        last_enqueued_primary_frame_id = None

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()

            # record mode
            if args.record and RECORD_CANCEL:
                RECORD_CANCEL = False
                if RECORD_RUNNING or recording_waiting_for_first_frame:
                    RECORD_RUNNING = False
                    recording_waiting_for_first_frame = False
                    record_start_monotonic_ns = None
                    pending_record_samples.clear()
                    last_enqueued_primary_frame_id = None
                    recorder.cancel_episode()
                    logger_mp.info("[RECORD_CANCEL] active episode canceled; next recording will reuse the episode index.")
                else:
                    logger_mp.info("[RECORD_CANCEL] ignored: no active recording to cancel.")

            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING and not recording_waiting_for_first_frame:
                    if recorder.create_episode():
                        record_start_monotonic_ns = int(time.monotonic_ns())
                        pending_record_samples.clear()
                        last_enqueued_primary_frame_id = None
                        recording_waiting_for_first_frame = True
                        logger_mp.info("[RECORD_ALIGN] episode armed, waiting for first post-start camera frame before recording.")
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recording_waiting_for_first_frame = False
                    record_start_monotonic_ns = None
                    pending_record_samples.clear()
                    last_enqueued_primary_frame_id = None
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
            state_read_t0_ns = time.monotonic_ns()
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            state_read_t1_ns = time.monotonic_ns()
            current_state_sample_ns = int((state_read_t0_ns + state_read_t1_ns) // 2)
            append_timed_sample(
                state_history,
                current_state_sample_ns,
                q=current_lr_arm_q.copy(),
                dq=current_lr_arm_dq.copy(),
            )
            if latency_tracker is not None:
                latency_tracker.maybe_timeout()
                latency_tracker.maybe_mark_execute(
                    current_lr_arm_q,
                    current_lr_arm_dq,
                    q_threshold=args.latency_exec_q_threshold,
                    dq_threshold=args.latency_exec_dq_threshold,
                )
            current_left_wrist_pose, current_right_wrist_pose = get_robot_wrist_poses(arm_ik, current_lr_arm_q)

            online_left_gripper_q = 0.0
            online_right_gripper_q = 0.0
            if (not args.no_gripper) and args.ee == "dex1":
                try:
                    with dual_gripper_data_lock:
                        online_left_gripper_q = float(dual_gripper_state_array[0])
                        online_right_gripper_q = float(dual_gripper_state_array[1])
                except Exception:
                    online_left_gripper_q = 0.0
                    online_right_gripper_q = 0.0
            camera_sources = {
                "head": head_remote_camera if head_remote_camera is not None else head_camera,
                "left_wrist": left_remote_camera if left_remote_camera is not None else left_camera,
                "right_wrist": right_remote_camera if right_remote_camera is not None else right_camera,
            }

            # get xr's tele data
            tele_fetch_start = time.perf_counter()
            sample = tv_wrapper.get_sample(
                current_left_robot_wrist_pose=current_left_wrist_pose,
                current_right_robot_wrist_pose=current_right_wrist_pose,
                current_state_host_monotonic_ns=current_state_sample_ns,
                current_arm_q=current_lr_arm_q,
                current_arm_dq=current_lr_arm_dq,
                current_left_gripper_width=online_left_gripper_q,
                current_right_gripper_width=online_right_gripper_q,
                camera_sources=camera_sources,
                dt=control_dt,
            )
            tele_fetch_dt = time.perf_counter() - tele_fetch_start
            timing_debugger.add_tele_fetch(tele_fetch_dt, sample is not None)
            if sample is None:
                if bool(getattr(tv_wrapper, "done", False)):
                    if args.input_provider == "lerobot_offline":
                        exit_go_home = args.offline_replay_end_action == "home"
                        logger_mp.info(
                            "[OFFLINE_REPLAY] input provider finished. end_action=%s, stopping main loop.",
                            args.offline_replay_end_action,
                        )
                    START = False
                    STOP = True
                    continue
                timing_debugger.maybe_report(arm_ctrl=arm_ctrl, gripper_ctrl=gripper_ctrl)
                time.sleep(0.01)
                continue
            tele_data = sample.tele_data
            motion_intent = sample.motion_intent
            if args.input_provider == "online_inference" and bool(getattr(sample, "done", False)):
                metadata = getattr(motion_intent, "metadata", {}) or {}
                logger_mp.error(
                    "[ONLINE_INFERENCE] provider entered fail-closed state: status=%s error=%s",
                    metadata.get("online_inference_status"),
                    metadata.get("error"),
                )
                START = False
                STOP = True
                continue
            tele_data_recv_ts_ns = time.perf_counter_ns()
            tele_fetch_ms = tele_fetch_dt * 1000.0

            takeover_logic_start = time.perf_counter()
            tele_data = operator_runtime.apply_to_tele_data(
                tele_data,
                started=bool(START),
                on_press=on_press,
            )

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
            provider_enabled_arms = None
            if motion_intent is not None:
                provider_enabled_arms = motion_intent.metadata.get("enabled_arms")
            if provider_enabled_arms is not None:
                provider_enabled_set = {str(side) for side in provider_enabled_arms}
                left_arm_enabled = left_arm_enabled and ("left" in provider_enabled_set)
                right_arm_enabled = right_arm_enabled and ("right" in provider_enabled_set)
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

            left_grip_pressed = bool(tele_data.left_ctrl_squeeze)
            right_grip_pressed = bool(tele_data.right_ctrl_squeeze)
            left_takeover_rising_edge = left_grip_pressed and (not prev_left_grip_pressed)
            right_takeover_rising_edge = right_grip_pressed and (not prev_right_grip_pressed)
            if left_takeover_rising_edge:
                left_takeover_settle_frames = TAKEOVER_SETTLE_FRAMES
            if right_takeover_rising_edge:
                right_takeover_settle_frames = TAKEOVER_SETTLE_FRAMES
            left_zero_takeover_this_frame = left_takeover_settle_frames > 0
            right_zero_takeover_this_frame = right_takeover_settle_frames > 0
            any_zero_takeover_this_frame = (
                left_zero_takeover_this_frame or right_zero_takeover_this_frame
            )
            prev_left_grip_pressed = left_grip_pressed
            prev_right_grip_pressed = right_grip_pressed
            takeover_logic_ms = (time.perf_counter() - takeover_logic_start) * 1000.0

            if (not args.no_gripper) and (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif (not args.no_gripper) and args.ee == "dex1" and args.input_mode == "controller":
                if args.input_provider == "online_inference":
                    left_trigger_value = (
                        tele_data.left_ctrl_triggerValue
                        if left_arm_enabled
                        else float(np.clip(online_left_gripper_q / 5.4 * 2.0 + 5.0, 5.0, 7.0))
                    )
                    right_trigger_value = (
                        tele_data.right_ctrl_triggerValue
                        if right_arm_enabled
                        else float(np.clip(online_right_gripper_q / 5.4 * 2.0 + 5.0, 5.0, 7.0))
                    )
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = left_trigger_value
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = right_trigger_value
                else:
                    if left_arm_enabled:
                        with left_gripper_value.get_lock():
                            left_gripper_value.value = tele_data.left_ctrl_triggerValue
                    if right_arm_enabled:
                        with right_gripper_value.get_lock():
                            right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif (not args.no_gripper) and args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # high level base control
            base_result = apply_base_command(
                args=args,
                tele_data=tele_data,
                home_return_active=home_return_active,
                loco_wrapper=loco_wrapper,
                agv_bridge=agv_bridge,
                timing_debugger=timing_debugger,
            )
            base_control_mode = base_result.base_control_mode
            base_control_ms = base_result.base_control_ms
            base_move_ms = base_result.base_move_ms
            base_height_ms = base_result.base_height_ms
            base_misc_ms = base_result.base_misc_ms
            base_async_cycle_avg_ms = base_result.base_async_cycle_avg_ms
            base_async_move_avg_ms = base_result.base_async_move_avg_ms
            base_async_height_avg_ms = base_result.base_async_height_avg_ms
            base_async_queue_avg_ms = base_result.base_async_queue_avg_ms
            base_async_publish_hz = base_result.base_async_publish_hz
            base_vx = base_result.base_vx
            base_vy = base_result.base_vy
            base_wz = base_result.base_wz
            base_z = base_result.base_z
            if base_result.stop_requested:
                START = False
                STOP = True
            if base_result.should_continue_frame:
                continue

            # solve arm command target, safety-limit it, and compute gravity compensation.
            try:
                arm_command = build_arm_command(
                    arm_ik=arm_ik,
                    motion_intent=motion_intent,
                    current_lr_arm_q=current_lr_arm_q,
                    current_lr_arm_dq=current_lr_arm_dq,
                    current_hold_q=current_hold_q,
                    current_hold_tauff=current_hold_tauff,
                    left_arm_enabled=left_arm_enabled,
                    right_arm_enabled=right_arm_enabled,
                    home_return_active=home_return_active,
                    home_target_q=home_target_q,
                    control_dt=control_dt,
                    input_provider=args.input_provider,
                    frequency=args.frequency,
                    max_arm_joint_speed=args.max_arm_joint_speed,
                    home_return_speed=args.home_return_speed,
                    workspace_limit_enabled=workspace_limit_enabled,
                    workspace_mode=workspace_mode,
                    workspace_min=workspace_min,
                    workspace_max=workspace_max,
                    tapered_workspace_params=tapered_workspace_params,
                    compute_arm_gravity_tauff=compute_arm_gravity_tauff,
                    timing_debugger=timing_debugger,
                    any_zero_takeover_this_frame=any_zero_takeover_this_frame,
                    left_zero_takeover_this_frame=left_zero_takeover_this_frame,
                    right_zero_takeover_this_frame=right_zero_takeover_this_frame,
                    left_takeover_rising_edge=left_takeover_rising_edge,
                    right_takeover_rising_edge=right_takeover_rising_edge,
                    left_takeover_settle_frames=left_takeover_settle_frames,
                    right_takeover_settle_frames=right_takeover_settle_frames,
                    takeover_settle_frames=TAKEOVER_SETTLE_FRAMES,
                    post_home_takeover_armed=post_home_takeover_armed,
                    normalized_head_mode=normalized_head_mode,
                    sync_reference_to_current_live_pose=tv_wrapper.sync_reference_to_current_live_pose,
                    log=logger_mp,
                )
            except ValueError:
                START = False
                STOP = True
                continue

            sol_q = arm_command.sol_q
            sol_tauff = arm_command.sol_tauff
            current_hold_q = arm_command.current_hold_q
            current_hold_tauff = arm_command.current_hold_tauff
            provider_feedback = arm_command.provider_feedback
            ik_ms = arm_command.ik_ms
            safety_ms = arm_command.safety_ms
            gravity_ms = arm_command.gravity_ms
            left_arm_enabled = arm_command.left_arm_enabled
            right_arm_enabled = arm_command.right_arm_enabled
            post_home_takeover_armed = arm_command.post_home_takeover_armed
            left_takeover_settle_frames = arm_command.left_takeover_settle_frames
            right_takeover_settle_frames = arm_command.right_takeover_settle_frames
            if args.input_provider == "online_inference" and provider_feedback is not None:
                report_feedback = getattr(tv_wrapper, "report_control_feedback", None)
                if callable(report_feedback):
                    report_feedback(provider_feedback)

            trace_seq = None
            if latency_tracker is not None and latency_tracker.can_start_new_trace():
                max_command_delta = float(np.max(np.abs(sol_q - current_lr_arm_q)))
                if max_command_delta >= args.latency_command_threshold:
                    online_trace_extra = {}
                    if args.input_provider == "online_inference" and motion_intent is not None:
                        online_trace_extra = {
                            key: value
                            for key, value in (getattr(motion_intent, "metadata", {}) or {}).items()
                            if str(key).startswith("online_")
                        }
                        online_trace_extra["online_provider_output_perf_ns"] = int(tele_data_recv_ts_ns)
                    trace_seq = latency_tracker.begin_trace(
                        recv_ts_ns=tele_data_recv_ts_ns,
                        recv_q=current_lr_arm_q,
                        extra={
                            **online_trace_extra,
                            "tele_fetch_ms": tele_fetch_ms,
                            "takeover_logic_ms": takeover_logic_ms,
                            "base_control_ms": base_control_ms,
                            "base_control_mode": base_control_mode,
                            "base_move_ms": base_move_ms,
                            "base_height_ms": base_height_ms,
                            "base_misc_ms": base_misc_ms,
                            "base_vx_cmd": float(base_vx),
                            "base_vy_cmd": float(base_vy),
                            "base_wz_cmd": float(base_wz),
                            "base_z_cmd": float(base_z),
                            "base_async_publish_hz": base_async_publish_hz,
                            "base_async_move_avg_ms": base_async_move_avg_ms,
                            "base_async_height_avg_ms": base_async_height_avg_ms,
                            "base_async_cycle_avg_ms": base_async_cycle_avg_ms,
                            "base_async_queue_avg_ms": base_async_queue_avg_ms,
                            "ik_ms": ik_ms,
                            "safety_ms": safety_ms,
                            "gravity_ms": gravity_ms,
                            "left_arm_enabled": bool(left_arm_enabled),
                            "right_arm_enabled": bool(right_arm_enabled),
                            "home_return_active": bool(home_return_active),
                            "left_takeover_rising_edge": bool(left_takeover_rising_edge),
                            "right_takeover_rising_edge": bool(right_takeover_rising_edge),
                            "left_zero_takeover_this_frame": bool(left_zero_takeover_this_frame),
                            "right_zero_takeover_this_frame": bool(right_zero_takeover_this_frame),
                            "max_command_delta": max_command_delta,
                            "left_joint_delta_norm": float(np.linalg.norm(sol_q[:7] - current_lr_arm_q[:7])),
                            "right_joint_delta_norm": float(np.linalg.norm(sol_q[-7:] - current_lr_arm_q[-7:])),
                        },
                    )

            action_command_monotonic_ns = int(time.monotonic_ns())
            append_timed_sample(
                action_history,
                action_command_monotonic_ns,
                q=sol_q.copy(),
                tauff=sol_tauff.copy(),
            )
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff, trace_seq=trace_seq)
            if home_return_active and np.all(np.abs(sol_q - home_target_q) < 0.05):
                home_return_active = False
                calibration_hold_q = current_hold_q.copy()
                calibration_hold_tauff = current_hold_tauff.copy()
                logger_mp.info("[HOME] reached ready/calibration pose. Waiting for both grips to release before teleop resumes.")

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if (not args.no_gripper) and args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                elif (not args.no_gripper) and args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                elif (not args.no_gripper) and args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                elif (not args.no_gripper) and (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []

                # arm state and action
                if RECORD_RUNNING or recording_waiting_for_first_frame:
                    head_source = head_remote_camera if head_remote_camera is not None else head_camera
                    left_source = left_remote_camera if left_remote_camera is not None else left_camera
                    right_source = right_remote_camera if right_remote_camera is not None else right_camera

                    if recording_waiting_for_first_frame:
                        wait_target_ns = int(time.monotonic_ns())
                        source_entries = [
                            ("head", head_source),
                            ("left_wrist", left_source),
                            ("right_wrist", right_source),
                        ]
                        required_sources = [(name, src) for name, src in source_entries if src is not None]
                        if not required_sources:
                            RECORD_RUNNING = True
                            recording_waiting_for_first_frame = False
                            logger_mp.info("[RECORD_ALIGN] no camera source configured, recording starts immediately.")
                        else:
                            first_frame_ready = True
                            first_frame_times = []
                            for camera_name, source in required_sources:
                                _, meta = source.get_nearest(
                                    wait_target_ns,
                                    max_delta_ns=None,
                                    min_monotonic_ns=record_start_monotonic_ns,
                                    copy=False,
                                )
                                meta_ts = camera_meta_monotonic_ns(meta)
                                if meta is None or meta_ts is None:
                                    first_frame_ready = False
                                    break
                                first_frame_times.append(meta_ts)
                            if not first_frame_ready:
                                now_ns = time.monotonic_ns()
                                if now_ns - last_record_wait_log_ns > 1_000_000_000:
                                    logger_mp.info("[RECORD_ALIGN] waiting for first post-start camera frame...")
                                    last_record_wait_log_ns = now_ns
                                continue
                            record_start_monotonic_ns = int(max(first_frame_times))
                            RECORD_RUNNING = True
                            recording_waiting_for_first_frame = False
                            logger_mp.info(
                                "[RECORD_ALIGN] first post-start camera frame received, recording begins at monotonic_ns=%d",
                                record_start_monotonic_ns,
                            )
                            continue

                    record_min_timestamp_ns = int(record_start_monotonic_ns or time.monotonic_ns())
                    primary_source_entry = None
                    for camera_name, source in (
                        ("head", head_source),
                        ("left_wrist", left_source),
                        ("right_wrist", right_source),
                    ):
                        if source is None:
                            continue
                        primary_source_entry = (camera_name, source)
                        break

                    if primary_source_entry is not None:
                        primary_camera_name, primary_source = primary_source_entry
                        primary_frame, primary_meta = primary_source.get_latest(copy=True)
                    else:
                        primary_camera_name = None
                        primary_frame = None
                        primary_meta = None

                    if primary_frame is not None and primary_meta is not None:
                        primary_ts = int(camera_meta_monotonic_ns(primary_meta))
                        primary_frame_seq = (primary_meta or {}).get("frame_seq")
                        primary_frame_id = camera_frame_identity(primary_camera_name, primary_meta)
                        already_pending = False
                        if pending_record_samples:
                            last_pending = pending_record_samples[-1]
                            already_pending = (
                                last_pending.get("primary_camera_name") == primary_camera_name
                                and last_pending.get("frame_seq") == primary_frame_seq
                                and last_pending.get("sample_monotonic_ns") == primary_ts
                            )
                        if (
                            not already_pending
                            and primary_ts >= record_min_timestamp_ns
                            and primary_frame_id is not None
                            and primary_frame_id != last_enqueued_primary_frame_id
                        ):
                            pending_record_samples.append(
                                {
                                    "enqueue_wall_time_ns": int(time.time_ns()),
                                    "sample_monotonic_ns": primary_ts,
                                    "primary_camera_name": primary_camera_name,
                                    "primary_frame": primary_frame,
                                    "primary_meta": primary_meta,
                                    "left_ee_state": list(left_ee_state),
                                    "right_ee_state": list(right_ee_state),
                                    "left_hand_action": list(left_hand_action),
                                    "right_hand_action": list(right_hand_action),
                                    "teleop_input_perf_counter_ns": int(tele_data_recv_ts_ns),
                                    "frame_seq": primary_frame_seq,
                                }
                            )
                            last_enqueued_primary_frame_id = primary_frame_id

                    while pending_record_samples:
                        pending = pending_record_samples[0]
                        sample_monotonic_ns = int(pending["sample_monotonic_ns"])
                        state_earliest, state_latest = timed_buffer_bounds(
                            state_history, min_timestamp_ns=record_min_timestamp_ns
                        )
                        action_earliest, action_latest = timed_buffer_bounds(
                            action_history, min_timestamp_ns=record_min_timestamp_ns
                        )
                        now_mono_ns = int(time.monotonic_ns())
                        sample_age_ns = now_mono_ns - sample_monotonic_ns

                        if state_earliest is None or action_earliest is None:
                            break

                        if sample_monotonic_ns < state_earliest or sample_monotonic_ns < action_earliest:
                            logger_mp.warning("[RECORD_ALIGN] drop pending sample: target timestamp fell out of state/action history window.")
                            pending_record_samples.popleft()
                            continue

                        if state_latest is None or action_latest is None:
                            break

                        waiting_for_future_coverage = (
                            sample_monotonic_ns > state_latest or sample_monotonic_ns > action_latest
                        )
                        if waiting_for_future_coverage and sample_age_ns <= record_future_wait_timeout_ns:
                            break

                        aligned_state = interpolate_timed_sample_strict(
                            state_history,
                            sample_monotonic_ns,
                            max_delta_ns=state_align_max_delta_ns,
                            min_timestamp_ns=record_min_timestamp_ns,
                        )
                        aligned_action = interpolate_timed_sample_strict(
                            action_history,
                            sample_monotonic_ns,
                            max_delta_ns=action_align_max_delta_ns,
                            min_timestamp_ns=record_min_timestamp_ns,
                        )
                        if aligned_state is None:
                            aligned_state = nearest_timed_sample(
                                state_history,
                                sample_monotonic_ns,
                                max_delta_ns=state_nearest_fallback_max_delta_ns,
                                min_timestamp_ns=record_min_timestamp_ns,
                            )
                            if aligned_state is not None:
                                aligned_state["interpolation_mode"] = "nearest_fallback"
                        if aligned_action is None:
                            aligned_action = nearest_timed_sample(
                                action_history,
                                sample_monotonic_ns,
                                max_delta_ns=action_nearest_fallback_max_delta_ns,
                                min_timestamp_ns=record_min_timestamp_ns,
                            )
                            if aligned_action is not None:
                                aligned_action["interpolation_mode"] = "nearest_fallback"
                        if aligned_state is None:
                            if sample_age_ns > pending_sample_timeout_ns:
                                logger_mp.warning("[RECORD_ALIGN] drop pending sample: no interpolated state found at primary camera timestamp.")
                                pending_record_samples.popleft()
                                continue
                            break
                        if aligned_action is None:
                            if sample_age_ns > pending_sample_timeout_ns:
                                logger_mp.warning("[RECORD_ALIGN] drop pending sample: no interpolated action found at primary camera timestamp.")
                                pending_record_samples.popleft()
                                continue
                            break

                        colors = {
                            pending["primary_camera_name"]: pending["primary_frame"]
                        }
                        depths = {}
                        camera_timestamps = {
                            pending["primary_camera_name"]: pending["primary_meta"]
                        }

                        if head_source is not None and pending["primary_camera_name"] != "head":
                            head_frame, head_meta = head_source.get_nearest(
                                sample_monotonic_ns,
                                max_delta_ns=camera_align_max_delta_ns,
                                min_monotonic_ns=record_min_timestamp_ns,
                                copy=True,
                            )
                            if head_frame is not None:
                                colors["head"] = head_frame
                                camera_timestamps["head"] = head_meta
                        if left_source is not None and pending["primary_camera_name"] != "left_wrist":
                            left_frame, left_meta = left_source.get_nearest(
                                sample_monotonic_ns,
                                max_delta_ns=camera_align_max_delta_ns,
                                min_monotonic_ns=record_min_timestamp_ns,
                                copy=True,
                            )
                            if left_frame is not None:
                                colors["left_wrist"] = left_frame
                                camera_timestamps["left_wrist"] = left_meta
                        if right_source is not None and pending["primary_camera_name"] != "right_wrist":
                            right_frame, right_meta = right_source.get_nearest(
                                sample_monotonic_ns,
                                max_delta_ns=camera_align_max_delta_ns,
                                min_monotonic_ns=record_min_timestamp_ns,
                                copy=True,
                            )
                            if right_frame is not None:
                                colors["right_wrist"] = right_frame
                                camera_timestamps["right_wrist"] = right_meta

                        aligned_lr_arm_q = np.asarray(aligned_state["q"], dtype=float)
                        aligned_sol_q = np.asarray(aligned_action["q"], dtype=float)
                        left_arm_state  = aligned_lr_arm_q[:7]
                        right_arm_state = aligned_lr_arm_q[-7:]
                        left_arm_action = aligned_sol_q[:7]
                        right_arm_action = aligned_sol_q[-7:]
                        record_arm_repr = args.record_arm_repr
                        need_arm_pose = record_arm_repr in {"pose", "both"}
                        left_state_pose = right_state_pose = None
                        left_action_pose = right_action_pose = None
                        if need_arm_pose:
                            left_state_pose, right_state_pose = get_robot_wrist_poses(arm_ik, aligned_lr_arm_q)
                            left_action_pose, right_action_pose = get_robot_wrist_poses(arm_ik, aligned_sol_q)

                        left_arm_state_entry = {
                            "qpos": left_arm_state.tolist() if record_arm_repr in {"qpos", "both"} else [],
                            "qvel": [],
                            "torque": [],
                        }
                        right_arm_state_entry = {
                            "qpos": right_arm_state.tolist() if record_arm_repr in {"qpos", "both"} else [],
                            "qvel": [],
                            "torque": [],
                        }
                        left_arm_action_entry = {
                            "qpos": left_arm_action.tolist() if record_arm_repr in {"qpos", "both"} else [],
                            "qvel": [],
                            "torque": [],
                        }
                        right_arm_action_entry = {
                            "qpos": right_arm_action.tolist() if record_arm_repr in {"qpos", "both"} else [],
                            "qvel": [],
                            "torque": [],
                        }
                        if need_arm_pose:
                            left_arm_state_entry["pose"] = pose_matrix_to_record(left_state_pose)
                            right_arm_state_entry["pose"] = pose_matrix_to_record(right_state_pose)
                            left_arm_action_entry["pose"] = pose_matrix_to_record(left_action_pose)
                            right_arm_action_entry["pose"] = pose_matrix_to_record(right_action_pose)

                        states = {
                            "left_arm": left_arm_state_entry,
                            "right_arm": right_arm_state_entry,
                            "left_ee": {
                                "qpos": pending["left_ee_state"],
                                "qvel": [],
                                "torque": [],
                            },
                            "right_ee": {
                                "qpos": pending["right_ee_state"],
                                "qvel": [],
                                "torque": [],
                            },
                        }
                        actions = {
                            "left_arm": left_arm_action_entry,
                            "right_arm": right_arm_action_entry,
                            "left_ee": {
                                "qpos": pending["left_hand_action"],
                                "qvel": [],
                                "torque": [],
                            },
                            "right_ee": {
                                "qpos": pending["right_hand_action"],
                                "qvel": [],
                                "torque": [],
                            },
                        }
                        timestamps = {
                            "sample_wall_time_ns": int(time.time_ns()),
                            "sample_monotonic_ns": sample_monotonic_ns,
                            "teleop_input_perf_counter_ns": int(pending["teleop_input_perf_counter_ns"]),
                            "primary_camera_name": pending["primary_camera_name"],
                            "state": build_alignment_timestamp_entry(aligned_state, sample_monotonic_ns),
                            "action": build_alignment_timestamp_entry(aligned_action, sample_monotonic_ns),
                            "camera": camera_timestamps,
                        }
                        aligned_arm_tauff = np.asarray(aligned_action["tauff"], dtype=float).reshape(-1)
                        if aligned_arm_tauff.shape[0] != 14 or not np.all(np.isfinite(aligned_arm_tauff)):
                            logger_mp.warning(
                                "[RECORD_ALIGN] drop pending sample: invalid aligned arm tauff for control sidecar."
                            )
                            pending_record_samples.popleft()
                            continue
                        control_extras = {
                            "arm_tauff": aligned_arm_tauff.tolist(),
                        }
                        if args.sim:
                            sim_state = sim_state_subscriber.read_data()
                            recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state, timestamps=timestamps, control_extras=control_extras)
                        else:
                            recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, timestamps=timestamps, control_extras=control_extras)
                        pending_record_samples.popleft()

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            timing_debugger.add_loop(time_elapsed, overrun=(sleep_time <= 1e-6))
            timing_debugger.maybe_report(arm_ctrl=arm_ctrl, gripper_ctrl=gripper_ctrl)
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
                if exit_go_home:
                    logger_mp.info("[EXIT] returning dual arms to home before shutdown...")
                    arm_ctrl.ctrl_dual_arm_go_home()
                logger_mp.warning(
                    "[EXIT] holding dual arms at home for %.1fs before shutdown. "
                    "Keep clear and support the robot if needed.",
                    exit_home_hold_sec,
                )
                time.sleep(exit_home_hold_sec)
        except Exception as e:
            logger_mp.error(f"Failed to hold dual arms at home before shutdown: {e}")
        
        try:
            stop_listening()
            if listen_keyboard_thread is not None:
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener: {e}")
        
        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close teleop input provider: {e}")

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
            if args.sim and sim_state_subscriber is not None:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")

        for camera in [head_remote_camera, left_remote_camera, right_remote_camera, head_camera, left_camera, right_camera]:
            try:
                if camera is not None:
                    camera.close()
            except Exception as e:
                logger_mp.error(f"Failed to close local camera: {e}")
        
        try:
            if args.record and recorder is not None:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
