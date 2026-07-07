"""真机遥操主入口：只编排启动、主循环和各职责模块的接口调用。"""

import time
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
from data_pipeline.recording.alignment import append_timed_sample
from core.input.teleop_input_provider import validate_lerobot_offline_episode
from teleop.debug.timing_debugger import TimingDebugger
from teleop.real.setup import (
    RealTeleopComponents,
    initialize_dds,
    log_workspace_config,
    setup_real_teleop_components,
)
from teleop.runtime.operator_runtime import OperatorRuntime
from teleop.control_flow.base_command import apply_base_command
from teleop.control_flow.arm_command_pipeline import build_arm_command
from teleop.control_flow.operator_state import OperatorStateFlow
from teleop.control_flow.end_effector_command import (
    apply_end_effector_command,
    read_online_gripper_widths,
)
from sshkeyboard import listen_keyboard, stop_listening

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


def start_keyboard_listener():
    listen_keyboard_thread = threading.Thread(
        target=listen_keyboard,
        kwargs={"on_press": on_press, "until": None, "sequential": False,},
        daemon=True,
    )
    listen_keyboard_thread.start()
    return listen_keyboard_thread


def cleanup_real_teleop_resources(
    *,
    args,
    arm_ctrl,
    tv_wrapper,
    listen_keyboard_thread,
    recorder,
    sim_state_subscriber,
    agv_bridge,
    cameras,
    exit_go_home: bool,
    exit_home_hold_sec: float,
):
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
        if agv_bridge is not None:
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

    for camera in cameras:
        try:
            if camera is not None:
                camera.close()
        except Exception as e:
            logger_mp.error(f"Failed to close camera: {e}")

    try:
        if args.record and recorder is not None:
            recorder.close()
    except Exception as e:
        logger_mp.error(f"Failed to close recorder: {e}")


if __name__ == '__main__':
    components = RealTeleopComponents()
    arm_ctrl = None
    arm_ik = None
    tv_wrapper = None
    listen_keyboard_thread = None
    gripper_ctrl = None
    loco_wrapper = None
    recorder = None
    recording_flow = None
    sim_state_subscriber = None
    agv_bridge = None
    exit_go_home = True
    exit_home_hold_sec = 5.0
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
    control_dt = 1.0 / max(args.frequency, 1e-6)
    if args.base_controller == "g1d_agv" and args.motion:
        raise ValueError("Do not combine --base-controller g1d_agv with --motion. G1D AGV base control should run with the arms kept in debug mode.")
    if args.input_provider == "lerobot_offline":
        validate_lerobot_offline_episode(
            args.offline_replay_dataset_root,
            args.offline_replay_episode_index,
            args.offline_replay_arm_source,
        )

    try:
        initialize_dds(args)
        listen_keyboard_thread = start_keyboard_listener()
        log_workspace_config(
            args,
            workspace_limit_enabled=workspace_limit_enabled,
            workspace_mode=workspace_mode,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            tapered_workspace_params=tapered_workspace_params,
            log=logger_mp,
        )

        setup_real_teleop_components(args, log=logger_mp, components=components)
        arm_ik = components.arm_ik
        arm_ctrl = components.arm_ctrl
        tv_wrapper = components.tv_wrapper
        gripper_ctrl = components.ee.gripper_ctrl
        loco_wrapper = components.loco_wrapper
        agv_bridge = components.agv_bridge
        recorder = components.recorder
        recording_flow = components.recording_flow
        sim_state_subscriber = components.sim_state_subscriber
        latency_tracker = components.latency_tracker

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

        home_return_active = False
        home_target_q = np.zeros_like(current_hold_q)
        post_home_takeover_armed = False
        TAKEOVER_SETTLE_FRAMES = 0 if (
            args.controller_mapping_mode == "legacy_main" or args.input_provider in {"lerobot_offline", "online_inference"}
        ) else 2
        operator_state_flow = OperatorStateFlow(
            takeover_settle_frames=TAKEOVER_SETTLE_FRAMES,
            log=logger_mp,
        )
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()

            # record mode
            if recording_flow is not None:
                command_recording = recording_flow.handle_commands(
                    record_running=RECORD_RUNNING,
                    record_toggle=RECORD_TOGGLE,
                    record_cancel=RECORD_CANCEL,
                )
                RECORD_RUNNING = command_recording.record_running
                RECORD_TOGGLE = command_recording.record_toggle
                RECORD_CANCEL = command_recording.record_cancel

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

            online_left_gripper_q, online_right_gripper_q = read_online_gripper_widths(
                args=args,
                dual_gripper_data_lock=components.ee.dual_gripper_data_lock,
                dual_gripper_state_array=components.ee.dual_gripper_state_array,
            )
            camera_sources = components.cameras.sources()

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

            operator_state = operator_state_flow.apply(
                args=args,
                operator_runtime=operator_runtime,
                tele_data=tele_data,
                started=bool(START),
                on_press=on_press,
                motion_intent=motion_intent,
                home_return_active=home_return_active,
                normalized_head_mode=normalized_head_mode,
                tv_wrapper=tv_wrapper,
                arm_ik=arm_ik,
                current_hold_q=current_hold_q,
                reset_arm_ik_state=reset_arm_ik_state,
                timer=time.perf_counter,
            )
            tele_data = operator_state.tele_data
            left_arm_enabled = operator_state.left_arm_enabled
            right_arm_enabled = operator_state.right_arm_enabled
            home_return_active = operator_state.home_return_active
            post_home_takeover_armed = operator_state.post_home_takeover_armed or post_home_takeover_armed
            left_takeover_settle_frames = operator_state.left_takeover_settle_frames
            right_takeover_settle_frames = operator_state.right_takeover_settle_frames
            left_takeover_rising_edge = operator_state.left_takeover_rising_edge
            right_takeover_rising_edge = operator_state.right_takeover_rising_edge
            left_zero_takeover_this_frame = operator_state.left_zero_takeover_this_frame
            right_zero_takeover_this_frame = operator_state.right_zero_takeover_this_frame
            any_zero_takeover_this_frame = operator_state.any_zero_takeover_this_frame
            takeover_logic_ms = operator_state.takeover_logic_ms

            end_effector_start = time.perf_counter()
            apply_end_effector_command(
                args=args,
                tele_data=tele_data,
                left_arm_enabled=left_arm_enabled,
                right_arm_enabled=right_arm_enabled,
                online_left_gripper_q=online_left_gripper_q,
                online_right_gripper_q=online_right_gripper_q,
                left_hand_pos_array=components.ee.left_hand_pos_array,
                right_hand_pos_array=components.ee.right_hand_pos_array,
                left_gripper_value=components.ee.left_gripper_value,
                right_gripper_value=components.ee.right_gripper_value,
            )
            end_effector_command_ms = (time.perf_counter() - end_effector_start) * 1000.0

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
            operator_sync_start = time.perf_counter()
            operator_state_flow.sync_from_arm_command(
                left_takeover_settle_frames=left_takeover_settle_frames,
                right_takeover_settle_frames=right_takeover_settle_frames,
            )
            operator_sync_ms = (time.perf_counter() - operator_sync_start) * 1000.0

            provider_feedback_ms = 0.0
            if args.input_provider == "online_inference" and provider_feedback is not None:
                provider_feedback_start = time.perf_counter()
                report_feedback = getattr(tv_wrapper, "report_control_feedback", None)
                if callable(report_feedback):
                    report_feedback(provider_feedback)
                provider_feedback_ms = (time.perf_counter() - provider_feedback_start) * 1000.0

            trace_seq = None
            if latency_tracker is not None and latency_tracker.can_start_new_trace():
                latency_trace_prepare_start = time.perf_counter()
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
                            "arm_cmd_input_ms": arm_command.arm_cmd_input_ms,
                            "arm_cmd_takeover_reset_ms": arm_command.arm_cmd_takeover_reset_ms,
                            "arm_cmd_target_extra_ms": arm_command.arm_cmd_target_extra_ms,
                            "arm_cmd_feedback_gate_ms": arm_command.arm_cmd_feedback_gate_ms,
                            "arm_cmd_enable_gating_ms": arm_command.arm_cmd_enable_gating_ms,
                            "arm_cmd_takeover_settle_ms": arm_command.arm_cmd_takeover_settle_ms,
                            "arm_cmd_speed_feedback_ms": arm_command.arm_cmd_speed_feedback_ms,
                            "arm_cmd_hold_update_ms": arm_command.arm_cmd_hold_update_ms,
                            "end_effector_command_ms": end_effector_command_ms,
                            "operator_sync_ms": operator_sync_ms,
                            "provider_feedback_ms": provider_feedback_ms,
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
                    latency_trace_prepare_ms = (time.perf_counter() - latency_trace_prepare_start) * 1000.0
                    latency_tracker.set_fields(trace_seq, latency_trace_prepare_ms=latency_trace_prepare_ms)

            action_history_start = time.perf_counter()
            action_command_monotonic_ns = int(time.monotonic_ns())
            append_timed_sample(
                action_history,
                action_command_monotonic_ns,
                q=sol_q.copy(),
                tauff=sol_tauff.copy(),
            )
            action_history_append_ms = (time.perf_counter() - action_history_start) * 1000.0
            if trace_seq is not None:
                latency_tracker.set_fields(trace_seq, action_history_append_ms=action_history_append_ms)
            ctrl_dual_arm_start = time.perf_counter()
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff, trace_seq=trace_seq)
            ctrl_dual_arm_call_ms = (time.perf_counter() - ctrl_dual_arm_start) * 1000.0
            if trace_seq is not None:
                latency_tracker.set_fields(trace_seq, ctrl_dual_arm_call_ms=ctrl_dual_arm_call_ms)
            if home_return_active and np.all(np.abs(sol_q - home_target_q) < 0.05):
                home_return_active = False
                calibration_hold_q = current_hold_q.copy()
                calibration_hold_tauff = current_hold_tauff.copy()
                logger_mp.info("[HOME] reached ready/calibration pose. Waiting for both grips to release before teleop resumes.")

            # record data
            if recording_flow is not None:
                frame_recording = recording_flow.process_frame(
                    record_running=RECORD_RUNNING,
                    camera_sources=camera_sources,
                    state_history=state_history,
                    action_history=action_history,
                    teleop_input_perf_counter_ns=tele_data_recv_ts_ns,
                    arm_ik=arm_ik,
                    get_wrist_poses=get_robot_wrist_poses,
                    sim_state_subscriber=sim_state_subscriber,
                )
                RECORD_RUNNING = frame_recording.record_running
                READY = frame_recording.ready
                if frame_recording.should_continue_frame:
                    continue

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
        cleanup_real_teleop_resources(
            args=args,
            arm_ctrl=arm_ctrl or components.arm_ctrl,
            tv_wrapper=tv_wrapper or components.tv_wrapper,
            listen_keyboard_thread=listen_keyboard_thread,
            recorder=recorder or components.recorder,
            sim_state_subscriber=sim_state_subscriber or components.sim_state_subscriber,
            agv_bridge=agv_bridge or components.agv_bridge,
            cameras=components.cameras.close_list(),
            exit_go_home=exit_go_home,
            exit_home_hold_sec=exit_home_hold_sec,
        )
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
