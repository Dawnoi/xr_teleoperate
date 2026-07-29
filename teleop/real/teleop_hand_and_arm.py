"""真机遥操主入口：只编排启动、主循环和各职责模块的接口调用。"""

import time
import threading
from copy import copy
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
from core.input.online_inference_provider import create_online_inference_provider
from core.input.base import BaseCommandIntent
from core.control.workspace_governor import WorkspaceGovernor, WorkspaceGovernorConfig
from core.input.teleop_input_provider import validate_lerobot_offline_episode
from teleop.debug.inference_pose_debug import wrist_pose_to_debug_sample
from teleop.debug.timing_debugger import TimingDebugger
from teleop.real.setup import (
    RealTeleopComponents,
    initialize_dds,
    log_workspace_config,
    setup_real_teleop_components,
    switch_recording_root,
)
from teleop.runtime.operator_runtime import OperatorRuntime
from teleop.control_flow.base_command import (
    apply_base_command,
    integrate_manual_torso_yaw_target,
    map_base_command,
    map_manual_torso_yaw_rate,
    resolve_runtime_base_command_source,
    stop_base_command,
)
from teleop.control_flow.mobile_manipulation_coordinator import (
    column_position_from_raw_height,
    G1DIkFrameKinematics,
    MobileManipulationCoordinator,
    MobileStateSample,
)
from teleop.control_flow.arm_workspace_config import build_arm_side_workspaces
from teleop.control_flow.arm_command_pipeline import build_arm_command
from teleop.control_flow.operator_state import OperatorStateFlow, rebase_xr_takeover_after_provider_switch
from teleop.control_flow.end_effector_command import (
    apply_end_effector_command,
    read_dual_gripper_snapshot,
)
from teleop.runtime.provider_switch import ActiveProviderKind, TeleopProviderRuntime
from teleop.ui.command_bus import UiCommandBus, UiCommandName
from teleop.ui.integration import (
    build_runtime_camera_status,
    build_runtime_web_payload,
    dispatch_ui_commands,
)
from teleop.ui.server import TeleopUiServer
from teleop.ui.state_store import UiStateStore
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


PROVIDER_UI_COMMANDS = {
    UiCommandName.SET_PROVIDER_HOLD,
    UiCommandName.SET_PROVIDER_XR,
    UiCommandName.START_RAW_REPLAY,
    UiCommandName.STOP_RAW_REPLAY,
    UiCommandName.START_ONLINE_INFERENCE,
    UiCommandName.STOP_ONLINE_INFERENCE,
    UiCommandName.SET_RECORD_ROOT,
}


def split_ui_commands(commands):
    keyboard_commands = []
    provider_commands = []
    for command in commands:
        if command.name in PROVIDER_UI_COMMANDS:
            provider_commands.append(command)
        else:
            keyboard_commands.append(command)
    return keyboard_commands, provider_commands


def recording_is_active_or_armed(record_running, recording_flow):
    if bool(record_running):
        return True
    flow_state = getattr(recording_flow, "state", None)
    return bool(getattr(flow_state, "waiting_for_first_frame", False))


def switch_recording_root_from_ui(*, args, components, recorder, recording_flow, record_running, payload, log):
    if recording_is_active_or_armed(record_running, recording_flow):
        log.error("[RECORD_ROOT] rejected: recording is active or armed.")
        return recorder, recording_flow
    root_dir = str(payload.get("root_dir", "")).strip()
    if not root_dir:
        raise ValueError("UI record root command requires root_dir")
    switch_recording_root(args, components, root_dir, log)
    return components.recorder, components.recording_flow

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


def get_camera_frame_by_id(cameras, camera_id: int):
    camera_name_by_id = {
        0: "head",
        1: "left_wrist",
        2: "right_wrist",
    }
    camera_name = camera_name_by_id.get(int(camera_id))
    if camera_name is None:
        return None, None
    source = cameras.sources().get(camera_name) if cameras is not None else None
    if source is None:
        return None, None
    return source.get_latest(copy=True)


def missing_online_inference_camera_names(cameras) -> list[str]:
    sources = cameras.sources() if cameras is not None else {}
    required = ("head", "left_wrist", "right_wrist")
    return [name for name in required if sources.get(name) is None]


def set_online_inference_gripper_mode(gripper_ctrl, enabled: bool) -> None:
    if gripper_ctrl is not None:
        gripper_ctrl.force_hold_extra_close_enabled = bool(enabled)


def cleanup_real_teleop_resources(
    *,
    args,
    arm_ctrl,
    tv_wrapper,
    listen_keyboard_thread,
    recorder,
    validation_manager,
    sim_state_subscriber,
    base_state_receiver,
    loco_wrapper,
    agv_bridge,
    cameras,
    ui_server,
    exit_go_home: bool,
    exit_home_hold_sec: float,
):
    if ui_server is not None:
        ui_server.stop()

    if args.base_motion:
        base_controller = str(args.base_controller)
        backend_ready = (
            (base_controller == "loco" and loco_wrapper is not None)
            or (base_controller == "g1d_agv" and agv_bridge is not None)
        )
        if backend_ready:
            stop_base_command(
                args=args,
                loco_wrapper=loco_wrapper,
                agv_bridge=agv_bridge,
            )
        elif base_controller != "none":
            logger_mp.error(
                "[EXIT] base STOP skipped because %s backend was not initialized; "
                "startup failed before this process owned the base command path.",
                base_controller,
            )

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

    if agv_bridge is not None:
        agv_bridge.close()

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

    if base_state_receiver is not None:
        base_state_receiver.close()

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

    if validation_manager is not None:
        validation_manager.close()


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
    base_state_receiver = None
    agv_bridge = None
    ui_command_bus = None
    ui_state_store = None
    ui_server = None
    provider_runtime = None
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
    side_workspaces = build_arm_side_workspaces(
        args,
        workspace_mode=workspace_mode,
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        tapered_workspace_params=tapered_workspace_params,
    )
    timing_debugger = TimingDebugger(enabled=args.timing_debug, interval_sec=args.timing_debug_interval)
    state_history_size = max(32, int(args.frequency * 6))
    action_history_size = max(32, int(args.frequency * 6))
    base_history_size = max(32, int(getattr(args, "base_history_size", 512)))
    state_history = deque(maxlen=state_history_size)
    action_history = deque(maxlen=action_history_size)
    base_action_history = deque(maxlen=base_history_size)
    control_dt = 1.0 / max(args.frequency, 1e-6)
    if args.input_provider == "lerobot_offline":
        validate_lerobot_offline_episode(
            args.offline_replay_dataset_root,
            args.offline_replay_episode_index,
            args.offline_replay_arm_source,
        )

    def publish_ui_payload(payload) -> None:
        if ui_state_store is not None:
            ui_state_store.update(payload)

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
            side_workspaces=side_workspaces,
            log=logger_mp,
        )

        setup_real_teleop_components(args, log=logger_mp, components=components)
        arm_ik = components.arm_ik
        arm_ctrl = components.arm_ctrl
        tv_wrapper = components.tv_wrapper
        def create_ui_online_provider(*, prompt: str):
            inference_args = copy(args)
            inference_args.input_provider = "online_inference"
            inference_args.online_inference_transport = "http"
            inference_args.online_inference_base_url = (
                str(getattr(args, "online_inference_base_url", "") or "").strip() or "http://127.0.0.1:18027"
            )
            inference_args.online_inference_protocol_profile = "pi05_dual_arm_20d"
            inference_args.online_inference_arm_side = "both"
            inference_args.online_inference_prompt = str(prompt)
            inference_args.online_inference_enable_motion = True
            inference_args.online_inference_dry_run = False
            inference_args.online_inference_transform_config = (
                str(getattr(args, "online_inference_transform_config", "") or "").strip()
                or "configs/inference/unitree_dual_arm_identity_transform.json"
            )
            return create_online_inference_provider(inference_args)

        provider_runtime = TeleopProviderRuntime(
            live_provider=tv_wrapper,
            live_provider_name=args.input_provider,
            online_provider_factory=create_ui_online_provider,
        )
        gripper_ctrl = components.ee.gripper_ctrl
        loco_wrapper = components.loco_wrapper
        agv_bridge = components.agv_bridge
        base_state_receiver = components.base_state_receiver
        base_stop_state = {"latched": True, "fault": False, "error": ""}
        manual_waist_yaw_limit_state = {"active": False}
        mobile_coordinator = None
        mobile_kinematics = None
        dex1_tcp_fk = components.dex1_tcp_fk
        need_g1d_kinematics = (
            args.mobile_manipulation_mode == "mobile_ik_qp"
            or bool(args.record_mobile_training_state)
        )
        if need_g1d_kinematics:
            mobile_kinematics = G1DIkFrameKinematics(
                torso_from_ik=np.array([
                    [1.0, 0.0, 0.0, 0.00396],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -0.044],
                    [0.0, 0.0, 0.0, 1.0],
                ])
            )
        if args.mobile_manipulation_mode == "mobile_ik_qp":
            if base_state_receiver is None:
                raise RuntimeError("mobile_ik_qp requires the G1D odometry and height receiver")
            mobile_governor = WorkspaceGovernor(
                WorkspaceGovernorConfig(
                    mode=workspace_mode,
                    workspace_min=workspace_min.copy(),
                    workspace_max=workspace_max.copy(),
                    tapered=dict(tapered_workspace_params),
                    side_workspaces=side_workspaces,
                    max_forward_speed=float(args.base_max_vx),
                    max_base_yaw_rate=float(args.base_max_wz),
                    max_column_command=float(args.base_max_z),
                    column_speed_mps=0.10,
                    max_torso_yaw_rate=float(args.mobile_max_torso_yaw_rate),
                    column_minimum=0.0,
                    column_maximum=float(args.mobile_column_travel_m),
                )
            )
            mobile_coordinator = MobileManipulationCoordinator(
                mobile_governor,
                state_timeout_sec=float(args.mobile_state_timeout_sec),
            )
            logger_mp.info("[MOBILE_IK_QP] enabled: 4D base/column/torso governor with legacy G1_29 IK")
        elif args.record_mobile_training_state:
            logger_mp.info("[MOBILE_RECORD] enabled: base_link-local EEF pose and SLAM map base state")

        def current_mobile_state() -> MobileStateSample:
            odom_sample, height_sample = base_state_receiver.snapshot_latest()
            if odom_sample is None or height_sample is None:
                raise RuntimeError("MOBILE_STATE_MISSING")
            now_monotonic_ns = time.monotonic_ns()
            odom_age_ms = (now_monotonic_ns - int(odom_sample["t_ns"])) / 1e6
            height_age_ms = (now_monotonic_ns - int(height_sample["t_ns"])) / 1e6
            timeout_ms = float(args.mobile_state_timeout_sec) * 1e3
            if odom_age_ms > timeout_ms or height_age_ms > timeout_ms:
                raise RuntimeError(
                    "MOBILE_STATE_STALE "
                    f"odom_age_ms={odom_age_ms:.1f} height_age_ms={height_age_ms:.1f} "
                    f"timeout_ms={timeout_ms:.1f}"
                )
            pose = odom_sample["world_pose"]
            yaw = float(pose["yaw"])
            cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
            odom_world_from_agv = np.array([
                [cosine, -sine, 0.0, float(pose["x"])],
                [sine, cosine, 0.0, float(pose["y"])],
                [0.0, 0.0, 1.0, float(pose["z"])],
                [0.0, 0.0, 0.0, 1.0],
            ])
            column_position = column_position_from_raw_height(
                raw_height=float(height_sample["height"]["z"]),
                raw_minimum=float(args.mobile_height_raw_minimum),
                raw_maximum=float(args.mobile_height_raw_maximum),
                column_travel_m=float(args.mobile_column_travel_m),
            )
            torso_yaw = arm_ctrl.get_current_waist_yaw()
            global_from_ik = mobile_kinematics.global_from_ik(
                odom_world_from_agv=odom_world_from_agv,
                column_position=column_position,
                torso_yaw=torso_yaw,
            )
            return MobileStateSample(
                global_from_ik=global_from_ik,
                monotonic_ns=min(int(odom_sample["t_ns"]), int(height_sample["t_ns"])),
                column_position=column_position,
                torso_yaw=torso_yaw,
            )

        def get_robot_wrist_poses_base_link(arm_ik_obj, arm_q, column_height_m, waist_yaw):
            if mobile_kinematics is None:
                raise RuntimeError("base_link wrist kinematics is not initialized")
            # Current collection calibration explicitly defines base_link == AGV_link.
            base_link_from_ik = mobile_kinematics.agv_from_ik(
                column_position=float(column_height_m),
                torso_yaw=float(waist_yaw),
            )
            left_ik_pose, right_ik_pose = get_robot_wrist_poses(arm_ik_obj, arm_q)
            return base_link_from_ik @ left_ik_pose, base_link_from_ik @ right_ik_pose

        def get_robot_dex1_tcp_poses_base_link(arm_q, column_height_m, waist_yaw):
            if dex1_tcp_fk is None:
                raise RuntimeError("Dex1 TCP FK is not initialized")
            return dex1_tcp_fk.compute_tcp_poses(
                arm_q,
                column_height_m,
                waist_yaw,
            )

        def stop_base_once(reason: str) -> bool:
            if base_stop_state["latched"]:
                return True
            stop_result = stop_base_command(
                args=args,
                loco_wrapper=loco_wrapper,
                agv_bridge=agv_bridge,
            )
            base_stop_state["latched"] = stop_result.confirmed
            base_stop_state["fault"] = not stop_result.confirmed
            base_stop_state["error"] = stop_result.error
            if stop_result.confirmed:
                logger_mp.info("[BASE_CTRL] stop confirmed: %s", reason)
                return True
            logger_mp.error("[BASE_CTRL] STOP not confirmed (%s): %s", reason, stop_result.error)
            return False

        recorder = components.recorder
        recording_flow = components.recording_flow
        sim_state_subscriber = components.sim_state_subscriber
        base_state_receiver = components.base_state_receiver
        latency_tracker = components.latency_tracker
        if args.ui:
            ui_command_bus = UiCommandBus()
            ui_state_store = UiStateStore(
                build_runtime_web_payload(
                    args=args,
                    recorder=recorder,
                    recording_flow=recording_flow,
                    record_running=RECORD_RUNNING,
                    started=START,
                    ready=READY,
                    stopping=STOP,
                    provider_status=provider_runtime.status(),
                    base_state_receiver=base_state_receiver,
                    base_stop_state=base_stop_state,
                )
            )
            ui_server = TeleopUiServer(
                command_bus=ui_command_bus,
                state_store=ui_state_store,
                camera_status_getter=lambda: build_runtime_camera_status(components.cameras),
                camera_frame_getter=lambda camera_id: get_camera_frame_by_id(components.cameras, camera_id),
                host=args.ui_host,
                port=args.ui_port,
                publish_rate_hz=args.ui_preview_fps,
            )
            ui_server.start()
            logger_mp.info("[UI] web control enabled at http://%s:%d", ui_server.host, ui_server.port)

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
            if ui_command_bus is not None:
                keyboard_commands, provider_commands = split_ui_commands(ui_command_bus.drain())
                dispatch_ui_commands(keyboard_commands, on_press)
                for command in provider_commands:
                    if command.name == UiCommandName.SET_PROVIDER_HOLD:
                        provider_runtime.set_hold(reason="ui_wait_hold")
                    elif command.name == UiCommandName.SET_PROVIDER_XR:
                        provider_runtime.set_live(reason="ui_wait_xr")
                    elif command.name == UiCommandName.START_RAW_REPLAY:
                        provider_runtime.fail_raw_replay("teleop is not started; start teleop before real replay")
                    elif command.name == UiCommandName.STOP_RAW_REPLAY:
                        provider_runtime.stop_raw_replay(reason="ui_wait_stop")
                    elif command.name == UiCommandName.SET_RECORD_ROOT:
                        recorder, recording_flow = switch_recording_root_from_ui(
                            args=args,
                            components=components,
                            recorder=recorder,
                            recording_flow=recording_flow,
                            record_running=RECORD_RUNNING,
                            payload=command.payload or {},
                            log=logger_mp,
                        )
            if ui_state_store is not None:
                publish_ui_payload(
                    build_runtime_web_payload(
                        args=args,
                        recorder=recorder,
                        recording_flow=recording_flow,
                        record_running=RECORD_RUNNING,
                        started=START,
                        ready=READY,
                        stopping=STOP,
                        provider_status=provider_runtime.status(),
                        base_state_receiver=base_state_receiver,
                        base_stop_state=base_stop_state,
                    )
                )
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
            ui_provider_commands = []
            if ui_command_bus is not None:
                keyboard_commands, ui_provider_commands = split_ui_commands(ui_command_bus.drain())
                dispatch_ui_commands(keyboard_commands, on_press)
                if STOP:
                    break

            # record mode
            if recording_flow is not None:
                command_recording = recording_flow.handle_commands(
                    record_running=RECORD_RUNNING,
                    record_toggle=RECORD_TOGGLE,
                    record_cancel=RECORD_CANCEL,
                    camera_sources=components.cameras.sources(),
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
                        stop_base_once("calibration_wait")
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
                    stop_base_once("calibration_wait")
                    arm_ctrl.ctrl_dual_arm(calibration_hold_q.copy(), calibration_hold_tauff.copy())
                    time.sleep(0.01)
                    continue

            # get current robot state data first so XR wrapper can anchor each new
            # grip takeover to the *current* robot wrist pose instead of a fixed home pose.
            state_read_t0_ns = time.monotonic_ns()
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            current_waist_yaw = arm_ctrl.get_current_waist_yaw()
            state_read_t1_ns = time.monotonic_ns()
            current_state_sample_ns = int((state_read_t0_ns + state_read_t1_ns) // 2)
            append_timed_sample(
                state_history,
                current_state_sample_ns,
                q=current_lr_arm_q.copy(),
                dq=current_lr_arm_dq.copy(),
                waist_yaw=float(current_waist_yaw),
            )
            (
                left_gripper_feedback_q,
                right_gripper_feedback_q,
                left_gripper_command_q,
                right_gripper_command_q,
            ) = read_dual_gripper_snapshot(
                args=args,
                dual_gripper_data_lock=components.ee.dual_gripper_data_lock,
                dual_gripper_state_array=components.ee.dual_gripper_state_array,
                dual_gripper_action_array=components.ee.dual_gripper_action_array,
            )
            online_left_gripper_q = 0.0 if left_gripper_feedback_q is None else left_gripper_feedback_q
            online_right_gripper_q = 0.0 if right_gripper_feedback_q is None else right_gripper_feedback_q
            if latency_tracker is not None:
                latency_tracker.maybe_timeout()
                latency_tracker.maybe_mark_execute(
                    current_lr_arm_q,
                    current_lr_arm_dq,
                    q_threshold=args.latency_exec_q_threshold,
                    dq_threshold=args.latency_exec_dq_threshold,
                )
                if provider_runtime.active_provider_kind == ActiveProviderKind.ONLINE_INFERENCE:
                    provider_runtime.note_online_inference_execution_trace(latency_tracker.get_snapshot())
                elif provider_runtime.active_provider_kind == ActiveProviderKind.RAW_REPLAY:
                    provider_runtime.note_raw_replay_execution_trace(latency_tracker.get_snapshot())
            if ui_state_store is not None:
                publish_ui_payload(
                    build_runtime_web_payload(
                        args=args,
                        recorder=recorder,
                        recording_flow=recording_flow,
                        record_running=RECORD_RUNNING,
                        current_lr_arm_q=current_lr_arm_q,
                        current_left_gripper_q=left_gripper_feedback_q,
                        current_right_gripper_q=right_gripper_feedback_q,
                        current_left_gripper_cmd=left_gripper_command_q,
                        current_right_gripper_cmd=right_gripper_command_q,
                        current_state_sample_ns=current_state_sample_ns,
                        started=START,
                        ready=READY,
                        stopping=STOP,
                        provider_status=provider_runtime.status(),
                        base_state_receiver=base_state_receiver,
                        base_stop_state=base_stop_state,
                    )
                )
            current_left_wrist_pose, current_right_wrist_pose = get_robot_wrist_poses(arm_ik, current_lr_arm_q)

            camera_sources = components.cameras.sources()

            for command in ui_provider_commands:
                payload = command.payload or {}
                if command.name == UiCommandName.SET_PROVIDER_HOLD:
                    current_hold_q = current_lr_arm_q.copy()
                    current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
                    home_return_active = False
                    post_home_takeover_armed = False
                    was_online_inference = provider_runtime.active_provider_kind == ActiveProviderKind.ONLINE_INFERENCE
                    stop_base_once("ui_hold")
                    provider_runtime.set_hold(reason=str(payload.get("reason", "ui_hold")))
                    if was_online_inference:
                        set_online_inference_gripper_mode(gripper_ctrl, False)
                    logger_mp.info("[UI_PROVIDER] switched to HOLD.")
                elif command.name == UiCommandName.SET_RECORD_ROOT:
                    recorder, recording_flow = switch_recording_root_from_ui(
                        args=args,
                        components=components,
                        recorder=recorder,
                        recording_flow=recording_flow,
                        record_running=RECORD_RUNNING,
                        payload=payload,
                        log=logger_mp,
                    )
                elif command.name == UiCommandName.SET_PROVIDER_XR:
                    home_return_active = False
                    post_home_takeover_armed = False
                    stop_base_once("ui_xr")
                    provider_runtime.set_live(reason=str(payload.get("reason", "ui_xr")))
                    set_online_inference_gripper_mode(gripper_ctrl, False)
                    rebase_xr_takeover_after_provider_switch(
                        tv_wrapper=tv_wrapper,
                        arm_ik=arm_ik,
                        current_arm_q=current_lr_arm_q,
                        reset_arm_ik_state=reset_arm_ik_state,
                    )
                    operator_state_flow = OperatorStateFlow(
                        takeover_settle_frames=TAKEOVER_SETTLE_FRAMES,
                        log=logger_mp,
                    )
                    logger_mp.info("[UI_PROVIDER] switched to XR live.")
                elif command.name == UiCommandName.START_RAW_REPLAY:
                    if recording_is_active_or_armed(RECORD_RUNNING, recording_flow):
                        stop_base_once("raw_replay_rejected_recording")
                        provider_runtime.fail_raw_replay("recording is active or armed; stop/cancel recording before real replay")
                        logger_mp.error("[UI_REPLAY] rejected: recording is active or armed.")
                    else:
                        replay_base_source = str(payload.get("base_source", "none") or "none")
                        if replay_base_source not in {"none", "action"}:
                            stop_base_once("raw_replay_rejected_base_source")
                            provider_runtime.fail_raw_replay(
                                f"unsupported UI raw replay base_source: {replay_base_source}"
                            )
                            logger_mp.error("[UI_REPLAY] rejected: unsupported base_source=%s", replay_base_source)
                            continue
                        if replay_base_source == "action" and not args.base_motion:
                            stop_base_once("raw_replay_rejected_base_motion")
                            provider_runtime.fail_raw_replay(
                                "base_source=action requires restarting with --base-motion"
                            )
                            logger_mp.error("[UI_REPLAY] rejected: base_source=action requires --base-motion.")
                            continue
                        if replay_base_source == "action" and str(args.base_controller) != "g1d_agv":
                            stop_base_once("raw_replay_rejected_base_controller")
                            provider_runtime.fail_raw_replay(
                                "base_source=action requires restarting with --base-controller g1d_agv"
                            )
                            logger_mp.error(
                                "[UI_REPLAY] rejected: base_source=action requires base_controller=g1d_agv, got %s.",
                                args.base_controller,
                            )
                            continue
                        replay_arm_source = str(payload.get("arm_source", "action") or "action")
                        if args.mobile_manipulation_mode == "mobile_ik_qp" and replay_arm_source not in {"action", "state"}:
                            stop_base_once("raw_replay_rejected_mobile_motion_repr")
                            provider_runtime.fail_raw_replay(
                                "mobile_ik_qp raw replay requires arm_source=action or state; "
                                "fk_cmd_pose would invoke IK"
                            )
                            logger_mp.error(
                                "[UI_REPLAY] rejected: mobile_ik_qp raw replay requires joint action/state, got %s.",
                                replay_arm_source,
                            )
                            continue
                        current_hold_q = current_lr_arm_q.copy()
                        current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
                        home_return_active = False
                        post_home_takeover_armed = False
                        stop_base_once("raw_replay_start")
                        if provider_runtime.active_provider_kind != ActiveProviderKind.HOLD:
                            provider_runtime.set_hold(reason="raw_replay_start_sync_hold")
                            set_online_inference_gripper_mode(gripper_ctrl, False)
                        provider_runtime.start_raw_replay(
                            dataset_root=str(payload.get("dataset_root", "")),
                            episode_index=int(payload.get("episode_index", -1)),
                            episode_name=str(payload.get("episode_name", "")),
                            arm_source=replay_arm_source,
                            base_source=replay_base_source,
                            speed_scale=float(payload.get("speed_scale", 1.0)),
                        )
                        operator_state_flow = OperatorStateFlow(
                            takeover_settle_frames=0,
                            log=logger_mp,
                        )
                        logger_mp.info(
                            "[UI_REPLAY] started raw replay: root=%s episode=%s arm_source=%s base_source=%s speed_scale=%.3f",
                            payload.get("dataset_root", ""),
                            payload.get("episode_name", ""),
                            replay_arm_source,
                            replay_base_source,
                            float(payload.get("speed_scale", 1.0)),
                        )
                        if args.mobile_manipulation_mode == "mobile_ik_qp":
                            logger_mp.info(
                                "[UI_REPLAY] mobile_ik_qp bypassed: replaying recorded joint/base actions without QP or IK."
                            )
                elif command.name == UiCommandName.STOP_RAW_REPLAY:
                    current_hold_q = current_lr_arm_q.copy()
                    current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
                    home_return_active = False
                    post_home_takeover_armed = False
                    stop_base_once("raw_replay_stop")
                    provider_runtime.stop_raw_replay(reason=str(payload.get("reason", "ui_stop")))
                    logger_mp.info("[UI_REPLAY] stopped raw replay -> HOLD.")
                elif command.name == UiCommandName.START_ONLINE_INFERENCE:
                    if recording_is_active_or_armed(RECORD_RUNNING, recording_flow):
                        stop_base_once("online_inference_rejected_recording")
                        provider_runtime.fail_online_inference(
                            "recording is active or armed; stop/cancel recording before starting online inference"
                        )
                        logger_mp.error("[UI_INFERENCE] rejected: recording is active or armed.")
                    elif provider_runtime.active_provider_kind == ActiveProviderKind.RAW_REPLAY:
                        stop_base_once("online_inference_rejected_raw_replay")
                        provider_runtime.fail_online_inference("raw replay is active; stop replay before starting online inference")
                        logger_mp.error("[UI_INFERENCE] rejected: raw replay is active.")
                    else:
                        missing_cameras = missing_online_inference_camera_names(components.cameras)
                        if missing_cameras:
                            stop_base_once("online_inference_rejected_missing_cameras")
                            provider_runtime.fail_online_inference(
                                "missing required online inference cameras: " + ", ".join(missing_cameras)
                            )
                            logger_mp.error("[UI_INFERENCE] rejected: missing cameras=%s", missing_cameras)
                        else:
                            current_hold_q = current_lr_arm_q.copy()
                            current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
                            home_return_active = False
                            post_home_takeover_armed = False
                            stop_base_once("online_inference_start")
                            if provider_runtime.active_provider_kind != ActiveProviderKind.HOLD:
                                provider_runtime.set_hold(reason="online_inference_start_sync_hold")
                            provider_runtime.start_online_inference(prompt=str(payload.get("prompt", "")))
                            set_online_inference_gripper_mode(gripper_ctrl, True)
                            START = True
                            operator_state_flow = OperatorStateFlow(takeover_settle_frames=0, log=logger_mp)
                            logger_mp.info("[UI_INFERENCE] started HTTP pi0.5 online inference.")
                elif command.name == UiCommandName.STOP_ONLINE_INFERENCE:
                    current_hold_q = current_lr_arm_q.copy()
                    current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
                    home_return_active = False
                    post_home_takeover_armed = False
                    stop_base_once("online_inference_stop")
                    provider_runtime.stop_online_inference(reason=str(payload.get("reason", "ui_stop")))
                    set_online_inference_gripper_mode(gripper_ctrl, False)
                    logger_mp.info("[UI_INFERENCE] stopped -> HOLD.")

            if provider_runtime.active_provider_kind == ActiveProviderKind.HOLD:
                stop_base_once("provider_hold")
                arm_ctrl.ctrl_dual_arm(current_hold_q.copy(), current_hold_tauff.copy())
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / args.frequency) - time_elapsed)
                time.sleep(sleep_time)
                continue

            active_provider = provider_runtime.active_provider()
            active_input_provider = provider_runtime.active_input_provider_name()
            is_ui_raw_replay = provider_runtime.active_provider_kind == ActiveProviderKind.RAW_REPLAY
            is_ui_online_inference = provider_runtime.active_provider_kind == ActiveProviderKind.ONLINE_INFERENCE
            raw_replay_base_source = "none"
            if is_ui_raw_replay:
                raw_replay_base_source = str(
                    (provider_runtime.status().get("real_replay") or {}).get("base_source", "none") or "none"
                )

            # get active provider tele data
            tele_fetch_start = time.perf_counter()
            provider_get_sample_kwargs = {
                "current_left_robot_wrist_pose": current_left_wrist_pose,
                "current_right_robot_wrist_pose": current_right_wrist_pose,
                "current_state_host_monotonic_ns": current_state_sample_ns,
                "current_arm_q": current_lr_arm_q,
                "current_arm_dq": current_lr_arm_dq,
                "current_left_gripper_width": online_left_gripper_q,
                "current_right_gripper_width": online_right_gripper_q,
                "camera_sources": camera_sources,
                "dt": control_dt,
            }
            if is_ui_raw_replay and ui_command_bus is not None:
                provider_get_sample_kwargs["raw_replay_stop_requested"] = ui_command_bus.raw_replay_stop_requested
            sample = active_provider.get_sample(
                **provider_get_sample_kwargs,
            )
            tele_fetch_dt = time.perf_counter() - tele_fetch_start
            timing_debugger.add_tele_fetch(tele_fetch_dt, sample is not None)
            if is_ui_online_inference and ui_command_bus is not None and ui_command_bus.online_inference_stop_requested():
                current_hold_q = current_lr_arm_q.copy()
                current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
                stop_base_once("online_inference_stop_pending")
                provider_runtime.stop_online_inference(reason="ui_stop_pending_after_inference_fetch")
                set_online_inference_gripper_mode(gripper_ctrl, False)
                arm_ctrl.ctrl_dual_arm(current_hold_q.copy(), current_hold_tauff.copy())
                logger_mp.info("[UI_INFERENCE] discarded fetched action because stop is pending -> HOLD.")
                continue
            if sample is None:
                if is_ui_raw_replay:
                    if bool(getattr(active_provider, "stop_interrupted", False)):
                        stop_base_once("raw_replay_stop_interrupted")
                        provider_runtime.stop_raw_replay(reason="ui_stop_interrupt")
                        logger_mp.info("[UI_REPLAY] stop request interrupted raw replay frame wait -> HOLD.")
                    else:
                        stop_base_once("raw_replay_finished")
                        provider_runtime.finish_raw_replay(reason="provider_done")
                    current_hold_q = current_lr_arm_q.copy()
                    current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
                    arm_ctrl.ctrl_dual_arm(current_hold_q.copy(), current_hold_tauff.copy())
                    logger_mp.info("[UI_REPLAY] raw replay provider finished -> HOLD.")
                    current_time = time.time()
                    time_elapsed = current_time - start_time
                    sleep_time = max(0, (1 / args.frequency) - time_elapsed)
                    time.sleep(sleep_time)
                    continue
                if bool(getattr(active_provider, "done", False)):
                    if active_input_provider == "lerobot_offline":
                        exit_go_home = args.offline_replay_end_action == "home"
                        logger_mp.info(
                            "[OFFLINE_REPLAY] input provider finished. end_action=%s, stopping main loop.",
                            args.offline_replay_end_action,
                        )
                    START = False
                    STOP = True
                    stop_base_once("offline_replay_finished")
                    continue
                timing_debugger.maybe_report(arm_ctrl=arm_ctrl, gripper_ctrl=gripper_ctrl)
                time.sleep(0.01)
                continue
            provider_runtime.note_sample(sample)
            tele_data = sample.tele_data
            motion_intent = sample.motion_intent
            if is_ui_raw_replay and not args.no_gripper and args.ee == "dex1":
                recorded_gripper_state_q = (getattr(motion_intent, "metadata", {}) or {}).get(
                    "raw_replay_recorded_gripper_state_q"
                )
                if not isinstance(recorded_gripper_state_q, list) or len(recorded_gripper_state_q) != 2:
                    raise ValueError("raw replay sample is missing recorded dual-gripper state")
                if left_gripper_feedback_q is None or right_gripper_feedback_q is None:
                    raise RuntimeError("Dex1 gripper feedback is unavailable during raw replay")
                provider_runtime.note_raw_replay_gripper_state_pair(
                    frame_index=int(getattr(motion_intent, "frame_index", -1)),
                    left_feedback_q=left_gripper_feedback_q,
                    right_feedback_q=right_gripper_feedback_q,
                    left_recorded_state_q=float(recorded_gripper_state_q[0]),
                    right_recorded_state_q=float(recorded_gripper_state_q[1]),
                )
            if active_input_provider == "online_inference" and bool(getattr(sample, "done", False)):
                metadata = getattr(motion_intent, "metadata", {}) or {}
                logger_mp.error(
                    "[ONLINE_INFERENCE] provider entered fail-closed state: status=%s error=%s",
                    metadata.get("online_inference_status"),
                    metadata.get("error"),
                )
                stop_base_once("online_inference_failed")
                provider_runtime.fail_online_inference(str(metadata.get("error", "online inference provider failed")))
                set_online_inference_gripper_mode(gripper_ctrl, False)
                current_hold_q = current_lr_arm_q.copy()
                current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
                arm_ctrl.ctrl_dual_arm(current_hold_q.copy(), current_hold_tauff.copy())
                continue
            tele_data_recv_ts_ns = time.perf_counter_ns()
            tele_fetch_ms = tele_fetch_dt * 1000.0

            if is_ui_raw_replay:
                left_arm_enabled = True
                right_arm_enabled = True
                home_return_active = False
                post_home_takeover_armed = False
                left_takeover_settle_frames = 0
                right_takeover_settle_frames = 0
                left_takeover_rising_edge = False
                right_takeover_rising_edge = False
                left_zero_takeover_this_frame = False
                right_zero_takeover_this_frame = False
                any_zero_takeover_this_frame = False
                takeover_logic_ms = 0.0
            else:
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
                    current_lr_arm_q=current_lr_arm_q,
                    reset_arm_ik_state=reset_arm_ik_state,
                    timer=time.perf_counter,
                )
                tele_data = operator_state.tele_data
                left_arm_enabled = operator_state.left_arm_enabled
                right_arm_enabled = operator_state.right_arm_enabled
                home_return_active = operator_state.home_return_active
                if operator_state.home_return_interrupted:
                    current_hold_q = current_lr_arm_q.copy()
                    current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
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

            # The direct path preserves legacy behavior. The mobile path forms one
            # coordinated body target before the existing hardware dispatch.
            runtime_base_source = resolve_runtime_base_command_source(
                active_input_provider=active_input_provider,
                is_ui_raw_replay=is_ui_raw_replay,
                raw_replay_base_source=raw_replay_base_source,
            )
            manual_torso_yaw_rate = map_manual_torso_yaw_rate(
                args=args,
                tele_data=tele_data,
                home_return_active=home_return_active,
            )
            if args.mobile_manipulation_mode == "mobile_ik_qp" and not is_ui_raw_replay:
                nominal_source = runtime_base_source or "controller"
                nominal_intent = map_base_command(
                    args=args,
                    tele_data=tele_data,
                    home_return_active=home_return_active,
                    base_intent=getattr(sample, "base_intent", None),
                    base_command_source=nominal_source,
                )
                coordinated = mobile_coordinator.step(
                    motion_intent=motion_intent,
                    enabled={"left": left_arm_enabled, "right": right_arm_enabled},
                    rising={"left": left_takeover_rising_edge, "right": right_takeover_rising_edge},
                    nominal_body_command=np.array([
                        nominal_intent.vx,
                        nominal_intent.wz,
                        nominal_intent.z,
                        manual_torso_yaw_rate,
                    ]),
                    mobile_state=current_mobile_state(),
                    now_monotonic_ns=time.monotonic_ns(),
                    dt=control_dt,
                    home_active=home_return_active,
                    stop_active=STOP,
                )
                motion_intent = coordinated.motion_intent_for_ik
                arm_ctrl.set_waist_yaw_target(coordinated.waist_yaw_target)
                final_base_intent = BaseCommandIntent(
                    vx=float(coordinated.final_body_command[0]),
                    vy=0.0,
                    wz=float(coordinated.final_body_command[1]),
                    z=float(coordinated.final_body_command[2]),
                    source="mobile_ik_qp",
                )
                runtime_base_source = "provider"
                base_intent = final_base_intent
                base_provider_active = True
            else:
                if not is_ui_raw_replay:
                    if arm_ctrl.waist_yaw_target is None:
                        raise RuntimeError("manual waist yaw requires an initialized waist yaw target")
                    waist_yaw_target, waist_yaw_saturated = integrate_manual_torso_yaw_target(
                        current_target_rad=float(arm_ctrl.waist_yaw_target),
                        yaw_rate_radps=manual_torso_yaw_rate,
                        dt=control_dt,
                    )
                    arm_ctrl.set_waist_yaw_target(waist_yaw_target)
                    if waist_yaw_saturated and not manual_waist_yaw_limit_state["active"]:
                        logger_mp.warning(
                            "[WAIST_YAW] manual target saturated at %.4f rad (limit=[-2.7053, 2.7053])",
                            waist_yaw_target,
                        )
                    manual_waist_yaw_limit_state["active"] = waist_yaw_saturated
                base_intent = getattr(sample, "base_intent", None)
                base_provider_active = active_input_provider in {"lerobot_offline", "online_inference"} or is_ui_raw_replay
            base_result = apply_base_command(
                args=args,
                tele_data=tele_data,
                home_return_active=home_return_active,
                loco_wrapper=loco_wrapper,
                agv_bridge=agv_bridge,
                timing_debugger=timing_debugger,
                base_intent=base_intent,
                base_provider_active=base_provider_active,
                base_stop_latched=base_stop_state["latched"],
                base_stop_fault=base_stop_state["fault"],
                base_command_source=runtime_base_source,
            )
            base_stop_state["latched"] = base_result.base_stop_latched
            base_stop_state["fault"] = base_result.base_stop_fault
            base_stop_state["error"] = base_result.base_stop_error
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
            if args.record_base:
                base_action_source = "g1d_agv_bridge" if base_control_mode == "g1d_agv_async" else base_control_mode
                append_timed_sample(
                    base_action_history,
                    int(time.monotonic_ns()),
                    vx_cmd=float(base_vx),
                    vy_cmd=float(base_vy),
                    wz_cmd=float(base_wz),
                    z_cmd=float(base_z),
                    frame_id="base_link",
                    source=base_action_source,
                    base_control_mode=base_control_mode,
                )
            if base_result.stop_requested:
                stop_base_once("controller_stop")
                START = False
                STOP = True
            if base_result.base_stop_fault:
                logger_mp.error("[BASE_CTRL] base stop fault; retrying STOP: %s", base_result.base_stop_error)
                time.sleep(control_dt)
                continue
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
                    input_provider=active_input_provider,
                    frequency=args.frequency,
                    max_arm_joint_speed=args.max_arm_joint_speed,
                    home_return_speed=args.home_return_speed,
                    workspace_limit_enabled=workspace_limit_enabled,
                    workspace_mode=workspace_mode,
                    workspace_min=workspace_min,
                    workspace_max=workspace_max,
                    tapered_workspace_params=tapered_workspace_params,
                    compute_arm_gravity_tauff=compute_arm_gravity_tauff,
                    side_workspaces=side_workspaces,
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
            if active_input_provider == "online_inference" and provider_feedback is not None:
                provider_feedback_start = time.perf_counter()
                report_feedback = getattr(active_provider, "report_control_feedback", None)
                if callable(report_feedback):
                    report_feedback(provider_feedback)
                provider_feedback_ms = (time.perf_counter() - provider_feedback_start) * 1000.0

            trace_seq = None
            if (
                latency_tracker is not None
                and latency_tracker.can_start_new_trace()
                and (
                    latency_tracker.tracks_input_provider(active_input_provider)
                )
            ):
                latency_trace_prepare_start = time.perf_counter()
                max_command_delta = float(np.max(np.abs(sol_q - current_lr_arm_q)))
                if max_command_delta >= args.latency_command_threshold:
                    provider_trace_extra = {}
                    if active_input_provider == "online_inference" and motion_intent is not None:
                        provider_trace_extra = {
                            key: value
                            for key, value in (getattr(motion_intent, "metadata", {}) or {}).items()
                            if str(key).startswith("online_")
                        }
                        provider_trace_extra["online_provider_output_perf_ns"] = int(tele_data_recv_ts_ns)
                    elif is_ui_raw_replay and motion_intent is not None:
                        provider_trace_extra = {
                            "raw_replay_frame_index": int(getattr(motion_intent, "frame_index", -1)),
                            "raw_replay_provider_output_perf_ns": int(tele_data_recv_ts_ns),
                        }
                    trace_seq = latency_tracker.begin_trace(
                        recv_ts_ns=tele_data_recv_ts_ns,
                        recv_q=current_lr_arm_q,
                        extra={
                            **provider_trace_extra,
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
            waist_yaw_target = arm_ctrl.waist_yaw_target
            if args.record_mobile_training_state and waist_yaw_target is None:
                raise RuntimeError("mobile training state requires an initialized waist yaw target")
            append_timed_sample(
                action_history,
                action_command_monotonic_ns,
                q=sol_q.copy(),
                tauff=sol_tauff.copy(),
                waist_yaw_target=(None if waist_yaw_target is None else float(waist_yaw_target)),
            )
            action_history_append_ms = (time.perf_counter() - action_history_start) * 1000.0
            if trace_seq is not None:
                latency_tracker.set_fields(trace_seq, action_history_append_ms=action_history_append_ms)
            ctrl_dual_arm_start = time.perf_counter()
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff, trace_seq=trace_seq)
            ctrl_dual_arm_call_ms = (time.perf_counter() - ctrl_dual_arm_start) * 1000.0
            if trace_seq is not None:
                latency_tracker.set_fields(trace_seq, ctrl_dual_arm_call_ms=ctrl_dual_arm_call_ms)
            if active_input_provider == "online_inference":
                metadata = getattr(motion_intent, "metadata", {}) or {}
                provider_runtime.note_online_inference_runtime_debug(
                    {
                        "updated_monotonic_ns": int(time.monotonic_ns()),
                        "latency": {
                            "http_roundtrip_ms": metadata.get("online_obs_send_to_action_recv_ms"),
                            "tele_fetch_ms": float(tele_fetch_ms),
                            "ik_ms": float(ik_ms),
                            "safety_ms": float(safety_ms),
                            "gravity_ms": float(gravity_ms),
                            "provider_feedback_ms": float(provider_feedback_ms),
                            "target_submit_ms": float(ctrl_dual_arm_call_ms),
                        },
                        "safety": {
                            "provider_feedback": provider_feedback,
                            "command_delta_max_abs": float(np.max(np.abs(sol_q - current_lr_arm_q))),
                            "command_delta_l2": float(np.linalg.norm(sol_q - current_lr_arm_q)),
                            "target_submitted": True,
                        },
                        "trajectory": {
                            "sample_monotonic_ns": int(time.monotonic_ns()),
                            "left_target": wrist_pose_to_debug_sample(motion_intent.left_wrist_pose),
                            "right_target": wrist_pose_to_debug_sample(motion_intent.right_wrist_pose),
                            "left_feedback": wrist_pose_to_debug_sample(current_left_wrist_pose),
                            "right_feedback": wrist_pose_to_debug_sample(current_right_wrist_pose),
                            "left_target_xyz": np.asarray(motion_intent.left_wrist_pose, dtype=float)[:3, 3].tolist(),
                            "right_target_xyz": np.asarray(motion_intent.right_wrist_pose, dtype=float)[:3, 3].tolist(),
                            "left_feedback_xyz": np.asarray(current_left_wrist_pose, dtype=float)[:3, 3].tolist(),
                            "right_feedback_xyz": np.asarray(current_right_wrist_pose, dtype=float)[:3, 3].tolist(),
                        },
                    }
                )
                if latency_tracker is not None:
                    provider_runtime.note_online_inference_execution_trace(latency_tracker.get_snapshot())
            elif is_ui_raw_replay and latency_tracker is not None:
                provider_runtime.note_raw_replay_execution_trace(latency_tracker.get_snapshot())
            if is_ui_raw_replay and bool(getattr(sample, "done", False)):
                current_hold_q = sol_q.copy()
                current_hold_tauff = sol_tauff.copy()
                stop_base_once("raw_replay_final_frame")
                provider_runtime.finish_raw_replay(reason="sample_done")
                logger_mp.info("[UI_REPLAY] raw replay reached final frame -> HOLD.")
            if home_return_active and np.all(np.abs(sol_q - home_target_q) < 0.05):
                home_return_active = False
                calibration_hold_q = current_hold_q.copy()
                calibration_hold_tauff = current_hold_tauff.copy()
                logger_mp.info("[HOME] reached ready/calibration pose. Waiting for both grips to release before teleop resumes.")

            # record data
            if recording_flow is not None:
                base_state_history_snapshot = None
                base_height_history_snapshot = None
                slam_tf_history_snapshot = None
                if args.record_base:
                    if base_state_receiver is None:
                        raise RuntimeError("--record-base is enabled but base_state_receiver is not initialized")
                    if not base_state_receiver.is_alive():
                        raise RuntimeError("--record-base receiver thread is not alive")
                    base_state_history_snapshot, base_height_history_snapshot = base_state_receiver.snapshot_histories()
                    slam_tf_history_snapshot = base_state_receiver.snapshot_slam_tf_history()
                frame_recording = recording_flow.process_frame(
                    record_running=RECORD_RUNNING,
                    camera_sources=camera_sources,
                    state_history=state_history,
                    action_history=action_history,
                    teleop_input_perf_counter_ns=tele_data_recv_ts_ns,
                    arm_ik=arm_ik,
                    get_wrist_poses=get_robot_wrist_poses,
                    get_wrist_poses_base_link=(
                        get_robot_wrist_poses_base_link
                        if args.record_mobile_training_state
                        else None
                    ),
                    get_dex1_tcp_poses_base_link=(
                        get_robot_dex1_tcp_poses_base_link
                        if args.record_mobile_training_state
                        else None
                    ),
                    sim_state_subscriber=sim_state_subscriber,
                    base_state_history=base_state_history_snapshot,
                    base_height_history=base_height_history_snapshot,
                    base_action_history=base_action_history if args.record_base else None,
                    slam_tf_history=slam_tf_history_snapshot,
                )
                RECORD_RUNNING = frame_recording.record_running
                READY = frame_recording.ready
                if ui_state_store is not None:
                    publish_ui_payload(
                        build_runtime_web_payload(
                            args=args,
                            recorder=recorder,
                            recording_flow=recording_flow,
                            record_running=RECORD_RUNNING,
                            current_lr_arm_q=current_lr_arm_q,
                            current_left_gripper_q=left_gripper_feedback_q,
                            current_right_gripper_q=right_gripper_feedback_q,
                            current_left_gripper_cmd=left_gripper_command_q,
                            current_right_gripper_cmd=right_gripper_command_q,
                            current_state_sample_ns=current_state_sample_ns,
                            started=START,
                            ready=READY,
                            stopping=STOP,
                            provider_status=provider_runtime.status(),
                            base_state_receiver=base_state_receiver,
                            base_stop_state=base_stop_state,
                        )
                    )
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
            validation_manager=components.validation_manager,
            sim_state_subscriber=sim_state_subscriber or components.sim_state_subscriber,
            base_state_receiver=base_state_receiver or components.base_state_receiver,
            loco_wrapper=loco_wrapper or components.loco_wrapper,
            agv_bridge=agv_bridge or components.agv_bridge,
            cameras=components.cameras.close_list(),
            ui_server=ui_server,
            exit_go_home=exit_go_home,
            exit_home_hold_sec=exit_home_hold_sec,
        )
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
