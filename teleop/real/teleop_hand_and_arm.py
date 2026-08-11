"""真机遥操主入口：只编排启动、主循环和各职责模块的接口调用。"""

import base64
import re
import shutil
import time
import threading
from collections import deque
from pathlib import Path
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
from core.input.base import BaseCommandIntent, MotionIntent
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
    resolve_waist_yaw_position_target,
    resolve_runtime_base_command_source,
    stop_base_command,
)
from teleop.control_flow.mobile_manipulation_coordinator import (
    G1DIkFrameKinematics,
    legacy_g1_29_torso_from_ik_urdf,
    MobileManipulationCoordinator,
)
from teleop.control_flow.mobile_state_provider import MobileStateProvider
from teleop.control_flow.mobile_base_kinematics import MobileBaseKinematicsProvider
from teleop.control_flow.arm_workspace_config import build_arm_side_workspaces
from teleop.control_flow.arm_command_pipeline import build_arm_command
from teleop.control_flow.operator_state import OperatorStateFlow, rebase_xr_takeover_after_provider_switch
from teleop.control_flow.end_effector_command import (
    apply_end_effector_command,
    read_dual_gripper_snapshot,
)
from teleop.runtime.provider_switch import ActiveProviderKind, TeleopProviderRuntime
from teleop.runtime.mobile_inference_inputs import MobileOnlineInferenceInputs
from teleop.ui.command_bus import UiCommandBus, UiCommandName
from teleop.ui.integration import (
    build_runtime_camera_status,
    build_runtime_web_payload,
    dispatch_ui_commands,
)
from teleop.ui.server import TeleopUiServer
from teleop.ui.episode_store import PlaybackSession, resolve_root
from teleop.ui.exporter import DEFAULT_UI_URDF_PATH, UiExportManager, UiExportRequest
from teleop.ui.nero_storage import (
    export_output_details,
    export_output_episodes,
    update_episode_task_description,
)
from teleop.ui.state_store import UiStateStore
from teleop.ui.online_inference_runtime import OnlineInferenceUiRuntime
from teleop.nero_console.extensions import map_xr_teleoperation_extension_action
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
    UiCommandName.LOAD_OFFLINE_REPLAY,
    UiCommandName.START_OFFLINE_REPLAY,
    UiCommandName.PAUSE_OFFLINE_REPLAY,
    UiCommandName.RESUME_OFFLINE_REPLAY,
    UiCommandName.STOP_OFFLINE_REPLAY,
    UiCommandName.SEEK_OFFLINE_REPLAY,
    UiCommandName.OFFLINE_REPLAY_CURVES,
    UiCommandName.OFFLINE_REPLAY_IMAGE,
    UiCommandName.LOAD_ONLINE_REPLAY,
    UiCommandName.DELETE_EPISODES,
    UiCommandName.UPDATE_EPISODE_TASK_DESCRIPTION,
    UiCommandName.START_EXPORT,
    UiCommandName.EXPORT_OUTPUTS,
    UiCommandName.EXPORT_OUTPUT_GET,
    UiCommandName.EXPORT_OUTPUT_EPISODES_GET,
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


RECORDING_UI_COMMANDS = {
    UiCommandName.START_RECORDING,
    UiCommandName.STOP_RECORDING,
}


def complete_ui_command(command_bus, command, *, accepted: bool, message: str, details: dict | None = None) -> None:
    """Complete only Nero-tracked commands; legacy UI remains fire-and-forget."""
    if command.track_completion:
        if not command_bus.has_pending_completion(command.request_id):
            return
        if command_bus.completion_for(command.request_id) is not None:
            return
        if accepted:
            command_bus.succeed(command, message, details=details)
        else:
            command_bus.reject(command, message, details=details)


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
    data = model.createData()
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
    if arm_ik_obj is None:
        raise RuntimeError("arm gravity compensation requires an initialized arm_ik")
    model = arm_ik_obj.reduced_robot.model
    data = arm_ik_obj.reduced_robot.data
    q = np.asarray(arm_q, dtype=float).reshape(-1)
    if q.shape != (model.nq,) or not np.all(np.isfinite(q)):
        raise ValueError(f"arm gravity compensation q must be a finite ({model.nq},) vector")
    dq = np.zeros(model.nv, dtype=float)
    ddq = np.zeros(model.nv, dtype=float)
    tauff = np.asarray(pin.rnea(model, data, q, dq, ddq), dtype=float).reshape(-1)
    if tauff.shape != (model.nv,) or not np.all(np.isfinite(tauff)):
        raise RuntimeError(f"arm gravity compensation must return a finite ({model.nv},) vector")
    return tauff


def ctrl_g1_29_dual_arm_go_home(*, args, arm_ctrl, arm_ik) -> None:
    if str(args.arm) != "G1_29":
        arm_ctrl.ctrl_dual_arm_go_home()
        return
    if arm_ik is None:
        raise RuntimeError("G1_29 Home requires an initialized arm_ik for gravity compensation")
    arm_ctrl.ctrl_dual_arm_go_home(
        gravity_tauff_fn=lambda arm_q: compute_arm_gravity_tauff(arm_ik, arm_q),
        position_tolerance_rad=float(args.home_position_tolerance_rad),
        gravity_update_hz=float(args.home_gravity_update_hz),
        gravity_torque_limit_nm=float(args.home_gravity_torque_limit_nm),
    )


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
    arm_ik,
    recording_flow,
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
    nero_console_runtime,
    nero_preview_cache,
    nero_episode_index,
    exit_go_home: bool,
    exit_home_hold_sec: float,
):
    if ui_server is not None:
        ui_server.stop()
    if nero_console_runtime is not None:
        nero_console_runtime.stop()
    if nero_preview_cache is not None:
        nero_preview_cache.stop()
    if nero_episode_index is not None:
        nero_episode_index.stop()

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

    if arm_ctrl is not None:
        if exit_go_home:
            logger_mp.info("[EXIT] returning dual arms to home before shutdown...")
            ctrl_g1_29_dual_arm_go_home(args=args, arm_ctrl=arm_ctrl, arm_ik=arm_ik)
        logger_mp.warning(
            "[EXIT] holding dual arms at home for %.1fs before shutdown. "
            "Keep clear and support the robot if needed.",
            exit_home_hold_sec,
        )
        time.sleep(exit_home_hold_sec)

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

    if recording_flow is not None:
        recording_flow.close()

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
    nero_console_runtime = None
    nero_projection = None
    nero_preview_cache = None
    nero_episode_index = None
    nero_episode_tracking = {"root": "", "active": False}
    offline_replay_session = None
    export_manager = None
    online_replay_selection = {"state": "idle", "loaded": False, "episode_name": "", "replay_rate": 1.0, "error": ""}
    provider_runtime = None
    exit_go_home = True
    exit_home_hold_sec = 5.0
    args = parse_args()
    online_replay_selection.update(
        {
            "arm_source": str(args.nero_online_replay_arm_source),
            "base_source": str(args.nero_online_replay_base_source),
        }
    )
    if args.nero_console_provider:
        from teleop.nero_console.runtime import require_rclpy

        require_rclpy()
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
    timing_debugger = TimingDebugger(
        enabled=args.timing_debug,
        interval_sec=args.timing_debug_interval,
        collect_for_ui=args.ui,
    )
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
        if nero_projection is not None:
            nero_projection.refresh_control_thread_cache(payload)
        if nero_episode_index is not None:
            recording = payload.get("recording") if isinstance(payload, dict) else None
            recording = recording if isinstance(recording, dict) else {}
            root = str(recording.get("active_root_dir") or recording.get("root_dir") or "").strip()
            active = bool(recording.get("active"))
            if root and root != nero_episode_tracking["root"] and not active:
                nero_episode_index.request_refresh(root)
            if root:
                nero_episode_tracking["root"] = root
            if nero_episode_tracking["active"] and not active and root:
                nero_episode_index.request_refresh(root)
            nero_episode_tracking["active"] = active

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
        gripper_ctrl = components.ee.gripper_ctrl
        loco_wrapper = components.loco_wrapper
        agv_bridge = components.agv_bridge
        base_state_receiver = components.base_state_receiver
        base_stop_state = {"latched": True, "fault": False, "error": ""}
        manual_waist_yaw_limit_state = {"active": False}
        mobile_coordinator = None
        mobile_kinematics = None
        dex1_tcp_fk = components.dex1_tcp_fk
        mobile_base_kinematics = None
        need_g1d_kinematics = (
            args.mobile_manipulation_mode == "mobile_ik_qp"
            or bool(args.record_mobile_training_state)
            or dex1_tcp_fk is not None
        )
        if need_g1d_kinematics:
            mobile_kinematics = G1DIkFrameKinematics(
                torso_from_ik=legacy_g1_29_torso_from_ik_urdf(),
            )
            mobile_base_kinematics = MobileBaseKinematicsProvider(
                g1d_kinematics=mobile_kinematics,
                legacy_wrist_pose_solver=get_robot_wrist_poses,
                dex1_tcp_fk=dex1_tcp_fk,
            )
        if args.mobile_manipulation_mode == "mobile_ik_qp":
            if base_state_receiver is None:
                raise RuntimeError("mobile_ik_qp requires the G1D odometry and height receiver")
            mobile_coordinator = MobileManipulationCoordinator(
                state_timeout_sec=float(args.mobile_state_timeout_sec),
                column_travel_m=float(args.mobile_column_travel_m),
                command_horizon_sec=float(args.mobile_wbc_command_horizon_sec),
                max_position_lead_rad=float(args.mobile_wbc_max_position_lead_rad),
                enable_collision_avoidance=not bool(args.disable_mobile_wbc_collision_avoidance),
            )
            logger_mp.info(
                "[MOBILE_IK_QP] enabled: measured-state whole-body QP; "
                "collision_avoidance=%s, command_horizon=%.3fs, max_position_lead=%.3frad",
                not bool(args.disable_mobile_wbc_collision_avoidance),
                float(args.mobile_wbc_command_horizon_sec),
                float(args.mobile_wbc_max_position_lead_rad),
            )
        elif args.record_mobile_training_state:
            logger_mp.info("[MOBILE_RECORD] enabled: base_link-local EEF pose and SLAM map base state")

        mobile_online_inputs = MobileOnlineInferenceInputs(
            args=args,
            base_state_receiver=base_state_receiver,
            mobile_base_kinematics=mobile_base_kinematics,
            waist_yaw_getter=arm_ctrl.get_current_waist_yaw,
        )
        ui_online_inference = OnlineInferenceUiRuntime(
            args=args,
            mobile_inputs=mobile_online_inputs,
        )

        provider_runtime = TeleopProviderRuntime(
            live_provider=tv_wrapper,
            live_provider_name=args.input_provider,
            online_provider_factory=ui_online_inference.create_provider,
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
        mobile_state_provider = None
        if args.mobile_manipulation_mode == "mobile_ik_qp":
            if base_state_receiver is None or mobile_kinematics is None:
                raise RuntimeError("mobile_ik_qp requires initialized base state and G1D kinematics")

            def enter_mobile_state_retry_hold() -> None:
                hold_q = arm_ctrl.get_current_dual_arm_q()
                hold_tauff = compute_arm_gravity_tauff(arm_ik, hold_q)
                arm_ctrl.ctrl_dual_arm(hold_q, hold_tauff)
                if not stop_base_once("mobile_state_stale_retry"):
                    raise RuntimeError("MOBILE_STATE_RETRY_STOP_UNCONFIRMED")

            mobile_state_provider = MobileStateProvider(
                receiver=base_state_receiver,
                kinematics=mobile_kinematics,
                state_timeout_sec=float(args.mobile_state_timeout_sec),
                retry_count=int(args.mobile_state_retry_count),
                retry_interval_sec=float(args.mobile_state_retry_interval_sec),
                raw_height_minimum=float(args.mobile_height_raw_minimum),
                raw_height_maximum=float(args.mobile_height_raw_maximum),
                column_travel_m=float(args.mobile_column_travel_m),
                enter_fail_closed_hold=enter_mobile_state_retry_hold,
                log=logger_mp,
            )
        if args.ui or args.nero_console_provider:
            ui_command_bus = UiCommandBus()
            offline_replay_session = PlaybackSession()
            export_manager = UiExportManager()
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
                    latency_snapshot=latency_tracker.get_snapshot() if latency_tracker is not None else None,
                    timing_snapshot=timing_debugger.snapshot(),
                    base_state_receiver=base_state_receiver,
                    base_stop_state=base_stop_state,
                )
            )
            if args.ui:
                ui_server = TeleopUiServer(
                    command_bus=ui_command_bus,
                    state_store=ui_state_store,
                    camera_status_getter=lambda: build_runtime_camera_status(components.cameras),
                    camera_frame_getter=lambda camera_id: get_camera_frame_by_id(components.cameras, camera_id),
                    inference_profiles_getter=ui_online_inference.profiles,
                    host=args.ui_host,
                    port=args.ui_port,
                    publish_rate_hz=args.ui_preview_fps,
                )
                ui_server.playback = offline_replay_session
                ui_server.convert_manager = export_manager
                ui_server.start()
                logger_mp.info("[UI] web control enabled at http://%s:%d", ui_server.host, ui_server.port)
            if args.nero_console_provider:
                from teleop.nero_console.projection import XrConsoleProjection
                from teleop.nero_console.cache_workers import EpisodeIndexCache, PreviewCache
                from teleop.nero_console.runtime import NeroConsoleRuntime

                nero_projection = XrConsoleProjection(
                    state_store=ui_state_store,
                    args=args,
                    camera_status_getter=lambda: build_runtime_camera_status(components.cameras),
                    camera_frame_getter=lambda camera_id: get_camera_frame_by_id(components.cameras, camera_id),
                    offline_replay_status_getter=offline_replay_session.status,
                    online_replay_status_getter=lambda: dict(online_replay_selection),
                    export_status_getter=export_manager.status,
                    inference_profiles_getter=ui_online_inference.profiles,
                )
                nero_preview_cache = PreviewCache(
                    cache=nero_projection.cache,
                    camera_frame_getter=lambda camera_id: get_camera_frame_by_id(components.cameras, camera_id),
                )
                nero_episode_index = EpisodeIndexCache(cache=nero_projection.cache)
                nero_projection.register_worker_health_source("preview", nero_preview_cache.status)
                nero_projection.register_worker_health_source("episode_index", nero_episode_index.status)
                nero_preview_cache.start()
                nero_episode_index.start()

                def map_nero_command(mode, action, params):
                    if action == "extension.invoke":
                        return map_xr_teleoperation_extension_action(params)
                    if mode == "collector" and action == "collect.start":
                        if set(params) - {"task_description"}:
                            return None
                        task_description = params.get("task_description", "")
                        if not isinstance(task_description, str):
                            return None
                        return UiCommandName.START_RECORDING, {"task_description": task_description}
                    if mode == "collector" and action == "collect.stop":
                        if params:
                            return None
                        return UiCommandName.STOP_RECORDING, dict(params)
                    if mode == "vla" and action == "runtime.start":
                        if params:
                            return None
                        return UiCommandName.START_ONLINE_INFERENCE, {
                            "prompt": str(args.online_inference_prompt),
                            "protocol_profile": str(args.online_inference_protocol_profile),
                        }
                    if mode == "vla" and action == "runtime.stop":
                        if params:
                            return None
                        return UiCommandName.STOP_ONLINE_INFERENCE, dict(params)
                    if mode != "collector":
                        return None
                    payload = dict(params)
                    if action == "global.config.set":
                        if set(payload) - {"record_root"}:
                            return None
                        root = str(payload.get("record_root", "")).strip()
                        if not root:
                            return None
                        return UiCommandName.SET_RECORD_ROOT, {"root_dir": root}
                    if action == "online_replay.load":
                        episode_name = str(payload.get("episode_name", "")).strip()
                        replay_rate = payload.get("replay_rate")
                        if not episode_name or isinstance(replay_rate, bool):
                            return None
                        return UiCommandName.LOAD_ONLINE_REPLAY, {
                            "episode_name": episode_name,
                            "speed_scale": replay_rate,
                        }
                    if action == "online_replay.start":
                        if not bool(online_replay_selection.get("loaded", False)):
                            return UiCommandName.START_RAW_REPLAY, {"nero_online_start_unloaded": True}
                        return UiCommandName.START_RAW_REPLAY, {
                            "dataset_root": str(online_replay_selection["dataset_root"]),
                            "episode_index": int(online_replay_selection["episode_index"]),
                            "episode_name": str(online_replay_selection["episode_name"]),
                            "arm_source": str(online_replay_selection["arm_source"]),
                            "base_source": str(online_replay_selection["base_source"]),
                            "speed_scale": float(online_replay_selection["replay_rate"]),
                        }
                    if action == "global.episodes.update":
                        if set(payload) != {"episode_name", "task_description"}:
                            return None
                        if not isinstance(payload["episode_name"], str) or not isinstance(payload["task_description"], str):
                            return None
                        return UiCommandName.UPDATE_EPISODE_TASK_DESCRIPTION, payload
                    if action in {"export.output.get", "export.output.episodes.get"}:
                        if set(payload) != {"output_root", "dataset_name"}:
                            return None
                        if not isinstance(payload["output_root"], str) or not isinstance(payload["dataset_name"], str):
                            return None
                        command_name = (
                            UiCommandName.EXPORT_OUTPUT_GET
                            if action == "export.output.get"
                            else UiCommandName.EXPORT_OUTPUT_EPISODES_GET
                        )
                        return command_name, payload
                    collector_actions = {
                        "global.episodes.delete": UiCommandName.DELETE_EPISODES,
                        "offline_replay.load": UiCommandName.LOAD_OFFLINE_REPLAY,
                        "offline_replay.start": UiCommandName.START_OFFLINE_REPLAY,
                        "offline_replay.pause": UiCommandName.PAUSE_OFFLINE_REPLAY,
                        "offline_replay.resume": UiCommandName.RESUME_OFFLINE_REPLAY,
                        "offline_replay.stop": UiCommandName.STOP_OFFLINE_REPLAY,
                        "offline_replay.position.set": UiCommandName.SEEK_OFFLINE_REPLAY,
                        "offline_replay.curves.get": UiCommandName.OFFLINE_REPLAY_CURVES,
                        "offline_replay.image.get": UiCommandName.OFFLINE_REPLAY_IMAGE,
                        "online_replay.load": UiCommandName.LOAD_ONLINE_REPLAY,
                        "online_replay.stop": UiCommandName.STOP_RAW_REPLAY,
                        "export.start": UiCommandName.START_EXPORT,
                        "export.outputs.get": UiCommandName.EXPORT_OUTPUTS,
                    }
                    command_name = collector_actions.get(action)
                    if command_name is not None:
                        return command_name, payload
                    return None

                nero_console_runtime = NeroConsoleRuntime(
                    command_bus=ui_command_bus,
                    config_supplier=nero_projection.config,
                    snapshot_supplier=nero_projection.snapshot,
                    command_mapper=map_nero_command,
                    stream_supplier=lambda mode, stream_id: nero_projection.stream_jpeg(stream_id),
                    stream_demand_sink=nero_preview_cache.set_stream_demand,
                )
                _, initial_ui_payload = ui_state_store.snapshot()
                nero_projection.refresh_control_thread_cache(initial_ui_payload)
                initial_recording = dict(initial_ui_payload.get("recording") or {})
                initial_root = str(initial_recording.get("active_root_dir") or initial_recording.get("root_dir") or "").strip()
                nero_episode_tracking["active"] = bool(initial_recording.get("active"))
                nero_episode_tracking["root"] = initial_root
                if initial_root and not nero_episode_tracking["active"]:
                    nero_episode_index.request_refresh(initial_root)
                nero_console_runtime.start()
                logger_mp.info("[NERO_PROVIDER] ROS providers enabled for xr_collector and xr_vla.")

        logger_mp.info("Move arms to home pose before entering teleop wait state...")
        ctrl_g1_29_dual_arm_go_home(args=args, arm_ctrl=arm_ctrl, arm_ik=arm_ik)

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
                for command in keyboard_commands:
                    if command.name in {UiCommandName.START, UiCommandName.STOP}:
                        dispatch_ui_commands([command], on_press)
                        complete_ui_command(
                            ui_command_bus,
                            command,
                            accepted=True,
                            message="keyboard-equivalent command was applied to the teleop state machine",
                        )
                    else:
                        complete_ui_command(
                            ui_command_bus,
                            command,
                            accepted=False,
                            message="teleop has not started; command is unavailable before start",
                        )
                for command in provider_commands:
                    if command.name == UiCommandName.SET_PROVIDER_HOLD:
                        provider_runtime.set_hold(reason="ui_wait_hold")
                        complete_ui_command(ui_command_bus, command, accepted=True, message="provider switched to hold")
                    elif command.name == UiCommandName.SET_PROVIDER_XR:
                        provider_runtime.set_live(reason="ui_wait_xr")
                        complete_ui_command(ui_command_bus, command, accepted=True, message="provider switched to XR live")
                    elif command.name == UiCommandName.START_RAW_REPLAY:
                        provider_runtime.fail_raw_replay("teleop is not started; start teleop before real replay")
                        complete_ui_command(ui_command_bus, command, accepted=False, message="teleop is not started; start teleop before real replay")
                    elif command.name == UiCommandName.STOP_RAW_REPLAY:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="raw replay is not active while teleop is waiting to start")
                    elif command.name == UiCommandName.START_ONLINE_INFERENCE:
                        provider_runtime.fail_online_inference("teleop is not started; start teleop before starting online inference")
                        complete_ui_command(ui_command_bus, command, accepted=False, message="teleop is not started; start teleop before starting online inference")
                    elif command.name == UiCommandName.STOP_ONLINE_INFERENCE:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="online inference is not active while teleop is waiting to start")
                    elif command.name == UiCommandName.SET_RECORD_ROOT:
                        if recording_is_active_or_armed(RECORD_RUNNING, recording_flow):
                            complete_ui_command(ui_command_bus, command, accepted=False, message="recording is active or armed; cannot change recording root")
                        else:
                            recorder, recording_flow = switch_recording_root_from_ui(
                                args=args,
                                components=components,
                                recorder=recorder,
                                recording_flow=recording_flow,
                                record_running=RECORD_RUNNING,
                                payload=command.payload or {},
                                log=logger_mp,
                            )
                            complete_ui_command(ui_command_bus, command, accepted=True, message="recording root was switched")
                    else:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="unsupported command while teleop is waiting")
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
                        latency_snapshot=latency_tracker.get_snapshot() if latency_tracker is not None else None,
                        timing_snapshot=timing_debugger.snapshot(),
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
        latest_teleop_input_perf_counter_ns = int(time.perf_counter_ns())
        last_xr_input_poll_start_ns = None
        xr_input_none_streak_start_ns = None
        xr_input_none_count = 0
        xr_input_poll_stall_threshold_ms = 2.5 * control_dt * 1000.0
        base_z = 0.0
        last_recorded_base_action = {
            "vx_cmd": 0.0,
            "vy_cmd": 0.0,
            "wz_cmd": 0.0,
            "z_cmd": 0.0,
            "frame_id": "base_link",
            "source": "initial_hold",
            "base_control_mode": "initial_hold",
        }

        def publish_recording_sources() -> None:
            if recording_flow is None or not recording_flow.needs_source_updates():
                return
            base_state_history_snapshot = None
            base_height_history_snapshot = None
            slam_tf_history_snapshot = None
            if args.record_base:
                if base_state_receiver is None:
                    raise RuntimeError("--record-base is enabled but base_state_receiver is not initialized")
                if not base_state_receiver.is_alive():
                    raise RuntimeError("--record-base receiver thread is not alive")
                cursors = recording_flow.source_history_cursors()
                snapshot_since = getattr(base_state_receiver, "snapshot_histories_since", None)
                if not callable(snapshot_since):
                    raise RuntimeError(
                        "BaseStateReceiver must provide snapshot_histories_since() for event-driven recording"
                    )
                (
                    base_state_history_snapshot,
                    base_height_history_snapshot,
                    slam_tf_history_snapshot,
                ) = snapshot_since(
                    base_state_after_t_ns=cursors["base_state"],
                    base_height_after_t_ns=cursors["base_height"],
                    slam_tf_after_t_ns=cursors["slam_tf"],
                )
            recording_flow.publish_sources(
                camera_sources=components.cameras.sources(),
                state_history=state_history,
                action_history=action_history,
                teleop_input_perf_counter_ns=latest_teleop_input_perf_counter_ns,
                arm_ik=arm_ik,
                get_wrist_poses=get_robot_wrist_poses,
                get_wrist_poses_base_link=(
                    mobile_base_kinematics.wrist_poses_base_link
                    if args.record_mobile_training_state
                    else None
                ),
                get_dex1_tcp_poses_base_link=(
                    mobile_base_kinematics.dex1_tcp_poses_base_link
                    if args.record_mobile_training_state
                    else None
                ),
                sim_state_subscriber=sim_state_subscriber,
                base_state_history=base_state_history_snapshot,
                base_height_history=base_height_history_snapshot,
                base_action_history=base_action_history if args.record_base else None,
                slam_tf_history=slam_tf_history_snapshot,
            )

        def publish_hold_recording_sources() -> None:
            if recording_flow is None or not recording_flow.needs_source_updates():
                return
            hold_ee_sample = recording_flow._read_end_effector_sample()
            hold_waist_yaw_target = arm_ctrl.waist_yaw_target
            hold_command_monotonic_ns = int(time.monotonic_ns())
            append_timed_sample(
                action_history,
                hold_command_monotonic_ns,
                q=current_hold_q.copy(),
                tauff=current_hold_tauff.copy(),
                waist_yaw_target=(
                    float(current_waist_yaw)
                    if hold_waist_yaw_target is None
                    else float(hold_waist_yaw_target)
                ),
                left_hand_action=list(hold_ee_sample["left_hand_action"]),
                right_hand_action=list(hold_ee_sample["right_hand_action"]),
            )
            if args.record_base:
                hold_base_action = {
                    "vx_cmd": 0.0,
                    "vy_cmd": 0.0,
                    "wz_cmd": 0.0,
                    "z_cmd": float(base_z),
                    "frame_id": "base_link",
                    "source": "hold",
                    "base_control_mode": "hold",
                }
                last_recorded_base_action.clear()
                last_recorded_base_action.update(hold_base_action)
                append_timed_sample(
                    base_action_history,
                    hold_command_monotonic_ns,
                    **hold_base_action,
                )
            publish_recording_sources()

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
            ui_keyboard_commands = []
            if ui_command_bus is not None:
                ui_keyboard_commands, ui_provider_commands = split_ui_commands(ui_command_bus.drain())
                dispatched_keyboard_commands = []
                for command in ui_keyboard_commands:
                    if command.name in RECORDING_UI_COMMANDS:
                        if command.name == UiCommandName.START_RECORDING:
                            if recording_flow is None:
                                complete_ui_command(ui_command_bus, command, accepted=False, message="recording is disabled; restart with --record")
                            elif recording_is_active_or_armed(RECORD_RUNNING, recording_flow):
                                complete_ui_command(ui_command_bus, command, accepted=False, message="recording is already active or armed")
                            else:
                                task_description = (command.payload or {}).get("task_description", "")
                                if not isinstance(task_description, str):
                                    complete_ui_command(ui_command_bus, command, accepted=False, message="task_description must be a string")
                                    continue
                                recorder.set_next_episode_task_description(task_description)
                                RECORD_TOGGLE = True
                        elif not recording_is_active_or_armed(RECORD_RUNNING, recording_flow):
                            complete_ui_command(ui_command_bus, command, accepted=False, message="recording is not active")
                        else:
                            RECORD_TOGGLE = True
                    else:
                        dispatched_keyboard_commands.append(command)
                dispatch_ui_commands(dispatched_keyboard_commands, on_press)
                for command in dispatched_keyboard_commands:
                    if command.name not in {UiCommandName.RECORD_TOGGLE, UiCommandName.RECORD_CANCEL}:
                        complete_ui_command(
                            ui_command_bus,
                            command,
                            accepted=True,
                            message="keyboard-equivalent command was applied to the teleop state machine",
                        )
                if STOP:
                    for command in (*ui_keyboard_commands, *ui_provider_commands):
                        complete_ui_command(ui_command_bus, command, accepted=False, message="teleop is stopping; command was not processed")
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
                if ui_command_bus is not None:
                    for command in ui_keyboard_commands:
                        if command.name in {UiCommandName.RECORD_TOGGLE, UiCommandName.RECORD_CANCEL, *RECORDING_UI_COMMANDS}:
                            complete_ui_command(
                                ui_command_bus,
                                command,
                                accepted=True,
                                message="recording command was processed by the recording state machine",
                            )

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
            if recording_flow is not None and recording_flow.needs_source_updates():
                end_effector_history_sample = recording_flow._read_end_effector_sample()
                append_timed_sample(
                    state_history,
                    current_state_sample_ns,
                    q=current_lr_arm_q.copy(),
                    dq=current_lr_arm_dq.copy(),
                    waist_yaw=float(current_waist_yaw),
                    left_ee_state=list(end_effector_history_sample["left_ee_state"]),
                    right_ee_state=list(end_effector_history_sample["right_ee_state"]),
                )
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
                        latency_snapshot=latency_tracker.get_snapshot() if latency_tracker is not None else None,
                        timing_snapshot=timing_debugger.snapshot(),
                        base_state_receiver=base_state_receiver,
                        base_stop_state=base_stop_state,
                    )
                )
            current_left_wrist_pose, current_right_wrist_pose = get_robot_wrist_poses(arm_ik, current_lr_arm_q)

            camera_sources = components.cameras.sources()

            for command in ui_provider_commands:
                payload = command.payload or {}
                if command.name == UiCommandName.LOAD_OFFLINE_REPLAY:
                    episode_name = str(payload.get("episode_name", "")).strip()
                    if re.fullmatch(r"episode_\d+", episode_name) is None:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="episode_name must use episode_NNNN format")
                    else:
                        result = offline_replay_session.load(
                            str(Path(str(args.task_dir)) / str(args.task_name)), episode_name
                        )
                        complete_ui_command(ui_command_bus, command, accepted=True, message="offline replay episode loaded", details=result)
                elif command.name == UiCommandName.START_OFFLINE_REPLAY:
                    if offline_replay_session.episode_dir is None:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="no offline replay episode is loaded")
                    else:
                        result = offline_replay_session.start()
                        complete_ui_command(ui_command_bus, command, accepted=True, message="offline replay started", details=result)
                elif command.name == UiCommandName.PAUSE_OFFLINE_REPLAY:
                    if offline_replay_session.episode_dir is None:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="no offline replay episode is loaded")
                    else:
                        result = offline_replay_session.pause()
                        complete_ui_command(ui_command_bus, command, accepted=True, message="offline replay paused", details=result)
                elif command.name == UiCommandName.RESUME_OFFLINE_REPLAY:
                    if offline_replay_session.episode_dir is None:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="no offline replay episode is loaded")
                    else:
                        result = offline_replay_session.start()
                        complete_ui_command(ui_command_bus, command, accepted=True, message="offline replay resumed", details=result)
                elif command.name == UiCommandName.STOP_OFFLINE_REPLAY:
                    if offline_replay_session.episode_dir is None:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="no offline replay episode is loaded")
                    else:
                        result = offline_replay_session.stop()
                        complete_ui_command(ui_command_bus, command, accepted=True, message="offline replay stopped", details=result)
                elif command.name == UiCommandName.SEEK_OFFLINE_REPLAY:
                    frame_index = payload.get("frame_index")
                    rate = payload.get("rate")
                    if offline_replay_session.episode_dir is None:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="no offline replay episode is loaded")
                    elif frame_index is None and rate is None:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="offline replay position requires frame_index or rate")
                    elif frame_index is not None and (isinstance(frame_index, bool) or not isinstance(frame_index, int)):
                        complete_ui_command(ui_command_bus, command, accepted=False, message="offline replay frame_index must be an integer")
                    elif rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float)) or float(rate) <= 0.0):
                        complete_ui_command(ui_command_bus, command, accepted=False, message="offline replay rate must be positive")
                    else:
                        result = offline_replay_session.set_position(frame_index=frame_index, rate=rate)
                        complete_ui_command(ui_command_bus, command, accepted=True, message="offline replay position updated", details=result)
                elif command.name == UiCommandName.OFFLINE_REPLAY_CURVES:
                    arm = str(payload.get("arm", "both")).strip()
                    max_points = payload.get("max_points", 900)
                    if offline_replay_session.episode_dir is None:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="no offline replay episode is loaded")
                    elif arm not in {"left", "right", "both"}:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="offline replay arm must be left, right, or both")
                    elif isinstance(max_points, bool) or not isinstance(max_points, int) or max_points <= 0:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="offline replay max_points must be a positive integer")
                    else:
                        result = offline_replay_session.curves(max_points=max_points)
                        if arm != "both":
                            result = {key: value for key, value in result.items() if key not in {"left", "right"} or key == arm}
                        complete_ui_command(ui_command_bus, command, accepted=True, message="offline replay curves ready", details=result)
                elif command.name == UiCommandName.OFFLINE_REPLAY_IMAGE:
                    camera_id = payload.get("camera_id")
                    frame_index = payload.get("frame_index")
                    rel_path = str(payload.get("rel_path", "")).strip()
                    if rel_path:
                        if rel_path.startswith("/") or "\\" in rel_path or offline_replay_session.episode_dir is None:
                            complete_ui_command(ui_command_bus, command, accepted=False, message="offline replay image path is invalid")
                        else:
                            target = (offline_replay_session.episode_dir / rel_path).resolve()
                            if offline_replay_session.episode_dir.resolve() not in target.parents or not target.is_file():
                                complete_ui_command(ui_command_bus, command, accepted=False, message="offline replay image was not found")
                            else:
                                complete_ui_command(
                                    ui_command_bus,
                                    command,
                                    accepted=True,
                                    message="offline replay image ready",
                                    details={"jpeg_base64": base64.b64encode(target.read_bytes()).decode("ascii")},
                                )
                    elif offline_replay_session.episode_dir is None:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="no offline replay episode is loaded")
                    elif isinstance(camera_id, bool) or not isinstance(camera_id, int) or isinstance(frame_index, bool) or not isinstance(frame_index, int):
                        complete_ui_command(ui_command_bus, command, accepted=False, message="offline replay image requires integer camera_id and frame_index")
                    else:
                        target = offline_replay_session.image_path(camera_id, frame_index)
                        complete_ui_command(
                            ui_command_bus,
                            command,
                            accepted=True,
                            message="offline replay image ready",
                            details={"jpeg_base64": base64.b64encode(target.read_bytes()).decode("ascii")},
                        )
                elif command.name == UiCommandName.LOAD_ONLINE_REPLAY:
                    episode_name = str(payload.get("episode_name", "")).strip()
                    speed_scale = payload.get("speed_scale")
                    if re.fullmatch(r"episode_\d+", episode_name) is None:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="episode_name must use episode_NNNN format")
                    elif isinstance(speed_scale, bool) or not isinstance(speed_scale, (int, float)) or float(speed_scale) <= 0.0:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="online replay replay_rate must be positive")
                    else:
                        index = int(episode_name.rsplit("_", 1)[1])
                        root_dir = str(Path(str(args.task_dir)) / str(args.task_name))
                        data_path = Path(root_dir) / episode_name / "data.json"
                        if not data_path.is_file():
                            complete_ui_command(ui_command_bus, command, accepted=False, message="online replay episode data.json was not found")
                        else:
                            online_replay_selection.update({
                                "state": "loaded",
                                "loaded": True,
                                "dataset_root": root_dir,
                                "episode_name": episode_name,
                                "episode_index": index,
                                "replay_rate": float(speed_scale),
                                "error": "",
                            })
                            complete_ui_command(ui_command_bus, command, accepted=True, message="online replay episode loaded", details=dict(online_replay_selection))
                elif command.name == UiCommandName.DELETE_EPISODES:
                    episode_names = payload.get("episode_names")
                    if not isinstance(episode_names, list) or not episode_names:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="episode_names must be a non-empty list")
                    elif any(re.fullmatch(r"episode_\d+", str(name)) is None for name in episode_names):
                        complete_ui_command(ui_command_bus, command, accepted=False, message="episode names must use episode_NNNN format")
                    elif offline_replay_session.state == "playing":
                        complete_ui_command(ui_command_bus, command, accepted=False, message="stop offline replay before deleting episodes")
                    else:
                        root = resolve_root(str(Path(str(args.task_dir)) / str(args.task_name))).resolve()
                        targets = [(root / str(name)).resolve() for name in episode_names]
                        if any(target.parent != root or not (target / "data.json").is_file() for target in targets):
                            complete_ui_command(ui_command_bus, command, accepted=False, message="one or more episode directories do not exist")
                        else:
                            if offline_replay_session.episode_dir is not None and offline_replay_session.episode_dir.resolve() in targets:
                                offline_replay_session.unload()
                            for target in targets:
                                shutil.rmtree(target)
                            if nero_episode_index is not None:
                                nero_episode_index.request_refresh(str(root))
                            complete_ui_command(ui_command_bus, command, accepted=True, message="episodes deleted", details={"deleted": [str(name) for name in episode_names]})
                elif command.name == UiCommandName.UPDATE_EPISODE_TASK_DESCRIPTION:
                    episode_name = str(payload.get("episode_name", "")).strip()
                    task_description = payload.get("task_description")
                    root = resolve_root(str(Path(str(args.task_dir)) / str(args.task_name))).resolve()
                    episode_data_path = root / episode_name / "data.json"
                    if re.fullmatch(r"episode_\d+", episode_name) is None:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="episode_name must use episode_NNNN format")
                    elif not isinstance(task_description, str) or not task_description.strip():
                        complete_ui_command(ui_command_bus, command, accepted=False, message="task_description must be a non-empty string")
                    elif recording_is_active_or_armed(RECORD_RUNNING, recording_flow):
                        complete_ui_command(ui_command_bus, command, accepted=False, message="recording is active or armed; cannot update an episode task description")
                    elif offline_replay_session.state == "playing":
                        complete_ui_command(ui_command_bus, command, accepted=False, message="stop offline replay before updating an episode task description")
                    elif provider_runtime.active_provider_kind == ActiveProviderKind.RAW_REPLAY:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="stop real replay before updating an episode task description")
                    elif episode_data_path.parent.resolve().parent != root or not episode_data_path.is_file():
                        complete_ui_command(ui_command_bus, command, accepted=False, message="episode data.json was not found")
                    else:
                        result = update_episode_task_description(
                            root_dir=root,
                            episode_name=episode_name,
                            task_description=task_description,
                        )
                        if nero_episode_index is not None:
                            nero_episode_index.request_refresh(str(root))
                        complete_ui_command(
                            ui_command_bus,
                            command,
                            accepted=True,
                            message="episode task description updated",
                            details=result,
                        )
                elif command.name == UiCommandName.START_EXPORT:
                    output_root = str(payload.get("output_root", "")).strip()
                    dataset_name = str(payload.get("dataset_name", "")).strip()
                    task = str(payload.get("default_task", payload.get("task", ""))).strip()
                    fps = payload.get("fps", args.frequency)
                    selected_episodes = payload.get("episode_names", payload.get("episodes", []))
                    if not output_root or not dataset_name or not task:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="export requires output_root, dataset_name, and default_task")
                    elif isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) <= 0.0:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="export fps must be positive")
                    elif not isinstance(selected_episodes, list) or any(re.fullmatch(r"episode_\d+", str(name)) is None for name in selected_episodes):
                        complete_ui_command(ui_command_bus, command, accepted=False, message="export episode_names must be a list of episode_NNNN names")
                    elif str(payload.get("training_profile", "auto")) != "auto" or bool(payload.get("export_collection_mode", False)):
                        complete_ui_command(ui_command_bus, command, accepted=False, message="XR LeRobot exporter does not support training_profile or export_collection_mode")
                    else:
                        result = export_manager.start(UiExportRequest(
                            source_root=Path(str(Path(str(args.task_dir)) / str(args.task_name))),
                            output_root=Path(output_root),
                            dataset_name=dataset_name,
                            task=task,
                            fps=float(fps),
                            format_version=str(payload.get("format_version", "v2")),
                            export_mode=str(payload.get("export_mode", "new")),
                            export_video=bool(payload.get("export_video", False)),
                            export_fk=bool(payload.get("export_fk", False)),
                            export_verify=bool(payload.get("export_verify", True)),
                            selected_episodes=tuple(str(name) for name in selected_episodes),
                            urdf_path=str(payload.get("urdf_path", DEFAULT_UI_URDF_PATH)),
                        ))
                        complete_ui_command(ui_command_bus, command, accepted=bool(result.get("ok")), message=str(result.get("message", "export request processed")), details=result)
                elif command.name == UiCommandName.EXPORT_OUTPUTS:
                    output_root = str(payload.get("output_root", "")).strip()
                    if not output_root:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="export output_root is required")
                    else:
                        root = Path(output_root).expanduser()
                        datasets = [item.name for item in sorted(root.iterdir()) if item.is_dir()] if root.is_dir() else []
                        complete_ui_command(ui_command_bus, command, accepted=True, message="export outputs ready", details={"output_root": str(root), "datasets": datasets})
                elif command.name in {UiCommandName.EXPORT_OUTPUT_GET, UiCommandName.EXPORT_OUTPUT_EPISODES_GET}:
                    output_root = str(payload.get("output_root", "")).strip()
                    dataset_name = str(payload.get("dataset_name", "")).strip()
                    root = Path(output_root).expanduser()
                    dataset = root / dataset_name
                    if not output_root:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="export output_root is required")
                    elif not dataset_name or dataset_name in {".", ".."} or "/" in dataset_name or "\\" in dataset_name:
                        complete_ui_command(ui_command_bus, command, accepted=False, message="dataset_name must be one plain directory name")
                    elif not root.is_dir():
                        complete_ui_command(ui_command_bus, command, accepted=False, message="export output_root is not a directory")
                    elif not dataset.is_dir():
                        complete_ui_command(ui_command_bus, command, accepted=False, message="export dataset was not found")
                    elif (
                        not (dataset / "export_summary.json").is_file()
                        or not (dataset / "meta" / "info.json").is_file()
                        or not (dataset / "meta" / "episodes.jsonl").is_file()
                    ):
                        complete_ui_command(ui_command_bus, command, accepted=False, message="export dataset is incomplete: required LeRobot metadata is missing")
                    elif command.name == UiCommandName.EXPORT_OUTPUT_GET:
                        result = export_output_details(output_root=output_root, dataset_name=dataset_name)
                        complete_ui_command(ui_command_bus, command, accepted=True, message="export output details ready", details=result)
                    else:
                        result = export_output_episodes(output_root=output_root, dataset_name=dataset_name)
                        complete_ui_command(ui_command_bus, command, accepted=True, message="export output episodes ready", details=result)
                elif command.name == UiCommandName.SET_PROVIDER_HOLD:
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
                    if bool(payload.get("nero_online_start_unloaded", False)):
                        complete_ui_command(ui_command_bus, command, accepted=False, message="load an online replay episode before starting")
                        continue
                    if recording_is_active_or_armed(RECORD_RUNNING, recording_flow):
                        stop_base_once("raw_replay_rejected_recording")
                        provider_runtime.fail_raw_replay("recording is active or armed; stop/cancel recording before real replay")
                        logger_mp.error("[UI_REPLAY] rejected: recording is active or armed.")
                        complete_ui_command(ui_command_bus, command, accepted=False, message="recording is active or armed; stop/cancel recording before real replay")
                    else:
                        replay_base_source = str(payload.get("base_source", "none") or "none")
                        if replay_base_source not in {"none", "action"}:
                            stop_base_once("raw_replay_rejected_base_source")
                            provider_runtime.fail_raw_replay(
                                f"unsupported UI raw replay base_source: {replay_base_source}"
                            )
                            logger_mp.error("[UI_REPLAY] rejected: unsupported base_source=%s", replay_base_source)
                            complete_ui_command(ui_command_bus, command, accepted=False, message=f"unsupported raw replay base_source: {replay_base_source}")
                            continue
                        if replay_base_source == "action" and not args.base_motion:
                            stop_base_once("raw_replay_rejected_base_motion")
                            provider_runtime.fail_raw_replay(
                                "base_source=action requires restarting with --base-motion"
                            )
                            logger_mp.error("[UI_REPLAY] rejected: base_source=action requires --base-motion.")
                            complete_ui_command(ui_command_bus, command, accepted=False, message="base_source=action requires --base-motion")
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
                            complete_ui_command(ui_command_bus, command, accepted=False, message="base_source=action requires --base-controller g1d_agv")
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
                            complete_ui_command(ui_command_bus, command, accepted=False, message="mobile_ik_qp raw replay requires arm_source=action or state")
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
                        if command.source.startswith("nero:"):
                            online_replay_selection["state"] = "running"
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
                    if command.source.startswith("nero:"):
                        online_replay_selection["state"] = "stopped"
                    logger_mp.info("[UI_REPLAY] stopped raw replay -> HOLD.")
                elif command.name == UiCommandName.START_ONLINE_INFERENCE:
                    if recording_is_active_or_armed(RECORD_RUNNING, recording_flow):
                        stop_base_once("online_inference_rejected_recording")
                        provider_runtime.fail_online_inference(
                            "recording is active or armed; stop/cancel recording before starting online inference"
                        )
                        logger_mp.error("[UI_INFERENCE] rejected: recording is active or armed.")
                        complete_ui_command(ui_command_bus, command, accepted=False, message="recording is active or armed; stop/cancel recording before starting online inference")
                    elif provider_runtime.active_provider_kind == ActiveProviderKind.RAW_REPLAY:
                        stop_base_once("online_inference_rejected_raw_replay")
                        provider_runtime.fail_online_inference("raw replay is active; stop replay before starting online inference")
                        logger_mp.error("[UI_INFERENCE] rejected: raw replay is active.")
                        complete_ui_command(ui_command_bus, command, accepted=False, message="raw replay is active; stop replay before starting online inference")
                    else:
                        prompt = str(payload.get("prompt", "")).strip()
                        protocol_profile = str(payload.get("protocol_profile", "")).strip()
                        profile_error = ui_online_inference.profile_error(protocol_profile)
                        missing_cameras = missing_online_inference_camera_names(components.cameras)
                        if not prompt:
                            stop_base_once("online_inference_rejected_missing_prompt")
                            provider_runtime.fail_online_inference("online inference prompt is required")
                            logger_mp.error("[UI_INFERENCE] rejected: prompt is missing.")
                            complete_ui_command(ui_command_bus, command, accepted=False, message="online inference prompt is required")
                        elif not bool(args.online_inference_enable_motion):
                            stop_base_once("online_inference_rejected_motion_authorization")
                            provider_runtime.fail_online_inference(
                                "online inference requires --online-inference-enable-motion authorization"
                            )
                            logger_mp.error("[UI_INFERENCE] rejected: --online-inference-enable-motion is required.")
                            complete_ui_command(
                                ui_command_bus,
                                command,
                                accepted=False,
                                message="online inference requires --online-inference-enable-motion authorization",
                            )
                        elif profile_error:
                            stop_base_once("online_inference_rejected_protocol")
                            provider_runtime.fail_online_inference(profile_error)
                            logger_mp.error("[UI_INFERENCE] rejected: %s", profile_error)
                            complete_ui_command(ui_command_bus, command, accepted=False, message=profile_error)
                        elif missing_cameras:
                            stop_base_once("online_inference_rejected_missing_cameras")
                            provider_runtime.fail_online_inference(
                                "missing required online inference cameras: " + ", ".join(missing_cameras)
                            )
                            logger_mp.error("[UI_INFERENCE] rejected: missing cameras=%s", missing_cameras)
                            complete_ui_command(ui_command_bus, command, accepted=False, message="missing required online inference cameras: " + ", ".join(missing_cameras))
                        else:
                            current_hold_q = current_lr_arm_q.copy()
                            current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
                            home_return_active = False
                            post_home_takeover_armed = False
                            stop_base_once("online_inference_start")
                            if provider_runtime.active_provider_kind != ActiveProviderKind.HOLD:
                                provider_runtime.set_hold(reason="online_inference_start_sync_hold")
                            provider_runtime.start_online_inference(
                                prompt=prompt,
                                protocol_profile=protocol_profile,
                            )
                            set_online_inference_gripper_mode(gripper_ctrl, True)
                            START = True
                            operator_state_flow = OperatorStateFlow(takeover_settle_frames=0, log=logger_mp)
                            logger_mp.info("[UI_INFERENCE] started HTTP pi0.5 online inference: protocol=%s", protocol_profile)
                elif command.name == UiCommandName.STOP_ONLINE_INFERENCE:
                    current_hold_q = current_lr_arm_q.copy()
                    current_hold_tauff = compute_arm_gravity_tauff(arm_ik, current_hold_q)
                    home_return_active = False
                    post_home_takeover_armed = False
                    stop_base_once("online_inference_stop")
                    provider_runtime.stop_online_inference(reason=str(payload.get("reason", "ui_stop")))
                    set_online_inference_gripper_mode(gripper_ctrl, False)
                    logger_mp.info("[UI_INFERENCE] stopped -> HOLD.")

                complete_ui_command(
                    ui_command_bus,
                    command,
                    accepted=True,
                    message="command was applied by the XR control loop",
                )

            if provider_runtime.active_provider_kind == ActiveProviderKind.HOLD:
                stop_base_once("provider_hold")
                arm_ctrl.ctrl_dual_arm(current_hold_q.copy(), current_hold_tauff.copy())
                publish_hold_recording_sources()
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
            input_poll_start_ns = int(time.perf_counter_ns())
            input_poll_gap_ms = None
            if active_input_provider == "xr" and last_xr_input_poll_start_ns is not None:
                input_poll_gap_ms = (input_poll_start_ns - last_xr_input_poll_start_ns) / 1e6
            if active_input_provider == "xr":
                last_xr_input_poll_start_ns = input_poll_start_ns
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
            if active_input_provider == "online_inference":
                active_session = getattr(active_provider, "session", None)
                active_profile = str(getattr(getattr(active_session, "config", None), "protocol_profile", ""))
                if active_profile == "mobile_tcp23":
                    provider_get_sample_kwargs.update(
                        mobile_online_inputs.current(current_lr_arm_q, include_tcp=True)
                    )
                elif active_profile == "mobile_pelvis_planar22":
                    provider_get_sample_kwargs.update(mobile_online_inputs.pelvis_planar22_current())
                elif active_profile == "mobile_joint_base":
                    provider_get_sample_kwargs.update(
                        mobile_online_inputs.current(current_lr_arm_q, include_tcp=False)
                    )
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
                publish_hold_recording_sources()
                logger_mp.info("[UI_INFERENCE] discarded fetched action because stop is pending -> HOLD.")
                continue
            if sample is None:
                if active_input_provider == "xr":
                    if xr_input_none_streak_start_ns is None:
                        xr_input_none_streak_start_ns = input_poll_start_ns
                    xr_input_none_count += 1
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
                    publish_hold_recording_sources()
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
                    publish_hold_recording_sources()
                    continue
                timing_debugger.maybe_report(arm_ctrl=arm_ctrl, gripper_ctrl=gripper_ctrl)
                publish_hold_recording_sources()
                time.sleep(0.01)
                continue
            xr_input_poll_diagnostics = {}
            if active_input_provider == "xr":
                input_none_streak_ms = (
                    0.0
                    if xr_input_none_streak_start_ns is None
                    else (input_poll_start_ns - xr_input_none_streak_start_ns) / 1e6
                )
                poll_gap_ms = 0.0 if input_poll_gap_ms is None else float(input_poll_gap_ms)
                xr_input_poll_diagnostics = {
                    "input_poll_gap_ms": poll_gap_ms,
                    "input_none_count": int(xr_input_none_count),
                    "input_none_streak_ms": float(input_none_streak_ms),
                    "input_poll_gap_threshold_ms": float(xr_input_poll_stall_threshold_ms),
                }
                if (
                    xr_input_none_count > 0
                    or poll_gap_ms > xr_input_poll_stall_threshold_ms
                ):
                    if latency_tracker is not None:
                        latency_tracker.record_input_poll_stall(
                            poll_start_ns=input_poll_start_ns,
                            poll_gap_ms=poll_gap_ms,
                            get_sample_ms=tele_fetch_dt * 1000.0,
                            none_count=xr_input_none_count,
                            none_streak_ms=input_none_streak_ms,
                            threshold_ms=xr_input_poll_stall_threshold_ms,
                        )
                xr_input_none_streak_start_ns = None
                xr_input_none_count = 0
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
                publish_hold_recording_sources()
                continue
            tele_data_recv_ts_ns = time.perf_counter_ns()
            latest_teleop_input_perf_counter_ns = int(tele_data_recv_ts_ns)
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
            active_protocol_profile = str(
                getattr(getattr(getattr(active_provider, "session", None), "config", None), "protocol_profile", "")
            )
            if (
                args.mobile_manipulation_mode == "mobile_ik_qp"
                or (active_input_provider == "online_inference" and active_protocol_profile == "mobile_joint_base")
            ):
                manual_torso_yaw_rate = 0.0
            else:
                manual_torso_yaw_rate = map_manual_torso_yaw_rate(
                    args=args,
                    tele_data=tele_data,
                    home_return_active=home_return_active,
                )
            wbc_ms = None
            if args.mobile_manipulation_mode == "mobile_ik_qp" and not is_ui_raw_replay:
                if mobile_coordinator is None:
                    raise RuntimeError("mobile_ik_qp coordinator is not initialized")
                if mobile_state_provider is None:
                    raise RuntimeError("mobile_ik_qp state provider is not initialized")
                nominal_source = runtime_base_source or "controller"
                nominal_intent = map_base_command(
                    args=args,
                    tele_data=tele_data,
                    home_return_active=home_return_active,
                    base_intent=getattr(sample, "base_intent", None),
                    base_command_source=nominal_source,
                )
                wbc_start = time.perf_counter()
                coordinated = mobile_coordinator.step(
                    motion_intent=motion_intent,
                    enabled={"left": left_arm_enabled, "right": right_arm_enabled},
                    rising={"left": left_takeover_rising_edge, "right": right_takeover_rising_edge},
                    settling={
                        "left": left_zero_takeover_this_frame,
                        "right": right_zero_takeover_this_frame,
                    },
                    nominal_body_command=np.array([
                        nominal_intent.vx,
                        nominal_intent.wz,
                        nominal_intent.z,
                    ]),
                    mobile_state=mobile_state_provider.current(),
                    arm_q=current_lr_arm_q,
                    now_monotonic_ns=time.monotonic_ns(),
                    dt=control_dt,
                    home_active=home_return_active,
                    stop_active=STOP,
                )
                wbc_ms = (time.perf_counter() - wbc_start) * 1000.0
                timing_debugger.add_wbc(wbc_ms / 1000.0)
                motion_intent = MotionIntent(
                    kind="joint_position",
                    arm_q=coordinated.arm_q_target,
                    gripper_q=motion_intent.gripper_q,
                    timestamp=motion_intent.timestamp,
                    frame_index=motion_intent.frame_index,
                    source="mobile_ik_qp:wbc",
                    metadata=motion_intent.metadata,
                )
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
                base_intent = getattr(sample, "base_intent", None)
                base_provider_active = active_input_provider in {"lerobot_offline", "online_inference"} or is_ui_raw_replay
            if not is_ui_raw_replay:
                waist_yaw_target, waist_yaw_saturated = resolve_waist_yaw_position_target(
                    mobile_manipulation_mode=str(args.mobile_manipulation_mode),
                    current_target_rad=arm_ctrl.waist_yaw_target,
                    manual_yaw_rate_radps=manual_torso_yaw_rate,
                    dt=control_dt,
                )
                arm_ctrl.set_waist_yaw_target(waist_yaw_target)
                if args.mobile_manipulation_mode != "mobile_ik_qp" and waist_yaw_saturated and not manual_waist_yaw_limit_state["active"]:
                    logger_mp.warning(
                        "[WAIST_YAW] manual target saturated at %.4f rad (limit=[-2.7053, 2.7053])",
                        waist_yaw_target,
                    )
                manual_waist_yaw_limit_state["active"] = waist_yaw_saturated
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
            if (
                args.record_base
                and recording_flow is not None
                and recording_flow.needs_source_updates()
            ):
                base_action_source = "g1d_agv_bridge" if base_control_mode == "g1d_agv_async" else base_control_mode
                last_recorded_base_action = {
                    "vx_cmd": float(base_vx),
                    "vy_cmd": float(base_vy),
                    "wz_cmd": float(base_wz),
                    "z_cmd": float(base_z),
                    "frame_id": "base_link",
                    "source": base_action_source,
                    "base_control_mode": base_control_mode,
                }
                append_timed_sample(
                    base_action_history,
                    int(time.monotonic_ns()),
                    **last_recorded_base_action,
                )
            if base_result.stop_requested:
                stop_base_once("controller_stop")
                START = False
                STOP = True
            if base_result.base_bridge_recovered:
                recovery_reason = str(base_result.base_stop_error or "G1D AGV bridge fault")
                logger_mp.error(
                    "[BASE_CTRL] bridge recovered with STOP confirmed; entering HOLD and canceling the active episode: %s",
                    recovery_reason,
                )
                if recording_flow is not None:
                    recording_flow.cancel_for_safety(f"g1d_agv_bridge_recovered: {recovery_reason}")
                RECORD_RUNNING = False
                RECORD_TOGGLE = False
                RECORD_CANCEL = False
                operator_state_flow.require_safety_rearm("g1d_agv_bridge_recovered")
                if active_input_provider != "xr":
                    provider_runtime.set_hold(reason="g1d_agv_bridge_recovered")
                START = False
            if base_result.base_stop_fault:
                logger_mp.error("[BASE_CTRL] base stop fault; retrying STOP: %s", base_result.base_stop_error)
                publish_hold_recording_sources()
                time.sleep(control_dt)
                continue
            if base_result.should_continue_frame:
                publish_hold_recording_sources()
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
                publish_hold_recording_sources()
                continue

            sol_q = arm_command.sol_q
            sol_tauff = arm_command.sol_tauff
            current_hold_q = arm_command.current_hold_q
            current_hold_tauff = arm_command.current_hold_tauff
            provider_feedback = arm_command.provider_feedback
            ik_ms = arm_command.ik_ms
            ik_ipopt_solve_ms = arm_command.ik_ipopt_solve_ms
            ik_ipopt_iterations = arm_command.ik_ipopt_iterations
            ik_filter_ms = arm_command.ik_filter_ms
            ik_rnea_ms = arm_command.ik_rnea_ms
            ik_total_ms = arm_command.ik_total_ms
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
                            **xr_input_poll_diagnostics,
                            "input_provider": str(active_input_provider),
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
                            "ik_ipopt_solve_ms": ik_ipopt_solve_ms,
                            "ik_ipopt_iterations": ik_ipopt_iterations,
                            "ik_filter_ms": ik_filter_ms,
                            "ik_rnea_ms": ik_rnea_ms,
                            "ik_total_ms": ik_total_ms,
                            "wbc_ms": wbc_ms,
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
            if recording_flow is not None and recording_flow.needs_source_updates():
                action_command_monotonic_ns = int(time.monotonic_ns())
                commanded_end_effector_sample = recording_flow._read_end_effector_sample()
                waist_yaw_target = arm_ctrl.waist_yaw_target
                if args.record_mobile_training_state and waist_yaw_target is None:
                    raise RuntimeError("mobile training state requires an initialized waist yaw target")
                append_timed_sample(
                    action_history,
                    action_command_monotonic_ns,
                    q=sol_q.copy(),
                    tauff=sol_tauff.copy(),
                    waist_yaw_target=(None if waist_yaw_target is None else float(waist_yaw_target)),
                    left_hand_action=list(commanded_end_effector_sample["left_hand_action"]),
                    right_hand_action=list(commanded_end_effector_sample["right_hand_action"]),
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
                if motion_intent.kind == "joint_position":
                    trajectory_debug = {
                        "sample_monotonic_ns": int(time.monotonic_ns()),
                        "representation": "joint_position",
                        "target_q": np.asarray(motion_intent.arm_q, dtype=float).tolist(),
                        "feedback_q": current_lr_arm_q.tolist(),
                    }
                else:
                    trajectory_debug = {
                        "sample_monotonic_ns": int(time.monotonic_ns()),
                        "representation": "wrist_pose",
                        "left_target": wrist_pose_to_debug_sample(motion_intent.left_wrist_pose),
                        "right_target": wrist_pose_to_debug_sample(motion_intent.right_wrist_pose),
                        "left_feedback": wrist_pose_to_debug_sample(current_left_wrist_pose),
                        "right_feedback": wrist_pose_to_debug_sample(current_right_wrist_pose),
                        "left_target_xyz": np.asarray(motion_intent.left_wrist_pose, dtype=float)[:3, 3].tolist(),
                        "right_target_xyz": np.asarray(motion_intent.right_wrist_pose, dtype=float)[:3, 3].tolist(),
                        "left_feedback_xyz": np.asarray(current_left_wrist_pose, dtype=float)[:3, 3].tolist(),
                        "right_feedback_xyz": np.asarray(current_right_wrist_pose, dtype=float)[:3, 3].tolist(),
                    }
                provider_runtime.note_online_inference_runtime_debug(
                    {
                        "updated_monotonic_ns": int(time.monotonic_ns()),
                        "latency": {
                            "http_roundtrip_ms": metadata.get("online_obs_send_to_action_recv_ms"),
                            "tele_fetch_ms": float(tele_fetch_ms),
                            "ik_ms": float(ik_ms),
                            "ik_ipopt_solve_ms": ik_ipopt_solve_ms,
                            "ik_ipopt_iterations": ik_ipopt_iterations,
                            "ik_filter_ms": ik_filter_ms,
                            "ik_rnea_ms": ik_rnea_ms,
                            "ik_total_ms": ik_total_ms,
                            "wbc_ms": wbc_ms,
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
                        "trajectory": trajectory_debug,
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
            if home_return_active and np.all(
                np.abs(sol_q - home_target_q) < float(args.home_position_tolerance_rad)
            ):
                home_return_active = False
                calibration_hold_q = current_hold_q.copy()
                calibration_hold_tauff = current_hold_tauff.copy()
                logger_mp.info("[HOME] reached ready/calibration pose. Waiting for both grips to release before teleop resumes.")

            # Publish the newly computed command as the future support for camera
            # frames received during this control tick.
            publish_recording_sources()
            if recording_flow is not None:
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
                            latency_snapshot=latency_tracker.get_snapshot() if latency_tracker is not None else None,
                            timing_snapshot=timing_debugger.snapshot(),
                            base_state_receiver=base_state_receiver,
                            base_stop_state=base_stop_state,
                        )
                    )
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
            arm_ik=arm_ik or components.arm_ik,
            recording_flow=recording_flow,
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
            nero_console_runtime=nero_console_runtime,
            nero_preview_cache=nero_preview_cache,
            nero_episode_index=nero_episode_index,
            exit_go_home=exit_go_home,
            exit_home_hold_sec=exit_home_hold_sec,
        )
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
