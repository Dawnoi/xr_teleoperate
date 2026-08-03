"""真机遥操启动装配：集中创建硬件控制器、相机、录制和调试组件。

Setup helpers for the real teleop entrypoint.

Keep hardware/process construction out of ``teleop_hand_and_arm.py`` while leaving
that file as the single runtime entrypoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from multiprocessing import Array, Lock, Value
from pathlib import Path

from core.camera.local_camera import LocalCameraStream
from core.control.g1d_agv_bridge import G1DAgvBridge
from core.control.motion_switcher import LocoClientWrapper, MotionSwitcher
from core.input.teleop_input_provider import create_teleop_input_provider
from data_pipeline.audit.episode_validation_manager import EpisodeValidationManager
from data_pipeline.recording.base_state_receiver import BaseStateReceiver
from data_pipeline.recording.episode_writer import EpisodeWriter, ZMQRawCameraReceiver
from data_pipeline.recording.teleop_recording_flow import TeleopRecordingFlow
from teleop.control_flow.dex1_tcp_fk import Dex1TcpFkProvider
from teleop.debug.latency_setup import setup_latency_tracker
from teleop.robot_control.robot_arm import (
    G1_23_ArmController,
    G1_29_ArmController,
    H1_2_ArmController,
    H1_ArmController,
    H2_ArmController,
)
from teleop.robot_control.robot_arm_ik import G1_23_ArmIK, G1_29_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


@dataclass
class EndEffectorRuntime:
    hand_ctrl: object | None = None
    gripper_ctrl: object | None = None
    left_hand_pos_array: object | None = None
    right_hand_pos_array: object | None = None
    dual_hand_data_lock: object | None = None
    dual_hand_state_array: object | None = None
    dual_hand_action_array: object | None = None
    left_gripper_value: object | None = None
    right_gripper_value: object | None = None
    dual_gripper_data_lock: object | None = None
    dual_gripper_state_array: object | None = None
    dual_gripper_action_array: object | None = None


@dataclass
class CameraRuntime:
    head_remote: object | None = None
    left_remote: object | None = None
    right_remote: object | None = None
    head_local: object | None = None
    left_local: object | None = None
    right_local: object | None = None

    def sources(self) -> dict[str, object | None]:
        return {
            "head": self.head_remote if self.head_remote is not None else self.head_local,
            "left_wrist": self.left_remote if self.left_remote is not None else self.left_local,
            "right_wrist": self.right_remote if self.right_remote is not None else self.right_local,
        }

    def close_list(self) -> list[object | None]:
        return [
            self.head_remote,
            self.left_remote,
            self.right_remote,
            self.head_local,
            self.left_local,
            self.right_local,
        ]


@dataclass
class RealTeleopComponents:
    arm_ik: object | None = None
    arm_ctrl: object | None = None
    tv_wrapper: object | None = None
    ee: EndEffectorRuntime = field(default_factory=EndEffectorRuntime)
    loco_wrapper: object | None = None
    agv_bridge: object | None = None
    reset_pose_publisher: object | None = None
    sim_state_subscriber: object | None = None
    base_state_receiver: object | None = None
    recorder: object | None = None
    recording_flow: object | None = None
    validation_manager: object | None = None
    cameras: CameraRuntime = field(default_factory=CameraRuntime)
    latency_tracker: object | None = None
    dex1_tcp_fk: object | None = None


MOBILE_HAND_EE = {"dex3", "inspire_dfx", "inspire_ftp", "brainco"}


def publish_reset_category(category: int, publisher, log):
    msg = String_(data=str(category))
    publisher.Write(msg)
    log.info(f"published reset category: {category}")


def initialize_dds(args):
    if args.sim:
        ChannelFactoryInitialize(1, networkInterface=args.network_interface)
    else:
        ChannelFactoryInitialize(0, networkInterface=args.network_interface)


def log_workspace_config(args, *, workspace_limit_enabled, workspace_mode, workspace_min, workspace_max, tapered_workspace_params, side_workspaces, log):
    if workspace_limit_enabled:
        if args.arm_workspace_layout == "per_arm":
            for side in ("left", "right"):
                workspace = side_workspaces[side]
                if workspace_mode == "box":
                    log.info("[ARM_WORKSPACE][%s] box min=%s max=%s", side, workspace["workspace_min"], workspace["workspace_max"])
                else:
                    tapered = workspace["tapered"]
                    log.info(
                        "[ARM_WORKSPACE][%s] tapered z=[%.3f, %.3f] x=[%.3f, %.3f->%.3f] y(low)=[%.3f, %.3f] y(high)=[%.3f, %.3f]",
                        side, tapered["z_min"], tapered["z_max"], tapered["x_min"], tapered["x_max_low"], tapered["x_max_high"],
                        tapered["y_min_low"], tapered["y_max_low"], tapered["y_min_high"], tapered["y_max_high"],
                    )
        elif workspace_mode == "box":
            log.info(
                "[ARM_WORKSPACE] enabled: forward box, min=(%.3f, %.3f, %.3f), max=(%.3f, %.3f, %.3f)",
                workspace_min[0], workspace_min[1], workspace_min[2],
                workspace_max[0], workspace_max[1], workspace_max[2],
            )
        else:
            log.info(
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
        log.info("[ARM_WORKSPACE] +z is arm-up in the IK/base frame.")
    else:
        log.info("[ARM_WORKSPACE] disabled.")
    if args.timing_debug:
        log.info(f"[TIMING] debug enabled, report interval = {args.timing_debug_interval:.1f}s")


def setup_real_teleop_components(args, *, log, components: RealTeleopComponents | None = None) -> RealTeleopComponents:
    components = components if components is not None else RealTeleopComponents()
    setup_base(args, components, log)
    setup_arm_and_input(args, components)
    components.ee = setup_end_effector(args, log)
    apply_affinity_if_requested(args, log)
    setup_sim(args, components)
    components.validation_manager = EpisodeValidationManager() if args.record else None
    components.recorder = setup_recorder(args, components.validation_manager)
    setup_dex1_tcp_fk(args, components, log)
    components.cameras = setup_cameras(args, log)
    components.base_state_receiver = setup_base_state_receiver(args, log)
    components.recording_flow = setup_recording_flow(args, components, log)
    components.latency_tracker = setup_latency_tracker(args, components.arm_ctrl, log)
    return components


def setup_base(args, components: RealTeleopComponents, log):
    base_command_source = str(getattr(args, "base_command_source", "controller") or "controller")
    if not args.motion:
        motion_switcher = MotionSwitcher()
        status, _ = motion_switcher.Enter_Debug_Mode()
        log.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

    if args.base_controller == "none":
        log.info("[BASE_CTRL] disabled: --base-controller none.")
        return
    if not args.base_motion:
        log.info("[BASE_CTRL] output disabled: add --base-motion to enable %s.", args.base_controller)
        return
    if args.base_controller == "loco":
        components.loco_wrapper = LocoClientWrapper()
        log.info(
            "[BASE_CTRL] enabled: Unitree loco backend, source=%s "
            "(left stick Y -> x(vx), left stick X -> y(vy), right stick X -> yaw(wz), "
            "max_vx=%.2f, max_vy=%.2f, max_wz=%.2f)",
            base_command_source,
            args.base_max_vx,
            args.base_max_vy,
            args.base_max_wz,
        )
        return
    if args.base_controller == "g1d_agv":
        components.agv_bridge = G1DAgvBridge(network_interface=args.network_interface, auto_build=True)
        log.info(
            "[BASE_CTRL] enabled: official G1D AgvClient bridge (async target queue), source=%s "
            "(controller mapping: left stick Y -> x(vx), left stick X -> yaw(wz), right stick Y -> z, "
            "max_vx=%.2f, max_wz=%.2f, max_z=%.2f)",
            base_command_source,
            args.base_max_vx,
            args.base_max_wz,
            args.base_max_z,
        )
        log.warning(
            "[BASE_CTRL] G1D AgvClient limitation: official API currently ignores vy, "
            "so this path maps left stick X to in-place yaw instead of lateral strafing."
        )
        return
    raise ValueError(f"unsupported base_controller: {args.base_controller}")


def setup_arm_and_input(args, components: RealTeleopComponents):
    if args.arm == "G1_29":
        components.arm_ik = G1_29_ArmIK()
        components.arm_ctrl = G1_29_ArmController(
            motion_mode=args.motion,
            simulation_mode=args.sim,
            control_hz=args.arm_control_hz,
        )
    elif args.arm == "G1_23":
        components.arm_ik = G1_23_ArmIK()
        components.arm_ctrl = G1_23_ArmController(
            motion_mode=args.motion,
            simulation_mode=args.sim,
            control_hz=args.arm_control_hz,
        )
    elif args.arm == "H1_2":
        components.arm_ik = H1_2_ArmIK()
        components.arm_ctrl = H1_2_ArmController(
            motion_mode=args.motion,
            simulation_mode=args.sim,
            control_hz=args.arm_control_hz,
        )
    elif args.arm == "H1":
        components.arm_ik = H1_ArmIK()
        components.arm_ctrl = H1_ArmController(
            simulation_mode=args.sim,
            control_hz=args.arm_control_hz,
        )
    elif args.arm == "H2":
        components.arm_ik = H2_ArmIK()
        components.arm_ctrl = H2_ArmController(
            motion_mode=args.motion,
            simulation_mode=args.sim,
            control_hz=args.arm_control_hz,
        )
    else:
        raise ValueError(f"Unsupported arm: {args.arm}")
    components.tv_wrapper = create_teleop_input_provider(args, arm_ik=components.arm_ik)


def setup_end_effector(args, log) -> EndEffectorRuntime:
    ee = EndEffectorRuntime()
    if args.no_gripper:
        log.info("[EE] --no-gripper enabled; end-effector controller is disabled.")
        return ee

    if args.ee == "dex3":
        from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
        ee.left_hand_pos_array = Array('d', 75, lock=True)
        ee.right_hand_pos_array = Array('d', 75, lock=True)
        ee.dual_hand_data_lock = Lock()
        ee.dual_hand_state_array = Array('d', 14, lock=False)
        ee.dual_hand_action_array = Array('d', 14, lock=False)
        ee.hand_ctrl = Dex3_1_Controller(
            ee.left_hand_pos_array,
            ee.right_hand_pos_array,
            ee.dual_hand_data_lock,
            ee.dual_hand_state_array,
            ee.dual_hand_action_array,
            simulation_mode=args.sim,
        )
    elif args.ee == "dex1":
        from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
        ee.left_gripper_value = Value('d', 0.0, lock=True)
        ee.right_gripper_value = Value('d', 0.0, lock=True)
        ee.dual_gripper_data_lock = Lock()
        ee.dual_gripper_state_array = Array('d', 2, lock=False)
        ee.dual_gripper_action_array = Array('d', 2, lock=False)
        ee.gripper_ctrl = Dex1_1_Gripper_Controller(
            ee.left_gripper_value,
            ee.right_gripper_value,
            ee.dual_gripper_data_lock,
            ee.dual_gripper_state_array,
            ee.dual_gripper_action_array,
            simulation_mode=args.sim,
            force_hold_extra_close_enabled=(args.input_provider == "online_inference"),
        )
    elif args.ee in {"inspire_dfx", "inspire_ftp"}:
        controller_cls = _inspire_controller_class(args.ee)
        ee.left_hand_pos_array = Array('d', 75, lock=True)
        ee.right_hand_pos_array = Array('d', 75, lock=True)
        ee.dual_hand_data_lock = Lock()
        ee.dual_hand_state_array = Array('d', 12, lock=False)
        ee.dual_hand_action_array = Array('d', 12, lock=False)
        ee.hand_ctrl = controller_cls(
            ee.left_hand_pos_array,
            ee.right_hand_pos_array,
            ee.dual_hand_data_lock,
            ee.dual_hand_state_array,
            ee.dual_hand_action_array,
            simulation_mode=args.sim,
        )
    elif args.ee == "brainco":
        from teleop.robot_control.robot_hand_brainco import Brainco_Controller
        ee.left_hand_pos_array = Array('d', 75, lock=True)
        ee.right_hand_pos_array = Array('d', 75, lock=True)
        ee.dual_hand_data_lock = Lock()
        ee.dual_hand_state_array = Array('d', 12, lock=False)
        ee.dual_hand_action_array = Array('d', 12, lock=False)
        ee.hand_ctrl = Brainco_Controller(
            ee.left_hand_pos_array,
            ee.right_hand_pos_array,
            ee.dual_hand_data_lock,
            ee.dual_hand_state_array,
            ee.dual_hand_action_array,
            simulation_mode=args.sim,
        )
    return ee


def _inspire_controller_class(ee_name: str):
    if ee_name == "inspire_dfx":
        from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
        return Inspire_Controller_DFX
    from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
    return Inspire_Controller_FTP


def apply_affinity_if_requested(args, log):
    if not args.affinity:
        return
    import psutil
    p = psutil.Process(os.getpid())
    p.cpu_affinity([0, 1, 2, 3])
    try:
        p.nice(-20)
        log.info("Set high priority successfully.")
    except psutil.AccessDenied:
        log.warning("Failed to set high priority. Please run as root.")

    for child in p.children(recursive=True):
        try:
            log.info(f"Child process {child.pid} name: {child.name()}")
            child.cpu_affinity([5, 6])
            child.nice(-20)
        except psutil.AccessDenied:
            pass


def setup_sim(args, components: RealTeleopComponents):
    if not args.sim:
        return
    components.reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
    components.reset_pose_publisher.Init()
    from teleop.sim.sim_state_topic import start_sim_state_subscribe
    components.sim_state_subscriber = start_sim_state_subscribe()


def setup_recorder(args, validation_manager=None):
    if not args.record:
        return None
    return EpisodeWriter(
        task_dir=os.path.join(args.task_dir, args.task_name),
        task_goal=args.task_goal,
        task_desc=args.task_desc,
        task_steps=args.task_steps,
        frequency=args.frequency,
        image_size=[args.camera_width, args.camera_height],
        rerun_log=not args.headless,
        episode_finalized_callback=(
            validation_manager.validate_after_finalize if validation_manager is not None else None
        ),
    )


def setup_dex1_tcp_fk(args, components: RealTeleopComponents, log) -> None:
    mobile_tcp23 = str(getattr(args, "online_inference_protocol_profile", "")) == "mobile_tcp23"
    if not bool(getattr(args, "record_mobile_training_state", False)) and not mobile_tcp23:
        return
    if bool(getattr(args, "record_mobile_training_state", False)) and not bool(getattr(args, "record", False)):
        raise ValueError("--record-mobile-training-state requires --record")
    if str(getattr(args, "arm", "")) != "G1_29":
        raise ValueError("Dex1 TCP recording requires --arm G1_29")
    if str(getattr(args, "ee", "")) != "dex1" or bool(getattr(args, "no_gripper", False)):
        raise ValueError("Dex1 TCP recording requires --ee dex1 without --no-gripper")
    repo_root = Path(__file__).resolve().parents[2]
    components.dex1_tcp_fk = Dex1TcpFkProvider(
        repo_root / "assets/g1_d/g1_d.urdf",
        repo_root / "assets/dex1_1/dex1_1.urdf",
    )
    if bool(getattr(args, "record_mobile_training_state", False)):
        apply_dex1_tcp_episode_metadata(args, components)
    metadata = components.dex1_tcp_fk.metadata()
    log.info("[DEX1_TCP_FK] G1D FK URDF: %s", metadata["robot_fk_urdf"])
    log.info("[DEX1_TCP_FK] Dex1.1 model URDF: %s", metadata["eef_model_urdf"])
    log.info(
        "[DEX1_TCP_FK] base_link == G1D URDF AGV_link; left wrist->tcp xyz=%s rpy=%s; right wrist->tcp xyz=%s rpy=%s",
        metadata["left_wrist_to_tcp_xyz_m"],
        metadata["left_wrist_to_tcp_rpy_rad"],
        metadata["right_wrist_to_tcp_xyz_m"],
        metadata["right_wrist_to_tcp_rpy_rad"],
    )


def apply_dex1_tcp_episode_metadata(args, components: RealTeleopComponents) -> None:
    if components.dex1_tcp_fk is None:
        return
    if components.recorder is None:
        raise RuntimeError("Dex1 TCP recording metadata requires an EpisodeWriter")
    metadata = components.dex1_tcp_fk.metadata()
    if bool(getattr(args, "record_mobile_training_state", False)):
        velocity_only = bool(getattr(args, "record_base_velocity_only", False))
        metadata.update(
            {
                "mobile_base_observation_mode": (
                    "velocity_only_base_link" if velocity_only else "global_slam_map_and_velocity"
                ),
                "mobile_base_absolute_pose_recorded": not velocity_only,
                "mobile_base_velocity_frame": str(getattr(args, "base_velocity_frame", "") or ""),
            }
        )
    components.recorder.update_episode_info(metadata)


def switch_recording_root(args, components: RealTeleopComponents, root_dir: str | Path, log) -> None:
    """Replace an idle writer so subsequent episodes use one exact root directory."""
    if not args.record:
        raise RuntimeError("recording root cannot change because --record is disabled")
    validation_manager = components.validation_manager
    if validation_manager is not None and validation_manager.is_busy():
        raise RuntimeError("recording root cannot change while episode validation is pending")
    previous_recorder = components.recorder
    if previous_recorder is None or not previous_recorder.is_ready():
        raise RuntimeError("recording root cannot change while an episode writer is active")

    root = Path(root_dir).expanduser().resolve()
    if not root.name:
        raise ValueError("recording root must not be the filesystem root")
    root.mkdir(parents=True, exist_ok=True)
    previous_recorder.close()
    args.task_dir = str(root.parent)
    args.task_name = root.name
    components.recorder = setup_recorder(args, validation_manager)
    apply_dex1_tcp_episode_metadata(args, components)
    components.recording_flow = setup_recording_flow(args, components, log)
    log.info("[RECORD_ROOT] switched recording root to %s", root)

def setup_cameras(args, log) -> CameraRuntime:
    cameras = CameraRuntime()
    if not bool(args.record or args.ui or args.input_provider == "online_inference"):
        return cameras
    cameras.head_remote = maybe_open_remote_camera("head", args.head_zmq_endpoint, log)
    cameras.left_remote = maybe_open_remote_camera("left_wrist", args.left_zmq_endpoint, log)
    cameras.right_remote = maybe_open_remote_camera("right_wrist", args.right_zmq_endpoint, log)
    cameras.head_local = maybe_open_local_camera("head", args.head_camera_id, args, log)
    cameras.left_local = maybe_open_local_camera("left_wrist", args.left_camera_id, args, log)
    cameras.right_local = maybe_open_local_camera("right_wrist", args.right_camera_id, args, log)
    return cameras


def setup_base_state_receiver(args, log):
    mobile_mode = str(getattr(args, "mobile_manipulation_mode", "direct_ik")) == "mobile_ik_qp"
    mobile_profile = str(getattr(args, "online_inference_protocol_profile", "")) in {"mobile_tcp23", "mobile_joint_base"}
    if not bool(getattr(args, "record_base", False)) and not mobile_mode and not mobile_profile:
        return None
    receiver = BaseStateReceiver(
        odom_topic=args.base_odom_topic,
        height_topic=args.base_height_topic,
        history_size=args.base_history_size,
        network_interface=args.network_interface,
        record_slam_map_pose=bool(args.record_slam_map_pose or mobile_profile),
        slam_pose_source_frame=args.slam_pose_source_frame,
        base_velocity_frame=args.base_velocity_frame,
        slam_chain_max_skew_ms=args.slam_chain_max_skew_ms,
    )
    receiver.start()
    timeout_sec = float(args.base_startup_timeout_sec)
    if not receiver.wait_until_ready(timeout_sec):
        receiver.close()
        raise RuntimeError(
            "[BASE_RECORD] failed to receive initial base data within "
            f"{timeout_sec:.1f}s "
            f"(odom_topic={args.base_odom_topic!r}, height_topic={args.base_height_topic!r})"
        )
    log.info(
        "[BASE_STATE] enabled: odom_topic=%s, height_topic=%s, slam_pose_source_frame=%s, mobile_ik_qp=%s",
        args.base_odom_topic,
        args.base_height_topic or "<disabled>",
        args.slam_pose_source_frame if (args.record_slam_map_pose or mobile_profile) else "<disabled>",
        mobile_mode,
    )
    return receiver


def maybe_open_local_camera(name: str, camera_id: int, args, log):
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
        log.warning(f"[CAM] failed to open local camera '{name}' (id={camera_id}): {e}")
        return None


def maybe_open_remote_camera(name: str, endpoint: str, log):
    endpoint = str(endpoint or "").strip()
    if not endpoint:
        return None
    try:
        return ZMQRawCameraReceiver(endpoint=endpoint, name=name)
    except Exception as e:
        log.warning(f"[CAM] failed to connect remote camera '{name}' ({endpoint}): {e}")
        return None


def setup_recording_flow(args, components: RealTeleopComponents, log):
    if not args.record:
        return None
    return TeleopRecordingFlow(
        args=args,
        recorder=components.recorder,
        log=log,
        reset_callback=(
            (lambda: publish_reset_category(1, components.reset_pose_publisher, log))
            if args.sim
            else None
        ),
        dual_hand_data_lock=components.ee.dual_hand_data_lock,
        dual_hand_state_array=components.ee.dual_hand_state_array,
        dual_hand_action_array=components.ee.dual_hand_action_array,
        dual_gripper_data_lock=components.ee.dual_gripper_data_lock,
        dual_gripper_state_array=components.ee.dual_gripper_state_array,
        dual_gripper_action_array=components.ee.dual_gripper_action_array,
        validation_manager=components.validation_manager,
    )
