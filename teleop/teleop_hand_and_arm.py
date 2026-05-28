import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
from collections import deque
import numpy as np
import cv2
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
from teleop.utils.episode_writer import EpisodeWriter, ZMQRawCameraReceiver
from teleop.utils.g1d_agv_bridge import G1DAgvBridge
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from teleop.utils.xr_robotics_wrapper import XRRoboticsWrapper
from teleop.utils.arm_target_safety import limit_arm_joint_target_velocity
from teleop.utils.arm_workspace_safety import (
    clamp_dual_wrist_poses_to_box,
    clamp_dual_wrist_poses_to_tapered_workspace,
)
from teleop.utils.local_camera import LocalCameraStream
from teleop.utils.simple_latency_trace import SimpleLatencyTracker
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


class TimingDebugger:
    def __init__(self, enabled: bool = False, interval_sec: float = 2.0, history_size: int = 400):
        self.enabled = enabled
        self.interval_sec = max(0.5, float(interval_sec))
        self.history_size = max(50, int(history_size))
        self._last_report_time = time.time()
        self._reset()

    def _make_window(self):
        return deque(maxlen=self.history_size)

    def _reset(self):
        self.loop_count = 0
        self.loop_total = 0.0
        self.loop_max = 0.0
        self.overrun_count = 0
        self.tele_fetch_total = 0.0
        self.tele_fetch_max = 0.0
        self.tele_none_count = 0
        self.ik_total = 0.0
        self.ik_max = 0.0
        self.ik_count = 0
        self.agv_total = 0.0
        self.agv_max = 0.0
        self.agv_count = 0
        self.loop_window = self._make_window()
        self.tele_window = self._make_window()
        self.ik_window = self._make_window()
        self.agv_window = self._make_window()

    def _p95_ms(self, values):
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=float), 95) * 1000.0)

    def add_loop(self, dt: float, overrun: bool):
        if not self.enabled:
            return
        self.loop_count += 1
        self.loop_total += dt
        self.loop_max = max(self.loop_max, dt)
        self.loop_window.append(float(dt))
        if overrun:
            self.overrun_count += 1

    def add_tele_fetch(self, dt: float, got_data: bool):
        if not self.enabled:
            return
        self.tele_fetch_total += dt
        self.tele_fetch_max = max(self.tele_fetch_max, dt)
        self.tele_window.append(float(dt))
        if not got_data:
            self.tele_none_count += 1

    def add_ik(self, dt: float):
        if not self.enabled:
            return
        self.ik_total += dt
        self.ik_max = max(self.ik_max, dt)
        self.ik_count += 1
        self.ik_window.append(float(dt))

    def add_agv(self, dt: float):
        if not self.enabled:
            return
        self.agv_total += dt
        self.agv_max = max(self.agv_max, dt)
        self.agv_count += 1
        self.agv_window.append(float(dt))

    def maybe_report(self, arm_ctrl=None, gripper_ctrl=None):
        if not self.enabled:
            return
        now = time.time()
        if (now - self._last_report_time) < self.interval_sec:
            return

        arm_age = None
        if arm_ctrl is not None and hasattr(arm_ctrl, "lowstate_buffer"):
            try:
                arm_age = arm_ctrl.lowstate_buffer.GetAge()
            except Exception:
                arm_age = None

        gripper_age = None
        if gripper_ctrl is not None and hasattr(gripper_ctrl, "get_state_age"):
            try:
                gripper_age = gripper_ctrl.get_state_age()
            except Exception:
                gripper_age = None

        loop_avg_ms = (self.loop_total / self.loop_count * 1000.0) if self.loop_count else 0.0
        tele_avg_ms = (self.tele_fetch_total / self.loop_count * 1000.0) if self.loop_count else 0.0
        ik_avg_ms = (self.ik_total / self.ik_count * 1000.0) if self.ik_count else 0.0
        agv_avg_ms = (self.agv_total / self.agv_count * 1000.0) if self.agv_count else 0.0

        timing_msg = (
            "[TIMING] "
            f"loop avg/p95/max={loop_avg_ms:.1f}/{self._p95_ms(self.loop_window):.1f}/{self.loop_max * 1000.0:.1f} ms, "
            f"tele avg/p95/max={tele_avg_ms:.1f}/{self._p95_ms(self.tele_window):.1f}/{self.tele_fetch_max * 1000.0:.1f} ms, "
            f"ik avg/p95/max={ik_avg_ms:.1f}/{self._p95_ms(self.ik_window):.1f}/{self.ik_max * 1000.0:.1f} ms ({self.ik_count} calls), "
            f"agv avg/p95/max={agv_avg_ms:.1f}/{self._p95_ms(self.agv_window):.1f}/{self.agv_max * 1000.0:.1f} ms ({self.agv_count} calls)"
        )
        if arm_age is not None:
            timing_msg += f", arm_state_age_ms={arm_age * 1000.0:.1f}"
        logger_mp.info(timing_msg)

        arm_timing = None
        if arm_ctrl is not None and hasattr(arm_ctrl, "get_timing_snapshot"):
            try:
                arm_timing = arm_ctrl.get_timing_snapshot()
            except Exception:
                arm_timing = None

        if arm_timing is not None:
            loop_stats = arm_timing.get("loop_stats") or {}
            write_stats = arm_timing.get("write_stats") or {}
            enqueue_stats = arm_timing.get("enqueue_stats") or {}
            logger_mp.info(
                "[TIMING_DDS] publish_hz=%.1f, ctrl_loop avg/p95/max=%.2f/%.2f/%.2f ms, dds_write avg/p95/max=%.3f/%.3f/%.3f ms, enqueue->publish avg/p95/max=%s/%s/%s ms",
                arm_timing.get("publish_hz", 0.0),
                loop_stats.get("avg_ms", 0.0),
                loop_stats.get("p95_ms", 0.0),
                loop_stats.get("max_ms", 0.0),
                write_stats.get("avg_ms", 0.0),
                write_stats.get("p95_ms", 0.0),
                write_stats.get("max_ms", 0.0),
                f"{enqueue_stats.get('avg_ms', 0.0):.2f}" if enqueue_stats else "N/A",
                f"{enqueue_stats.get('p95_ms', 0.0):.2f}" if enqueue_stats else "N/A",
                f"{enqueue_stats.get('max_ms', 0.0):.2f}" if enqueue_stats else "N/A",
            )

        state_parts = [f"tele_none={self.tele_none_count}", f"overrun={self.overrun_count}/{self.loop_count}"]
        if arm_age is not None:
            state_parts.insert(0, f"arm_state_age_ms={arm_age * 1000.0:.1f}")
        if gripper_age is not None:
            state_parts.insert(1 if arm_age is not None else 0, f"gripper_state_age_ms={gripper_age * 1000.0:.1f}")
        logger_mp.info("[TIMING_STATE] " + ", ".join(state_parts))

        self._last_report_time = now
        self._reset()


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
    recorder = None
    ipc_server = None
    sim_state_subscriber = None
    head_camera = None
    left_camera = None
    right_camera = None
    head_remote_camera = None
    left_remote_camera = None
    right_remote_camera = None
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
    parser.add_argument('--disable-arm-workspace-limit', action='store_true',
                        help='Disable wrist workspace clamping before IK.')
    parser.add_argument('--arm-workspace-mode', type=str, choices=['tapered', 'box'], default='tapered',
                        help='Workspace shape before IK. "tapered" = lower narrow / upper wide inverted-trapezoid prism. "box" = fixed rectangular box.')
    parser.add_argument('--arm-workspace-min', type=float, nargs=3, default=[0.10, -0.32, -0.08],
                        metavar=('XMIN', 'YMIN', 'ZMIN'),
                        help='Forward box workspace lower bound in the arm IK/base frame, applied before IK when --arm-workspace-mode box.')
    parser.add_argument('--arm-workspace-max', type=float, nargs=3, default=[0.45, 0.32, 0.42],
                        metavar=('XMAX', 'YMAX', 'ZMAX'),
                        help='Forward box workspace upper bound in the arm IK/base frame, applied before IK when --arm-workspace-mode box.')
    parser.add_argument('--arm-workspace-z-min', type=float, default=-0.05,
                        help='Tapered workspace lower z bound in the arm IK/base frame.')
    parser.add_argument('--arm-workspace-z-max', type=float, default=0.45,
                        help='Tapered workspace upper z bound in the arm IK/base frame.')
    parser.add_argument('--arm-workspace-x-min', type=float, default=0.10,
                        help='Tapered workspace minimum forward x bound.')
    parser.add_argument('--arm-workspace-x-max-low', type=float, default=0.38,
                        help='Tapered workspace forward x upper bound at z_min.')
    parser.add_argument('--arm-workspace-x-max-high', type=float, default=0.52,
                        help='Tapered workspace forward x upper bound at z_max.')
    parser.add_argument('--arm-workspace-y-max-low', type=float, default=0.24,
                        help='Tapered workspace lateral |y| bound at z_min.')
    parser.add_argument('--arm-workspace-y-max-high', type=float, default=0.38,
                        help='Tapered workspace lateral |y| bound at z_max.')
    parser.add_argument('--timing-debug', action='store_true',
                        help='Enable periodic timing / staleness logs for diagnosing wireless lag and runtime stalls.')
    parser.add_argument('--timing-debug-interval', type=float, default=2.0,
                        help='Seconds between timing debug reports when --timing-debug is enabled.')
    parser.add_argument('--latency-trace', action='store_true',
                        help='Enable simple arm latency tracing: 收到输入 -> DDS下发 -> 执行响应.')
    parser.add_argument('--latency-trace-path', type=str, default='./utils/data/latency_trace.jsonl',
                        help='Path to save latency trace jsonl records.')
    parser.add_argument('--latency-command-threshold', type=float, default=0.02,
                        help='Minimum max joint delta (rad) to start a new latency trace sample.')
    parser.add_argument('--latency-exec-q-threshold', type=float, default=0.01,
                        help='Execution-detected threshold on joint position delta (rad).')
    parser.add_argument('--latency-exec-dq-threshold', type=float, default=0.05,
                        help='Execution-detected threshold on joint velocity magnitude (rad/s).')
    parser.add_argument('--latency-timeout', type=float, default=2.0,
                        help='Timeout in seconds for one latency trace sample.')
    parser.add_argument('--latency-summary-every', type=int, default=10,
                        help='Print percentile summary every N completed/timeout latency samples.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--record-arm-repr', type=str, choices=['qpos', 'pose', 'both'], default='qpos',
                        help='Recording representation for arm data: joint angles (qpos), wrist pose, or both.')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')
    parser.add_argument('--head-camera-id', type=int, default=-1, help='Local head RGB camera id for cv2.VideoCapture, e.g. 0. <0 disables.')
    parser.add_argument('--left-camera-id', type=int, default=-1, help='Local left wrist RGB camera id for cv2.VideoCapture. <0 disables.')
    parser.add_argument('--right-camera-id', type=int, default=-1, help='Local right wrist RGB camera id for cv2.VideoCapture. <0 disables.')
    parser.add_argument('--camera-width', type=int, default=640, help='Requested local camera frame width.')
    parser.add_argument('--camera-height', type=int, default=480, help='Requested local camera frame height.')
    parser.add_argument('--camera-fps', type=int, default=30, help='Requested local camera FPS.')
    parser.add_argument('--camera-fourcc', type=str, default='MJPG', help='Requested local camera FOURCC, e.g. MJPG/YUYV.')
    parser.add_argument('--camera-buffer-size', type=int, default=1, help='Requested local camera driver buffer size.')
    parser.add_argument('--head-zmq-endpoint', type=str, default='', help='Remote ZMQ raw endpoint for head camera, e.g. tcp://192.168.1.10:5555')
    parser.add_argument('--left-zmq-endpoint', type=str, default='', help='Remote ZMQ raw endpoint for left wrist camera.')
    parser.add_argument('--right-zmq-endpoint', type=str, default='', help='Remote ZMQ raw endpoint for right wrist camera.')

    args = parser.parse_args()
    logger_mp.debug(f"args: {args}")
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
    record_start_monotonic_ns = None
    camera_align_max_delta_ns = int(max(20_000_000, (1.5 / max(args.frequency, 1e-6)) * 1e9))
    state_align_max_delta_ns = int(max(10_000_000, (0.75 / max(args.frequency, 1e-6)) * 1e9))
    action_align_max_delta_ns = state_align_max_delta_ns
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

    def append_timed_sample(buffer: deque, timestamp_ns: int, **payload):
        entry = {"t_ns": int(timestamp_ns)}
        entry.update(payload)
        buffer.append(entry)

    def nearest_timed_sample(buffer: deque, target_ns: int, max_delta_ns: int | None = None, min_timestamp_ns: int | None = None):
        best = None
        best_abs_delta = None
        target_ns = int(target_ns)
        for entry in reversed(buffer):
            t_ns = int(entry["t_ns"])
            if min_timestamp_ns is not None and t_ns < int(min_timestamp_ns):
                continue
            abs_delta = abs(t_ns - target_ns)
            if max_delta_ns is not None and abs_delta > int(max_delta_ns):
                continue
            if best_abs_delta is None or abs_delta < best_abs_delta:
                best = entry
                best_abs_delta = abs_delta
        if best is None:
            return None
        result = dict(best)
        result["delta_to_target_ns"] = int(best["t_ns"] - target_ns)
        result["target_monotonic_ns"] = target_ns
        return result

    def camera_meta_monotonic_ns(meta):
        if meta is None:
            return None
        value = meta.get("host_recv_monotonic_ns")
        if value is None:
            value = meta.get("host_monotonic_ns")
        return int(value) if value is not None else None

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
                                     image_size = [args.camera_width, args.camera_height],
                                     rerun_log = not args.headless)
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
        post_home_takeover_armed = False
        prev_left_grip_pressed = False
        prev_right_grip_pressed = False
        left_takeover_settle_frames = 0
        right_takeover_settle_frames = 0
        TAKEOVER_SETTLE_FRAMES = 0 if args.controller_mapping_mode == "legacy_main" else 2
        recording_waiting_for_first_frame = False
        last_record_wait_log_ns = 0

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING and not recording_waiting_for_first_frame:
                    if recorder.create_episode():
                        record_start_monotonic_ns = int(time.monotonic_ns())
                        recording_waiting_for_first_frame = True
                        logger_mp.info("[RECORD_ALIGN] episode armed, waiting for first post-start camera frame before recording.")
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recording_waiting_for_first_frame = False
                    record_start_monotonic_ns = None
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

            # get xr's tele data
            tele_fetch_start = time.perf_counter()
            tele_data = tv_wrapper.get_tele_data(
                current_left_robot_wrist_pose=current_left_wrist_pose,
                current_right_robot_wrist_pose=current_right_wrist_pose,
            )
            tele_fetch_dt = time.perf_counter() - tele_fetch_start
            timing_debugger.add_tele_fetch(tele_fetch_dt, tele_data is not None)
            if tele_data is None:
                timing_debugger.maybe_report(arm_ctrl=arm_ctrl, gripper_ctrl=gripper_ctrl)
                time.sleep(0.01)
                continue
            tele_data_recv_ts_ns = time.perf_counter_ns()
            tele_fetch_ms = tele_fetch_dt * 1000.0

            takeover_logic_start = time.perf_counter()

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
            base_control_start = time.perf_counter()
            base_control_mode = "none"
            base_move_ms = 0.0
            base_height_ms = 0.0
            base_misc_ms = 0.0
            base_async_cycle_avg_ms = None
            base_async_move_avg_ms = None
            base_async_height_avg_ms = None
            base_async_queue_avg_ms = None
            base_async_publish_hz = None
            base_vx = 0.0
            base_vy = 0.0
            base_wz = 0.0
            base_z = 0.0
            if args.input_mode == "controller" and args.motion:
                base_control_mode = "loco"
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

                base_move_start = time.perf_counter()
                loco_wrapper.Move(base_vx, base_vy, base_wz)
                base_move_ms = (time.perf_counter() - base_move_start) * 1000.0
            elif args.input_mode == "controller" and args.base_controller == "g1d_agv":
                base_control_mode = "g1d_agv_async"
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
                    agv_send_start = time.perf_counter()
                    base_move_start = time.perf_counter()
                    agv_bridge.set_target(base_vx, base_vy, base_wz, base_z)
                    base_move_ms = (time.perf_counter() - base_move_start) * 1000.0
                    timing_debugger.add_agv(time.perf_counter() - agv_send_start)
                    try:
                        agv_timing_snapshot = agv_bridge.get_timing_snapshot()
                    except Exception:
                        agv_timing_snapshot = None
                    if agv_timing_snapshot is not None:
                        base_async_publish_hz = float(agv_timing_snapshot.get("publish_hz", 0.0))
                        move_stats = agv_timing_snapshot.get("move_stats") or {}
                        height_stats = agv_timing_snapshot.get("height_stats") or {}
                        cycle_stats = agv_timing_snapshot.get("cycle_stats") or {}
                        queue_stats = agv_timing_snapshot.get("queue_delay_stats") or {}
                        base_async_move_avg_ms = move_stats.get("avg_ms")
                        base_async_height_avg_ms = height_stats.get("avg_ms")
                        base_async_cycle_avg_ms = cycle_stats.get("avg_ms")
                        base_async_queue_avg_ms = queue_stats.get("avg_ms")
            base_control_ms = (time.perf_counter() - base_control_start) * 1000.0
            base_misc_ms = max(0.0, base_control_ms - base_move_ms - base_height_ms)

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            ik_ms = 0.0
            if any_zero_takeover_this_frame:
                if post_home_takeover_armed and normalized_head_mode in {"head_coupled", "hybrid"}:
                    tv_wrapper.sync_reference_to_current_live_pose(require_live=False)
                reset_arm_ik_state(arm_ik, current_lr_arm_q)
                post_home_takeover_armed = False
            if home_return_active:
                sol_q = home_target_q.copy()
                sol_tauff = current_hold_tauff.copy()
            elif left_arm_enabled or right_arm_enabled:
                left_target_pose = tele_data.left_wrist_pose
                right_target_pose = tele_data.right_wrist_pose
                if workspace_limit_enabled:
                    if workspace_mode == "box":
                        left_target_pose, right_target_pose, _ = clamp_dual_wrist_poses_to_box(
                            left_target_pose,
                            right_target_pose,
                            workspace_min,
                            workspace_max,
                        )
                    else:
                        left_target_pose, right_target_pose, _ = clamp_dual_wrist_poses_to_tapered_workspace(
                            left_target_pose,
                            right_target_pose,
                            tapered_workspace_params["z_min"],
                            tapered_workspace_params["z_max"],
                            tapered_workspace_params["x_min"],
                            tapered_workspace_params["x_max_low"],
                            tapered_workspace_params["x_max_high"],
                            tapered_workspace_params["y_max_low"],
                            tapered_workspace_params["y_max_high"],
                        )
                time_ik_start = time.perf_counter()
                sol_q, sol_tauff  = arm_ik.solve_ik(left_target_pose, right_target_pose, current_lr_arm_q, current_lr_arm_dq)
                ik_dt = time.perf_counter() - time_ik_start
                ik_ms = ik_dt * 1000.0
                timing_debugger.add_ik(ik_dt)
                logger_mp.debug(f"ik:\t{round(ik_dt, 6)}")
            else:
                sol_q = current_hold_q.copy()
                sol_tauff = current_hold_tauff.copy()

            if ((not left_arm_enabled) or left_zero_takeover_this_frame) and (not home_return_active):
                sol_q[:7] = current_hold_q[:7]
                sol_tauff[:7] = current_hold_tauff[:7]
            if ((not right_arm_enabled) or right_zero_takeover_this_frame) and (not home_return_active):
                sol_q[-7:] = current_hold_q[-7:]
                sol_tauff[-7:] = current_hold_tauff[-7:]

            if left_zero_takeover_this_frame:
                if left_takeover_rising_edge:
                    logger_mp.info(
                        f"[TAKEOVER][LEFT] grip rising edge -> zero-delta hold for "
                        f"{TAKEOVER_SETTLE_FRAMES} frames."
                    )
                left_takeover_settle_frames -= 1
            if right_zero_takeover_this_frame:
                if right_takeover_rising_edge:
                    logger_mp.info(
                        f"[TAKEOVER][RIGHT] grip rising edge -> zero-delta hold for "
                        f"{TAKEOVER_SETTLE_FRAMES} frames."
                    )
                right_takeover_settle_frames -= 1

            safety_start = time.perf_counter()
            sol_q = limit_arm_joint_target_velocity(
                sol_q,
                current_lr_arm_q,
                max_joint_speed=(args.home_return_speed if home_return_active else args.max_arm_joint_speed),
                control_frequency=args.frequency,
            )
            safety_ms = (time.perf_counter() - safety_start) * 1000.0
            gravity_start = time.perf_counter()
            sol_tauff = compute_arm_gravity_tauff(arm_ik, sol_q)
            gravity_ms = (time.perf_counter() - gravity_start) * 1000.0
            current_hold_q = sol_q.copy()
            current_hold_tauff = sol_tauff.copy()

            trace_seq = None
            if latency_tracker is not None and latency_tracker.can_start_new_trace():
                max_command_delta = float(np.max(np.abs(sol_q - current_lr_arm_q)))
                if max_command_delta >= args.latency_command_threshold:
                    trace_seq = latency_tracker.begin_trace(
                        recv_ts_ns=tele_data_recv_ts_ns,
                        recv_q=current_lr_arm_q,
                        extra={
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
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
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

                    sample_monotonic_ns = int(time.monotonic_ns())
                    record_min_timestamp_ns = int(record_start_monotonic_ns or sample_monotonic_ns)
                    aligned_state = nearest_timed_sample(
                        state_history,
                        sample_monotonic_ns,
                        max_delta_ns=state_align_max_delta_ns,
                        min_timestamp_ns=record_min_timestamp_ns,
                    )
                    aligned_action = nearest_timed_sample(
                        action_history,
                        sample_monotonic_ns,
                        max_delta_ns=action_align_max_delta_ns,
                        min_timestamp_ns=record_min_timestamp_ns,
                    )
                    if aligned_state is None:
                        logger_mp.warning("[RECORD_ALIGN] skip sample: no aligned state found near sample timestamp.")
                        continue
                    if aligned_action is None:
                        logger_mp.warning("[RECORD_ALIGN] skip sample: no aligned action found near sample timestamp.")
                        continue

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

                    colors = {}
                    depths = {}
                    camera_timestamps = {}
                    required_camera_missing = False
                    if head_source is not None:
                        head_frame, head_meta = head_source.get_nearest(
                            sample_monotonic_ns,
                            max_delta_ns=camera_align_max_delta_ns,
                            min_monotonic_ns=record_min_timestamp_ns,
                            copy=True,
                        )
                        if head_frame is not None:
                            colors["head"] = head_frame
                            camera_timestamps["head"] = head_meta
                        else:
                            required_camera_missing = True
                    if left_source is not None:
                        left_frame, left_meta = left_source.get_nearest(
                            sample_monotonic_ns,
                            max_delta_ns=camera_align_max_delta_ns,
                            min_monotonic_ns=record_min_timestamp_ns,
                            copy=True,
                        )
                        if left_frame is not None:
                            colors["left_wrist"] = left_frame
                            camera_timestamps["left_wrist"] = left_meta
                        else:
                            required_camera_missing = True
                    if right_source is not None:
                        right_frame, right_meta = right_source.get_nearest(
                            sample_monotonic_ns,
                            max_delta_ns=camera_align_max_delta_ns,
                            min_monotonic_ns=record_min_timestamp_ns,
                            copy=True,
                        )
                        if right_frame is not None:
                            colors["right_wrist"] = right_frame
                            camera_timestamps["right_wrist"] = right_meta
                        else:
                            required_camera_missing = True
                    if required_camera_missing:
                        logger_mp.warning("[RECORD_ALIGN] skip sample: no aligned camera frame found after record start.")
                        continue

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
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                    }
                    actions = {
                        "left_arm": left_arm_action_entry,
                        "right_arm": right_arm_action_entry,
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
                    }
                    timestamps = {
                        "sample_wall_time_ns": int(time.time_ns()),
                        "sample_monotonic_ns": sample_monotonic_ns,
                        "teleop_input_perf_counter_ns": int(tele_data_recv_ts_ns),
                        "state": {
                            "host_monotonic_ns": int(aligned_state["t_ns"]),
                            "delta_to_sample_ns": int(aligned_state["delta_to_target_ns"]),
                        },
                        "action": {
                            "host_monotonic_ns": int(aligned_action["t_ns"]),
                            "delta_to_sample_ns": int(aligned_action["delta_to_target_ns"]),
                        },
                        "camera": camera_timestamps,
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state, timestamps=timestamps)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, timestamps=timestamps)

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
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc and ipc_server is not None:
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
