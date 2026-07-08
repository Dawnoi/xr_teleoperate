"""Argument parser for the real hand-and-arm teleoperation entrypoint."""

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand',
                        help='Select XR device input tracking source. hand uses XR hand wrist as arm locator and pinch as the grip/deadman signal.')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    parser.add_argument('--controller-deadman', type=str, choices=['grip', 'none'], default='grip',
                        help='Controller safety enable logic. "grip" means each arm/ee only moves while the same-side grip is held.')
    parser.add_argument('--max-arm-joint-speed', type=float, default=1.5,
                        help='Outer-loop arm target speed limit in rad/s. Lower values reduce sudden jumps from teleop/IK.')
    parser.add_argument('--arm-control-hz', type=float, default=250.0,
                        help='Arm DDS publish frequency in Hz. Use 500 to test lower ctrl_wait.')
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
    parser.add_argument('--input-provider', type=str, choices=['xr', 'lerobot_offline', 'online_inference', 'online_inference_rtc'], default='xr',
                        help='Teleop input provider backend: live XR input, offline LeRobot replay, or Pika-style online inference.')
    parser.add_argument('--offline-replay-dataset-root', type=str, default='',
                        help='Dataset root for offline replay provider.')
    parser.add_argument('--offline-replay-episode-index', type=int, default=0,
                        help='Episode index for offline replay provider.')
    parser.add_argument('--offline-replay-arm-source', type=str, choices=['action', 'state', 'fk_cmd_pose'], default='action',
                        help='Arm source used by offline replay provider.')
    parser.add_argument('--offline-replay-speed-scale', type=float, default=1.0,
                        help='Playback speed scale for offline replay provider.')
    parser.add_argument('--offline-replay-end-action', type=str, choices=['home', 'hold'], default='home',
                        help='Robot action to take when offline replay reaches the end.')
    parser.add_argument('--online-inference-host', type=str, default='127.0.0.1',
                        help='Pika-compatible online inference TCP server host.')
    parser.add_argument('--online-inference-port', type=int, default=5555,
                        help='Pika-compatible online inference TCP server port.')
    parser.add_argument('--online-inference-transport', type=str, choices=['tcp_jsonl', 'http'], default='tcp_jsonl',
                        help='Online inference transport. Use http for pi0.5 service on /handshake and /infer.')
    parser.add_argument('--online-inference-base-url', type=str, default='',
                        help='HTTP base URL for online inference, e.g. http://115.190.134.186:8017. If empty, host/port are used.')
    parser.add_argument('--online-inference-http-handshake-path', type=str, default='/handshake',
                        help='HTTP handshake path for online inference.')
    parser.add_argument('--online-inference-http-infer-path', type=str, default='/infer',
                        help='HTTP infer path for online inference.')
    parser.add_argument('--online-inference-protocol-profile', type=str, choices=['pika_pose7', 'pi05_dual_arm_20d'], default='pika_pose7',
                        help='Online inference payload/action schema profile.')
    parser.add_argument('--online-inference-prompt', type=str, default='',
                        help='Task prompt sent to online inference services such as pi0.5.')
    parser.add_argument('--online-inference-arm-side', type=str, choices=['left', 'right', 'both'], default='both',
                        help='Arm side controlled by online inference.')
    parser.add_argument('--online-inference-n-obs-steps', type=int, default=2,
                        help='Number of observation history steps sent to online inference.')
    parser.add_argument('--online-inference-camera-freq', type=float, default=30.0,
                        help='Nominal camera frequency used for online inference observation history.')
    parser.add_argument('--online-inference-jpeg-quality', type=int, default=85,
                        help='JPEG quality for online inference observation images.')
    parser.add_argument('--online-inference-action-step-sec', type=float, default=0.10,
                        help='Nominal server action step duration in seconds.')
    parser.add_argument('--online-inference-chunk-step-mode', type=str, choices=['timed', 'per_tick'], default='per_tick',
                        help='How to advance action steps inside an online inference chunk.')
    parser.add_argument('--online-inference-interp-sec', type=float, default=0.01,
                        help='Nominal server interpolation interval in seconds.')
    parser.add_argument('--online-inference-post-action-delay-ms', type=int, default=75,
                        help='Delay after executing an action chunk before sending the next observation.')
    parser.add_argument('--online-inference-response-timeout-sec', type=float, default=2.0,
                        help='Timeout while waiting for an online inference action response.')
    parser.add_argument('--online-inference-transform-config', type=str, default='',
                        help='JSON pose transform config. Required with --online-inference-enable-motion.')
    parser.add_argument('--online-inference-enable-motion', action='store_true',
                        help='Allow online inference provider to enable robot motion when all safety checks pass.')
    parser.add_argument('--online-inference-dry-run', action='store_true',
                        help='Send observations and parse actions without enabling arm motion.')
    parser.add_argument('--online-inference-rtc-horizon', type=int, default=50,
                        help='RTC action chunk horizon. Only used by --input-provider online_inference_rtc.')
    parser.add_argument('--online-inference-rtc-s', type=int, default=10,
                        help='RTC request step s. Only used by --input-provider online_inference_rtc.')
    parser.add_argument('--online-inference-rtc-d', type=int, default=7,
                        help='RTC predicted delay steps d sent to the server. The local takeover cursor is computed from the actual executed old-chunk cursor, not fixed to d.')
    parser.add_argument('--auto-start', action='store_true',
                        help='Enter START state automatically after initialization. Intended for offline replay entrypoints.')
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
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    parser.add_argument('--no-gripper', action='store_true',
                        help='Disable end-effector controller initialization and commands. Useful for arm-only offline replay.')
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

    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)
