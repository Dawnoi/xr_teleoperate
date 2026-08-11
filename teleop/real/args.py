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
    parser.add_argument('--home-position-tolerance-rad', type=float, default=0.015,
                        help='Maximum absolute joint error accepted as G1_29 Home completion.')
    parser.add_argument('--home-gravity-update-hz', type=float, default=20.0,
                        help='Rate at which G1_29 Home refreshes gravity feed-forward from measured arm joints.')
    parser.add_argument('--home-gravity-torque-limit-nm', type=float, default=10.0,
                        help='Maximum absolute per-joint gravity feed-forward torque accepted during G1_29 Home; values beyond this abort Home.')
    parser.add_argument('--mobile-manipulation-mode', choices=['direct_ik', 'mobile_ik_qp'], default='direct_ik',
                        help='direct_ik preserves the legacy path; mobile_ik_qp runs the measured-state G1-D whole-body velocity QP for both TCPs, arms, column, and base. mobile_ik_qp leaves waist yaw at its existing hardware hold target.')
    parser.add_argument('--mobile-state-timeout-sec', type=float, default=0.50,
                        help='Maximum age of required odom/column measurements in mobile_ik_qp mode.')
    parser.add_argument('--mobile-state-retry-count', type=int, default=3,
                        help='Number of fail-closed fresh-state retries after mobile_ik_qp detects stale odom or column feedback.')
    parser.add_argument('--mobile-state-retry-interval-sec', type=float, default=0.50,
                        help='Maximum wait for each fresh-state retry after mobile_ik_qp has entered arm/base hold.')
    parser.add_argument('--mobile-height-raw-minimum', type=float, default=-0.0015,
                        help='Calibrated rt/hispeed_state.y lower endpoint with a safety margin for the fully lowered G1D column.')
    parser.add_argument('--mobile-height-raw-maximum', type=float, default=0.431062324,
                        help='Calibrated rt/hispeed_state.y upper endpoint with a safety margin for the fully raised G1D column.')
    parser.add_argument('--mobile-column-travel-m', type=float, default=0.42,
                        help='Total physical G1D column travel used by mobile_ik_qp.')
    parser.add_argument('--mobile-max-torso-yaw-rate', type=float, default=0.50,
                        help='Maximum independent waist-yaw rate in rad/s from the right thumbstick X axis.')
    parser.add_argument('--mobile-wbc-command-horizon-sec', type=float, default=0.10,
                        help='Position lookahead horizon applied to mobile_ik_qp arm QP velocities.')
    parser.add_argument('--mobile-wbc-max-position-lead-rad', type=float, default=0.12,
                        help='Maximum QP position lookahead from measured arm joint state in radians.')
    parser.add_argument('--disable-mobile-wbc-collision-avoidance', action='store_true',
                        help='Disable mobile_ik_qp WBC collision constraints. This is a high-risk explicit override.')
    parser.add_argument('--base-max-vx', type=float, default=0.3,
                        help='Maximum commanded chassis x velocity in m/s from the left thumbstick Y axis.')
    parser.add_argument('--base-max-vy', type=float, default=0.3,
                        help='Maximum commanded chassis y velocity in m/s from the left thumbstick X axis.')
    parser.add_argument('--base-max-wz', type=float, default=0.3,
                        help='Maximum commanded chassis angular z velocity in rad/s from the right thumbstick X axis.')
    parser.add_argument('--base-max-z', type=float, default=1.0,
                        help='Maximum normalized G1D AGV HeightAdjust command from the right thumbstick Y axis when --base-controller g1d_agv is enabled.')
    parser.add_argument('--base-stick-deadzone', type=float, default=0.08,
                        help='Thumbstick deadzone for chassis motion commands in motion mode.')
    parser.add_argument('--base-controller', type=str, choices=['none', 'loco', 'g1d_agv'], default='none',
                        help='Mobile-base command backend. none disables chassis output, loco uses Unitree loco Move, g1d_agv uses the official G1D AGV bridge.')
    parser.add_argument('--base-command-source', type=str, choices=['none', 'controller', 'provider'], default='controller',
                        help='Mobile-base command source. controller uses XR controller sticks, provider uses the active replay/inference provider base action.')
    parser.add_argument('--base-motion', action='store_true',
                        help='Permit selected mobile-base backend output without changing the arm DDS command route.')
    parser.add_argument('--head-reference-mode', type=str, choices=['head_coupled', 'fixed_per_grip', 'live_head_reference', 'head_decoupled_live', 'hybrid', 'calibrated', 'live'], default='live_head_reference',
                        help='Controller wrist reference frame. "head_coupled" = fixed once after calibration. "fixed_per_grip" = freeze the current head translation at each grip takeover, release it when grip is released. "live_head_reference" = current head translation is always the live reference. "hybrid" = live when idle, frozen while gripping. Legacy aliases: calibrated=head_coupled, live/head_decoupled_live=live_head_reference semantics.')
    parser.add_argument('--controller-orientation-mode', type=str, choices=['absolute', 'relative', 'neutral'], default='absolute',
                        help='Wrist orientation control. "absolute" matches the original main-branch controller feel most closely (controller orientation directly drives wrist orientation). "relative" uses controller rotation delta from the current grip anchor. "neutral" fixes wrist orientation.')
    parser.add_argument('--controller-mapping-mode', type=str, choices=['legacy_main', 'anchored_safe'], default='anchored_safe',
                        help='"legacy_main" reproduces the original main-branch controller mapping semantics as closely as possible. "anchored_safe" uses the newer grip-anchor based takeover-safe mapping.')
    parser.add_argument('--calibration-mode', type=str, choices=['manual', 'auto'], default='manual',
                        help='Calibration trigger in head_coupled/hybrid mode. "manual" waits for key c after r; "auto" calibrates once live pose data is available. fixed_per_grip/live_head_reference do not require manual calibration.')
    parser.add_argument('--input-provider', type=str, choices=['xr', 'lerobot_offline', 'online_inference'], default='xr',
                        help='Teleop input provider backend: live XR input, offline LeRobot replay, or Pika-style online inference.')
    parser.add_argument('--offline-replay-dataset-root', type=str, default='',
                        help='Dataset root for offline replay provider.')
    parser.add_argument('--offline-replay-episode-index', type=int, default=0,
                        help='Episode index for offline replay provider.')
    parser.add_argument('--offline-replay-arm-source', type=str, choices=['action', 'state', 'fk_cmd_pose'], default='action',
                        help='Arm source used by offline replay provider.')
    parser.add_argument('--offline-replay-base-source', type=str, choices=['none', 'action'], default='none',
                        help='Base source used by raw offline replay provider. action replays actions.base as provider base commands.')
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
    parser.add_argument('--online-inference-protocol-profile', type=str, choices=['pika_pose7', 'pi05_dual_arm_20d', 'mobile_tcp23', 'mobile_pelvis_planar22', 'mobile_joint_base'], default='pika_pose7',
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
    parser.add_argument('--auto-start', action='store_true',
                        help='Enter START state automatically after initialization. Intended for offline replay entrypoints.')
    parser.add_argument('--disable-arm-workspace-limit', action='store_true',
                        help='Disable wrist workspace clamping before IK.')
    parser.add_argument('--arm-workspace-mode', type=str, choices=['tapered', 'box'], default='tapered',
                        help='Workspace shape before IK. "tapered" = lower narrow / upper wide inverted-trapezoid prism. "box" = fixed rectangular box.')
    parser.add_argument('--arm-workspace-layout', choices=['shared', 'per_arm'], default='shared',
                        help='shared uses one workspace for both wrists. per_arm requires explicit left/right workspace arguments and lets either gripped wrist independently drive mobile_ik_qp.')
    parser.add_argument('--arm-workspace-min', type=float, nargs=3, default=[0.10, -0.28, -0.055],
                        metavar=('XMIN', 'YMIN', 'ZMIN'),
                        help='Forward box workspace lower bound in the arm IK/base frame, applied before IK when --arm-workspace-mode box.')
    parser.add_argument('--arm-workspace-max', type=float, nargs=3, default=[0.45, 0.28, 0.245],
                        metavar=('XMAX', 'YMAX', 'ZMAX'),
                        help='Forward box workspace upper bound in the arm IK/base frame, applied before IK when --arm-workspace-mode box.')
    parser.add_argument('--arm-workspace-z-min', type=float, default=-0.055,
                        help='Tapered workspace lower z bound in the arm IK/base frame. Default is the G1D URDF zero-pose Dex1 TCP height (0.095226 m) minus 0.15 m.')
    parser.add_argument('--arm-workspace-z-max', type=float, default=0.245,
                        help='Tapered workspace upper z bound in the arm IK/base frame. Default is the G1D URDF zero-pose Dex1 TCP height (0.095226 m) plus 0.15 m.')
    parser.add_argument('--arm-workspace-x-min', type=float, default=0.154,
                        help='Tapered workspace minimum forward x bound. Default is the G1D URDF zero-pose J6 wrist-pitch joint center (0.153778 m in the arm IK frame), rounded to millimetres.')
    parser.add_argument('--arm-workspace-x-max-low', type=float, default=0.38,
                        help='Tapered workspace forward x upper bound at z_min.')
    parser.add_argument('--arm-workspace-x-max-high', type=float, default=0.52,
                        help='Tapered workspace forward x upper bound at z_max.')
    parser.add_argument('--arm-workspace-y-max-low', type=float, default=0.20,
                        help='Tapered workspace lateral |y| bound at z_min.')
    parser.add_argument('--arm-workspace-y-max-high', type=float, default=0.28,
                        help='Tapered workspace lateral |y| bound at z_max.')
    parser.add_argument('--left-arm-workspace-min', type=float, nargs=3, default=None,
                        metavar=('XMIN', 'YMIN', 'ZMIN'),
                        help='Required left box workspace lower bound when --arm-workspace-layout per_arm --arm-workspace-mode box.')
    parser.add_argument('--left-arm-workspace-max', type=float, nargs=3, default=None,
                        metavar=('XMAX', 'YMAX', 'ZMAX'),
                        help='Required left box workspace upper bound when --arm-workspace-layout per_arm --arm-workspace-mode box.')
    parser.add_argument('--right-arm-workspace-min', type=float, nargs=3, default=None,
                        metavar=('XMIN', 'YMIN', 'ZMIN'),
                        help='Required right box workspace lower bound when --arm-workspace-layout per_arm --arm-workspace-mode box.')
    parser.add_argument('--right-arm-workspace-max', type=float, nargs=3, default=None,
                        metavar=('XMAX', 'YMAX', 'ZMAX'),
                        help='Required right box workspace upper bound when --arm-workspace-layout per_arm --arm-workspace-mode box.')
    parser.add_argument('--left-arm-workspace-tapered', type=float, nargs=9, default=None,
                        metavar=('ZMIN', 'ZMAX', 'XMIN', 'XMAXLOW', 'XMAXHIGH', 'YMINLOW', 'YMINHIGH', 'YMAXLOW', 'YMAXHIGH'),
                        help='Required left tapered workspace when layout=per_arm: z_min z_max x_min x_max_low x_max_high y_min_low y_min_high y_max_low y_max_high.')
    parser.add_argument('--right-arm-workspace-tapered', type=float, nargs=9, default=None,
                        metavar=('ZMIN', 'ZMAX', 'XMIN', 'XMAXLOW', 'XMAXHIGH', 'YMINLOW', 'YMINHIGH', 'YMAXLOW', 'YMAXHIGH'),
                        help='Required right tapered workspace when layout=per_arm: z_min z_max x_min x_max_low x_max_high y_min_low y_min_high y_max_low y_max_high.')
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
    parser.add_argument('--ui', action='store_true',
                        help='Enable optional local web UI control server. Commands are queued into the existing teleop control loop. Also enables --headless to avoid starting the Rerun live viewer.')
    parser.add_argument('--ui-host', type=str, default='127.0.0.1',
                        help='Bind host for the optional web UI server.')
    parser.add_argument('--ui-port', type=int, default=8085,
                        help='Bind port for the optional web UI server.')
    parser.add_argument('--ui-preview-fps', type=float, default=5.0,
                        help='Maximum state publish rate for the optional web UI event stream.')
    parser.add_argument('--nero-console-provider', action='store_true',
                        help='Enable the Nero ROS Provider bridge. The Nero Web host runs separately.')
    parser.add_argument('--nero-online-replay-arm-source', type=str, choices=['action', 'state', 'fk_cmd_pose'], default='action',
                        help='Arm representation used by Nero online replay.')
    parser.add_argument('--nero-online-replay-base-source', type=str, choices=['none', 'action'], default='none',
                        help='Base representation used by Nero online replay. action requires explicit base-motion authorization at replay start.')
    # mode flags
    parser.add_argument('--motion', action='store_true',
                        help='Use the arm SDK DDS command route (rt/arm_sdk) for real-robot arms.')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--rerun-live', action='store_true',
                        help='Enable live Rerun logging and viewer for recorded episodes. Disabled by default.')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    parser.add_argument('--no-gripper', action='store_true',
                        help='Disable end-effector controller initialization and commands. Useful for arm-only offline replay.')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--record-arm-repr', type=str, choices=['qpos', 'pose', 'both'], default='qpos',
                        help='Recording representation for arm data: joint angles (qpos), wrist pose, or both.')
    parser.add_argument('--record-base', action='store_true',
                        help='Record mobile-base pose/velocity/action into raw episode JSON. Requires --record and Unitree DDS access to --base-odom-topic.')
    parser.add_argument('--record-base-velocity-only', action='store_true',
                        help='Record only base_link velocity and base commands. Omits odom world_pose and SLAM map pose from raw episodes.')
    parser.add_argument('--base-odom-topic', type=str, default='rt/agv/odom',
                        help='Unitree DDS odometry topic used for recorded base world pose and velocity.')
    parser.add_argument('--base-height-topic', type=str, default='rt/hispeed_state',
                        help='Unitree DDS Point32 topic used for recorded base height. Set empty string to disable height recording.')
    parser.add_argument('--record-slam-map-pose', action='store_true',
                        help='Record SLAM map pose from ROS2 /tf. Fails startup when slamware_map->--slam-pose-source-frame is unavailable.')
    parser.add_argument('--slam-pose-source-frame', type=str, default='laser',
                        help='TF source mode for map pose. Use odom to compose slamware_map->odom->base_link; other values record direct slamware_map->source under an explicit identity-extrinsic assumption.')
    parser.add_argument('--slam-chain-max-skew-ms', type=float, default=75.0,
                        help='Maximum TF header timestamp skew when --slam-pose-source-frame odom composes map->odom with odom->base_link.')
    parser.add_argument('--record-mobile-training-state', action='store_true',
                        help='Add base_link-local EEF poses plus calibrated column/waist state for the mobile-manipulation training schema.')
    parser.add_argument('--base-velocity-frame', choices=['base_link', 'world'], default=None,
                        help='Coordinate frame of rt/agv/odom vx/vy/wz. Required for --record-mobile-training-state.')
    parser.add_argument('--base-state-max-age-ms', type=float, default=131.578947,
                        help='Maximum base odometry timestamp delta from the camera sample in milliseconds (2.5 periods at 19 Hz).')
    parser.add_argument('--slam-tf-max-age-ms', type=float, default=125.0,
                        help='Maximum composed SLAM TF timestamp delta from the camera sample in milliseconds (2.5 periods at 20 Hz).')
    parser.add_argument('--base-history-size', type=int, default=512,
                        help='Maximum number of base state/action samples retained for timestamp alignment.')
    parser.add_argument('--base-startup-timeout-sec', type=float, default=3.0,
                        help='Seconds to wait for initial base odom/height messages when --record-base is enabled.')
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
    args = build_arg_parser().parse_args(argv)
    if (args.ui or args.nero_console_provider) and not args.rerun_live:
        args.headless = True
    if args.headless and args.rerun_live:
        raise ValueError('--rerun-live cannot be combined with --headless; use --ui --rerun-live to enable both explicitly')
    if args.home_position_tolerance_rad <= 0.0:
        raise ValueError('--home-position-tolerance-rad must be positive')
    if args.home_gravity_update_hz <= 0.0:
        raise ValueError('--home-gravity-update-hz must be positive')
    if args.home_gravity_torque_limit_nm <= 0.0:
        raise ValueError('--home-gravity-torque-limit-nm must be positive')
    if args.base_motion and args.base_controller == 'none':
        raise ValueError('--base-motion requires --base-controller loco or g1d_agv')
    if args.mobile_manipulation_mode == 'mobile_ik_qp':
        if args.arm != 'G1_29' or args.base_controller != 'g1d_agv' or not args.base_motion:
            raise ValueError('mobile_ik_qp requires --arm G1_29 --base-controller g1d_agv --base-motion')
        if args.mobile_state_timeout_sec <= 0.0 or args.mobile_column_travel_m <= 0.0:
            raise ValueError('mobile_ik_qp state timeout and column travel must be positive')
        if args.mobile_state_retry_count < 0 or args.mobile_state_retry_interval_sec <= 0.0:
            raise ValueError('mobile_ik_qp state retry count must be non-negative and retry interval must be positive')
        if args.mobile_height_raw_maximum <= args.mobile_height_raw_minimum:
            raise ValueError('mobile_ik_qp raw height limits must be ordered')
        if args.mobile_wbc_command_horizon_sec <= 0.0 or args.mobile_wbc_max_position_lead_rad <= 0.0:
            raise ValueError('mobile_ik_qp WBC position lookahead settings must be positive')
    if args.record_base and not args.record:
        raise ValueError("--record-base requires --record")
    if args.record_base and float(args.base_state_max_age_ms) <= 0.0:
        raise ValueError("--base-state-max-age-ms must be positive")
    if args.record_slam_map_pose and float(args.slam_tf_max_age_ms) <= 0.0:
        raise ValueError("--slam-tf-max-age-ms must be positive")
    if args.record_base and int(args.base_history_size) <= 0:
        raise ValueError("--base-history-size must be positive")
    if args.record_base and float(args.base_startup_timeout_sec) <= 0.0:
        raise ValueError("--base-startup-timeout-sec must be positive")
    if args.record_slam_map_pose and not args.record_base:
        raise ValueError("--record-slam-map-pose requires --record-base")
    if args.record_base_velocity_only and not args.record_base:
        raise ValueError("--record-base-velocity-only requires --record-base")
    if args.record_base_velocity_only and args.record_slam_map_pose:
        raise ValueError("--record-base-velocity-only cannot be combined with --record-slam-map-pose")
    if args.record_slam_map_pose and not str(args.slam_pose_source_frame or "").strip():
        raise ValueError("--slam-pose-source-frame must not be empty with --record-slam-map-pose")
    if args.record_slam_map_pose and float(args.slam_chain_max_skew_ms) <= 0.0:
        raise ValueError("--slam-chain-max-skew-ms must be positive")
    if args.record_mobile_training_state:
        if not args.record_base:
            raise ValueError("--record-mobile-training-state requires --record-base")
        if not args.record_slam_map_pose and not args.record_base_velocity_only:
            raise ValueError(
                "--record-mobile-training-state requires --record-slam-map-pose or --record-base-velocity-only"
            )
        if not str(args.base_height_topic or "").strip():
            raise ValueError("--record-mobile-training-state requires --base-height-topic")
        if args.base_velocity_frame is None:
            raise ValueError("--record-mobile-training-state requires --base-velocity-frame base_link or world")
        if args.record_base_velocity_only and args.base_velocity_frame != "base_link":
            raise ValueError("--record-base-velocity-only requires --base-velocity-frame base_link")
    if args.online_inference_protocol_profile == 'mobile_tcp23':
        if args.arm != 'G1_29' or args.ee != 'dex1' or args.no_gripper:
            raise ValueError("mobile_tcp23 requires --arm G1_29 --ee dex1 without --no-gripper")
        if args.online_inference_transport != 'http':
            raise ValueError("mobile_tcp23 requires --online-inference-transport http")
        if args.online_inference_arm_side != 'both':
            raise ValueError("mobile_tcp23 requires --online-inference-arm-side both")
        if args.mobile_manipulation_mode != 'direct_ik':
            raise ValueError("mobile_tcp23 requires --mobile-manipulation-mode direct_ik; model base actions must not pass through QP")
        if args.base_controller != 'g1d_agv' or not args.base_motion:
            raise ValueError("mobile_tcp23 requires --base-controller g1d_agv --base-motion")
        if args.base_command_source != 'provider':
            raise ValueError("mobile_tcp23 requires --base-command-source provider")
        if args.base_velocity_frame != 'base_link':
            raise ValueError("mobile_tcp23 requires --base-velocity-frame base_link")
    if args.online_inference_protocol_profile == 'mobile_pelvis_planar22':
        if args.arm != 'G1_29' or args.ee != 'dex1' or args.no_gripper:
            raise ValueError("mobile_pelvis_planar22 requires --arm G1_29 --ee dex1 without --no-gripper")
        if args.online_inference_transport != 'http':
            raise ValueError("mobile_pelvis_planar22 requires --online-inference-transport http")
        if args.online_inference_arm_side != 'both':
            raise ValueError("mobile_pelvis_planar22 requires --online-inference-arm-side both")
        if args.mobile_manipulation_mode != 'direct_ik':
            raise ValueError("mobile_pelvis_planar22 requires --mobile-manipulation-mode direct_ik; model base actions must not pass through QP")
        if args.base_controller != 'g1d_agv' or not args.base_motion:
            raise ValueError("mobile_pelvis_planar22 requires --base-controller g1d_agv --base-motion")
        if args.base_command_source != 'provider':
            raise ValueError("mobile_pelvis_planar22 requires --base-command-source provider")
        if args.base_velocity_frame != 'base_link':
            raise ValueError("mobile_pelvis_planar22 requires --base-velocity-frame base_link")
    if args.online_inference_protocol_profile == 'mobile_joint_base':
        if args.arm != 'G1_29' or args.ee != 'dex1' or args.no_gripper:
            raise ValueError("mobile_joint_base requires --arm G1_29 --ee dex1 without --no-gripper")
        if args.online_inference_transport != 'http':
            raise ValueError("mobile_joint_base requires --online-inference-transport http")
        if args.online_inference_arm_side != 'both':
            raise ValueError("mobile_joint_base requires --online-inference-arm-side both")
        if args.mobile_manipulation_mode == 'mobile_ik_qp':
            raise ValueError("mobile_joint_base cannot run with --mobile-manipulation-mode mobile_ik_qp")
        if args.online_inference_chunk_step_mode != 'per_tick':
            raise ValueError("mobile_joint_base requires --online-inference-chunk-step-mode per_tick")
        if args.base_controller != 'g1d_agv' or not args.base_motion:
            raise ValueError("mobile_joint_base requires --base-controller g1d_agv --base-motion")
        if args.base_command_source != 'provider':
            raise ValueError("mobile_joint_base requires --base-command-source provider")
        if args.base_velocity_frame != 'base_link':
            raise ValueError("mobile_joint_base requires --base-velocity-frame base_link")
    return args
