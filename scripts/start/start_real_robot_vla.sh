#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-tv}"
fi

unset PYTHONPATH || true

if [[ "${VLA_SOURCE_ROS_HUMBLE:-0}" == "1" ]]; then
  ROS_SETUP="/opt/ros/humble/setup.bash"
  if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "[START_VLA] VLA_SOURCE_ROS_HUMBLE=1 requires ${ROS_SETUP}." >&2
    exit 1
  fi
  # ROS Humble setup references unset variables on this host.
  set +u
  # shellcheck disable=SC1091
  source "${ROS_SETUP}"
  set -u
fi

REAL_TELEOP_ENTRY="${REAL_TELEOP_ENTRY:-teleop/real/teleop_hand_and_arm.py}"

NETWORK_INTERFACE="${NETWORK_INTERFACE:-enx9c69d3212b05}"
SENDER_IP="${SENDER_IP:-192.168.123.164}"

HEAD_ZMQ_PORT="${HEAD_ZMQ_PORT:-5556}"
LEFT_WRIST_ZMQ_PORT="${LEFT_WRIST_ZMQ_PORT:-5557}"
RIGHT_WRIST_ZMQ_PORT="${RIGHT_WRIST_ZMQ_PORT:-5558}"

VLA_TRANSPORT="${VLA_TRANSPORT:-http}"
VLA_BASE_URL="${VLA_BASE_URL:-http://127.0.0.1:18027}"
VLA_PROTOCOL_PROFILE="${VLA_PROTOCOL_PROFILE:-pi05_dual_arm_20d}"
VLA_PROMPT="${VLA_PROMPT:-pick up the pink octagonal prism with the right hand, hand it over to the left hand, and place it in the bowl}"
VLA_ARM_SIDE="${VLA_ARM_SIDE:-both}"
VLA_ENABLE_MOTION="${VLA_ENABLE_MOTION:-1}"
VLA_AUTO_START="${VLA_AUTO_START:-0}"
ARM_CONTROL_HZ="${ARM_CONTROL_HZ:-250}"

if [[ -z "${VLA_TRANSFORM_CONFIG:-}" ]]; then
  if [[ "${VLA_ARM_SIDE}" == "right" ]]; then
    VLA_TRANSFORM_CONFIG="configs/inference/unitree_right_arm_identity_transform.json"
  else
    VLA_TRANSFORM_CONFIG="configs/inference/unitree_dual_arm_identity_transform.json"
  fi
fi

MOTION_ARGS=(--online-inference-dry-run)
if [[ "${VLA_ENABLE_MOTION}" == "1" ]]; then
  MOTION_ARGS=(--online-inference-enable-motion)
fi

AUTO_START_ARGS=()
if [[ "${VLA_AUTO_START}" == "1" ]]; then
  AUTO_START_ARGS=(--auto-start)
fi

cd "${REPO_DIR}"

echo "[START_VLA] real teleop entry: ${REAL_TELEOP_ENTRY}"
echo "[START_VLA] network interface: ${NETWORK_INTERFACE}"
echo "[START_VLA] sender ip: ${SENDER_IP}"
echo "[START_VLA] base url: ${VLA_BASE_URL}"
echo "[START_VLA] protocol profile: ${VLA_PROTOCOL_PROFILE}"
echo "[START_VLA] arm side: ${VLA_ARM_SIDE}"
echo "[START_VLA] transform config: ${VLA_TRANSFORM_CONFIG}"
echo "[START_VLA] enable motion: ${VLA_ENABLE_MOTION} (0=dry-run, 1=real motion)"
echo "[START_VLA] dex1 adaptive force-hold: enabled for online_inference"
echo "[START_VLA] auto start: ${VLA_AUTO_START} (0=press r, 1=start immediately)"
echo "[START_VLA] arm control hz: ${ARM_CONTROL_HZ}"

exec python "${REAL_TELEOP_ENTRY}" \
  --input-provider online_inference \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface "${NETWORK_INTERFACE}" \
  --base-controller g1d_agv \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --arm-control-hz "${ARM_CONTROL_HZ}" \
  --max-arm-joint-speed "${MAX_ARM_JOINT_SPEED:-1.0}" \
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.10 \
  --head-zmq-endpoint "tcp://${SENDER_IP}:${HEAD_ZMQ_PORT}" \
  --left-zmq-endpoint "tcp://${SENDER_IP}:${LEFT_WRIST_ZMQ_PORT}" \
  --right-zmq-endpoint "tcp://${SENDER_IP}:${RIGHT_WRIST_ZMQ_PORT}" \
  --online-inference-transport "${VLA_TRANSPORT}" \
  --online-inference-base-url "${VLA_BASE_URL}" \
  --online-inference-protocol-profile "${VLA_PROTOCOL_PROFILE}" \
  --online-inference-prompt "${VLA_PROMPT}" \
  --online-inference-arm-side "${VLA_ARM_SIDE}" \
  --online-inference-transform-config "${VLA_TRANSFORM_CONFIG}" \
  "${MOTION_ARGS[@]}" \
  "${AUTO_START_ARGS[@]}" \
  "$@"
