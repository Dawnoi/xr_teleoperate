#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate tv
fi

unset PYTHONPATH || true

RECORD_SLAM_MAP_POSE=0
for arg in "$@"; do
  case "${arg}" in
    --record-slam-map-pose)
      RECORD_SLAM_MAP_POSE=1
      ;;
    --mobile-manipulation-mode|--mobile-manipulation-mode=*)
      echo "[START] this script fixes --mobile-manipulation-mode direct_ik; use start_real_robot_wired_3cams_zmq.sh for mobile_ik_qp" >&2
      exit 2
      ;;
  esac
done
if [[ "${RECORD_SLAM_MAP_POSE}" == "1" ]]; then
  ROS_SETUP="/opt/ros/humble/setup.bash"
  if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "[START] --record-slam-map-pose requires ${ROS_SETUP}" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  set +u
  source "${ROS_SETUP}"
  set -u
fi

REAL_TELEOP_ENTRY="${REAL_TELEOP_ENTRY:-teleop/real/teleop_hand_and_arm.py}"

NETWORK_INTERFACE="${NETWORK_INTERFACE:-eno1}"
SENDER_IP="${SENDER_IP:-192.168.123.164}"

HEAD_ZMQ_PORT="${HEAD_ZMQ_PORT:-5556}"
LEFT_WRIST_ZMQ_PORT="${LEFT_WRIST_ZMQ_PORT:-5557}"
RIGHT_WRIST_ZMQ_PORT="${RIGHT_WRIST_ZMQ_PORT:-5558}"

TASK_DIR="${TASK_DIR:-./utils/data}"
TASK_NAME="${TASK_NAME:-multi_cam_record}"
RECORD_ARM_REPR="${RECORD_ARM_REPR:-both}"

cd "${REPO_DIR}"

echo "[START] real teleop entry: ${REAL_TELEOP_ENTRY}"
echo "[START] network interface: ${NETWORK_INTERFACE}"
echo "[START] motion mode: direct_ik; legacy shared workspace"

exec python "${REAL_TELEOP_ENTRY}" \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface "${NETWORK_INTERFACE}" \
  --base-controller g1d_agv \
  --mobile-manipulation-mode direct_ik \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --max-arm-joint-speed 5.0 \
  --arm-workspace-mode tapered \
  --arm-workspace-layout shared \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.45 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --base-max-vx 0.10 \
  --base-max-wz 0.35 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.10 \
  --record \
  --task-dir "${TASK_DIR}" \
  --task-name "${TASK_NAME}" \
  --record-arm-repr "${RECORD_ARM_REPR}" \
  --head-zmq-endpoint "tcp://${SENDER_IP}:${HEAD_ZMQ_PORT}" \
  --left-zmq-endpoint "tcp://${SENDER_IP}:${LEFT_WRIST_ZMQ_PORT}" \
  --right-zmq-endpoint "tcp://${SENDER_IP}:${RIGHT_WRIST_ZMQ_PORT}" \
  "$@"
