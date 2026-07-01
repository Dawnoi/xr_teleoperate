#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate tv
fi

unset PYTHONPATH || true

NETWORK_INTERFACE="${NETWORK_INTERFACE:-enx9c69d3212b05}"
SENDER_IP="${SENDER_IP:-192.168.123.164}"
XR_POSE_SOURCE="${XR_POSE_SOURCE:-controller}"
LEFT_MOTION_TRACKER_SN="${LEFT_MOTION_TRACKER_SN:-PC2310MLL2250062G}"
RIGHT_MOTION_TRACKER_SN="${RIGHT_MOTION_TRACKER_SN:-PC2310MLL2250124G}"
LEFT_MOTION_TRACKER_INDEX="${LEFT_MOTION_TRACKER_INDEX:-0}"
RIGHT_MOTION_TRACKER_INDEX="${RIGHT_MOTION_TRACKER_INDEX:-1}"
CONTROLLER_GRIP_THRESHOLD="${CONTROLLER_GRIP_THRESHOLD:-0.5}"
MOTION_TRACKER_MAX_STEP_LINEAR="${MOTION_TRACKER_MAX_STEP_LINEAR:-0.03}"
MOTION_TRACKER_MAX_STEP_ANGULAR="${MOTION_TRACKER_MAX_STEP_ANGULAR:-0.35}"
MAX_ARM_JOINT_SPEED="${MAX_ARM_JOINT_SPEED:-5.0}"
HOME_RETURN_SPEED="${HOME_RETURN_SPEED:-0.6}"

HEAD_ZMQ_PORT="${HEAD_ZMQ_PORT:-5556}"
LEFT_WRIST_ZMQ_PORT="${LEFT_WRIST_ZMQ_PORT:-5557}"
RIGHT_WRIST_ZMQ_PORT="${RIGHT_WRIST_ZMQ_PORT:-5558}"

TASK_DIR="${TASK_DIR:-./utils/data}"
TASK_NAME="${TASK_NAME:-multi_cam_record}"
RECORD_ARM_REPR="${RECORD_ARM_REPR:-both}"

cd "${REPO_DIR}"

exec python teleop/teleop_hand_and_arm.py \
  --input-provider xr \
  --input-mode controller \
  --xr-pose-source "${XR_POSE_SOURCE}" \
  --left-motion-tracker-sn "${LEFT_MOTION_TRACKER_SN}" \
  --right-motion-tracker-sn "${RIGHT_MOTION_TRACKER_SN}" \
  --left-motion-tracker-index "${LEFT_MOTION_TRACKER_INDEX}" \
  --right-motion-tracker-index "${RIGHT_MOTION_TRACKER_INDEX}" \
  --controller-grip-threshold "${CONTROLLER_GRIP_THRESHOLD}" \
  --motion-tracker-max-step-linear "${MOTION_TRACKER_MAX_STEP_LINEAR}" \
  --motion-tracker-max-step-angular "${MOTION_TRACKER_MAX_STEP_ANGULAR}" \
  --arm G1_29 \
  --ee dex1 \
  --network-interface "${NETWORK_INTERFACE}" \
  --base-controller g1d_agv \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --max-arm-joint-speed "${MAX_ARM_JOINT_SPEED}" \
  --home-return-speed "${HOME_RETURN_SPEED}" \
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.45 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
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
