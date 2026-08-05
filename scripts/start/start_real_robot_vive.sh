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

REAL_TELEOP_ENTRY="${REAL_TELEOP_ENTRY:-teleop/real/teleop_hand_and_arm.py}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-eno1}"
VIVE_CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME}/.config}/xr_teleoperate"
VIVE_CALIBRATION_FILE="${VIVE_CALIBRATION_FILE:-${VIVE_CONFIG_HOME}/vive_calibration.json}"
VIVE_ENABLE_LEFT_TOPIC="${VIVE_ENABLE_LEFT_TOPIC:-/vive/enable_left}"
VIVE_ENABLE_RIGHT_TOPIC="${VIVE_ENABLE_RIGHT_TOPIC:-/vive/enable_right}"

cd "${REPO_DIR}"

if [[ ! -f "${VIVE_CALIBRATION_FILE}" ]]; then
  echo "[START_VIVE] missing calibration file: ${VIVE_CALIBRATION_FILE}" >&2
  echo "[START_VIVE] run: python scripts/vive_axis_calibrator.py --output-file ${VIVE_CALIBRATION_FILE}" >&2
  exit 2
fi

echo "[START_VIVE] real teleop entry: ${REAL_TELEOP_ENTRY}"
echo "[START_VIVE] network interface: ${NETWORK_INTERFACE}"
echo "[START_VIVE] calibration file: ${VIVE_CALIBRATION_FILE}"
echo "[START_VIVE] pedal deadman required in another terminal: python scripts/vive_keyboard_enable.py --grab-input-devices"
echo "[START_VIVE] add --swap-sides if the physical left/right pedals are reversed"

exec python "${REAL_TELEOP_ENTRY}" \
  --input-provider vive \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface "${NETWORK_INTERFACE}" \
  --base-controller g1d_agv \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --vive-calibration-file "${VIVE_CALIBRATION_FILE}" \
  --vive-enable-left-topic "${VIVE_ENABLE_LEFT_TOPIC}" \
  --vive-enable-right-topic "${VIVE_ENABLE_RIGHT_TOPIC}" \
  --max-arm-joint-speed "${MAX_ARM_JOINT_SPEED:-1.0}" \
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
  "$@"
