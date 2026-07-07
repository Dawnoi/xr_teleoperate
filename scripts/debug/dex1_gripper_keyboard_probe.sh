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

VLA_START_SCRIPT="${REPO_DIR}/scripts/start/start_real_robot_vla.sh"
DEFAULT_NETWORK_INTERFACE="$(
  sed -n -E 's/^NETWORK_INTERFACE="\$\{NETWORK_INTERFACE:-([^}]+)\}"$/\1/p' "${VLA_START_SCRIPT}" \
    | head -n 1
)"

if [[ -z "${DEFAULT_NETWORK_INTERFACE}" ]]; then
  echo "[DEX1_PROBE] failed to read default NETWORK_INTERFACE from ${VLA_START_SCRIPT}" >&2
  exit 1
fi

NETWORK_INTERFACE="${NETWORK_INTERFACE:-${DEFAULT_NETWORK_INTERFACE}}"

cd "${REPO_DIR}"

echo "[DEX1_PROBE] network interface: ${NETWORK_INTERFACE}"
echo "[DEX1_PROBE] VLA default source: ${VLA_START_SCRIPT}"

exec python teleop/debug/dex1_gripper_keyboard_probe.py \
  --network-interface "${NETWORK_INTERFACE}" \
  "$@"
