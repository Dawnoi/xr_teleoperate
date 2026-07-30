#!/usr/bin/env bash
set -euo pipefail

# One-command VIVE locator startup. Reuse libsurvive's saved room calibration
# by default; set VIVE_RECALIBRATE=1 only after moving a base station.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UNITREE_WS="${UNITREE_WS:-$(cd "${REPO_DIR}/../.." && pwd)}"
VIVE_LOCATOR_SRC="${VIVE_LOCATOR_SRC:-${UNITREE_WS}/src/vive_locator}"
WAIT_SEC="${VIVE_LOCATOR_WAIT_SEC:-5}"
VERBOSE_LEVEL="${VIVE_SURVIVE_VERBOSE:-5}"
CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME}/.config}"
SURVIVE_CACHE="${CONFIG_HOME}/libsurvive/config.json"

if [[ ! -f "${UNITREE_WS}/install/setup.bash" ]]; then
  echo "[VIVE] missing ${UNITREE_WS}/install/setup.bash; build vive_locator first." >&2
  exit 2
fi

if [[ "${VIVE_RECALIBRATE:-0}" == "1" ]]; then
  CALIBRATE_SCRIPT="${VIVE_LOCATOR_SRC}/scripts/recalibrate_after_move.bash"
  echo "[VIVE] forcing libsurvive recalibration..."
elif [[ ! -f "${SURVIVE_CACHE}" ]]; then
  CALIBRATE_SCRIPT="${VIVE_LOCATOR_SRC}/scripts/calibration_first_time.bash"
  echo "[VIVE] no libsurvive cache found; running first-time calibration..."
else
  CALIBRATE_SCRIPT=""
  echo "[VIVE] reusing libsurvive cache: ${SURVIVE_CACHE}"
fi

if [[ -n "${CALIBRATE_SCRIPT}" ]]; then
  if [[ ! -x "${CALIBRATE_SCRIPT}" ]]; then
    echo "[VIVE] calibration script not found or not executable: ${CALIBRATE_SCRIPT}" >&2
    exit 2
  fi
  bash "${CALIBRATE_SCRIPT}" --v "${VERBOSE_LEVEL}"
fi

echo "[VIVE] starting vive_locator in ${WAIT_SEC}s..."
sleep "${WAIT_SEC}"

# ament setup scripts intentionally read optional unset variables, which is
# incompatible with this wrapper's `set -u`.
set +u
source /opt/ros/humble/setup.bash
source "${UNITREE_WS}/install/setup.bash"
set -u

# Keep plugin discovery working even when the launch file is resolved through
# a symlinked install tree.
LOCATOR_PREFIX="${UNITREE_WS}/install/vive_locator"
export LD_LIBRARY_PATH="${LOCATOR_PREFIX}/lib:${LOCATOR_PREFIX}/lib/plugins${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export SURVIVE_PLUGINS="${LOCATOR_PREFIX}/lib/plugins${SURVIVE_PLUGINS:+:${SURVIVE_PLUGINS}}"

exec ros2 launch vive_locator vive_dual_locator.launch.py "$@"
