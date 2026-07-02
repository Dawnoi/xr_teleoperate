#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LEROBOT_ROOT="${LEROBOT_ROOT:-utils/data/multi_cam_record_lerobot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-utils/data/multi_cam_record_umi_dp}"
TCP_SOURCE="${TCP_SOURCE:-cmd}"
GRIPPER_SOURCE="${GRIPPER_SOURCE:-action}"
TIMESTAMP_SOURCE="${TIMESTAMP_SOURCE:-sample}"
TIMEZONE="${TIMEZONE:-local}"
OVERWRITE="${OVERWRITE:-1}"
EPISODES="${EPISODES:-}"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-tv}"
LOG_FILE="${LOG_FILE:-utils/data/export_lerobot_to_umi_dp.log}"

cd "${REPO_DIR}"

echo "[EXPORT_UMI_DP] lerobot_root=${LEROBOT_ROOT}"
echo "[EXPORT_UMI_DP] output_root=${OUTPUT_ROOT}"
echo "[EXPORT_UMI_DP] tcp_source=${TCP_SOURCE}"
echo "[EXPORT_UMI_DP] gripper_source=${GRIPPER_SOURCE}"
echo "[EXPORT_UMI_DP] timestamp_source=${TIMESTAMP_SOURCE}"
echo "[EXPORT_UMI_DP] timezone=${TIMEZONE}"
echo "[EXPORT_UMI_DP] overwrite=${OVERWRITE}"
echo "[EXPORT_UMI_DP] episodes=${EPISODES:-<all>}"

mkdir -p "$(dirname "${LOG_FILE}")"

ARGS=(
  --lerobot-root "${LEROBOT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --tcp-source "${TCP_SOURCE}"
  --gripper-source "${GRIPPER_SOURCE}"
  --timestamp-source "${TIMESTAMP_SOURCE}"
  --timezone "${TIMEZONE}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

if [[ -n "${EPISODES}" ]]; then
  # Supports both "0 1 2" and "0,1,2" forms accepted by the Python exporter.
  # shellcheck disable=SC2206
  EPISODE_ARGS=(${EPISODES})
  ARGS+=(--episodes "${EPISODE_ARGS[@]}")
fi

echo "[EXPORT_UMI_DP] writing log to ${LOG_FILE}"
env -u PYTHONPATH "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -u data_pipeline/export/lerobot_to_umi_dp.py \
  "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

echo "[EXPORT_UMI_DP] done"
