#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET="${DATASET:-utils/data/multi_cam_record/transfer_black}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"
EPISODE_START="${EPISODE_START:-}"
EPISODE_END="${EPISODE_END:-}"
DECODE_IMAGES="${DECODE_IMAGES:-0}"
STATE_GRIPPER_REVIEW="${STATE_GRIPPER_REVIEW:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-tv}"

cd "${REPO_DIR}"

ARGS=(
  --dataset "${DATASET}"
  --decode-images "${DECODE_IMAGES}"
  --state-gripper-review "${STATE_GRIPPER_REVIEW}"
  --progress-interval "${PROGRESS_INTERVAL}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${OUTPUT_PREFIX}" ]]; then
  ARGS+=(--output-prefix "${OUTPUT_PREFIX}")
fi
if [[ -n "${EPISODE_START}" ]]; then
  ARGS+=(--episode-start "${EPISODE_START}")
fi
if [[ -n "${EPISODE_END}" ]]; then
  ARGS+=(--episode-end "${EPISODE_END}")
fi

echo "[AUDIT_MULTI_CAM] dataset=${DATASET}"
echo "[AUDIT_MULTI_CAM] decode_images=${DECODE_IMAGES}"
echo "[AUDIT_MULTI_CAM] state_gripper_review=${STATE_GRIPPER_REVIEW}"
echo "[AUDIT_MULTI_CAM] progress_interval=${PROGRESS_INTERVAL}"
echo "[AUDIT_MULTI_CAM] episode_start=${EPISODE_START:-all}"
echo "[AUDIT_MULTI_CAM] episode_end=${EPISODE_END:-all}"

env -u PYTHONPATH "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u data_pipeline/audit/multi_cam_record.py "${ARGS[@]}"
