#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RAW_ROOT="${RAW_ROOT:-utils/data/multi_cam_record}"
OUTPUT_ROOT="${OUTPUT_ROOT:-utils/data/multi_cam_record_lerobot}"
TASK="${TASK:-pick up the pink octagonal prism with the right hand, hand it over to the left hand, and place it in the bowl}"
FPS="${FPS:-30}"
OVERWRITE="${OVERWRITE:-1}"

CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-tv}"
LOG_FILE="${LOG_FILE:-utils/data/export_raw_to_lerobot.log}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
FRAME_PROGRESS_EVERY="${FRAME_PROGRESS_EVERY:-100}"
STRICT_IMAGE_VALIDATE="${STRICT_IMAGE_VALIDATE:-0}"
VERIFY_EXPORT="${VERIFY_EXPORT:-1}"
VERIFY_VIDEO_FRAMES="${VERIFY_VIDEO_FRAMES:-0}"
EXPORT_FK="${EXPORT_FK:-0}"
URDF_PATH="${URDF_PATH:-assets/g1_d/g1_d.urdf}"

cd "${REPO_DIR}"

echo "[EXPORT_RAW_LEROBOT] raw_root=${RAW_ROOT}"
echo "[EXPORT_RAW_LEROBOT] output_root=${OUTPUT_ROOT}"
echo "[EXPORT_RAW_LEROBOT] task=${TASK}"
echo "[EXPORT_RAW_LEROBOT] fps=${FPS}"
echo "[EXPORT_RAW_LEROBOT] overwrite=${OVERWRITE}"
echo "[EXPORT_RAW_LEROBOT] progress_every=${PROGRESS_EVERY}"
echo "[EXPORT_RAW_LEROBOT] frame_progress_every=${FRAME_PROGRESS_EVERY}"
echo "[EXPORT_RAW_LEROBOT] strict_image_validate=${STRICT_IMAGE_VALIDATE}"
echo "[EXPORT_RAW_LEROBOT] verify_export=${VERIFY_EXPORT}"
echo "[EXPORT_RAW_LEROBOT] verify_video_frames=${VERIFY_VIDEO_FRAMES}"
echo "[EXPORT_RAW_LEROBOT] export_fk=${EXPORT_FK}"

if [[ ! -d "${RAW_ROOT}" ]]; then
  echo "[EXPORT_RAW_LEROBOT][ERROR] raw root not found: ${RAW_ROOT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"

ARGS=(
  --input-task-dir "${RAW_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --task "${TASK}"
  --fps "${FPS}"
  --progress-every "${PROGRESS_EVERY}"
  --frame-progress-every "${FRAME_PROGRESS_EVERY}"
  --strict-image-validate "${STRICT_IMAGE_VALIDATE}"
  --verify-export "${VERIFY_EXPORT}"
  --verify-video-frames "${VERIFY_VIDEO_FRAMES}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${EXPORT_FK}" == "1" ]]; then
  ARGS+=(--export-fk 1 --urdf-path "${URDF_PATH}")
fi

echo "[EXPORT_RAW_LEROBOT] writing log to ${LOG_FILE}"
env -u PYTHONPATH "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -u data_pipeline/export/raw_to_lerobot_v2.py \
  "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

echo "[EXPORT_RAW_LEROBOT] done"
echo "[EXPORT_RAW_LEROBOT] summary=${OUTPUT_ROOT}/export_summary.json"
