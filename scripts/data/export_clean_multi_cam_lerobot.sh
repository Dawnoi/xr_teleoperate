#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-utils/data/multi_cam_record}"
PROBLEM_FILE="${PROBLEM_FILE:-$RAW_ROOT/problem_episodes.txt}"
VIEW_ROOT="${VIEW_ROOT:-utils/data/multi_cam_record_clean_view}"
OUTPUT_ROOT="${OUTPUT_ROOT:-utils/data/multi_cam_record_lerobot_clean}"
TASK="${TASK:-pick up the pink octagonal prism with the right hand, hand it over to the left hand, and place it in the bowl}"
FPS="${FPS:-30}"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-tv}"
LOG_FILE="${LOG_FILE:-utils/data/export_clean_multi_cam_lerobot.log}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
FRAME_PROGRESS_EVERY="${FRAME_PROGRESS_EVERY:-100}"
STRICT_IMAGE_VALIDATE="${STRICT_IMAGE_VALIDATE:-0}"
VERIFY_EXPORT="${VERIFY_EXPORT:-1}"
VERIFY_VIDEO_FRAMES="${VERIFY_VIDEO_FRAMES:-0}"

cd "$(dirname "$0")/../.."

echo "[EXPORT_CLEAN] raw_root=$RAW_ROOT"
echo "[EXPORT_CLEAN] problem_file=$PROBLEM_FILE"
echo "[EXPORT_CLEAN] view_root=$VIEW_ROOT"
echo "[EXPORT_CLEAN] output_root=$OUTPUT_ROOT"
echo "[EXPORT_CLEAN] task=$TASK"
echo "[EXPORT_CLEAN] fps=$FPS"
echo "[EXPORT_CLEAN] progress_every=$PROGRESS_EVERY"
echo "[EXPORT_CLEAN] frame_progress_every=$FRAME_PROGRESS_EVERY"
echo "[EXPORT_CLEAN] strict_image_validate=$STRICT_IMAGE_VALIDATE"
echo "[EXPORT_CLEAN] verify_export=$VERIFY_EXPORT"
echo "[EXPORT_CLEAN] verify_video_frames=$VERIFY_VIDEO_FRAMES"

python3 - "$RAW_ROOT" "$PROBLEM_FILE" "$VIEW_ROOT" "$OUTPUT_ROOT" <<'PY'
import shutil
import sys
from pathlib import Path

raw_root = Path(sys.argv[1])
problem_file = Path(sys.argv[2])
view_root = Path(sys.argv[3])
output_root = Path(sys.argv[4])

if not raw_root.is_dir():
    raise FileNotFoundError(f"raw root not found: {raw_root}")
if not problem_file.is_file():
    raise FileNotFoundError(f"problem list not found: {problem_file}")
if output_root.resolve() == raw_root.resolve():
    raise ValueError("output root must be different from raw root")

exclude = set()
in_dedup = False
for line in problem_file.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped.startswith("ALL_PROBLEM_OR_REVIEW_EPISODES_DEDUP"):
        in_dedup = True
        continue
    if in_dedup and stripped.startswith("["):
        break
    if in_dedup and stripped.startswith("episode_"):
        exclude.add(stripped.split()[0])

if not exclude:
    raise RuntimeError(f"no excluded episodes parsed from {problem_file}")

if view_root.exists():
    if view_root.is_symlink() or view_root.is_file():
        view_root.unlink()
    else:
        shutil.rmtree(view_root)
view_root.mkdir(parents=True)

source_episodes = sorted(path for path in raw_root.glob("episode_*") if path.is_dir())
if not source_episodes:
    raise FileNotFoundError(f"no episode_* directories found under {raw_root}")

included = []
for src in source_episodes:
    if src.name in exclude:
        continue
    dst = view_root / src.name
    dst.symlink_to(src.resolve(), target_is_directory=True)
    included.append(src.name)

manifest = view_root / "clean_view_manifest.txt"
manifest.write_text(
    "source_root=" + str(raw_root.resolve()) + "\n"
    "problem_file=" + str(problem_file.resolve()) + "\n"
    "output_root=" + str(output_root) + "\n"
    "source_count=" + str(len(source_episodes)) + "\n"
    "excluded_count=" + str(len(exclude)) + "\n"
    "included_count=" + str(len(included)) + "\n"
    "\n[EXCLUDED]\n" + "\n".join(sorted(exclude)) + "\n"
    "\n[INCLUDED]\n" + "\n".join(included) + "\n",
    encoding="utf-8",
)

print(f"[EXPORT_CLEAN] source_count={len(source_episodes)}")
print(f"[EXPORT_CLEAN] excluded_count={len(exclude)}")
print(f"[EXPORT_CLEAN] included_count={len(included)}")
print(f"[EXPORT_CLEAN] manifest={manifest}")
PY

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$(dirname "$LOG_FILE")"
echo "[EXPORT_CLEAN] writing log to $LOG_FILE"

"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -u data_pipeline/export/raw_to_lerobot_v2.py \
  --input-task-dir "$VIEW_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --task "$TASK" \
  --fps "$FPS" \
  --overwrite \
  --progress-every "$PROGRESS_EVERY" \
  --frame-progress-every "$FRAME_PROGRESS_EVERY" \
  --strict-image-validate "$STRICT_IMAGE_VALIDATE" \
  --verify-export "$VERIFY_EXPORT" \
  --verify-video-frames "$VERIFY_VIDEO_FRAMES" 2>&1 | tee "$LOG_FILE"

echo "[EXPORT_CLEAN] done"
echo "[EXPORT_CLEAN] summary=$OUTPUT_ROOT/export_summary.json"
