#!/usr/bin/env python3
"""Build an episode-level training task from persisted validation results.

The raw task is never modified. Every accepted episode is represented by a
directory symlink, so images and raw metadata are never copied. Episodes whose
UI-equivalent validation level is ``error`` are omitted as a whole.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.ui.episode_store import load_persisted_validation


DEFAULT_MAX_SAMPLE_INTERVAL_MS = 250.0
DEFAULT_MAX_LONG_GAP_FRACTION = 0.01
DEFAULT_MIN_DURATION_SEC = 30.0
DEFAULT_MAX_DURATION_SEC = 60.0
DEFAULT_MAX_EPISODE: int | None = None


def _finite_nonnegative(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and numeric >= 0.0 else None


def _time_distribution_metrics(report: dict[str, Any]) -> dict[str, float] | None:
    time_alignment = report.get("time_alignment")
    sample_interval = time_alignment.get("sample_interval") if isinstance(time_alignment, dict) else None
    if not isinstance(sample_interval, dict):
        return None
    duration_sec = _finite_nonnegative(report.get("duration_sec"))
    interval_count_value = sample_interval.get("interval_count")
    long_gap_count_value = sample_interval.get("long_gap_count")
    interval_max_ns = _finite_nonnegative(sample_interval.get("interval_max_ns"))
    if (
        duration_sec is None
        or not isinstance(interval_count_value, int)
        or isinstance(interval_count_value, bool)
        or interval_count_value <= 0
        or not isinstance(long_gap_count_value, int)
        or isinstance(long_gap_count_value, bool)
        or long_gap_count_value < 0
        or interval_max_ns is None
    ):
        return None
    return {
        "duration_sec": duration_sec,
        "interval_count": float(interval_count_value),
        "long_gap_count": float(long_gap_count_value),
        "long_gap_fraction": float(long_gap_count_value) / float(interval_count_value),
        "max_sample_interval_ms": interval_max_ns / 1e6,
    }


def _duration_distribution(
    records: list[dict[str, Any]],
) -> dict[str, float]:
    durations = [
        float(record["time_distribution"]["duration_sec"])
        for record in records
        if record["validation_level"] != "error" and record["time_distribution"] is not None
    ]
    if len(durations) < 4:
        raise RuntimeError("training_quality requires at least four validation-nonerror episodes with time-distribution metrics")
    mean = statistics.mean(durations)
    std = statistics.pstdev(durations)
    return {
        "reference_episode_count": float(len(durations)),
        "mean_duration_sec": mean,
        "std_duration_sec": std,
        "min_duration_sec": min(durations),
        "max_duration_sec": max(durations),
    }


def _training_quality_reasons(
    record: dict[str, Any],
    *,
    max_sample_interval_ms: float,
    max_long_gap_fraction: float,
    min_duration_sec: float,
    max_duration_sec: float,
) -> list[str]:
    if record["validation_level"] == "error":
        return ["validation_level_error"]
    metrics = record["time_distribution"]
    if metrics is None:
        return ["time_distribution_metadata_missing"]
    reasons = []
    if metrics["max_sample_interval_ms"] > max_sample_interval_ms:
        reasons.append("max_sample_interval_exceeded")
    if metrics["long_gap_fraction"] >= max_long_gap_fraction:
        reasons.append("long_gap_fraction_exceeded")
    if metrics["duration_sec"] < min_duration_sec:
        reasons.append("episode_duration_too_short")
    if metrics["duration_sec"] > max_duration_sec:
        reasons.append("episode_duration_too_long")
    return reasons


def _validation_record(episode_dir: Path) -> dict[str, Any]:
    report = load_persisted_validation(episode_dir)
    level = report.get("level")
    if not isinstance(level, str) or level not in {"ok", "warning", "error"}:
        raise ValueError(
            f"episode={episode_dir} validation.json has invalid level={level!r}; "
            "expected one of ok, warning, error"
        )
    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    if not isinstance(errors, list) or not isinstance(warnings, list):
        raise ValueError(f"episode={episode_dir} validation errors and warnings must be lists")
    return {
        "episode": episode_dir.name,
        "source_episode_dir": str(episode_dir.resolve()),
        "validation_level": level,
        "errors": errors,
        "warnings": warnings,
        "time_distribution": _time_distribution_metrics(report),
    }


def clean_task_dir(
    task_dir: Path,
    output_task_dir: Path,
    *,
    selection_mode: str = "validation_error",
    excluded_episode_names: set[str] | None = None,
    overwrite: bool = False,
    max_sample_interval_ms: float = DEFAULT_MAX_SAMPLE_INTERVAL_MS,
    max_long_gap_fraction: float = DEFAULT_MAX_LONG_GAP_FRACTION,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    max_episode: int | None = DEFAULT_MAX_EPISODE,
) -> dict[str, Any]:
    task_dir = task_dir.resolve()
    output_task_dir = output_task_dir.resolve()
    if not task_dir.is_dir():
        raise ValueError(f"--task-dir is not a directory: {task_dir}")
    if task_dir == output_task_dir:
        raise ValueError("--output-task-dir must differ from --task-dir")
    if selection_mode not in {"validation_error", "training_quality", "explicit"}:
        raise ValueError(f"unsupported selection_mode={selection_mode!r}")
    for name, value, inclusive in (
        ("max_sample_interval_ms", max_sample_interval_ms, False),
        ("max_long_gap_fraction", max_long_gap_fraction, False),
        ("min_duration_sec", min_duration_sec, False),
        ("max_duration_sec", max_duration_sec, False),
    ):
        numeric = _finite_nonnegative(value)
        if numeric is None or (not inclusive and numeric <= 0.0):
            comparison = "non-negative" if inclusive else "positive"
            raise ValueError(f"{name} must be finite and {comparison}")
    if float(max_duration_sec) < float(min_duration_sec):
        raise ValueError("max_duration_sec must be greater than or equal to min_duration_sec")
    if max_episode is not None and (
        isinstance(max_episode, bool) or not isinstance(max_episode, int) or max_episode <= 0
    ):
        raise ValueError("max_episode must be a positive integer or None")
    excluded_episode_names = set(excluded_episode_names or ())
    if selection_mode == "explicit" and not excluded_episode_names:
        raise ValueError("--selection-mode explicit requires at least one --exclude-episode")

    episode_dirs = sorted(
        path
        for path in task_dir.glob("episode_*")
        if (path / "data.json").is_file()
        and (
            max_episode is None
            or path.name.removeprefix("episode_").isdigit()
            and int(path.name.removeprefix("episode_")) <= max_episode
        )
    )
    if not episode_dirs:
        raise RuntimeError(f"no episode_xxxx/data.json found under: {task_dir}")
    if output_task_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"output task directory already exists: {output_task_dir}; rerun with --overwrite to replace it"
            )
        shutil.rmtree(output_task_dir)

    output_task_dir.mkdir(parents=True)
    records = [_validation_record(episode_dir) for episode_dir in episode_dirs]
    duration_distribution = (
        _duration_distribution(records)
        if selection_mode == "training_quality"
        else None
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for episode_dir, record in zip(episode_dirs, records):
        explicit_rejection = episode_dir.name in excluded_episode_names
        validation_rejection = selection_mode == "validation_error" and record["validation_level"] == "error"
        training_quality_reasons = (
            _training_quality_reasons(
                record,
                max_sample_interval_ms=float(max_sample_interval_ms),
                max_long_gap_fraction=float(max_long_gap_fraction),
                min_duration_sec=float(min_duration_sec),
                max_duration_sec=float(max_duration_sec),
            )
            if selection_mode == "training_quality"
            else []
        )
        if explicit_rejection or validation_rejection or training_quality_reasons:
            rejection_reasons = ["explicit_exclusion"] if explicit_rejection else training_quality_reasons or ["validation_level_error"]
            record["rejection_reason"] = rejection_reasons[0]
            record["rejection_reasons"] = rejection_reasons
            rejected.append(record)
            continue
        target = output_task_dir / episode_dir.name
        relative_source = os.path.relpath(episode_dir.resolve(), start=output_task_dir)
        target.symlink_to(relative_source, target_is_directory=True)
        accepted.append(record)

    report = {
        "schema_version": 1,
        "source_task_dir": str(task_dir),
        "output_task_dir": str(output_task_dir),
        "selection_mode": selection_mode,
        "max_episode": max_episode,
        "explicit_excluded_episode_names": sorted(excluded_episode_names),
        "training_quality": (
            {
                "max_sample_interval_ms": float(max_sample_interval_ms),
                "max_long_gap_fraction": float(max_long_gap_fraction),
                "min_duration_sec": float(min_duration_sec),
                "max_duration_sec": float(max_duration_sec),
                "duration_distribution": duration_distribution,
            }
            if selection_mode == "training_quality"
            else None
        ),
        "source_episode_count": len(episode_dirs),
        "accepted_episode_count": len(accepted),
        "rejected_episode_count": len(rejected),
        "accepted_episodes": accepted,
        "rejected_episodes": rejected,
    }
    report_path = output_task_dir / "clean_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[CLEAN_EPISODES] source={len(episode_dirs)} accepted={len(accepted)} "
        f"rejected={len(rejected)} output={output_task_dir}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True, help="包含 raw episode_xxxx 的任务目录")
    parser.add_argument("--output-task-dir", type=Path, required=True, help="输出的软链接清洗任务目录")
    parser.add_argument(
        "--selection-mode",
        choices=("validation_error", "training_quality", "explicit"),
        default="validation_error",
        help="validation_error 排除 validation error；training_quality 额外排除时间分布异常；explicit 只排除 --exclude-episode 指定集",
    )
    parser.add_argument("--max-sample-interval-ms", type=float, default=DEFAULT_MAX_SAMPLE_INTERVAL_MS)
    parser.add_argument("--max-long-gap-fraction", type=float, default=DEFAULT_MAX_LONG_GAP_FRACTION)
    parser.add_argument("--min-duration-sec", type=float, default=DEFAULT_MIN_DURATION_SEC)
    parser.add_argument("--max-duration-sec", type=float, default=DEFAULT_MAX_DURATION_SEC)
    parser.add_argument(
        "--max-episode",
        type=int,
        default=DEFAULT_MAX_EPISODE,
        help="只处理 episode 编号不大于该值的样本（默认处理全部 episode）",
    )
    parser.add_argument(
        "--exclude-episode",
        action="append",
        default=[],
        help="要排除的 episode 名称；可重复传入，仅 explicit 模式使用",
    )
    parser.add_argument("--overwrite", action="store_true", help="明确允许替换已存在的输出任务目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_task_dir(
        args.task_dir,
        args.output_task_dir,
        selection_mode=args.selection_mode,
        excluded_episode_names=set(args.exclude_episode),
        overwrite=args.overwrite,
        max_sample_interval_ms=args.max_sample_interval_ms,
        max_long_gap_fraction=args.max_long_gap_fraction,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        max_episode=args.max_episode,
    )


if __name__ == "__main__":
    main()
