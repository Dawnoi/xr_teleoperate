#!/usr/bin/env python3
"""Build an episode-level training task from persisted validation results.

The raw task is never modified. Every accepted episode is represented by a
directory symlink, so images and raw metadata are never copied. Episodes whose
UI-equivalent validation level is ``error`` are omitted as a whole.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.ui.episode_store import load_persisted_validation


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
    }


def clean_task_dir(
    task_dir: Path,
    output_task_dir: Path,
    *,
    selection_mode: str = "validation_error",
    excluded_episode_names: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    task_dir = task_dir.resolve()
    output_task_dir = output_task_dir.resolve()
    if not task_dir.is_dir():
        raise ValueError(f"--task-dir is not a directory: {task_dir}")
    if task_dir == output_task_dir:
        raise ValueError("--output-task-dir must differ from --task-dir")
    if selection_mode not in {"validation_error", "explicit"}:
        raise ValueError(f"unsupported selection_mode={selection_mode!r}")
    excluded_episode_names = set(excluded_episode_names or ())
    if selection_mode == "explicit" and not excluded_episode_names:
        raise ValueError("--selection-mode explicit requires at least one --exclude-episode")

    episode_dirs = sorted(path for path in task_dir.glob("episode_*") if (path / "data.json").is_file())
    if not episode_dirs:
        raise RuntimeError(f"no episode_xxxx/data.json found under: {task_dir}")
    if output_task_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"output task directory already exists: {output_task_dir}; rerun with --overwrite to replace it"
            )
        shutil.rmtree(output_task_dir)

    output_task_dir.mkdir(parents=True)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        record = _validation_record(episode_dir)
        explicit_rejection = episode_dir.name in excluded_episode_names
        validation_rejection = selection_mode == "validation_error" and record["validation_level"] == "error"
        if explicit_rejection or validation_rejection:
            record["rejection_reason"] = "explicit_exclusion" if explicit_rejection else "validation_level_error"
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
        "explicit_excluded_episode_names": sorted(excluded_episode_names),
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
        choices=("validation_error", "explicit"),
        default="validation_error",
        help="validation_error 排除全部 validation error；explicit 只排除 --exclude-episode 指定集",
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
    )


if __name__ == "__main__":
    main()
