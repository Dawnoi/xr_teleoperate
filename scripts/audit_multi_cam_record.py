#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_CAMERAS = ("head", "left_wrist", "right_wrist")
ARM_GROUPS = ("left_arm", "right_arm")
EE_GROUPS = ("left_ee", "right_ee")


@dataclass(frozen=True)
class Issue:
    episode: str
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class EpisodeSummary:
    episode: str
    frame_count: int | None
    duration_s: float | None
    max_gap_ms: float | None
    json_path: Path


def percentile(sorted_values: list[int], pct: float) -> float | None:
    if not sorted_values:
        return None
    if pct <= 0:
        return float(sorted_values[0])
    if pct >= 100:
        return float(sorted_values[-1])
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    alpha = pos - lo
    return float(sorted_values[lo] * (1.0 - alpha) + sorted_values[hi] * alpha)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def finite_vector(value: Any, expected_len: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected_len
        and all(finite_number(item) for item in value)
    )


def append_issue(issues: list[Issue], episode: str, severity: str, code: str, message: str) -> None:
    issues.append(Issue(episode=episode, severity=severity, code=code, message=message))


def json_tool_error(data_json: Path) -> str | None:
    result = subprocess.run(
        [sys.executable, "-m", "json.tool", str(data_json)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    stderr = result.stderr.strip()
    if not stderr:
        return "json.tool returned non-zero without stderr"
    return stderr.splitlines()[-1]


def load_payload(data_json: Path) -> dict[str, Any]:
    payload = json.loads(data_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"top-level JSON must be an object: {data_json}")
    return payload


def validate_qpos(
    issues: list[Issue],
    episode: str,
    item: dict[str, Any],
    frame_index: int,
    source_name: str,
) -> None:
    source = item.get(source_name)
    if not isinstance(source, dict):
        append_issue(issues, episode, "FAIL", "FAIL_MOTION_SCHEMA", f"frame {frame_index} missing {source_name}")
        return

    for group_name in ARM_GROUPS:
        group = source.get(group_name)
        if not isinstance(group, dict):
            append_issue(
                issues,
                episode,
                "FAIL",
                "FAIL_QPOS",
                f"frame {frame_index} missing {source_name}.{group_name}",
            )
            continue
        qpos = group.get("qpos")
        if not finite_vector(qpos, 7):
            actual_len = len(qpos) if isinstance(qpos, list) else "not-list"
            append_issue(
                issues,
                episode,
                "FAIL",
                "FAIL_QPOS",
                f"frame {frame_index} {source_name}.{group_name}.qpos len={actual_len}, expected=7, finite={isinstance(qpos, list) and all(finite_number(v) for v in qpos)}",
            )

    for group_name in EE_GROUPS:
        group = source.get(group_name)
        if not isinstance(group, dict):
            append_issue(
                issues,
                episode,
                "FAIL",
                "FAIL_QPOS",
                f"frame {frame_index} missing {source_name}.{group_name}",
            )
            continue
        qpos = group.get("qpos")
        if not finite_vector(qpos, 1):
            actual_len = len(qpos) if isinstance(qpos, list) else "not-list"
            append_issue(
                issues,
                episode,
                "FAIL",
                "FAIL_QPOS",
                f"frame {frame_index} {source_name}.{group_name}.qpos len={actual_len}, expected=1, finite={isinstance(qpos, list) and all(finite_number(v) for v in qpos)}",
            )


def right_gripper_value(
    issues: list[Issue],
    episode: str,
    item: dict[str, Any],
    frame_index: int,
) -> float | None:
    source = item.get("actions")
    if not isinstance(source, dict):
        append_issue(issues, episode, "FAIL", "FAIL_MOTION_SCHEMA", f"frame {frame_index} missing actions")
        return None
    group = source.get("right_ee")
    if not isinstance(group, dict):
        append_issue(issues, episode, "FAIL", "FAIL_QPOS", f"frame {frame_index} missing actions.right_ee")
        return None
    qpos = group.get("qpos")
    if not finite_vector(qpos, 1):
        return None
    return float(qpos[0])


def validate_right_gripper_event(
    issues: list[Issue],
    episode: str,
    values: list[float],
    open_threshold: float,
    close_threshold: float,
    min_close_run: int,
    min_motion_range: float,
) -> None:
    if not values:
        append_issue(issues, episode, "FAIL", "FAIL_RIGHT_GRIPPER_EMPTY", "actions.right_ee.qpos has no valid values")
        return

    max_value = max(values)
    min_value = min(values)
    motion_range = max_value - min_value

    if max_value < open_threshold:
        append_issue(
            issues,
            episode,
            "FAIL",
            "FAIL_RIGHT_GRIPPER_NO_OPEN_STATE",
            f"max={max_value:.6f} < open_threshold={open_threshold:.6f}",
        )

    if motion_range < min_motion_range:
        append_issue(
            issues,
            episode,
            "FAIL",
            "FAIL_RIGHT_GRIPPER_MOTION_RANGE_TOO_SMALL",
            f"range={motion_range:.6f} < min_motion_range={min_motion_range:.6f}; min={min_value:.6f}, max={max_value:.6f}",
        )

    longest_run = 0
    longest_start = None
    current_run = 0
    current_start = None
    for frame_index, value in enumerate(values):
        if value < close_threshold:
            if current_run == 0:
                current_start = frame_index
            current_run += 1
            if current_run > longest_run:
                longest_run = current_run
                longest_start = current_start
        else:
            current_run = 0
            current_start = None

    if longest_run < min_close_run:
        start_text = "None" if longest_start is None else str(longest_start)
        append_issue(
            issues,
            episode,
            "FAIL",
            "FAIL_RIGHT_GRIPPER_NO_SUSTAINED_CLOSE",
            f"longest_run_below_close={longest_run} < min_close_run={min_close_run}; close_threshold={close_threshold:.6f}; longest_start={start_text}",
        )


def audit_episode(
    episode_dir: Path,
    min_frames: int,
    warn_gap_ms: float,
    fail_gap_ms: float,
    check_image_paths: bool,
    right_gripper_open_threshold: float,
    right_gripper_close_threshold: float,
    right_gripper_min_close_run: int,
    right_gripper_min_motion_range: float,
) -> tuple[EpisodeSummary, list[Issue]]:
    episode = episode_dir.name
    data_json = episode_dir / "data.json"
    issues: list[Issue] = []

    if not data_json.is_file():
        append_issue(issues, episode, "FAIL", "FAIL_MISSING_JSON", f"missing {data_json}")
        return EpisodeSummary(episode, None, None, None, data_json), issues

    json_error = json_tool_error(data_json)
    if json_error is not None:
        append_issue(issues, episode, "FAIL", "FAIL_BAD_JSON", json_error)
        return EpisodeSummary(episode, None, None, None, data_json), issues

    payload = load_payload(data_json)
    data = payload.get("data")
    if not isinstance(data, list):
        append_issue(issues, episode, "FAIL", "FAIL_SCHEMA", "top-level data must be a list")
        return EpisodeSummary(episode, None, None, None, data_json), issues
    if not data:
        append_issue(issues, episode, "FAIL", "FAIL_EMPTY", "data list is empty")
        return EpisodeSummary(episode, 0, None, None, data_json), issues

    frame_count = len(data)
    if min_frames > 0 and frame_count < min_frames:
        append_issue(issues, episode, "WARN", "WARN_SHORT_EPISODE", f"frames={frame_count} < min_frames={min_frames}")

    first_sample_ns: int | None = None
    previous_sample_ns: int | None = None
    max_gap_ns = 0
    right_gripper_values: list[float] = []

    for frame_index, item in enumerate(data):
        if not isinstance(item, dict):
            append_issue(issues, episode, "FAIL", "FAIL_SCHEMA", f"frame {frame_index} is not an object")
            continue

        idx = item.get("idx")
        if idx != frame_index:
            append_issue(issues, episode, "FAIL", "FAIL_IDX", f"frame {frame_index} idx={idx}, expected={frame_index}")

        timestamps = item.get("timestamps")
        if not isinstance(timestamps, dict) or "sample_monotonic_ns" not in timestamps:
            append_issue(
                issues,
                episode,
                "FAIL",
                "FAIL_TIMESTAMP",
                f"frame {frame_index} missing timestamps.sample_monotonic_ns",
            )
        else:
            sample_ns = int(timestamps["sample_monotonic_ns"])
            if first_sample_ns is None:
                first_sample_ns = sample_ns
            if sample_ns < 0:
                append_issue(
                    issues,
                    episode,
                    "FAIL",
                    "FAIL_TIMESTAMP",
                    f"frame {frame_index} sample_monotonic_ns is negative: {sample_ns}",
                )
            if previous_sample_ns is not None:
                gap_ns = sample_ns - previous_sample_ns
                if gap_ns <= 0:
                    append_issue(
                        issues,
                        episode,
                        "FAIL",
                        "FAIL_TIMESTAMP",
                        f"frame {frame_index} sample_monotonic_ns not increasing: {sample_ns} <= {previous_sample_ns}",
                    )
                elif gap_ns > max_gap_ns:
                    max_gap_ns = gap_ns
            previous_sample_ns = sample_ns

        colors = item.get("colors")
        if not isinstance(colors, dict):
            append_issue(issues, episode, "FAIL", "FAIL_COLORS", f"frame {frame_index} missing colors")
        else:
            for camera_name in REQUIRED_CAMERAS:
                rel_path = colors.get(camera_name)
                if not isinstance(rel_path, str) or not rel_path.strip():
                    append_issue(
                        issues,
                        episode,
                        "FAIL",
                        "FAIL_IMAGE_PATH",
                        f"frame {frame_index} missing colors.{camera_name}",
                    )
                    continue
                if check_image_paths and not (episode_dir / rel_path).is_file():
                    append_issue(
                        issues,
                        episode,
                        "FAIL",
                        "FAIL_IMAGE_MISSING",
                        f"frame {frame_index} colors.{camera_name} not found: {rel_path}",
                    )

        validate_qpos(issues, episode, item, frame_index, "states")
        validate_qpos(issues, episode, item, frame_index, "actions")
        value = right_gripper_value(
            issues,
            episode=episode,
            item=item,
            frame_index=frame_index,
        )
        if value is not None:
            right_gripper_values.append(value)

    validate_right_gripper_event(
        issues=issues,
        episode=episode,
        values=right_gripper_values,
        open_threshold=right_gripper_open_threshold,
        close_threshold=right_gripper_close_threshold,
        min_close_run=right_gripper_min_close_run,
        min_motion_range=right_gripper_min_motion_range,
    )

    if max_gap_ns > int(fail_gap_ms * 1e6):
        append_issue(
            issues,
            episode,
            "FAIL",
            "FAIL_TIME_GAP",
            f"max sample gap={max_gap_ns / 1e6:.3f} ms > fail_gap_ms={fail_gap_ms:.3f}",
        )
    elif max_gap_ns > int(warn_gap_ms * 1e6):
        append_issue(
            issues,
            episode,
            "WARN",
            "WARN_TIME_GAP",
            f"max sample gap={max_gap_ns / 1e6:.3f} ms > warn_gap_ms={warn_gap_ms:.3f}",
        )

    duration_s = None
    if first_sample_ns is not None and previous_sample_ns is not None:
        duration_s = max(0.0, float(previous_sample_ns - first_sample_ns) / 1e9)
    max_gap_ms = float(max_gap_ns) / 1e6 if max_gap_ns > 0 else None
    return EpisodeSummary(episode, frame_count, duration_s, max_gap_ms, data_json), issues


def frame_count_bounds(frame_counts: list[int], iqr_scale: float) -> tuple[float | None, float | None]:
    sorted_counts = sorted(frame_counts)
    q1 = percentile(sorted_counts, 25)
    q3 = percentile(sorted_counts, 75)
    if q1 is None or q3 is None:
        return None, None
    iqr = q3 - q1
    return max(0.0, q1 - iqr_scale * iqr), q3 + iqr_scale * iqr


def print_frame_stats(summaries: list[EpisodeSummary], iqr_scale: float) -> tuple[float | None, float | None]:
    counts = sorted(summary.frame_count for summary in summaries if summary.frame_count is not None)
    print("FRAME_COUNT_STATS")
    if not counts:
        print("readable_episode_count=0")
        print("")
        return None, None
    low_bound, high_bound = frame_count_bounds(counts, iqr_scale)
    print(f"readable_episode_count={len(counts)}")
    print(f"min={counts[0]}")
    print(f"q1={percentile(counts, 25):.3f}")
    print(f"median={statistics.median(counts):.3f}")
    print(f"q3={percentile(counts, 75):.3f}")
    print(f"p90={percentile(counts, 90):.3f}")
    print(f"p95={percentile(counts, 95):.3f}")
    print(f"max={counts[-1]}")
    print(f"iqr_low_bound={low_bound:.3f}")
    print(f"iqr_high_bound={high_bound:.3f}")
    print("")
    return low_bound, high_bound


def print_table(title: str, rows: list[str]) -> None:
    print(title)
    if rows:
        for row in rows:
            print(row)
    else:
        print("(none)")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit raw utils/data/multi_cam_record episodes without decoding images or creating a clean dataset."
    )
    parser.add_argument("--dataset", default="utils/data/multi_cam_record", help="Path to multi_cam_record directory.")
    parser.add_argument("--min-frames", type=int, default=100, help="Warn when an episode has fewer frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="Warn when an episode has more frames. 0 disables manual max.")
    parser.add_argument("--warn-gap-ms", type=float, default=200.0, help="Warn when sample timestamp gap exceeds this.")
    parser.add_argument("--fail-gap-ms", type=float, default=500.0, help="Fail when sample timestamp gap exceeds this.")
    parser.add_argument("--iqr-scale", type=float, default=1.5, help="IQR multiplier for automatic frame-count outliers.")
    parser.add_argument(
        "--right-gripper-open-threshold",
        type=float,
        default=4.8,
        help="Fail if actions.right_ee.qpos never reaches this open-state value.",
    )
    parser.add_argument(
        "--right-gripper-close-threshold",
        type=float,
        default=2.8,
        help="Close-event threshold for actions.right_ee.qpos.",
    )
    parser.add_argument(
        "--right-gripper-min-close-run",
        type=int,
        default=100,
        help="Fail if actions.right_ee.qpos is not below close threshold for at least this many consecutive frames.",
    )
    parser.add_argument(
        "--right-gripper-min-motion-range",
        type=float,
        default=2.5,
        help="Fail if max(actions.right_ee.qpos) - min(actions.right_ee.qpos) is below this value.",
    )
    parser.add_argument(
        "--skip-image-path-check",
        action="store_true",
        help="Do not check whether referenced image files exist. JSON fields are still checked.",
    )
    parser.add_argument("--no-frame-table", action="store_true", help="Do not print the per-episode frame table.")
    parser.add_argument("--full-issues", action="store_true", help="Print every raw issue line instead of only grouped issues.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress to stderr every N episodes. Use 0 to disable.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser()
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset}")
    if args.min_frames < 0:
        raise ValueError("--min-frames must be non-negative")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.warn_gap_ms <= 0:
        raise ValueError("--warn-gap-ms must be positive")
    if args.fail_gap_ms <= 0:
        raise ValueError("--fail-gap-ms must be positive")
    if args.warn_gap_ms > args.fail_gap_ms:
        raise ValueError("--warn-gap-ms must be <= --fail-gap-ms")
    if args.iqr_scale <= 0:
        raise ValueError("--iqr-scale must be positive")
    if not math.isfinite(args.right_gripper_open_threshold):
        raise ValueError("--right-gripper-open-threshold must be finite")
    if not math.isfinite(args.right_gripper_close_threshold):
        raise ValueError("--right-gripper-close-threshold must be finite")
    if args.right_gripper_open_threshold <= args.right_gripper_close_threshold:
        raise ValueError("--right-gripper-open-threshold must be greater than --right-gripper-close-threshold")
    if args.right_gripper_min_close_run <= 0:
        raise ValueError("--right-gripper-min-close-run must be positive")
    if args.right_gripper_min_motion_range <= 0:
        raise ValueError("--right-gripper-min-motion-range must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    episode_dirs = sorted(path for path in dataset.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"no episode_* directories found under: {dataset}")

    summaries: list[EpisodeSummary] = []
    issues: list[Issue] = []
    total_episodes = len(episode_dirs)
    if args.progress_every > 0:
        print(f"[progress] scanning {total_episodes} episodes under {dataset}", file=sys.stderr, flush=True)
    for episode_index, episode_dir in enumerate(episode_dirs, start=1):
        if args.progress_every > 0 and (episode_index == 1 or episode_index % args.progress_every == 0 or episode_index == total_episodes):
            print(f"[progress] {episode_index}/{total_episodes} {episode_dir.name}", file=sys.stderr, flush=True)
        summary, episode_issues = audit_episode(
            episode_dir=episode_dir,
            min_frames=args.min_frames,
            warn_gap_ms=args.warn_gap_ms,
            fail_gap_ms=args.fail_gap_ms,
            check_image_paths=not args.skip_image_path_check,
            right_gripper_open_threshold=args.right_gripper_open_threshold,
            right_gripper_close_threshold=args.right_gripper_close_threshold,
            right_gripper_min_close_run=args.right_gripper_min_close_run,
            right_gripper_min_motion_range=args.right_gripper_min_motion_range,
        )
        summaries.append(summary)
        issues.extend(episode_issues)

    low_bound, high_bound = print_frame_stats(summaries, args.iqr_scale)

    frame_outlier_issues: list[Issue] = []
    for summary in summaries:
        if summary.frame_count is None:
            continue
        if low_bound is not None and summary.frame_count < low_bound:
            frame_outlier_issues.append(
                Issue(summary.episode, "WARN", "WARN_FRAME_COUNT_IQR_LOW", f"frames={summary.frame_count} < iqr_low_bound={low_bound:.3f}")
            )
        if high_bound is not None and summary.frame_count > high_bound:
            frame_outlier_issues.append(
                Issue(summary.episode, "WARN", "WARN_FRAME_COUNT_IQR_HIGH", f"frames={summary.frame_count} > iqr_high_bound={high_bound:.3f}")
            )
        if args.max_frames > 0 and summary.frame_count > args.max_frames:
            frame_outlier_issues.append(
                Issue(summary.episode, "WARN", "WARN_LONG_EPISODE", f"frames={summary.frame_count} > max_frames={args.max_frames}")
            )
    issues.extend(frame_outlier_issues)

    fail_episodes = sorted({issue.episode for issue in issues if issue.severity == "FAIL"})
    warn_episodes = sorted({issue.episode for issue in issues if issue.severity == "WARN"} - set(fail_episodes))
    issue_episodes = sorted(set(fail_episodes) | set(warn_episodes))
    pass_episodes = sorted(summary.episode for summary in summaries if summary.episode not in issue_episodes)
    code_counts = Counter(issue.code for issue in issues)

    print("SUMMARY")
    print(f"dataset={dataset}")
    print(f"episode_count={len(summaries)}")
    print(f"pass={len(pass_episodes)}")
    print(f"fail={len(fail_episodes)}")
    print(f"warn_only={len(warn_episodes)}")
    print(f"issue_count={len(issues)}")
    print("")

    print_table("FAIL_EPISODES", fail_episodes)
    print_table("WARN_ONLY_EPISODES", warn_episodes)

    short_rows = []
    long_rows = []
    for summary in summaries:
        if summary.frame_count is None:
            continue
        short_by_manual = args.min_frames > 0 and summary.frame_count < args.min_frames
        short_by_iqr = low_bound is not None and summary.frame_count < low_bound
        long_by_manual = args.max_frames > 0 and summary.frame_count > args.max_frames
        long_by_iqr = high_bound is not None and summary.frame_count > high_bound
        detail = (
            f"{summary.episode}\tframes={summary.frame_count}"
            f"\tduration_s={summary.duration_s:.3f}" if summary.duration_s is not None else f"{summary.episode}\tframes={summary.frame_count}\tduration_s=None"
        )
        if short_by_manual or short_by_iqr:
            short_rows.append(detail)
        if long_by_manual or long_by_iqr:
            long_rows.append(detail)
    print_table("SHORT_EPISODES", short_rows)
    print_table("LONG_EPISODES", long_rows)

    print("ISSUE_CODE_COUNTS")
    if code_counts:
        for code, count in sorted(code_counts.items()):
            print(f"{code}\t{count}")
    else:
        print("(none)")
    print("")

    grouped_issues: dict[tuple[str, str, str], list[Any]] = defaultdict(lambda: [0, ""])
    for issue in issues:
        key = (issue.episode, issue.severity, issue.code)
        grouped_issues[key][0] += 1
        if not grouped_issues[key][1]:
            grouped_issues[key][1] = issue.message

    print("ISSUE_GROUPS")
    if grouped_issues:
        for (episode, severity, code), (count, first_message) in sorted(grouped_issues.items()):
            print(f"{episode}\t{severity}\t{code}\tcount={count}\tfirst={first_message}")
    else:
        print("(none)")
    print("")

    if not args.no_frame_table:
        print("FRAME_COUNT_BY_EPISODE")
        for summary in summaries:
            frame_count = "None" if summary.frame_count is None else str(summary.frame_count)
            duration_s = "None" if summary.duration_s is None else f"{summary.duration_s:.3f}"
            max_gap_ms = "None" if summary.max_gap_ms is None else f"{summary.max_gap_ms:.3f}"
            print(f"{summary.episode}\tframes={frame_count}\tduration_s={duration_s}\tmax_gap_ms={max_gap_ms}")
        print("")

    if args.full_issues:
        print("ISSUES")
        if issues:
            for issue in sorted(issues, key=lambda item: (item.episode, item.severity, item.code, item.message)):
                print(f"{issue.episode}\t{issue.severity}\t{issue.code}\t{issue.message}")
        else:
            print("(none)")


if __name__ == "__main__":
    main()
