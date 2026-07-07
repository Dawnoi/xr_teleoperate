#!/usr/bin/env python3
"""Audit raw multi-camera episode datasets without modifying the source data."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any


CAMERA_KEYS = ("head", "left_wrist", "right_wrist")
CAMERA_DIRS = {
    "head": "colors/head",
    "left_wrist": "colors/wrist_left",
    "right_wrist": "colors/wrist_right",
}
QPOS_SPECS = (
    ("states", "left_arm", 7),
    ("states", "right_arm", 7),
    ("states", "left_ee", 1),
    ("states", "right_ee", 1),
    ("actions", "left_arm", 7),
    ("actions", "right_arm", 7),
    ("actions", "left_ee", 1),
    ("actions", "right_ee", 1),
)
EPISODE_RE = re.compile(r"^episode_(\d+)$")
GRIPPER_SOURCES = ("states", "actions")
GRIPPER_SIDES = ("left_ee", "right_ee")


@dataclass(frozen=True)
class Issue:
    code: str
    message: str


@dataclass
class EpisodeAudit:
    episode: str
    hard: list[Issue] = field(default_factory=list)
    review: list[Issue] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def parse_bool_flag(value: Any, label: str) -> bool:
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{label} must be one of 0/1/true/false/yes/no/on/off, got {value!r}")


def episode_number(path: Path) -> int | None:
    match = EPISODE_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * pct
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    alpha = position - low
    return float(sorted_values[low] * (1.0 - alpha) + sorted_values[high] * alpha)


def finite_vector(value: Any, expected_len: int) -> bool:
    if not isinstance(value, list) or len(value) != expected_len:
        return False
    return all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


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


def load_frames(data_json: Path) -> list[dict[str, Any]] | None:
    payload = json.loads(data_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        data = payload.get("data")
    elif isinstance(payload, list):
        data = payload
    else:
        return None
    if not isinstance(data, list):
        return None
    if not all(isinstance(item, dict) for item in data):
        return None
    return data


def longest_run_below(values: list[float], threshold: float) -> int:
    longest = 0
    current = 0
    for value in values:
        if value < threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def count_below(values: list[float], threshold: float) -> int:
    return sum(1 for value in values if value < threshold)


def min_with_index(values: list[float]) -> tuple[float, int]:
    index = min(range(len(values)), key=values.__getitem__)
    return values[index], index


def max_step_with_index(values: list[float]) -> tuple[float, int]:
    if len(values) < 2:
        return 0.0, 0
    best = abs(values[1] - values[0])
    best_index = 1
    for index in range(2, len(values)):
        step = abs(values[index] - values[index - 1])
        if step > best:
            best = step
            best_index = index
    return best, best_index


def qpos_scalar(frame: dict[str, Any], source: str, group: str) -> float | None:
    source_value = frame.get(source)
    if not isinstance(source_value, dict):
        return None
    group_value = source_value.get(group)
    if not isinstance(group_value, dict):
        return None
    qpos = group_value.get("qpos")
    if not finite_vector(qpos, 1):
        return None
    return float(qpos[0])


def validate_qpos(frame: dict[str, Any], frame_index: int, audit: EpisodeAudit) -> None:
    for source, group, expected_len in QPOS_SPECS:
        source_value = frame.get(source)
        if not isinstance(source_value, dict):
            audit.hard.append(Issue("BAD_QPOS", f"frame {frame_index} missing {source}"))
            continue
        group_value = source_value.get(group)
        if not isinstance(group_value, dict):
            audit.hard.append(Issue("BAD_QPOS", f"frame {frame_index} missing {source}.{group}"))
            continue
        qpos = group_value.get("qpos")
        if not finite_vector(qpos, expected_len):
            actual_len = len(qpos) if isinstance(qpos, list) else "not-list"
            audit.hard.append(
                Issue(
                    "BAD_QPOS",
                    f"frame {frame_index} {source}.{group}.qpos len={actual_len}, expected={expected_len}",
                )
            )


def validate_images(
    episode_dir: Path,
    frames: list[dict[str, Any]],
    audit: EpisodeAudit,
    decode_images: bool,
    cv2_module: Any,
    expected_image_shape: tuple[int, int, int],
) -> None:
    missing_count = 0
    first_missing = ""
    bad_decode_count = 0
    first_bad_decode = ""
    bad_shape_count = 0
    first_bad_shape = ""
    missing_delta_count = 0
    first_missing_delta = ""
    bad_delta_count = 0
    first_bad_delta = ""
    camera_delta_ms: dict[str, list[float]] = {key: [] for key in CAMERA_KEYS}

    for frame_index, frame in enumerate(frames):
        colors = frame.get("colors")
        if not isinstance(colors, dict):
            missing_count += 1
            if not first_missing:
                first_missing = f"frame {frame_index} missing colors"
            continue

        for camera_key in CAMERA_KEYS:
            if camera_key not in colors:
                missing_count += 1
                if not first_missing:
                    first_missing = f"frame {frame_index} missing colors.{camera_key}"
                continue

            relative_path = colors[camera_key]
            if not isinstance(relative_path, str) or not relative_path:
                missing_count += 1
                if not first_missing:
                    first_missing = f"frame {frame_index} colors.{camera_key} is not a non-empty path"
                continue

            image_path = episode_dir / relative_path
            if not image_path.is_file():
                missing_count += 1
                if not first_missing:
                    first_missing = f"frame {frame_index} colors.{camera_key} not found: {relative_path}"
                continue

            if decode_images:
                image = cv2_module.imread(str(image_path), cv2_module.IMREAD_COLOR)
                if image is None:
                    bad_decode_count += 1
                    if not first_bad_decode:
                        first_bad_decode = f"frame {frame_index} colors.{camera_key}: {relative_path}"
                elif tuple(image.shape) != expected_image_shape:
                    bad_shape_count += 1
                    if not first_bad_shape:
                        first_bad_shape = (
                            f"frame {frame_index} colors.{camera_key} shape={tuple(image.shape)}, "
                            f"expected={expected_image_shape}"
                        )

            if camera_key == "head":
                camera_delta_ms[camera_key].append(0.0)
                continue

            timestamps = frame.get("timestamps")
            camera_timestamps = timestamps.get("camera") if isinstance(timestamps, dict) else None
            camera_meta = camera_timestamps.get(camera_key) if isinstance(camera_timestamps, dict) else None
            if not isinstance(camera_meta, dict) or "delta_to_sample_ns" not in camera_meta:
                missing_delta_count += 1
                if not first_missing_delta:
                    first_missing_delta = f"frame {frame_index} missing timestamps.camera.{camera_key}.delta_to_sample_ns"
                continue

            delta_ns = camera_meta["delta_to_sample_ns"]
            if not finite_number(delta_ns):
                bad_delta_count += 1
                if not first_bad_delta:
                    first_bad_delta = f"frame {frame_index} timestamps.camera.{camera_key}.delta_to_sample_ns={delta_ns!r}"
                continue

            camera_delta_ms[camera_key].append(abs(float(delta_ns)) / 1_000_000.0)

    if missing_count:
        audit.hard.append(Issue("MISSING_IMAGE", f"{first_missing}; count={missing_count}"))
    if bad_decode_count:
        audit.hard.append(Issue("BAD_IMAGE_DECODE", f"{first_bad_decode}; count={bad_decode_count}"))
    if bad_shape_count:
        audit.hard.append(Issue("BAD_IMAGE_SHAPE", f"{first_bad_shape}; count={bad_shape_count}"))
    if missing_delta_count:
        audit.review.append(Issue("MISSING_CAMERA_DELTA", f"{first_missing_delta}; count={missing_delta_count}"))
    if bad_delta_count:
        audit.review.append(Issue("BAD_CAMERA_DELTA", f"{first_bad_delta}; count={bad_delta_count}"))

    folder_counts = {
        key: len(list((episode_dir / CAMERA_DIRS[key]).glob("*.jpg")))
        for key in CAMERA_KEYS
    }
    audit.details["folder_counts"] = folder_counts
    mismatched_counts = {key: value for key, value in folder_counts.items() if value != len(frames)}
    if mismatched_counts:
        audit.review.append(Issue("IMAGE_FOLDER_COUNT_MISMATCH", str(mismatched_counts)))

    audit.details["camera_max_delta_ms"] = {
        key: (max(values) if values else None)
        for key, values in camera_delta_ms.items()
    }


def audit_episode(
    episode_dir: Path,
    args: argparse.Namespace,
    cv2_module: Any,
    expected_image_shape: tuple[int, int, int],
) -> EpisodeAudit:
    audit = EpisodeAudit(episode=episode_dir.name)
    data_json = episode_dir / "data.json"
    if not data_json.is_file():
        audit.hard.append(Issue("MISSING_DATA_JSON", str(data_json)))
        return audit

    json_error = json_tool_error(data_json)
    if json_error is not None:
        audit.hard.append(Issue("BAD_DATA_JSON", json_error))
        return audit

    frames = load_frames(data_json)
    if frames is None:
        audit.hard.append(Issue("BAD_DATA_SCHEMA", "data.json must be a list or an object with a data list of frame objects"))
        return audit
    if not frames:
        audit.hard.append(Issue("EMPTY_DATA", "data list is empty"))
        return audit

    audit.details["frames"] = len(frames)

    sample_ns_values: list[int] = []
    for frame_index, frame in enumerate(frames):
        if frame.get("idx") != frame_index:
            audit.review.append(Issue("IDX_NOT_CONTIGUOUS", f"frame {frame_index} idx={frame.get('idx')}"))
            break

    for frame_index, frame in enumerate(frames):
        timestamps = frame.get("timestamps")
        if not isinstance(timestamps, dict) or "sample_monotonic_ns" not in timestamps:
            audit.hard.append(Issue("MISSING_SAMPLE_TS", f"frame {frame_index} missing timestamps.sample_monotonic_ns"))
            break
        sample_ns_raw = timestamps["sample_monotonic_ns"]
        if not finite_number(sample_ns_raw):
            audit.hard.append(Issue("BAD_SAMPLE_TS", f"frame {frame_index} sample_monotonic_ns={sample_ns_raw!r}"))
            break
        sample_ns = int(sample_ns_raw)
        if sample_ns < 0:
            audit.hard.append(Issue("BAD_SAMPLE_TS", f"frame {frame_index} sample_monotonic_ns is negative"))
            break
        sample_ns_values.append(sample_ns)

    if len(sample_ns_values) == len(frames):
        max_gap_ms = 0.0
        for index in range(1, len(sample_ns_values)):
            if sample_ns_values[index] <= sample_ns_values[index - 1]:
                audit.hard.append(Issue("NON_MONOTONIC_SAMPLE_TS", f"frame {index} timestamp is not increasing"))
                break
            max_gap_ms = max(max_gap_ms, (sample_ns_values[index] - sample_ns_values[index - 1]) / 1_000_000.0)
        duration_s = (sample_ns_values[-1] - sample_ns_values[0]) / 1_000_000_000.0 if len(sample_ns_values) > 1 else 0.0
        fps = (len(sample_ns_values) - 1) / duration_s if duration_s > 0 else 0.0
        audit.details["duration_s"] = duration_s
        audit.details["fps"] = fps
        audit.details["max_gap_ms"] = max_gap_ms
        if max_gap_ms > args.fail_gap_ms:
            audit.hard.append(Issue("SAMPLE_GAP_GT_FAIL", f"max_gap_ms={max_gap_ms:.3f} > {args.fail_gap_ms:.3f}"))
        elif max_gap_ms > args.warn_gap_ms:
            audit.review.append(Issue("SAMPLE_GAP_GT_WARN", f"max_gap_ms={max_gap_ms:.3f} > {args.warn_gap_ms:.3f}"))

    for frame_index, frame in enumerate(frames):
        validate_qpos(frame, frame_index, audit)

    validate_images(
        episode_dir=episode_dir,
        frames=frames,
        audit=audit,
        decode_images=args.decode_images,
        cv2_module=cv2_module,
        expected_image_shape=expected_image_shape,
    )

    for source in GRIPPER_SOURCES:
        for side in GRIPPER_SIDES:
            values: list[float] = []
            for frame in frames:
                value = qpos_scalar(frame, source, side)
                if value is not None:
                    values.append(value)
            label = f"{source}.{side}"
            if values:
                minimum, minimum_index = min_with_index(values)
                max_step, max_step_index = max_step_with_index(values)
                audit.details[f"{label}.min"] = minimum
                audit.details[f"{label}.min_i"] = minimum_index
                audit.details[f"{label}.below_count"] = count_below(values, args.gripper_close_threshold)
                audit.details[f"{label}.below_run"] = longest_run_below(values, args.gripper_close_threshold)
                audit.details[f"{label}.max_step"] = max_step
                audit.details[f"{label}.max_step_i"] = max_step_index

    review_gripper_sources = GRIPPER_SOURCES if args.state_gripper_review else ("actions",)
    for source in review_gripper_sources:
        for side in GRIPPER_SIDES:
            minimum = audit.details.get(f"{source}.{side}.min")
            below_run = audit.details.get(f"{source}.{side}.below_run")
            if minimum is None or below_run is None:
                continue
            side_name = "LEFT" if side == "left_ee" else "RIGHT"
            source_name = "STATE" if source == "states" else "ACTION"
            min_close_run = args.left_gripper_min_close_run if side == "left_ee" else args.right_gripper_min_close_run
            if minimum >= args.gripper_close_threshold:
                audit.review.append(Issue(f"{side_name}_GRIPPER_{source_name}_NEVER_CLOSE", f"{source}.{side}.min={minimum:.6f}"))
            elif below_run < min_close_run:
                audit.review.append(
                    Issue(
                        f"{side_name}_GRIPPER_{source_name}_CLOSE_TOO_SHORT",
                        f"{source}.{side}.min={minimum:.6f}, longest_run={below_run} < {min_close_run}",
                    )
                )

    if args.ee_action_step_review > 0:
        for side in ("left_ee", "right_ee"):
            step = audit.details.get(f"actions.{side}.max_step")
            step_index = audit.details.get(f"actions.{side}.max_step_i")
            if step is not None and step > args.ee_action_step_review:
                audit.review.append(
                    Issue(
                        f"{side.upper()}_ACTION_STEP_GT_THRESHOLD",
                        f"step={step:.6f}@{step_index} > {args.ee_action_step_review:.6f}",
                    )
                )

    for camera_key, delta_ms in audit.details.get("camera_max_delta_ms", {}).items():
        if delta_ms is not None and delta_ms > args.camera_delta_review_ms:
            audit.review.append(
                Issue(
                    "CAMERA_ALIGN_DELTA_GT_REVIEW",
                    f"{camera_key} max_abs_delta_ms={delta_ms:.3f} > {args.camera_delta_review_ms:.3f}",
                )
            )

    return audit


def list_episode_dirs(dataset: Path, start: int | None, end: int | None) -> list[Path]:
    episode_dirs = []
    for path in dataset.iterdir():
        if not path.is_dir():
            continue
        number = episode_number(path)
        if number is None:
            continue
        if start is not None and number < start:
            continue
        if end is not None and number > end:
            continue
        episode_dirs.append(path)
    return sorted(episode_dirs, key=lambda item: int(EPISODE_RE.fullmatch(item.name).group(1)))


def apply_frame_count_rules(results: list[EpisodeAudit], args: argparse.Namespace) -> dict[str, float]:
    frame_counts = sorted(int(result.details["frames"]) for result in results if "frames" in result.details)
    if not frame_counts:
        return {}
    q1 = percentile(frame_counts, 0.25)
    q3 = percentile(frame_counts, 0.75)
    iqr = q3 - q1
    low_bound = q1 - args.iqr_scale * iqr
    high_bound = q3 + args.iqr_scale * iqr

    for result in results:
        frame_count = result.details.get("frames")
        if frame_count is None:
            continue
        if args.min_frames > 0 and frame_count < args.min_frames:
            result.hard.append(Issue("FRAME_COUNT_LT_MIN", f"frames={frame_count} < min_frames={args.min_frames}"))
        if args.max_frames > 0 and frame_count > args.max_frames:
            result.review.append(Issue("FRAME_COUNT_GT_MAX", f"frames={frame_count} > max_frames={args.max_frames}"))
        if frame_count < low_bound:
            if args.short_frame_severity == "hard":
                result.hard.append(Issue("FRAME_COUNT_IQR_LOW", f"frames={frame_count} < iqr_low_bound={low_bound:.3f}"))
            elif args.short_frame_severity == "review":
                result.review.append(Issue("FRAME_COUNT_IQR_LOW", f"frames={frame_count} < iqr_low_bound={low_bound:.3f}"))
        if frame_count > high_bound:
            if args.long_frame_severity == "hard":
                result.hard.append(Issue("FRAME_COUNT_IQR_HIGH", f"frames={frame_count} > iqr_high_bound={high_bound:.3f}"))
            elif args.long_frame_severity == "review":
                result.review.append(Issue("FRAME_COUNT_IQR_HIGH", f"frames={frame_count} > iqr_high_bound={high_bound:.3f}"))

    return {
        "min": float(min(frame_counts)),
        "median": float(median(frame_counts)),
        "max": float(max(frame_counts)),
        "q1": q1,
        "q3": q3,
        "iqr_low_bound": low_bound,
        "iqr_high_bound": high_bound,
    }


def issue_text(issues: list[Issue]) -> str:
    return "; ".join(f"{issue.code}: {issue.message}" for issue in issues)


def status(result: EpisodeAudit) -> str:
    if result.hard:
        return "HARD_FAIL"
    if result.review:
        return "REVIEW"
    return "PASS"


def format_value(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "None"
    return str(value)


def write_reports(results: list[EpisodeAudit], frame_stats: dict[str, float], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{args.output_prefix}.txt"
    problem_path = args.output_dir / f"{args.output_prefix}_problem_or_review.txt"
    hard_path = args.output_dir / f"{args.output_prefix}_hard_fail_episodes.txt"

    hard_results = [result for result in results if result.hard]
    review_results = [result for result in results if (not result.hard) and result.review]
    pass_results = [result for result in results if (not result.hard) and (not result.review)]

    lines = [
        "AUDIT_MULTI_CAM_RECORD",
        f"dataset={args.dataset}",
        f"episode_count={len(results)}",
        f"pass={len(pass_results)}",
        f"hard_fail={len(hard_results)}",
        f"review_only={len(review_results)}",
        f"decode_images={args.decode_images}",
        f"state_gripper_review={args.state_gripper_review}",
        f"gripper_close_threshold={args.gripper_close_threshold}",
        f"left_gripper_min_close_run={args.left_gripper_min_close_run} (review only)",
        f"right_gripper_min_close_run={args.right_gripper_min_close_run} (review only)",
        f"short_frame_severity={args.short_frame_severity}",
        f"long_frame_severity={args.long_frame_severity}",
    ]
    if frame_stats:
        lines.extend([
            f"frame_min={frame_stats['min']:.0f}",
            f"frame_median={frame_stats['median']:.3f}",
            f"frame_max={frame_stats['max']:.0f}",
            f"frame_iqr_low_bound={frame_stats['iqr_low_bound']:.3f}",
            f"frame_iqr_high_bound={frame_stats['iqr_high_bound']:.3f}",
        ])
    lines.extend([
        "",
        "HARD_FAIL_RULES",
        "- data.json missing, invalid, empty, or wrong schema.",
        "- qpos missing, non-finite, or wrong vector length.",
        "- required image path missing; optional image decode/shape failure when --decode-images=1.",
        "- sample_monotonic_ns missing, non-finite, negative, non-increasing, or gap above fail threshold.",
        "- short frame count when short_frame_severity=hard.",
        "",
        "MANUAL_REVIEW_RULES",
        "- left/right gripper action never goes below gripper_close_threshold.",
        "- left/right gripper action close run is shorter than the configured minimum.",
        "- state gripper close checks are optional and only enabled by --state-gripper-review=1.",
        "- long frame count, camera alignment delta warning, idx discontinuity, or optional ee action jump.",
        "- left gripper ineffective close is REVIEW-only; it is not written to hard_fail_episodes.txt.",
        "",
        "PER_EPISODE",
        (
            "episode\tstatus\tframes\tduration_s\tfps\tmax_gap_ms\t"
            "left_action_min@i\tleft_action_count/run\tleft_state_min@i\tleft_state_count/run\t"
            "right_action_min@i\tright_action_count/run\tright_state_min@i\tright_state_count/run\t"
            "camera_max_delta_ms\tissues"
        ),
    ])

    for result in results:
        details = result.details
        lines.append(
            "\t".join([
                result.episode,
                status(result),
                format_value(details.get("frames")),
                format_value(details.get("duration_s")),
                format_value(details.get("fps")),
                format_value(details.get("max_gap_ms")),
                f"{format_value(details.get('actions.left_ee.min'), 6)}@{format_value(details.get('actions.left_ee.min_i'))}",
                f"{format_value(details.get('actions.left_ee.below_count'))}/{format_value(details.get('actions.left_ee.below_run'))}",
                f"{format_value(details.get('states.left_ee.min'), 6)}@{format_value(details.get('states.left_ee.min_i'))}",
                f"{format_value(details.get('states.left_ee.below_count'))}/{format_value(details.get('states.left_ee.below_run'))}",
                f"{format_value(details.get('actions.right_ee.min'), 6)}@{format_value(details.get('actions.right_ee.min_i'))}",
                f"{format_value(details.get('actions.right_ee.below_count'))}/{format_value(details.get('actions.right_ee.below_run'))}",
                f"{format_value(details.get('states.right_ee.min'), 6)}@{format_value(details.get('states.right_ee.min_i'))}",
                f"{format_value(details.get('states.right_ee.below_count'))}/{format_value(details.get('states.right_ee.below_run'))}",
                json.dumps(details.get("camera_max_delta_ms", {}), sort_keys=True),
                issue_text(result.hard + result.review) if (result.hard or result.review) else "PASS",
            ])
        )

    lines.extend(["", "HARD_FAIL_EXCLUDE"])
    if hard_results:
        lines.extend(f"{result.episode}\t{issue_text(result.hard)}" for result in hard_results)
    else:
        lines.append("(none)")

    lines.extend(["", "REVIEW"])
    if review_results:
        lines.extend(f"{result.episode}\t{issue_text(result.review)}" for result in review_results)
    else:
        lines.append("(none)")

    lines.extend(["", "PASS_EPISODES"])
    if pass_results:
        lines.extend(result.episode for result in pass_results)
    else:
        lines.append("(none)")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    problem_lines = ["[HARD_FAIL_EXCLUDE]"]
    if hard_results:
        problem_lines.extend(f"{result.episode}\t{issue_text(result.hard)}" for result in hard_results)
    else:
        problem_lines.append("(none)")
    problem_lines.extend(["", "[REVIEW]"])
    if review_results:
        problem_lines.extend(f"{result.episode}\t{issue_text(result.review)}" for result in review_results)
    else:
        problem_lines.append("(none)")
    problem_lines.extend([
        "",
        "[HARD_FAIL_EPISODES_DEDUP]",
    ])
    if hard_results:
        problem_lines.extend(result.episode for result in hard_results)
    else:
        problem_lines.append("(none)")
    problem_lines.extend([
        "",
        "[NOTES]",
        "Only HARD_FAIL_EXCLUDE episodes should be removed automatically.",
        "Gripper close failures, including left gripper ineffective close, are REVIEW-only for the transfer_black handover task.",
        "The report separates actions.*_ee command values from states.*_ee feedback values.",
    ])
    problem_path.write_text("\n".join(problem_lines) + "\n", encoding="utf-8")
    hard_path.write_text("\n".join(result.episode for result in hard_results) + ("\n" if hard_results else ""), encoding="utf-8")

    print(f"WROTE_REPORT={report_path}")
    print(f"WROTE_PROBLEM={problem_path}")
    print(f"WROTE_HARD_FAIL_LIST={hard_path}")
    print(f"SUMMARY pass={len(pass_results)} hard_fail={len(hard_results)} review_only={len(review_results)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit raw multi_cam_record episodes without changing source data.")
    parser.add_argument("--dataset", type=Path, default=Path("utils/data/multi_cam_record/transfer_black"))
    parser.add_argument("--episode-start", type=int, default=None, help="Inclusive episode number, e.g. 207.")
    parser.add_argument("--episode-end", type=int, default=None, help="Inclusive episode number, e.g. 239.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--decode-images", type=lambda value: parse_bool_flag(value, "--decode-images"), default=False)
    parser.add_argument("--state-gripper-review", type=lambda value: parse_bool_flag(value, "--state-gripper-review"), default=False)
    parser.add_argument("--expected-image-height", type=int, default=480)
    parser.add_argument("--expected-image-width", type=int, default=640)
    parser.add_argument("--warn-gap-ms", type=float, default=200.0)
    parser.add_argument("--fail-gap-ms", type=float, default=500.0)
    parser.add_argument("--camera-delta-review-ms", type=float, default=200.0)
    parser.add_argument("--iqr-scale", type=float, default=1.5)
    parser.add_argument("--short-frame-severity", choices=("hard", "review", "off"), default="hard")
    parser.add_argument("--long-frame-severity", choices=("hard", "review", "off"), default="review")
    parser.add_argument("--min-frames", type=int, default=0, help="Hard fail below this manual frame count; 0 disables.")
    parser.add_argument("--max-frames", type=int, default=0, help="Review above this manual frame count; 0 disables.")
    parser.add_argument("--gripper-close-threshold", type=float, default=2.8)
    parser.add_argument("--left-gripper-min-close-run", type=int, default=15)
    parser.add_argument("--right-gripper-min-close-run", type=int, default=1)
    parser.add_argument("--ee-action-step-review", type=float, default=0.0, help="0 disables EE action jump review.")
    parser.add_argument("--progress-interval", type=int, default=10, help="Print progress every N episodes; 0 disables.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.dataset = args.dataset.expanduser()
    if not args.dataset.is_dir():
        raise NotADirectoryError(f"dataset not found: {args.dataset}")
    if args.episode_start is not None and args.episode_end is not None and args.episode_start > args.episode_end:
        raise ValueError("--episode-start must be <= --episode-end")
    if args.min_frames < 0 or args.max_frames < 0:
        raise ValueError("--min-frames and --max-frames must be non-negative")
    if args.left_gripper_min_close_run < 0 or args.right_gripper_min_close_run < 0:
        raise ValueError("gripper min close runs must be non-negative")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be non-negative")

    if args.output_dir is None:
        args.output_dir = args.dataset.parent / f"{args.dataset.name}_audit"
    if args.output_prefix is None:
        if args.episode_start is not None or args.episode_end is not None:
            start_text = "start" if args.episode_start is None else f"{args.episode_start:04d}"
            end_text = "end" if args.episode_end is None else f"{args.episode_end:04d}"
            args.output_prefix = f"audit_{start_text}_{end_text}"
        else:
            args.output_prefix = "audit"

    cv2_module = None
    if args.decode_images:
        import cv2 as cv2_module

    episode_dirs = list_episode_dirs(args.dataset, args.episode_start, args.episode_end)
    if not episode_dirs:
        raise FileNotFoundError(f"no matching episode directories under {args.dataset}")

    expected_image_shape = (args.expected_image_height, args.expected_image_width, 3)
    results = []
    total = len(episode_dirs)
    for index, episode_dir in enumerate(episode_dirs, start=1):
        result = audit_episode(episode_dir, args, cv2_module, expected_image_shape)
        results.append(result)
        if args.progress_interval > 0 and (index == 1 or index == total or index % args.progress_interval == 0):
            print(
                (
                    f"[AUDIT] {index}/{total} {episode_dir.name} "
                    f"status={status(result)} frames={format_value(result.details.get('frames'))} "
                    f"hard={len(result.hard)} review={len(result.review)}"
                ),
                flush=True,
            )
    frame_stats = apply_frame_count_rules(results, args)
    write_reports(results, frame_stats, args)


if __name__ == "__main__":
    main()
