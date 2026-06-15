#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone


def ns_to_iso8601(ns):
    if ns is None:
        return None
    try:
        return datetime.fromtimestamp(float(ns) / 1e9, tz=timezone.utc).astimezone().isoformat()
    except Exception:
        return None


def canonical_camera_name(name):
    key = str(name or "").strip()
    mapping = {
        "head": "head",
        "left_wrist": "wrist_left",
        "right_wrist": "wrist_right",
        "wrist_left": "wrist_left",
        "wrist_right": "wrist_right",
    }
    return mapping.get(key, key)


def percentile(values, pct):
    if not values:
        return None
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    sorted_vals = sorted(float(v) for v in values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    alpha = pos - lo
    return float(sorted_vals[lo] * (1.0 - alpha) + sorted_vals[hi] * alpha)


def summarize_ms_abs(values_ns):
    abs_ms = [abs(float(v)) / 1e6 for v in values_ns if v is not None]
    if not abs_ms:
        return {
            "count": 0,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(abs_ms),
        "p50": round(percentile(abs_ms, 50), 3),
        "p90": round(percentile(abs_ms, 90), 3),
        "p99": round(percentile(abs_ms, 99), 3),
        "max": round(max(abs_ms), 3),
        "mean": round(sum(abs_ms) / len(abs_ms), 3),
    }


def summarize_ms(values_ns):
    ms = [float(v) / 1e6 for v in values_ns if v is not None]
    if not ms:
        return {
            "count": 0,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(ms),
        "p50": round(percentile(ms, 50), 3),
        "p90": round(percentile(ms, 90), 3),
        "p99": round(percentile(ms, 99), 3),
        "max": round(max(ms), 3),
        "mean": round(sum(ms) / len(ms), 3),
    }


def extract_alignment_metrics(meta):
    meta = meta or {}
    raw_delta_ns = meta.get("delta_to_sample_ns")
    support_max_abs_delta_ns = meta.get("support_max_abs_delta_ns")
    support_span_ns = meta.get("support_span_ns")
    support_prev_delta_ns = meta.get("support_prev_delta_to_sample_ns")
    support_next_delta_ns = meta.get("support_next_delta_to_sample_ns")
    source_count = meta.get("support_source_count")

    effective_abs_delta_ns = (
        int(support_max_abs_delta_ns)
        if support_max_abs_delta_ns is not None
        else (abs(int(raw_delta_ns)) if raw_delta_ns is not None else None)
    )
    return {
        "raw_delta_ns": int(raw_delta_ns) if raw_delta_ns is not None else None,
        "effective_abs_delta_ns": effective_abs_delta_ns,
        "support_span_ns": int(support_span_ns) if support_span_ns is not None else None,
        "support_prev_delta_ns": int(support_prev_delta_ns) if support_prev_delta_ns is not None else None,
        "support_next_delta_ns": int(support_next_delta_ns) if support_next_delta_ns is not None else None,
        "support_source_count": int(source_count) if source_count is not None else None,
    }


def infer_arm_repr_mode(samples):
    has_qpos = False
    has_pose = False
    for item in samples:
        for section in ("states", "actions"):
            part_map = item.get(section, {}) or {}
            for key in ("left_arm", "right_arm"):
                entry = part_map.get(key, {}) or {}
                if entry.get("qpos"):
                    has_qpos = True
                if entry.get("pose"):
                    has_pose = True
    if has_qpos and has_pose:
        return "both"
    if has_pose:
        return "pose"
    if has_qpos:
        return "qpos"
    return "unknown"


def build_quality_flags(report):
    flags = []
    state_summary = ((report.get("alignment_summary", {}) or {}).get("state_delta_ms_abs", {}) or {})
    action_summary = ((report.get("alignment_summary", {}) or {}).get("action_delta_ms_abs", {}) or {})
    modes = report.get("alignment_modes", {}) or {}
    state_modes = modes.get("state", {}) or {}
    action_modes = modes.get("action", {}) or {}
    sample_count = int((report.get("episode", {}) or {}).get("num_samples", 0) or 0)

    state_p90 = state_summary.get("p90")
    action_p90 = action_summary.get("p90")
    state_p99 = state_summary.get("p99")
    action_p99 = action_summary.get("p99")

    nearest_state_ratio = (
        float(state_modes.get("nearest_fallback", 0)) / sample_count if sample_count > 0 else 0.0
    )
    nearest_action_ratio = (
        float(action_modes.get("nearest_fallback", 0)) / sample_count if sample_count > 0 else 0.0
    )
    nearest_ratio = max(nearest_state_ratio, nearest_action_ratio)

    if state_p90 is not None and action_p90 is not None and state_p90 <= 10.0 and action_p90 <= 10.0:
        flags.append("PASS: state/action alignment P90 is within 10 ms on the host monotonic timeline.")
    else:
        flags.append("WARN: state/action alignment P90 exceeds 10 ms or is unavailable.")

    if state_p99 is not None and action_p99 is not None and (state_p99 > 25.0 or action_p99 > 25.0):
        flags.append(
            f"WARN: tail alignment jitter is elevated (state P99={state_p99} ms, action P99={action_p99} ms)."
        )

    if nearest_ratio > 0.10:
        flags.append(
            f"WARN: nearest_fallback ratio is high ({round(nearest_ratio * 100.0, 2)}%), inspect network/control jitter."
        )
    elif nearest_ratio > 0.0:
        flags.append(
            f"INFO: nearest_fallback is present ({round(nearest_ratio * 100.0, 2)}%); most samples still aligned by exact/linear methods."
        )

    retention = report.get("retention", {}) or {}
    if retention.get("num_total_attempted_samples") is None:
        flags.append("INFO: current dataset schema does not persist attempted/drop counts; retention is not directly auditable offline.")

    camera_summary = ((report.get("alignment_summary", {}) or {}).get("camera_delta_ms_abs", {}) or {})
    for camera_name, stats in camera_summary.items():
        p99 = (stats or {}).get("p99")
        if p99 is not None and p99 > 30.0:
            flags.append(f"WARN: {camera_name} camera tail residual is elevated (P99={p99} ms).")

    return flags


def load_episode_json(episode_dir):
    data_json_path = os.path.join(episode_dir, "data.json")
    if not os.path.isfile(data_json_path):
        raise FileNotFoundError(f"data.json not found: {data_json_path}")
    with open(data_json_path, "r", encoding="utf-8") as f:
        return json.load(f), data_json_path


def normalize_episode_dir(input_path):
    abs_path = os.path.abspath(input_path)
    if os.path.isfile(abs_path):
        if os.path.basename(abs_path) != "data.json":
            raise ValueError(f"Expected episode directory or data.json path, got file: {abs_path}")
        return os.path.dirname(abs_path)
    return abs_path


def write_csv(csv_path, rows, camera_names):
    fieldnames = [
        "idx",
        "primary_camera_name",
        "sample_monotonic_ns",
        "sample_wall_time_ns",
        "sample_wall_time_iso8601",
        "teleop_input_perf_counter_ns",
        "state_monotonic_ns",
        "state_delta_ms",
        "state_abs_delta_ms",
        "state_effective_abs_delta_ms",
        "state_support_span_ms",
        "state_support_prev_delta_ms",
        "state_support_next_delta_ms",
        "state_support_source_count",
        "state_interpolation_mode",
        "action_monotonic_ns",
        "action_delta_ms",
        "action_abs_delta_ms",
        "action_effective_abs_delta_ms",
        "action_support_span_ms",
        "action_support_prev_delta_ms",
        "action_support_next_delta_ms",
        "action_support_source_count",
        "action_interpolation_mode",
    ]
    for camera_name in camera_names:
        fieldnames.extend(
            [
                f"{camera_name}_camera_monotonic_ns",
                f"{camera_name}_camera_delta_ms",
                f"{camera_name}_camera_abs_delta_ms",
                f"{camera_name}_frame_seq",
                f"{camera_name}_file",
            ]
        )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(md_path, report):
    episode = report.get("episode", {}) or {}
    align = report.get("alignment_summary", {}) or {}
    modes = report.get("alignment_modes", {}) or {}
    retention = report.get("retention", {}) or {}
    primary_counts = report.get("primary_camera_counts", {}) or {}
    camera_presence = report.get("camera_presence_counts", {}) or {}
    flags = report.get("quality_flags", []) or []

    def fmt_summary(stats):
        if not stats or stats.get("count", 0) == 0:
            return "N/A"
        return (
            f"P50 {stats['p50']} ms, P90 {stats['p90']} ms, "
            f"P99 {stats['p99']} ms, max {stats['max']} ms"
        )

    lines = []
    lines.append(f"# Alignment Report: {episode.get('episode_name', 'episode')}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Samples saved: {episode.get('num_samples', 0)}")
    lines.append(f"- Duration: {episode.get('duration_sec') if episode.get('duration_sec') is not None else 'N/A'} s")
    lines.append(f"- Record frequency target: {episode.get('record_frequency_target_hz', 'N/A')} Hz")
    lines.append(f"- Arm representation: {episode.get('arm_repr_mode', 'unknown')}")
    lines.append(f"- Primary camera policy (runtime): {episode.get('primary_camera_policy', 'unknown')}")
    if retention.get("num_total_attempted_samples") is not None:
        lines.append(
            f"- Retention: {retention.get('num_saved_samples')} / {retention.get('num_total_attempted_samples')} "
            f"({retention.get('save_ratio')})"
        )
    else:
        lines.append("- Retention: unavailable in current dataset schema")
    lines.append("")
    lines.append("## Alignment Quality")
    metric_note = align.get("metric_note")
    if metric_note:
        lines.append(f"- Metric note: {metric_note}")
    lines.append(f"- State support max abs delta: {fmt_summary(align.get('state_delta_ms_abs'))}")
    lines.append(f"- Action support max abs delta: {fmt_summary(align.get('action_delta_ms_abs'))}")
    lines.append(f"- State support span: {fmt_summary(align.get('state_support_span_ms'))}")
    lines.append(f"- Action support span: {fmt_summary(align.get('action_support_span_ms'))}")
    for camera_name, stats in (align.get("camera_delta_ms_abs", {}) or {}).items():
        lines.append(f"- {camera_name} camera residual abs: {fmt_summary(stats)}")
    lines.append("")
    lines.append("## Alignment Modes")
    for kind in ("state", "action"):
        kind_modes = modes.get(kind, {}) or {}
        mode_parts = [f"{mode} {count}" for mode, count in sorted(kind_modes.items())]
        lines.append(f"- {kind}: {', '.join(mode_parts) if mode_parts else 'N/A'}")
    lines.append("")
    lines.append("## Camera Coverage")
    for camera_name, count in sorted(camera_presence.items()):
        lines.append(f"- {camera_name}: {count}")
    lines.append("")
    lines.append("## Primary Camera Counts")
    for camera_name, count in sorted(primary_counts.items()):
        lines.append(f"- {camera_name}: {count}")
    lines.append("")
    lines.append("## Flags")
    for flag in flags:
        lines.append(f"- {flag}")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate an auditable alignment report for one recorded episode.")
    parser.add_argument("episode_path", help="Path to episode_xxxx directory or its data.json")
    parser.add_argument(
        "--output-prefix",
        default="alignment_report",
        help="Output prefix inside the episode directory. Generates <prefix>.json/.md and <prefix>_samples.csv",
    )
    args = parser.parse_args()

    episode_dir = normalize_episode_dir(args.episode_path)
    if not os.path.isdir(episode_dir):
        raise FileNotFoundError(f"episode directory not found: {episode_dir}")

    episode_name = os.path.basename(episode_dir.rstrip("/"))
    if not episode_name.startswith("episode_"):
        raise ValueError(f"Expected episode_xxxx directory, got: {episode_name}")

    episode_json, data_json_path = load_episode_json(episode_dir)
    info = episode_json.get("info", {}) or {}
    text = episode_json.get("text", {}) or {}
    samples = episode_json.get("data", []) or []

    sample_rows = []
    state_deltas_ns = []
    action_deltas_ns = []
    state_support_spans_ns = []
    action_support_spans_ns = []
    camera_deltas_ns = defaultdict(list)
    primary_camera_counts = Counter()
    camera_presence_counts = Counter()
    state_mode_counts = Counter()
    action_mode_counts = Counter()
    camera_name_set = set()
    sample_monotonic_values = []
    sample_wall_values = []

    for item in samples:
        timestamps = item.get("timestamps", {}) or {}
        cameras = timestamps.get("camera", {}) or {}
        sample_monotonic_ns = timestamps.get("sample_monotonic_ns")
        sample_wall_time_ns = timestamps.get("sample_wall_time_ns")
        teleop_input_perf_counter_ns = timestamps.get("teleop_input_perf_counter_ns")
        primary_camera_name = canonical_camera_name(timestamps.get("primary_camera_name"))

        if sample_monotonic_ns is not None:
            sample_monotonic_values.append(int(sample_monotonic_ns))
        if sample_wall_time_ns is not None:
            sample_wall_values.append(int(sample_wall_time_ns))
        if primary_camera_name:
            primary_camera_counts[primary_camera_name] += 1

        state_meta = timestamps.get("state", {}) or {}
        action_meta = timestamps.get("action", {}) or {}
        state_metrics = extract_alignment_metrics(state_meta)
        action_metrics = extract_alignment_metrics(action_meta)
        state_delta_ns = state_metrics["raw_delta_ns"]
        action_delta_ns = action_metrics["raw_delta_ns"]
        state_mode = state_meta.get("interpolation_mode", "unknown")
        action_mode = action_meta.get("interpolation_mode", "unknown")

        if state_metrics["effective_abs_delta_ns"] is not None:
            state_deltas_ns.append(int(state_metrics["effective_abs_delta_ns"]))
        if action_metrics["effective_abs_delta_ns"] is not None:
            action_deltas_ns.append(int(action_metrics["effective_abs_delta_ns"]))
        if state_metrics["support_span_ns"] is not None:
            state_support_spans_ns.append(int(state_metrics["support_span_ns"]))
        if action_metrics["support_span_ns"] is not None:
            action_support_spans_ns.append(int(action_metrics["support_span_ns"]))
        state_mode_counts[state_mode] += 1
        action_mode_counts[action_mode] += 1

        row = {
            "idx": item.get("idx"),
            "primary_camera_name": primary_camera_name,
            "sample_monotonic_ns": sample_monotonic_ns,
            "sample_wall_time_ns": sample_wall_time_ns,
            "sample_wall_time_iso8601": ns_to_iso8601(sample_wall_time_ns),
            "teleop_input_perf_counter_ns": teleop_input_perf_counter_ns,
            "state_monotonic_ns": state_meta.get("host_monotonic_ns"),
            "state_delta_ms": round(float(state_delta_ns) / 1e6, 3) if state_delta_ns is not None else None,
            "state_abs_delta_ms": round(abs(float(state_delta_ns)) / 1e6, 3) if state_delta_ns is not None else None,
            "state_effective_abs_delta_ms": round(float(state_metrics["effective_abs_delta_ns"]) / 1e6, 3) if state_metrics["effective_abs_delta_ns"] is not None else None,
            "state_support_span_ms": round(float(state_metrics["support_span_ns"]) / 1e6, 3) if state_metrics["support_span_ns"] is not None else None,
            "state_support_prev_delta_ms": round(float(state_metrics["support_prev_delta_ns"]) / 1e6, 3) if state_metrics["support_prev_delta_ns"] is not None else None,
            "state_support_next_delta_ms": round(float(state_metrics["support_next_delta_ns"]) / 1e6, 3) if state_metrics["support_next_delta_ns"] is not None else None,
            "state_support_source_count": state_metrics["support_source_count"],
            "state_interpolation_mode": state_mode,
            "action_monotonic_ns": action_meta.get("host_monotonic_ns"),
            "action_delta_ms": round(float(action_delta_ns) / 1e6, 3) if action_delta_ns is not None else None,
            "action_abs_delta_ms": round(abs(float(action_delta_ns)) / 1e6, 3) if action_delta_ns is not None else None,
            "action_effective_abs_delta_ms": round(float(action_metrics["effective_abs_delta_ns"]) / 1e6, 3) if action_metrics["effective_abs_delta_ns"] is not None else None,
            "action_support_span_ms": round(float(action_metrics["support_span_ns"]) / 1e6, 3) if action_metrics["support_span_ns"] is not None else None,
            "action_support_prev_delta_ms": round(float(action_metrics["support_prev_delta_ns"]) / 1e6, 3) if action_metrics["support_prev_delta_ns"] is not None else None,
            "action_support_next_delta_ms": round(float(action_metrics["support_next_delta_ns"]) / 1e6, 3) if action_metrics["support_next_delta_ns"] is not None else None,
            "action_support_source_count": action_metrics["support_source_count"],
            "action_interpolation_mode": action_mode,
        }

        color_files = item.get("colors", {}) or {}
        for camera_key, camera_meta in cameras.items():
            canonical_name = canonical_camera_name(camera_key)
            camera_name_set.add(canonical_name)
            camera_presence_counts[canonical_name] += 1
            cam_delta_ns = camera_meta.get("delta_to_sample_ns")
            if cam_delta_ns is not None:
                camera_deltas_ns[canonical_name].append(int(cam_delta_ns))
            row[f"{canonical_name}_camera_monotonic_ns"] = camera_meta.get("host_recv_monotonic_ns") or camera_meta.get("host_monotonic_ns")
            row[f"{canonical_name}_camera_delta_ms"] = round(float(cam_delta_ns) / 1e6, 3) if cam_delta_ns is not None else None
            row[f"{canonical_name}_camera_abs_delta_ms"] = round(abs(float(cam_delta_ns)) / 1e6, 3) if cam_delta_ns is not None else None
            row[f"{canonical_name}_frame_seq"] = camera_meta.get("frame_seq")
            row[f"{canonical_name}_file"] = color_files.get(camera_key) or color_files.get(canonical_name)

        for color_key, file_name in color_files.items():
            canonical_name = canonical_camera_name(color_key)
            camera_name_set.add(canonical_name)
            row.setdefault(f"{canonical_name}_file", file_name)

        sample_rows.append(row)

    sample_monotonic_values.sort()
    sample_wall_values.sort()
    duration_sec = None
    if len(sample_monotonic_values) >= 2:
        duration_sec = round((sample_monotonic_values[-1] - sample_monotonic_values[0]) / 1e9, 3)

    camera_names = sorted(camera_name_set)
    episode_report = {
        "episode": {
            "episode_name": episode_name,
            "episode_dir": episode_dir,
            "data_json_path": data_json_path,
            "task_name": os.path.basename(os.path.dirname(episode_dir)),
            "task_goal": text.get("goal"),
            "task_desc": text.get("desc"),
            "task_steps": text.get("steps"),
            "num_samples": len(samples),
            "duration_sec": duration_sec,
            "record_frequency_target_hz": ((info.get("image", {}) or {}).get("fps")),
            "record_image_size": {
                "width": ((info.get("image", {}) or {}).get("width")),
                "height": ((info.get("image", {}) or {}).get("height")),
            },
            "camera_names": camera_names,
            "primary_camera_policy": "head > left_wrist > right_wrist",
            "arm_repr_mode": infer_arm_repr_mode(samples),
            "first_sample_wall_time_ns": sample_wall_values[0] if sample_wall_values else None,
            "last_sample_wall_time_ns": sample_wall_values[-1] if sample_wall_values else None,
            "first_sample_wall_time_iso8601": ns_to_iso8601(sample_wall_values[0]) if sample_wall_values else None,
            "last_sample_wall_time_iso8601": ns_to_iso8601(sample_wall_values[-1]) if sample_wall_values else None,
            "first_sample_monotonic_ns": sample_monotonic_values[0] if sample_monotonic_values else None,
            "last_sample_monotonic_ns": sample_monotonic_values[-1] if sample_monotonic_values else None,
        },
        "alignment_summary": {
            "metric_note": "state/action delta uses support_max_abs_delta_ns when available; for linear interpolation this is the larger absolute distance from sample time to the two support timestamps.",
            "state_delta_ms_abs": summarize_ms_abs(state_deltas_ns),
            "action_delta_ms_abs": summarize_ms_abs(action_deltas_ns),
            "state_support_span_ms": summarize_ms(state_support_spans_ns),
            "action_support_span_ms": summarize_ms(action_support_spans_ns),
            "camera_delta_ms_abs": {
                camera_name: summarize_ms_abs(values)
                for camera_name, values in sorted(camera_deltas_ns.items())
            },
        },
        "alignment_modes": {
            "state": dict(sorted(state_mode_counts.items())),
            "action": dict(sorted(action_mode_counts.items())),
        },
        "primary_camera_counts": dict(sorted(primary_camera_counts.items())),
        "camera_presence_counts": dict(sorted(camera_presence_counts.items())),
        "retention": {
            "num_total_attempted_samples": None,
            "num_saved_samples": len(samples),
            "num_dropped_samples": None,
            "save_ratio": None,
            "note": "Current dataset schema stores saved samples only. Attempted/drop counts are not persisted offline.",
        },
    }
    episode_report["quality_flags"] = build_quality_flags(episode_report)

    output_prefix = os.path.join(episode_dir, args.output_prefix)
    report_json_path = f"{output_prefix}.json"
    report_md_path = f"{output_prefix}.md"
    report_csv_path = f"{output_prefix}_samples.csv"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(episode_report, f, ensure_ascii=False, indent=2)
    write_markdown(report_md_path, episode_report)
    write_csv(report_csv_path, sample_rows, camera_names)

    print(report_json_path)
    print(report_md_path)
    print(report_csv_path)


if __name__ == "__main__":
    main()
