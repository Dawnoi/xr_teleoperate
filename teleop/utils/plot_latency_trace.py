import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


TOP_LEVEL_ORDER = [
    "recv_to_pub_ms",
    "pub_to_exec_ms",
    "recv_to_exec_ms",
    "fetch_to_exec_ms",
]

TOP_LEVEL_LABELS = {
    "recv_to_pub_ms": "Receive -> DDS Publish",
    "pub_to_exec_ms": "DDS Publish -> Motion Detected",
    "recv_to_exec_ms": "Receive -> Motion Detected",
    "fetch_to_exec_ms": "XR Fetch Start -> Motion Detected",
}

TOP_LEVEL_STYLES = {
    "recv_to_pub_ms": {"color": "#4C78A8", "marker": "o"},
    "pub_to_exec_ms": {"color": "#F58518", "marker": "s"},
    "recv_to_exec_ms": {"color": "#54A24B", "marker": "^"},
    "fetch_to_exec_ms": {"color": "#B279A2", "marker": "D"},
}

SEGMENT_SPECS = [
    ("tele_fetch_ms", "tele_fetch", "#B279A2"),
    ("takeover_logic_ms", "takeover", "#9D755D"),
    ("base_control_ms", "base_ctrl", "#72B7B2"),
    ("ik_ms", "ik", "#E45756"),
    ("safety_ms", "safety", "#F2CF5B"),
    ("gravity_ms", "gravity", "#54A24B"),
    ("controller_wait_ms", "ctrl_wait", "#4C78A8"),
    ("dds_write_ms", "dds_write", "#1F77B4"),
    ("unaccounted_pre_publish_ms", "unknown_pre_pub", "#BAB0AC"),
    ("pub_to_exec_ms", "pub_to_exec", "#F58518"),
]

BASE_SEGMENT_SPECS = [
    ("base_move_ms", "base_move", "#4C78A8"),
    ("base_height_ms", "base_height", "#F58518"),
    ("base_misc_ms", "base_misc", "#BAB0AC"),
]

TREND_KEYS = [
    ("base_control_ms", "base_ctrl", "#72B7B2"),
    ("base_move_ms", "base_move", "#4C78A8"),
    ("base_height_ms", "base_height", "#F58518"),
    ("ik_ms", "ik", "#E45756"),
    ("enqueue_to_publish_ms", "queue_total", "#7F3C8D"),
    ("pub_to_exec_ms", "pub_to_exec", "#11A579"),
]

MODE_COLORS = {
    "g1d_agv": "#E45756",
    "g1d_agv_async": "#C44E52",
    "loco": "#4C78A8",
    "none": "#9D9D9D",
}


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records



def completed_records(records):
    return [r for r in records if r.get("status") == "completed"]



def _metric_array(records, key):
    return np.array([float(r[key]) for r in records if r.get(key) is not None], dtype=float)



def _metric_pairs(records, key):
    seq = []
    values = []
    for record in records:
        if record.get(key) is None:
            continue
        seq.append(int(record["seq"]))
        values.append(float(record[key]))
    return np.asarray(seq, dtype=int), np.asarray(values, dtype=float)



def _segment_value(record, key):
    if key == "controller_wait_ms":
        enqueue = record.get("enqueue_to_publish_ms")
        dds = record.get("dds_write_ms")
        if enqueue is not None:
            enqueue = max(0.0, float(enqueue))
            if dds is not None:
                dds = min(max(0.0, float(dds)), enqueue)
                return max(0.0, enqueue - dds)
            return enqueue
        return max(0.0, float(record.get("controller_wait_ms") or 0.0))
    if key == "dds_write_ms":
        enqueue = record.get("enqueue_to_publish_ms")
        dds = record.get("dds_write_ms")
        if enqueue is not None and dds is not None:
            return min(max(0.0, float(dds)), max(0.0, float(enqueue)))
    return max(0.0, float(record.get(key) or 0.0))



def _stats(arr):
    if arr.size == 0:
        return None
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }



def _fmt_stats(name, stats, unit="ms"):
    if stats is None:
        return f"{name:<22} n/a"
    return (
        f"{name:<22} avg={stats['avg']:>6.2f}  p50={stats['p50']:>6.2f}  "
        f"p95={stats['p95']:>6.2f}  max={stats['max']:>6.2f} {unit}"
    )



def _rolling_stat(values, window=25):
    if values.size == 0:
        return values
    if values.size < 3:
        return values.copy()
    radius = max(1, min(window, values.size) // 2)
    out = np.empty_like(values)
    for idx in range(values.size):
        lo = max(0, idx - radius)
        hi = min(values.size, idx + radius + 1)
        out[idx] = float(np.median(values[lo:hi]))
    return out



def _safe_corr(x, y):
    if x.size < 2 or y.size < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])



def _combined_top_values(completed):
    arrays = []
    for key in TOP_LEVEL_ORDER:
        arr = _metric_array(completed, key)
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return np.array([], dtype=float)
    return np.concatenate(arrays)



def _top_latency_records(completed, top_k=20, key="recv_to_exec_ms"):
    valid = [r for r in completed if r.get(key) is not None]
    valid.sort(key=lambda item: float(item.get(key) or 0.0), reverse=True)
    return valid[:max(1, int(top_k))]



def _base_mode_counts(completed):
    counts = {}
    for record in completed:
        mode = record.get("base_control_mode") or "none"
        counts[mode] = counts.get(mode, 0) + 1
    return counts



def _base_inference_lines(completed):
    if not completed:
        return ["base ctrl inference: no completed traces"]

    active = [
        r for r in completed
        if _segment_value(r, "base_control_ms") > 0.05 or (r.get("base_control_mode") not in (None, "none"))
    ]
    if not active:
        return ["base ctrl inference: base path inactive in traced samples"]

    move = np.array([_segment_value(r, "base_move_ms") for r in active], dtype=float)
    height = np.array([_segment_value(r, "base_height_ms") for r in active], dtype=float)
    misc = np.array([_segment_value(r, "base_misc_ms") for r in active], dtype=float)
    total = np.array([_segment_value(r, "base_control_ms") for r in active], dtype=float)
    recv_to_pub = np.array([float(r.get("recv_to_pub_ms") or 0.0) for r in active], dtype=float)

    total_mean = max(1e-6, float(np.mean(total)))
    move_share = float(np.mean(move)) / total_mean
    height_share = float(np.mean(height)) / total_mean
    misc_share = float(np.mean(misc)) / total_mean
    bridge_share = float(np.mean(move + height)) / total_mean
    corr = _safe_corr(total, recv_to_pub)

    lines = [
        f"base active traces: {len(active)}/{len(completed)}",
        f"base share avg: move={move_share * 100:.0f}% height={height_share * 100:.0f}% misc={misc_share * 100:.0f}%",
    ]
    if corr is not None:
        lines.append(f"corr(base_ctrl, recv->pub) = {corr:.2f}")

    active_modes = sorted({(r.get("base_control_mode") or "none") for r in active})
    if any(mode.startswith("g1d_agv_async") for mode in active_modes):
        lines.append("async agv bridge enabled: base_ctrl now mainly reflects queue/enqueue cost, not blocking RPC wait.")
        async_queue = _metric_array(active, "base_async_queue_avg_ms")
        async_cycle = _metric_array(active, "base_async_cycle_avg_ms")
        async_move = _metric_array(active, "base_async_move_avg_ms")
        async_height = _metric_array(active, "base_async_height_avg_ms")
        if async_queue.size:
            lines.append(f"async queue avg={float(np.mean(async_queue)):.2f} ms")
        if async_cycle.size:
            lines.append(f"async worker cycle avg={float(np.mean(async_cycle)):.2f} ms")
        if async_move.size or async_height.size:
            lines.append(
                f"async worker move/height avg={float(np.mean(async_move)) if async_move.size else 0.0:.2f}/"
                f"{float(np.mean(async_height)) if async_height.size else 0.0:.2f} ms"
            )
        return lines

    if bridge_share >= 0.85:
        if height_share > move_share * 1.2 and height_share > 0.35:
            lines.append("inference: base_ctrl mostly blocked in synchronous AGV bridge, HEIGHT path dominates.")
        elif move_share > 0.5:
            lines.append("inference: base_ctrl mostly blocked in synchronous MOVE/bridge ack.")
        else:
            lines.append("inference: base_ctrl mainly comes from synchronous bridge round-trip.")
    elif misc_share > 0.20:
        lines.append("inference: non-bridge overhead is visible; check host scheduling / Python side waits.")
    else:
        lines.append("inference: base_ctrl is mixed; inspect recent base split timeline for per-trace cause.")
    return lines



def _dominant_segments(record, top_n=4):
    pairs = []
    for key, label, _ in SEGMENT_SPECS:
        pairs.append((label, float(_segment_value(record, key))))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs[:max(1, int(top_n))]



def _summary_text(records):
    completed = completed_records(records)
    timeout = sum(1 for r in records if r.get("status") == "timeout")
    lines = [
        f"total traces: {len(records)}",
        f"completed:    {len(completed)}",
        f"timeout:      {timeout}",
        "",
        "Top-level latency:",
    ]

    for key in TOP_LEVEL_ORDER:
        lines.append(_fmt_stats(TOP_LEVEL_LABELS[key], _stats(_metric_array(completed, key))))

    lines.extend(["", "Breakdown:"])
    for key, label, _ in SEGMENT_SPECS:
        values = np.array([_segment_value(r, key) for r in completed], dtype=float)
        lines.append(_fmt_stats(label, _stats(values)))

    lines.extend(["", "Base ctrl focus:"])
    counts = _base_mode_counts(completed)
    lines.append("modes: " + "  ".join(f"{mode}={counts[mode]}" for mode in sorted(counts)))
    for line in _base_inference_lines(completed):
        lines.append(line)

    top_records = _top_latency_records(completed, top_k=5)
    if top_records:
        lines.extend(["", "Worst traces (top5 recv->exec):"])
        for idx, record in enumerate(top_records, start=1):
            lines.append(
                f"#{idx} seq={record['seq']} recv->exec={float(record.get('recv_to_exec_ms') or 0.0):.2f} "
                f"base={_segment_value(record, 'base_control_ms'):.2f} ik={_segment_value(record, 'ik_ms'):.2f} "
                f"pub->exec={float(record.get('pub_to_exec_ms') or 0.0):.2f}"
            )

    q_delta = _metric_array(completed, "q_delta_trigger")
    dq_peak = _metric_array(completed, "dq_peak_trigger")
    cmd_delta = _metric_array(completed, "max_command_delta")
    lines.extend(["", "Trigger / command magnitude:"])
    lines.append(_fmt_stats("q_delta_trigger", _stats(q_delta), unit="rad"))
    lines.append(_fmt_stats("dq_peak_trigger", _stats(dq_peak), unit="rad/s"))
    lines.append(_fmt_stats("max_command_delta", _stats(cmd_delta), unit="rad"))
    return "\n".join(lines)



def _plot_top_level_full(ax, completed, top_records):
    for key in TOP_LEVEL_ORDER:
        seq, values = _metric_pairs(completed, key)
        if values.size == 0:
            continue
        style = TOP_LEVEL_STYLES[key]
        ax.scatter(seq, values, s=18, alpha=0.55, color=style["color"], marker=style["marker"], label=TOP_LEVEL_LABELS[key])
        ax.plot(seq, _rolling_stat(values, window=31), color=style["color"], linewidth=1.6, alpha=0.95)
    for idx, record in enumerate(top_records, start=1):
        seq = int(record["seq"])
        value = float(record.get("recv_to_exec_ms") or 0.0)
        ax.scatter([seq], [value], s=130, facecolors="none", edgecolors="black", linewidths=1.3, zorder=6)
        ax.text(seq, value + 0.35, str(idx), fontsize=8, ha="center", va="bottom", color="black")
    ax.set_title("Top-level latency (scatter + rolling median, worst traces highlighted)")
    ax.set_xlabel("seq")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)



def _plot_top_level_zoom(ax, completed):
    combined = _combined_top_values(completed)
    if combined.size == 0:
        ax.axis("off")
        return
    lo = float(np.percentile(combined, 10))
    hi = float(np.percentile(combined, 90))
    pad = max(0.8, (hi - lo) * 0.15)
    for key in TOP_LEVEL_ORDER:
        seq, values = _metric_pairs(completed, key)
        if values.size == 0:
            continue
        style = TOP_LEVEL_STYLES[key]
        ax.scatter(seq, values, s=20, alpha=0.70, color=style["color"], marker=style["marker"], label=TOP_LEVEL_LABELS[key])
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title("Top-level latency dense-band zoom (P10-P90)")
    ax.set_xlabel("seq")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)



def _plot_top_level_distribution(ax, completed, hist_bins):
    for key in TOP_LEVEL_ORDER:
        values = _metric_array(completed, key)
        if values.size == 0:
            continue
        style = TOP_LEVEL_STYLES[key]
        ax.hist(values, bins=hist_bins, alpha=0.30, label=TOP_LEVEL_LABELS[key], color=style["color"])
    ax.set_title("Top-level latency distributions")
    ax.set_xlabel("ms")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)



def _plot_recent_breakdown(ax, records):
    if not records:
        ax.axis("off")
        return

    y = np.arange(len(records))
    tele_fetch = np.array([_segment_value(r, "tele_fetch_ms") for r in records], dtype=float)
    ax.barh(y, tele_fetch, left=-tele_fetch, color="#B279A2", alpha=0.85, label="tele_fetch (pre-receive)")

    left = np.zeros(len(records), dtype=float)
    for key, label, color in SEGMENT_SPECS[1:]:
        values = np.array([_segment_value(r, key) for r in records], dtype=float)
        ax.barh(y, values, left=left, color=color, alpha=0.92, label=label)
        left += values

    for idx, record in enumerate(records):
        recv_to_pub = float(record.get("recv_to_pub_ms") or 0.0)
        recv_to_exec = float(record.get("recv_to_exec_ms") or 0.0)
        ax.vlines(recv_to_pub, idx - 0.38, idx + 0.38, color="black", linewidth=1.1)
        ax.text(recv_to_exec + 0.35, idx, f"pub {recv_to_pub:.1f} / exec {recv_to_exec:.1f}", va="center", fontsize=8)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"seq {r['seq']}" for r in records])
    ax.set_xlabel("ms (0 = tele_data ready / receive time)")
    ax.set_title("Recent full breakdown timeline")
    ax.grid(True, axis="x", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower right", ncol=2)



def _plot_recent_base_breakdown(ax, records):
    if not records:
        ax.axis("off")
        return

    y = np.arange(len(records))
    left = np.zeros(len(records), dtype=float)
    for key, label, color in BASE_SEGMENT_SPECS:
        values = np.array([_segment_value(r, key) for r in records], dtype=float)
        ax.barh(y, values, left=left, color=color, alpha=0.9, label=label)
        left += values

    for idx, record in enumerate(records):
        total = _segment_value(record, "base_control_ms")
        mode = record.get("base_control_mode") or "none"
        z_cmd = float(record.get("base_z_cmd") or 0.0)
        ax.text(total + 0.15, idx, f"{total:.1f} ms | {mode} | z={z_cmd:.2f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([f"seq {r['seq']}" for r in records])
    ax.set_xlabel("ms")
    ax.set_title("Recent base_ctrl split (move / height / misc)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)



def _plot_stage_stats(ax, records):
    labels = []
    avg_vals = []
    p95_vals = []
    max_vals = []
    colors = []
    for key, label, color in SEGMENT_SPECS:
        arr = np.array([_segment_value(r, key) for r in records], dtype=float)
        labels.append(label)
        avg_vals.append(float(np.mean(arr)) if arr.size else 0.0)
        p95_vals.append(float(np.percentile(arr, 95)) if arr.size else 0.0)
        max_vals.append(float(np.max(arr)) if arr.size else 0.0)
        colors.append(color)

    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, avg_vals, width=width, color=colors, alpha=0.88, label="avg")
    ax.bar(x + width / 2, p95_vals, width=width, color=colors, alpha=0.34, label="p95")
    ax.scatter(x, max_vals, marker="x", s=48, color="black", label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("ms")
    ax.set_title("Per-segment statistics")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)



def _plot_recent_segment_trends(ax, records):
    if not records:
        ax.axis("off")
        return
    seq = np.array([r["seq"] for r in records], dtype=int)
    for key, label, color in TREND_KEYS:
        values = np.array([_segment_value(r, key) for r in records], dtype=float)
        ax.plot(seq, values, marker="o", ms=3, lw=1.5, label=label, color=color)
    ax.set_xlabel("seq")
    ax.set_ylabel("ms")
    ax.set_title("Recent key segment trends")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="upper right")



def _plot_base_ctrl_scatter(ax, completed):
    max_height = max(1e-6, np.max([_segment_value(r, "base_height_ms") for r in completed]) if completed else 1.0)
    corr_values_x = []
    corr_values_y = []
    modes = sorted({(r.get("base_control_mode") or "none") for r in completed})
    for mode in modes:
        subset = [r for r in completed if (r.get("base_control_mode") or "none") == mode]
        if not subset:
            continue
        x = np.array([_segment_value(r, "base_control_ms") for r in subset], dtype=float)
        y = np.array([float(r.get("recv_to_pub_ms") or 0.0) for r in subset], dtype=float)
        size = 24 + 140 * np.array([_segment_value(r, "base_height_ms") for r in subset], dtype=float) / max_height
        ax.scatter(x, y, s=size, alpha=0.72, color=MODE_COLORS.get(mode, "#999999"), label=f"{mode} ({len(subset)})", edgecolors="none")
        corr_values_x.append(x)
        corr_values_y.append(y)

    all_x = np.concatenate(corr_values_x) if corr_values_x else np.array([], dtype=float)
    all_y = np.concatenate(corr_values_y) if corr_values_y else np.array([], dtype=float)
    corr = _safe_corr(all_x, all_y)
    if all_x.size >= 2 and not np.allclose(all_x, all_x[0]):
        coeff = np.polyfit(all_x, all_y, 1)
        xs = np.linspace(float(np.min(all_x)), float(np.max(all_x)), 100)
        ax.plot(xs, coeff[0] * xs + coeff[1], color="black", linestyle="--", linewidth=1.2, alpha=0.8)

    title = "base_ctrl vs recv->pub"
    if corr is not None:
        title += f" (corr={corr:.2f}, marker size=base_height_ms)"
    ax.set_title(title)
    ax.set_xlabel("base_control_ms")
    ax.set_ylabel("recv_to_pub_ms")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)



def _plot_base_focus_stats(ax, completed):
    labels = ["base_ctrl", "base_move", "base_height", "base_misc"]
    keys = ["base_control_ms", "base_move_ms", "base_height_ms", "base_misc_ms"]
    colors = ["#72B7B2", "#4C78A8", "#F58518", "#BAB0AC"]
    avg_vals = []
    p95_vals = []
    max_vals = []
    for key in keys:
        arr = np.array([_segment_value(r, key) for r in completed], dtype=float)
        avg_vals.append(float(np.mean(arr)) if arr.size else 0.0)
        p95_vals.append(float(np.percentile(arr, 95)) if arr.size else 0.0)
        max_vals.append(float(np.max(arr)) if arr.size else 0.0)

    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, avg_vals, width=width, color=colors, alpha=0.88, label="avg")
    ax.bar(x + width / 2, p95_vals, width=width, color=colors, alpha=0.36, label="p95")
    ax.scatter(x, max_vals, marker="x", s=48, color="black", label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ms")
    ax.set_title("base_ctrl breakdown statistics")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)



def _format_detail_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)



def write_top_latency_report(records, out_path: Path, top_k: int = 20):
    completed = completed_records(records)
    top_records = _top_latency_records(completed, top_k=top_k)
    lines = [
        "# xr_teleoperate latency top-k report",
        "",
        f"- samples: {len(completed)} completed / {len(records)} total",
        f"- ranking metric: recv_to_exec_ms",
        f"- top_k: {len(top_records)}",
        "",
        "## Ranked overview",
        "",
        "| rank | seq | recv->exec | recv->pub | pub->exec | fetch->exec | base | ik | mode | dominant |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for idx, record in enumerate(top_records, start=1):
        dominant = ", ".join(f"{name}:{value:.1f}" for name, value in _dominant_segments(record, top_n=2))
        lines.append(
            f"| {idx} | {record['seq']} | {float(record.get('recv_to_exec_ms') or 0.0):.2f} | "
            f"{float(record.get('recv_to_pub_ms') or 0.0):.2f} | {float(record.get('pub_to_exec_ms') or 0.0):.2f} | "
            f"{float(record.get('fetch_to_exec_ms') or 0.0):.2f} | {_segment_value(record, 'base_control_ms'):.2f} | "
            f"{_segment_value(record, 'ik_ms'):.2f} | {record.get('base_control_mode') or 'none'} | {dominant} |"
        )

    key_order = [
        "seq", "status",
        "recv_to_pub_ms", "pub_to_exec_ms", "recv_to_exec_ms", "fetch_to_exec_ms",
        "tele_fetch_ms", "takeover_logic_ms", "base_control_ms", "base_move_ms", "base_height_ms", "base_misc_ms",
        "base_control_mode", "base_vx_cmd", "base_vy_cmd", "base_wz_cmd", "base_z_cmd",
        "base_async_publish_hz", "base_async_move_avg_ms", "base_async_height_avg_ms",
        "base_async_cycle_avg_ms", "base_async_queue_avg_ms",
        "ik_ms", "safety_ms", "gravity_ms", "enqueue_to_publish_ms", "controller_wait_ms", "dds_write_ms",
        "known_pre_publish_ms", "unaccounted_pre_publish_ms", "known_post_receive_ms", "unaccounted_post_receive_ms",
        "max_command_delta", "left_joint_delta_norm", "right_joint_delta_norm",
        "q_delta_trigger", "dq_peak_trigger",
        "left_arm_enabled", "right_arm_enabled", "home_return_active",
        "left_takeover_rising_edge", "right_takeover_rising_edge",
        "left_zero_takeover_this_frame", "right_zero_takeover_this_frame",
        "t_recv_ns", "t_pub_ns", "t_exec_ns",
    ]
    lines.extend(["", "## Detailed records", ""])
    for idx, record in enumerate(top_records, start=1):
        lines.append(f"### Rank {idx} / seq {record['seq']}")
        lines.append("")
        lines.append("- dominant segments: " + ", ".join(f"{name}={value:.3f} ms" for name, value in _dominant_segments(record, top_n=4)))
        lines.append("- all fields:")
        lines.append("")
        lines.append("| key | value |")
        lines.append("|---|---|")
        seen = set()
        for key in key_order:
            if key in record:
                lines.append(f"| {key} | {_format_detail_value(record[key])} |")
                seen.add(key)
        extra_keys = sorted(k for k in record.keys() if k not in seen)
        for key in extra_keys:
            lines.append(f"| {key} | {_format_detail_value(record[key])} |")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path



def plot_records(records, out_path: Path, tail: int, hist_bins: int, top_k: int = 20):
    completed = completed_records(records)
    if not completed:
        raise ValueError("No completed records found in trace file.")

    tail_completed = completed[-tail:]
    top_records = _top_latency_records(completed, top_k=top_k)

    fig = plt.figure(figsize=(24, 26), constrained_layout=True)
    gs = fig.add_gridspec(5, 2, height_ratios=[1.1, 1.0, 1.15, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    _plot_top_level_full(ax, completed, top_records)

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("Summary")
    ax.text(0.0, 1.0, _summary_text(records), va="top", ha="left", family="monospace", fontsize=9.2)

    ax = fig.add_subplot(gs[1, 0])
    _plot_top_level_zoom(ax, completed)

    ax = fig.add_subplot(gs[1, 1])
    _plot_top_level_distribution(ax, completed, hist_bins)

    ax = fig.add_subplot(gs[2, 0])
    _plot_recent_breakdown(ax, tail_completed)

    ax = fig.add_subplot(gs[2, 1])
    _plot_recent_base_breakdown(ax, tail_completed)

    ax = fig.add_subplot(gs[3, 0])
    _plot_stage_stats(ax, completed)

    ax = fig.add_subplot(gs[3, 1])
    _plot_recent_segment_trends(ax, tail_completed)

    ax = fig.add_subplot(gs[4, 0])
    _plot_base_ctrl_scatter(ax, completed)

    ax = fig.add_subplot(gs[4, 1])
    _plot_base_focus_stats(ax, completed)

    fig.suptitle("xr_teleoperate latency visualization (top-level clarity + base_ctrl deep dive)", fontsize=18)
    fig.savefig(out_path, dpi=180)
    print(f"saved plot to: {out_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_path", type=str, help="Path to latency trace jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--tail", type=int, default=20, help="How many recent completed traces to show in recent plots")
    parser.add_argument("--hist-bins", type=int, default=20, help="Histogram bins for latency distribution plots")
    parser.add_argument("--top-k", type=int, default=20, help="How many worst-latency traces to include in report/highlight")
    parser.add_argument("--top-report", type=str, default=None, help="Optional markdown path for top-k detailed latency report")
    args = parser.parse_args()

    trace_path = Path(args.trace_path).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve() if args.output else trace_path.with_suffix(".png")
    top_report_path = (
        Path(args.top_report).expanduser().resolve()
        if args.top_report
        else trace_path.with_name(trace_path.stem + "_top_latency.md")
    )
    records = load_records(trace_path)
    plot_records(records, out_path, tail=max(1, args.tail), hist_bins=max(5, args.hist_bins), top_k=max(1, args.top_k))
    report_path = write_top_latency_report(records, top_report_path, top_k=max(1, args.top_k))
    print(f"saved top-latency report to: {report_path}")


if __name__ == "__main__":
    main()
