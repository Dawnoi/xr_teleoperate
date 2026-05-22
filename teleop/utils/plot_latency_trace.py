import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


TOP_LEVEL_LABELS = {
    "recv_to_pub_ms": "Receive -> DDS Publish",
    "pub_to_exec_ms": "DDS Publish -> Motion Detected",
    "recv_to_exec_ms": "Receive -> Motion Detected",
    "fetch_to_exec_ms": "XR Fetch Start -> Motion Detected",
}

TOP_LEVEL_COLORS = {
    "recv_to_pub_ms": "#4C78A8",
    "pub_to_exec_ms": "#F58518",
    "recv_to_exec_ms": "#54A24B",
    "fetch_to_exec_ms": "#B279A2",
    "timeout": "#E45756",
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

TREND_KEYS = [
    ("tele_fetch_ms", "tele_fetch", "#B279A2"),
    ("ik_ms", "ik", "#E45756"),
    ("enqueue_to_publish_ms", "queue_total", "#4C78A8"),
    ("dds_write_ms", "dds_write", "#1F77B4"),
    ("pub_to_exec_ms", "pub_to_exec", "#F58518"),
    ("unaccounted_pre_publish_ms", "unknown_pre_pub", "#BAB0AC"),
]


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

    for key in ["recv_to_pub_ms", "pub_to_exec_ms", "recv_to_exec_ms", "fetch_to_exec_ms"]:
        lines.append(_fmt_stats(TOP_LEVEL_LABELS[key], _stats(_metric_array(completed, key))))

    lines.extend(["", "Breakdown:"])
    for key, label, _ in SEGMENT_SPECS:
        values = np.array([_segment_value(r, key) for r in completed], dtype=float)
        lines.append(_fmt_stats(label, _stats(values)))

    q_delta = _metric_array(completed, "q_delta_trigger")
    dq_peak = _metric_array(completed, "dq_peak_trigger")
    cmd_delta = _metric_array(completed, "max_command_delta")
    lines.extend(["", "Trigger / command magnitude:"])
    lines.append(_fmt_stats("q_delta_trigger", _stats(q_delta), unit="rad"))
    lines.append(_fmt_stats("dq_peak_trigger", _stats(dq_peak), unit="rad/s"))
    lines.append(_fmt_stats("max_command_delta", _stats(cmd_delta), unit="rad"))
    return "\n".join(lines)



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
        ax.barh(y, values, left=left, color=color, alpha=0.9, label=label)
        left += values

    for idx, record in enumerate(records):
        recv_to_pub = float(record.get("recv_to_pub_ms") or 0.0)
        recv_to_exec = float(record.get("recv_to_exec_ms") or 0.0)
        ax.vlines(recv_to_pub, idx - 0.38, idx + 0.38, color="black", linewidth=1.1)
        ax.text(
            recv_to_exec + 0.5,
            idx,
            f"pub {recv_to_pub:.1f} / exec {recv_to_exec:.1f} ms",
            va="center",
            fontsize=8,
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"seq {r['seq']}" for r in records])
    ax.set_xlabel("ms (0 = tele_data ready / receive time)")
    ax.set_title("Recent breakdown timeline (tele_fetch is left of 0, publish marker is black line)")
    ax.grid(True, axis="x", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower right", ncol=2)



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
    ax.bar(x - width / 2, avg_vals, width=width, color=colors, alpha=0.85, label="avg")
    ax.bar(x + width / 2, p95_vals, width=width, color=colors, alpha=0.35, label="p95")
    ax.scatter(x, max_vals, marker="x", s=48, color="black", label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("ms")
    ax.set_title("Per-segment statistics")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")



def _plot_recent_segment_trends(ax, records):
    if not records:
        ax.axis("off")
        return
    seq = np.array([r["seq"] for r in records], dtype=int)
    for key, label, color in TREND_KEYS:
        values = np.array([_segment_value(r, key) for r in records], dtype=float)
        ax.plot(seq, values, marker="o", ms=3, lw=1.4, label=label, color=color)
    ax.set_xlabel("seq")
    ax.set_ylabel("ms")
    ax.set_title("Recent key segment trends")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="upper right")



def _plot_scatter(ax, records):
    scatter_records = [r for r in records if r.get("max_command_delta") is not None and r.get("recv_to_exec_ms") is not None]
    if not scatter_records:
        ax.axis("off")
        return
    x = np.array([float(r["max_command_delta"]) for r in scatter_records], dtype=float)
    y = np.array([float(r["recv_to_exec_ms"]) for r in scatter_records], dtype=float)
    color = np.array([_segment_value(r, "ik_ms") for r in scatter_records], dtype=float)
    size = 28 + 120 * np.array([_segment_value(r, "pub_to_exec_ms") for r in scatter_records], dtype=float) / max(np.max([_segment_value(r, "pub_to_exec_ms") for r in scatter_records]), 1e-6)
    sc = ax.scatter(x, y, c=color, s=size, cmap="viridis", alpha=0.82, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("ik_ms")
    ax.set_xlabel("max_command_delta (rad)")
    ax.set_ylabel("recv_to_exec_ms")
    ax.set_title("Command magnitude vs end-to-end latency\n(color=IK, size=pub_to_exec)")
    ax.grid(True, alpha=0.25)



def _plot_dominant_stage(ax, records):
    keys = [key for key, _, _ in SEGMENT_SPECS]
    labels = {key: label for key, label, _ in SEGMENT_SPECS}
    colors = {key: color for key, _, color in SEGMENT_SPECS}
    counts = {key: 0 for key in keys}
    for record in records:
        values = {key: _segment_value(record, key) for key in keys}
        dominant = max(values, key=values.get)
        counts[dominant] += 1
    non_zero = [(key, count) for key, count in counts.items() if count > 0]
    if not non_zero:
        ax.axis("off")
        return
    x_labels = [labels[key] for key, _ in non_zero]
    x = np.arange(len(non_zero))
    y = [count for _, count in non_zero]
    ax.bar(x, y, color=[colors[key] for key, _ in non_zero], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Dominant segment count per trace")
    ax.grid(True, axis="y", alpha=0.25)



def plot_records(records, out_path: Path, tail: int, hist_bins: int):
    completed = completed_records(records)
    if not completed:
        raise ValueError("No completed records found in trace file.")

    seq = np.array([r["seq"] for r in completed], dtype=int)
    recv_to_pub = _metric_array(completed, "recv_to_pub_ms")
    pub_to_exec = _metric_array(completed, "pub_to_exec_ms")
    recv_to_exec = _metric_array(completed, "recv_to_exec_ms")
    fetch_seq, fetch_to_exec = _metric_pairs(completed, "fetch_to_exec_ms")

    tail_completed = completed[-tail:]

    fig = plt.figure(figsize=(22, 18), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.15, 1.05])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(seq, recv_to_pub, marker="o", ms=3, lw=1.4, label=TOP_LEVEL_LABELS["recv_to_pub_ms"], color=TOP_LEVEL_COLORS["recv_to_pub_ms"])
    ax.plot(seq, pub_to_exec, marker="o", ms=3, lw=1.4, label=TOP_LEVEL_LABELS["pub_to_exec_ms"], color=TOP_LEVEL_COLORS["pub_to_exec_ms"])
    ax.plot(seq, recv_to_exec, marker="o", ms=3, lw=1.7, label=TOP_LEVEL_LABELS["recv_to_exec_ms"], color=TOP_LEVEL_COLORS["recv_to_exec_ms"])
    if fetch_to_exec.size:
        ax.plot(fetch_seq, fetch_to_exec, marker="o", ms=3, lw=1.3, linestyle="--", label=TOP_LEVEL_LABELS["fetch_to_exec_ms"], color=TOP_LEVEL_COLORS["fetch_to_exec_ms"])
    ax.set_title("Top-level latency curves")
    ax.set_xlabel("seq")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("Summary")
    ax.text(0.0, 1.0, _summary_text(records), va="top", ha="left", family="monospace", fontsize=9.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(recv_to_pub, bins=hist_bins, alpha=0.58, label=TOP_LEVEL_LABELS["recv_to_pub_ms"], color=TOP_LEVEL_COLORS["recv_to_pub_ms"])
    ax.hist(pub_to_exec, bins=hist_bins, alpha=0.58, label=TOP_LEVEL_LABELS["pub_to_exec_ms"], color=TOP_LEVEL_COLORS["pub_to_exec_ms"])
    ax.hist(recv_to_exec, bins=hist_bins, alpha=0.42, label=TOP_LEVEL_LABELS["recv_to_exec_ms"], color=TOP_LEVEL_COLORS["recv_to_exec_ms"])
    if fetch_to_exec.size:
        ax.hist(fetch_to_exec, bins=hist_bins, alpha=0.28, label=TOP_LEVEL_LABELS["fetch_to_exec_ms"], color=TOP_LEVEL_COLORS["fetch_to_exec_ms"])
    ax.set_title("Latency distributions")
    ax.set_xlabel("ms")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    _plot_recent_breakdown(ax, tail_completed)

    ax = fig.add_subplot(gs[2, 0])
    _plot_stage_stats(ax, completed)

    ax = fig.add_subplot(gs[2, 1])
    _plot_recent_segment_trends(ax, tail_completed)

    ax = fig.add_subplot(gs[3, 0])
    _plot_scatter(ax, completed)

    ax = fig.add_subplot(gs[3, 1])
    _plot_dominant_stage(ax, completed)

    fig.suptitle("xr_teleoperate latency visualization (fine-grained breakdown)", fontsize=18)
    fig.savefig(out_path, dpi=180)
    print(f"saved plot to: {out_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_path", type=str, help="Path to latency trace jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--tail", type=int, default=20, help="How many recent completed traces to show in recent plots")
    parser.add_argument("--hist-bins", type=int, default=20, help="Histogram bins for latency distribution plots")
    args = parser.parse_args()

    trace_path = Path(args.trace_path).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve() if args.output else trace_path.with_suffix(".png")
    records = load_records(trace_path)
    plot_records(records, out_path, tail=max(1, args.tail), hist_bins=max(5, args.hist_bins))


if __name__ == "__main__":
    main()
