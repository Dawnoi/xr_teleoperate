import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "recv_to_pub_ms": "Receive -> DDS Publish",
    "pub_to_exec_ms": "DDS Publish -> Motion Detected",
    "recv_to_exec_ms": "Receive -> Motion Detected",
}

COLORS = {
    "recv_to_pub_ms": "#4C78A8",
    "pub_to_exec_ms": "#F58518",
    "recv_to_exec_ms": "#54A24B",
    "timeout": "#E45756",
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
    return np.array([r[key] for r in records if r.get(key) is not None], dtype=float)


def _summary_text(records):
    completed = completed_records(records)
    total = len(records)
    done = len(completed)
    timeout = sum(1 for r in records if r.get("status") == "timeout")
    lines = [
        f"total traces: {total}",
        f"completed: {done}",
        f"timeout: {timeout}",
    ]
    if done:
        for key in ["recv_to_pub_ms", "pub_to_exec_ms", "recv_to_exec_ms"]:
            arr = _metric_array(completed, key)
            lines.append(
                f"{LABELS[key]}\n  avg={np.mean(arr):.2f}  p50={np.percentile(arr, 50):.2f}  "
                f"p95={np.percentile(arr, 95):.2f}  max={np.max(arr):.2f} ms"
            )
        q_delta = _metric_array(completed, "q_delta_trigger")
        dq_peak = _metric_array(completed, "dq_peak_trigger")
        if q_delta.size:
            lines.append(
                f"trigger q_delta avg/p95/max={np.mean(q_delta):.4f}/{np.percentile(q_delta,95):.4f}/{np.max(q_delta):.4f} rad"
            )
        if dq_peak.size:
            lines.append(
                f"trigger dq_peak avg/p95/max={np.mean(dq_peak):.4f}/{np.percentile(dq_peak,95):.4f}/{np.max(dq_peak):.4f} rad/s"
            )
    return "\n".join(lines)


def plot_records(records, out_path: Path, tail: int, hist_bins: int):
    completed = completed_records(records)
    if not completed:
        raise ValueError("No completed records found in trace file.")

    seq = np.array([r["seq"] for r in completed], dtype=int)
    recv_to_pub = _metric_array(completed, "recv_to_pub_ms")
    pub_to_exec = _metric_array(completed, "pub_to_exec_ms")
    recv_to_exec = _metric_array(completed, "recv_to_exec_ms")
    q_delta = _metric_array(completed, "q_delta_trigger")
    dq_peak = _metric_array(completed, "dq_peak_trigger")

    tail_records = records[-tail:]
    tail_completed = completed[-tail:]

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.2])

    # 1) overall curves
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(seq, recv_to_pub, marker="o", ms=3, lw=1.5, label=LABELS["recv_to_pub_ms"], color=COLORS["recv_to_pub_ms"])
    ax.plot(seq, pub_to_exec, marker="o", ms=3, lw=1.5, label=LABELS["pub_to_exec_ms"], color=COLORS["pub_to_exec_ms"])
    ax.plot(seq, recv_to_exec, marker="o", ms=3, lw=1.8, label=LABELS["recv_to_exec_ms"], color=COLORS["recv_to_exec_ms"])
    ax.set_title("Per-trace latency curves")
    ax.set_xlabel("seq")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # 2) summary text
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("Summary")
    ax.text(
        0.0,
        1.0,
        _summary_text(records),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    # 3) histogram
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(recv_to_pub, bins=hist_bins, alpha=0.6, label=LABELS["recv_to_pub_ms"], color=COLORS["recv_to_pub_ms"])
    ax.hist(pub_to_exec, bins=hist_bins, alpha=0.6, label=LABELS["pub_to_exec_ms"], color=COLORS["pub_to_exec_ms"])
    ax.hist(recv_to_exec, bins=hist_bins, alpha=0.45, label=LABELS["recv_to_exec_ms"], color=COLORS["recv_to_exec_ms"])
    ax.set_title("Latency distributions")
    ax.set_xlabel("ms")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # 4) scatter: trigger magnitudes vs end-to-end latency
    ax = fig.add_subplot(gs[1, 1])
    if q_delta.size and dq_peak.size:
        size = 20 + 120 * np.clip(q_delta / max(np.max(q_delta), 1e-6), 0.0, 1.0)
        sc = ax.scatter(q_delta, recv_to_exec, c=dq_peak, s=size, cmap="viridis", alpha=0.8, edgecolors="none")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("dq_peak_trigger (rad/s)")
        ax.set_xlabel("q_delta_trigger (rad)")
        ax.set_ylabel("Receive -> Motion Detected (ms)")
        ax.set_title("Trigger magnitude vs end-to-end latency")
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")

    # 5) boxplot for quick comparison
    ax = fig.add_subplot(gs[2, 0])
    ax.boxplot(
        [recv_to_pub, pub_to_exec, recv_to_exec],
        labels=["recv->pub", "pub->exec", "recv->exec"],
        showfliers=True,
        patch_artist=True,
        boxprops={"facecolor": "#DDEEFF"},
        medianprops={"color": "#CC3311", "linewidth": 2},
    )
    ax.set_title("Latency boxplot comparison")
    ax.set_ylabel("ms")
    ax.grid(True, axis="y", alpha=0.3)

    # 6) recent timeline with timeout annotation
    ax = fig.add_subplot(gs[2, 1])
    if tail_records:
        y = np.arange(len(tail_records))
        seq_tail = [r["seq"] for r in tail_records]
        recv_pub_tail = np.array([float(r.get("recv_to_pub_ms") or 0.0) for r in tail_records], dtype=float)
        pub_exec_tail = np.array([float(r.get("pub_to_exec_ms") or 0.0) for r in tail_records], dtype=float)
        statuses = [r.get("status", "unknown") for r in tail_records]

        ax.barh(y, recv_pub_tail, left=np.zeros_like(y, dtype=float), label=LABELS["recv_to_pub_ms"], color=COLORS["recv_to_pub_ms"])
        ax.barh(y, pub_exec_tail, left=recv_pub_tail, label=LABELS["pub_to_exec_ms"], color=COLORS["pub_to_exec_ms"])

        for idx, r in enumerate(tail_records):
            total = r.get("recv_to_exec_ms")
            if total is not None:
                ax.text(float(total) + 0.4, idx, f"{float(total):.1f} ms", va="center", fontsize=8)
            else:
                timeout_x = float(r.get("recv_to_pub_ms") or 0.0)
                ax.scatter([timeout_x], [idx], marker="x", s=50, color=COLORS["timeout"], label="timeout" if idx == 0 else None)
                ax.text(timeout_x + 0.4, idx, "timeout", va="center", fontsize=8, color=COLORS["timeout"])

        ax.set_yticks(y)
        ax.set_yticklabels([f"seq {s} ({st})" for s, st in zip(seq_tail, statuses)])
        ax.set_xlabel("ms (relative to receive)")
        ax.set_title(f"Recent {len(tail_records)} trace timelines")
        ax.grid(True, axis="x", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="lower right")
    else:
        ax.axis("off")

    fig.suptitle("xr_teleoperate latency visualization (detailed)", fontsize=16)
    fig.savefig(out_path, dpi=180)
    print(f"saved plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_path", type=str, help="Path to latency trace jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--tail", type=int, default=20, help="How many recent traces to show in the timeline subplot")
    parser.add_argument("--hist-bins", type=int, default=20, help="Histogram bins for latency distribution plots")
    args = parser.parse_args()

    trace_path = Path(args.trace_path).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve() if args.output else trace_path.with_suffix(".png")
    records = load_records(trace_path)
    plot_records(records, out_path, tail=max(1, args.tail), hist_bins=max(5, args.hist_bins))


if __name__ == "__main__":
    main()
