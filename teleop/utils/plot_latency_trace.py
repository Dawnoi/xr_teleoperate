import argparse
import json
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


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("status") == "completed":
                records.append(item)
    return records


def plot_records(records, out_path: Path, tail: int):
    if not records:
        raise ValueError("No completed records found in trace file.")

    seq = np.array([r["seq"] for r in records], dtype=int)
    recv_to_pub = np.array([r["recv_to_pub_ms"] for r in records], dtype=float)
    pub_to_exec = np.array([r["pub_to_exec_ms"] for r in records], dtype=float)
    recv_to_exec = np.array([r["recv_to_exec_ms"] for r in records], dtype=float)

    tail_records = records[-tail:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    ax = axes[0]
    ax.plot(seq, recv_to_pub, marker="o", label=LABELS["recv_to_pub_ms"])
    ax.plot(seq, pub_to_exec, marker="o", label=LABELS["pub_to_exec_ms"])
    ax.plot(seq, recv_to_exec, marker="o", label=LABELS["recv_to_exec_ms"])
    ax.set_title("Latency Curves")
    ax.set_xlabel("seq")
    ax.set_ylabel("ms")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    y = np.arange(len(tail_records))
    left = np.zeros(len(tail_records), dtype=float)
    recv_pub_tail = np.array([r["recv_to_pub_ms"] for r in tail_records], dtype=float)
    pub_exec_tail = np.array([r["pub_to_exec_ms"] for r in tail_records], dtype=float)
    total_tail = np.array([r["recv_to_exec_ms"] for r in tail_records], dtype=float)
    seq_tail = [r["seq"] for r in tail_records]

    ax.barh(y, recv_pub_tail, left=left, label=LABELS["recv_to_pub_ms"], color="#4C78A8")
    left = left + recv_pub_tail
    ax.barh(y, pub_exec_tail, left=left, label=LABELS["pub_to_exec_ms"], color="#F58518")

    for idx, total in enumerate(total_tail):
        ax.text(total + 0.5, idx, f"total {total:.1f} ms", va="center", fontsize=9)
        ax.text(0.1, idx, "recv", va="center", ha="left", fontsize=8, color="white")
        ax.text(recv_pub_tail[idx] + 0.1, idx, "pub", va="center", ha="left", fontsize=8)
        ax.text(total + 0.1, idx, "exec", va="center", ha="left", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([f"seq {s}" for s in seq_tail])
    ax.set_xlabel("ms (relative to receive)")
    ax.set_title(f"Recent {len(tail_records)} Trace Timelines")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    fig.suptitle("xr_teleoperate Receive -> Publish -> Execute Latency", fontsize=14)
    fig.savefig(out_path, dpi=180)
    print(f"saved plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_path", type=str, help="Path to latency trace jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    parser.add_argument("--tail", type=int, default=20, help="How many recent traces to show in the timeline subplot")
    args = parser.parse_args()

    trace_path = Path(args.trace_path).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve() if args.output else trace_path.with_suffix(".png")
    records = load_records(trace_path)
    plot_records(records, out_path, tail=max(1, args.tail))


if __name__ == "__main__":
    main()
