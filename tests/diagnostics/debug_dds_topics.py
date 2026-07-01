#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HgLowState
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_


TOPICS = {
    "lowstate": ("rt/lowstate", HgLowState),
    "lf_lowstate": ("rt/lf/lowstate", HgLowState),
    "dex1_left_state": ("rt/dex1/left/state", MotorStates_),
    "dex1_right_state": ("rt/dex1/right/state", MotorStates_),
}


def _run_cmd(cmd):
    try:
        out = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return {
            "cmd": cmd,
            "returncode": out.returncode,
            "output": out.stdout,
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "returncode": -1,
            "output": f"<exception> {e}",
        }


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _summarize_lowstate(msg):
    summary = {
        "motor_count": None,
        "first_motor_q": None,
        "first_motor_dq": None,
        "sample_q_head": [],
        "sample_dq_head": [],
    }
    try:
        motor_state = msg.motor_state
        summary["motor_count"] = len(motor_state)
        if len(motor_state) > 0:
            summary["first_motor_q"] = _safe_float(motor_state[0].q)
            summary["first_motor_dq"] = _safe_float(motor_state[0].dq)
        for s in motor_state[:6]:
            summary["sample_q_head"].append(_safe_float(s.q))
            summary["sample_dq_head"].append(_safe_float(s.dq))
    except Exception as e:
        summary["parse_error"] = str(e)
    return summary


def _summarize_motorstates(msg):
    summary = {
        "state_count": None,
        "first_q": None,
        "sample_q_head": [],
    }
    try:
        states = msg.states
        summary["state_count"] = len(states)
        if len(states) > 0:
            summary["first_q"] = _safe_float(states[0].q)
        for s in states[:4]:
            summary["sample_q_head"].append(_safe_float(s.q))
    except Exception as e:
        summary["parse_error"] = str(e)
    return summary


def _summarize_sample(topic_key, msg):
    if topic_key in ("lowstate", "lf_lowstate"):
        return _summarize_lowstate(msg)
    return _summarize_motorstates(msg)


def _make_output_dir(base_dir: Path, iface: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{ts}_{iface}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Passive DDS probe for xr_teleoperate / Unitree topics.")
    parser.add_argument("--network-interface", required=True, help="DDS network interface, e.g. eno1 / wlo1")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id (real robot usually 0)")
    parser.add_argument("--seconds", type=float, default=8.0, help="Probe duration in seconds")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "dds_debug"),
        help="Directory to store logs and summary",
    )
    args = parser.parse_args()

    out_dir = _make_output_dir(Path(args.output_dir), args.network_interface)
    log_path = out_dir / "probe.log"
    summary_path = out_dir / "summary.json"

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    summary = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "python": sys.executable,
        "argv": sys.argv,
        "network_interface": args.network_interface,
        "domain": args.domain,
        "seconds": args.seconds,
        "commands": {
            "ip_brief_addr": _run_cmd(["ip", "-brief", "addr"]),
            "ip_addr_iface": _run_cmd(["ip", "addr", "show", args.network_interface]),
            "ip_route": _run_cmd(["ip", "route"]),
        },
        "topics": {},
    }

    try:
        log(f"Output dir: {out_dir}")
        log(f"Initializing DDS: domain={args.domain}, iface={args.network_interface}")
        ChannelFactoryInitialize(args.domain, networkInterface=args.network_interface)
        log("DDS init OK")
    except Exception as e:
        summary["dds_init_error"] = str(e)
        log(f"DDS init FAILED: {e}")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return 1

    subscribers = {}
    for topic_key, (topic_name, topic_type) in TOPICS.items():
        try:
            sub = ChannelSubscriber(topic_name, topic_type)
            sub.Init()
            subscribers[topic_key] = sub
            summary["topics"][topic_key] = {
                "topic_name": topic_name,
                "samples": 0,
                "first_sample_after_sec": None,
                "last_sample_after_sec": None,
                "first_sample_summary": None,
                "last_sample_summary": None,
                "errors": [],
            }
            log(f"subscriber ready: {topic_key} -> {topic_name}")
        except Exception as e:
            summary["topics"][topic_key] = {
                "topic_name": topic_name,
                "subscriber_init_error": str(e),
                "samples": 0,
                "errors": [f"subscriber_init_error: {e}"],
            }
            log(f"subscriber init FAILED: {topic_key} -> {topic_name}: {e}")

    t0 = time.time()
    next_heartbeat = t0 + 1.0
    sleep_dt = 0.02

    read_timeout_sec = 0.005

    while time.time() - t0 < args.seconds:
        for topic_key, sub in subscribers.items():
            try:
                msg = sub.Read(timeout=read_timeout_sec)
                if msg is None:
                    continue
                elapsed = time.time() - t0
                topic_summary = summary["topics"][topic_key]
                topic_summary["samples"] += 1
                topic_summary["last_sample_after_sec"] = round(elapsed, 4)
                sample_summary = _summarize_sample(topic_key, msg)
                topic_summary["last_sample_summary"] = sample_summary
                if topic_summary["first_sample_after_sec"] is None:
                    topic_summary["first_sample_after_sec"] = round(elapsed, 4)
                    topic_summary["first_sample_summary"] = sample_summary
                    log(f"FIRST sample on {topic_key} at {elapsed:.3f}s: {sample_summary}")
            except Exception as e:
                summary["topics"][topic_key]["errors"].append(str(e))
                log(f"read error on {topic_key}: {e}")
        now = time.time()
        if now >= next_heartbeat:
            counts = ", ".join(
                f"{topic_key}={summary['topics'][topic_key].get('samples', 0)}"
                for topic_key in summary["topics"]
            )
            log(f"heartbeat: {counts}")
            next_heartbeat += 1.0
        time.sleep(sleep_dt)

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["duration_sec_actual"] = round(time.time() - t0, 4)

    cdds_log = Path("/tmp/cdds.LOG")
    if cdds_log.exists():
        dst = out_dir / "cdds.LOG"
        try:
            shutil.copy2(cdds_log, dst)
            summary["cdds_log"] = str(dst)
            log(f"copied CycloneDDS log to {dst}")
        except Exception as e:
            summary["cdds_log_copy_error"] = str(e)
            log(f"failed to copy CycloneDDS log: {e}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"summary written to {summary_path}")

    print("\n=== RESULT SUMMARY ===")
    for topic_key, topic_summary in summary["topics"].items():
        print(
            f"{topic_key:16s} "
            f"samples={topic_summary.get('samples', 0):4d} "
            f"first={topic_summary.get('first_sample_after_sec')}"
        )
    print(f"Artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
