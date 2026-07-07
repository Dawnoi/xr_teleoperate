#!/usr/bin/env python3
import argparse
import curses
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


DEX1_MIN_Q = 0.0
DEX1_MAX_Q = 5.40
DEX1_RAD_PER_CM = 0.60

TOPICS = {
    "left": {
        "cmd": "rt/dex1/left/cmd",
        "state": "rt/dex1/left/state",
    },
    "right": {
        "cmd": "rt/dex1/right/cmd",
        "state": "rt/dex1/right/state",
    },
}


@dataclass
class GripperState:
    q: float
    dq: float
    ddq: float
    q_raw: float
    dq_raw: float
    mode: int
    tau_est: float
    temperature: int
    lost: int
    host_time: float


def q_to_width_mm(q: float) -> float:
    return float(q) / DEX1_RAD_PER_CM * 10.0


def clamp_q(q: float) -> float:
    return min(max(float(q), DEX1_MIN_Q), DEX1_MAX_Q)


def active_sides(side: str) -> list[str]:
    if side == "both":
        return ["left", "right"]
    return [side]


class Dex1KeyboardProbe:
    def __init__(self, args):
        self.args = args
        self.sides = active_sides(args.side)
        self.states: dict[str, GripperState | None] = {"left": None, "right": None}
        self.targets: dict[str, float | None] = {"left": None, "right": None}
        self.publishers: dict[str, ChannelPublisher] = {}
        self.subscribers: dict[str, ChannelSubscriber] = {}
        self.cmd_msgs: dict[str, MotorCmds_] = {}
        self.last_key_text = "none"

    def init_dds(self) -> None:
        ChannelFactoryInitialize(self.args.domain, networkInterface=self.args.network_interface)
        for side in ("left", "right"):
            publisher = ChannelPublisher(TOPICS[side]["cmd"], MotorCmds_)
            publisher.Init()
            subscriber = ChannelSubscriber(TOPICS[side]["state"], MotorStates_)
            subscriber.Init()
            msg = MotorCmds_()
            msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
            msg.cmds[0].mode = 1
            msg.cmds[0].dq = 0.0
            msg.cmds[0].tau = 0.0
            msg.cmds[0].kp = self.args.kp
            msg.cmds[0].kd = self.args.kd
            self.publishers[side] = publisher
            self.subscribers[side] = subscriber
            self.cmd_msgs[side] = msg

    def read_states(self) -> None:
        for side, subscriber in self.subscribers.items():
            msg = subscriber.Read(timeout=0.001)
            if msg is None:
                continue
            if len(msg.states) < 1:
                raise RuntimeError(f"{TOPICS[side]['state']} returned MotorStates_ with no states")
            state = msg.states[0]
            self.states[side] = GripperState(
                q=float(state.q),
                dq=float(state.dq),
                ddq=float(state.ddq),
                q_raw=float(state.q_raw),
                dq_raw=float(state.dq_raw),
                mode=int(state.mode),
                tau_est=float(state.tau_est),
                temperature=int(state.temperature),
                lost=int(state.lost),
                host_time=time.time(),
            )

    def initialize_targets_from_state(self) -> None:
        for side in self.sides:
            if self.targets[side] is not None:
                continue
            if self.args.initial_q is not None:
                self.targets[side] = clamp_q(self.args.initial_q)
                continue
            state = self.states[side]
            if state is not None:
                self.targets[side] = clamp_q(state.q)

    def adjust_targets(self, delta_q: float) -> None:
        for side in self.sides:
            current_target = self.targets[side]
            if current_target is None:
                state = self.states[side]
                if state is None:
                    continue
                current_target = state.q
            self.targets[side] = clamp_q(current_target + delta_q)

    def hold_current_state(self) -> None:
        for side in self.sides:
            state = self.states[side]
            if state is not None:
                self.targets[side] = clamp_q(state.q)

    def handle_key(self, key: int) -> bool:
        if key == -1:
            return True
        if key in (ord("q"), ord("Q"), 27):
            self.last_key_text = "quit"
            return False
        if key in (curses.KEY_LEFT, ord("a"), ord("A")):
            self.adjust_targets(-self.args.step)
            self.last_key_text = "close"
            return True
        if key in (curses.KEY_RIGHT, ord("d"), ord("D")):
            self.adjust_targets(self.args.step)
            self.last_key_text = "open"
            return True
        if key == ord(" "):
            self.hold_current_state()
            self.last_key_text = "hold-current"
            return True
        self.last_key_text = str(key)
        return True

    def publish_targets(self) -> None:
        for side in self.sides:
            target_q = self.targets[side]
            if target_q is None:
                continue
            msg = self.cmd_msgs[side]
            msg.cmds[0].q = clamp_q(target_q)
            msg.cmds[0].mode = 1
            msg.cmds[0].dq = 0.0
            msg.cmds[0].tau = 0.0
            msg.cmds[0].kp = self.args.kp
            msg.cmds[0].kd = self.args.kd
            self.publishers[side].Write(msg)

    def render(self, screen) -> None:
        now = time.time()
        screen.erase()
        lines = [
            "Dex1 keyboard gripper probe",
            "",
            "WARNING: stop teleop/VLA before running this script. This process publishes rt/dex1/*/cmd.",
            "",
            f"iface={self.args.network_interface} domain={self.args.domain} side={self.args.side} rate={self.args.rate_hz:.1f}Hz",
            f"q_step={self.args.step:.3f} rad ({q_to_width_mm(self.args.step):.2f} mm) kp={self.args.kp:.3f} kd={self.args.kd:.3f}",
            "",
            "keys: LEFT/a close | RIGHT/d open | SPACE set target=current q | q/ESC quit",
            f"last_key={self.last_key_text}",
            "",
            "side    target_q  state_q   err_q  width_mm       dq   tau_est  tau/|dq| mode temp lost age_ms",
        ]
        for side in ("left", "right"):
            target = self.targets[side]
            state = self.states[side]
            if target is None:
                target_q_text = "waiting"
                error_q_text = "waiting"
            else:
                target_q_text = f"{target:8.3f}"
            if state is None:
                state_q_text = "waiting"
                error_q_text = "waiting"
                width_text = "waiting"
                dq_text = "waiting"
                tau_text = "waiting"
                load_text = "waiting"
                mode_text = "-"
                temp_text = "-"
                lost_text = "-"
                age_text = "-"
            else:
                error_q = None if target is None else float(target - state.q)
                load_index = abs(state.tau_est) / max(abs(state.dq), self.args.dq_epsilon)
                state_q_text = f"{state.q:8.3f}"
                error_q_text = "waiting" if error_q is None else f"{error_q:7.3f}"
                width_text = f"{q_to_width_mm(state.q):8.1f}"
                dq_text = f"{state.dq:8.3f}"
                tau_text = f"{state.tau_est:8.3f}"
                load_text = f"{load_index:8.1f}"
                mode_text = f"{state.mode:4d}"
                temp_text = f"{state.temperature:4d}"
                lost_text = f"{state.lost:4d}"
                age_text = f"{(now - state.host_time) * 1000.0:6.1f}"
            lines.append(
                f"{side:<7} {target_q_text:>8} {state_q_text:>8} {error_q_text:>7} "
                f"{width_text:>8} {dq_text:>8} {tau_text:>8} {load_text:>8} "
                f"{mode_text:>4} {temp_text:>4} {lost_text:>4} {age_text:>6}"
            )
        lines.extend(
            [
                "",
                "q smaller means closing; q larger means opening.",
                "dq is motor speed. q_error=target_q-state_q; negative error means commanded closing.",
                "tau/|dq| is only a rough stall/load hint, not physical force. Large value with dq near 0 means hard contact or limit.",
                "width_mm is a linear estimate from 0.60 rad ~= 10 mm.",
            ]
        )
        height, width = screen.getmaxyx()
        for row, line in enumerate(lines[: height - 1]):
            screen.addstr(row, 0, line[: max(0, width - 1)])
        screen.refresh()

    def run_curses(self, screen) -> None:
        curses.curs_set(0)
        screen.nodelay(True)
        screen.keypad(True)
        sleep_dt = 1.0 / self.args.rate_hz
        running = True
        while running:
            self.read_states()
            self.initialize_targets_from_state()
            running = self.handle_key(screen.getch())
            self.publish_targets()
            self.render(screen)
            time.sleep(sleep_dt)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyboard Dex1 gripper q probe with live tau_est display.")
    parser.add_argument("--network-interface", required=True, help="DDS network interface, e.g. eno1 / enx... / wlo1")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id. Real robot usually uses 0.")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both", help="Which gripper command topic to publish.")
    parser.add_argument("--step", type=float, default=0.05, help="q increment per key press in rad.")
    parser.add_argument("--initial-q", type=float, default=None, help="Optional initial target q. By default the script starts from current q.")
    parser.add_argument("--rate-hz", type=float, default=30.0, help="Publish and display rate.")
    parser.add_argument("--kp", type=float, default=5.0, help="Position-control kp used for this probe.")
    parser.add_argument("--kd", type=float, default=0.05, help="Position-control kd used for this probe.")
    parser.add_argument("--dq-epsilon", type=float, default=0.02, help="Minimum |dq| denominator for the displayed tau/|dq| load hint.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.rate_hz <= 0.0:
        raise ValueError("--rate-hz must be positive")
    if args.step <= 0.0:
        raise ValueError("--step must be positive")
    if args.dq_epsilon <= 0.0:
        raise ValueError("--dq-epsilon must be positive")
    probe = Dex1KeyboardProbe(args)
    probe.init_dds()
    curses.wrapper(probe.run_curses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
