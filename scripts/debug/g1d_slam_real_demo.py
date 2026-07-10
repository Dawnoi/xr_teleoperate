#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from core.control.g1d_slam_client import G1DSlamClient


class DemoCase:
    def __init__(
        self,
        func: Callable[[Any, argparse.Namespace], dict[str, Any]],
        *,
        requires_motion: bool = False,
        requires_abort: bool = False,
    ) -> None:
        self.func = func
        self.requires_motion = bool(requires_motion)
        self.requires_abort = bool(requires_abort)


class DemoTask:
    def __init__(
        self,
        name: str,
        func: Callable[[Any, argparse.Namespace], dict[str, Any]],
        *,
        requires_motion: bool = True,
    ) -> None:
        self.name = name
        self.func = func
        self.requires_motion = bool(requires_motion)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run high-level G1D SLAM HTTP checks against a real robot.",
    )
    parser.add_argument("--robot-ip", required=True, help="G1D robot IP, for example 192.168.123.163")
    parser.add_argument("--port", type=int, default=1448, help="G1D SLAM HTTP API port")
    parser.add_argument("--timeout-sec", type=float, default=5.0, help="HTTP timeout in seconds")
    parser.add_argument("--debug-http", action="store_true", help="Print G1D HTTP request/response logs to stderr.")
    parser.add_argument(
        "--task",
        type=int,
        choices=sorted(TASKS),
        default=None,
        help="High-level real-robot task id. Task 1: p0 -> p1 -> go_home(no dock). Task 2: relocalize -> p0 -> p1 -> P2 -> P3 -> P5.",
    )
    parser.add_argument(
        "--case",
        choices=sorted(CASE_HANDLERS),
        default="status",
        help="Low-level debug case to run when --task is not set. Motion cases require --allow-motion.",
    )
    parser.add_argument(
        "--allow-motion",
        action="store_true",
        help="Required for cases that can move the chassis.",
    )
    parser.add_argument(
        "--allow-abort",
        action="store_true",
        help="Required for abort-current-action because it sends DELETE to the current action endpoint.",
    )
    parser.add_argument(
        "--allow-dock",
        action="store_true",
        help="Required with --dock because the robot will try to get on the charger.",
    )
    parser.add_argument("--wait", action="store_true", help="Wait for action completion when supported.")
    parser.add_argument("--action-timeout-sec", type=float, default=180.0, help="Wait timeout for each route action.")
    parser.add_argument("--poll-interval-sec", type=float, default=1.0, help="Action polling interval.")
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=250,
        help="MoveByAction pulse duration. Demo limit: 1..300 ms.",
    )
    parser.add_argument("--angle-rad", type=float, default=0.2, help="Small rotate angle in radians.")
    parser.add_argument("--x", type=float, default=None, help="MoveTo target x. Required by --case move-to.")
    parser.add_argument("--y", type=float, default=None, help="MoveTo target y. Required by --case move-to.")
    parser.add_argument("--z", type=float, default=0.0, help="MoveTo target z.")
    parser.add_argument("--yaw", type=float, default=None, help="MoveTo target yaw. Enables precise yaw mode.")
    parser.add_argument("--speed-ratio", type=float, default=0.3, help="MoveTo speed ratio.")
    parser.add_argument(
        "--poi",
        action="append",
        default=None,
        help="POI display name for --case poi-route. Repeatable. Default: p0 then p1.",
    )
    parser.add_argument("--dock", action="store_true", help="Append GoHomeAction dock mode at the end of poi-route.")
    parser.add_argument(
        "--max-recover-time-ms",
        type=int,
        default=30000,
        help="RecoverLocalizationAction max recover time.",
    )
    parser.add_argument(
        "--max-parking-wait-ms",
        type=int,
        default=60000,
        help="ReturnToParkingAction max wait time.",
    )
    return parser.parse_args(argv)


def run_demo(args: argparse.Namespace, *, client: Any | None = None) -> dict[str, Any]:
    slam = client if client is not None else _make_client(args)
    if args.task is not None:
        return run_task(slam, args)
    return run_case(slam, args)


def _make_client(args: argparse.Namespace) -> G1DSlamClient:
    return G1DSlamClient(
        robot_ip=args.robot_ip,
        port=args.port,
        timeout_sec=args.timeout_sec,
        debug_http=args.debug_http,
    )


def run_task(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    task_id = int(args.task)
    task = TASKS[task_id]
    if task.requires_motion and not bool(args.allow_motion):
        raise PermissionError(f"task {task_id} can move the chassis; rerun with --allow-motion after clearing the area")
    if bool(args.dock) and not bool(args.allow_dock):
        raise PermissionError("--dock will try to get on the charger; rerun with --allow-dock to confirm")
    _task_log(args, f"start task {task_id}: {task.name}")
    result = task.func(slam, args)
    _task_log(args, f"finish task {task_id}: {task.name}")
    return {"task": task_id, "task_name": task.name, "result": result}


def run_case(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    case_name = str(args.case)
    case = CASE_HANDLERS[case_name]
    if case.requires_motion and not bool(args.allow_motion):
        raise PermissionError(f"{case_name} can move the chassis; rerun with --allow-motion after clearing the area")
    if case.requires_abort and not bool(args.allow_abort):
        raise PermissionError("abort-current-action cancels robot state; rerun with --allow-abort to confirm")
    if bool(args.dock) and not bool(args.allow_dock):
        raise PermissionError("--dock will try to get on the charger; rerun with --allow-dock to confirm")
    if case_name.startswith("move-by") and int(args.duration_ms) > 300:
        raise ValueError("--duration-ms must be <= 300 for real-robot MoveByAction demos")
    return case.func(slam, args)


def case_status(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "case": "status",
        "robot_info": slam.robot_info(),
        "capabilities": slam.capabilities(),
        "health": slam.health(),
        "current_action": slam.current_action(),
    }


def case_robot_info(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {"case": "robot-info", "result": slam.robot_info()}


def case_capabilities(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {"case": "capabilities", "result": slam.capabilities()}


def case_health(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {"case": "health", "result": slam.health()}


def case_current_action(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {"case": "current-action", "result": slam.current_action()}


def case_list_pois(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {"case": "list-pois", "result": slam.pois()}


def case_abort_current_action(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    return {"case": "abort-current-action", "result": slam.abort_current_action()}


def case_move_by_forward(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    health = slam.health()
    result = slam.move_by(G1DSlamClient.MOVE_BY_FORWARD, duration_ms=args.duration_ms, wait=args.wait)
    return {"case": "move-by-forward", "health": health, "result": result}


def case_move_by_backward(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    health = slam.health()
    result = slam.move_by(G1DSlamClient.MOVE_BY_BACKWARD, duration_ms=args.duration_ms, wait=args.wait)
    return {"case": "move-by-backward", "health": health, "result": result}


def case_rotate_small(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    health = slam.health()
    result = slam.rotate(args.angle_rad, wait=args.wait)
    return {"case": "rotate-small", "health": health, "result": result}


def case_move_to(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.x is None or args.y is None:
        raise ValueError("--case move-to requires explicit --x and --y")
    health = slam.health()
    result = slam.move_to(
        x=args.x,
        y=args.y,
        z=args.z,
        yaw=args.yaw,
        precise=args.yaw is not None,
        speed_ratio=args.speed_ratio,
        wait=args.wait,
    )
    return {"case": "move-to", "health": health, "result": result}


def case_recover_localization(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    health = slam.health()
    result = slam.recover_localization(max_recover_time_ms=args.max_recover_time_ms, wait=args.wait)
    return {"case": "recover-localization", "health": health, "result": result}


def case_go_home_no_dock(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    health = slam.health()
    result = slam.go_home(dock=False, wait=args.wait)
    return {"case": "go-home-no-dock", "health": health, "result": result}


def case_poi_route(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    health = slam.health()
    result = run_poi_route(slam, args)
    return {"case": "poi-route", "health": health, "result": result}


def case_return_to_parking(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    health = slam.health()
    result = slam.return_to_parking(
        wait_for_parking=True,
        max_wait_time_ms=args.max_parking_wait_ms,
        wait=args.wait,
    )
    return {"case": "return-to-parking", "health": health, "result": result}


def task_1_p0_p1_go_home(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    health = slam.health()
    steps = _run_poi_sequence(slam, args, poi_names=["p0", "p1"])
    home_response = slam.go_home(dock=False, wait=False)
    home_final_state = _wait_created_action(slam, args, home_response)
    steps.append(
        {
            "step": "go-home:no-dock",
            "create_response": home_response,
            "final_state": home_final_state,
        }
    )
    return {"health": health, "pois": ["p0", "p1"], "go_home": True, "dock": False, "steps": steps}


def task_2_relocalize_p0_p1_p2_p3_p5(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    health = slam.health()
    poi_names = ["p0", "p1", "P2", "P3", "P5"]
    poi_steps = _resolve_poi_steps(slam.pois(), poi_names=poi_names)
    recover_response = slam.recover_localization(max_recover_time_ms=args.max_recover_time_ms, wait=False)
    recover_final_state = _wait_created_action(slam, args, recover_response)
    steps = [
        {
            "step": "recover-localization",
            "create_response": recover_response,
            "final_state": recover_final_state,
        }
    ]
    steps.extend(_run_resolved_poi_steps(slam, args, poi_steps=poi_steps))
    return {
        "health": health,
        "recover_localization": True,
        "pois": poi_names,
        "steps": steps,
    }


def run_poi_route(slam: Any, args: argparse.Namespace) -> dict[str, Any]:
    poi_names = list(args.poi or ["p0", "p1"])
    steps = _run_poi_sequence(slam, args, poi_names=poi_names)
    if bool(args.dock):
        home_response = slam.go_home(dock=True, wait=False)
        home_final_state = _wait_created_action(slam, args, home_response)
        steps.append(
            {
                "step": "go-home:dock",
                "create_response": home_response,
                "final_state": home_final_state,
            }
        )
    return {"pois": poi_names, "dock": bool(args.dock), "steps": steps}


def _run_poi_sequence(slam: Any, args: argparse.Namespace, *, poi_names: list[str]) -> list[dict[str, Any]]:
    return _run_resolved_poi_steps(slam, args, poi_steps=_resolve_poi_steps(slam.pois(), poi_names=poi_names))


def _run_resolved_poi_steps(slam: Any, args: argparse.Namespace, *, poi_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = []
    for poi_step in poi_steps:
        requested_name = str(poi_step["requested_name"])
        resolved_name = str(poi_step["resolved_name"])
        pose = poi_step["pose"]
        _task_log(
            args,
            f"move to POI {requested_name} resolved={resolved_name}: x={pose['x']} y={pose['y']} yaw={pose['yaw']}",
        )
        create_response = slam.move_to(
            x=pose["x"],
            y=pose["y"],
            z=0.0,
            yaw=pose["yaw"],
            precise=True,
            speed_ratio=args.speed_ratio,
            wait=False,
        )
        final_state = _wait_created_action(slam, args, create_response)
        steps.append(
            {
                "step": f"move-to-poi:{requested_name}",
                "resolved_name": resolved_name,
                "pose": pose,
                "create_response": create_response,
                "final_state": final_state,
            }
        )
    return steps


def _wait_created_action(slam: Any, args: argparse.Namespace, create_response: dict[str, Any]) -> dict[str, Any]:
    action_id = str(create_response["action_id"])
    _task_log(args, f"wait action {action_id}")
    return slam.wait_action(
        action_id,
        timeout_sec=args.action_timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )


def _task_log(args: argparse.Namespace, message: str) -> None:
    if bool(args.debug_http):
        print(f"[G1D_TASK] {message}", file=sys.stderr, flush=True)


def _resolve_poi_steps(pois: list[Any], *, poi_names: list[str]) -> list[dict[str, Any]]:
    poi_by_name = _pois_by_display_name(pois)
    steps = []
    for name in poi_names:
        resolved_name, poi = _resolve_poi_by_name(poi_by_name, name)
        steps.append(
            {
                "requested_name": name,
                "resolved_name": resolved_name,
                "pose": _poi_pose(poi, resolved_name),
            }
        )
    return steps


def _resolve_poi_by_name(poi_by_name: dict[str, dict[str, Any]], name: str) -> tuple[str, dict[str, Any]]:
    if name in poi_by_name:
        return name, poi_by_name[name]
    lower_name = name.lower()
    case_matches = [candidate for candidate in poi_by_name if candidate.lower() == lower_name]
    if len(case_matches) == 1:
        resolved_name = case_matches[0]
        return resolved_name, poi_by_name[resolved_name]
    if len(case_matches) > 1:
        raise ValueError(f"POI {name!r} is ambiguous by case-insensitive match: {sorted(case_matches)}")
    raise ValueError(f"POI {name!r} not found; available={sorted(poi_by_name)}")


def _pois_by_display_name(pois: list[Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for poi in pois:
        if not isinstance(poi, dict):
            raise ValueError(f"POI entry must be object, got {type(poi).__name__}")
        metadata = poi.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"POI metadata must be object: {poi!r}")
        display_name = metadata.get("display_name")
        if isinstance(display_name, str) and display_name:
            if display_name in out:
                raise ValueError(f"duplicate POI display_name: {display_name!r}")
            out[display_name] = poi
    return out


def _poi_pose(poi: dict[str, Any], name: str) -> dict[str, float]:
    pose = poi.get("pose")
    if not isinstance(pose, dict):
        raise ValueError(f"POI {name!r} missing object pose: {poi!r}")
    for key in ("x", "y", "yaw"):
        if key not in pose:
            raise ValueError(f"POI {name!r} missing pose.{key}: {poi!r}")
    return {"x": float(pose["x"]), "y": float(pose["y"]), "yaw": float(pose["yaw"])}


CASE_HANDLERS = {
    "status": DemoCase(case_status),
    "robot-info": DemoCase(case_robot_info),
    "capabilities": DemoCase(case_capabilities),
    "health": DemoCase(case_health),
    "current-action": DemoCase(case_current_action),
    "list-pois": DemoCase(case_list_pois),
    "abort-current-action": DemoCase(case_abort_current_action, requires_abort=True),
    "move-by-forward": DemoCase(case_move_by_forward, requires_motion=True),
    "move-by-backward": DemoCase(case_move_by_backward, requires_motion=True),
    "rotate-small": DemoCase(case_rotate_small, requires_motion=True),
    "move-to": DemoCase(case_move_to, requires_motion=True),
    "recover-localization": DemoCase(case_recover_localization, requires_motion=True),
    "go-home-no-dock": DemoCase(case_go_home_no_dock, requires_motion=True),
    "poi-route": DemoCase(case_poi_route, requires_motion=True),
    "return-to-parking": DemoCase(case_return_to_parking, requires_motion=True),
}


TASKS = {
    1: DemoTask("p0-p1-go-home", task_1_p0_p1_go_home),
    2: DemoTask("relocalize-p0-p1-P2-P3-P5", task_2_relocalize_p0_p1_p2_p3_p5),
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_demo(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
