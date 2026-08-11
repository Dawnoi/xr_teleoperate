"""Measured mobile observations and protocol availability for online inference."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable
import time

import numpy as np

from teleop.control_flow.mobile_manipulation_coordinator import column_position_from_raw_height


class MobileOnlineInferenceInputs:
    """Build strict mobile observations for the supported mobile profiles."""

    def __init__(
        self,
        *,
        args: Any,
        base_state_receiver: Any,
        mobile_base_kinematics: Any,
        waist_yaw_getter: Callable[[], float],
    ) -> None:
        if not callable(waist_yaw_getter):
            raise ValueError("mobile inference inputs requires a waist-yaw getter")
        self._args = args
        self._base_state_receiver = base_state_receiver
        self._mobile_base_kinematics = mobile_base_kinematics
        self._waist_yaw_getter = waist_yaw_getter

    def current(self, arm_q: np.ndarray, *, include_tcp: bool) -> dict[str, Any]:
        profile_name = "mobile_tcp23" if include_tcp else "mobile_joint_base"
        receiver = self._require_base_state_receiver(profile_name)
        kinematics = self._require_tcp_kinematics() if include_tcp else None
        odom_sample, height_sample = receiver.snapshot_latest()
        slam_history = receiver.snapshot_slam_tf_history()
        if odom_sample is None or height_sample is None or slam_history is None or not slam_history:
            raise RuntimeError(f"{profile_name.upper()}_STATE_MISSING")
        slam_sample = dict(slam_history[-1])
        self._validate_state_ages(profile_name, odom_sample, height_sample, slam_sample)
        if str(slam_sample.get("frame_id")) != "slamware_map" or str(slam_sample.get("child_frame_id")) != "base_link":
            raise RuntimeError(
                f"{profile_name.upper()}_SLAM_FRAME_INVALID "
                f"frame_id={slam_sample.get('frame_id')!r} child_frame_id={slam_sample.get('child_frame_id')!r}"
            )
        velocity = odom_sample.get("velocity")
        if not isinstance(velocity, dict) or str(velocity.get("frame_id")) != "base_link":
            raise RuntimeError(f"{profile_name.upper()}_BASE_VELOCITY_FRAME_INVALID: expected base_link")
        column_height = column_position_from_raw_height(
            raw_height=float(height_sample["height"]["z"]),
            raw_minimum=float(self._args.mobile_height_raw_minimum),
            raw_maximum=float(self._args.mobile_height_raw_maximum),
            column_travel_m=float(self._args.mobile_column_travel_m),
        )
        map_pose = np.asarray([slam_sample["x"], slam_sample["y"], slam_sample["yaw"]], dtype=float)
        base_velocity = np.asarray([velocity["vx"], velocity["wz"]], dtype=float)
        if not np.all(np.isfinite(map_pose)) or not np.all(np.isfinite(base_velocity)):
            raise RuntimeError(f"{profile_name.upper()}_BASE_STATE_NONFINITE")
        result: dict[str, Any] = {
            "current_map_base_pose": map_pose,
            "current_base_velocity_base_link": base_velocity,
            "current_column_height_m": column_height,
        }
        if not include_tcp:
            return result

        waist_yaw = float(self._waist_yaw_getter())
        if not np.isfinite(waist_yaw):
            raise RuntimeError("MOBILE_TCP23_WAIST_YAW_NONFINITE")
        left_tcp, right_tcp = kinematics.dex1_tcp_poses_base_link(arm_q, column_height, waist_yaw)
        result.update(
            {
                "current_left_robot_tcp_pose_base_link": left_tcp,
                "current_right_robot_tcp_pose_base_link": right_tcp,
                "mobile_tcp_to_wrist_transformer": partial(
                    self._tcp_target_to_ik_wrist,
                    column_height_m=column_height,
                    waist_yaw_rad=waist_yaw,
                ),
            }
        )
        return result

    def tcp23_runtime_error(self) -> str:
        missing: list[str] = []
        if str(self._args.arm) != "G1_29":
            missing.append("--arm G1_29")
        if str(self._args.ee) != "dex1" or bool(self._args.no_gripper):
            missing.append("--ee dex1 without --no-gripper")
        if self._args.mobile_manipulation_mode != "direct_ik":
            missing.append("--mobile-manipulation-mode direct_ik")
        if self._args.base_controller != "g1d_agv" or not self._args.base_motion:
            missing.append("--base-controller g1d_agv --base-motion")
        if self._args.base_velocity_frame != "base_link":
            missing.append("--base-velocity-frame base_link")
        if not self._args.record_slam_map_pose:
            missing.append("--record-slam-map-pose")
        if self._mobile_base_kinematics is None or not self._mobile_base_kinematics.dex1_tcp_available:
            missing.append("Dex1 TCP FK")
        if self._mobile_base_kinematics is None:
            missing.append("G1D base_link kinematics")
        missing.extend(self._base_state_receiver_requirements("base state receiver (--record-base)"))
        return "" if not missing else "mobile_tcp23 unavailable: requires " + ", ".join(missing)

    def joint_base_runtime_error(self) -> str:
        missing: list[str] = []
        if str(self._args.arm) != "G1_29":
            missing.append("--arm G1_29")
        if str(self._args.ee) != "dex1" or bool(self._args.no_gripper):
            missing.append("--ee dex1 without --no-gripper")
        if self._args.mobile_manipulation_mode == "mobile_ik_qp":
            missing.append("mobile_ik_qp disabled")
        if self._args.base_controller != "g1d_agv" or not self._args.base_motion:
            missing.append("--base-controller g1d_agv --base-motion")
        if self._args.base_velocity_frame != "base_link":
            missing.append("--base-velocity-frame base_link")
        missing.extend(self._base_state_receiver_requirements("base state receiver"))
        return "" if not missing else "mobile_joint_base unavailable: requires " + ", ".join(missing)

    def pelvis_planar22_current(self) -> dict[str, Any]:
        """Return the measured planar base state for legacy pelvis-frame EEF inference.

        The arm EEF poses are intentionally not built here.  The main loop already
        owns the G1_29 IK-frame feedback poses; those are the training dataset's
        ``pelvis -> gripper_flange`` convention and must not be recomputed through
        the newer base-link Dex1 TCP calibration.
        """
        profile_name = "mobile_pelvis_planar22"
        receiver = self._require_base_state_receiver(profile_name)
        odom_sample, _height_sample = receiver.snapshot_latest()
        slam_history = receiver.snapshot_slam_tf_history()
        if odom_sample is None or slam_history is None or not slam_history:
            raise RuntimeError(f"{profile_name.upper()}_STATE_MISSING")
        slam_sample = dict(slam_history[-1])
        self._validate_odom_slam_state_ages(profile_name, odom_sample, slam_sample)
        if str(slam_sample.get("frame_id")) != "slamware_map" or str(slam_sample.get("child_frame_id")) != "base_link":
            raise RuntimeError(
                f"{profile_name.upper()}_SLAM_FRAME_INVALID "
                f"frame_id={slam_sample.get('frame_id')!r} child_frame_id={slam_sample.get('child_frame_id')!r}"
            )
        velocity = odom_sample.get("velocity")
        if not isinstance(velocity, dict) or str(velocity.get("frame_id")) != "base_link":
            raise RuntimeError(f"{profile_name.upper()}_BASE_VELOCITY_FRAME_INVALID: expected base_link")
        map_pose = np.asarray([slam_sample["x"], slam_sample["y"], slam_sample["yaw"]], dtype=float)
        base_velocity = np.asarray([velocity["vx"], velocity["wz"]], dtype=float)
        if not np.all(np.isfinite(map_pose)) or not np.all(np.isfinite(base_velocity)):
            raise RuntimeError(f"{profile_name.upper()}_BASE_STATE_NONFINITE")
        return {
            "current_map_base_pose": map_pose,
            "current_base_velocity_base_link": base_velocity,
        }

    def pelvis_planar22_runtime_error(self) -> str:
        missing: list[str] = []
        if str(self._args.arm) != "G1_29":
            missing.append("--arm G1_29")
        if str(self._args.ee) != "dex1" or bool(self._args.no_gripper):
            missing.append("--ee dex1 without --no-gripper")
        if self._args.mobile_manipulation_mode != "direct_ik":
            missing.append("--mobile-manipulation-mode direct_ik")
        if self._args.base_controller != "g1d_agv" or not self._args.base_motion:
            missing.append("--base-controller g1d_agv --base-motion")
        if self._args.base_command_source != "provider":
            missing.append("--base-command-source provider")
        if self._args.base_velocity_frame != "base_link":
            missing.append("--base-velocity-frame base_link")
        if not self._args.record_slam_map_pose:
            missing.append("--record-slam-map-pose")
        missing.extend(self._base_state_receiver_requirements("base state receiver (--record-base)"))
        return "" if not missing else "mobile_pelvis_planar22 unavailable: requires " + ", ".join(missing)

    def _require_base_state_receiver(self, profile_name: str):
        receiver = self._base_state_receiver
        if receiver is None:
            raise RuntimeError(f"{profile_name} requires initialized base state")
        if not receiver.is_alive():
            raise RuntimeError(f"{profile_name} base state receiver is not alive")
        return receiver

    def _require_tcp_kinematics(self):
        provider = self._mobile_base_kinematics
        if provider is None or not provider.dex1_tcp_available:
            raise RuntimeError("mobile_tcp23 requires initialized TCP FK and G1D kinematics")
        return provider

    def _validate_state_ages(self, profile_name: str, odom_sample, height_sample, slam_sample) -> None:
        now_ns = time.monotonic_ns()
        timeout_ns = int(float(self._args.mobile_state_timeout_sec) * 1e9)
        ages = {
            "odom": now_ns - int(odom_sample["t_ns"]),
            "height": now_ns - int(height_sample["t_ns"]),
            "slam_tf": now_ns - int(slam_sample["t_ns"]),
        }
        stale = {name: age for name, age in ages.items() if age < 0 or age > timeout_ns}
        if stale:
            raise RuntimeError(
                f"{profile_name.upper()}_STATE_STALE "
                + " ".join(f"{name}_age_ms={age / 1e6:.1f}" for name, age in stale.items())
                + f" timeout_ms={timeout_ns / 1e6:.1f}"
            )

    def _validate_odom_slam_state_ages(self, profile_name: str, odom_sample, slam_sample) -> None:
        now_ns = time.monotonic_ns()
        timeout_ns = int(float(self._args.mobile_state_timeout_sec) * 1e9)
        ages = {
            "odom": now_ns - int(odom_sample["t_ns"]),
            "slam_tf": now_ns - int(slam_sample["t_ns"]),
        }
        stale = {name: age for name, age in ages.items() if age < 0 or age > timeout_ns}
        if stale:
            raise RuntimeError(
                f"{profile_name.upper()}_STATE_STALE "
                + " ".join(f"{name}_age_ms={age / 1e6:.1f}" for name, age in stale.items())
                + f" timeout_ms={timeout_ns / 1e6:.1f}"
            )

    def _tcp_target_to_ik_wrist(
        self,
        side: str,
        tcp_target: np.ndarray,
        *,
        column_height_m: float,
        waist_yaw_rad: float,
    ) -> np.ndarray:
        return self._require_tcp_kinematics().legacy_ik_ee_pose_from_tcp_target(
            side,
            tcp_target,
            column_height_m=column_height_m,
            waist_yaw_rad=waist_yaw_rad,
        )

    def _base_state_receiver_requirements(self, missing_label: str) -> list[str]:
        if self._base_state_receiver is None:
            return [missing_label]
        if not self._base_state_receiver.is_alive():
            return ["live base state receiver"]
        return []
