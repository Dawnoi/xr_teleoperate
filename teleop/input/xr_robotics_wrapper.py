import os
import sys
import numpy as np


def _bootstrap_xrobotoolkit():
    try:
        import xrobotoolkit_sdk as xrt  # noqa: F401
        return
    except ModuleNotFoundError:
        candidates = [
            "/home/dx/pico_ws/src/XRoboToolkit-PC-Service-Pybind/build/lib.linux-x86_64-cpython-310",
            "/home/dx/Desktop/pika_ros/build/MyPybind11Project",
        ]
        for path in candidates:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)


_bootstrap_xrobotoolkit()
import xrobotoolkit_sdk as xrt  # noqa: E402

from teleop.input.xr_input_types import (
    TeleData,
    T_ROBOT_OPENXR,
    T_OPENXR_ROBOT,
    CONST_HEAD_POSE,
    CONST_LEFT_ARM_POSE,
    CONST_RIGHT_ARM_POSE,
    ROBOT_HOME_LEFT_WRIST_POSE,
    ROBOT_HOME_RIGHT_WRIST_POSE,
)


class XRRoboticsWrapper:
    """
    XR-Robotics raw poses -> local TeleData used by this project.
    """

    def __init__(
        self,
        use_hand_tracking: bool = False,
        head_reference_mode: str = "calibrated",
        controller_orientation_mode: str = "neutral",
        controller_mapping_mode: str = "anchored_safe",
        **kwargs,
    ):
        self.use_hand_tracking = use_hand_tracking
        self.head_reference_mode = head_reference_mode
        self.controller_orientation_mode = controller_orientation_mode
        self.controller_mapping_mode = controller_mapping_mode
        xrt.init()
        self._last_head_pose = CONST_HEAD_POSE.copy()
        self._last_left_pose = CONST_LEFT_ARM_POSE.copy()
        self._last_right_pose = CONST_RIGHT_ARM_POSE.copy()
        self._last_head_valid = False
        self._last_left_valid = False
        self._last_right_valid = False
        self._head_reference_pose = None
        self._hybrid_reference_frozen = False
        self._grip_frozen_head_pose = None
        self._prev_left_grip_pressed = False
        self._prev_right_grip_pressed = False
        self._left_operation_controller_anchor_pose = None
        self._right_operation_controller_anchor_pose = None
        self._left_robot_wrist_anchor_pose = ROBOT_HOME_LEFT_WRIST_POSE.copy()
        self._right_robot_wrist_anchor_pose = ROBOT_HOME_RIGHT_WRIST_POSE.copy()
        if self.use_hand_tracking:
            raise NotImplementedError(
                "XRRoboticsWrapper currently supports controller mode first. "
                "Hand-tracking adaptation can be added later if needed."
            )
        if self.head_reference_mode not in {
            "calibrated",
            "live",
            "hybrid",
            "head_coupled",
            "head_decoupled_live",
            "live_head_reference",
            "fixed_per_grip",
        }:
            raise ValueError(
                f"Unsupported head_reference_mode: {self.head_reference_mode}. "
                "Use 'fixed_per_grip', 'live_head_reference', "
                "'head_coupled', 'head_decoupled_live', or 'hybrid'. "
                "Legacy aliases: 'calibrated' -> 'head_coupled', 'live' -> 'head_decoupled_live'."
            )
        if self.controller_orientation_mode not in {"neutral", "relative", "absolute"}:
            raise ValueError(
                f"Unsupported controller_orientation_mode: {self.controller_orientation_mode}. "
                "Use 'absolute', 'relative', or 'neutral'."
            )
        if self.controller_mapping_mode not in {"anchored_safe", "legacy_main"}:
            raise ValueError(
                f"Unsupported controller_mapping_mode: {self.controller_mapping_mode}. "
                "Use 'legacy_main' or 'anchored_safe'."
            )

    @staticmethod
    def _pose7_to_matrix(pose7):
        pose7 = np.asarray(pose7, dtype=float).reshape(-1)
        if pose7.size < 7:
            return None
        if not np.all(np.isfinite(pose7[:7])):
            return None
        T = np.eye(4, dtype=float)
        T[:3, 3] = pose7[:3]
        qx, qy, qz, qw = pose7[3:7]
        quat = np.array([qx, qy, qz, qw], dtype=float)
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return None
        quat = quat / norm
        # same qx qy qz qw raw ordering as XR-Robotics SDK examples
        from scipy.spatial.transform import Rotation as R

        T[:3, :3] = R.from_quat(quat).as_matrix()
        return T

    def _safe_pose_matrix(self, raw_pose7, last_pose):
        mat = self._pose7_to_matrix(raw_pose7)
        if mat is None:
            return last_pose.copy(), False
        return mat, True

    def _read_robot_basis_poses(self):
        head_raw, head_valid = self._safe_pose_matrix(xrt.get_headset_pose(), self._last_head_pose)
        left_raw, left_valid = self._safe_pose_matrix(xrt.get_left_controller_pose(), self._last_left_pose)
        right_raw, right_valid = self._safe_pose_matrix(xrt.get_right_controller_pose(), self._last_right_pose)
        self._last_head_valid = bool(head_valid)
        self._last_left_valid = bool(left_valid)
        self._last_right_valid = bool(right_valid)
        if head_valid:
            self._last_head_pose = head_raw.copy()
        if left_valid:
            self._last_left_pose = left_raw.copy()
        if right_valid:
            self._last_right_pose = right_raw.copy()

        return (
            T_ROBOT_OPENXR @ head_raw @ T_OPENXR_ROBOT,
            T_ROBOT_OPENXR @ left_raw @ T_OPENXR_ROBOT,
            T_ROBOT_OPENXR @ right_raw @ T_OPENXR_ROBOT,
        )

    def has_live_pose_data(self):
        self._read_robot_basis_poses()
        return self._last_head_valid and self._last_left_valid and self._last_right_valid

    def _normalized_head_reference_mode(self):
        if self.head_reference_mode == "calibrated":
            return "head_coupled"
        if self.head_reference_mode == "live":
            return "live_head_reference"
        if self.head_reference_mode == "head_decoupled_live":
            return "live_head_reference"
        return self.head_reference_mode

    def calibrate_head_reference(self, require_live: bool = False):
        robot_head_pose, _, _ = self._read_robot_basis_poses()
        if require_live and not (self._last_head_valid and self._last_left_valid and self._last_right_valid):
            return None
        self._head_reference_pose = robot_head_pose.copy()
        self._hybrid_reference_frozen = False
        self._grip_frozen_head_pose = None
        self._prev_left_grip_pressed = False
        self._prev_right_grip_pressed = False
        self._left_operation_controller_anchor_pose = None
        self._right_operation_controller_anchor_pose = None
        return self._head_reference_pose[0:3, 3].copy()

    def get_head_reference_translation(self):
        if self._head_reference_pose is None:
            return None
        return self._head_reference_pose[0:3, 3].copy()

    def sync_reference_to_current_live_pose(self, require_live: bool = False):
        """
        Soft-reset the controller/head reference to the *current* live XR pose.
        This is lighter than a user-facing recalibration: it does not change any
        robot state, it only clears stale hybrid/calibrated reference caches so
        the next grip takeover starts from the current human pose instead of an
        older frozen operation anchor.
        """
        Brobot_world_head, left_Brobot_world_arm, right_Brobot_world_arm = self._read_robot_basis_poses()
        if require_live and not (self._last_head_valid and self._last_left_valid and self._last_right_valid):
            return None
        self._set_reference_from_current(
            Brobot_world_head,
            left_Brobot_world_arm,
            right_Brobot_world_arm,
        )
        self._hybrid_reference_frozen = False
        self._grip_frozen_head_pose = None
        self._prev_left_grip_pressed = False
        self._prev_right_grip_pressed = False
        self._left_operation_controller_anchor_pose = None
        self._right_operation_controller_anchor_pose = None
        return self._head_reference_pose[0:3, 3].copy()

    def _set_reference_from_current(
        self,
        Brobot_world_head,
        left_Brobot_world_arm,
        right_Brobot_world_arm,
    ):
        self._head_reference_pose = Brobot_world_head.copy()

    @staticmethod
    def _pose_with_head_translation_reference(reference_head_pose, world_pose):
        pose = world_pose.copy()
        pose[0:3, 3] = pose[0:3, 3] - reference_head_pose[0:3, 3]
        return pose

    def get_tele_data(
        self,
        current_left_robot_wrist_pose=None,
        current_right_robot_wrist_pose=None,
    ):
        Brobot_world_head, left_Brobot_world_arm, right_Brobot_world_arm = self._read_robot_basis_poses()
        left_trigger = float(xrt.get_left_trigger())
        right_trigger = float(xrt.get_right_trigger())
        left_grip = float(xrt.get_left_grip())
        right_grip = float(xrt.get_right_grip())
        left_axis = np.array(xrt.get_left_axis(), dtype=float)
        right_axis = np.array(xrt.get_right_axis(), dtype=float)
        left_axis_click = bool(xrt.get_left_axis_click())
        right_axis_click = bool(xrt.get_right_axis_click())
        left_primary = bool(xrt.get_X_button())
        left_secondary = bool(xrt.get_Y_button())
        right_primary = bool(xrt.get_A_button())
        right_secondary = bool(xrt.get_B_button())

        # Align to the controller-mode semantics used by this project:
        # 1) OpenXR basis -> robot basis
        # 2) keep controller initial pose convention as arm convention
        # 3) subtract head translation only
        # 4) shift from head origin to Unitree IK waist origin (+x 0.15, +z 0.45)
        left_grip_pressed = bool(left_grip > 1e-3)
        right_grip_pressed = bool(right_grip > 1e-3)

        normalized_mode = self._normalized_head_reference_mode()

        if normalized_mode in {"head_coupled", "hybrid", "fixed_per_grip"}:
            if self._head_reference_pose is None:
                if self.calibrate_head_reference(require_live=True) is None:
                    return None
            if normalized_mode == "hybrid":
                # Hybrid semantics:
                # - idle   (no grip): keep redefining the reference to the user's
                #   current pose so body walking/repositioning does not require
                #   explicit recalibration.
                # - active (any grip): freeze the reference at the current frame
                #   and use controller deltas relative to that frozen pose.
                hybrid_frozen_now = left_grip_pressed or right_grip_pressed
                should_refresh_reference = (
                    (not hybrid_frozen_now)
                    or ((not self._hybrid_reference_frozen) and hybrid_frozen_now)
                )
                if should_refresh_reference:
                    self._set_reference_from_current(
                        Brobot_world_head,
                        left_Brobot_world_arm,
                        right_Brobot_world_arm,
                    )
                self._hybrid_reference_frozen = hybrid_frozen_now
            elif normalized_mode == "fixed_per_grip":
                grip_active_now = left_grip_pressed or right_grip_pressed
                grip_rising_now = grip_active_now and (not self._hybrid_reference_frozen)
                if grip_rising_now or self._grip_frozen_head_pose is None:
                    self._grip_frozen_head_pose = Brobot_world_head.copy()
                if not grip_active_now:
                    self._grip_frozen_head_pose = None
                self._hybrid_reference_frozen = grip_active_now
            reference_head_pose = self._head_reference_pose.copy()
            if self._head_reference_pose is None:
                self._set_reference_from_current(
                    Brobot_world_head,
                    left_Brobot_world_arm,
                    right_Brobot_world_arm,
                )
                reference_head_pose = self._head_reference_pose.copy()
        else:
            reference_head_pose = Brobot_world_head.copy()

        # Important: follow the original main-branch controller semantics:
        # controller teleop uses HEAD TRANSLATION only.
        effective_left_world_arm = left_Brobot_world_arm.copy()
        effective_right_world_arm = right_Brobot_world_arm.copy()

        left_Brobot_head_arm = self._pose_with_head_translation_reference(reference_head_pose, effective_left_world_arm)
        right_Brobot_head_arm = self._pose_with_head_translation_reference(reference_head_pose, effective_right_world_arm)

        if normalized_mode == "fixed_per_grip":
            # Previous working version: keep position mapping in world space so
            # head translation does not drag the arms during a grip session.
            # Do NOT compensate head rotation here. For fixed_per_grip +
            # relative-orientation, the desired semantics are:
            #   - head motion alone should not rotate the gripper
            #   - only controller rotation relative to grip takeover should
            #     change wrist orientation
            # If we inject head-rotation compensation here, relative mode will
            # still inherit head yaw/pitch/roll and look like "the gripper
            # follows the headset".
            left_mapping_pose = left_Brobot_world_arm.copy()
            right_mapping_pose = right_Brobot_world_arm.copy()
        else:
            left_mapping_pose = left_Brobot_head_arm
            right_mapping_pose = right_Brobot_head_arm

        if self.controller_mapping_mode == "legacy_main":
            left_wrist_pose = left_Brobot_head_arm.copy()
            right_wrist_pose = right_Brobot_head_arm.copy()
            left_wrist_pose[0, 3] += 0.15
            right_wrist_pose[0, 3] += 0.15
            left_wrist_pose[2, 3] += 0.45
            right_wrist_pose[2, 3] += 0.45
            if self.controller_orientation_mode == "neutral":
                left_wrist_pose[0:3, 0:3] = ROBOT_HOME_LEFT_WRIST_POSE[0:3, 0:3]
                right_wrist_pose[0:3, 0:3] = ROBOT_HOME_RIGHT_WRIST_POSE[0:3, 0:3]
            # For legacy_main, "absolute" and "relative" both intentionally keep
            # the original main-branch direct controller-orientation behavior.
        elif normalized_mode in {"head_coupled", "live_head_reference", "hybrid", "fixed_per_grip"}:
            left_grip_rising = left_grip_pressed and (not self._prev_left_grip_pressed)
            right_grip_rising = right_grip_pressed and (not self._prev_right_grip_pressed)

            if left_grip_rising:
                self._left_operation_controller_anchor_pose = left_mapping_pose.copy()
                if current_left_robot_wrist_pose is not None:
                    self._left_robot_wrist_anchor_pose = np.asarray(current_left_robot_wrist_pose, dtype=float).copy()
            if right_grip_rising:
                self._right_operation_controller_anchor_pose = right_mapping_pose.copy()
                if current_right_robot_wrist_pose is not None:
                    self._right_robot_wrist_anchor_pose = np.asarray(current_right_robot_wrist_pose, dtype=float).copy()

            if self._left_operation_controller_anchor_pose is None:
                self._left_operation_controller_anchor_pose = left_mapping_pose.copy()
            if self._right_operation_controller_anchor_pose is None:
                self._right_operation_controller_anchor_pose = right_mapping_pose.copy()

            left_wrist_pose = self._left_robot_wrist_anchor_pose.copy()
            right_wrist_pose = self._right_robot_wrist_anchor_pose.copy()

            left_delta_pos = left_mapping_pose[0:3, 3] - self._left_operation_controller_anchor_pose[0:3, 3]
            right_delta_pos = right_mapping_pose[0:3, 3] - self._right_operation_controller_anchor_pose[0:3, 3]
            left_wrist_pose[0:3, 3] += left_delta_pos
            right_wrist_pose[0:3, 3] += right_delta_pos
            if self.controller_orientation_mode == "absolute":
                # Match the original main-branch controller semantics as closely
                # as possible: controller orientation directly defines wrist
                # orientation after only the chosen head-translation reference is
                # removed from position.
                left_wrist_pose[0:3, 0:3] = left_mapping_pose[0:3, 0:3]
                right_wrist_pose[0:3, 0:3] = right_mapping_pose[0:3, 0:3]
            elif self.controller_orientation_mode == "relative":
                left_delta_rot = left_mapping_pose[0:3, 0:3] @ self._left_operation_controller_anchor_pose[0:3, 0:3].T
                right_delta_rot = right_mapping_pose[0:3, 0:3] @ self._right_operation_controller_anchor_pose[0:3, 0:3].T
                left_wrist_pose[0:3, 0:3] = left_delta_rot @ self._left_robot_wrist_anchor_pose[0:3, 0:3]
                right_wrist_pose[0:3, 0:3] = right_delta_rot @ self._right_robot_wrist_anchor_pose[0:3, 0:3]
            else:
                left_wrist_pose[0:3, 0:3] = self._left_robot_wrist_anchor_pose[0:3, 0:3]
                right_wrist_pose[0:3, 0:3] = self._right_robot_wrist_anchor_pose[0:3, 0:3]
        else:
            left_wrist_pose = left_Brobot_head_arm.copy()
            right_wrist_pose = right_Brobot_head_arm.copy()
            left_wrist_pose[0, 3] += 0.15
            right_wrist_pose[0, 3] += 0.15
            left_wrist_pose[2, 3] += 0.42
            right_wrist_pose[2, 3] += 0.42

        self._prev_left_grip_pressed = left_grip_pressed
        self._prev_right_grip_pressed = right_grip_pressed

        return TeleData(
            head_pose=Brobot_world_head,
            left_wrist_pose=left_wrist_pose,
            right_wrist_pose=right_wrist_pose,
            left_ctrl_trigger=bool(left_trigger > 1e-3),
            left_ctrl_triggerValue=10.0 - left_trigger * 10.0,
            left_ctrl_squeeze=left_grip_pressed,
            left_ctrl_squeezeValue=left_grip,
            left_ctrl_aButton=left_primary,
            left_ctrl_bButton=left_secondary,
            left_ctrl_thumbstick=left_axis_click,
            left_ctrl_thumbstickValue=left_axis,
            right_ctrl_trigger=bool(right_trigger > 1e-3),
            right_ctrl_triggerValue=10.0 - right_trigger * 10.0,
            right_ctrl_squeeze=right_grip_pressed,
            right_ctrl_squeezeValue=right_grip,
            right_ctrl_aButton=right_primary,
            right_ctrl_bButton=right_secondary,
            right_ctrl_thumbstick=right_axis_click,
            right_ctrl_thumbstickValue=right_axis,
        )

    def render_to_xr(self, img):
        # XR-Robotics input source is used only for motion data in this adapter.
        return None

    def close(self):
        try:
            xrt.close()
        except Exception:
            pass
