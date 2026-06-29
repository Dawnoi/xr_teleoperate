import numpy as np
from dataclasses import dataclass, field


T_ROBOT_OPENXR = np.array([[0, 0, -1, 0],
                           [-1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]])

T_OPENXR_ROBOT = np.array([[0, -1, 0, 0],
                           [0, 0, 1, 0],
                           [-1, 0, 0, 0],
                           [0, 0, 0, 1]])

CONST_HEAD_POSE = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 1.5],
                            [0, 0, 1, -0.2],
                            [0, 0, 0, 1]])

CONST_RIGHT_ARM_POSE = np.array([[1, 0, 0, 0.15],
                                 [0, 1, 0, 1.13],
                                 [0, 0, 1, -0.3],
                                 [0, 0, 0, 1]])

CONST_LEFT_ARM_POSE = np.array([[1, 0, 0, -0.15],
                                [0, 1, 0, 1.13],
                                [0, 0, 1, -0.3],
                                [0, 0, 0, 1]])

# G1_29 arm home EE pose in robot base frame (q = 0).
# These are used as the controller-space anchor so that, after calibration,
# keeping both controllers still will keep the MuJoCo / real robot arms at home.
ROBOT_HOME_LEFT_WRIST_POSE = np.array([[1, 0, 0, 0.24977],
                                       [0, 1, 0, 0.14865],
                                       [0, 0, 1, 0.09523],
                                       [0, 0, 0, 1]])

ROBOT_HOME_RIGHT_WRIST_POSE = np.array([[1, 0, 0, 0.24977],
                                        [0, 1, 0, -0.14864],
                                        [0, 0, 1, 0.09523],
                                        [0, 0, 0, 1]])


@dataclass
class TeleData:
    head_pose: np.ndarray
    left_wrist_pose: np.ndarray
    right_wrist_pose: np.ndarray

    left_hand_pos: np.ndarray = None
    right_hand_pos: np.ndarray = None
    left_hand_rot: np.ndarray = None
    right_hand_rot: np.ndarray = None

    left_hand_pinch: bool = False
    left_hand_pinchValue: float = 10.0
    left_hand_squeeze: bool = False
    left_hand_squeezeValue: float = 0.0

    right_hand_pinch: bool = False
    right_hand_pinchValue: float = 10.0
    right_hand_squeeze: bool = False
    right_hand_squeezeValue: float = 0.0

    left_ctrl_trigger: bool = False
    left_ctrl_triggerValue: float = 10.0
    left_ctrl_squeeze: bool = False
    left_ctrl_squeezeValue: float = 0.0
    left_ctrl_aButton: bool = False
    left_ctrl_bButton: bool = False
    left_ctrl_thumbstick: bool = False
    left_ctrl_thumbstickValue: np.ndarray = field(default_factory=lambda: np.zeros(2))

    right_ctrl_trigger: bool = False
    right_ctrl_triggerValue: float = 10.0
    right_ctrl_squeeze: bool = False
    right_ctrl_squeezeValue: float = 0.0
    right_ctrl_aButton: bool = False
    right_ctrl_bButton: bool = False
    right_ctrl_thumbstick: bool = False
    right_ctrl_thumbstickValue: np.ndarray = field(default_factory=lambda: np.zeros(2))
