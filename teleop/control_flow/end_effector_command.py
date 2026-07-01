"""末端执行器命令映射：把遥操输入写入夹爪或灵巧手控制量。

End-effector command mapping for teleop.
"""

import numpy as np


def read_online_gripper_widths(*, args, dual_gripper_data_lock, dual_gripper_state_array):
    if args.no_gripper or args.ee != "dex1":
        return 0.0, 0.0
    try:
        with dual_gripper_data_lock:
            return float(dual_gripper_state_array[0]), float(dual_gripper_state_array[1])
    except Exception:
        return 0.0, 0.0


def apply_end_effector_command(
    *,
    args,
    tele_data,
    left_arm_enabled: bool,
    right_arm_enabled: bool,
    online_left_gripper_q: float,
    online_right_gripper_q: float,
    left_hand_pos_array=None,
    right_hand_pos_array=None,
    left_gripper_value=None,
    right_gripper_value=None,
):
    if args.no_gripper:
        return

    if args.ee in {"dex3", "inspire_dfx", "inspire_ftp", "brainco"} and args.input_mode == "hand":
        with left_hand_pos_array.get_lock():
            left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
        with right_hand_pos_array.get_lock():
            right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
        return

    if args.ee == "dex1" and args.input_mode == "controller":
        if args.input_provider == "online_inference":
            left_trigger_value = (
                tele_data.left_ctrl_triggerValue
                if left_arm_enabled
                else float(np.clip(online_left_gripper_q / 5.4 * 2.0 + 5.0, 5.0, 7.0))
            )
            right_trigger_value = (
                tele_data.right_ctrl_triggerValue
                if right_arm_enabled
                else float(np.clip(online_right_gripper_q / 5.4 * 2.0 + 5.0, 5.0, 7.0))
            )
            with left_gripper_value.get_lock():
                left_gripper_value.value = left_trigger_value
            with right_gripper_value.get_lock():
                right_gripper_value.value = right_trigger_value
            return

        if left_arm_enabled:
            with left_gripper_value.get_lock():
                left_gripper_value.value = tele_data.left_ctrl_triggerValue
        if right_arm_enabled:
            with right_gripper_value.get_lock():
                right_gripper_value.value = tele_data.right_ctrl_triggerValue
        return

    if args.ee == "dex1" and args.input_mode == "hand":
        with left_gripper_value.get_lock():
            left_gripper_value.value = tele_data.left_hand_pinchValue
        with right_gripper_value.get_lock():
            right_gripper_value.value = tele_data.right_hand_pinchValue
