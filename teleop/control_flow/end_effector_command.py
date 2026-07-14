"""末端执行器命令映射：把遥操输入写入夹爪或灵巧手控制量。

End-effector command mapping for teleop.
"""

import numpy as np


def read_dual_gripper_snapshot(
    *,
    args,
    dual_gripper_data_lock,
    dual_gripper_state_array,
    dual_gripper_action_array,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Read Dex1 feedback and the control thread's processed command atomically.

    The action array is written by the gripper control thread after trigger mapping,
    rate limiting, force-hold handling, and smoothing.  It is therefore the only
    command suitable for comparing against the measured state in the UI.
    """

    if args.no_gripper or args.ee != "dex1":
        return None, None, None, None
    if dual_gripper_data_lock is None:
        raise RuntimeError("Dex1 gripper data lock is unavailable")
    if dual_gripper_state_array is None or dual_gripper_action_array is None:
        raise RuntimeError("Dex1 gripper state/action arrays are unavailable")

    with dual_gripper_data_lock:
        left_feedback = float(dual_gripper_state_array[0])
        right_feedback = float(dual_gripper_state_array[1])
        left_command = float(dual_gripper_action_array[0])
        right_command = float(dual_gripper_action_array[1])

    values = np.asarray([left_feedback, right_feedback, left_command, right_command], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Dex1 gripper snapshot contains non-finite values: {values.tolist()}")
    return left_feedback, right_feedback, left_command, right_command


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
