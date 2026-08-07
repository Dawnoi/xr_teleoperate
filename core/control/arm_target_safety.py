import numpy as np


def limit_arm_joint_target_velocity(
    target_q,
    reference_q,
    max_joint_speed,
    control_frequency,
    synchronize_per_arm=False,
):
    """
    Limit single-step joint target changes so that sudden teleop / IK jumps do not
    instantly propagate to the robot or MuJoCo viewer.

    Args:
        target_q: desired joint targets, shape (14,)
        reference_q: current / last executed joint state, shape (14,)
        max_joint_speed: per-joint speed limit in rad/s
        control_frequency: outer-loop frequency in Hz
        synchronize_per_arm: advance each seven-joint arm at one common progress
            rate. This preserves the IK posture while limiting its speed.

    Returns:
        clipped_q: shape (14,)
    """
    target_q = np.asarray(target_q, dtype=float)
    reference_q = np.asarray(reference_q, dtype=float)

    if control_frequency <= 0 or max_joint_speed <= 0:
        return target_q.copy()

    max_delta = float(max_joint_speed) / float(control_frequency)
    delta = target_q - reference_q
    if not synchronize_per_arm or delta.shape != (14,):
        return reference_q + np.clip(delta, -max_delta, max_delta)

    limited_q = reference_q.copy()
    for start in (0, 7):
        arm_delta = delta[start : start + 7]
        scale = np.max(np.abs(arm_delta)) / max_delta
        limited_q[start : start + 7] += arm_delta / max(scale, 1.0)
    return limited_q
