import unittest

import numpy as np

from core.control.arm_target_safety import limit_arm_joint_target_velocity


class ArmTargetSafetyTest(unittest.TestCase):
    def test_per_arm_synchronized_limit_reaches_the_ik_posture_together(self):
        reference_q = np.zeros(14)
        target_q = np.array(
            [0.1, 0.02, 0.0, 0.4, 0.0, 0.5, 0.0, -0.1, 0.0, 0.0, -0.2, 0.0, 0.3, 0.0]
        )

        limited_q = limit_arm_joint_target_velocity(
            target_q,
            reference_q,
            max_joint_speed=1.0,
            control_frequency=10.0,
            synchronize_per_arm=True,
        )

        self.assertTrue(np.allclose(limited_q[:7], target_q[:7] * 0.2))
        self.assertTrue(np.allclose(limited_q[7:], target_q[7:] / 3.0))
        self.assertLessEqual(np.max(np.abs(limited_q - reference_q)), 0.1)

    def test_default_limit_remains_per_joint(self):
        limited_q = limit_arm_joint_target_velocity(
            np.array([0.5] * 14),
            np.zeros(14),
            max_joint_speed=1.0,
            control_frequency=10.0,
        )

        self.assertTrue(np.allclose(limited_q, 0.1))
