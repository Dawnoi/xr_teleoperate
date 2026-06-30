import json
import math
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop.inference.pose_transform import (
    PoseTransformer,
    matrix_to_pose9_rot6d,
    matrix_to_rot6d,
    load_pose_transformer,
    matrix_to_pose7_xyzw,
    pose7_xyzw_to_matrix,
    pose7_xyzw_to_rot6d,
    pose9_rot6d_to_matrix,
    rot6d_to_matrix,
    validate_matrix4x4,
)


def _translation_matrix(x: float, y: float, z: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _write_config(root: pathlib.Path, payload: dict) -> pathlib.Path:
    path = root / "pose_transform.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _assert_matrix_close(test_case: unittest.TestCase, actual, expected, places: int = 6):
    test_case.assertEqual(len(actual), len(expected))
    for actual_row, expected_row in zip(actual, expected):
        test_case.assertEqual(len(actual_row), len(expected_row))
        for actual_value, expected_value in zip(actual_row, expected_row):
            test_case.assertAlmostEqual(actual_value, expected_value, places=places)


def _assert_vector_close(test_case: unittest.TestCase, actual, expected, places: int = 6):
    test_case.assertEqual(len(actual), len(expected))
    for actual_value, expected_value in zip(actual, expected):
        test_case.assertAlmostEqual(actual_value, expected_value, places=places)


class _ArrayLikeMatrix:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _ArrayLikeVector:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class PoseTransformTest(unittest.TestCase):
    def test_pose7_matrix_roundtrip_xyzw(self):
        pose7 = [0.25, -0.5, 1.2, 0.0, 0.0, math.sqrt(0.5) * 2.0, math.sqrt(0.5) * 2.0]

        matrix = pose7_xyzw_to_matrix(pose7)
        roundtrip_pose7 = matrix_to_pose7_xyzw(matrix)

        _assert_matrix_close(self, matrix, pose7_xyzw_to_matrix(roundtrip_pose7))
        _assert_vector_close(self, roundtrip_pose7[:3], pose7[:3])
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in roundtrip_pose7[3:])), 1.0)

    def test_rot6d_roundtrip_preserves_rotation_matrix(self):
        pose7 = [0.25, -0.5, 1.2, 0.2, 0.3, 0.4, 0.5]

        rot6d = pose7_xyzw_to_rot6d(pose7)
        matrix = rot6d_to_matrix(rot6d)
        roundtrip_rot6d = matrix_to_rot6d(matrix)

        _assert_vector_close(self, roundtrip_rot6d, rot6d)

    def test_pose9_rot6d_packs_xyz_and_rot6d(self):
        matrix = pose7_xyzw_to_matrix([0.25, -0.5, 1.2, 0.0, 0.0, 0.0, 1.0])

        pose9 = matrix_to_pose9_rot6d(matrix)

        self.assertEqual(len(pose9), 9)
        _assert_vector_close(self, pose9[:3], [0.25, -0.5, 1.2])
        _assert_vector_close(self, pose9[3:], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_pose9_rot6d_roundtrip_preserves_translation_and_rotation(self):
        pose7 = [0.25, -0.5, 1.2, 0.2, 0.3, 0.4, 0.5]

        pose9 = matrix_to_pose9_rot6d(pose7_xyzw_to_matrix(pose7))
        matrix = pose9_rot6d_to_matrix(pose9)
        roundtrip_pose9 = matrix_to_pose9_rot6d(matrix)

        _assert_vector_close(self, roundtrip_pose9, pose9)

    def test_pose7_to_matrix_accepts_array_like_vector(self):
        pose7 = _ArrayLikeVector([0.25, -0.5, 1.2, 0.0, 0.0, 0.0, 1.0])

        matrix = pose7_xyzw_to_matrix(pose7)

        _assert_vector_close(self, [matrix[0][3], matrix[1][3], matrix[2][3]], [0.25, -0.5, 1.2])

    def test_loader_requires_config_when_motion_enabled(self):
        with self.assertRaises(ValueError):
            load_pose_transformer(enable_motion=True, transform_config_path=None)

    def test_loader_rejects_implicit_identity_for_active_side_motion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_side_path = _write_config(
                pathlib.Path(tmp_dir),
                {
                    "sides": {
                        "left": {
                            "server_to_unitree": _translation_matrix(0.0, 0.0, 0.0),
                            "tcp_to_wrist": _translation_matrix(0.0, 0.0, 0.0),
                        }
                    }
                },
            )

            with self.assertRaises(ValueError):
                load_pose_transformer(
                    enable_motion=True,
                    transform_config_path=str(missing_side_path),
                    arm_side="both",
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_tool_path = _write_config(
                pathlib.Path(tmp_dir),
                {
                    "sides": {
                        "left": {
                            "server_to_unitree": _translation_matrix(0.0, 0.0, 0.0),
                        },
                        "right": {
                            "server_to_unitree": _translation_matrix(0.0, 0.0, 0.0),
                            "tcp_to_wrist": _translation_matrix(0.0, 0.0, 0.0),
                        },
                    }
                },
            )

            with self.assertRaises(ValueError):
                load_pose_transformer(
                    enable_motion=True,
                    transform_config_path=str(missing_tool_path),
                    arm_side="left",
                )

    def test_loader_allows_single_arm_config_for_single_arm_motion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = _write_config(
                pathlib.Path(tmp_dir),
                {
                    "sides": {
                        "left": {
                            "server_to_unitree": _translation_matrix(1.0, 0.0, 0.0),
                            "tcp_to_wrist": _translation_matrix(0.0, 0.0, 0.0),
                        }
                    }
                },
            )

            transformer = load_pose_transformer(
                enable_motion=True,
                transform_config_path=str(config_path),
                arm_side="left",
            )

        pose = transformer.action_to_unitree("left", [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        _assert_vector_close(self, [pose[0][3], pose[1][3], pose[2][3]], [1.5, 0.0, 0.0])
        with self.assertRaises(ValueError):
            transformer.action_to_unitree("right", [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def test_dry_run_without_config_returns_identity_disabled_transformer(self):
        transformer = load_pose_transformer(enable_motion=False, transform_config_path=None)
        pose7 = [0.1, 0.2, -0.3, 0.0, 0.0, 0.0, 1.0]

        self.assertIsInstance(transformer, PoseTransformer)
        self.assertFalse(transformer.enabled)
        self.assertEqual(transformer.status, "disabled")
        _assert_matrix_close(
            self,
            transformer.action_to_unitree("left", pose7),
            pose7_xyzw_to_matrix(pose7),
        )
        _assert_matrix_close(
            self,
            pose7_xyzw_to_matrix(transformer.observation_to_server("right", pose7_xyzw_to_matrix(pose7))),
            pose7_xyzw_to_matrix(pose7),
        )

    def test_disabled_transformer_accepts_array_like_wrist_matrix_for_observation(self):
        transformer = load_pose_transformer(enable_motion=False, transform_config_path=None)
        pose7 = [0.1, 0.2, -0.3, 0.0, 0.0, 0.0, 1.0]
        wrist_matrix = _ArrayLikeMatrix(pose7_xyzw_to_matrix(pose7))

        recovered_pose7 = transformer.observation_to_server("right", wrist_matrix)

        _assert_matrix_close(
            self,
            pose7_xyzw_to_matrix(recovered_pose7),
            pose7_xyzw_to_matrix(pose7),
        )

    def test_per_side_action_transform_applies_server_to_unitree_and_tcp_to_wrist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = _write_config(
                pathlib.Path(tmp_dir),
                {
                    "sides": {
                        "left": {
                            "server_to_unitree": _translation_matrix(10.0, 0.0, 0.0),
                            "tcp_to_wrist": _translation_matrix(0.0, 1.0, 0.0),
                        },
                        "right": {
                            "server_to_unitree": _translation_matrix(-10.0, 0.0, 0.0),
                            "tcp_to_wrist": _translation_matrix(0.0, -1.0, 0.0),
                        },
                    }
                },
            )
            transformer = load_pose_transformer(enable_motion=True, transform_config_path=str(config_path))

        pose7 = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        left_pose = transformer.action_to_unitree("left", pose7)
        right_pose = transformer.action_to_unitree("right", pose7)

        _assert_vector_close(self, [left_pose[0][3], left_pose[1][3], left_pose[2][3]], [11.0, 3.0, 3.0])
        _assert_vector_close(self, [right_pose[0][3], right_pose[1][3], right_pose[2][3]], [-9.0, 1.0, 3.0])

    def test_observation_transform_uses_inverse_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = _write_config(
                pathlib.Path(tmp_dir),
                {
                    "sides": {
                        "left": {
                            "server_to_unitree": _translation_matrix(1.5, -2.0, 0.25),
                            "tcp_to_wrist": _translation_matrix(0.0, 0.0, 0.5),
                        },
                        "right": {
                            "server_to_unitree": _translation_matrix(-1.0, 0.5, 0.0),
                            "tcp_to_wrist": _translation_matrix(0.0, 0.25, 0.0),
                        },
                    }
                },
            )
            transformer = load_pose_transformer(enable_motion=True, transform_config_path=str(config_path))

        original_server_pose7 = [0.6, -0.4, 1.1, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
        unitree_wrist = transformer.action_to_unitree("left", original_server_pose7)

        recovered_pose7 = transformer.observation_to_server("left", unitree_wrist)

        _assert_matrix_close(
            self,
            pose7_xyzw_to_matrix(recovered_pose7),
            pose7_xyzw_to_matrix(original_server_pose7),
        )

    def test_invalid_matrix_rejected(self):
        with self.assertRaises(ValueError):
            validate_matrix4x4([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        invalid = _translation_matrix(0.0, 0.0, 0.0)
        invalid[0][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_matrix4x4(invalid)


if __name__ == "__main__":
    unittest.main()
