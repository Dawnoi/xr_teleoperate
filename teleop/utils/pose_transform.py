import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _identity_matrix() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_matrix4x4(matrix: Any) -> List[List[float]]:
    if not isinstance(matrix, (list, tuple)) and hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 4:
        raise ValueError("matrix must be a 4x4 list")
    validated: List[List[float]] = []
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError("matrix must be a 4x4 list")
        validated_row: List[float] = []
        for value in row:
            if not _is_finite_number(value):
                raise ValueError("matrix must contain only finite values")
            validated_row.append(float(value))
        validated.append(validated_row)
    return validated


def _matmul4(left: List[List[float]], right: List[List[float]]) -> List[List[float]]:
    return [
        [
            sum(left[row][idx] * right[idx][col] for idx in range(4))
            for col in range(4)
        ]
        for row in range(4)
    ]


def _transpose3(rotation: List[List[float]]) -> List[List[float]]:
    return [
        [rotation[col][row] for col in range(3)]
        for row in range(3)
    ]


def _invert_rigid_transform(matrix: List[List[float]]) -> List[List[float]]:
    validated = validate_matrix4x4(matrix)
    rotation = [row[:3] for row in validated[:3]]
    translation = [validated[row][3] for row in range(3)]
    rotation_t = _transpose3(rotation)
    inverted = _identity_matrix()
    for row in range(3):
        for col in range(3):
            inverted[row][col] = rotation_t[row][col]
        inverted[row][3] = -sum(rotation_t[row][idx] * translation[idx] for idx in range(3))
    return inverted


def pose7_xyzw_to_matrix(pose7: Any) -> List[List[float]]:
    if not isinstance(pose7, (list, tuple)) and hasattr(pose7, "tolist"):
        pose7 = pose7.tolist()
    if not isinstance(pose7, (list, tuple)) or len(pose7) != 7:
        raise ValueError("pose7 must be [x, y, z, qx, qy, qz, qw]")
    values = [float(value) for value in pose7]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pose7 must contain finite values")
    x, y, z, qx, qy, qz, qw = values
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("quaternion norm must be non-zero")
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), x],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), y],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_to_pose7_xyzw(matrix: Any) -> List[float]:
    validated = validate_matrix4x4(matrix)
    r00, r01, r02 = validated[0][:3]
    r10, r11, r12 = validated[1][:3]
    r20, r21, r22 = validated[2][:3]
    trace = r00 + r11 + r22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("matrix quaternion norm must be non-zero")
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    return [
        validated[0][3],
        validated[1][3],
        validated[2][3],
        qx,
        qy,
        qz,
        qw,
    ]


def _normalized_side(side: str) -> str:
    normalized = str(side).strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"unsupported side: {side}")
    return normalized


def _parse_side_config(
    side_name: str,
    side_config: Dict[str, Any],
    *,
    require_explicit: bool = False,
) -> Dict[str, List[List[float]]]:
    if not isinstance(side_config, dict):
        raise ValueError(f"side config for {side_name} must be a mapping")

    if require_explicit and "server_to_unitree" not in side_config:
        raise ValueError(f"side config for {side_name} must explicitly define server_to_unitree")
    server_to_unitree = validate_matrix4x4(side_config.get("server_to_unitree", _identity_matrix()))
    unitree_to_server = side_config.get("unitree_to_server")
    if unitree_to_server is None:
        unitree_to_server_matrix = _invert_rigid_transform(server_to_unitree)
    else:
        unitree_to_server_matrix = validate_matrix4x4(unitree_to_server)

    has_tcp_to_wrist = "tcp_to_wrist" in side_config
    has_wrist_to_tcp = "wrist_to_tcp" in side_config
    if require_explicit and not (has_tcp_to_wrist or has_wrist_to_tcp):
        raise ValueError(f"side config for {side_name} must explicitly define tcp_to_wrist or wrist_to_tcp")
    if has_tcp_to_wrist:
        tcp_to_wrist = validate_matrix4x4(side_config["tcp_to_wrist"])
    elif has_wrist_to_tcp:
        tcp_to_wrist = _invert_rigid_transform(side_config["wrist_to_tcp"])
    else:
        tcp_to_wrist = validate_matrix4x4(_identity_matrix())
    wrist_to_tcp = side_config.get("wrist_to_tcp")
    if wrist_to_tcp is None:
        wrist_to_tcp_matrix = _invert_rigid_transform(tcp_to_wrist)
    else:
        wrist_to_tcp_matrix = validate_matrix4x4(wrist_to_tcp)

    return {
        "server_to_unitree": server_to_unitree,
        "unitree_to_server": unitree_to_server_matrix,
        "tcp_to_wrist": tcp_to_wrist,
        "wrist_to_tcp": wrist_to_tcp_matrix,
    }


@dataclass
class PoseTransformer:
    side_matrices: Dict[str, Dict[str, List[List[float]]]]
    enabled: bool
    status: str

    @classmethod
    def disabled_identity(cls) -> "PoseTransformer":
        identity_side = _parse_side_config("identity", {})
        return cls(
            side_matrices={"left": identity_side, "right": identity_side},
            enabled=False,
            status="disabled",
        )

    def _side(self, side: str) -> Dict[str, List[List[float]]]:
        normalized = _normalized_side(side)
        try:
            return self.side_matrices[normalized]
        except KeyError as exc:
            raise ValueError(f"missing transform config for side: {normalized}") from exc

    def action_to_unitree(self, side: str, pose7: Any) -> List[List[float]]:
        server_tcp = pose7_xyzw_to_matrix(pose7)
        side_matrices = self._side(side)
        return _matmul4(
            _matmul4(side_matrices["server_to_unitree"], server_tcp),
            side_matrices["tcp_to_wrist"],
        )

    def observation_to_server(self, side: str, unitree_wrist_matrix: Any) -> List[float]:
        unitree_wrist = validate_matrix4x4(unitree_wrist_matrix)
        side_matrices = self._side(side)
        server_tcp = _matmul4(
            _matmul4(side_matrices["unitree_to_server"], unitree_wrist),
            side_matrices["wrist_to_tcp"],
        )
        return matrix_to_pose7_xyzw(server_tcp)


def load_pose_transformer(
    enable_motion: bool,
    transform_config_path: Optional[str],
    arm_side: str = "both",
) -> PoseTransformer:
    if not enable_motion:
        return PoseTransformer.disabled_identity()

    normalized_arm_side = str(arm_side or "both").strip().lower()
    if normalized_arm_side not in {"left", "right", "both"}:
        raise ValueError(f"unsupported arm_side: {arm_side!r}")
    active_sides = ["left", "right"] if normalized_arm_side == "both" else [normalized_arm_side]

    if not transform_config_path:
        raise ValueError("transform_config_path is required when motion is enabled")

    with open(transform_config_path, "r", encoding="utf-8") as handle:
        raw_config = json.load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError("transform config root must be an object")
    sides = raw_config.get("sides")
    if not isinstance(sides, dict):
        raise ValueError("transform config must contain a sides object")

    side_matrices: Dict[str, Dict[str, List[List[float]]]] = {}
    for side in active_sides:
        side_config = sides.get(side)
        if side_config is None:
            raise ValueError(f"transform config must explicitly contain active side: {side}")
        side_matrices[side] = _parse_side_config(side, side_config, require_explicit=True)

    return PoseTransformer(
        side_matrices=side_matrices,
        enabled=True,
        status="enabled",
    )
