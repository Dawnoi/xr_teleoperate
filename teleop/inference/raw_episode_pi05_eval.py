from __future__ import annotations

from contextlib import nullcontext
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


_PREFERRED_PYTHON_PATHS = [
    "/home/dx/miniconda3/envs/tv/lib/python3.10/site-packages",
    "/home/dx/miniconda3/envs/tv/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages",
    "/home/dx/miniconda3/envs/gmr/lib/python3.10/site-packages",
    "/home/luopengcheng/miniconda3/envs/tv/lib/python3.10/site-packages",
    "/home/luopengcheng/miniconda3/envs/tv/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages",
    "/home/luopengcheng/miniconda3/envs/gmr/lib/python3.10/site-packages",
]
for _python_path in reversed(_PREFERRED_PYTHON_PATHS):
    if os.path.isdir(_python_path):
        if _python_path in sys.path:
            sys.path.remove(_python_path)
        sys.path.insert(0, _python_path)

from teleop.inference.online_session import (
    CameraSample,
    OnlineInferenceConfig,
    OnlineInferenceSession,
    RobotStateSample,
)
from teleop.inference.pi05_protocol import build_pi05_observation_payload
from teleop.inference.pose_transform import matrix_to_pose9_rot6d
from teleop.inference.protocol import HttpJsonInferenceTransport


ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

DEX1_JOINT_NAMES = {
    "left": ["left_dex1_finger_joint_1", "left_dex1_finger_joint_2"],
    "right": ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"],
}

DEX1_OPEN_Q = 0.0245
DEX1_CLOSED_Q = 0.0
DEX1_REAL_MAX_Q = 5.40
G1_IK_EE_OFFSET = np.array([0.05, 0.0, 0.0], dtype=np.float64)
RGBA_GT_LEFT = np.array([0.05, 0.55, 1.00, 1.0], dtype=np.float32)
RGBA_PRED_LEFT = np.array([1.00, 0.20, 0.12, 1.0], dtype=np.float32)
RGBA_GT_RIGHT = np.array([0.05, 0.95, 0.85, 1.0], dtype=np.float32)
RGBA_PRED_RIGHT = np.array([1.00, 0.68, 0.10, 1.0], dtype=np.float32)
RGBA_GT_CURRENT = np.array([0.85, 0.95, 1.00, 1.0], dtype=np.float32)
RGBA_PRED_CURRENT = np.array([1.00, 0.96, 0.65, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class RawEpisodePayload:
    episode_dir: Path
    text: dict[str, Any] | None
    items: list[dict[str, Any]]


@dataclass(frozen=True)
class RawEpisodeWindow:
    anchor_frame: int
    obs_indices: list[int]
    sample_monotonic_ns_history: list[int]
    left_state_pose_history: list[list[list[float]]]
    right_state_pose_history: list[list[list[float]]]
    left_gripper_history: list[float]
    right_gripper_history: list[float]
    head_image_paths: list[Path]
    left_wrist_image_paths: list[Path]
    right_wrist_image_paths: list[Path]
    gt_left_action_pose_targets: list[list[list[float]]]
    gt_right_action_pose_targets: list[list[list[float]]]


@dataclass
class MujocoEvalRuntime:
    model: Any
    data: Any
    base_qpos: np.ndarray
    arm_joint_qpos_indices: list[int]
    dex1_qpos_indices: dict[str, list[int] | None] | None
    torso_body_id: int
    left_wrist_body_id: int
    right_wrist_body_id: int
    arm_ik: Any
    control_frequency: float
    max_arm_joint_speed: float
    apply_speed_limit: bool


@dataclass(frozen=True)
class EvalVideoConfig:
    path: Path
    fps: float
    step_repeat: int
    width: int
    height: int
    camera_distance: float
    camera_azimuth: float
    camera_elevation: float
    camera_lookat: tuple[float, float, float]


class MujocoEvalVideoSink:
    def __init__(self, runtime: MujocoEvalRuntime, config: EvalVideoConfig) -> None:
        import mujoco as mj

        if config.fps <= 0.0:
            raise ValueError("video fps must be positive")
        if config.step_repeat <= 0:
            raise ValueError("video step_repeat must be positive")
        if config.width <= 0 or config.height <= 0:
            raise ValueError("video width and height must be positive")
        if config.camera_distance <= 0.0:
            raise ValueError("video camera distance must be positive")

        self.runtime = runtime
        self.config = config
        self._frame_count = 0
        self.config.path.parent.mkdir(parents=True, exist_ok=True)

        self._camera = mj.MjvCamera()
        self._camera.type = mj.mjtCamera.mjCAMERA_FREE
        self._camera.distance = float(config.camera_distance)
        self._camera.azimuth = float(config.camera_azimuth)
        self._camera.elevation = float(config.camera_elevation)
        self._camera.lookat[:] = np.asarray(config.camera_lookat, dtype=np.float64)

        runtime.model.vis.global_.offwidth = max(int(runtime.model.vis.global_.offwidth), int(config.width))
        runtime.model.vis.global_.offheight = max(int(runtime.model.vis.global_.offheight), int(config.height))
        self._renderer = mj.Renderer(runtime.model, height=int(config.height), width=int(config.width))
        self._writer = cv2.VideoWriter(
            str(config.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(config.fps),
            (int(config.width), int(config.height)),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {config.path}")

    def __enter__(self) -> "MujocoEvalVideoSink":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        return int(self._frame_count)

    def close(self) -> None:
        self._writer.release()
        if hasattr(self._renderer, "close"):
            self._renderer.close()

    def assert_has_frames(self) -> None:
        if self._frame_count <= 0:
            raise RuntimeError(f"video sink wrote no frames: {self.config.path}")

    def write_step(self, runtime: MujocoEvalRuntime, overlay: dict[str, Any]) -> None:
        self._renderer.update_scene(runtime.data, camera=self._camera)
        self._overlay_trajectories(self._renderer.scene, overlay)
        rgb_frame = np.asarray(self._renderer.render(), dtype=np.uint8)
        expected_shape = (int(self.config.height), int(self.config.width), 3)
        if rgb_frame.shape != expected_shape:
            raise ValueError(f"renderer returned shape {rgb_frame.shape}, expected {expected_shape}")
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        self._draw_overlay(bgr_frame, overlay)
        for _ in range(int(self.config.step_repeat)):
            self._writer.write(bgr_frame)
            self._frame_count += 1

    def _draw_overlay(self, frame: np.ndarray, overlay: dict[str, Any]) -> None:
        lines = [
            f"anchor={int(overlay['anchor_frame'])} step={int(overlay['step_index'])} frame={int(overlay['frame_index'])}",
            f"pred_xyz={float(overlay['pred_xyz_error_m']):.4f}m gt_vs_pred",
            f"pred_rot={float(overlay['pred_rot_error_rad']):.4f}rad gt_vs_pred",
            f"traj_step={int(overlay.get('trajectory_step_index', overlay['step_index']))} repeat={int(self.config.step_repeat)}",
        ]
        panel_top = 18
        panel_left = 18
        panel_width = 640
        panel_height = 138
        cv2.rectangle(
            frame,
            (panel_left, panel_top),
            (panel_left + panel_width, panel_top + panel_height),
            (8, 8, 8),
            thickness=-1,
        )
        for line_index, text in enumerate(lines):
            y = panel_top + 30 + line_index * 30
            cv2.putText(
                frame,
                text,
                (panel_left + 14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

    def _overlay_trajectories(self, scene: Any, overlay: dict[str, Any]) -> None:
        gt_left_points = _trajectory_points_array(overlay.get("gt_left_points"))
        gt_right_points = _trajectory_points_array(overlay.get("gt_right_points"))
        pred_left_points = _trajectory_points_array(overlay.get("pred_left_points"))
        pred_right_points = _trajectory_points_array(overlay.get("pred_right_points"))
        current_index = int(overlay.get("trajectory_step_index", overlay.get("step_index", 0)))

        _overlay_trajectory(scene, gt_left_points, RGBA_GT_LEFT, radius=0.0048)
        _overlay_trajectory(scene, gt_right_points, RGBA_GT_RIGHT, radius=0.0048)
        _overlay_trajectory(scene, pred_left_points, RGBA_PRED_LEFT, radius=0.0062)
        _overlay_trajectory(scene, pred_right_points, RGBA_PRED_RIGHT, radius=0.0062)

        _add_current_marker(scene, gt_left_points, current_index, radius=0.013, rgba=RGBA_GT_CURRENT)
        _add_current_marker(scene, gt_right_points, current_index, radius=0.013, rgba=RGBA_GT_CURRENT)
        _add_current_marker(scene, pred_left_points, current_index, radius=0.016, rgba=RGBA_PRED_CURRENT)
        _add_current_marker(scene, pred_right_points, current_index, radius=0.016, rgba=RGBA_PRED_CURRENT)


class PinocchioDualArmIK:
    def __init__(
        self,
        reduced_robot: Any,
        left_frame_id: int,
        right_frame_id: int,
    ) -> None:
        self.reduced_robot = reduced_robot
        self.L_hand_id = int(left_frame_id)
        self.R_hand_id = int(right_frame_id)
        self.init_data = np.zeros(self.reduced_robot.model.nq, dtype=np.float64)

    @classmethod
    def create_g1_29(cls) -> "PinocchioDualArmIK":
        import pinocchio as pin

        repo_root = Path(__file__).resolve().parents[2]
        urdf_path = repo_root / "assets/g1/g1_body29_hand14.urdf"
        model_dir = str(repo_root / "assets/g1")
        robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), model_dir)
        joints_to_lock = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]
        reduced_robot = robot.buildReducedRobot(
            list_of_joints_to_lock=joints_to_lock,
            reference_configuration=np.array([0.0] * robot.model.nq),
        )
        reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0], dtype=np.float64)),
                pin.FrameType.OP_FRAME,
            )
        )
        reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0], dtype=np.float64)),
                pin.FrameType.OP_FRAME,
            )
        )
        reduced_robot.data = reduced_robot.model.createData()
        left_frame_id = int(reduced_robot.model.getFrameId("L_ee"))
        right_frame_id = int(reduced_robot.model.getFrameId("R_ee"))
        return cls(reduced_robot=reduced_robot, left_frame_id=left_frame_id, right_frame_id=right_frame_id)

    def solve_ik(
        self,
        left_wrist: Any,
        right_wrist: Any,
        current_lr_arm_motor_q: Any = None,
        current_lr_arm_motor_dq: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import pinocchio as pin
        from scipy.optimize import least_squares

        if current_lr_arm_motor_q is not None:
            self.init_data = _vector_from_field(
                np.asarray(current_lr_arm_motor_q, dtype=np.float64).reshape(-1).tolist(),
                self.reduced_robot.model.nq,
                "current_lr_arm_motor_q",
            )
        target_left = _matrix4x4(left_wrist, "left_wrist")
        target_right = _matrix4x4(right_wrist, "right_wrist")
        q0 = self.init_data.copy()
        model = self.reduced_robot.model
        data = self.reduced_robot.data
        lower = np.asarray(model.lowerPositionLimit, dtype=np.float64)
        upper = np.asarray(model.upperPositionLimit, dtype=np.float64)

        def residual(q: np.ndarray) -> np.ndarray:
            pin.framesForwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            left_pose = data.oMf[self.L_hand_id]
            right_pose = data.oMf[self.R_hand_id]
            left_translation_residual = np.asarray(left_pose.translation - target_left[:3, 3], dtype=np.float64)
            right_translation_residual = np.asarray(right_pose.translation - target_right[:3, 3], dtype=np.float64)
            left_rotation_residual = np.asarray(
                pin.log3(np.asarray(target_left[:3, :3] @ left_pose.rotation.T, dtype=np.float64)),
                dtype=np.float64,
            )
            right_rotation_residual = np.asarray(
                pin.log3(np.asarray(target_right[:3, :3] @ right_pose.rotation.T, dtype=np.float64)),
                dtype=np.float64,
            )
            regularization = 1e-2 * (q - q0)
            return np.concatenate(
                [
                    left_translation_residual,
                    left_rotation_residual,
                    right_translation_residual,
                    right_rotation_residual,
                    regularization,
                ]
            )

        result = least_squares(
            residual,
            q0,
            bounds=(lower, upper),
            xtol=1e-5,
            ftol=1e-5,
            gtol=1e-5,
            max_nfev=80,
        )
        if result.status <= 0:
            raise RuntimeError(f"pinocchio IK failed: status={result.status} message={result.message}")
        sol_q = np.asarray(result.x, dtype=np.float64)
        self.init_data = sol_q.copy()
        tauff = np.asarray(
            pin.rnea(
                model,
                data,
                sol_q,
                np.zeros(model.nv, dtype=np.float64),
                np.zeros(model.nv, dtype=np.float64),
            ),
            dtype=np.float64,
        )
        return sol_q, tauff


def load_raw_episode_payload(dataset_root: str | Path, episode_index: int) -> RawEpisodePayload:
    root = Path(dataset_root)
    episode_dir = root / f"episode_{int(episode_index):04d}"
    data_path = episode_dir / "data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"raw episode data.json not found: {data_path}")
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    items = payload.get("data")
    if not isinstance(items, list) or not items:
        raise ValueError(f"raw episode data list must be non-empty: {data_path}")
    text = payload.get("text")
    if text is not None and not isinstance(text, dict):
        raise ValueError(f"raw episode text field must be an object when present: {data_path}")
    return RawEpisodePayload(episode_dir=episode_dir, text=text, items=items)


def build_prompt_from_payload(payload: RawEpisodePayload) -> str:
    text = payload.text or {}
    if not text:
        return ""
    parts = []
    for key in ("goal", "desc", "steps"):
        value = text.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts).strip()


def build_anchor_indices(
    episode_len: int,
    n_obs_steps: int,
    chunk_size: int,
    stride: int,
    start_frame: int | None,
    end_frame: int | None,
) -> list[int]:
    if episode_len <= 0:
        raise ValueError("episode_len must be positive")
    if n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    first = max(n_obs_steps - 1, 0 if start_frame is None else int(start_frame))
    last_exclusive = episode_len - chunk_size + 1
    if end_frame is not None:
        last_exclusive = min(last_exclusive, int(end_frame))
    indices = list(range(first, last_exclusive, stride))
    if not indices:
        raise ValueError("no anchor frames selected")
    return indices


def make_gt_window(payload: RawEpisodePayload, anchor_frame: int, n_obs_steps: int, chunk_size: int) -> RawEpisodeWindow:
    items = payload.items
    if anchor_frame < n_obs_steps - 1:
        raise ValueError("anchor_frame is too early for requested history")
    if anchor_frame + chunk_size > len(items):
        raise ValueError("anchor_frame + chunk_size exceeds episode length")

    obs_indices = list(range(anchor_frame - n_obs_steps + 1, anchor_frame + 1))
    target_indices = list(range(anchor_frame, anchor_frame + chunk_size))

    return RawEpisodeWindow(
        anchor_frame=anchor_frame,
        obs_indices=obs_indices,
        sample_monotonic_ns_history=[int(items[idx]["timestamps"]["sample_monotonic_ns"]) for idx in obs_indices],
        left_state_pose_history=[items[idx]["states"]["left_arm"]["pose"]["matrix4x4"] for idx in obs_indices],
        right_state_pose_history=[items[idx]["states"]["right_arm"]["pose"]["matrix4x4"] for idx in obs_indices],
        left_gripper_history=[float(items[idx]["states"]["left_ee"]["qpos"][0]) for idx in obs_indices],
        right_gripper_history=[float(items[idx]["states"]["right_ee"]["qpos"][0]) for idx in obs_indices],
        head_image_paths=[payload.episode_dir / items[idx]["colors"]["head"] for idx in obs_indices],
        left_wrist_image_paths=[payload.episode_dir / items[idx]["colors"]["left_wrist"] for idx in obs_indices],
        right_wrist_image_paths=[payload.episode_dir / items[idx]["colors"]["right_wrist"] for idx in obs_indices],
        gt_left_action_pose_targets=[items[idx]["actions"]["left_arm"]["pose"]["matrix4x4"] for idx in target_indices],
        gt_right_action_pose_targets=[items[idx]["actions"]["right_arm"]["pose"]["matrix4x4"] for idx in target_indices],
    )


def _read_bgr_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"decoded image must be HxWx3: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_pi05_observation_from_window(window: RawEpisodeWindow, prompt: str) -> dict[str, Any]:
    images = {
        "head_fpv": _read_bgr_rgb(window.head_image_paths[-1]),
        "left_hand": _read_bgr_rgb(window.left_wrist_image_paths[-1]),
        "right_hand": _read_bgr_rgb(window.right_wrist_image_paths[-1]),
        "third_front": _read_bgr_rgb(window.right_wrist_image_paths[-1]),
    }
    return build_pi05_observation_payload(
        images=images,
        poses_left=[matrix_to_pose9_rot6d(pose) for pose in window.left_state_pose_history],
        grippers_left=[[value] for value in window.left_gripper_history],
        poses_right=[matrix_to_pose9_rot6d(pose) for pose in window.right_state_pose_history],
        grippers_right=[[value] for value in window.right_gripper_history],
        prompt=prompt,
    )


def build_session_inputs_from_window(window: RawEpisodeWindow) -> tuple[RobotStateSample, list[CameraSample]]:
    host_monotonic_ns = int(window.sample_monotonic_ns_history[-1])
    state_sample = RobotStateSample(
        host_monotonic_ns=host_monotonic_ns,
        left_pose=np.asarray(window.left_state_pose_history[-1], dtype=np.float64),
        right_pose=np.asarray(window.right_state_pose_history[-1], dtype=np.float64),
        left_gripper_width=float(window.left_gripper_history[-1]),
        right_gripper_width=float(window.right_gripper_history[-1]),
    )
    camera_samples = [
        CameraSample(name="head", frame=_read_bgr_rgb(window.head_image_paths[-1]), host_monotonic_ns=host_monotonic_ns),
        CameraSample(name="left_wrist", frame=_read_bgr_rgb(window.left_wrist_image_paths[-1]), host_monotonic_ns=host_monotonic_ns),
        CameraSample(name="right_wrist", frame=_read_bgr_rgb(window.right_wrist_image_paths[-1]), host_monotonic_ns=host_monotonic_ns),
    ]
    return state_sample, camera_samples


def build_session_input_history_from_window(window: RawEpisodeWindow) -> list[tuple[RobotStateSample, list[CameraSample]]]:
    history: list[tuple[RobotStateSample, list[CameraSample]]] = []
    for index in range(len(window.obs_indices)):
        host_monotonic_ns = int(window.sample_monotonic_ns_history[index])
        state_sample = RobotStateSample(
            host_monotonic_ns=host_monotonic_ns,
            left_pose=np.asarray(window.left_state_pose_history[index], dtype=np.float64),
            right_pose=np.asarray(window.right_state_pose_history[index], dtype=np.float64),
            left_gripper_width=float(window.left_gripper_history[index]),
            right_gripper_width=float(window.right_gripper_history[index]),
        )
        camera_samples = [
            CameraSample(name="head", frame=_read_bgr_rgb(window.head_image_paths[index]), host_monotonic_ns=host_monotonic_ns),
            CameraSample(
                name="left_wrist",
                frame=_read_bgr_rgb(window.left_wrist_image_paths[index]),
                host_monotonic_ns=host_monotonic_ns,
            ),
            CameraSample(
                name="right_wrist",
                frame=_read_bgr_rgb(window.right_wrist_image_paths[index]),
                host_monotonic_ns=host_monotonic_ns,
            ),
        ]
        history.append((state_sample, camera_samples))
    return history


def pose_error_metrics(pred_pose: Any, gt_pose: Any) -> dict[str, float]:
    pred = np.asarray(pred_pose, dtype=np.float64)
    gt = np.asarray(gt_pose, dtype=np.float64)
    if pred.shape != (4, 4) or gt.shape != (4, 4):
        raise ValueError("pose_error_metrics expects 4x4 poses")

    delta_xyz = pred[:3, 3] - gt[:3, 3]
    xyz_error_m = float(np.linalg.norm(delta_xyz))

    rel_rot = pred[:3, :3] @ gt[:3, :3].T
    trace = float(np.trace(rel_rot))
    cos_theta = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    rot_error_rad = float(np.arccos(cos_theta))
    return {
        "xyz_error_m": xyz_error_m,
        "rot_error_rad": rot_error_rad,
    }


def summarize_step_records(episode_index: int, records: list[dict[str, Any]], num_requests: int, num_anchor_frames: int) -> dict[str, Any]:
    if not records:
        raise ValueError("records must be non-empty")

    pred_xyz_values = [0.5 * (record["pred_left_xyz_error_m"] + record["pred_right_xyz_error_m"]) for record in records]
    exec_xyz_values = [0.5 * (record["exec_left_xyz_error_m"] + record["exec_right_xyz_error_m"]) for record in records]
    pred_rot_values = [0.5 * (record["pred_left_rot_error_rad"] + record["pred_right_rot_error_rad"]) for record in records]
    exec_rot_values = [0.5 * (record["exec_left_rot_error_rad"] + record["exec_right_rot_error_rad"]) for record in records]
    return {
        "episode": int(episode_index),
        "num_anchor_frames": int(num_anchor_frames),
        "num_requests": int(num_requests),
        "num_executed_steps": int(len(records)),
        "mean_pred_xyz_error_m": float(sum(pred_xyz_values) / len(pred_xyz_values)),
        "mean_exec_xyz_error_m": float(sum(exec_xyz_values) / len(exec_xyz_values)),
        "mean_pred_rot_error_rad": float(sum(pred_rot_values) / len(pred_rot_values)),
        "mean_exec_rot_error_rad": float(sum(exec_rot_values) / len(exec_rot_values)),
        "max_pred_xyz_error_m": float(max(pred_xyz_values)),
        "max_exec_xyz_error_m": float(max(exec_xyz_values)),
    }


def write_eval_outputs(output_dir: str | Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    steps_path = out_dir / "steps.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with steps_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary_path, steps_path


def _vector_from_field(value: Any, expected_len: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != expected_len:
        actual_len = len(value) if isinstance(value, list) else "not-list"
        raise ValueError(f"{label} must be length {expected_len}, got {actual_len}")
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains non-finite values")
    return arr


def _matrix4x4(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{label} must be 4x4")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains non-finite values")
    return arr


def _frame_arm_and_ee_qpos(item: dict[str, Any], source: str) -> tuple[np.ndarray, np.ndarray]:
    source_value = item.get(source)
    if not isinstance(source_value, dict):
        raise ValueError(f"frame missing {source}")
    left_arm = _vector_from_field(source_value["left_arm"]["qpos"], 7, f"{source}.left_arm.qpos")
    right_arm = _vector_from_field(source_value["right_arm"]["qpos"], 7, f"{source}.right_arm.qpos")
    left_ee = _vector_from_field(source_value["left_ee"]["qpos"], 1, f"{source}.left_ee.qpos")
    right_ee = _vector_from_field(source_value["right_ee"]["qpos"], 1, f"{source}.right_ee.qpos")
    arm_q = np.concatenate([left_arm, right_arm]).astype(np.float64)
    gripper_q = np.asarray([float(left_ee[0]), float(right_ee[0])], dtype=np.float64)
    return arm_q, gripper_q


def gripper_state_to_mujoco_q(gripper_q: float) -> float:
    open_ratio = float(np.clip(float(gripper_q) / DEX1_REAL_MAX_Q, 0.0, 1.0))
    return DEX1_CLOSED_Q + (DEX1_OPEN_Q - DEX1_CLOSED_Q) * open_ratio


def joint_qpos_indices(model: Any, joint_names: list[str]) -> list[int]:
    import mujoco as mj

    qpos_indices: list[int] = []
    for name in joint_names:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in model: {name}")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
    return qpos_indices


def joint_qpos_indices_if_present(model: Any, joint_names: list[str]) -> list[int] | None:
    import mujoco as mj

    qpos_indices: list[int] = []
    for name in joint_names:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            return None
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
    return qpos_indices


def make_pose(rotation: Any, translation: Any) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return pose


def invert_rigid_pose(pose: Any) -> np.ndarray:
    matrix = _matrix4x4(pose, "pose")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def mujoco_body_pose(data: Any, body_id: int) -> np.ndarray:
    xpos = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    xmat = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()
    return make_pose(xmat, xpos)


def wrist_body_pose_to_ee_pose(body_pose: Any) -> np.ndarray:
    pose = _matrix4x4(body_pose, "wrist body pose").copy()
    pose[:3, 3] = pose[:3, 3] + pose[:3, :3] @ G1_IK_EE_OFFSET
    return pose


def torso_local_pose_to_world_pose(torso_world_pose: Any, local_pose: Any) -> np.ndarray:
    torso_world = _matrix4x4(torso_world_pose, "torso_world_pose")
    local = _matrix4x4(local_pose, "local_pose")
    return torso_world @ local


def _trajectory_points_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"trajectory points must have shape (N, 3), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("trajectory points must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("trajectory points contain non-finite values")
    return arr


def _add_sphere(scene: Any, pos: Any, radius: float, rgba: Any) -> None:
    import mujoco as mj

    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(f"MuJoCo scene geom capacity exceeded: {scene.ngeom} >= {scene.maxgeom}")
    mj.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mj.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_segment(scene: Any, start: Any, end: Any, radius: float, rgba: Any) -> None:
    import mujoco as mj

    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(f"MuJoCo scene geom capacity exceeded: {scene.ngeom} >= {scene.maxgeom}")
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        mj.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mj.mjv_connector(
        geom,
        mj.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
    scene.ngeom += 1


def _overlay_trajectory(scene: Any, points: Any, rgba: Any, radius: float) -> None:
    points_arr = _trajectory_points_array(points)
    for index in range(points_arr.shape[0] - 1):
        _add_segment(scene, points_arr[index], points_arr[index + 1], radius, rgba)
    for index, point in enumerate(points_arr):
        point_radius = radius * (1.8 if index in {0, points_arr.shape[0] - 1} else 1.15)
        _add_sphere(scene, point, point_radius, rgba)


def _add_current_marker(scene: Any, points: Any, step_index: int, radius: float, rgba: Any) -> None:
    points_arr = _trajectory_points_array(points)
    clamped_index = int(np.clip(int(step_index), 0, points_arr.shape[0] - 1))
    _add_sphere(scene, points_arr[clamped_index], radius, rgba)


def reset_arm_ik_state(arm_ik: Any, arm_q: Any) -> None:
    q = _vector_from_field(np.asarray(arm_q, dtype=np.float64).reshape(-1).tolist(), 14, "arm_q")
    if hasattr(arm_ik, "init_data"):
        arm_ik.init_data = q.copy()
    smooth_filter = getattr(arm_ik, "smooth_filter", None)
    if smooth_filter is not None:
        window_size = int(getattr(smooth_filter, "_window_size"))
        smooth_filter._data_queue = [q.copy() for _ in range(window_size)]
        smooth_filter._filtered_data = q.copy()


def build_http_pi05_session(
    base_url: str,
    prompt: str,
    n_obs_steps: int,
    camera_freq: float,
    action_step_sec: float,
    response_timeout_sec: float,
    jpeg_quality: int = 85,
) -> OnlineInferenceSession:
    transport = HttpJsonInferenceTransport.connect(
        base_url=base_url,
        handshake_path="/handshake",
        infer_path="/infer",
        timeout_sec=response_timeout_sec,
        handshake_payload={
            "action_dim": 20,
            "action_space": "pose20",
            "robot": "nero_dual_arm",
            "transport": "http",
        },
    )
    session = OnlineInferenceSession(
        config=OnlineInferenceConfig(
            arm_side="both",
            protocol_profile="pi05_dual_arm_20d",
            task_prompt=prompt,
            n_obs_steps=n_obs_steps,
            camera_freq=camera_freq,
            action_step_sec=action_step_sec,
            chunk_step_mode="per_tick",
            interpolation_interval_sec=max(action_step_sec / 10.0, 1e-3),
            post_action_delay_ms=0,
            response_timeout_sec=response_timeout_sec,
            jpeg_quality=jpeg_quality,
            enable_motion=False,
            dry_run=True,
        ),
        transport=transport,
        pose_transformer=None,
    )
    session.set_required_camera_names(["head", "left_wrist", "right_wrist"])
    return session


def run_session_for_window(session: OnlineInferenceSession, window: RawEpisodeWindow) -> list[Any]:
    history_inputs = build_session_input_history_from_window(window)
    last_state_sample = None
    last_camera_samples = None
    for state_sample, camera_samples in history_inputs:
        session.tick(state_sample=state_sample, camera_samples=camera_samples)
        last_state_sample = state_sample
        last_camera_samples = camera_samples

    if last_state_sample is None or last_camera_samples is None:
        raise ValueError("window history must be non-empty")

    first_step = None
    for _ in range(64):
        step = session.tick(state_sample=last_state_sample, camera_samples=last_camera_samples)
        if step.status == "executing_chunk":
            first_step = step
            break
        if step.status == "failed":
            raise ValueError(f"online inference session failed: {session.error}")
    if first_step is None:
        raise ValueError(f"session did not reach executing_chunk, last status={session.status}")

    chunk_size = int(first_step.metadata.get("online_chunk_size", 0))
    if chunk_size <= 0:
        raise ValueError("online inference returned non-positive chunk size")

    steps = [first_step]
    while len(steps) < chunk_size:
        step = session.tick(state_sample=last_state_sample, camera_samples=last_camera_samples)
        if step.status != "executing_chunk":
            raise ValueError(f"expected executing_chunk while draining action chunk, got {step.status}")
        steps.append(step)
    return steps


def build_mujoco_eval_runtime(
    control_frequency: float,
    max_arm_joint_speed: float,
    apply_speed_limit: bool = False,
) -> MujocoEvalRuntime:
    import mujoco as mj

    from teleop.sim.g1d_mujoco_builder import prepare_g1d_mobile_scene

    if control_frequency <= 0.0:
        raise ValueError("control_frequency must be positive")
    if max_arm_joint_speed <= 0.0:
        raise ValueError("max_arm_joint_speed must be positive")

    xml_path = prepare_g1d_mobile_scene(use_dex1=True)
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    arm_qpos_indices = joint_qpos_indices(model, ARM_JOINT_NAMES)
    dex1_qpos_indices = {
        side: joint_qpos_indices_if_present(model, names)
        for side, names in DEX1_JOINT_NAMES.items()
    }
    if dex1_qpos_indices["left"] is None or dex1_qpos_indices["right"] is None:
        raise ValueError(f"XML has no complete Dex1 joint set: {xml_path}")
    torso_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "torso_link")
    left_wrist_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_wrist_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    if torso_body_id < 0 or left_wrist_body_id < 0 or right_wrist_body_id < 0:
        raise ValueError("MuJoCo model missing wrist bodies for EE pose probing")

    return MujocoEvalRuntime(
        model=model,
        data=data,
        base_qpos=np.asarray(data.qpos, dtype=np.float64).copy(),
        arm_joint_qpos_indices=arm_qpos_indices,
        dex1_qpos_indices=dex1_qpos_indices,
        torso_body_id=int(torso_body_id),
        left_wrist_body_id=int(left_wrist_body_id),
        right_wrist_body_id=int(right_wrist_body_id),
        arm_ik=PinocchioDualArmIK.create_g1_29(),
        control_frequency=float(control_frequency),
        max_arm_joint_speed=float(max_arm_joint_speed),
        apply_speed_limit=bool(apply_speed_limit),
    )


def reset_runtime_to_gt_state(runtime: MujocoEvalRuntime, item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    import mujoco as mj

    arm_q, gripper_q = _frame_arm_and_ee_qpos(item, "states")
    runtime.data.qpos[:] = runtime.base_qpos
    for index, qpos_index in enumerate(runtime.arm_joint_qpos_indices):
        runtime.data.qpos[qpos_index] = float(arm_q[index])
    if runtime.dex1_qpos_indices is not None:
        left_q = gripper_state_to_mujoco_q(float(gripper_q[0]))
        right_q = gripper_state_to_mujoco_q(float(gripper_q[1]))
        for qpos_index in runtime.dex1_qpos_indices["left"] or []:
            runtime.data.qpos[qpos_index] = left_q
        for qpos_index in runtime.dex1_qpos_indices["right"] or []:
            runtime.data.qpos[qpos_index] = right_q
    if runtime.data.qvel is not None and runtime.data.qvel.size:
        runtime.data.qvel[:] = 0.0
    if runtime.data.ctrl is not None and runtime.data.ctrl.size:
        runtime.data.ctrl[:] = 0.0
    mj.mj_forward(runtime.model, runtime.data)
    reset_arm_ik_state(runtime.arm_ik, arm_q)
    return arm_q.copy(), gripper_q.copy()


def execute_chunk_step_in_mujoco(
    runtime: MujocoEvalRuntime,
    current_arm_q: np.ndarray,
    left_target_pose: Any,
    right_target_pose: Any,
    left_gripper_q: float,
    right_gripper_q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import mujoco as mj

    from teleop.control_utils.arm_target_safety import limit_arm_joint_target_velocity

    current_q = _vector_from_field(np.asarray(current_arm_q, dtype=np.float64).reshape(-1).tolist(), 14, "current_arm_q")
    left_pose = _matrix4x4(left_target_pose, "left_target_pose")
    right_pose = _matrix4x4(right_target_pose, "right_target_pose")
    arm_dq = np.zeros(14, dtype=np.float64)
    solved_q, _ = runtime.arm_ik.solve_ik(left_pose, right_pose, current_q, arm_dq)
    solved_q = _vector_from_field(np.asarray(solved_q, dtype=np.float64).reshape(-1).tolist(), 14, "solved_q")
    if runtime.apply_speed_limit:
        solved_q = limit_arm_joint_target_velocity(
            solved_q,
            current_q,
            max_joint_speed=runtime.max_arm_joint_speed,
            control_frequency=runtime.control_frequency,
        )

    for index, qpos_index in enumerate(runtime.arm_joint_qpos_indices):
        runtime.data.qpos[qpos_index] = float(solved_q[index])
    if runtime.dex1_qpos_indices is not None:
        left_q = gripper_state_to_mujoco_q(float(left_gripper_q))
        right_q = gripper_state_to_mujoco_q(float(right_gripper_q))
        for qpos_index in runtime.dex1_qpos_indices["left"] or []:
            runtime.data.qpos[qpos_index] = left_q
        for qpos_index in runtime.dex1_qpos_indices["right"] or []:
            runtime.data.qpos[qpos_index] = right_q
    if runtime.data.qvel is not None and runtime.data.qvel.size:
        runtime.data.qvel[:] = 0.0
    mj.mj_forward(runtime.model, runtime.data)
    torso_world_pose = mujoco_body_pose(runtime.data, runtime.torso_body_id)
    world_to_torso = invert_rigid_pose(torso_world_pose)
    left_exec_pose = world_to_torso @ wrist_body_pose_to_ee_pose(mujoco_body_pose(runtime.data, runtime.left_wrist_body_id))
    right_exec_pose = world_to_torso @ wrist_body_pose_to_ee_pose(mujoco_body_pose(runtime.data, runtime.right_wrist_body_id))
    return solved_q.copy(), left_exec_pose, right_exec_pose


def run_anchor_evaluation(
    runtime: MujocoEvalRuntime,
    session: OnlineInferenceSession,
    anchor_item: dict[str, Any],
    window: RawEpisodeWindow,
    video_sink: MujocoEvalVideoSink | None = None,
) -> list[dict[str, Any]]:
    current_arm_q, _ = reset_runtime_to_gt_state(runtime, anchor_item)
    action_steps = run_session_for_window(session, window)
    eval_steps = min(
        len(action_steps),
        len(window.gt_left_action_pose_targets),
        len(window.gt_right_action_pose_targets),
    )
    if eval_steps <= 0:
        raise ValueError("no overlapping eval steps between model chunk and GT targets")
    records: list[dict[str, Any]] = []
    gt_left_points = []
    gt_right_points = []
    pred_left_points = []
    pred_right_points = []
    torso_world_pose = mujoco_body_pose(runtime.data, runtime.torso_body_id)
    for step_index, action_step in enumerate(action_steps[:eval_steps]):
        gt_left_pose = _matrix4x4(window.gt_left_action_pose_targets[step_index], "gt_left_pose")
        gt_right_pose = _matrix4x4(window.gt_right_action_pose_targets[step_index], "gt_right_pose")
        pred_left_pose = _matrix4x4(action_step.left_pose, "pred_left_pose")
        pred_right_pose = _matrix4x4(action_step.right_pose, "pred_right_pose")
        current_arm_q, exec_left_pose, exec_right_pose = execute_chunk_step_in_mujoco(
            runtime=runtime,
            current_arm_q=current_arm_q,
            left_target_pose=pred_left_pose,
            right_target_pose=pred_right_pose,
            left_gripper_q=float(action_step.left_gripper_width),
            right_gripper_q=float(action_step.right_gripper_width),
        )
        pred_left_metrics = pose_error_metrics(pred_left_pose, gt_left_pose)
        pred_right_metrics = pose_error_metrics(pred_right_pose, gt_right_pose)
        exec_left_metrics = pose_error_metrics(exec_left_pose, gt_left_pose)
        exec_right_metrics = pose_error_metrics(exec_right_pose, gt_right_pose)
        gt_left_world_pose = torso_local_pose_to_world_pose(torso_world_pose, gt_left_pose)
        gt_right_world_pose = torso_local_pose_to_world_pose(torso_world_pose, gt_right_pose)
        pred_left_world_pose = torso_local_pose_to_world_pose(torso_world_pose, pred_left_pose)
        pred_right_world_pose = torso_local_pose_to_world_pose(torso_world_pose, pred_right_pose)
        gt_left_points.append(np.asarray(gt_left_world_pose[:3, 3], dtype=np.float64).tolist())
        gt_right_points.append(np.asarray(gt_right_world_pose[:3, 3], dtype=np.float64).tolist())
        pred_left_points.append(np.asarray(pred_left_world_pose[:3, 3], dtype=np.float64).tolist())
        pred_right_points.append(np.asarray(pred_right_world_pose[:3, 3], dtype=np.float64).tolist())
        if video_sink is not None:
            video_sink.write_step(
                runtime=runtime,
                overlay={
                    "anchor_frame": int(window.anchor_frame),
                    "step_index": int(step_index),
                    "frame_index": int(window.anchor_frame + step_index),
                    "trajectory_step_index": int(step_index),
                    "pred_xyz_error_m": 0.5 * (pred_left_metrics["xyz_error_m"] + pred_right_metrics["xyz_error_m"]),
                    "exec_xyz_error_m": 0.5 * (exec_left_metrics["xyz_error_m"] + exec_right_metrics["xyz_error_m"]),
                    "pred_rot_error_rad": 0.5 * (
                        pred_left_metrics["rot_error_rad"] + pred_right_metrics["rot_error_rad"]
                    ),
                    "exec_rot_error_rad": 0.5 * (
                        exec_left_metrics["rot_error_rad"] + exec_right_metrics["rot_error_rad"]
                    ),
                    "gt_left_points": gt_left_points,
                    "gt_right_points": gt_right_points,
                    "pred_left_points": pred_left_points,
                    "pred_right_points": pred_right_points,
                },
            )
        records.append(
            {
                "anchor_frame": int(window.anchor_frame),
                "step_index": int(step_index),
                "frame_index": int(window.anchor_frame + step_index),
                "sample_monotonic_ns": int(window.sample_monotonic_ns_history[-1]),
                "pred_left_xyz_error_m": pred_left_metrics["xyz_error_m"],
                "pred_right_xyz_error_m": pred_right_metrics["xyz_error_m"],
                "exec_left_xyz_error_m": exec_left_metrics["xyz_error_m"],
                "exec_right_xyz_error_m": exec_right_metrics["xyz_error_m"],
                "pred_left_rot_error_rad": pred_left_metrics["rot_error_rad"],
                "pred_right_rot_error_rad": pred_right_metrics["rot_error_rad"],
                "exec_left_rot_error_rad": exec_left_metrics["rot_error_rad"],
                "exec_right_rot_error_rad": exec_right_metrics["rot_error_rad"],
                "pred_left_pose": pred_left_pose.tolist(),
                "pred_right_pose": pred_right_pose.tolist(),
                "exec_left_pose": exec_left_pose.tolist(),
                "exec_right_pose": exec_right_pose.tolist(),
                "gt_left_pose": gt_left_pose.tolist(),
                "gt_right_pose": gt_right_pose.tolist(),
                "pred_left_gripper_q": float(action_step.left_gripper_width),
                "pred_right_gripper_q": float(action_step.right_gripper_width),
                "online_observation_seq": int(action_step.metadata.get("online_observation_seq", 0)),
                "online_chunk_seq": int(action_step.metadata.get("online_chunk_seq", 0)),
                "online_chunk_index": int(action_step.metadata.get("online_chunk_index", step_index)),
                "online_chunk_size": int(action_step.metadata.get("online_chunk_size", len(action_steps))),
                "eval_overlap_steps": int(eval_steps),
            }
        )
    return records


def evaluate_raw_episode_pi05_mujoco(
    dataset_root: str | Path,
    episode_index: int,
    base_url: str,
    prompt: str,
    output_dir: str | Path,
    stride: int = 10,
    chunk_size: int = 10,
    n_obs_steps: int = 2,
    start_frame: int | None = None,
    end_frame: int | None = None,
    camera_freq: float = 30.0,
    action_step_sec: float | None = None,
    response_timeout_sec: float = 30.0,
    max_arm_joint_speed: float = 1.5,
    apply_speed_limit: bool = False,
    save_video: str | Path | None = None,
    video_step_repeat: int = 4,
    video_width: int = 1280,
    video_height: int = 720,
    video_camera_distance: float = 3.0,
    video_camera_azimuth: float = -135.0,
    video_camera_elevation: float = -18.0,
    video_camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.95),
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    if action_step_sec is None:
        action_step_sec = 1.0 / camera_freq
    payload = load_raw_episode_payload(dataset_root, episode_index)
    prompt_value = str(prompt or "").strip() or build_prompt_from_payload(payload)
    anchor_frames = build_anchor_indices(
        episode_len=len(payload.items),
        n_obs_steps=n_obs_steps,
        chunk_size=chunk_size,
        stride=stride,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    runtime = build_mujoco_eval_runtime(
        control_frequency=camera_freq,
        max_arm_joint_speed=max_arm_joint_speed,
        apply_speed_limit=apply_speed_limit,
    )
    all_records: list[dict[str, Any]] = []
    video_context = nullcontext(None)
    if save_video is not None:
        video_context = MujocoEvalVideoSink(
            runtime=runtime,
            config=EvalVideoConfig(
                path=Path(save_video),
                fps=float(camera_freq),
                step_repeat=int(video_step_repeat),
                width=int(video_width),
                height=int(video_height),
                camera_distance=float(video_camera_distance),
                camera_azimuth=float(video_camera_azimuth),
                camera_elevation=float(video_camera_elevation),
                camera_lookat=tuple(float(value) for value in video_camera_lookat),
            ),
        )
    with video_context as video_sink:
        for request_index, anchor_frame in enumerate(anchor_frames, start=1):
            window = make_gt_window(payload, anchor_frame=anchor_frame, n_obs_steps=n_obs_steps, chunk_size=chunk_size)
            effective_camera_freq = float(camera_freq)
            if len(window.sample_monotonic_ns_history) >= 2:
                sample_deltas = np.diff(np.asarray(window.sample_monotonic_ns_history, dtype=np.int64))
                mean_delta_ns = float(np.mean(sample_deltas))
                if mean_delta_ns <= 0.0:
                    raise ValueError(f"window has non-positive sample delta at anchor_frame={anchor_frame}")
                effective_camera_freq = 1e9 / mean_delta_ns
            effective_action_step_sec = float(action_step_sec) if action_step_sec is not None else 1.0 / effective_camera_freq
            session = build_http_pi05_session(
                base_url=base_url,
                prompt=prompt_value,
                n_obs_steps=n_obs_steps,
                camera_freq=effective_camera_freq,
                action_step_sec=effective_action_step_sec,
                response_timeout_sec=response_timeout_sec,
            )
            anchor_records = run_anchor_evaluation(
                runtime=runtime,
                session=session,
                anchor_item=payload.items[anchor_frame],
                window=window,
                video_sink=video_sink,
            )
            close_transport = getattr(session.transport, "close", None)
            if callable(close_transport):
                close_transport()
            all_records.extend(anchor_records)
            mean_exec_xyz = float(
                sum(0.5 * (record["exec_left_xyz_error_m"] + record["exec_right_xyz_error_m"]) for record in all_records)
                / len(all_records)
            )
            print(
                "[RAW_PI05_EVAL] "
                f"episode={episode_index:04d} request={request_index}/{len(anchor_frames)} "
                f"anchor={anchor_frame} steps={len(anchor_records)} mean_exec_xyz_error_m={mean_exec_xyz:.6f}",
                flush=True,
            )
        if video_sink is not None:
            video_sink.assert_has_frames()

    summary = summarize_step_records(
        episode_index=episode_index,
        records=all_records,
        num_requests=len(anchor_frames),
        num_anchor_frames=len(anchor_frames),
    )
    summary_path, steps_path = write_eval_outputs(output_dir=output_dir, summary=summary, records=all_records)
    return summary, all_records, summary_path, steps_path
