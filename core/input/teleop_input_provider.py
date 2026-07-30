from __future__ import annotations

from typing import Any

import logging_mp

from core.input.base import (
    ACTION_SIZE,
    ARM_SIZE,
    CHUNK_NAME,
    POSE6_SIZE,
    BaseTeleopInputProvider,
    MotionIntent,
    TeleopInputSample,
    _build_offline_tele_data,
    _finite_pose_matrix,
    dex1_q_to_trigger_value,
    dex1_width_m_to_trigger_value,
    finite_vector,
    pose6_to_matrix,
    split_lerobot_action,
)
from core.input.lerobot_offline import (
    LeRobotOfflineInputProvider,
    episode_parquet_path,
    required_lerobot_columns,
    validate_lerobot_offline_episode,
)
from core.input.online_inference_provider import OnlineInferenceInputProvider, create_online_inference_provider
from core.input.xr_provider import XRTeleopInputProvider
from core.input.vive_provider import ViveTrackerInputProvider, vive_config_from_args


logger_mp = logging_mp.getLogger(__name__)


def _load_xr_robotics_wrapper():
    from core.input.xr_robotics_wrapper import XRRoboticsWrapper

    return XRRoboticsWrapper


def _create_xr_provider(args) -> XRTeleopInputProvider:
    logger_mp.info("Using XR-Robotics as the teleop input provider.")
    xr_wrapper_cls = _load_xr_robotics_wrapper()
    xr_wrapper = xr_wrapper_cls(
        use_hand_tracking=getattr(args, "input_mode", "controller") == "hand",
        head_reference_mode=getattr(args, "head_reference_mode", "calibrated"),
        controller_orientation_mode=getattr(args, "controller_orientation_mode", "neutral"),
        controller_mapping_mode=getattr(args, "controller_mapping_mode", "anchored_safe"),
    )
    return XRTeleopInputProvider(xr_wrapper)


def _create_lerobot_offline_provider(args, arm_ik=None) -> BaseTeleopInputProvider:
    dataset_root = getattr(args, "offline_replay_dataset_root", None)
    if not dataset_root:
        raise ValueError("offline_replay_dataset_root is required for lerobot_offline input provider")
    if not hasattr(args, "offline_replay_episode_index"):
        raise ValueError("offline_replay_episode_index is required for lerobot_offline input provider")
    episode_index = getattr(args, "offline_replay_episode_index")
    arm_source = getattr(args, "offline_replay_arm_source", "action")
    validation = validate_lerobot_offline_episode(dataset_root, episode_index, arm_source)
    logger_mp.info(
        "Using offline teleop input provider (%s), dataset_root=%s, episode_index=%d.",
        "lerobot_offline",
        dataset_root,
        episode_index,
    )
    if validation.get("dataset_kind") == "raw_episode":
        from core.input.raw_offline import RawEpisodeInputProvider

        logger_mp.info(
            "Detected raw episode replay input, episode_dir=%s arm_source=%s base_source=%s.",
            validation.get("episode_dir"),
            arm_source,
            getattr(args, "offline_replay_base_source", "none"),
        )
        return RawEpisodeInputProvider(
            dataset_root=dataset_root,
            episode_index=episode_index,
            arm_source=arm_source,
            speed_scale=getattr(args, "offline_replay_speed_scale", 1.0),
            motion_repr="pose" if str(arm_source) == "fk_cmd_pose" else "qpos",
            base_source=getattr(args, "offline_replay_base_source", "none"),
        )
    return LeRobotOfflineInputProvider(
        dataset_root=dataset_root,
        episode_index=episode_index,
        arm_source=arm_source,
        speed_scale=getattr(args, "offline_replay_speed_scale", 1.0),
        arm_ik=arm_ik,
    )


def _create_vive_provider(args) -> ViveTrackerInputProvider:
    logger_mp.info("Using VIVE tracker pose topics as the teleop input provider.")
    config = vive_config_from_args(args)
    return ViveTrackerInputProvider(
        left_topic=getattr(args, "vive_left_tracker_topic", "/vive_pose_l"),
        right_topic=getattr(args, "vive_right_tracker_topic", "/vive_pose_r"),
        timeout_sec=getattr(args, "vive_tracker_timeout_sec", 0.25),
        position_scale=config["position_scale"],
        orientation_mode=getattr(args, "controller_orientation_mode", "relative"),
        rotation_robot_from_vive=config["rotation_robot_from_vive"],
        offset_xyz=config["offset_xyz"],
        left_mount_rotation=config["left_mount_rotation"],
        right_mount_rotation=config["right_mount_rotation"],
        enable_left_topic=getattr(args, "vive_enable_left_topic", "/vive/enable_left"),
        enable_right_topic=getattr(args, "vive_enable_right_topic", "/vive/enable_right"),
    )


def create_teleop_input_provider(args, arm_ik=None) -> BaseTeleopInputProvider:
    input_provider = getattr(args, "input_provider", "xr") or "xr"

    if input_provider == "xr":
        return _create_xr_provider(args)
    if input_provider == "lerobot_offline":
        return _create_lerobot_offline_provider(args, arm_ik=arm_ik)
    if input_provider == "online_inference":
        return create_online_inference_provider(args)
    if input_provider == "vive":
        return _create_vive_provider(args)

    raise ValueError(f"unsupported input_provider: {input_provider}")


__all__ = [
    "ACTION_SIZE",
    "ARM_SIZE",
    "CHUNK_NAME",
    "POSE6_SIZE",
    "BaseTeleopInputProvider",
    "LeRobotOfflineInputProvider",
    "MotionIntent",
    "OnlineInferenceInputProvider",
    "TeleopInputSample",
    "ViveTrackerInputProvider",
    "XRTeleopInputProvider",
    "_build_offline_tele_data",
    "_finite_pose_matrix",
    "_load_xr_robotics_wrapper",
    "create_teleop_input_provider",
    "dex1_q_to_trigger_value",
    "dex1_width_m_to_trigger_value",
    "episode_parquet_path",
    "finite_vector",
    "pose6_to_matrix",
    "required_lerobot_columns",
    "split_lerobot_action",
    "validate_lerobot_offline_episode",
]
