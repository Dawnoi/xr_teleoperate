from __future__ import annotations

from typing import Any

import logging_mp

from teleop.inference.online_session import OnlineInferenceConfig, OnlineInferenceSession
from teleop.inference.pose_transform import load_pose_transformer
from teleop.inference.protocol import HttpJsonInferenceTransport, TcpJsonTransport
from teleop.input.base import (
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
from teleop.input.lerobot_offline import (
    LeRobotOfflineInputProvider,
    episode_parquet_path,
    required_lerobot_columns,
    validate_lerobot_offline_episode,
)
from teleop.input.online_inference_provider import OnlineInferenceInputProvider
from teleop.input.xr_provider import XRTeleopInputProvider


logger_mp = logging_mp.getLogger(__name__)


def _online_inference_http_handshake_payload(protocol_profile: str) -> dict[str, Any]:
    profile = str(protocol_profile or "pika_pose7").strip()
    if profile == "pi05_dual_arm_20d":
        return {
            "action_dim": 20,
            "action_space": "pose20",
            "robot": "nero_dual_arm",
            "transport": "http",
        }
    return {"transport": "http"}


def _load_xr_robotics_wrapper():
    from teleop.input.xr_robotics_wrapper import XRRoboticsWrapper

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
        from teleop.input.raw_offline import RawEpisodeInputProvider

        logger_mp.info(
            "Detected raw episode replay input, episode_dir=%s arm_source=%s.",
            validation.get("episode_dir"),
            arm_source,
        )
        return RawEpisodeInputProvider(
            dataset_root=dataset_root,
            episode_index=episode_index,
            arm_source=arm_source,
            speed_scale=getattr(args, "offline_replay_speed_scale", 1.0),
            motion_repr="pose" if str(arm_source) == "fk_cmd_pose" else "qpos",
        )
    return LeRobotOfflineInputProvider(
        dataset_root=dataset_root,
        episode_index=episode_index,
        arm_source=arm_source,
        speed_scale=getattr(args, "offline_replay_speed_scale", 1.0),
        arm_ik=arm_ik,
    )


def _create_online_inference_transport(args, protocol_profile: str, transport_kind: str):
    if transport_kind in {"http", "http_json"}:
        base_url = getattr(args, "online_inference_base_url", "")
        if not base_url:
            base_url = f"http://{getattr(args, 'online_inference_host', '127.0.0.1')}:{getattr(args, 'online_inference_port', 5555)}"
        return HttpJsonInferenceTransport.connect(
            base_url=base_url,
            handshake_path=getattr(args, "online_inference_http_handshake_path", "/handshake"),
            infer_path=getattr(args, "online_inference_http_infer_path", "/infer"),
            timeout_sec=getattr(args, "online_inference_response_timeout_sec", 2.0),
            handshake_payload=_online_inference_http_handshake_payload(protocol_profile),
        )
    if transport_kind in {"tcp", "tcp_jsonl", "jsonl"}:
        return TcpJsonTransport.connect(
            host=getattr(args, "online_inference_host", "127.0.0.1"),
            port=getattr(args, "online_inference_port", 5555),
            connect_timeout_sec=getattr(args, "online_inference_response_timeout_sec", 2.0),
        )
    raise ValueError(f"unsupported online_inference_transport: {transport_kind}")


def _create_online_inference_provider(args) -> OnlineInferenceInputProvider:
    enable_motion = bool(getattr(args, "online_inference_enable_motion", False))
    dry_run = bool(getattr(args, "online_inference_dry_run", False))
    protocol_profile = str(getattr(args, "online_inference_protocol_profile", "pika_pose7") or "pika_pose7").strip()
    transport_kind = str(getattr(args, "online_inference_transport", "tcp_jsonl") or "tcp_jsonl").strip()
    transform_arm_side = "both" if protocol_profile == "pi05_dual_arm_20d" else getattr(args, "online_inference_arm_side", "both")
    transformer = load_pose_transformer(
        enable_motion=enable_motion and not dry_run,
        transform_config_path=getattr(args, "online_inference_transform_config", None),
        arm_side=transform_arm_side,
    )
    config = OnlineInferenceConfig(
        arm_side=getattr(args, "online_inference_arm_side", "both"),
        protocol_profile=protocol_profile,
        task_prompt=getattr(args, "online_inference_prompt", ""),
        n_obs_steps=getattr(args, "online_inference_n_obs_steps", 2),
        camera_freq=getattr(args, "online_inference_camera_freq", 30.0),
        action_step_sec=getattr(args, "online_inference_action_step_sec", 0.10),
        chunk_step_mode=getattr(args, "online_inference_chunk_step_mode", "timed"),
        interpolation_interval_sec=getattr(args, "online_inference_interp_sec", 0.01),
        post_action_delay_ms=getattr(args, "online_inference_post_action_delay_ms", 75),
        response_timeout_sec=getattr(args, "online_inference_response_timeout_sec", 2.0),
        jpeg_quality=getattr(args, "online_inference_jpeg_quality", 85),
        enable_motion=enable_motion,
        dry_run=dry_run,
    )
    transport = _create_online_inference_transport(args, protocol_profile, transport_kind)
    session = OnlineInferenceSession(
        config=config,
        transport=transport,
        pose_transformer=transformer,
    )
    logger_mp.info(
        "Using online inference input provider: transport=%s protocol_profile=%s host=%s port=%s arm_side=%s enable_motion=%s dry_run=%s.",
        transport_kind,
        config.protocol_profile,
        getattr(args, "online_inference_host", "127.0.0.1"),
        getattr(args, "online_inference_port", 5555),
        config.arm_side,
        config.enable_motion,
        config.dry_run,
    )
    return OnlineInferenceInputProvider(
        session=session,
        arm_side=config.arm_side,
        ee=getattr(args, "ee", None),
        no_gripper=getattr(args, "no_gripper", False),
    )


def create_teleop_input_provider(args, arm_ik=None) -> BaseTeleopInputProvider:
    input_provider = getattr(args, "input_provider", "xr") or "xr"

    if input_provider == "xr":
        return _create_xr_provider(args)
    if input_provider == "lerobot_offline":
        return _create_lerobot_offline_provider(args, arm_ik=arm_ik)
    if input_provider == "online_inference":
        return _create_online_inference_provider(args)

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
    "XRTeleopInputProvider",
    "_build_offline_tele_data",
    "_finite_pose_matrix",
    "_load_xr_robotics_wrapper",
    "_online_inference_http_handshake_payload",
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
