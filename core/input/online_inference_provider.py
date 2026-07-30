from __future__ import annotations

from typing import Any, Mapping
import time

import logging_mp
import numpy as np

from inference.online_session import CameraSample, OnlineInferenceConfig, OnlineInferenceSession, RobotStateSample
from inference.pose_transform import load_pose_transformer, matrix_to_pose9_rot6d
from inference.transport import HttpJsonInferenceTransport, TcpJsonTransport
from core.input.base import (
    BaseCommandIntent,
    BaseTeleopInputProvider,
    MotionIntent,
    TeleopInputSample,
    _finite_gripper_width,
    _finite_pose_matrix,
    _make_identity_pose,
    dex1_q_to_trigger_value,
)
from core.input.xr_input_types import TeleData


logger_mp = logging_mp.getLogger(__name__)


def _base_intent_from_online_metadata(metadata: Mapping[str, Any], frame_index: int) -> BaseCommandIntent | None:
    value = metadata.get("online_base_action")
    status = str(metadata.get("online_inference_status") or "")
    if value is None:
        return BaseCommandIntent(
            source="online_inference:missing_base_action",
            frame_index=frame_index,
            metadata={
                "online_inference_status": status,
                "online_chunk_index": metadata.get("online_chunk_index"),
                "online_chunk_size": metadata.get("online_chunk_size"),
                "online_chunk_seq": metadata.get("online_chunk_seq"),
                "online_base_action_available": False,
            },
        )
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape[0] != 4:
        raise ValueError(f"online_base_action must have length 4, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("online_base_action contains NaN or Inf")
    return BaseCommandIntent(
        vx=arr[0],
        vy=arr[1],
        wz=arr[2],
        z=arr[3],
        source="online_inference",
        frame_index=frame_index,
        metadata={
            "online_inference_status": status,
            "online_chunk_index": metadata.get("online_chunk_index"),
            "online_chunk_size": metadata.get("online_chunk_size"),
            "online_chunk_seq": metadata.get("online_chunk_seq"),
        },
    )


def _http_handshake_payload(protocol_profile: str) -> dict[str, Any]:
    profile = str(protocol_profile or "pika_pose7").strip()
    if profile == "pi05_dual_arm_20d":
        return {
            "action_dim": 20,
            "action_space": "pose20",
            "robot": "nero_dual_arm",
            "transport": "http",
        }
    if profile == "mobile_tcp23":
        return {
            "action_dim": 20,
            "action_space": "mobile_tcp23_pose20_base4",
            "model_action_dim": 23,
            "base_action_dim": 4,
            "robot": "g1d_dex1_mobile",
            "transport": "http",
        }
    if profile == "mobile_joint_base":
        return {
            "action_dim": 16,
            "action_space": "mobile_joint_base_q16_base4",
            "model_action_dim": 19,
            "base_action_dim": 4,
            "robot": "g1d_dex1_mobile",
            "transport": "http",
        }
    return {"transport": "http"}


def _create_online_inference_transport(args, protocol_profile: str, transport_kind: str):
    if transport_kind in {"http", "http_json"}:
        base_url = str(getattr(args, "online_inference_base_url", "") or "").strip()
        if not base_url:
            base_url = f"http://{getattr(args, 'online_inference_host', '127.0.0.1')}:{getattr(args, 'online_inference_port', 5555)}"
        transport = HttpJsonInferenceTransport.connect(
            base_url=base_url,
            handshake_path=getattr(args, "online_inference_http_handshake_path", "/handshake"),
            infer_path=getattr(args, "online_inference_http_infer_path", "/infer"),
            timeout_sec=getattr(args, "online_inference_response_timeout_sec", 2.0),
            handshake_payload=_http_handshake_payload(protocol_profile),
        )
        transport.reset()
        return transport, True
    if transport_kind in {"tcp", "tcp_jsonl", "jsonl"}:
        return (
            TcpJsonTransport.connect(
                host=getattr(args, "online_inference_host", "127.0.0.1"),
                port=getattr(args, "online_inference_port", 5555),
                connect_timeout_sec=getattr(args, "online_inference_response_timeout_sec", 2.0),
            ),
            False,
        )
    raise ValueError(f"unsupported online_inference_transport: {transport_kind}")


def create_online_inference_provider(args) -> "OnlineInferenceInputProvider":
    enable_motion = bool(getattr(args, "online_inference_enable_motion", False))
    dry_run = bool(getattr(args, "online_inference_dry_run", False))
    protocol_profile = str(getattr(args, "online_inference_protocol_profile", "pika_pose7") or "pika_pose7").strip()
    transport_kind = str(getattr(args, "online_inference_transport", "tcp_jsonl") or "tcp_jsonl").strip()
    transform_arm_side = "both" if protocol_profile in {"pi05_dual_arm_20d", "mobile_tcp23"} else getattr(args, "online_inference_arm_side", "both")
    transformer = load_pose_transformer(
        enable_motion=(
            enable_motion
            and not dry_run
            and protocol_profile != "mobile_joint_base"
        ),
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
        chunk_step_mode=getattr(args, "online_inference_chunk_step_mode", "per_tick"),
        interpolation_interval_sec=getattr(args, "online_inference_interp_sec", 0.01),
        post_action_delay_ms=getattr(args, "online_inference_post_action_delay_ms", 75),
        response_timeout_sec=getattr(args, "online_inference_response_timeout_sec", 2.0),
        jpeg_quality=getattr(args, "online_inference_jpeg_quality", 85),
        enable_motion=enable_motion,
        dry_run=dry_run,
    )
    transport, handshake_complete = _create_online_inference_transport(args, protocol_profile, transport_kind)
    session = OnlineInferenceSession(
        config=config,
        transport=transport,
        pose_transformer=transformer,
        transport_reset_done=handshake_complete,
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


class OnlineInferenceInputProvider(BaseTeleopInputProvider):
    def __init__(
        self,
        session: OnlineInferenceSession,
        arm_side: str,
        ee: str | None = None,
        no_gripper: bool = False,
        dex1_max_width_m: float = 0.054,
    ):
        self.session = session
        self.arm_side = str(arm_side)
        if self.arm_side not in {"left", "right", "both"}:
            raise ValueError(f"unsupported online inference arm_side: {arm_side!r}")
        self.ee = ee
        self.no_gripper = bool(no_gripper)
        self.dex1_max_width_m = float(dex1_max_width_m)
        self._frame_index = 0

        if not self.no_gripper and self.ee != "dex1":
            raise ValueError("online inference gripper control only supports ee='dex1'; use --no-gripper to ignore gripper actions")

    def get_sample(self, *args, **kwargs) -> TeleopInputSample | None:
        del args
        left_pose = _finite_pose_matrix(kwargs.get("current_left_robot_wrist_pose"), "current_left_robot_wrist_pose")
        right_pose = _finite_pose_matrix(kwargs.get("current_right_robot_wrist_pose"), "current_right_robot_wrist_pose")
        host_monotonic_ns = kwargs.get("current_state_host_monotonic_ns")
        if host_monotonic_ns is None:
            host_monotonic_ns = time.monotonic_ns()

        state_sample = RobotStateSample(
            host_monotonic_ns=int(host_monotonic_ns),
            left_pose=left_pose,
            right_pose=right_pose,
            left_gripper_width=_finite_gripper_width(
                kwargs.get("current_left_gripper_width", 0.0),
                "current_left_gripper_width",
            ),
            right_gripper_width=_finite_gripper_width(
                kwargs.get("current_right_gripper_width", 0.0),
                "current_right_gripper_width",
            ),
            mobile_state26=self._mobile_state26(kwargs),
            mobile_joint_state22=self._mobile_joint_state22(kwargs),
        )
        camera_samples = self._coerce_camera_samples(kwargs)
        step = self.session.tick(state_sample=state_sample, camera_samples=camera_samples)

        enabled_arms = [str(side) for side in step.enabled_arms if str(side) in {"left", "right"}]
        trigger_values = [10.0, 10.0]
        trigger_pressed = [False, False]
        if not self.no_gripper:
            if "left" in enabled_arms:
                trigger_values[0] = dex1_q_to_trigger_value(step.left_gripper_width)
                trigger_pressed[0] = True
            if "right" in enabled_arms:
                trigger_values[1] = dex1_q_to_trigger_value(step.right_gripper_width)
                trigger_pressed[1] = True

        tele_data = TeleData(
            head_pose=_make_identity_pose(),
            left_wrist_pose=np.asarray(step.left_pose, dtype=float),
            right_wrist_pose=np.asarray(step.right_pose, dtype=float),
            left_ctrl_trigger=trigger_pressed[0],
            left_ctrl_triggerValue=float(trigger_values[0]),
            left_ctrl_squeeze="left" in enabled_arms,
            left_ctrl_squeezeValue=1.0 if "left" in enabled_arms else 0.0,
            left_ctrl_thumbstickValue=np.zeros(2, dtype=float),
            right_ctrl_trigger=trigger_pressed[1],
            right_ctrl_triggerValue=float(trigger_values[1]),
            right_ctrl_squeeze="right" in enabled_arms,
            right_ctrl_squeezeValue=1.0 if "right" in enabled_arms else 0.0,
            right_ctrl_thumbstickValue=np.zeros(2, dtype=float),
        )
        metadata = dict(step.metadata or {})
        metadata.update(
            {
                "enabled_arms": enabled_arms,
                "online_inference_status": step.status,
                "arm_side": self.arm_side,
            }
        )
        if self._protocol_profile() == "mobile_joint_base":
            current_joint_q = self._current_joint_q_from_mobile_state(state_sample.mobile_joint_state22)
            target_joint_q = current_joint_q if step.joint_arm_q is None else np.asarray(step.joint_arm_q, dtype=float)
            motion_intent = MotionIntent(
                kind="joint_position",
                arm_q=target_joint_q,
                gripper_q=np.asarray([step.left_gripper_width, step.right_gripper_width], dtype=float),
                timestamp=time.time(),
                frame_index=self._frame_index,
                source="online_inference:mobile_joint_base",
                metadata=metadata,
            )
        else:
            motion_intent = MotionIntent(
                kind="pose",
                left_wrist_pose=np.asarray(step.left_pose, dtype=float),
                right_wrist_pose=np.asarray(step.right_pose, dtype=float),
                timestamp=time.time(),
                frame_index=self._frame_index,
                source="online_inference",
                metadata=metadata,
            )
        base_intent = _base_intent_from_online_metadata(metadata, self._frame_index)
        self._frame_index += 1
        done = step.status == "failed"
        return TeleopInputSample(tele_data=tele_data, motion_intent=motion_intent, done=done, base_intent=base_intent)

    def _mobile_state26(self, kwargs: Mapping[str, Any]) -> np.ndarray | None:
        protocol_profile = self._protocol_profile()
        if protocol_profile != "mobile_tcp23":
            return None
        transformer = kwargs.get("mobile_tcp_to_wrist_transformer")
        self.session.set_mobile_tcp_to_wrist_transformer(transformer)

        left_tcp = _finite_pose_matrix(
            kwargs.get("current_left_robot_tcp_pose_base_link"),
            "current_left_robot_tcp_pose_base_link",
        )
        right_tcp = _finite_pose_matrix(
            kwargs.get("current_right_robot_tcp_pose_base_link"),
            "current_right_robot_tcp_pose_base_link",
        )
        map_base = np.asarray(kwargs.get("current_map_base_pose"), dtype=float).reshape(-1)
        if map_base.shape != (3,) or not np.all(np.isfinite(map_base)):
            raise ValueError("current_map_base_pose must be a finite [map_x, map_y, map_yaw]")
        base_velocity = np.asarray(kwargs.get("current_base_velocity_base_link"), dtype=float).reshape(-1)
        if base_velocity.shape != (2,) or not np.all(np.isfinite(base_velocity)):
            raise ValueError("current_base_velocity_base_link must be a finite [vx, wz]")
        column_height = float(kwargs.get("current_column_height_m"))
        if not np.isfinite(column_height):
            raise ValueError("current_column_height_m must be finite")

        left_gripper = _finite_gripper_width(
            kwargs.get("current_left_gripper_width", 0.0),
            "current_left_gripper_width",
        )
        right_gripper = _finite_gripper_width(
            kwargs.get("current_right_gripper_width", 0.0),
            "current_right_gripper_width",
        )
        state = np.concatenate(
            [
                np.asarray(matrix_to_pose9_rot6d(left_tcp), dtype=float),
                [left_gripper],
                np.asarray(matrix_to_pose9_rot6d(right_tcp), dtype=float),
                [right_gripper],
                map_base,
                base_velocity,
                [column_height],
            ]
        ).astype(np.float32)
        if state.shape != (26,) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"mobile_tcp23 state construction failed: shape={state.shape}")
        return state

    def _mobile_joint_state22(self, kwargs: Mapping[str, Any]) -> np.ndarray | None:
        if self._protocol_profile() != "mobile_joint_base":
            return None
        arm_q = np.asarray(kwargs.get("current_arm_q"), dtype=float).reshape(-1)
        if arm_q.shape != (14,) or not np.all(np.isfinite(arm_q)):
            raise ValueError("current_arm_q must be a finite 14D [left_q7, right_q7] vector")
        map_base = np.asarray(kwargs.get("current_map_base_pose"), dtype=float).reshape(-1)
        if map_base.shape != (3,) or not np.all(np.isfinite(map_base)):
            raise ValueError("current_map_base_pose must be a finite [map_x, map_y, map_yaw]")
        base_velocity = np.asarray(kwargs.get("current_base_velocity_base_link"), dtype=float).reshape(-1)
        if base_velocity.shape != (2,) or not np.all(np.isfinite(base_velocity)):
            raise ValueError("current_base_velocity_base_link must be a finite [vx, wz]")
        column_height = float(kwargs.get("current_column_height_m"))
        if not np.isfinite(column_height):
            raise ValueError("current_column_height_m must be finite")
        left_gripper = _finite_gripper_width(
            kwargs.get("current_left_gripper_width", 0.0),
            "current_left_gripper_width",
        )
        right_gripper = _finite_gripper_width(
            kwargs.get("current_right_gripper_width", 0.0),
            "current_right_gripper_width",
        )
        state = np.concatenate(
            [
                arm_q[:7],
                [left_gripper],
                arm_q[7:14],
                [right_gripper],
                map_base,
                base_velocity,
                [column_height],
            ]
        ).astype(np.float32)
        if state.shape != (22,) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"mobile_joint_base state construction failed: shape={state.shape}")
        return state

    def _protocol_profile(self) -> str:
        return str(
            getattr(getattr(self.session, "config", None), "protocol_profile", "pika_pose7")
        ).strip()

    @staticmethod
    def _current_joint_q_from_mobile_state(mobile_joint_state22: np.ndarray | None) -> np.ndarray:
        if mobile_joint_state22 is None:
            raise RuntimeError("mobile_joint_base current state is missing")
        state = np.asarray(mobile_joint_state22, dtype=float)
        if state.shape != (22,) or not np.all(np.isfinite(state)):
            raise ValueError("mobile_joint_base current state must be a finite 22D vector")
        return np.concatenate([state[:7], state[8:15]])

    def _coerce_camera_samples(self, kwargs: dict[str, Any]) -> list[CameraSample]:
        explicit_samples = kwargs.get("camera_samples")
        if explicit_samples is not None:
            samples = list(explicit_samples)
            self._set_required_camera_names([getattr(sample, "name", "") for sample in samples])
            return samples

        camera_sources = kwargs.get("camera_sources") or {}
        target_monotonic_ns = kwargs.get("current_state_host_monotonic_ns")
        samples: list[CameraSample] = []
        required_names: list[str] = []
        iterable = camera_sources.items() if isinstance(camera_sources, Mapping) else camera_sources
        for entry in iterable:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("camera_sources entries must be (name, source) pairs")
            name, source = entry
            if source is None:
                continue
            name = str(name)
            required_names.append(name)
            frame = None
            meta = None
            get_nearest = getattr(source, "get_nearest", None)
            if callable(get_nearest) and target_monotonic_ns is not None:
                frame, meta = get_nearest(int(target_monotonic_ns), max_delta_ns=self.session.max_camera_delta_ns, copy=True)
            if frame is None or meta is None:
                get_latest = getattr(source, "get_latest", None)
                if not callable(get_latest):
                    raise TypeError(f"camera source {name!r} must provide get_latest(copy=True)")
                frame, meta = get_latest(copy=True)
            if frame is None or meta is None:
                continue
            host_ns = meta.get("host_recv_monotonic_ns", meta.get("host_monotonic_ns"))
            if host_ns is None:
                raise KeyError(f"camera source {name!r} metadata missing host monotonic timestamp")
            if target_monotonic_ns is not None and abs(int(host_ns) - int(target_monotonic_ns)) > self.session.max_camera_delta_ns:
                continue
            samples.append(CameraSample(name=name, frame=frame, host_monotonic_ns=int(host_ns)))
        self._set_required_camera_names(required_names)
        return samples

    def _set_required_camera_names(self, names: list[str]) -> None:
        unique_names = list(dict.fromkeys(str(name) for name in names if str(name)))
        if not unique_names:
            return
        set_required = getattr(self.session, "set_required_camera_names", None)
        if callable(set_required):
            set_required(unique_names)

    def report_control_feedback(self, feedback: dict) -> None:
        report = getattr(self.session, "report_control_feedback", None)
        if callable(report):
            report(feedback)

    def get_debug_snapshot(self) -> dict:
        snapshot = getattr(self.session, "get_debug_snapshot", None)
        if callable(snapshot):
            return snapshot()
        return {}

    def close(self) -> None:
        transport = getattr(self.session, "transport", None)
        close_transport = getattr(transport, "close", None)
        if callable(close_transport):
            close_transport()

    def has_live_pose_data(self) -> bool:
        return True

    @property
    def done(self) -> bool:
        return False


__all__ = ["OnlineInferenceInputProvider", "create_online_inference_provider"]
