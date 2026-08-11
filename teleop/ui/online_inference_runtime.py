"""UI-facing online inference profiles and provider construction."""

from __future__ import annotations

from copy import copy
from typing import Any

from core.input.online_inference_provider import create_online_inference_provider


class OnlineInferenceUiRuntime:
    """Expose selectable UI profiles and build their online input providers."""

    def __init__(self, *, args: Any, mobile_inputs: Any) -> None:
        if mobile_inputs is None:
            raise ValueError("online inference UI runtime requires mobile inference inputs")
        self._args = args
        self._mobile_inputs = mobile_inputs

    def profiles(self) -> list[dict[str, object]]:
        mobile_error = self._mobile_inputs.tcp23_runtime_error()
        mobile_pelvis_error = self._mobile_inputs.pelvis_planar22_runtime_error()
        mobile_joint_error = self._mobile_inputs.joint_base_runtime_error()
        return [
            {"id": "pi05_dual_arm_20d", "label": "pi0.5 双臂 20D", "available": True, "reason": ""},
            {
                "id": "mobile_tcp23",
                "label": "移动操作 BaseLink TCP 26D/23D",
                "available": not bool(mobile_error),
                "reason": mobile_error,
            },
            {
                "id": "mobile_pelvis_planar22",
                "label": "移动操作 Pelvis22",
                "available": not bool(mobile_pelvis_error),
                "reason": mobile_pelvis_error,
            },
            {
                "id": "mobile_joint_base",
                "label": "移动操作 Joint19",
                "available": not bool(mobile_joint_error),
                "reason": mobile_joint_error,
            },
        ]

    def profile_error(self, protocol_profile: str) -> str:
        selected = str(protocol_profile or "").strip()
        for profile in self.profiles():
            if selected == profile["id"]:
                return "" if bool(profile["available"]) else str(profile["reason"])
        return f"unknown UI online inference protocol_profile: {selected!r}"

    def create_provider(self, *, prompt: str, protocol_profile: str):
        profile_error = self.profile_error(protocol_profile)
        if profile_error:
            raise RuntimeError(profile_error)
        inference_args = copy(self._args)
        inference_args.input_provider = "online_inference"
        inference_args.online_inference_transport = "http"
        inference_args.online_inference_base_url = (
            str(getattr(self._args, "online_inference_base_url", "") or "").strip()
            or "http://127.0.0.1:18027"
        )
        inference_args.online_inference_protocol_profile = str(protocol_profile)
        inference_args.online_inference_arm_side = "both"
        inference_args.online_inference_prompt = str(prompt)
        inference_args.online_inference_enable_motion = bool(
            getattr(self._args, "online_inference_enable_motion", False)
        )
        inference_args.online_inference_dry_run = bool(
            getattr(self._args, "online_inference_dry_run", False)
        )
        inference_args.online_inference_transform_config = (
            str(getattr(self._args, "online_inference_transform_config", "") or "").strip()
            or "configs/inference/unitree_dual_arm_identity_transform.json"
        )
        return create_online_inference_provider(inference_args)
