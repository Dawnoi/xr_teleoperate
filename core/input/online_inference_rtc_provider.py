from __future__ import annotations

from core.input.online_inference_provider import OnlineInferenceInputProvider


class OnlineInferenceRTCInputProvider(OnlineInferenceInputProvider):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("motion_source", "online_inference_rtc")
        super().__init__(*args, **kwargs)
