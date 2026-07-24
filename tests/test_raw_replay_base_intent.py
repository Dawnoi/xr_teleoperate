import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.input.base import BaseCommandIntent, MotionIntent, TeleopInputSample, _build_offline_tele_data
from core.input.raw_offline import RawEpisodeInputProvider


def _write_raw_episode(tmp_path: Path, *, include_base: bool = True) -> Path:
    episode_dir = tmp_path / "episode_0001"
    episode_dir.mkdir(parents=True)
    image = np.full((8, 10, 3), 80, dtype=np.uint8)
    for camera_name in ("head", "wrist_left", "wrist_right"):
        (episode_dir / "colors" / camera_name).mkdir(parents=True)
        image_path = episode_dir / "colors" / camera_name / f"000000_{camera_name}.jpg"
        ok = cv2.imwrite(str(image_path), image)
        if not ok:
            raise RuntimeError(f"failed to write test image: {image_path}")

    actions = {
        "left_arm": {"qpos": [0.1] * 7},
        "right_arm": {"qpos": [0.2] * 7},
        "left_ee": {"qpos": [0.3]},
        "right_ee": {"qpos": [0.4]},
    }
    if include_base:
        actions["base"] = {
            "vx_cmd": 0.11,
            "vy_cmd": -0.22,
            "wz_cmd": 0.33,
            "z_cmd": -0.44,
            "source": "g1d_agv_bridge",
        }
    frame = {
        "idx": 0,
        "colors": {
            "head": "colors/head/000000_head.jpg",
            "left_wrist": "colors/wrist_left/000000_wrist_left.jpg",
            "right_wrist": "colors/wrist_right/000000_wrist_right.jpg",
        },
        "states": {
            "left_arm": {"qpos": [1.1] * 7},
            "right_arm": {"qpos": [1.2] * 7},
            "left_ee": {"qpos": [1.3]},
            "right_ee": {"qpos": [1.4]},
        },
        "actions": actions,
        "timestamps": {"sample_monotonic_ns": 1_000_000_000},
    }
    (episode_dir / "data.json").write_text(json.dumps({"data": [frame]}), encoding="utf-8")
    return episode_dir


def test_raw_replay_action_base_source_emits_base_intent(tmp_path):
    _write_raw_episode(tmp_path, include_base=True)

    provider = RawEpisodeInputProvider(tmp_path, 1, arm_source="action", base_source="action")
    sample = provider.get_sample()

    assert isinstance(sample.base_intent, BaseCommandIntent)
    assert sample.base_intent.vx == 0.11
    assert sample.base_intent.vy == -0.22
    assert sample.base_intent.wz == 0.33
    assert sample.base_intent.z == -0.44
    assert sample.base_intent.source == "raw_episode:actions.base"
    assert sample.base_intent.frame_index == 0


def test_raw_replay_default_base_source_keeps_episodes_without_base_compatible(tmp_path):
    _write_raw_episode(tmp_path, include_base=False)

    provider = RawEpisodeInputProvider(tmp_path, 1, arm_source="action")
    sample = provider.get_sample()

    assert sample.base_intent is None


def test_raw_replay_action_base_source_rejects_missing_actions_base(tmp_path):
    _write_raw_episode(tmp_path, include_base=False)

    with pytest.raises(KeyError, match="frame 0 missing actions.base"):
        RawEpisodeInputProvider(tmp_path, 1, arm_source="action", base_source="action")


def test_teleop_input_sample_preserves_legacy_done_positional_argument():
    motion_intent = MotionIntent(kind="joint_position", arm_q=np.zeros(14))

    sample = TeleopInputSample(
        _build_offline_tele_data(np.eye(4), np.eye(4), np.zeros(2)),
        motion_intent,
        True,
    )

    assert sample.done is True
    assert sample.base_intent is None
