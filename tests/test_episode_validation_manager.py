import json
from threading import Event
from pathlib import Path
from types import SimpleNamespace

import pytest

import data_pipeline.audit.episode_validation as episode_validation
from data_pipeline.audit.episode_validation import validate_finalized_episode, write_validation_report
from data_pipeline.audit.episode_validation_manager import EpisodeValidationManager
from data_pipeline.recording.episode_writer import EpisodeWriter
from data_pipeline.recording.teleop_recording_flow import TeleopRecordingFlow
from teleop.ui.episode_store import validate_episode
import teleop.validation.mobile_training as mobile_training
from teleop.validation.mobile_training import validate_mobile_training


def test_validation_manager_serializes_report_and_clears_pending(tmp_path):
    episode_dir = tmp_path / "episode_0001"
    episode_dir.mkdir()
    (episode_dir / "data.json").write_text('{"data": []}\n', encoding="utf-8")

    report = {
        "checked_at_ns": 1,
        "episode_name": episode_dir.name,
        "level": "ok",
        "source_data": {"size_bytes": (episode_dir / "data.json").stat().st_size, "mtime_ns": (episode_dir / "data.json").stat().st_mtime_ns},
        "errors": [],
        "warnings": [],
        "action_semantics": {"status": "ok", "errors": [], "warnings": []},
    }
    started = Event()
    release = Event()

    def validator(_):
        started.set()
        release.wait(timeout=1.0)
        return report

    manager = EpisodeValidationManager(validator=validator)
    manager.submit(episode_dir)

    assert started.wait(timeout=1.0)
    assert manager.is_busy()
    release.set()
    manager.close()

    assert not manager.is_busy()
    assert json.loads((episode_dir / "validation.json").read_text(encoding="utf-8")) == report
    assert not list(episode_dir.glob(".validation.json.*.tmp"))
    assert manager.status()["last_validation"] == report


def test_recording_flow_rejects_start_while_validation_is_pending():
    class Recorder:
        def create_episode(self):
            raise AssertionError("create_episode must not be called during validation")

    class Manager:
        def is_busy(self):
            return True

        def status(self):
            return {"current_episode_dir": "/tmp/task/episode_0001"}

    logs = []
    flow = TeleopRecordingFlow(
        args=SimpleNamespace(frequency=30.0),
        recorder=Recorder(),
        log=SimpleNamespace(warning=lambda *args: logs.append(args), info=lambda *args: None),
        validation_manager=Manager(),
    )

    result = flow.handle_commands(
        record_running=False,
        record_toggle=True,
        record_cancel=False,
        camera_sources={},
    )

    assert not result.record_running
    assert not result.record_toggle
    assert logs


def test_malformed_data_json_becomes_a_durable_validation_error(tmp_path):
    episode_dir = tmp_path / "episode_0001"
    episode_dir.mkdir()
    (episode_dir / "data.json").write_text('{"data": [\n', encoding="utf-8")

    report = validate_finalized_episode(episode_dir)
    output = write_validation_report(episode_dir, report)

    assert report["level"] == "error"
    assert report["action_semantics"]["status"] == "not_applicable"
    assert "parse error" in report["errors"][0]
    assert json.loads(output.read_text(encoding="utf-8"))["level"] == "error"


def test_base_record_without_mobile_tcp_schema_is_an_error():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/data_quality/g1_d_mobile_training.json").read_text(
            encoding="utf-8"
        )
    )
    report = validate_mobile_training(
        [{"states": {"base": {}}, "actions": {}}],
        config,
        episode_info={"enabled_cameras": ["head"]},
    )

    assert report["status"] == "error"
    assert any(issue["code"] == "MOBILE_TCP_METADATA_MISMATCH" for issue in report["errors"])


def test_mobile_training_alignment_uses_recorded_delta_to_sample_ns():
    issues = []

    delta_ms = mobile_training._validate_alignment(
        {"base_state": {"delta_to_sample_ns": -31_670_866}},
        key="base_state",
        max_abs_delta_ms=150.0,
        require_past=False,
        frame_index=0,
        issues=issues,
    )

    assert delta_ms == -31.670866
    assert issues == []


def test_base_action_hold_last_requires_past_source_and_respects_age_limit():
    issues = []
    age_ms = mobile_training._validate_base_action_alignment(
        {
            "sample_monotonic_ns": 100_000_000,
            "base_action": {
                "interpolation_mode": "hold_last",
                "support_source_t_ns": 40_000_000,
                "support_max_abs_delta_ns": 60_000_000,
            },
        },
        max_support_delta_ms=83.333333,
        frame_index=0,
        issues=issues,
    )

    assert age_ms == 60.0
    assert issues == []

    future_issues = []
    mobile_training._validate_base_action_alignment(
        {
            "sample_monotonic_ns": 100_000_000,
            "base_action": {
                "interpolation_mode": "hold_last",
                "support_source_t_ns": 100_000_001,
                "support_max_abs_delta_ns": 1,
            },
        },
        max_support_delta_ms=83.333333,
        frame_index=1,
        issues=future_issues,
    )

    assert [issue["code"] for issue in future_issues] == ["MOBILE_BASE_ACTION_HOLD_LAST_FUTURE"]


def test_mobile_training_derives_alignment_limits_from_nominal_periods():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/data_quality/g1_d_mobile_training.json").read_text(
            encoding="utf-8"
        )
    )

    validated = mobile_training._validate_config(config)

    assert validated["timing"] == {
        "base_state_period_ms": 60.0,
        "base_height_period_ms": 33.333333,
        "slam_tf_period_ms": 60.0,
        "base_action_period_ms": 33.333333,
        "max_alignment_periods": 2.5,
        "slam_tf_max_alignment_periods": 3.0,
        "slam_tf_max_age_periods": 3.0,
        "slam_tf_max_future_periods": 0.2,
    }
    assert validated["limits"]["base_state_max_delta_ms"] == pytest.approx(150.0)
    assert validated["limits"]["base_height_max_delta_ms"] == pytest.approx(83.3333325)
    assert validated["limits"]["slam_tf_max_delta_ms"] == pytest.approx(180.0)
    assert validated["limits"]["base_action_max_support_delta_ms"] == pytest.approx(83.3333325)
    assert validated["limits"]["slam_tf_max_age_ms"] == pytest.approx(180.0)
    assert validated["limits"]["slam_tf_max_future_ms"] == pytest.approx(12.0)


def test_episode_validation_requires_full_camera_coverage(tmp_path):
    episode_dir = tmp_path / "episode_0001"
    image_path = episode_dir / "colors" / "head" / "000000_head.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    payload = {
        "data": [
            {
                "colors": {"head": "colors/head/000000_head.jpg"},
                "states": {
                    "left_arm": {"qpos": [0.0] * 7},
                    "right_arm": {"qpos": [0.0] * 7},
                },
                "timestamps": {"sample_monotonic_ns": 0},
            },
            {
                "colors": {"head": "colors/head/000001_head.jpg"},
                "states": {
                    "left_arm": {"qpos": [0.0] * 7},
                    "right_arm": {"qpos": [0.0] * 7},
                },
                "timestamps": {"sample_monotonic_ns": 1},
            },
        ]
    }

    report = validate_episode(episode_dir, payload)

    assert report["level"] == "error"
    assert "camera head coverage 1/2 below 100%" in report["errors"]


def test_episode_writer_persists_explicit_enabled_camera_manifest(tmp_path):
    writer = EpisodeWriter(tmp_path / "task", rerun_log=False)

    assert writer.create_episode(enabled_cameras=["right_wrist", "head", "left_wrist"])
    writer.close()

    payload = json.loads((tmp_path / "task" / "episode_0000" / "data.json").read_text(encoding="utf-8"))
    assert payload["info"]["enabled_cameras"] == ["head", "left_wrist", "right_wrist"]


def test_recording_flow_passes_explicit_camera_manifest_to_writer():
    calls = []

    class Recorder:
        def create_episode(self, *, enabled_cameras):
            calls.append(enabled_cameras)
            return True

    flow = TeleopRecordingFlow(
        args=SimpleNamespace(frequency=30.0),
        recorder=Recorder(),
        log=SimpleNamespace(warning=lambda *args: None, info=lambda *args: None),
    )

    result = flow.handle_commands(
        record_running=False,
        record_toggle=True,
        record_cancel=False,
        camera_sources={"right_wrist": object(), "head": object(), "left_wrist": None},
    )

    assert result.record_running is False
    assert calls == [["head", "right_wrist"]]


def test_validation_merges_structural_alignment_and_action_reports(tmp_path, monkeypatch):
    episode_dir = tmp_path / "episode_0001"
    episode_dir.mkdir()
    image_path = episode_dir / "colors" / "head" / "000000_head.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    (episode_dir / "data.json").write_text(
        json.dumps(
            {
                "info": {"enabled_cameras": ["head"]},
                "data": [
                    {
                        "idx": 0,
                        "colors": {"head": "colors/head/000000_head.jpg"},
                        "timestamps": {"camera": {"head": {"host_recv_monotonic_ns": 1}}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    alignment_calls = []

    monkeypatch.setattr(
        episode_validation,
        "validate_episode",
        lambda _episode_dir, _payload: {
            "level": "warning",
            "errors": [],
            "warnings": ["structural warning"],
            "checks": {"frame_count": {"ok": True}},
        },
    )

    def fake_alignment(items, enabled_cameras, *, config_path):
        alignment_calls.append((items, enabled_cameras, config_path))
        return {"status": "error", "errors": ["alignment error"], "warnings": []}

    monkeypatch.setattr(episode_validation, "_run_time_alignment", fake_alignment)
    monkeypatch.setattr(
        episode_validation,
        "validate_action_semantics",
        lambda **kwargs: {"status": "ok", "issues": []},
    )

    report = validate_finalized_episode(
        episode_dir,
        time_alignment_config=tmp_path / "time_alignment.json",
        action_semantics_config=tmp_path / "action_semantics.json",
    )

    assert alignment_calls == [(
        [{
            "idx": 0,
            "colors": {"head": "colors/head/000000_head.jpg"},
            "timestamps": {"camera": {"head": {"host_recv_monotonic_ns": 1}}},
        }],
        ["head"],
        tmp_path / "time_alignment.json",
    )]
    assert report["time_alignment"]["status"] == "error"
    assert report["action_semantics"]["status"] == "ok"
    assert report["errors"] == ["alignment error"]
    assert report["warnings"] == ["structural warning"]
    assert report["level"] == "error"
