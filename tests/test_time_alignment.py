import json
from pathlib import Path

import pytest

from teleop.validation.time_alignment import validate_time_alignment


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "data_quality" / "g1_29_time_alignment.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _item(
    sample_ns,
    *,
    state_ns=None,
    action_ns=None,
    camera_ns=None,
    camera_path="colors/head/frame.jpg",
):
    if state_ns is None:
        state_ns = sample_ns
    if action_ns is None:
        action_ns = sample_ns
    if camera_ns is None:
        camera_ns = sample_ns
    return {
        "colors": {"head": camera_path},
        "timestamps": {
            "sample_monotonic_ns": sample_ns,
            "state": {"host_monotonic_ns": state_ns},
            "action": {
                "host_monotonic_ns": action_ns,
                "interpolation_mode": "exact",
                "support_source_count": 1,
                "support_span_ns": 0,
                "support_max_abs_delta_ns": 0,
            },
            "camera": {
                "head": {
                    "host_recv_monotonic_ns": camera_ns,
                }
            },
        },
    }


def _items_with_intervals(intervals, **kwargs):
    timestamp_ns = 0
    items = [_item(timestamp_ns, **kwargs)]
    for interval_ns in intervals:
        timestamp_ns += interval_ns
        items.append(_item(timestamp_ns, **kwargs))
    return items


def _issue_codes(report):
    return [issue["code"] for issue in report["errors"] + report["warnings"]]


def test_aligned_items_report_interval_and_each_source_statistics_with_config_snapshot():
    items = [
        _item(1_000_000_000 + index * 100_000_000, state_ns=1_000_000_000 + index * 100_000_000 + 2_000_000, action_ns=1_000_000_000 + index * 100_000_000 - 3_000_000, camera_ns=1_000_000_000 + index * 100_000_000 + 5_000_000)
        for index in range(4)
    ]

    report = validate_time_alignment(items, ["head"], CONFIG)

    assert report["status"] == "ok"
    assert report["timestamp_source"] == "host_monotonic_only"
    assert report["config"] == CONFIG
    interval = report["sample_interval"]
    assert interval["T_ns"] == pytest.approx(100_000_000)
    assert interval["average_fps"] == pytest.approx(10.0)
    assert interval["interval_median_ns"] == pytest.approx(100_000_000)
    assert interval["interval_p95_ns"] == pytest.approx(100_000_000)
    assert interval["interval_max_ns"] == 100_000_000
    assert interval["long_gap_count"] == 0
    assert json.dumps(report, allow_nan=False)
    for source_name in ("state", "action", "camera.head"):
        source = report["sources"][source_name]
        assert source["sample_count"] == 4
        assert source["mean_error_ns"] is not None
        assert source["p95_error_ns"] is not None
        assert source["p99_error_ns"] is not None
        assert source["max_error_ns"] is not None
        assert source["above_warning_threshold_count"] == 0
        assert source["above_error_threshold_count"] == 0
        assert source["offending_frame_indices"] == []


def test_action_support_reports_nearest_fallback_and_interpolation_distance():
    items = [_item(index * 100_000_000) for index in range(4)]
    items[1]["timestamps"]["action"].update(
        {
            "interpolation_mode": "nearest_fallback",
            "support_source_count": 1,
            "support_span_ns": 0,
            "support_max_abs_delta_ns": 4_000_000,
        }
    )
    items[2]["timestamps"]["action"].update(
        {
            "interpolation_mode": "linear",
            "support_source_count": 2,
            "support_span_ns": 12_000_000,
            "support_max_abs_delta_ns": 6_000_000,
        }
    )

    report = validate_time_alignment(items, ["head"], CONFIG)

    support = report["action_support"]
    assert support["status"] == "warning"
    assert support["nearest_fallback_count"] == 1
    assert support["nearest_fallback_frame_indices"] == [1]
    assert support["support_nearest_abs_delta_p95_ns"] is not None
    assert support["support_nearest_abs_delta_max_ns"] is not None
    assert support["support_max_abs_delta_p95_ns"] is not None
    assert support["support_span_p95_ns"] is not None


def test_sparse_action_support_error_threshold_crossings_remain_warning_when_p99_is_below_t():
    items = [_item(index * 100_000_000) for index in range(541)]
    for frame_index in (47, 53, 54, 287, 303, 318):
        items[frame_index]["timestamps"]["action"].update(
            {
                "interpolation_mode": "linear",
                "support_source_count": 2,
                "support_span_ns": 202_000_000,
                "support_max_abs_delta_ns": 101_000_000,
            }
        )

    report = validate_time_alignment(items, ["head"], CONFIG)

    support = report["action_support"]
    assert support["status"] == "warning"
    assert support["error_frame_indices"] == [47, 53, 54, 287, 303, 318]


def test_one_long_gap_warns_but_three_consecutive_long_gaps_error():
    one_gap = validate_time_alignment(
        _items_with_intervals([100_000_000, 100_000_000, 100_000_000, 1_000_000_000]),
        ["head"],
        config=CONFIG,
    )
    assert one_gap["sample_interval"]["status"] == "warning"
    assert one_gap["sample_interval"]["long_gap_count"] == 1
    assert one_gap["sample_interval"]["offending_frame_indices"] == [4]
    assert one_gap["sample_interval"]["longest_consecutive_long_gap_run"] == 1

    persistent = validate_time_alignment(
        _items_with_intervals([100_000_000] * 6 + [1_000_000_000] * 3 + [100_000_000]),
        ["head"],
        config=CONFIG,
    )
    assert persistent["sample_interval"]["status"] == "error"
    assert persistent["sample_interval"]["long_gap_count"] == 3
    assert persistent["sample_interval"]["longest_consecutive_long_gap_run"] == 3
    assert persistent["sample_interval"]["offending_frame_indices"] == [7, 8, 9]


def test_one_alignment_spike_warns_but_persistent_source_mismatch_errors():
    one_spike_items = [
        _item(index * 100_000_000, state_ns=state_ns)
        for index, state_ns in enumerate([0, 100_000_000, 200_000_000, 300_000_000, 400_000_000, 620_000_000, 630_000_000, 700_000_000, 800_000_000, 900_000_000])
    ]
    one_spike = validate_time_alignment(one_spike_items, ["head"], CONFIG)
    state = one_spike["sources"]["state"]
    assert one_spike["status"] == "warning"
    assert state["above_error_threshold_count"] == 1
    assert state["offending_frame_indices"] == [5]
    assert state["longest_consecutive_error_run"] == 1

    persistent_items = [
        _item(index * 100_000_000, state_ns=state_ns)
        for index, state_ns in enumerate([0, 100_000_000, 200_000_000, 300_000_000, 400_000_000, 620_000_000, 720_000_000, 820_000_000, 830_000_000, 900_000_000])
    ]
    persistent = validate_time_alignment(persistent_items, ["head"], CONFIG)
    assert persistent["status"] == "error"
    assert persistent["sources"]["state"]["above_error_threshold_count"] == 3
    assert persistent["sources"]["state"]["longest_consecutive_error_run"] == 3


@pytest.mark.parametrize(
    ("mutator", "code", "frame_index"),
    [
        (lambda item: item["timestamps"].pop("state"), "missing_state_timestamp", 1),
        (lambda item: item["timestamps"].pop("action"), "missing_action_timestamp", 1),
        (lambda item: item["colors"].pop("head"), "missing_camera_path", 1),
        (lambda item: item["timestamps"]["camera"]["head"].pop("host_recv_monotonic_ns"), "missing_camera_timestamp", 1),
    ],
)
def test_required_source_values_are_errors_at_their_frame(mutator, code, frame_index):
    items = [_item(0), _item(100_000_000)]
    mutator(items[frame_index])

    report = validate_time_alignment(items, ["head"], CONFIG)

    assert report["status"] == "error"
    assert code in _issue_codes(report)
    assert any(issue["code"] == code and issue["frame_index"] == frame_index for issue in report["errors"])


def test_camera_host_monotonic_fallback_is_rejected():
    item = _item(0)
    item["timestamps"]["camera"]["head"] = {"host_monotonic_ns": 0}

    report = validate_time_alignment([item], ["head"], CONFIG)

    assert report["status"] == "error"
    assert any(issue["code"] == "missing_camera_timestamp" for issue in report["errors"])


def test_reusing_the_same_camera_image_is_reported_as_quality_warning_not_timestamp_error():
    items = [_item(0), _item(100_000_000)]
    items[1]["timestamps"]["camera"]["head"]["host_recv_monotonic_ns"] = 0
    items[1]["colors"]["head"] = "colors/head/copied_output.jpg"
    items[0]["timestamps"]["camera"]["head"]["frame_seq"] = 17
    items[1]["timestamps"]["camera"]["head"]["frame_seq"] = 17

    report = validate_time_alignment(items, ["head"], CONFIG)

    camera = report["sources"]["camera.head"]
    assert report["status"] == "warning"
    assert camera["status"] == "warning"
    assert camera["camera_frame_reuse_count"] == 1
    assert camera["camera_frame_reuse_indices"] == [1]
    assert not any(issue["code"] == "repeated_camera_timestamp" for issue in report["errors"])
    assert any(issue["code"] == "camera_frame_reuse" for issue in report["warnings"])


def test_equal_camera_timestamp_for_different_images_is_an_error():
    items = [_item(0), _item(100_000_000, camera_path="colors/head/another.jpg")]
    items[1]["timestamps"]["camera"]["head"]["host_recv_monotonic_ns"] = 0

    report = validate_time_alignment(items, ["head"], CONFIG)

    assert report["status"] == "error"
    assert any(issue["code"] == "repeated_camera_timestamp" for issue in report["errors"])


@pytest.mark.parametrize(
    ("timestamps", "code", "frame_index"),
    [
        ([0, 100_000_000, 100_000_000], "repeated_sample_timestamp", 2),
        ([0, 100_000_000, 50_000_000], "backward_sample_timestamp", 2),
    ],
)
def test_repeated_or_backward_sample_timestamps_are_errors(timestamps, code, frame_index):
    report = validate_time_alignment(
        [_item(timestamp_ns) for timestamp_ns in timestamps],
        ["head"],
        config=CONFIG,
    )

    assert report["status"] == "error"
    assert report["sample_interval"]["status"] == "error"
    assert any(issue["code"] == code and issue["frame_index"] == frame_index for issue in report["errors"])


def test_repeated_or_backward_source_timestamps_are_errors():
    items = [
        _item(index * 100_000_000, state_ns=state_ns)
        for index, state_ns in enumerate([0, 100_000_000, 50_000_000])
    ]

    report = validate_time_alignment(items, ["head"], CONFIG)

    assert report["status"] == "error"
    assert any(issue["code"] == "backward_state_timestamp" and issue["frame_index"] == 2 for issue in report["errors"])


def test_every_manifest_camera_is_reported_independently_and_missing_values_are_not_inferred():
    items = [_item(0), _item(100_000_000)]
    for item in items:
        item["colors"]["left_wrist"] = "colors/left_wrist/frame.jpg"
        item["timestamps"]["camera"]["left_wrist"] = {"host_recv_monotonic_ns": item["timestamps"]["sample_monotonic_ns"]}
    items[1]["colors"].pop("left_wrist")
    items[1]["timestamps"]["camera"].pop("left_wrist")

    report = validate_time_alignment(items, ["head", "left_wrist"], CONFIG)

    assert "camera.head" in report["sources"]
    assert "camera.left_wrist" in report["sources"]
    assert report["sources"]["camera.left_wrist"]["status"] == "error"
    assert any(issue["code"] == "missing_camera_path" and issue["frame_index"] == 1 for issue in report["errors"])
    assert any(issue["code"] == "missing_camera_timestamp" and issue["frame_index"] == 1 for issue in report["errors"])


def test_negative_alignment_warning_scale_is_rejected_without_a_fallback(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["alignment_p95_warning_scale"] = -0.1
    with pytest.raises(ValueError, match="alignment_p95_warning_scale"):
        validate_time_alignment([], [], config)
