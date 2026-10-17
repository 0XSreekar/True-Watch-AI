"""The two field-measurement tools, proven on generated footage (no real footage exists yet).

They run the service's own path (ingest -> CameraService -> rules); the footage and the detector are
scripted, so the expected counts are known exactly and the harness itself is what is being tested.
"""

from __future__ import annotations

import json

import pytest

from tools.harness import poisson_interval, render_crossings, render_quiet, run_recording, wilson_interval
from tools.measure_fence_misses import _rules, load_truth, match
from tools.measure_fence_misses import summarise as fence_summary
from tools.measure_false_alerts import parse_recordings
from tools.measure_false_alerts import summarise as alert_summary


def test_wilson_interval_matches_published_values():
    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0 and hi == pytest.approx(0.0713, abs=1e-4)
    lo, hi = wilson_interval(2, 50)
    assert (lo, hi) == pytest.approx((0.0110, 0.1346), abs=1e-4)
    assert wilson_interval(0, 189)[1] < 0.02 < wilson_interval(0, 188)[1]
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_poisson_interval_is_the_exact_garwood_interval():
    assert poisson_interval(0) == pytest.approx((0.0, 3.689), abs=1e-3)
    assert poisson_interval(3) == pytest.approx((0.619, 8.767), abs=1e-3)


def test_matching_is_one_to_one_within_tolerance():
    pairs, missed, extra = match([10.0, 20.0, 30.0], [10.4, 10.6, 31.5, 50.0], tolerance=1.0)
    assert pairs == [(10.0, 10.4)] and missed == [20.0, 30.0] and extra == [10.6, 31.5, 50.0]


def test_truth_file_accepts_plain_times_or_objects(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("[3.5, 1.0]")
    b.write_text(json.dumps({"crossings": [{"t": 2.0, "direction": "in"}]}))
    assert load_truth(a) == [1.0, 3.5] and load_truth(b) == [2.0]


def test_recordings_take_an_optional_camera_id():
    assert parse_recordings(["CAM-A=/x/a.mp4", "/x/night b.mp4"])[0][0] == "CAM-A"
    cam, path = parse_recordings(["/x/night.mp4"])[0]
    assert cam == "CAM-NIGHT" and path.name == "night.mp4"


@pytest.mark.parametrize("fps", [20.0, 8.0])
def test_fence_harness_counts_scripted_crossings_and_the_blind_ones_as_missed(tmp_path, fps):
    script = render_crossings(tmp_path / "x.mp4", 6, blind={2}, lead_s=5.5)
    run = run_recording(script.path, script.appearance(), camera_id="SYN", target_fps=fps, rules=_rules(script.fence))
    s = fence_summary(run, [c["t"] for c in script.crossings], 1.0, fps)
    assert (s["n"], s["detected"], s["missed"]) == (6, 5, 1)
    assert s["missed_at"] == [script.crossings[2]["t"]]
    assert s["unmatched_fence_alerts"] == []
    assert s["timing_error_s"]["max_abs"] <= 2.0 / fps + 1e-6
    assert s["wilson95"] == pytest.approx(wilson_interval(1, 6), abs=1e-4)
    assert s["verdict"].startswith("fails")          # 1 of 6: the Wilson lower bound (3.0%) is above 2%


def test_false_alert_harness_counts_nothing_on_quiet_footage_and_every_scripted_walk(tmp_path):
    quiet = render_quiet(tmp_path / "q.mp4", 12.0)
    control = render_quiet(tmp_path / "c.mp4", 12.0, intrusions=1, seed=5)
    q = run_recording(quiet.path, quiet.appearance(), camera_id="Q", target_fps=8.0)
    c = run_recording(control.path, control.appearance(), camera_id="C", target_fps=8.0)
    assert q.events == [] and q.warmup_s == pytest.approx(5.0, abs=0.3)
    assert [e["kind"] for e in c.events] == ["detection"]
    s = alert_summary([q], 8.0)
    assert s["alerts"] == 0 and s["cameras"] == 1
    assert s["alertable_camera_hours"] == pytest.approx(q.alertable_s / 3600, abs=1e-4)
    lo, hi = s["ci95_per_camera_per_24h"]
    assert lo == 0.0 and hi == pytest.approx(3.689 / (q.alertable_s / 3600) * 24, rel=1e-2)
    assert "camera-hours" in s["extrapolation"]
