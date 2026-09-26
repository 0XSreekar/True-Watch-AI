"""Rules run on every frame against fusion-agreed tracks, not only on the one-off fusion snapshot."""

from __future__ import annotations

import json

import pytest

from pipeline.events import EventSink
from pipeline.rules_adapter import RuleTrack, rule_tracks, to_rule_track
from pipeline.service import CameraService
from rules import hours as hours_mod
from rules.config import CameraRuleConfig, ConfigStore
from rules.tracks import ground_point
from tests.synth import Scene, make_track
from tests.test_service import FakeAppearance, check_event, ingest_frames

FENCE_X = 160.0


def fence_config(camera_id: str = "CAM-T", **extra) -> dict:
    return {"camera_id": camera_id,
            "fences": [{"name": "test-line", "points": [[FENCE_X, -1000.0], [FENCE_X, 1000.0]], "direction": "both"}],
            "loitering": {"dwell_s": 30.0, "net_displacement_px": 5.0}, **extra}


def run_service(tmp_path, rules, n=130):
    scene = Scene(speed=1.5, start=(20.0, 110.0))
    sink = EventSink()
    svc = CameraService(FakeAppearance(scene), camera_id="CAM-T", post_id="RXL", sink=sink, stride=1,
                        calibration_dir=tmp_path / "cal", warmup_frames=30, rules=rules)
    per_frame = []
    for f in ingest_frames(scene, n):
        _, events = svc.process(f)
        per_frame.append(events)
    svc.close()
    return svc, sink, per_frame


def test_fence_crossed_after_the_fusion_event_is_still_caught(tmp_path):
    svc, sink, per_frame = run_service(tmp_path, CameraRuleConfig.from_dict(fence_config()))
    fused_at = next(i for i, evs in enumerate(per_frame) if any(e["rule"] is None for e in evs))
    fence_at = next(i for i, evs in enumerate(per_frame)
                    if any(e["rule"] and "fence crossed" in e["rule"]["fired"] for e in evs))
    # the object's ground point reaches the fence well after fusion first agreed on it
    assert fence_at > fused_at + 10
    assert svc.stats.rule_evaluations > svc.stats.fused_events, "rules must run every frame, not per fusion event"
    assert svc.stats.rule_fires == {"fence crossed": 1}
    ev = next(e for e in per_frame[fence_at] if e["rule"])
    check_event(ev)
    assert ev["rule"]["fired"] == ["fence crossed"]
    assert set(ev["rule"]["metrics"]) == {"dwell_s", "net_px", "path_px"}
    assert ev["detection"]["bbox"][0] * 320 + ev["detection"]["bbox"][2] * 320 / 2 >= FENCE_X - 2


def test_rules_come_from_the_hot_reloaded_store_and_missing_file_turns_them_off(tmp_path):
    store_dir = tmp_path / "rules"
    store_dir.mkdir()
    svc, sink, _ = run_service(tmp_path, ConfigStore(directory=store_dir), n=60)
    assert svc.stats.rule_evaluations == 0 and sink.total >= 1, "no rule file: detections still flow, rules off"
    (store_dir / "CAM-T.json").write_text(json.dumps(fence_config()))
    svc, sink, _ = run_service(tmp_path, ConfigStore(directory=store_dir))
    assert svc.stats.rule_fires.get("fence crossed") == 1


def test_adapter_shifts_monotonic_history_to_epoch_for_the_hours_rule():
    t = make_track(4, (10, 10, 30, 60), t=6.9)
    epoch = 1_760_000_000.0          # 2025-10-09 08:53 UTC
    rt = to_rule_track(t, epoch - 6.9)
    assert rt.history[0][0] == pytest.approx(epoch)
    assert rt.bbox == t.bbox and rt.history[0][1:] == t.history[0][1:]
    # read raw, the monotonic 6.9 s is 1970-01-01; shifted, it is the frame's real wall-clock time
    assert hours_mod.local_time_at(t.history[0][0]).year == 1970
    assert hours_mod.local_time_at(rt.history[0][0]).year >= 2025


def test_tentative_tracks_never_reach_the_rules():
    confirmed = make_track(1, (0, 0, 10, 10))
    tentative = make_track(2, (0, 0, 10, 10)).__class__(**{**make_track(2, (0, 0, 10, 10)).__dict__, "state": "tentative"})
    assert [r.track_id for r in rule_tracks([confirmed, tentative], 0.0)] == [1]
    assert isinstance(rule_tracks([confirmed], 0.0)[0], RuleTrack)


def test_ground_point_reads_corner_boxes():
    assert ground_point((10.0, 20.0, 30.0, 80.0)) == (20.0, 80.0)


# ---------------------------------------------------------------- homography frame size

SQUARE_1080 = {"image_points": [[420.0, 900.0], [860.0, 900.0], [900.0, 620.0], [460.0, 620.0]],
               "world_points": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]]}


def test_homography_without_frame_size_is_rejected():
    from rules.config import RuleConfigError

    with pytest.raises(RuleConfigError, match="frame_size"):
        CameraRuleConfig.from_dict({"camera_id": "C", "homography": dict(SQUARE_1080)})


def test_homography_rescales_to_a_same_aspect_stream():
    cfg = CameraRuleConfig.from_dict({"camera_id": "C", "homography": {**SQUARE_1080, "frame_size": [1920, 1080]}})
    live = cfg.for_frame(960, 540).homography
    assert live.frame_size == (960, 540)
    # the same ground point, clicked at half resolution, maps to the same metres
    for (u, v), (x, y) in zip(SQUARE_1080["image_points"], SQUARE_1080["world_points"]):
        wx, wy = live.to_world_m(u / 2, v / 2)
        assert wx == pytest.approx(x, abs=1e-6) and wy == pytest.approx(y, abs=1e-6)
    assert cfg.for_frame(1920, 1080).homography is cfg.homography


def test_homography_refuses_a_different_aspect_stream():
    from rules.config import RuleConfigError

    cfg = CameraRuleConfig.from_dict({"camera_id": "C", "homography": {**SQUARE_1080, "frame_size": [1920, 1080]}})
    with pytest.raises(RuleConfigError, match="1920x1080.*640x512"):
        cfg.for_frame(640, 512)


def test_a_1080p_calibration_on_a_640x512_stream_falls_back_to_pixels_and_no_false_loitering(tmp_path):
    """The walking object below moves ~120 px in 5 s. Read through a 1080p homography on a 640x512 frame
    that is well under 2 m, so it used to be flagged as loitering. Refused, the pixel threshold applies."""
    raw = {"camera_id": "CAM-T", "loitering": {"dwell_s": 3.0, "net_displacement_m": 2.0, "net_displacement_px": 90.0},
           "homography": {**SQUARE_1080, "frame_size": [1920, 1080]}}
    svc, sink, _ = run_service(tmp_path, CameraRuleConfig.from_dict(raw), n=130)
    assert svc.stats.homography_error and "re-pick" in svc.stats.homography_error
    assert "loitering" not in svc.stats.rule_fires
    # control: the unscaled 1080p homography applied blindly to the same frames does report loitering
    from dataclasses import replace

    from rules.homography import Homography

    blind = CameraRuleConfig.from_dict(raw)
    blind_h = Homography(camera_id="CAM-T", matrix=blind.homography.matrix, inverse=blind.homography.inverse,
                         residual=blind.homography.residual, frame_size=(320, 256))  # mislabelled as live-size
    svc2, _, _ = run_service(tmp_path, replace(blind, homography=blind_h), n=130)
    assert "loitering" in svc2.stats.rule_fires
