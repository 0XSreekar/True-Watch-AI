"""Fusion: an alert needs BOTH channels, at the same place, for N consecutive frames.

The four cases the Phase 3 brief requires, on synthetic scenes run through the real motion channel:
  (i)   appearance only (a confident box on a static object)         -> no alert
  (ii)  motion only (a moving object with no box, or a box below the appearance floor) -> no alert
  (iii) both agree                                                   -> alert, with both channel scores
  (iv)  agreement shorter than the temporal gate                     -> no alert
plus the spatial gate, the night weighting and the camera-state stand-down.
"""

from __future__ import annotations

import pytest

from pipeline.fusion import Fusion, FusionConfig, modality_weights
from pipeline.motion import MotionChannel, motion_reliability
from tests.synth import Scene, make_track

CFG = FusionConfig(threshold=0.55, n_consecutive=3, appearance_max_age=1)


def run(scene: Scene, frames: int, tracks_for, appearance_for, fusion: Fusion, moving_until=None, with_object=True,
        status="ok"):
    """Drive `frames` frames; returns (all decisions per frame, all events)."""
    mc = MotionChannel()
    decisions, events = [], []
    for k in range(frames):
        field = mc.update(scene.frame(k, moving_until, with_object), k / 20.0)
        tracks = tracks_for(k)
        out = fusion.evaluate(tracks, field, appearance_of=appearance_for(k), frame_index=k, timestamp=k / 20.0,
                              captured_at=0.0, frame_size=(320, 256), status=status)
        decisions.append(out.decisions)
        events.extend(out.events)
    return decisions, events


def test_appearance_only_does_not_alert():
    scene = Scene()
    fusion = Fusion("TEST", "visible", CFG)
    parked = scene.parked_box()
    decisions, events = run(scene, 12, lambda k: [make_track(1, parked, "car", k, k / 20)],
                            lambda k: {1: (0.95, k)}, fusion)
    assert events == []
    scored = [d for ds in decisions[1:] for d in ds]
    assert scored and all(d.appearance.score == pytest.approx(0.95) for d in scored)
    assert all(not d.m_ok for d in scored), "a static object must not pass the motion floor"


def test_motion_only_does_not_alert():
    scene = Scene()
    # (a) the moving object has no box at all: motion sees it, fusion has no candidate
    mc = MotionChannel()
    fusion = Fusion("TEST", "visible", CFG)
    events, saw_motion = [], False
    for k in range(12):
        field = mc.update(scene.frame(k), k / 20.0)
        if field is not None:
            saw_motion |= any(r.bbox[0] < scene.obj_box(k)[2] and r.bbox[2] > scene.obj_box(k)[0] for r in field.regions)
        events += fusion.evaluate([], field, appearance_of={}, frame_index=k, timestamp=k / 20.0, captured_at=0.0,
                                  frame_size=(320, 256)).events
    assert saw_motion, "the motion channel should have seen the moving object"
    assert events == []

    # (b) a box on the moving object, but the detector's score is below the appearance floor
    fusion = Fusion("TEST", "visible", CFG)
    decisions, events = run(scene, 12, lambda k: [make_track(2, scene.obj_box(k), "person", k, k / 20)],
                            lambda k: {2: (0.05, k)}, fusion)
    assert events == []
    assert any(d.m_ok for ds in decisions for d in ds), "motion vouched, appearance did not"


def test_both_agree_alerts_with_both_scores():
    scene = Scene()
    fusion = Fusion("TEST", "visible", CFG)
    decisions, events = run(scene, 10, lambda k: [make_track(3, scene.obj_box(k), "person", k, k / 20)],
                            lambda k: {3: (0.8, k)}, fusion)
    assert len(events) == 1, "one alert per track"
    ev = events[0]
    assert ev.appearance.channel == "appearance" and ev.appearance.score == pytest.approx(0.8)
    assert ev.motion.channel == "motion" and ev.motion.score >= CFG.m_min
    assert ev.fused_score >= ev.threshold == CFG.threshold
    assert ev.streak == CFG.n_consecutive
    assert ev.spatial_iou >= CFG.iou_min
    assert ev.weights == modality_weights("visible") and not ev.night
    assert ev.best_frame.fused_score >= ev.fused_score - 1e-9
    assert 0 <= ev.bbox_normalised[0] <= 1 and ev.kind == "DETECTION"
    # the rule engine's FusedEventLike shape: .track, .camera_id, .timestamp_s
    assert ev.track is not None and ev.track.track_id == 3 and ev.timestamp_s == ev.timestamp


def test_agreement_below_temporal_gate_does_not_alert():
    scene = Scene()
    fusion = Fusion("TEST", "visible", CFG)
    # the object moves on frames 0..2 (flow exists on frames 1 and 2 only: 2 agreeing frames), then stops
    decisions, events = run(scene, 12, lambda k: [make_track(4, scene.obj_box(k, 2), "person", k, k / 20)],
                            lambda k: {4: (0.8, k)}, fusion, moving_until=2)
    streaks = [ds[0].streak for ds in decisions if ds]
    assert max(streaks) == CFG.n_consecutive - 1
    assert events == []


def test_spatial_gate_rejects_motion_elsewhere():
    scene = Scene()
    fusion = Fusion("TEST", "visible", CFG)
    # a confident box on static background far from the moving object: the flow peak is not in it
    decisions, events = run(scene, 10, lambda k: [make_track(5, (150, 200, 180, 250), "person", k, k / 20)],
                            lambda k: {5: (0.9, k)}, fusion)
    assert events == []
    assert all(not d.agree for ds in decisions for d in ds)


def test_camera_state_stands_fusion_down():
    scene = Scene()
    fusion = Fusion("TEST", "visible", CFG)
    decisions, events = run(scene, 10, lambda k: [make_track(6, scene.obj_box(k), "person", k, k / 20)],
                            lambda k: {6: (0.8, k)}, fusion, status="camera_state")
    assert events == []


def test_night_weights_follow_measured_contrast():
    # MEASUREMENTS.md section 1: visible 5.17 / 2.74 px (1.9x), LWIR 2.73 / 0.81 px (3.4x)
    assert motion_reliability("visible") == pytest.approx(1 - 2.74 / 5.17)
    assert motion_reliability("lwir") == pytest.approx(1 - 0.81 / 2.73)
    wa_d, wm_d = modality_weights("visible")
    wa_n, wm_n = modality_weights("lwir")
    assert wm_n > wm_d and wa_n < wa_d
    assert wa_d + wm_d == pytest.approx(1.0) and wa_n + wm_n == pytest.approx(1.0)


def test_night_weighting_changes_the_verdict():
    """The same weak box on the same strongly moving object: alerts under LWIR weights, not visible ones."""
    scene = Scene()
    cfg = FusionConfig(threshold=0.6, n_consecutive=3, appearance_max_age=1)
    out = {}
    for modality in ("visible", "lwir"):
        fusion = Fusion("TEST", modality, cfg)
        decisions, events = run(scene, 10, lambda k: [make_track(7, scene.obj_box(k), "person", k, k / 20)],
                                lambda k: {7: (0.2, k)}, fusion)
        out[modality] = (decisions, events)
    ms = [d.motion.score for ds in out["lwir"][0] for d in ds]
    assert max(ms) > 0.9
    assert len(out["lwir"][1]) == 1 and out["lwir"][1][0].night
    assert out["visible"][1] == []
