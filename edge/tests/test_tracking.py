"""ByteTrack id stability, and the MEASUREMENTS.md section 2 loitering signals."""

from __future__ import annotations

import math

import pytest

from pipeline.tracking import ByteTracker, TrackerConfig, dwell_seconds, net_displacement, path_length
from pipeline.types import Detection

FPS = 20.0


def det(box, score=0.8, cls="person", k=0):
    return Detection(bbox=tuple(float(v) for v in box), class_name=cls, score=score, camera_id="T", frame_index=k,
                     timestamp=k / FPS, modality="visible")


def box_at(x, y, w=20, h=50):
    return (x, y, x + w, y + h)


def test_track_id_survives_an_occlusion():
    tr = ByteTracker(TrackerConfig(track_buffer_s=1.5))
    ids_a, ids_b = [], []
    for k in range(40):
        ts = k / FPS
        dets = [det(box_at(400, 50), k=k)]                       # a second, static person elsewhere
        occluded = 15 <= k < 25                                  # person A disappears for 10 frames (0.5 s)
        if not occluded:
            dets.append(det(box_at(20 + 4 * k, 200), k=k))
        tracks = tr.update(dets, k, ts)
        for t in tracks:
            if t.observed_now and t.bbox[0] < 350:
                ids_a.append(t.track_id)
            elif t.observed_now:
                ids_b.append(t.track_id)
    assert len(set(ids_a)) == 1, f"person A changed id across the occlusion: {sorted(set(ids_a))}"
    assert len(set(ids_b)) == 1
    assert ids_a[0] != ids_b[0]


def test_low_score_box_keeps_the_track_alive():
    """ByteTrack's second association: a blurred or half-occluded box with a low score keeps the id."""
    tr = ByteTracker()
    ids = []
    for k in range(30):
        score = 0.15 if 10 <= k < 20 else 0.8                  # below track_high, above track_low
        tracks = tr.update([det(box_at(20 + 3 * k, 100), score=score, k=k)], k, k / FPS)
        ids += [t.track_id for t in tracks if t.observed_now]
    assert len(set(ids)) == 1 and len(ids) >= 28


def test_predict_only_carries_tracks_across_skipped_frames():
    tr = ByteTracker()
    seen = set()
    for k in range(30):
        box = box_at(20 + 3 * k, 100)
        tracks = tr.update([det(box, k=k)], k, k / FPS) if k % 2 == 0 else tr.predict_only(k, k / FPS)
        for t in tracks:
            seen.add(t.track_id)
            if not t.observed_now:
                assert abs(t.bbox[0] - box[0]) < 6, "the Kalman prediction should follow the walker"
    assert seen == {1}


def test_dwell_net_and_path_follow_section_2_definitions():
    # Section 2 track 3: 33 contiguous frames at 20 fps -> 1.6 s (t_last - t_first, 32 intervals)
    assert round(dwell_seconds(0, 32, 1 / FPS), 1) == 1.6
    # track 1: 70 contiguous frames -> 3.45 s, printed 3.5 s
    assert dwell_seconds(0, 69, 1 / FPS) == pytest.approx(3.45) and round(dwell_seconds(0, 69, 1 / FPS), 1) == 3.5
    # track 12: 11 frames observed with gaps across 24 intervals -> 1.2 s
    assert dwell_seconds(100, 124, 1 / FPS) == pytest.approx(1.2)
    # net vs path on a back-and-forth walk
    pts = [(0, 0), (30, 0), (10, 0), (40, 0)]
    assert net_displacement(pts) == pytest.approx(40)
    assert path_length(pts) == pytest.approx(30 + 20 + 30)


def test_tracker_reports_section_2_signals_for_a_pacing_person():
    """A person pacing in place for 5 s: large path, small net -> the loitering signature (section 2 track 6)."""
    tr = ByteTracker()
    centres = []
    last = None
    for k in range(100):                                         # 100 frames at 20 fps = 5.0 s
        x = 300 + 60 * math.sin(2 * math.pi * k / 50)            # 60 px back and forth, two full cycles
        x += 0.7 * k                                             # slowly drifting: net ~70 px
        tracks = tr.update([det(box_at(x - 10, 150), k=k)], k, k / FPS)
        centres.append((x, 175.0))
        last = tracks[0]
    assert last.frames == 100
    assert last.dwell_s == pytest.approx(99 / FPS, abs=1e-6)          # 4.95 s, printed 5.0 s like track 6
    assert last.net_px == pytest.approx(net_displacement(centres), abs=2.0)
    assert last.path_px == pytest.approx(path_length(centres), rel=0.05)
    loitering = last.dwell_s >= 3.0 and last.net_px < 90.0            # section 2 rule
    assert loitering and last.path_px > 2 * last.net_px
    # rule-engine shape: history is (timestamp_s, x, y) bottom-centre ground points, oldest first
    t_last, gx, gy = last.history[-1]
    assert t_last == pytest.approx(99 / FPS) and gy == pytest.approx(200.0, abs=1.0)
    assert len(last.history) == len(last.observations) == 100


def test_class_gate_keeps_rider_and_bike_apart():
    tr = ByteTracker()
    for k in range(10):
        b = box_at(50 + 3 * k, 100, 30, 60)
        tracks = tr.update([det(b, cls="person", k=k), det(b, cls="two_wheeler", k=k)], k, k / FPS)
    assert {t.class_name for t in tracks} == {"person", "two_wheeler"}
    assert len({t.track_id for t in tracks}) == 2


def test_section_2_table_reproduces_from_the_signal_definitions(capsys):
    from tools.reproduce_rule_sanity import run_table

    assert run_table() == 0, capsys.readouterr().out
