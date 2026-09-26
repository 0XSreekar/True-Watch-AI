"""ANPR runs on fusion-agreed vehicle TRACKS, so every plate reading carries its track id."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anpr import detect_plate
from anpr.stage import PlateStage
from pipeline.events import EventSink
from pipeline.service import CameraService
from pipeline.types import Detection
from tests.synth import Scene, make_track
from tests.test_service import FakeAppearance, check_event, ingest_frames

PLATE_TEXT = "बा१२प१२३४"          # parses as a Nepal plate (anpr/postprocess.py)


@dataclass
class FakeReading:
    text: str
    script: str = "devanagari"
    line_confidences: tuple = (0.97, 0.95)


def plated_scene() -> Scene:
    """A car-sized moving object with a light plate carrying dark glyph strokes."""
    scene = Scene(obj_size=(80, 60), speed=1.5, start=(20.0, 110.0))
    h, w = scene.obj.shape
    ph, pw = 18, 44
    y0, x0 = h - ph - 6, (w - pw) // 2
    scene.obj[y0:y0 + ph, x0:x0 + pw] = 240
    for k in range(5):
        scene.obj[y0 + 4:y0 + ph - 4, x0 + 4 + k * 8:x0 + 8 + k * 8] = 15
    return scene


def test_detections_have_no_track_id_but_tracks_do():
    scene = plated_scene()
    img = scene.frame(10)
    det = Detection(bbox=scene.obj_box(10), class_name="car", score=0.8, camera_id="C", frame_index=10,
                    timestamp=0.5, modality="visible")
    trk = make_track(42, scene.obj_box(10), class_name="car")
    assert detect_plate.detect_plates(img, [det])[0].source_track_id is None
    assert detect_plate.detect_plates(img, [trk])[0].source_track_id == 42


def test_plate_stage_reads_once_per_track_and_bounds_attempts():
    scene = plated_scene()
    calls = []
    stage = PlateStage(reader=lambda crop: calls.append(crop.shape) or FakeReading(PLATE_TEXT))
    person = make_track(1, (0, 0, 60, 120), class_name="person")
    car = make_track(7, scene.obj_box(10), class_name="car")
    reads = stage.run(scene.frame(10), [person, car])
    assert [(r.track_id, r.text, r.valid) for r in reads] == [(7, PLATE_TEXT, True)]
    assert reads[0].contract_block() == {"text": PLATE_TEXT, "script": "devanagari", "line_confidences": [0.97, 0.95]}
    assert stage.run(scene.frame(11), [car]) == [], "a track with a valid reading is not read again"
    junk = PlateStage(reader=lambda crop: FakeReading("??"), max_attempts_per_track=2)
    for k in range(5):
        junk.run(scene.frame(k), [make_track(9, scene.obj_box(k), class_name="car")])
    assert junk.attempts == 2 and junk.valid_reads == 0


def test_missing_recogniser_is_reported_not_fatal():
    def reader(_crop):
        raise ImportError("No module named 'paddleocr'")

    scene = plated_scene()
    stage = PlateStage(reader=reader)
    reads = stage.run(scene.frame(10), [make_track(3, scene.obj_box(10), class_name="car")])
    assert reads and reads[0].text is None and reads[0].track_id == 3
    assert "paddleocr" in stage.stats()["recogniser_error"] and stage.candidates == 1


def test_service_attaches_the_plate_to_the_tracks_event(tmp_path):
    scene = plated_scene()
    sink = EventSink()
    svc = CameraService(FakeAppearance(scene, class_name="car"), camera_id="CAM-T", post_id="RXL", sink=sink,
                        stride=1, calibration_dir=tmp_path / "cal", warmup_frames=30,
                        anpr=PlateStage(reader=lambda crop: FakeReading(PLATE_TEXT)))
    for f in ingest_frames(scene, 90):
        svc.process(f)
    svc.close()
    events = sink.recent(100)
    with_plate = [e for e in events if e["anpr"]]
    assert with_plate, "the agreed car track's plate must reach an event"
    ev = with_plate[0]
    check_event(ev)
    assert ev["anpr"]["text"] == PLATE_TEXT and ev["detection"]["class"] == "car"
    fused_ids = {e["detection"]["track_id"] for e in events if e["anpr"] is None}
    assert ev["detection"]["track_id"] in fused_ids, "the reading belongs to the fusion-agreed track"
    assert svc.stats.anpr["attempts"] >= 1 and svc.stats.anpr["valid_reads"] == 1
