"""ANPR: the MEASUREMENTS.md section 4 comparison, grammar validation, and plumbing.

The comparison test downloads/loads the pretrained PP-OCRv5 Devanagari and
Latin recognition heads (cached under ~/.paddlex/official_models after the
first run) and is marked `slow` so a quick local loop can skip it; CI and the
pre-push check still run it, because it is the one test the PDF names by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

EDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDGE_ROOT))

from anpr import compare_latin, detect_plate, postprocess, rectify  # noqa: E402


@pytest.mark.slow
def test_measurements_section_4_reproduces(tmp_path):
    """Devanagari reads the confirmed plate correctly; Latin does not."""
    image_path = tmp_path / "plate.png"
    compare_latin.render_confirmed_sample(image_path)
    assert compare_latin.run_comparison(image_path) == 0


class _FakeDetection:
    """Stands in for pipeline.types.Detection (Phase 3, concurrent branch)."""

    def __init__(self, bbox, cls, track_id=None):
        self.bbox = bbox
        self.cls = cls
        self.track_id = track_id


def test_vehicle_detection_protocol_accepts_a_minimal_duck_type():
    detection = _FakeDetection(bbox=(10.0, 10.0, 90.0, 90.0), cls="car", track_id=7)
    assert isinstance(detection, detect_plate.VehicleDetection)
    assert detect_plate.is_plate_bearing(detection)


def test_non_vehicle_class_is_not_plate_bearing():
    detection = _FakeDetection(bbox=(0.0, 0.0, 50.0, 50.0), cls="person")
    assert not detect_plate.is_plate_bearing(detection)


def test_detect_plates_finds_nothing_in_a_blank_frame():
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    detections = [_FakeDetection(bbox=(20.0, 20.0, 180.0, 180.0), cls="car", track_id=1)]
    # A flat grey frame has no plate-shaped gradient structure; the detector
    # should return nothing rather than hallucinate a box.
    candidates = detect_plate.detect_plates(frame, detections)
    assert candidates == []


def test_detect_plates_skips_out_of_frame_bbox():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    detections = [_FakeDetection(bbox=(1000.0, 1000.0, 1010.0, 1010.0), cls="car")]
    assert detect_plate.detect_plates(frame, detections) == []


def _synthetic_two_line_plate() -> np.ndarray:
    """A plate-shaped image with two bands of 'ink' rows and a clear gap between them."""
    image = np.full((120, 300, 3), 235, dtype=np.uint8)
    image[15:45, 20:280] = 20  # top line
    image[75:105, 20:280] = 20  # bottom line
    return image


def test_split_lines_finds_two_lines():
    image = _synthetic_two_line_plate()
    lines = rectify.split_lines(image)
    assert len(lines) == 2
    assert lines[0].shape[0] < image.shape[0]
    assert lines[1].shape[0] < image.shape[0]


def test_split_lines_returns_one_line_when_no_valley():
    # A single line of text: a thin ink band (minority of the plate area, as
    # on a real plate) with no second band and so no inter-line gap.
    image = np.full((60, 300, 3), 235, dtype=np.uint8)
    image[25:35, 20:280] = 20
    lines = rectify.split_lines(image)
    assert len(lines) == 1


def test_upscale_for_recognition_grows_a_narrow_plate():
    small = np.zeros((30, 60, 3), dtype=np.uint8)
    upscaled = rectify.upscale_for_recognition(small, min_width=240)
    assert upscaled.shape[1] >= 240


def test_upscale_for_recognition_leaves_a_wide_plate_alone():
    wide = np.zeros((80, 400, 3), dtype=np.uint8)
    upscaled = rectify.upscale_for_recognition(wide, min_width=240)
    assert upscaled.shape == wide.shape


def test_rectify_handles_a_plain_rectangle_crop():
    plate = np.full((100, 200, 3), 200, dtype=np.uint8)
    result = rectify.rectify(plate)
    assert result.image.shape[:2] == (rectify.RECT_HEIGHT, rectify.RECT_WIDTH)


# --- Grammar validation ---------------------------------------------------------------------


def test_nepal_grammar_accepts_the_confirmed_plate():
    result = postprocess.validate_nepal("बा१२प१२३४")
    assert result.valid
    assert result.fields["zone"] == "बा"
    assert result.fields["lot"] == "१२"
    assert result.fields["vehicle_class"] == "प"
    assert result.fields["serial"] == "१२३४"


def test_nepal_grammar_rejects_latin_digits():
    result = postprocess.validate_nepal("बा12प1234")
    assert not result.valid


def test_bhutan_grammar_accepts_bp_n_annnn():
    result = postprocess.validate_bhutan("BP-1-A2345")
    assert result.valid
    assert result.fields["prefix"] == "BP"
    assert result.fields["district"] == "1"
    assert result.fields["letter"] == "A"
    assert result.fields["serial"] == "2345"


def test_bhutan_grammar_is_case_insensitive():
    assert postprocess.validate_bhutan("bp-1-a2345").valid


def test_bhutan_grammar_rejects_wrong_shape():
    assert not postprocess.validate_bhutan("BP-12-A234").valid
    assert not postprocess.validate_bhutan("NP-1-A2345").valid


def test_validate_dispatches_by_script():
    assert postprocess.validate("बा१२प१२३४", "devanagari").valid
    assert postprocess.validate("BP-1-A2345", "latin").valid


def test_expected_kinds_for_nepal_matches_field_lengths():
    match = postprocess.validate_nepal("बा१२प१२३४")
    kinds = postprocess.expected_kinds_for_nepal(match.fields)
    assert kinds == ["letter", "letter", "digit", "digit", "letter", "digit", "digit", "digit", "digit"]


def test_apply_corrections_uses_the_confusion_table_when_present(tmp_path, monkeypatch):
    confusions_path = tmp_path / "confusions.json"
    confusions_path.write_text(
        '{"schema": "truewatch.anpr_confusions.v1", "confusions": {"9": ["\\u0967"]}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(postprocess, "CONFUSIONS_PATH", confusions_path)
    text = "9२"
    corrected, corrections = postprocess.apply_corrections(text, char_confidences=[0.5, 0.99])
    assert corrected == "१२"
    assert len(corrections) == 1
    assert corrections[0].original == "9"
    assert corrections[0].corrected == "१"


def test_apply_corrections_leaves_high_confidence_characters_alone(tmp_path, monkeypatch):
    confusions_path = tmp_path / "confusions.json"
    confusions_path.write_text(
        '{"schema": "truewatch.anpr_confusions.v1", "confusions": {"9": ["\\u0967"]}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(postprocess, "CONFUSIONS_PATH", confusions_path)
    corrected, corrections = postprocess.apply_corrections("9२", char_confidences=[0.99, 0.99])
    assert corrected == "9२"
    assert corrections == []


def test_apply_corrections_with_no_table_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(postprocess, "CONFUSIONS_PATH", tmp_path / "missing.json")
    corrected, corrections = postprocess.apply_corrections("9२", char_confidences=[0.1, 0.1])
    assert corrected == "9२"
    assert corrections == []
