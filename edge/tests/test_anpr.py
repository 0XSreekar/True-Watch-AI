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


def _bordered_plate(line_rows, width=520, height=300, border=8):
    """A light plate with a heavy printed border and dark text bands at the given row ranges."""
    image = np.full((height, width, 3), 225, dtype=np.uint8)
    image[:border, :] = image[-border:, :] = 15
    image[:, :border] = image[:, -border:] = 15
    for top, bottom in line_rows:
        for x in range(40, width - 40, 36):  # separate glyph blocks, like characters
            image[top:bottom, x:x + 24] = 25
    return image


def test_split_lines_ignores_a_heavy_border_that_outweighs_the_text():
    # The regression behind 11.8% unsplit validation plates: the border rows carried more ink
    # than either text line, so the lines looked too faint to count as a second band.
    image = _bordered_plate([(55, 125), (175, 245)])
    lines = rectify.split_lines(image)
    assert len(lines) == 2
    cut = lines[0].shape[0]
    assert 125 <= cut <= 175, f"cut at row {cut}, expected inside the inter-line gap"


def test_split_lines_keeps_a_bordered_single_line_plate_whole():
    # A long single-line plate (Bhutan-style proportions) must not be cut in half.
    image = _bordered_plate([(40, 80)], width=520, height=120)
    assert len(rectify.split_lines(image)) == 1


def test_nepal_grammar_accepts_the_provincial_format_from_the_mock_data():
    # backend/src/data/mockData.js carries "प्र १ ख २३४५": a 3-code-point zone and a 1-digit lot.
    result = postprocess.validate_nepal("प्र१ख२३४५")
    assert result.valid
    assert result.fields["zone"] == "प्र" and result.fields["lot"] == "१"


def test_routing_prefers_the_head_whose_reading_is_a_valid_plate():
    from anpr.recognise import LineReading, _route_by_grammar

    nepali = [LineReading("बा १२", 0.40), LineReading("च ४५६७", 0.40)]
    latin_noise = [LineReading("8A 12", 0.95), LineReading("F 4567", 0.95)]
    assert _route_by_grammar(nepali, latin_noise) == "devanagari"
    bhutan = [LineReading("BP-1-A1234", 0.60)]
    devanagari_noise = [LineReading("बप", 0.90)]
    assert _route_by_grammar(devanagari_noise, bhutan) == "latin"
    assert _route_by_grammar([LineReading("क", 0.5)], [LineReading("X", 0.5)]) is None


# --- One-line plates ------------------------------------------------------------------------


def test_rectify_keeps_a_one_line_plate_at_its_own_aspect():
    # A 4.3:1 plate squeezed into the 1.73:1 two-line canvas had its glyphs stretched to over twice their height.
    plate = _bordered_plate([(40, 80)], width=520, height=120)
    result = rectify.rectify(plate)
    assert result.one_line
    assert result.image.shape[1] == rectify.RECT_WIDTH
    assert result.image.shape[0] < rectify.RECT_HEIGHT // 2


def test_prepare_lines_never_splits_a_one_line_plate():
    # One text band plus the Devanagari headline above it looked like two lines to the splitter: 43% of synthetic
    # one-line plates were cut through the text before the source shape was consulted.
    plate = _bordered_plate([(22, 34), (40, 95)], width=520, height=120)
    lines, _ = rectify.prepare_lines(plate)
    assert len(lines) == 1


def test_prepare_lines_layout_override_forces_the_other_reading():
    two_line = _bordered_plate([(55, 125), (175, 245)])
    assert len(rectify.prepare_lines(two_line)[0]) == 2
    assert len(rectify.prepare_lines(two_line, "one")[0]) == 1
    one_line = _bordered_plate([(40, 80)], width=520, height=120)
    assert rectify.rectify(one_line, "two").image.shape[:2] == (rectify.RECT_HEIGHT, rectify.RECT_WIDTH)


def test_nepal_grammar_accepts_real_province_plates():
    assert postprocess.validate_nepal("बागमतीप्रदेश०२०३७प१६४३").valid
    assert postprocess.validate_nepal("प्रदेश३०२०१३प७१७३").valid
    assert not postprocess.validate_nepal("बागमतीप्रदेश०२३७प१६४३").valid  # the lot is three digits


def test_latin_grammar_accepts_embossed_nepali_plates_and_drops_the_header():
    assert postprocess.strip_latin_header("BAGMATI B AC 5763") == "BAC5763"
    assert postprocess.validate_latin("BAGMATIBAC5763").valid
    assert postprocess.validate_latin("BP-1-A1234").valid
    assert not postprocess.validate_latin("BAC576").valid


def test_strip_latin_header_drops_stray_edge_characters():
    assert postprocess.strip_latin_header("EBAB3985") == "BAB3985"
    assert postprocess.strip_latin_header("BAC5297)") == "BAC5297"


def test_three_line_split_is_opt_in_and_finds_a_province_plate():
    # Header, lot line and serial line, like "बागमती प्रदेश-०२" / "०३१ प" / "२०५०".
    province = _bordered_plate([(30, 60), (95, 150), (190, 270)])
    assert len(rectify.split_lines(province, max_lines=3)) == 3
    two_line = _bordered_plate([(55, 125), (175, 245)])
    assert len(rectify.split_lines(two_line)) == 2


def test_extract_nepal_plate_cuts_junk_around_a_registration():
    assert postprocess.extract_nepal_plate("बतपरेश०४बा२च९५८५") == "बा२च९५८५"
    assert postprocess.extract_nepal_plate("बा१६च९३४५बागमतीरदेश०") == "बा१६च९३४५"
    assert postprocess.extract_nepal_plate("ह६") is None


def test_deskew_levels_a_tilted_two_line_plate():
    plate = _bordered_plate([(55, 125), (175, 245)])
    canvas = np.full((460, 680, 3), 225, dtype=np.uint8)
    canvas[80:380, 80:600] = plate
    matrix = cv2.getRotationMatrix2D((340, 230), 15, 1.0)
    tilted = cv2.warpAffine(canvas, matrix, (680, 460), borderValue=(225, 225, 225))
    assert len(rectify.prepare_lines(tilted, level=True)[0]) == 2


def test_province_header_and_core_patterns():
    assert postprocess.province_header("बागमती प्रदेश-०२") == "बागमतीप्रदेश०२"
    assert postprocess.province_header("प्रदेश ३-०१") == "प्रदेश३०१"
    assert postprocess.province_header("बा२च") is None
    assert postprocess.PROVINCE_CORE_RE.match("०४७प४१९३")
    assert postprocess.validate_nepal("बागमतीप्रदेश०२" + "०४७प४१९३").valid
