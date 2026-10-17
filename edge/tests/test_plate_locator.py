"""The learned plate locator: pinned download, decoding, and the fall-back to the classical finder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from anpr import detect_plate, plate_locator

PAYLOAD = b"onnx bytes stand-in"


def _mirror(tmp_path: Path, tamper: bool = False, published: bool = True) -> Path:
    src = tmp_path / "mirror"
    src.mkdir()
    (src / "plate_locator.onnx").write_bytes(PAYLOAD + (b"x" if tamper else b""))
    manifest = tmp_path / "plate_locator_model.json"
    manifest.write_text(json.dumps({
        "schema": plate_locator.MANIFEST_SCHEMA,
        "published": published,
        "revision": "0123456789abcdef0123",
        "file": "plate_locator.onnx",
        "resolve_url": (src / "plate_locator.onnx").as_uri(),
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
    }), encoding="utf-8")
    return manifest


def test_downloads_verifies_and_reuses(tmp_path):
    manifest, cache = _mirror(tmp_path), tmp_path / "cache"
    first = plate_locator.ensure_local(manifest, cache)
    assert first.parent.name == "anpr-plate-locator-0123456789ab"
    assert first.read_bytes() == PAYLOAD

    def no_network(*args, **kwargs):
        raise AssertionError("a verified cache must not download again")

    assert plate_locator.ensure_local(manifest, cache, opener=no_network) == first


def test_tampered_download_is_refused(tmp_path):
    manifest, cache = _mirror(tmp_path, tamper=True), tmp_path / "cache"
    with pytest.raises(plate_locator.PlateLocatorError, match="SHA-256"):
        plate_locator.ensure_local(manifest, cache)
    assert not list(cache.rglob("*.onnx"))
    assert not list(cache.rglob(".download-*"))


def test_unpublished_manifest_degrades(tmp_path, monkeypatch):
    monkeypatch.setenv("ANPR_FETCH_PLATE_LOCATOR", "1")
    monkeypatch.setenv("ANPR_PLATE_LOCATOR_MANIFEST", str(_mirror(tmp_path, published=False)))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))
    assert plate_locator.resolve_model_path() is None
    assert plate_locator.get_locator() is None


def test_resolution_order(tmp_path, monkeypatch):
    monkeypatch.setenv("ANPR_PLATE_LOCATOR_MODEL", "/operator/plate.onnx")
    assert plate_locator.resolve_model_path() == Path("/operator/plate.onnx")
    monkeypatch.delenv("ANPR_PLATE_LOCATOR_MODEL")
    assert plate_locator.resolve_model_path() is None  # ANPR_FETCH_PLATE_LOCATOR=0 from conftest
    monkeypatch.setenv("ANPR_FETCH_PLATE_LOCATOR", "1")
    monkeypatch.setenv("ANPR_PLATE_LOCATOR_MANIFEST", str(_mirror(tmp_path)))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))
    assert plate_locator.resolve_model_path().read_bytes() == PAYLOAD


def test_corrupt_model_file_degrades_to_none(tmp_path, monkeypatch):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx file")
    monkeypatch.setenv("ANPR_PLATE_LOCATOR_MODEL", str(bad))
    assert plate_locator.get_locator() is None


def test_committed_manifest_is_well_formed():
    doc = json.loads(plate_locator.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert doc["schema"] == plate_locator.MANIFEST_SCHEMA
    assert len(doc["sha256"]) == 64
    assert doc["resolve_url"].startswith("https://huggingface.co/")


class _FakeSession:
    """One confident plate (cx=320, cy=320, 160x80 in the 640 canvas), a weak box and a duplicate."""

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def run(self, _outputs, feeds):
        assert feeds["images"].shape == (1, 3, 640, 640)
        out = np.zeros((1, 5, 3), np.float32)
        out[0, :, 0] = [320, 320, 160, 80, 0.9]
        out[0, :, 1] = [100, 100, 40, 20, 0.05]       # below the confidence threshold
        out[0, :, 2] = [322, 322, 158, 82, 0.6]       # duplicate of the first: removed by NMS
        return [out]


def _locator_with_fake_session():
    loc = plate_locator.PlateLocator.__new__(plate_locator.PlateLocator)
    loc._session, loc._input, loc.conf, loc.size = _FakeSession(), "images", 0.25, 640
    return loc


def test_locate_decodes_letterboxed_output_into_image_coordinates():
    # 320x160 image: scale 2.0, content 640x320 centred vertically (top pad 160).
    found = _locator_with_fake_session().locate(np.zeros((160, 320, 3), np.uint8))
    assert len(found) == 1
    box, score = found[0]
    assert score == pytest.approx(0.9)
    assert box == pytest.approx([120, 60, 200, 100], abs=0.01)


def test_detect_plates_uses_the_locator_when_available(monkeypatch):
    monkeypatch.setattr(detect_plate, "get_locator", _locator_with_fake_session)
    frame = np.zeros((400, 600, 3), np.uint8)
    det = SimpleNamespace(bbox=(100, 50, 420, 210), cls="car", track_id=7)
    (cand,) = detect_plate.detect_plates(frame, [det])
    assert cand.bbox == (220, 110, 300, 150)
    assert cand.source_track_id == 7


def test_detect_plates_falls_back_to_classical_without_a_locator(monkeypatch):
    monkeypatch.setattr(detect_plate, "get_locator", lambda: None)
    calls = []
    monkeypatch.setattr(detect_plate, "find_plates_in_crop", lambda crop: calls.append(crop.shape) or [])
    det = SimpleNamespace(bbox=(0, 0, 300, 200), cls="truck", track_id=1)
    assert detect_plate.detect_plates(np.zeros((400, 600, 3), np.uint8), [det]) == []
    assert calls == [(200, 300, 3)]


def test_a_failing_locator_falls_back_to_classical(monkeypatch):
    class Broken:
        def locate(self, _crop):
            raise RuntimeError("boom")

    monkeypatch.setattr(detect_plate, "get_locator", lambda: Broken())
    monkeypatch.setattr(detect_plate, "find_plates_in_crop", lambda crop: [((1, 2, 30, 20), 0.5)])
    det = SimpleNamespace(bbox=(0, 0, 300, 200), cls="car", track_id=1)
    (cand,) = detect_plate.detect_plates(np.zeros((400, 600, 3), np.uint8), [det])
    assert cand.bbox == (1, 2, 30, 20)
