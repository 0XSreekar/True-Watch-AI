"""The pinned plate recogniser is fetched, verified and reused; a bad file never loads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anpr import ocr_weights

FILES = ("inference.json", "inference.pdiparams", "inference.yml")


def _mirror(tmp_path: Path, tamper: str | None = None) -> Path:
    """A file:// mirror of three model files and a manifest pinning their hashes."""
    src = tmp_path / "mirror"
    src.mkdir()
    hashes = {}
    for name in FILES:
        data = f"payload of {name}".encode()
        hashes[name] = hashlib.sha256(data).hexdigest()
        (src / name).write_bytes(data + (b"x" if name == tamper else b""))
    manifest = tmp_path / "hf_ocr_model.json"
    manifest.write_text(json.dumps({
        "schema": ocr_weights.MANIFEST_SCHEMA,
        "revision": "0123456789abcdef0123",
        "files": list(FILES),
        "resolve_urls": [(src / n).as_uri() for n in FILES],
        "sha256": hashes,
    }), encoding="utf-8")
    return manifest


def test_downloads_verifies_and_reuses(tmp_path):
    manifest = _mirror(tmp_path)
    cache = tmp_path / "cache"
    first = ocr_weights.ensure_local(manifest, cache)
    assert first.name == "anpr-devanagari-0123456789ab"
    assert sorted(p.name for p in first.iterdir()) == sorted(FILES)

    calls = []

    def no_network(*args, **kwargs):
        calls.append(args)
        raise AssertionError("a verified cache must not download again")

    assert ocr_weights.ensure_local(manifest, cache, opener=no_network) == first
    assert calls == []


def test_tampered_file_is_refused_and_not_left_behind(tmp_path):
    manifest = _mirror(tmp_path, tamper="inference.pdiparams")
    cache = tmp_path / "cache"
    with pytest.raises(ocr_weights.OcrWeightsError, match="SHA-256"):
        ocr_weights.ensure_local(manifest, cache)
    target = ocr_weights.model_dir_for(json.loads(manifest.read_text()), cache)
    assert not (target / "inference.pdiparams").exists()
    assert not list(target.glob(".download-*"))


def test_corrupted_cache_is_replaced(tmp_path):
    manifest = _mirror(tmp_path)
    cache = tmp_path / "cache"
    directory = ocr_weights.ensure_local(manifest, cache)
    (directory / "inference.yml").write_bytes(b"corrupted")
    ocr_weights.ensure_local(manifest, cache)
    assert (directory / "inference.yml").read_bytes() == b"payload of inference.yml"


def test_resolution_order_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("ANPR_DEVANAGARI_MODEL_DIR", "/operator/model")
    assert ocr_weights.resolve_model_dir() == "/operator/model"

    monkeypatch.delenv("ANPR_DEVANAGARI_MODEL_DIR")
    monkeypatch.setenv("ANPR_FETCH_FINETUNED", "0")
    assert ocr_weights.resolve_model_dir() is None

    monkeypatch.setenv("ANPR_FETCH_FINETUNED", "1")
    monkeypatch.setenv("ANPR_OCR_MANIFEST", str(tmp_path / "missing.json"))
    assert ocr_weights.resolve_model_dir() is None          # degrades to the pretrained head

    monkeypatch.setenv("ANPR_OCR_MANIFEST", str(_mirror(tmp_path)))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))
    assert ocr_weights.resolve_model_dir().endswith("anpr-devanagari-0123456789ab")


def test_committed_manifest_is_well_formed():
    doc = ocr_weights.load_manifest()
    assert set(doc["files"]) == set(FILES)
    assert all(u.startswith("https://huggingface.co/") and doc["revision"] in u for u in doc["resolve_urls"])
