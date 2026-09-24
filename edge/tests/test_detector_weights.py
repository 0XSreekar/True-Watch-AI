"""The detector loader: resolve, download over HTTP(S), verify SHA-256, rename atomically, open.

A tiny ONNX graph is built in the test and served from a local HTTP server on 127.0.0.1, so the
real download path (urllib, streaming, hashing, the temporary file and the rename) runs end to end
without the network.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest

from config import Config, ConfigError
from models import detector_weights as D

EDGE_ROOT = Path(__file__).resolve().parents[1]
NAMES = ["person", "two_wheeler", "car", "truck"]
IMGSZ = 64


def _tiny_detector(path: Path, imgsz: int = IMGSZ, names=NAMES) -> Path:
    """A one-node graph with the exported detector's interface: images (batch,3,s,s) -> output0, plus metadata."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["output0"], axes=[1], keepdims=1)],
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, imgsz, imgsz])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", 1, imgsz, imgsz])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    for key, value in {"class_names": json.dumps(list(names)), "imgsz": json.dumps([imgsz, imgsz])}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def served(tmp_path):
    """Serve tmp_path/www on 127.0.0.1; yields (base url, served dir, request log)."""
    www = tmp_path / "www"
    www.mkdir()
    hits: list[str] = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(www), **kwargs)

        def do_GET(self):  # noqa: N802 - http.server API
            hits.append(self.path)
            hits.append("auth:" + str(self.headers.get("Authorization")))
            super().do_GET()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", www, hits
    finally:
        server.shutdown()
        server.server_close()


def _manifest(path: Path, url: str, model: Path, **over) -> Path:
    doc = {"schema": D.MANIFEST_SCHEMA, "repo_id": "someone/detector", "revision": "a" * 40, "file": model.name,
           "resolve_url": url, "sha256": _sha(model), "size": model.stat().st_size, "opset": 17, "imgsz": IMGSZ,
           "class_names": NAMES, **over}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------
# Download, verify, open
# --------------------------------------------------------------------------------------------


def test_downloads_verifies_renames_and_opens_a_session(served, tmp_path):
    base, www, hits = served
    model = _tiny_detector(www / "det.onnx")
    spec = D.spec_from_manifest(_manifest(tmp_path / "hf_model.json", f"{base}/det.onnx", model))
    cache = tmp_path / "cache"
    detector = D.load_detector(spec, cache, expected_imgsz=IMGSZ)
    assert detector.downloaded and detector.path.parent == cache and detector.path.name.endswith(".onnx")
    assert _sha(detector.path) == spec.sha256
    assert sorted(p.name for p in cache.iterdir()) == [detector.path.name]    # no temporary file left behind
    assert detector.class_names == NAMES and detector.imgsz == IMGSZ and detector.input_name == "images"
    out = detector.session.run(None, {"images": np.ones((2, 3, IMGSZ, IMGSZ), np.float32)})[0]
    assert out.shape == (2, 1, IMGSZ, IMGSZ)
    assert "auth:None" in hits                                               # no token is ever sent
    assert detector.describe()["sha256"] == spec.sha256


def test_a_verified_cache_is_reused_without_a_request(served, tmp_path):
    base, www, hits = served
    model = _tiny_detector(www / "det.onnx")
    spec = D.resolve_spec(f"{base}/det.onnx", _sha(model))
    D.fetch(spec, tmp_path / "cache")
    n = len(hits)
    path, downloaded = D.fetch(spec, tmp_path / "cache")
    assert not downloaded and len(hits) == n


def test_a_tampered_cache_is_replaced(served, tmp_path):
    base, www, _ = served
    model = _tiny_detector(www / "det.onnx")
    spec = D.resolve_spec(f"{base}/det.onnx", _sha(model))
    path, _ = D.fetch(spec, tmp_path / "cache")
    path.write_bytes(b"tampered")
    again, downloaded = D.fetch(spec, tmp_path / "cache")
    assert downloaded and _sha(again) == spec.sha256


def test_a_hash_mismatch_is_refused_and_leaves_nothing_behind(served, tmp_path):
    base, www, _ = served
    _tiny_detector(www / "det.onnx")
    spec = D.resolve_spec(f"{base}/det.onnx", "0" * 64)
    cache = tmp_path / "cache"
    with pytest.raises(D.DetectorWeightsError, match="SHA-256"):
        D.fetch(spec, cache)
    assert list(cache.iterdir()) == []


def test_a_size_mismatch_is_refused(served, tmp_path):
    base, www, _ = served
    model = _tiny_detector(www / "det.onnx")
    spec = D.spec_from_manifest(_manifest(tmp_path / "m.json", f"{base}/det.onnx", model, size=10))
    with pytest.raises(D.DetectorWeightsError, match="more than the 10 bytes"):
        D.fetch(spec, tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_a_missing_file_is_one_clear_error(served, tmp_path):
    base, _, _ = served
    with pytest.raises(D.DetectorWeightsError, match="failed"):
        D.fetch(D.resolve_spec(f"{base}/nope.onnx", "1" * 64), tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_a_file_url_serves_an_offline_mirror(tmp_path):
    model = _tiny_detector(tmp_path / "mirror.onnx")
    spec = D.resolve_spec(model.resolve().as_uri(), _sha(model))
    detector = D.load_detector(spec, tmp_path / "cache")
    assert detector.path.exists() and detector.class_names == NAMES


def test_the_graph_must_agree_with_the_manifest_and_the_configured_size(served, tmp_path):
    base, www, _ = served
    model = _tiny_detector(www / "det.onnx")
    spec = D.spec_from_manifest(_manifest(tmp_path / "m.json", f"{base}/det.onnx", model, class_names=NAMES + ["cart"]))
    with pytest.raises(D.DetectorWeightsError, match="carries classes"):
        D.load_detector(spec, tmp_path / "c1")
    ok = D.spec_from_manifest(_manifest(tmp_path / "m2.json", f"{base}/det.onnx", model))
    with pytest.raises(D.DetectorWeightsError, match="YOLO_IMG_SIZE"):
        D.load_detector(ok, tmp_path / "c2", expected_imgsz=640)


# --------------------------------------------------------------------------------------------
# Resolution rules
# --------------------------------------------------------------------------------------------


def test_environment_wins_over_the_manifest_and_needs_both_values(tmp_path):
    model = _tiny_detector(tmp_path / "m.onnx")
    manifest = _manifest(tmp_path / "hf_model.json", "https://huggingface.co/someone/detector/resolve/" + "a" * 40 + "/m.onnx", model)
    from_manifest = D.resolve_spec("", "", manifest)
    assert from_manifest.origin.startswith("manifest") and from_manifest.revision == "a" * 40
    from_env = D.resolve_spec("https://example.org/x/other.onnx", "B" * 64, manifest)
    assert from_env.origin == "environment" and from_env.sha256 == "b" * 64 and from_env.file == "other.onnx"
    with pytest.raises(D.DetectorWeightsError, match="YOLO_MODEL_SHA256"):
        D.resolve_spec("https://example.org/x.onnx", "", manifest)
    with pytest.raises(D.DetectorWeightsError, match="YOLO_MODEL_URL"):
        D.resolve_spec("", "c" * 64, manifest)
    assert D.resolve_spec("", "", tmp_path / "absent.json") is None


@pytest.mark.parametrize("url", ["http://example.org/m.onnx", "ftp://example.org/m.onnx", "https:///m.onnx", "m.onnx"])
def test_only_https_localhost_http_and_file_urls_are_accepted(url):
    with pytest.raises(D.DetectorWeightsError, match="refusing"):
        D.check_url(url)


def test_a_bad_hash_or_foreign_manifest_is_refused(tmp_path):
    with pytest.raises(D.DetectorWeightsError, match="64-character"):
        D.resolve_spec("https://example.org/m.onnx", "not-a-hash")
    bad = tmp_path / "hf_model.json"
    bad.write_text(json.dumps({"schema": "something.else"}), encoding="utf-8")
    with pytest.raises(D.DetectorWeightsError, match="schema"):
        D.resolve_spec("", "", bad)


def test_config_requires_url_and_hash_together():
    base = dict(ingest_mode="file", rtsp_url="", target_fps=8.0, env="development", ingest_key="")
    Config(**base, yolo_model_url="", yolo_model_sha256="").validate()
    with pytest.raises(ConfigError, match="YOLO_MODEL_URL and YOLO_MODEL_SHA256"):
        Config(**base, yolo_model_url="https://example.org/m.onnx", yolo_model_sha256="").validate()


def test_boot_without_a_detector_configured_returns_none(tmp_path):
    cfg = Config(ingest_mode="file", rtsp_url="", target_fps=8.0, env="development", ingest_key="",
                 yolo_model_url="", yolo_model_sha256="", yolo_model_manifest=tmp_path / "absent.json",
                 model_cache_dir=tmp_path / "cache")
    assert D.load_at_boot(cfg) is None


def test_boot_with_the_manifest_loads_the_detector(served, tmp_path):
    base, www, _ = served
    model = _tiny_detector(www / "det.onnx")
    cfg = Config(ingest_mode="file", rtsp_url="", target_fps=8.0, env="development", ingest_key="",
                 yolo_model_url="", yolo_model_sha256="",
                 yolo_model_manifest=_manifest(tmp_path / "hf_model.json", f"{base}/det.onnx", model),
                 model_cache_dir=tmp_path / "cache", yolo_img_size=IMGSZ)
    detector = D.load_at_boot(cfg)
    assert detector is not None and detector.spec.repo_id == "someone/detector"


# --------------------------------------------------------------------------------------------
# Licence boundary
# --------------------------------------------------------------------------------------------


def test_nothing_in_edge_imports_ultralytics_or_torch():
    offenders = []
    for path in EDGE_ROOT.rglob("*.py"):
        if any(part in {".venv", "venv", "cache"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(import|from)\s+(ultralytics|torch)\b", text, re.M):
            offenders.append(str(path.relative_to(EDGE_ROOT)))
    assert offenders == []
