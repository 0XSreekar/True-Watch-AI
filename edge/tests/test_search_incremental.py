"""Live indexing of event keyframes (search/incremental.py) into the index /api/search reads."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pipeline.events import EventSink
from search.incremental import MAX_KEYFRAMES_PER_CLIP, KeyframeIndexer
from search.index import VectorIndex
from search.query import run_search

DIM = 8


class FakeEmbedder:
    """Deterministic 'embeddings': a frame's mean colour becomes its vector, and text maps to a colour."""

    logit_scale, logit_bias = 20.0, -10.0
    dim = DIM

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.gate.set()
        self.calls = 0

    @staticmethod
    def _norm(v):
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def embed_images(self, images):
        self.gate.wait(10)
        self.calls += 1
        out = np.zeros((len(images), DIM), dtype=np.float32)
        for i, img in enumerate(images):
            out[i, :3] = img.reshape(-1, 3).mean(axis=0) + 1e-3        # RGB mean
            out[i] = self._norm(out[i])
        return out

    def embed_text(self, texts):
        return np.stack([self._norm([1, 0, 0, 0, 0, 0, 0, 0] if "red" in t else [0, 0, 1, 0, 0, 0, 0, 0]) for t in texts])


def ev(event_id: str, track: int | None, camera: str = "RXL-01", **extra) -> dict:
    base = {"event_id": event_id, "camera_id": camera, "captured_at": "2026-08-14T02:41:07.312Z",
            "detection": {"track_id": track, "class": "truck", "ir": False}, "rule": None, "anpr": None}
    return {**base, **extra}


def frame(rgb) -> np.ndarray:
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:, :] = rgb[::-1]           # stored BGR, as the pipeline produces it
    return img


@pytest.fixture
def parts(tmp_path):
    emb, index = FakeEmbedder(), VectorIndex(DIM)
    idx = KeyframeIndexer(emb, index, tmp_path / "index.npz", save_every_s=0.0)
    yield emb, index, idx, tmp_path / "index.npz"
    idx.close()


def test_event_keyframes_become_searchable_in_the_frozen_result_shape(parts):
    emb, index, idx, path = parts
    assert idx.submit(ev("e-red", 1, rule={"fired": ["fence crossed"]}, anpr={"text": "BA 1 PA 2345"}),
                      frame((255, 0, 0)))
    assert idx.submit(ev("e-blue", 2, camera="PNT-07"), frame((0, 0, 255)))
    assert idx.flush(10)
    assert len(index) == 2 and path.exists()
    body = run_search("a red truck", index, emb, limit=6)
    assert set(body) == {"query", "tookMs", "windowScanned", "results"}
    top = body["results"][0]
    assert set(top) == {"id", "camera", "time", "chips", "score"}
    assert top["camera"]["id"] == "RXL-01" and top["time"] == "14 AUG · 02:41"
    assert "truck" in top["chips"] and "fence crossed" in top["chips"] and "plate BA 1 PA 2345" in top["chips"]
    assert VectorIndex.load(path).contains_clip("e-red")          # persisted for the next start


def test_events_on_one_track_share_a_clip_and_are_capped(parts):
    emb, index, idx, _ = parts
    for i in range(MAX_KEYFRAMES_PER_CLIP + 3):
        idx.submit(ev(f"e{i}", 7), frame((255, 0, 0)))
    assert idx.flush(10)
    assert len(index) == MAX_KEYFRAMES_PER_CLIP
    assert {m["clip_id"] for m in index.all_metadata()} == {"e0"}
    assert len(run_search("red", index, emb)["results"]) == 1     # one person/vehicle, one result


def test_submit_never_blocks_and_the_queue_is_bounded(tmp_path):
    emb, index = FakeEmbedder(), VectorIndex(DIM)
    emb.gate.clear()                                              # the embedder is stuck
    idx = KeyframeIndexer(emb, index, tmp_path / "i.npz", queue_max=4)
    t0 = time.perf_counter()
    for i in range(60):
        idx.submit(ev(f"e{i}", i), frame((255, 0, 0)))
        assert idx._q.qsize() <= 4
    assert time.perf_counter() - t0 < 2.0
    assert idx.dropped >= 60 - 4 - 8                              # at most one in-flight batch plus the queue
    emb.gate.set()
    idx.close()


def test_an_embedding_failure_is_counted_not_raised(tmp_path):
    class Broken(FakeEmbedder):
        def embed_images(self, images):
            raise RuntimeError("no model")

    idx = KeyframeIndexer(Broken(), VectorIndex(DIM), tmp_path / "i.npz")
    assert idx.submit(ev("e1", 1), frame((1, 2, 3)))
    idx.flush(5)
    assert idx.errors == 1 and idx.indexed == 0
    idx.close()


def test_the_service_hands_each_event_frame_to_the_indexer(tmp_path):
    pytest.importorskip("scipy")
    from pipeline.service import CameraService
    from tests.synth import Scene
    from tests.test_service import FakeAppearance, ingest_frames

    emb, index = FakeEmbedder(), VectorIndex(DIM)
    idx = KeyframeIndexer(emb, index, tmp_path / "i.npz", save_every_s=0.0)
    scene = Scene(speed=1.5, start=(20.0, 110.0))
    sink = EventSink()
    svc = CameraService(FakeAppearance(scene), camera_id="CAM-T", post_id="RXL", sink=sink, stride=1,
                        calibration_dir=tmp_path / "cal", warmup_frames=30, keyframes=idx)
    for f in ingest_frames(scene, 90):
        svc.process(f)
    svc.close()
    # under ALERT_POLICY=rules a fusion-only event is an observation, but it is still indexed for search
    assert sink.total == 0 and sink.observations >= 1
    assert idx.flush(10) and idx.submitted == sink.observations and len(index) >= 1
    idx.close()


def test_concurrent_add_and_search_never_see_a_torn_index():
    index = VectorIndex(DIM)
    stop = threading.Event()
    errors = []

    def reader():
        q = np.ones(DIM, dtype=np.float32)
        while not stop.is_set():
            try:
                index.search(q, k=3)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    t = threading.Thread(target=reader)
    t.start()
    for i in range(300):
        index.add([f"c{i}#0"], np.random.rand(1, DIM).astype(np.float32), [{"clip_id": f"c{i}"}])
    stop.set()
    t.join()
    assert not errors and len(index) == 300
