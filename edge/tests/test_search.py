"""Tests for edge/search/: relevance against the real SigLIP model, and the
response-shape contract against backend/src/services/search.service.js (the
mock FootageSearch already renders).

The relevance test (`test_pillar_42_query_ranks_the_seeded_clip_in_top_3`)
never mocks the embedding — it loads the real `google/siglip-base-patch16-224`
model and ranks real image embeddings against a real text embedding, because a
mocked embedding would prove nothing about whether search actually works.
SigLIP is trained on natural photographs, not vector-art scenes, so the
synthetic clips below lean on two signals it is reliably sensitive to even on
rendered scenes: colour words ("white" vs each distractor's distinct colour)
and rendered text (SigLIP's pretraining corpus is web imagery, much of it
captioned or containing on-image text, so it does respond to legible text
baked into the pixels). Every distractor differs from the target in *both*
signals at once, which is what keeps a top-3 assertion meaningful rather than
lucky: a model with no real discrimination would be about as likely to place
the target outside the top 3 of 5 as inside it.

A separate, explicitly-labelled unit test further down mocks the embedder to
exercise the index/backfill plumbing in isolation; it is additional coverage,
not a replacement for the real-model test.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from search import query as search_query
from search.backfill import ClipMeta, backfill, index_clips
from search.embed import get_embedder
from search.index import VectorIndex

FRAME_SIZE = 256


def _scene(bg, shape_color, shape, label) -> np.ndarray:
    """A small synthetic 'camera frame': a coloured shape plus a text label."""
    frame = np.full((FRAME_SIZE, FRAME_SIZE, 3), bg, dtype=np.uint8)
    if shape == "pickup":
        # A boxy truck silhouette: cab + bed, unmistakably vehicle-shaped.
        cv2.rectangle(frame, (60, 130), (200, 190), shape_color, -1)
        cv2.rectangle(frame, (130, 95), (200, 130), shape_color, -1)
        cv2.circle(frame, (90, 190), 16, (30, 30, 30), -1)
        cv2.circle(frame, (175, 190), 16, (30, 30, 30), -1)
    elif shape == "motorcycle":
        cv2.line(frame, (70, 170), (190, 170), shape_color, 10)
        cv2.circle(frame, (80, 185), 22, shape_color, 4)
        cv2.circle(frame, (180, 185), 22, shape_color, 4)
    elif shape == "sedan":
        cv2.rectangle(frame, (55, 140), (205, 185), shape_color, -1)
        cv2.ellipse(frame, (130, 140), (55, 25), 0, 180, 360, shape_color, -1)
        cv2.circle(frame, (90, 185), 15, (30, 30, 30), -1)
        cv2.circle(frame, (175, 185), 15, (30, 30, 30), -1)
    elif shape == "person":
        cv2.circle(frame, (128, 90), 18, shape_color, -1)
        cv2.rectangle(frame, (108, 108), (148, 190), shape_color, -1)

    cv2.putText(
        frame, label, (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA
    )
    return frame


def _write_clip(path, frame, frames=6, fps=6.0):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_SIZE, FRAME_SIZE)
    )
    assert writer.isOpened(), "OpenCV could not open an mp4 writer"
    for _ in range(frames):
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0


# Distinct colour AND distinct text for every clip, target included — see the
# module docstring for why that double difference is what makes "top 3"
# a meaningful bar rather than a coin flip.
SCENES = [
    # (clip_id, camera_id, shape, bgr_color, label, chips)
    ("clip-target-pillar42", "RXL-01", "pickup", (235, 235, 235), "PILLAR 42",
     ["white", "pickup", "northbound"]),
    ("clip-distractor-gate7", "JGB-03", "sedan", (30, 30, 200), "GATE 7",
     ["red", "sedan", "gate"]),
    ("clip-distractor-riverbank", "PNT-07", "motorcycle", (200, 130, 30), "RIVER BANK",
     ["blue", "motorcycle", "riverbank"]),
    ("clip-distractor-lane2", "SNL-01", "sedan", (20, 20, 20), "LANE 2",
     ["black", "sedan", "lane"]),
    ("clip-distractor-culvert", "GLG-05", "person", (40, 160, 40), "CULVERT",
     ["green", "person", "culvert"]),
]


@pytest.fixture(scope="module")
def clips_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("clips")
    for clip_id, camera_id, shape, color, label, chips in SCENES:
        frame = _scene(bg=(210, 210, 210), shape_color=color, shape=shape, label=label)
        clip_path = directory / f"{clip_id}.mp4"
        _write_clip(clip_path, frame)
        sidecar = directory / f"{clip_id}.json"
        sidecar.write_text(
            f'{{"clip_id": "{clip_id}", "camera_id": "{camera_id}", '
            f'"captured_at": "2026-08-14T02:41:00+00:00", "chips": {chips}}}'.replace("'", '"')
        )
    return directory


@pytest.fixture(scope="module")
def built_index(clips_dir, tmp_path_factory):
    """Backfill the seeded corpus once, with the real SigLIP model, and reuse it."""
    embedder = get_embedder()
    index_path = tmp_path_factory.mktemp("index") / "index.npz"
    summary = backfill(clips_dir, index_path, keyframes_per_clip=2)
    assert summary["clips_indexed"] == len(SCENES)
    index = VectorIndex.load(index_path)
    return index, embedder


def test_pillar_42_query_ranks_the_seeded_clip_in_top_3(built_index):
    index, embedder = built_index
    result = search_query.run_search(
        "white pickup near pillar 42", index, embedder, limit=len(SCENES)
    )

    ranked_camera_ids = [r["camera"]["id"] for r in result["results"]]
    print("\nranked cameras for 'white pickup near pillar 42':", ranked_camera_ids)
    for r in result["results"]:
        print(f"  {r['camera']['id']:>8}  score={r['score']:>3}  time={r['time']}")

    assert "RXL-01" in ranked_camera_ids[:3], (
        "expected the seeded white-pickup-at-pillar-42 clip's camera (RXL-01) "
        f"in the top 3, got {ranked_camera_ids}"
    )


def test_response_shape_matches_the_frontend_contract(built_index):
    """Key-by-key, type-by-type match against backend/src/services/search.service.js.

    That mock returns:
        { query, tookMs, windowScanned,
          results: [{ id, camera: {id,name,sector,ir,road}, time, chips, score }] }
    """
    index, embedder = built_index
    result = search_query.run_search("white pickup near pillar 42", index, embedder, limit=3)

    assert isinstance(result["query"], str)
    assert isinstance(result["tookMs"], (int, float))
    assert isinstance(result["windowScanned"], str)
    assert isinstance(result["results"], list)
    assert len(result["results"]) >= 1

    for entry in result["results"]:
        assert set(entry.keys()) == {"id", "camera", "time", "chips", "score"}
        assert isinstance(entry["id"], int)
        assert isinstance(entry["time"], str)
        assert isinstance(entry["score"], int)
        assert 0 <= entry["score"] <= 100
        assert isinstance(entry["chips"], list)
        assert all(isinstance(c, str) for c in entry["chips"])

        camera = entry["camera"]
        assert set(camera.keys()) == {"id", "name", "sector", "ir", "road"}
        assert isinstance(camera["id"], str)
        assert isinstance(camera["name"], str)
        assert isinstance(camera["sector"], str)
        assert isinstance(camera["ir"], bool)
        assert isinstance(camera["road"], bool)


def test_no_query_returns_empty_results_not_an_error(built_index):
    index, embedder = built_index
    result = search_query.run_search("", index, embedder, limit=6)
    assert result["results"] == []
    assert result["query"] == ""


# --- mocked-embedding unit test: index/backfill plumbing, not relevance -----


class _FakeEmbedder:
    """A deterministic stand-in for SiglipEmbedder, used only to test the
    incremental-indexing mechanics in isolation from any real model — never
    for the relevance assertion above."""

    dim = 8

    def embed_images(self, images):
        images = list(images)
        return np.stack([np.full(self.dim, (i + 1) / 10.0, dtype=np.float32) for i, _ in enumerate(images)]) if images else np.zeros((0, self.dim), dtype=np.float32)

    def embed_text(self, texts):
        texts = list(texts)
        return np.stack([np.full(self.dim, 0.1, dtype=np.float32) for _ in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)


def test_backfill_is_incremental_and_idempotent(clips_dir, tmp_path):
    from search.backfill import discover_clips

    fake = _FakeEmbedder()
    index = VectorIndex(dim=fake.dim)
    clips = discover_clips(clips_dir)

    first = index_clips(clips, index, fake, keyframes_per_clip=2)
    assert first == len(SCENES)
    count_after_first = len(index)

    second = index_clips(clips, index, fake, keyframes_per_clip=2)
    assert second == 0, "re-running over the same clips must not duplicate entries"
    assert len(index) == count_after_first


def test_index_persists_atomically_and_round_trips(tmp_path):
    fake = _FakeEmbedder()
    index = VectorIndex(dim=fake.dim)
    vectors = fake.embed_images([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)])
    index.add(["c1#0", "c2#0", "c3#0"], vectors, [
        {"clip_id": "c1", "camera_id": "RXL-01", "time_label": "t1", "chips": ["a"], "captured_at": "2026-01-01T00:00:00+00:00"},
        {"clip_id": "c2", "camera_id": "RXL-01", "time_label": "t2", "chips": ["b"], "captured_at": "2026-01-02T00:00:00+00:00"},
        {"clip_id": "c3", "camera_id": "RXL-01", "time_label": "t3", "chips": ["c"], "captured_at": "2026-01-03T00:00:00+00:00"},
    ])
    path = tmp_path / "index.npz"
    index.save(path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp*")), "no temp file should survive a completed atomic save"

    reloaded = VectorIndex.load(path)
    assert len(reloaded) == len(index)
    hits = reloaded.search(np.full(fake.dim, 0.3, dtype=np.float32), k=3)
    assert {h.clip_id for h in hits} == {"c1", "c2", "c3"}


def test_index_scales_to_100k_synthetic_vectors():
    """Measures and records numpy-brute-force query latency at the size the
    build plan asks index.py to justify against (< 100k keyframes). See
    edge/search/index.py's module docstring for the design rationale this
    number backs up."""
    rng = np.random.default_rng(0)
    dim = 768  # SiglipModel's projection_dim for the base checkpoint
    n = 100_000

    index = VectorIndex(dim=dim)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"clip-{i}#0" for i in range(n)]
    metadata = [{"clip_id": f"clip-{i}", "camera_id": "RXL-01", "time_label": "t"} for i in range(n)]
    index.add(ids, vectors, metadata)

    query_vector = rng.standard_normal(dim).astype(np.float32)
    query_vector /= np.linalg.norm(query_vector)

    t0 = time.time()
    hits = index.search(query_vector, k=6)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"\n100k-vector brute-force search latency: {elapsed_ms:.1f} ms")
    assert len(hits) == 6
    assert elapsed_ms < 2000, f"query took {elapsed_ms:.1f} ms, expected well under a human-perceptible search delay"
