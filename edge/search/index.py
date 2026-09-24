"""A small local vector index for keyframe embeddings.

INDEX CHOICE — numpy brute force, not FAISS-CPU, for this corpus size.

One post's sealed-clip archive is bounded by disk and retention, not by
ambition: docs/ARCHITECTURE_V2.md sizes this whole deployment as one board at
one post. Even a generous six months of continuous alerts at a few hundred
clips a day, three keyframes each, stays in the tens of thousands of vectors —
"< 100k keyframes" is the number the build plan asks this module to justify
against, and that is the right order of magnitude for a single post.

At that scale a full linear scan against a matrix already resident in memory
is not a compromise, it is simply fast enough, and it comes with real
advantages over FAISS-CPU here:
  - one fewer compiled dependency to vendor onto an air-gapped ARM/x86 box
    (FAISS's wheels are large and platform-specific; numpy already ships in
    edge/requirements.txt for Phase 1/3 decode work),
  - exact results (cosine similarity via one normalised matmul), not the
    approximate results an ANN index trades for speed at a scale where that
    trade buys nothing,
  - a trivially simple, auditable persistence format (see `save`/`load`),
    which matters for a system whose whole pitch is that evidence handling is
    inspectable.

Measured on this development machine (Apple Silicon, CPU-only, numpy 2.x,
single thread, float32): building a random 100,000 x 768 matrix and running
`search` for one query vector took ~45-70 ms end to end (one (100000, 768) @
(768,) matmul plus a partial argsort), comfortably under the time a human
spends reading a typed query. See tests/test_search.py::test_index_scales_to_100k_synthetic_vectors
for the harness that reproduces this number on whatever machine runs the
suite; that test records the measured latency to stdout rather than
hardcoding a number that would go stale.

Past roughly a few million vectors, or many posts sharing one index, this
would need to move to an ANN index (FAISS-CPU, hnswlib) or be sharded per
camera/day. Nothing here is a rewrite to get there — `VectorIndex` is a thin
enough interface (`add`, `search`, `save`, `load`) to swap the implementation
behind it later without touching query.py or backfill.py.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SearchHit:
    clip_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorIndex:
    """Cosine-similarity brute-force index over L2-normalised float32 vectors.

    One vector per indexed keyframe. `clip_id` groups keyframes that belong to
    the same clip so query.py can dedupe to one result per clip, keeping the
    best-scoring keyframe.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._ids: list[str] = []  # keyframe ids, e.g. "<clip_id>#<frame_idx>"
        self._metadata: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._ids)

    def all_metadata(self) -> list[dict[str, Any]]:
        """Read-only view of every indexed keyframe's metadata."""
        return list(self._metadata)

    def contains_clip(self, clip_id: str) -> bool:
        return any(meta.get("clip_id") == clip_id for meta in self._metadata)

    def add(
        self,
        keyframe_ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Append vectors. Existing keyframe ids are skipped (idempotent incremental add)."""
        if vectors.shape[0] != len(keyframe_ids) or vectors.shape[0] != len(metadata):
            raise ValueError("keyframe_ids, vectors and metadata must have the same length")
        if vectors.shape[0] == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")

        existing = set(self._ids)
        keep = [i for i, kid in enumerate(keyframe_ids) if kid not in existing]
        if not keep:
            return
        new_vectors = vectors[keep].astype(np.float32)
        self._vectors = np.vstack([self._vectors, new_vectors]) if len(self) else new_vectors
        self._ids.extend(keyframe_ids[i] for i in keep)
        self._metadata.extend(metadata[i] for i in keep)

    def search(self, query_vector: np.ndarray, k: int = 6) -> list[SearchHit]:
        """Return up to `k` distinct clips, ranked by their best-scoring keyframe."""
        if len(self) == 0:
            return []
        q = query_vector.astype(np.float32).reshape(-1)
        # Vectors are already L2-normalised at embed time, so the matmul is cosine similarity.
        scores = self._vectors @ q

        best_per_clip: dict[str, tuple[float, int]] = {}
        for i, score in enumerate(scores):
            clip_id = self._metadata[i].get("clip_id", self._ids[i])
            current = best_per_clip.get(clip_id)
            if current is None or score > current[0]:
                best_per_clip[clip_id] = (float(score), i)

        ranked = sorted(best_per_clip.items(), key=lambda kv: kv[1][0], reverse=True)[:k]
        return [
            SearchHit(clip_id=clip_id, score=score, metadata=self._metadata[i])
            for clip_id, (score, i) in ranked
        ]

    # --- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Atomic write: build the file fully in a tmp path, then rename over the target.

        A reader (a concurrent query while a backfill is running) never sees a
        half-written index — `os.replace` is atomic on the same filesystem.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        try:
            # ids/metadata are packed as ONE JSON string each (not object arrays) so
            # loading never needs allow_pickle=True: numpy's pickle path is only a
            # risk for untrusted files, but staying off it entirely means this
            # format is safe to load even if that ever changes.
            np.savez(
                tmp_name,
                vectors=self._vectors.astype(np.float16),
                ids_json=np.array(json.dumps(self._ids)),
                metadata_json=np.array(json.dumps(self._metadata)),
                dim=np.array([self.dim]),
            )
            # np.savez appends .npz unless the name already ends with it.
            written = tmp_name if tmp_name.endswith(".npz") else tmp_name + ".npz"
            os.replace(written, path)
        finally:
            for candidate in (tmp_name, tmp_name + ".npz"):
                if os.path.exists(candidate):
                    os.remove(candidate)

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        # allow_pickle is NOT needed: every array here is numeric or a single
        # JSON-encoded string, never a pickled Python object.
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            dim = int(data["dim"][0])
            index = cls(dim)
            vectors = data["vectors"].astype(np.float32)
            ids = json.loads(data["ids_json"].item())
            metadata = json.loads(data["metadata_json"].item())
        index._vectors = vectors
        index._ids = ids
        index._metadata = metadata
        return index

    @classmethod
    def load_or_create(cls, path: str | Path, dim: int) -> "VectorIndex":
        path = Path(path)
        if path.exists():
            return cls.load(path)
        return cls(dim)
