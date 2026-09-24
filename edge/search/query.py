"""Natural-language query -> SigLIP text embedding -> ranked clips.

The return shape is not designed here; it is copied from the contract the
frontend already renders. `docs/ARCHITECTURE_V2.md` section 4 freezes it:

    POST /api/search {query} -> {query, tookMs, windowScanned,
                                  results:[{id,camera,time,chips,score}]}

and `backend/src/services/search.service.js` (the mock the console runs
against today) is the executable version of that contract:

    { id, camera: {id,name,sector,ir,road}, time: "1<n> AUG · HH:MM", chips: [...], score }

`tests/test_search.py::test_response_shape_matches_the_frontend_contract`
asserts this module's output against that exact shape, key by key and type by
type, so a change on either side that breaks the other fails loudly.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from search import cameras
from search.embed import SiglipEmbedder, get_embedder
from search.index import VectorIndex


def _score_percents(cosines: list[float]) -> list[int]:
    """Map a batch of cosine similarities to 0-100 "match" percentages.

    SigLIP is trained with a sigmoid (not softmax) loss, so raw cosine
    similarity between a genuinely matching image/text pair is typically a
    modest positive number (single digits to low tens of percent of [0, 1]),
    not the near-1.0 a softmax-contrastive model like CLIP tends to produce
    for a confident match. Displaying that raw number as a percentage would
    read as "12% match" for the single best, correctly-ranked result, which
    misrepresents the actual confidence of the ranking to an operator even
    though the *ordering* is trustworthy.

    So the displayed percentage is a min-max rescaling of this result batch's
    own cosine scores into a fixed display band ([55, 99]): the best match in
    *any* batch reads as a strong match, the weakest as a weak one, and the
    ordering — the only thing this system actually claims — is preserved
    exactly. This mirrors how the existing mock (search.service.js, `96 -
    i*4`) already displays scores: a descending band anchored near the top
    entry, not an absolute confidence figure.
    """
    if not cosines:
        return []
    lo, hi = min(cosines), max(cosines)
    if hi - lo < 1e-9:
        return [90 for _ in cosines]
    band_lo, band_hi = 55.0, 99.0
    return [
        int(round(band_lo + (c - lo) / (hi - lo) * (band_hi - band_lo))) for c in cosines
    ]


def _window_scanned(index: VectorIndex) -> str:
    dates = []
    for meta in index.all_metadata():
        captured_at = meta.get("captured_at")
        if not captured_at:
            continue
        try:
            dates.append(datetime.fromisoformat(str(captured_at).replace("Z", "+00:00")))
        except ValueError:
            continue
    if not dates:
        return "no footage indexed"
    span_days = max(1, (max(dates) - min(dates)).days)
    if span_days >= 14:
        return f"{span_days // 7} weeks"
    return f"{span_days} days"


def run_search(
    query: str,
    index: VectorIndex,
    embedder: SiglipEmbedder,
    limit: int = 6,
) -> dict:
    """Embed `query` and return results in the exact FootageSearch contract shape."""
    t0 = time.time()
    query = (query or "").strip()

    results: list[dict] = []
    if query and len(index) > 0:
        query_vector = embedder.embed_text([query])[0]
        hits = index.search(query_vector, k=limit)
        percents = _score_percents([hit.score for hit in hits])
        for position, (hit, score) in enumerate(zip(hits, percents)):
            camera = cameras.lookup(hit.metadata.get("camera_id", ""))
            results.append(
                {
                    "id": position,
                    "camera": camera,
                    "time": hit.metadata.get("time_label", ""),
                    "chips": list(hit.metadata.get("chips") or []),
                    "score": score,
                }
            )

    return {
        "query": query,
        "tookMs": round((time.time() - t0) * 1000, 1),
        "windowScanned": _window_scanned(index),
        "results": results,
    }


# --- process-wide singletons for the FastAPI route (edge/app.py) -----------

_index_cache: dict[str, VectorIndex] = {}


def get_index(index_path: Path, dim: int) -> VectorIndex:
    key = str(index_path)
    index = _index_cache.get(key)
    if index is None:
        index = VectorIndex.load_or_create(index_path, dim=dim)
        _index_cache[key] = index
    return index


def search_with_config(cfg, query: str) -> dict:
    """Convenience entry point: resolve the embedder+index from a Config and search."""
    embedder = get_embedder(cfg.embed_model_id)
    index = get_index(cfg.search_index_path, dim=embedder.dim)
    return run_search(query, index, embedder, limit=cfg.search_default_limit)
