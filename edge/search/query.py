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

import math
import time
from datetime import datetime
from pathlib import Path

from search import cameras
from search.embed import SiglipEmbedder, get_embedder
from search.index import VectorIndex


def _score_percents(cosines: list[float], embedder: SiglipEmbedder) -> list[int]:
    """Map cosine similarities to 0-100 "match" percentages using SigLIP's own
    calibration, not a per-batch rescale.

    SigLIP is trained with a sigmoid loss, not a softmax: the checkpoint
    itself learns a temperature (``logit_scale``) and an additive bias
    (``logit_bias``) so that ``sigmoid(cos * logit_scale + logit_bias)`` is
    literally the probability the model was trained to output for "does this
    image match this text" (see `SiglipModel.forward`, `logits_per_text =
    cos * logit_scale.exp() + logit_bias`, and its own docstring example:
    ``probs = torch.sigmoid(logits_per_image)``). That is an absolute,
    checkpoint-calibrated score — every query, every batch, every corpus size
    is put through the exact same function, so an unrelated query legitimately
    scores low instead of being stretched up to look confident.

    This module used to min-max rescale each batch's own cosine scores into a
    fixed [55, 99] display band. That guaranteed the top hit of *any* batch —
    including a batch of five totally irrelevant clips — displayed as roughly
    "99% match", which is dishonest: the number stopped meaning "confidence"
    and became "rank within this particular result set". Using the model's
    own sigmoid probability fixes that while leaving the *ordering* (the one
    thing a rescale actually needs to preserve) untouched, because sigmoid is
    monotonic in its input.
    """
    scale, bias = embedder.logit_scale, embedder.logit_bias
    out = []
    for c in cosines:
        logit = c * scale + bias
        prob = 1.0 / (1.0 + math.exp(-logit)) if abs(logit) < 700 else (1.0 if logit > 0 else 0.0)
        out.append(int(round(prob * 100)))
    return out


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
        percents = _score_percents([hit.score for hit in hits], embedder)
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
