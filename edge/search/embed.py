"""SigLIP embeddings for keyframes and search queries.

MODEL CHOICE — why SigLIP, why this checkpoint, why plain transformers+torch
instead of an ONNX export, for an air-gapped CPU box.

- The PDF (slide 3) names SigLIP, pretrained, for semantic embeddings used in
  archived-footage search. `google/siglip-base-patch16-224` is the canonical
  pretrained checkpoint: an open-weights dual encoder (image tower + text
  tower) that projects both into one 768-d space, so "white pickup near
  pillar 42" and a keyframe of exactly that scene land close together under
  cosine similarity. It is ~400 MB, runs comfortably on CPU, and — unlike
  CLIP's softmax-over-batch contrastive loss — SigLIP's sigmoid loss makes its
  similarity scores well-behaved for one-query-vs-many-candidates search
  (the case here), not just batch classification.
- Downloaded once, cached locally (the standard Hugging Face cache directory
  under HF_HOME, or wherever the box's cache lives). `from_pretrained` resolves
  from that local cache without a network round trip once the weights are
  present; deployment sets `HF_HUB_OFFLINE=1` (transformers' own offline
  switch) so a cache miss fails loudly instead of silently reaching out —
  see edge/README.md for the one-time provisioning step on a box that still
  has network, before it is air-gapped.
- Plain `transformers` + CPU `torch`, not an ONNX export: this repo prefers
  onnxruntime (see edge/requirements.txt), and that remains the right call for
  the single-tower appearance model in edge/models (Phase 2), which needs to
  run per-frame at TARGET_FPS. SigLIP here runs only at index time (once per
  keyframe, in a slow offline backfill) and at query time (once per typed
  search, a human waiting on a text box, not a frame budget). Exporting a
  *dual*-tower SigLIP to ONNX correctly — two separate graphs, two separate
  pooling heads, a shared projection space — is real additional engineering
  risk for zero latency benefit at this call rate. CPU torch inference for a
  base-size SigLIP on a handful of images, or a handful of words, is low
  hundreds of milliseconds (measured in tests/test_search.py and
  edge/search/index.py's docstring) and is not a bottleneck anywhere in the
  ~30 ms alert path — search runs asynchronously, off to the side, never on
  it. If a future phase needs SigLIP embedding at frame-rate, that is the
  moment to invest in an ONNX export; today it would be premature.
- No paid API, no network at query time: the only network access this module
  ever performs is the one-time weight download, identical to how the rest of
  the stack already treats Moondream 2, PaddleOCR and YuNet (see
  edge/requirements.txt's Phase 7 comment).
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

import numpy as np

log = logging.getLogger("truewatch.edge.search.embed")

_DEFAULT_MODEL_ID = "google/siglip-base-patch16-224"

# One model per process. Loading it is the expensive part (hundreds of ms to a
# few seconds); every embed call after that reuses it.
_lock = threading.Lock()
_cache: dict[str, "SiglipEmbedder"] = {}


def get_embedder(model_id: str | None = None) -> "SiglipEmbedder":
    """Return the process-wide embedder for `model_id`, loading it if needed."""
    key = model_id or _DEFAULT_MODEL_ID
    with _lock:
        embedder = _cache.get(key)
        if embedder is None:
            embedder = SiglipEmbedder(key)
            _cache[key] = embedder
        return embedder


class SiglipEmbedder:
    """Wraps a pretrained SigLIP dual encoder for CPU inference.

    Lazy: the actual model weights load on first use (`_ensure_loaded`), not
    at import time, so importing this module never touches disk or network
    and stays cheap for callers (like edge/app.py) that only need the search
    route mounted, not necessarily hit yet.
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch  # local import: keep torch out of processes that never search
            from transformers import SiglipModel, SiglipProcessor

            log.info("loading SigLIP %s for local CPU inference", self.model_id)
            self._torch = torch
            self._processor = SiglipProcessor.from_pretrained(self.model_id)
            model = SiglipModel.from_pretrained(self.model_id)
            model.eval()
            self._model = model

    @staticmethod
    def _as_array(features) -> np.ndarray:
        """Newer `transformers` releases wrap get_*_features in a pooling output
        object instead of returning the tensor directly; older ones return the
        tensor. Handle both so this module isn't pinned to one exact minor
        version's return type."""
        tensor = getattr(features, "pooler_output", features)
        return tensor.numpy().astype(np.float32)

    @staticmethod
    def _l2_normalise(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def embed_images(self, images: Iterable[np.ndarray]) -> np.ndarray:
        """Embed a batch of images (HxWx3, RGB, uint8) into (N, D) float32, L2-normalised."""
        from PIL import Image

        images = list(images)
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        pil_images = [Image.fromarray(img).convert("RGB") for img in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
        vectors = self._as_array(features)
        return self._l2_normalise(vectors)

    def embed_text(self, texts: Iterable[str]) -> np.ndarray:
        """Embed a batch of query strings into (N, D) float32, L2-normalised."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_loaded()
        inputs = self._processor(
            text=texts, return_tensors="pt", padding="max_length", truncation=True
        )
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
        vectors = self._as_array(features)
        return self._l2_normalise(vectors)

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        config = self._model.config
        # SigLIP has no separate projection head: both towers share one hidden
        # size, and different transformers releases expose it under different
        # attribute names, so try the ones that have existed and fall back to
        # asking the vision tower directly rather than guessing a constant.
        for attr in ("projection_dim", "hidden_size"):
            value = getattr(config, attr, None)
            if isinstance(value, int):
                return value
        return int(config.vision_config.hidden_size)
