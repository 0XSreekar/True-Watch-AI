"""Download and verify the fine-tuned Devanagari plate recogniser pinned in hf_ocr_model.json.

Capability 4 (slide 2) depends on the fine-tuned head: the pretrained PP-OCRv5 Devanagari head
reads real Nepali plates poorly (edge/anpr/results/ANPR_METRICS.md). Weights are never committed;
what IS committed is edge/anpr/results/hf_ocr_model.json, which pins the exported inference files
to one commit of a public Hugging Face repository and carries each file's SHA-256. This module
turns that record into a local model directory PaddleOCR's TextRecognition(model_dir=...) reads:

  1. resolve   the manifest at ANPR_OCR_MANIFEST (default results/hf_ocr_model.json beside this file)
  2. download  plain HTTPS with the standard library into
               MODEL_CACHE_DIR/anpr-devanagari-<revision[:12]>/ (default edge/models/cache), one file
               at a time into a temporary name, renamed only after its SHA-256 matches the manifest
  3. reuse     a cached directory whose files all still hash correctly is returned without a download

ANPR_DEVANAGARI_MODEL_DIR still wins when set (an operator-installed checkpoint), and
ANPR_FETCH_FINETUNED=0 turns the download off (the pretrained head is then used, with a warning).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

from models.detector_weights import check_url, sha256_file

log = logging.getLogger("truewatch.edge.anpr")

MANIFEST_SCHEMA = "truewatch.hf_ocr_model.v1"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "results" / "hf_ocr_model.json"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "cache"
DOWNLOAD_TIMEOUT_S = 60.0
CHUNK = 1 << 20


class OcrWeightsError(RuntimeError):
    """The pinned recogniser cannot be resolved, downloaded or verified. The message says why."""


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OcrWeightsError(f"cannot read the recogniser manifest {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != MANIFEST_SCHEMA:
        raise OcrWeightsError(f"{path} is not a {MANIFEST_SCHEMA} manifest")
    files, urls, hashes = doc.get("files"), doc.get("resolve_urls"), doc.get("sha256")
    if not (isinstance(files, list) and isinstance(urls, list) and isinstance(hashes, dict)) or len(files) != len(urls):
        raise OcrWeightsError(f"{path} must list files, one resolve_url per file, and a sha256 per file")
    for name, url in zip(files, urls):
        check_url(str(url))
        if not str(url).endswith("/" + name):
            raise OcrWeightsError(f"{path}: resolve_url {url!r} does not serve {name!r}")
        if len(str(hashes.get(name, ""))) != 64:
            raise OcrWeightsError(f"{path}: no SHA-256 recorded for {name!r}")
    if not doc.get("revision"):
        raise OcrWeightsError(f"{path} has no revision")
    return doc


def model_dir_for(doc: dict, cache_dir: Path) -> Path:
    """Revision-addressed directory: a newer model never overwrites one an older boot verified."""
    return Path(cache_dir) / f"anpr-devanagari-{str(doc['revision'])[:12]}"


def _verified(path: Path, sha256: str) -> bool:
    return path.is_file() and sha256_file(path) == sha256


def _download(url: str, target: Path, sha256: str, timeout: float, opener: Callable[..., Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=target.parent)
    tmp = Path(tmp_name)
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "truewatch-edge"})
        with os.fdopen(fd, "wb") as out, opener(request, timeout=timeout) as response:
            for block in iter(lambda: response.read(CHUNK), b""):
                digest.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if digest.hexdigest() != sha256:
            raise OcrWeightsError(f"{url} has SHA-256 {digest.hexdigest()}, the manifest pins {sha256}; discarded")
        os.replace(tmp, target)
    except OcrWeightsError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:  # network, TLS, disk: one error type naming the URL, no partial file left
        tmp.unlink(missing_ok=True)
        raise OcrWeightsError(f"downloading {url} failed: {type(exc).__name__}: {exc}") from exc


def ensure_local(
    manifest: Path = DEFAULT_MANIFEST,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    timeout: float = DOWNLOAD_TIMEOUT_S,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Path:
    """Return a local directory holding every pinned file with a matching SHA-256. Raises OcrWeightsError."""
    doc = load_manifest(manifest)
    target_dir = model_dir_for(doc, cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, url in zip(doc["files"], doc["resolve_urls"]):
        target, sha = target_dir / name, str(doc["sha256"][name])
        if _verified(target, sha):
            continue
        if target.exists():
            log.warning("cached %s does not match its pinned SHA-256; downloading it again", target)
            target.unlink()
        _download(str(url), target, sha, timeout, opener)
    return target_dir


def resolve_model_dir() -> str | None:
    """The fine-tuned model directory to load, or None for the pretrained head.

    ANPR_DEVANAGARI_MODEL_DIR (operator-installed) wins; otherwise the pinned manifest is fetched
    unless ANPR_FETCH_FINETUNED=0. A failure falls back to the pretrained head with a warning, so
    plate reading degrades instead of stopping the pipeline.
    """
    explicit = os.environ.get("ANPR_DEVANAGARI_MODEL_DIR", "").strip()
    if explicit:
        return explicit
    if os.environ.get("ANPR_FETCH_FINETUNED", "1").strip() == "0":
        log.warning("anpr: ANPR_FETCH_FINETUNED=0, using the pretrained Devanagari head (weaker on real plates)")
        return None
    manifest = Path(os.environ.get("ANPR_OCR_MANIFEST", "").strip() or DEFAULT_MANIFEST)
    cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", "").strip() or DEFAULT_CACHE_DIR)
    try:
        path = ensure_local(manifest, cache_dir)
    except OcrWeightsError as exc:
        log.warning("anpr: fine-tuned recogniser unavailable (%s); using the pretrained head", exc)
        return None
    return str(path)
