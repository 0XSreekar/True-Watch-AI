"""Fetch and verify the YuNet face detector weights, pinned by URL and SHA-256.

YuNet ships pretrained in the OpenCV Zoo (Apache-2.0) — the PDF (slide 2,
capability 3) specifies it pretrained, with no fine-tuning. There is nothing to
fine-tune and nothing to train, so this module only ever fetches the one
upstream file, the same way `edge/models/detector_weights.py` fetches the
appearance-channel detector: plain HTTPS, no token, SHA-256 verified before the
file is used, streamed into a temp file and renamed atomically so a crashed
download never leaves a half-written file behind.

The URL and hash are pinned as constants (not read from a manifest, unlike the
trained appearance detector): YuNet is a fixed third-party asset with no
training run of ours behind it, so there is no `hf_model.json`-style record to
point at. `FACE_MODEL_URL` / `FACE_MODEL_SHA256` in the environment override
the pin, for an air-gapped deployment mirroring the file locally.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("truewatch.edge.face")

# opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx,
# pinned at the `main` branch commit resolved when this module was written.
# Verified: size 232589 bytes, sha256 below.
PINNED_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
PINNED_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
PINNED_SIZE = 232589

CHUNK = 1 << 20
DOWNLOAD_TIMEOUT_S = 60.0
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

# edge/models/cache/ already exists and is gitignored (models/.gitignore: `cache/`);
# the detector download shares it, and so does this one — one cache dir, no new
# tracked path.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "cache"


class FaceWeightsError(RuntimeError):
    """The YuNet weights could not be resolved, downloaded, verified or opened."""


@dataclass(frozen=True)
class ModelSpec:
    url: str
    sha256: str
    size: int | None


def _check_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https" and parsed.netloc:
        return url
    if parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS:
        return url
    if parsed.scheme == "file" and parsed.path:
        return url
    raise FaceWeightsError(f"refusing to download YuNet from {url!r}: use https://, http:// to localhost, or file://")


def resolve_spec() -> ModelSpec:
    url = os.environ.get("FACE_MODEL_URL", "").strip() or PINNED_URL
    sha256 = os.environ.get("FACE_MODEL_SHA256", "").strip().lower() or PINNED_SHA256
    size = PINNED_SIZE if url == PINNED_URL else None
    return ModelSpec(url=_check_url(url), sha256=sha256, size=size)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_path(spec: ModelSpec, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"yunet-{spec.sha256[:16]}.onnx"


def fetch(
    spec: ModelSpec | None = None,
    cache_dir: Path | None = None,
    timeout: float = DOWNLOAD_TIMEOUT_S,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Path:
    """Return a verified local path to the YuNet ONNX file, downloading if needed."""
    spec = spec or resolve_spec()
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _cached_path(spec, cache_dir)

    if target.exists():
        if _sha256_file(target) == spec.sha256:
            return target
        log.warning("cached YuNet weights %s do not match the pinned SHA-256; re-downloading", target.name)
        target.unlink()

    fd, tmp_name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=cache_dir)
    tmp = Path(tmp_name)
    digest, received = hashlib.sha256(), 0
    try:
        request = urllib.request.Request(spec.url, headers={"User-Agent": "truewatch-edge"})
        with os.fdopen(fd, "wb") as out, opener(request, timeout=timeout) as response:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                received += len(block)
                digest.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if spec.size is not None and received != spec.size:
            raise FaceWeightsError(f"{spec.url} sent {received} bytes, expected {spec.size}")
        got = digest.hexdigest()
        if got != spec.sha256:
            raise FaceWeightsError(
                f"YuNet download from {spec.url} has SHA-256 {got}, expected {spec.sha256}; discarded."
            )
        os.replace(tmp, target)
    except FaceWeightsError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:  # network, TLS, disk
        tmp.unlink(missing_ok=True)
        raise FaceWeightsError(f"downloading YuNet from {spec.url} failed: {type(exc).__name__}: {exc}") from exc
    return target
