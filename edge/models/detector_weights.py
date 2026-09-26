"""Resolve, download, verify and open the appearance-channel detector at boot.

The detector is the ONNX file that training/scripts/export_onnx.py exports and pushes to a public
Hugging Face repository. Weights are never committed; what IS committed is
training/results/hf_model.json, which pins the file to one commit of that repository and carries
its SHA-256. This module turns that record (or the same two facts given as environment variables)
into an onnxruntime InferenceSession:

  1. resolve   YOLO_MODEL_URL + YOLO_MODEL_SHA256 when both are set, else the manifest at
               YOLO_MODEL_MANIFEST (default ../training/results/hf_model.json, i.e. the committed
               record in a repository checkout). Neither -> no detector is configured.
  2. download  plain HTTPS with the standard library into MODEL_CACHE_DIR (default models/cache).
               No token is sent and none is needed: the repository is public by construction
               (export_onnx.py refuses --private). http:// is accepted for localhost only, and
               file:// for an offline mirror.
  3. verify    SHA-256 (and the size when the manifest records it) of the downloaded bytes, BEFORE
               the file takes its final name: the download streams into a temporary file in the
               cache directory and is renamed atomically only after the digest matches. A cached
               file is re-hashed on every boot and replaced if it does not match.
  4. open      onnxruntime with ONNX_PROVIDER; the graph's own metadata (input size, class names)
               is checked against the manifest so a mismatched file cannot start silently.

Nothing here imports Ultralytics or torch. Ultralytics is AGPL-3.0 and stays inside training/; the
serving path runs the exported graph with onnxruntime only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("truewatch.edge.detector")

MANIFEST_SCHEMA = "truewatch.hf_model.v1"
CHUNK = 1 << 20
DOWNLOAD_TIMEOUT_S = 60.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class DetectorWeightsError(RuntimeError):
    """The detector cannot be resolved, downloaded, verified or opened. The message says what to do."""


@dataclass(frozen=True)
class ModelSpec:
    """Where the detector comes from and what its bytes must hash to."""

    url: str
    sha256: str
    file: str
    origin: str                               # "environment" or "manifest <path>"
    size: int | None = None
    imgsz: int | None = None
    class_names: tuple[str, ...] | None = None
    repo_id: str | None = None
    revision: str | None = None


@dataclass
class Detector:
    """An open session plus the facts the pipeline needs to feed it."""

    session: Any                              # onnxruntime.InferenceSession
    path: Path
    spec: ModelSpec
    input_name: str
    imgsz: int
    class_names: list[str] = field(default_factory=list)
    downloaded: bool = False

    def describe(self) -> dict:
        """What /healthz reports: provenance, never a secret."""
        return {
            "file": self.spec.file,
            "sha256": self.spec.sha256,
            "origin": self.spec.origin,
            "repo_id": self.spec.repo_id,
            "revision": self.spec.revision,
            "imgsz": self.imgsz,
            "class_names": self.class_names,
            "providers": list(self.session.get_providers()),
        }


# --------------------------------------------------------------------------------------------
# 1. Resolve
# --------------------------------------------------------------------------------------------


def _check_sha256(value: str, where: str) -> str:
    sha = (value or "").strip().lower()
    if not _SHA256_RE.match(sha):
        raise DetectorWeightsError(f"{where} must be a 64-character hex SHA-256, got {value!r}")
    return sha


def check_url(url: str) -> str:
    """Accept https://, http:// to this machine only, and file:// (an offline mirror). Refuse the rest."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https" and parsed.netloc:
        return url
    if parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS:
        return url
    if parsed.scheme == "file" and parsed.path:
        return url
    raise DetectorWeightsError(
        f"refusing to download the detector from {url!r}: use an https:// URL (the resolve_url in "
        "training/results/hf_model.json), http:// to localhost only, or file:// for a local mirror"
    )


def _file_name(url: str) -> str:
    name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    return name if name.endswith(".onnx") else "detector.onnx"


def spec_from_manifest(path: Path) -> ModelSpec:
    """Read training/results/hf_model.json (written by export_onnx.py --push-to-hub)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DetectorWeightsError(f"cannot read the model manifest {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != MANIFEST_SCHEMA:
        got = doc.get("schema") if isinstance(doc, dict) else type(doc).__name__
        raise DetectorWeightsError(f"{path} has schema {got!r}, expected {MANIFEST_SCHEMA!r}; re-run export_onnx.py --push-to-hub")
    url = doc.get("resolve_url")
    if not url:
        raise DetectorWeightsError(f"{path} has no resolve_url")
    names = doc.get("class_names")
    return ModelSpec(
        url=check_url(str(url)),
        sha256=_check_sha256(str(doc.get("sha256", "")), f"sha256 in {path}"),
        file=str(doc.get("file") or _file_name(str(url))),
        origin=f"manifest {path}",
        size=int(doc["size"]) if doc.get("size") is not None else None,
        imgsz=int(doc["imgsz"]) if doc.get("imgsz") is not None else None,
        class_names=tuple(str(n) for n in names) if isinstance(names, list) else None,
        repo_id=doc.get("repo_id"),
        revision=doc.get("revision"),
    )


def resolve_spec(url: str = "", sha256: str = "", manifest: Path | None = None) -> ModelSpec | None:
    """The detector to load: both environment values, else the manifest file, else None (not configured)."""
    url, sha256 = (url or "").strip(), (sha256 or "").strip()
    if url or sha256:
        if not (url and sha256):
            missing = "YOLO_MODEL_SHA256" if url else "YOLO_MODEL_URL"
            raise DetectorWeightsError(
                f"{missing} is empty while its partner is set. A detector URL is only usable together with the "
                "SHA-256 of the file it serves; set both, or neither to use the committed manifest."
            )
        return ModelSpec(url=check_url(url), sha256=_check_sha256(sha256, "YOLO_MODEL_SHA256"),
                         file=_file_name(url), origin="environment")
    if manifest is not None and Path(manifest).is_file():
        return spec_from_manifest(Path(manifest))
    return None


# --------------------------------------------------------------------------------------------
# 2-3. Download and verify
# --------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_path(spec: ModelSpec, cache_dir: Path) -> Path:
    """Content-addressed name: a new model never overwrites the file an older boot verified."""
    return Path(cache_dir) / f"{Path(spec.file).stem}-{spec.sha256[:16]}.onnx"


def fetch(
    spec: ModelSpec,
    cache_dir: Path,
    timeout: float = DOWNLOAD_TIMEOUT_S,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[Path, bool]:
    """Return (verified local path, whether it was downloaded now). Raises DetectorWeightsError."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cached_path(spec, cache_dir)
    if target.exists():
        if sha256_file(target) == spec.sha256:
            return target, False
        log.warning("cached detector %s does not match its SHA-256; downloading it again", target.name)
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
                if spec.size is not None and received > spec.size:
                    raise DetectorWeightsError(f"{spec.url} sent more than the {spec.size} bytes the manifest records")
                digest.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if spec.size is not None and received != spec.size:
            raise DetectorWeightsError(f"{spec.url} sent {received} bytes, the manifest records {spec.size}")
        got = digest.hexdigest()
        if got != spec.sha256:
            raise DetectorWeightsError(
                f"the file at {spec.url} has SHA-256 {got}, expected {spec.sha256} ({spec.origin}). It was discarded. "
                "The URL or the recorded hash is wrong, or the download was altered."
            )
        os.replace(tmp, target)
    except DetectorWeightsError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:  # network, TLS, disk: one error type with the URL, no partial file left behind
        tmp.unlink(missing_ok=True)
        raise DetectorWeightsError(f"downloading the detector from {spec.url} failed: {type(exc).__name__}: {exc}") from exc
    return target, True


# --------------------------------------------------------------------------------------------
# 4. Open
# --------------------------------------------------------------------------------------------


def open_session(path: Path, provider: str = "CPUExecutionProvider"):
    """An onnxruntime session on `provider`. Refuses a provider this onnxruntime build does not have."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider not in available:
        raise DetectorWeightsError(f"ONNX_PROVIDER={provider!r} is not available in this onnxruntime build; available: {available}")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=[provider])


def _graph_facts(session) -> tuple[str, int | None, list[str] | None]:
    spec = session.get_inputs()[0]
    height = spec.shape[2] if len(spec.shape) == 4 and isinstance(spec.shape[2], int) else None
    meta = session.get_modelmeta().custom_metadata_map or {}
    names = None
    if meta.get("class_names"):
        try:
            names = [str(n) for n in json.loads(meta["class_names"])]
        except ValueError:
            names = None
    return spec.name, height, names


def load_detector(spec: ModelSpec, cache_dir: Path, provider: str = "CPUExecutionProvider",
                  expected_imgsz: int | None = None) -> Detector:
    """Fetch, verify and open `spec`; check the graph agrees with the manifest and the configured size."""
    path, downloaded = fetch(spec, cache_dir)
    session = open_session(path, provider)
    input_name, height, names = _graph_facts(session)
    for label, want in (("the manifest", spec.imgsz), ("YOLO_IMG_SIZE", expected_imgsz)):
        if want is not None and height is not None and height != want:
            raise DetectorWeightsError(f"{path.name} takes {height} px input but {label} says {want}")
    if spec.class_names is not None and names is not None and list(spec.class_names) != names:
        raise DetectorWeightsError(f"{path.name} carries classes {names} but the manifest lists {list(spec.class_names)}")
    imgsz = height or spec.imgsz or expected_imgsz or 640
    return Detector(session=session, path=path, spec=spec, input_name=input_name, imgsz=int(imgsz),
                    class_names=names or list(spec.class_names or ()), downloaded=downloaded)


def load_at_boot(cfg) -> Detector | None:
    """The service's entry point. None when no detector is configured; raises when one is but cannot load."""
    spec = resolve_spec(cfg.yolo_model_url, cfg.yolo_model_sha256, cfg.yolo_model_manifest)
    if spec is None:
        log.warning(
            "no detector configured: set YOLO_MODEL_URL and YOLO_MODEL_SHA256, or provide %s "
            "(export_onnx.py --push-to-hub writes it). The appearance channel is off.",
            cfg.yolo_model_manifest,
        )
        return None
    detector = load_detector(spec, cfg.model_cache_dir, cfg.onnx_provider, cfg.yolo_img_size)
    log.info("detector %s ready (%s, sha256 %s, %s)", detector.path.name,
             "downloaded" if detector.downloaded else "cached", spec.sha256[:12], spec.origin)
    return detector
