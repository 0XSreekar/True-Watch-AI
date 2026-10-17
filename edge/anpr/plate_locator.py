"""Learned licence-plate locator: a single-class detector exported to ONNX and run with onnxruntime.

The model is trained under training/ (see training/plate_locator/) and is never committed. What IS
committed is edge/anpr/results/plate_locator_model.json, which pins the exported file to one
revision of a Hugging Face repository and carries its SHA-256. Resolution mirrors
anpr/ocr_weights.py: download over plain HTTPS (or a file:// mirror) into
MODEL_CACHE_DIR/anpr-plate-locator-<revision[:12]>/, rename only after the hash matches, reuse a
verified cache without touching the network.

Environment:
  ANPR_PLATE_LOCATOR_MODEL     path to an operator-installed .onnx (wins over the manifest)
  ANPR_PLATE_LOCATOR_MANIFEST  alternative manifest (default results/plate_locator_model.json)
  ANPR_FETCH_PLATE_LOCATOR=0   never download; the classical finder is used
  MODEL_CACHE_DIR              cache root shared with the other edge models

Any failure (not published yet, offline, hash mismatch, onnxruntime missing) returns None from
`get_locator()` and the caller falls back to the classical finder in detect_plate.py.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable
import urllib.request

import cv2
import numpy as np

from anpr.ocr_weights import DEFAULT_CACHE_DIR, DOWNLOAD_TIMEOUT_S, _download, _verified
from models.detector_weights import check_url

log = logging.getLogger("truewatch.edge.anpr")

MANIFEST_SCHEMA = "truewatch.plate_locator_model.v1"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "results" / "plate_locator_model.json"

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_IOU = 0.5


class PlateLocatorError(RuntimeError):
    """The pinned locator cannot be resolved, downloaded or verified. The message says why."""


# --- weights --------------------------------------------------------------------------------------

def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PlateLocatorError(f"cannot read the plate locator manifest {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != MANIFEST_SCHEMA:
        raise PlateLocatorError(f"{path} is not a {MANIFEST_SCHEMA} manifest")
    if not doc.get("published", True):
        raise PlateLocatorError(f"{path}: the locator weights are not published yet")
    name, url, sha = doc.get("file"), doc.get("resolve_url"), str(doc.get("sha256", ""))
    if not (name and url and len(sha) == 64 and doc.get("revision")):
        raise PlateLocatorError(f"{path} must carry file, resolve_url, revision and a 64-hex sha256")
    check_url(str(url))
    if not str(url).endswith("/" + str(name)):
        raise PlateLocatorError(f"{path}: resolve_url {url!r} does not serve {name!r}")
    return doc


def ensure_local(
    manifest: Path = DEFAULT_MANIFEST,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    timeout: float = DOWNLOAD_TIMEOUT_S,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Path:
    """Path of the verified .onnx file. Raises PlateLocatorError."""
    doc = load_manifest(manifest)
    target_dir = Path(cache_dir) / f"anpr-plate-locator-{str(doc['revision'])[:12]}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target, sha = target_dir / str(doc["file"]), str(doc["sha256"])
    if _verified(target, sha):
        return target
    if target.exists():
        log.warning("cached %s does not match its pinned SHA-256; downloading it again", target)
        target.unlink()
    try:
        _download(str(doc["resolve_url"]), target, sha, timeout, opener)
    except Exception as exc:  # ocr_weights raises its own error type; one type for this module
        raise PlateLocatorError(str(exc)) from exc
    return target


def resolve_model_path() -> Path | None:
    explicit = os.environ.get("ANPR_PLATE_LOCATOR_MODEL", "").strip()
    if explicit:
        return Path(explicit)
    if os.environ.get("ANPR_FETCH_PLATE_LOCATOR", "1").strip() == "0":
        return None
    manifest = Path(os.environ.get("ANPR_PLATE_LOCATOR_MANIFEST", "").strip() or DEFAULT_MANIFEST)
    cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", "").strip() or DEFAULT_CACHE_DIR)
    try:
        return ensure_local(manifest, cache_dir)
    except PlateLocatorError as exc:
        log.warning("anpr: plate locator unavailable (%s); using the classical finder", exc)
        return None


# --- inference ------------------------------------------------------------------------------------

class PlateLocator:
    """YOLO-style single-class ONNX detector. `locate(bgr)` returns [([x1,y1,x2,y2], score), ...] best first."""

    def __init__(self, model_path: str | Path, conf: float = CONF_THRESHOLD, size: int = INPUT_SIZE):
        import onnxruntime as ort  # imported here so the classical path never needs it

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        self.conf = conf
        self.size = size

    def locate(self, bgr: np.ndarray) -> list[tuple[list[float], float]]:
        if bgr is None or bgr.size == 0:
            return []
        h, w = bgr.shape[:2]
        scale = self.size / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # Letterbox into a size x size canvas padded to a multiple of 32 only where needed.
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self._session.run(None, {self._input: np.ascontiguousarray(blob)})[0]
        pred = np.squeeze(out, 0)  # (5, N) for a one-class head
        if pred.shape[0] != 5 and pred.shape[1] == 5:
            pred = pred.T
        if pred.shape[0] != 5:
            raise PlateLocatorError(f"unexpected locator output shape {out.shape}")
        scores = pred[4]
        keep = scores >= self.conf
        if not keep.any():
            return []
        cx, cy, bw, bh = pred[0][keep], pred[1][keep], pred[2][keep], pred[3][keep]
        sc = scores[keep]
        x1 = np.clip((cx - bw / 2 - left) / scale, 0, w)
        y1 = np.clip((cy - bh / 2 - top) / scale, 0, h)
        x2 = np.clip((cx + bw / 2 - left) / scale, 0, w)
        y2 = np.clip((cy + bh / 2 - top) / scale, 0, h)
        boxes = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
        idx = cv2.dnn.NMSBoxes(boxes, [float(s) for s in sc], self.conf, NMS_IOU)
        result = []
        for i in np.array(idx).reshape(-1):
            bx, by, bw_, bh_ = boxes[int(i)]
            if bw_ >= 2 and bh_ >= 2:
                result.append(([bx, by, bx + bw_, by + bh_], float(sc[int(i)])))
        result.sort(key=lambda t: -t[1])
        return result


_LOCATOR: PlateLocator | None = None
_RESOLVED = False


def get_locator() -> PlateLocator | None:
    """The shared locator, or None when none is available (classical fallback)."""
    global _LOCATOR, _RESOLVED
    if _RESOLVED:
        return _LOCATOR
    _RESOLVED = True
    path = resolve_model_path()
    if path is None:
        return None
    try:
        _LOCATOR = PlateLocator(path)
    except Exception as exc:  # onnxruntime missing, corrupt file: degrade, do not stop the pipeline
        log.warning("anpr: plate locator %s failed to load (%s); using the classical finder", path, exc)
    return _LOCATOR


def reset_locator() -> None:
    """Forget the cached resolution (tests, or after an operator swaps the model)."""
    global _LOCATOR, _RESOLVED
    _LOCATOR, _RESOLVED = None, False
