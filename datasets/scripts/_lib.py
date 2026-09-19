"""Shared helpers for the dataset pipeline.

Every script in this directory uses the same CLI shape, the same structured log
format, the same manifest writer and the same failure discipline: a broken file is
logged and counted, never silently skipped, and a run that loses files exits non-zero.

CPU only. No Ultralytics import is permitted anywhere under datasets/ because that
package is AGPL-3.0 and belongs to training/ alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / "datasets"
CONFIG_DIR = DATASETS_ROOT / "config"
MANIFEST_DIR = DATASETS_ROOT / "manifests"
REPORT_DIR = DATASETS_ROOT / "reports"
DEFAULT_RAW = REPO_ROOT / "var" / "datasets"
DEFAULT_PROCESSED = DATASETS_ROOT / "processed"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- logging


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def log(level: str, script: str, message: str, **fields: Any) -> None:
    """One structured line on stdout. Machine-greppable, human-readable."""
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"{_stamp()} {level:<5} {script:<18} {message}"
    if extra:
        line = f"{line} | {extra}"
    stream = sys.stderr if level in ("ERROR", "FATAL") else sys.stdout
    print(line, file=stream, flush=True)


class Logger:
    def __init__(self, script: str) -> None:
        self.script = script

    def info(self, message: str, **f: Any) -> None:
        log("INFO", self.script, message, **f)

    def warn(self, message: str, **f: Any) -> None:
        log("WARN", self.script, message, **f)

    def error(self, message: str, **f: Any) -> None:
        log("ERROR", self.script, message, **f)

    def fatal(self, message: str, **f: Any) -> None:
        log("FATAL", self.script, message, **f)


# --------------------------------------------------------------------------- counters


@dataclass
class Counters:
    """Nothing is skipped silently: every skip lands in a named bucket and is printed."""

    counts: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def get(self, key: str) -> int:
        return self.counts.get(key, 0)

    def report(self, logger: Logger, title: str = "counters") -> None:
        if not self.counts:
            logger.info(f"{title}: none")
            return
        width = max(len(k) for k in self.counts)
        logger.info(f"{title}:")
        for key in sorted(self.counts):
            print(f"    {key:<{width}}  {self.counts[key]}", flush=True)

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


# --------------------------------------------------------------------------- cli


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dry-run", action="store_true", help="plan the work, write nothing")
    ap.add_argument("--seed", type=int, default=42, help="seed for every random operation")
    ap.add_argument(
        "--raw",
        default=str(DEFAULT_RAW),
        help="directory holding the downloaded source datasets",
    )
    ap.add_argument(
        "--processed",
        default=str(DEFAULT_PROCESSED),
        help="directory holding pipeline output",
    )
    ap.add_argument("--limit", type=int, default=None, help="process at most this many items")
    ap.add_argument("--force", action="store_true", help="redo work already done")
    return ap


# --------------------------------------------------------------------------- io


def read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:  # pragma: no cover - dependency is pinned in requirements.txt
        print("PyYAML is not installed: pip install -r datasets/requirements.txt", file=sys.stderr)
        raise SystemExit(2)
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_schema() -> dict:
    return read_yaml(CONFIG_DIR / "schema.yaml")


def load_sources() -> dict:
    return read_yaml(CONFIG_DIR / "sources.yaml")


def load_splits_config() -> dict:
    return read_yaml(CONFIG_DIR / "splits.yaml")


def load_augment() -> dict:
    return read_yaml(CONFIG_DIR / "augment.yaml")


def class_ids() -> dict[str, int]:
    return {c["name"]: int(c["id"]) for c in load_schema()["classes"]}


def class_names() -> list[str]:
    return [c["name"] for c in sorted(load_schema()["classes"], key=lambda c: c["id"])]


def write_json(path: Path, payload: Any, dry_run: bool = False) -> None:
    if dry_run:
        log("INFO", "io", "dry-run: would write", path=str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_lines(path: Path, lines: Iterable[str], dry_run: bool = False) -> None:
    if dry_run:
        log("INFO", "io", "dry-run: would write", path=str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.rstrip("\n") + "\n")
    tmp.replace(path)


def merge_jsonl(path: Path, new_lines: list[str], key: str = "image") -> list[str]:
    """Union an index with what is already on disk, keyed by `key`.

    Resumability depends on this. A converter that skips already-done work produces a
    record list covering only the new work, and writing that straight out would silently
    truncate the index to the last increment."""
    merged: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                merged[json.loads(line).get(key)] = line
            except json.JSONDecodeError:
                continue
    for raw in new_lines:
        line = raw.strip()
        if not line:
            continue
        merged[json.loads(line).get(key)] = line
    return [merged[k] for k in sorted(merged, key=lambda v: str(v))]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_images(root: Path) -> Iterator[Path]:
    """Stream image paths. Never builds a list of the whole corpus in memory."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def count_images(root: Path) -> int:
    return sum(1 for _ in iter_images(root))


# --------------------------------------------------------------------------- labels


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Clip to the frame, then normalise. Returns None when the clipped box is degenerate."""
    if width <= 0 or height <= 0:
        return None
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    x1 = max(0.0, min(x1, width))
    x2 = max(0.0, min(x2, width))
    y1 = max(0.0, min(y1, height))
    y2 = max(0.0, min(y2, height))
    bw, bh = x2 - x1, y2 - y1
    if bw * bh < 16.0:  # schema.yaml box.min_area_px2
        return None
    cx = clamp01((x1 + bw / 2.0) / width)
    cy = clamp01((y1 + bh / 2.0) / height)
    nw = clamp01(bw / width)
    nh = clamp01(bh / height)
    if nw <= 0.0005 or nh <= 0.0005:
        return None
    return cx, cy, nw, nh


def format_label_line(class_id: int, cx: float, cy: float, w: float, h: float) -> str:
    return f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def parse_label_file(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Return (rows, errors). A malformed row is an error, never a silent drop."""
    rows: list[tuple[int, float, float, float, float]] = []
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return rows, [f"unreadable: {exc}"]
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {number}: expected 5 fields, found {len(parts)}")
            continue
        try:
            rows.append((int(parts[0]), *(float(p) for p in parts[1:])))
        except ValueError:
            errors.append(f"line {number}: non-numeric field")
    return rows, errors


# --------------------------------------------------------------------------- images


def imread_gray(path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def imread_any(path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: Path, image) -> bool:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        return False
    buffer.tofile(str(path))
    return True


def phash64(gray) -> int:
    """64-bit DCT perceptual hash. Used by 05_dedupe.py and by the LLVIP scene fallback."""
    import cv2
    import numpy as np

    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    flat = dct.flatten()
    median = np.median(flat[1:])
    bits = 0
    for index, value in enumerate(flat):
        if value > median:
            bits |= 1 << index
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- state


def state_path(processed: Path, script: str) -> Path:
    return processed / "_state" / f"{script}.json"


def load_state(processed: Path, script: str) -> dict:
    return read_json(state_path(processed, script), default={}) or {}


def save_state(processed: Path, script: str, state: dict, dry_run: bool = False) -> None:
    write_json(state_path(processed, script), state, dry_run=dry_run)


def run(main: Callable[[], int]) -> None:
    """Uniform exit discipline: an unhandled error is a non-zero exit, never a traceback swallowed."""
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)


def require_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")
