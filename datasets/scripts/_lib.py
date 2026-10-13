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
# Manifests and reports default to the repository. A Kaggle build points both into its output
# directory (TRUEWATCH_MANIFEST_DIR / TRUEWATCH_REPORT_DIR) so they travel with the dataset
# instead of dying with the notebook's repository clone.
MANIFEST_DIR = Path(os.environ.get("TRUEWATCH_MANIFEST_DIR") or DATASETS_ROOT / "manifests").resolve()
REPORT_DIR = Path(os.environ.get("TRUEWATCH_REPORT_DIR") or DATASETS_ROOT / "reports").resolve()
DEFAULT_RAW = REPO_ROOT / "var" / "datasets"
DEFAULT_PROCESSED = Path(os.environ.get("TRUEWATCH_PROCESSED") or DATASETS_ROOT / "processed")

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
    ap.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="worker processes for image-heavy steps (1 = run in-process)",
    )
    return ap


def default_workers() -> int:
    env = os.environ.get("TRUEWATCH_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(4, os.cpu_count() or 1))


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


# --------------------------------------------------------------------------- source roots
#
# A source root may be a downloaded directory under --raw, or a read-only directory some
# other system mounted (a Kaggle input). Nothing in this pipeline writes under a source root.


def roots_state_path(processed: Path) -> Path:
    return processed / "_state" / "source_roots.json"


def locate_root(base: Path, marker: list[str], max_depth: int = 6) -> Path | None:
    """First directory at or below `base` (breadth first, sorted) holding every marker entry."""
    base = Path(base)
    if not base.exists():
        return None
    frontier = [base]
    for _depth in range(max_depth + 1):
        next_frontier: list[Path] = []
        for directory in frontier:
            if all((directory / m).exists() for m in marker):
                return directory
            try:
                children = sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith("."))
            except OSError:
                continue
            next_frontier.extend(children)
        frontier = next_frontier
        if not frontier:
            break
    return None


def resolve_source_root(name: str, args, log: "Logger") -> Path | None:
    """--source, else the root 00_fetch.py recorded, else <raw>/<name>; then find the marker."""
    spec = load_sources()["sources"].get(name, {})
    marker = list(spec.get("marker") or [])
    processed = Path(args.processed).resolve()
    candidates: list[tuple[str, Path]] = []
    if getattr(args, "source", None):
        candidates.append(("--source", Path(args.source)))
    recorded = (read_json(roots_state_path(processed), default={}) or {}).get(name)
    if recorded:
        candidates.append(("00_fetch record", Path(recorded)))
    candidates.append(("--raw", Path(args.raw) / name))
    for origin, candidate in candidates:
        root = locate_root(candidate, marker) if marker else (candidate if candidate.exists() else None)
        if root is not None:
            log.info("source root", source=name, root=str(root), origin=origin, writable=os.access(root, os.W_OK))
            return root.resolve()
        log.warn("candidate root has no marker", source=name, candidate=str(candidate), marker=",".join(marker))
    return None


def stage_link(target: Path, link: Path, counters: "Counters", dry_run: bool = False) -> bool:
    """Give a read-only source file a collision-proof name without copying it.

    IDD and KAIST reuse frame names across drives and videos (IDD: 2,694 stems appear in more
    than one drive). Every later step, and training/, names files after Path(image).stem, so the
    record's image path must itself carry a unique stem. A symlink costs no disk; when the
    filesystem refuses one the file is copied and the copy is counted."""
    if dry_run:
        return True
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        try:
            if link.resolve() == target.resolve():
                counters.bump("stage.already_linked")
                return True
        except OSError:
            pass
        link.unlink()
    try:
        link.symlink_to(target)
        counters.bump("stage.linked")
        return True
    except OSError:
        import shutil

        try:
            shutil.copy2(target, link)
            counters.bump("stage.copied_symlink_refused")
            return True
        except OSError:
            counters.bump("stage.failed")
            return False


def assert_unique(keys: Iterable[str], what: str) -> None:
    """Fail loudly on a duplicate. A duplicate final name means one file overwrote another."""
    seen: dict[str, int] = {}
    for key in keys:
        seen[key] = seen.get(key, 0) + 1
    dupes = sorted(k for k, n in seen.items() if n > 1)
    if dupes:
        raise AssertionError(f"{len(dupes)} duplicate {what}: {dupes[:5]}")


def final_stem(record: dict) -> str:
    """The one naming rule for a placed file: {source}_{stem of record['image']}_{modality}.

    training/scripts (eval_hardset.final_stem_of, _common.MetaResolver) re-derive final names
    with exactly this rule, so it must not change. Uniqueness is guaranteed upstream by giving
    every record an image path whose stem is unique within its source (see stage_link)."""
    return f"{record.get('source', 'x')}_{Path(record['image']).stem}_{record.get('modality', 'visible')}"


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- infrared


def ir_conversion_params() -> tuple[float, int]:
    conv = load_augment()["infrared_conversion"]
    if conv.get("false_colour"):
        raise SystemExit("augment.yaml enables false colour; DATASET_SPEC.md section 3.4 forbids it")
    return float(conv.get("clahe_clip", 2.0)), int(conv.get("clahe_tile_grid", 8))


def ir_to_three_channel(gray, clip: float, grid: int):
    """Section 3.4: CLAHE, then B = G = R. No false colour, ever."""
    import cv2
    import numpy as np

    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    return np.repeat(clahe.apply(gray)[:, :, None], 3, axis=2)


def load_for_output(record: dict, ir_params: tuple[float, int] | None = None):
    """The pixels a record contributes to the dataset. LWIR is converted here, at write time,
    so no 3-channel intermediate is ever stored (at Kaggle scale that intermediate alone would
    exceed the 20 GB output limit). Tiles are cut from this same function's output, so a tile
    is already converted and is read as is."""
    path = Path(record["image"])
    if record.get("modality") == "lwir" and not record.get("tiled"):
        gray = imread_gray(path)
        if gray is None:
            return None
        clip, grid = ir_params or ir_conversion_params()
        return ir_to_three_channel(gray, clip, grid)
    return imread_any(path)


def encode_jpeg(image, quality: int) -> bytes | None:
    import cv2

    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buffer.tobytes() if ok else None


def cap_long_side(image, cap: int):
    import cv2

    height, width = image.shape[:2]
    longest = max(height, width)
    if cap and longest > cap:
        scale = cap / float(longest)
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA), True
    return image, False


def detector_px(h_norm: float, img_w: int, img_h: int, imgsz: int = 640) -> float:
    """Normalised box height -> pixels at the detector input (long side letterboxed to imgsz).

    MEASUREMENTS.md section 3's 19/14/9 px cliff is measured at the detector input, so every
    small-object count in this pipeline is expressed on that scale."""
    longest = max(1, max(int(img_w or 0), int(img_h or 0)))
    return float(h_norm) * float(img_h or 0) * imgsz / longest


# --------------------------------------------------------------------------- composition


def load_composition() -> dict:
    return load_splits_config().get("composition", {})


def tile_windows(width: int, height: int, tile: int, overlap: float, strip_top: float, strip_bottom: float):
    """Tile origins over the far-field strip. Empty when the frame is smaller than a tile."""
    if width < tile or height < tile:
        return []
    y1 = int(height * strip_top)
    y2 = int(height * strip_bottom)
    if y2 - y1 < tile:
        y2 = min(height, y1 + tile)  # strip thinner than a tile: one row anchored at the strip top
    if y2 - y1 < tile:
        return []

    def origins(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        step = max(1, int(round(tile * (1.0 - overlap))))
        out = list(range(0, max(1, extent - tile + 1), step))
        if out[-1] != extent - tile:
            out.append(extent - tile)
        return out

    return [(ox, y1 + oy) for oy in origins(y2 - y1) for ox in origins(width)]


def clip_box_to_tile(cx, cy, w, h, width, height, ox, oy, tile, min_retained):
    x1, y1 = (cx - w / 2.0) * width, (cy - h / 2.0) * height
    x2, y2 = (cx + w / 2.0) * width, (cy + h / 2.0) * height
    area = max(1e-9, (x2 - x1) * (y2 - y1))
    ix1, iy1 = max(x1, ox), max(y1, oy)
    ix2, iy2 = min(x2, ox + tile), min(y2, oy + tile)
    if ix2 <= ix1 or iy2 <= iy1:
        return None, False
    if (ix2 - ix1) * (iy2 - iy1) / area < min_retained:
        return None, True  # touches the tile but too little survives
    return (((ix1 + ix2) / 2.0 - ox) / tile, ((iy1 + iy2) / 2.0 - oy) / tile, (ix2 - ix1) / tile, (iy2 - iy1) / tile), True


def plan_object_tiles(train_records: list[dict], seed: int) -> list[dict]:
    """Deterministic far-field tile plan, shared by 07 (which sizes the negative pool against
    it) and 08 (which cuts it). Same inputs and seed -> the same list, element for element.

    Tiles holding a small object (<= small_object_px at native resolution) are taken first,
    because the small-object tail is what tiling exists for; the budget is split by modality
    at composition.tiles.lwir_share so tiles do not push the visible share out of band."""
    import random

    cfg = load_composition().get("tiles", {})
    tile = int(cfg.get("size", 640))
    overlap = float(cfg.get("overlap", 0.20))
    top, bottom = float(cfg.get("strip_top", 0.0)), float(cfg.get("strip_bottom", 0.40))
    min_retained = float(cfg.get("min_box_area_retained", 0.30))
    budget = int(cfg.get("max_tiles", 6000))
    lwir_share = float(cfg.get("lwir_share", 0.35))
    small_px = float(cfg.get("small_object_px", 27))

    pools: dict[str, list[dict]] = {"visible": [], "lwir": []}
    for record in sorted(train_records, key=lambda r: r["image"]):
        if record.get("objects", 0) <= 0 or record.get("is_negative") or record.get("tiled"):
            continue
        width, height = int(record.get("img_w") or 0), int(record.get("img_h") or 0)
        windows = tile_windows(width, height, tile, overlap, top, bottom)
        if not windows:
            continue
        rows, _errors = parse_label_file(Path(record["label"]))
        for ox, oy in windows:
            kept = []
            small = False
            for class_id, cx, cy, w, h in rows:
                clipped, _touch = clip_box_to_tile(cx, cy, w, h, width, height, ox, oy, tile, min_retained)
                if clipped is None:
                    continue
                kept.append((class_id, *clipped))
                if h * height <= small_px:
                    small = True
            if not kept:
                continue
            pools.setdefault(record.get("modality", "visible"), []).append(
                {"parent": record["image"], "ox": ox, "oy": oy, "rows": kept, "small": small,
                 "modality": record.get("modality", "visible")}
            )

    rng = random.Random(seed)
    ordered: dict[str, list[dict]] = {}
    for modality, pool in sorted(pools.items()):
        smalls = [c for c in pool if c["small"]]
        others = [c for c in pool if not c["small"]]
        rng.shuffle(smalls)
        rng.shuffle(others)
        ordered[modality] = smalls + others

    want_ir = int(round(budget * lwir_share))
    take_ir = ordered.get("lwir", [])[:want_ir]
    take_vis = ordered.get("visible", [])[: budget - len(take_ir)]
    if len(take_vis) + len(take_ir) < budget:  # visible short: top up with more infrared
        take_ir = ordered.get("lwir", [])[: budget - len(take_vis)]
    chosen = take_vis + take_ir
    chosen.sort(key=lambda c: (c["parent"], c["oy"], c["ox"]))
    return chosen


# --------------------------------------------------------------------------- cart gate


CART_SHEET_FIELDS = ["uid", "image", "xmin", "ymin", "xmax", "ymax", "verdict"]


def cart_sheet_path(processed: Path) -> Path:
    return processed / "cart_review" / "vehicle_fallback.csv"


def read_cart_sheet(path: Path) -> list[dict]:
    import csv

    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def cart_gate_status(processed: Path) -> dict:
    """DATASET_SPEC.md section 1.5, evaluated. Class 4 ships only at >= min_verified_instances
    rows marked `cart` by a human; otherwise it is withdrawn and id 4 stays reserved."""
    gate = load_schema().get("cart_gate", {})
    threshold = int(gate.get("min_verified_instances", 300))
    rows = read_cart_sheet(cart_sheet_path(processed))
    verdicts = [str(r.get("verdict", "")).strip().lower() for r in rows]
    verified = sum(1 for v in verdicts if v == "cart")
    rejected = sum(1 for v in verdicts if v in ("not_cart", "notcart", "no"))
    unreviewed = len(rows) - verified - rejected
    shipped = verified >= threshold
    return {
        "candidates": len(rows),
        "verified_cart": verified,
        "rejected_not_cart": rejected,
        "unreviewed": unreviewed,
        "threshold": threshold,
        "shipped": shipped,
        "decision": "ship_class_4" if shipped else "withdraw_class_4",
        "reason": (
            f"{verified} hand-verified instances >= {threshold}"
            if shipped
            else f"{verified} hand-verified instances < {threshold} "
            f"({unreviewed} of {len(rows)} candidates never reviewed); class 4 withdrawn, id 4 reserved"
        ),
    }


# --------------------------------------------------------------------------- parallel


def parallel_map(func: Callable, items: list, workers: int, chunksize: int = 16) -> Iterator:
    """Ordered map over a process pool; in-process when workers <= 1 or the list is tiny.

    The worker function must live at module level in an importable module (this one), because
    macOS starts workers with spawn and re-imports it."""
    if workers <= 1 or len(items) < 2 * chunksize:
        for item in items:
            yield func(item)
        return
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(processes=workers) as pool:
        for result in pool.imap(func, items, chunksize=chunksize):
            yield result


def phash_worker(path: str):
    gray = imread_gray(Path(path))
    return None if gray is None else phash64(gray)


def ir_probe_worker(path: str):
    """04: decode one LWIR frame, report bit depth and intensity spread, prove B = G = R."""
    import cv2
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return {"ok": False, "why": "empty"}
    raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return {"ok": False, "why": "undecodable"}
    colour_input = raw.ndim == 3
    gray = imread_gray(Path(path))
    clip, grid = ir_conversion_params()
    three = ir_to_three_channel(gray, clip, grid)
    replicated = bool(np.array_equal(three[..., 0], three[..., 1]) and np.array_equal(three[..., 1], three[..., 2]))
    return {
        "ok": replicated,
        "why": "" if replicated else "channels differ after conversion",
        "bit16": bool(raw.dtype != np.uint8),
        "colour_input": colour_input,
        "ir_std": round(float(gray.std()), 3),
    }


def place_worker(job: dict) -> dict:
    """09: write one final image (JPEG) and its label. Returns sizes for the projection."""
    record = job["record"]
    target = Path(job["target_image"])
    label_target = Path(job["target_label"])
    result = {"ok": False, "bytes": 0, "downscaled": False, "why": ""}
    if job.get("skip_image") and target.exists():
        result.update(ok=True, bytes=target.stat().st_size, why="already_placed")
    else:
        image = load_for_output(record, tuple(job["ir_params"]))
        if image is None:
            result["why"] = "unreadable"
            return result
        downscaled = False
        if not record.get("tiled"):
            image, downscaled = cap_long_side(image, int(job["long_side_cap"]))
        payload = encode_jpeg(image, int(job["quality"]))
        if payload is None:
            result["why"] = "encode_failed"
            return result
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.replace(target)
        result.update(ok=True, bytes=len(payload), downscaled=downscaled)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.write_text("".join(line + "\n" for line in job["lines"]), encoding="utf-8")
    return result


def tile_worker(job: dict) -> dict:
    """08: cut every planned tile of one parent frame and write them as JPEG."""
    record = job["record"]
    image = load_for_output(record, tuple(job["ir_params"]))
    if image is None:
        return {"ok": False, "written": 0, "why": "unreadable"}
    tile = int(job["tile"])
    written = 0
    wrong = 0
    for spec in job["tiles"]:
        ox, oy = int(spec["ox"]), int(spec["oy"])
        crop = image[oy : oy + tile, ox : ox + tile]
        if crop.shape[0] != tile or crop.shape[1] != tile:
            wrong += 1
            continue
        payload = encode_jpeg(crop, int(job["quality"]))
        if payload is None:
            wrong += 1
            continue
        out = Path(spec["image"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        Path(spec["label"]).parent.mkdir(parents=True, exist_ok=True)
        Path(spec["label"]).write_text("".join(line + "\n" for line in spec["lines"]), encoding="utf-8")
        written += 1
    return {"ok": wrong == 0, "written": written, "wrong": wrong, "why": "" if wrong == 0 else "wrong_shape"}
