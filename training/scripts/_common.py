"""Shared helpers for training/scripts.

Ultralytics is AGPL-3.0 and lives in training/ only (see training/README.md). This module does
not import it at all, so the pure-Python parts of the toolchain (metrics, verdicts, the CPU
benchmark) run on a machine that has only numpy and onnxruntime.

Three ideas in here carry the reporting rules the Phase 2 brief makes non-negotiable:

  * SLICES. Every metric belongs to exactly one modality slice, "day" (the visible camera) or
    "ir" (LWIR replicated to 3 channels). There is no function anywhere that produces a metric
    over both. See `assert_single_modality` in _metrics.py for the structural guard.
  * SIZE_BUCKETS_PX. The seven person heights of docs/MEASUREMENTS.md section 3, verbatim.
    tests/test_common.py parses that document and fails if these constants drift from it.
  * Atomic writes. A Kaggle session can be killed at any instant; every JSON this toolchain
    writes goes to a temp file first and is renamed into place.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = SCRIPTS_DIR.parent
REPO_ROOT = TRAINING_ROOT.parent
CONFIG_DIR = TRAINING_ROOT / "configs"
RESULTS_DIR = TRAINING_ROOT / "results"
AUGMENT_YAML = REPO_ROOT / "datasets" / "config" / "augment.yaml"
SPLITS_YAML = REPO_ROOT / "datasets" / "config" / "splits.yaml"
SCHEMA_YAML = REPO_ROOT / "datasets" / "config" / "schema.yaml"
HARD_SET_MANIFEST = REPO_ROOT / "datasets" / "manifests" / "hard_set.txt"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# --------------------------------------------------------------------------------------------
# Object-size buckets. docs/MEASUREMENTS.md section 3, "Person height" column, in print order.
# --------------------------------------------------------------------------------------------

SIZE_BUCKETS_PX: tuple[int, ...] = (77, 54, 38, 27, 19, 14, 9)

# MEASUREMENTS.md section 3 recall against native detections (COCO-pretrained, 23 KAIST frames,
# conf 0.35). Carried only so METRICS.md can print the pre-fine-tuning cliff beside the measured
# one. It is a different quantity (recall against native detections, not against ground truth).
MEASURED_PRETRAINED_RECALL_VS_NATIVE: dict[int, float] = {
    77: 1.00, 54: 1.00, 38: 1.00, 27: 1.02, 19: 0.90, 14: 0.80, 9: 0.49,
}


def bucket_upper_edges(levels: Sequence[int] = SIZE_BUCKETS_PX) -> tuple[float, ...]:
    """Lower edge of each bucket, largest first: the geometric mean of adjacent levels.

    The seven levels are a geometric ladder (each about 0.7x the last), so an object belongs to
    the level it is nearest to on a log scale. The tallest bucket is open above, the smallest
    open below. Returned as (edge between 77 and 54, ..., edge between 14 and 9).
    """
    return tuple(math.sqrt(a * b) for a, b in zip(levels[:-1], levels[1:]))


_EDGES = bucket_upper_edges()


def size_bucket(height_px: float) -> int:
    """Map an object height in pixels to one of SIZE_BUCKETS_PX (the level, not an index)."""
    for level, edge in zip(SIZE_BUCKETS_PX, _EDGES):
        if height_px >= edge:
            return level
    return SIZE_BUCKETS_PX[-1]


def bucket_range_label(level: int) -> str:
    """Human-readable range for a bucket, e.g. '45.3-64.5 px' or '>= 64.5 px'."""
    i = SIZE_BUCKETS_PX.index(level)
    if i == 0:
        return f">= {_EDGES[0]:.1f} px"
    if i == len(SIZE_BUCKETS_PX) - 1:
        return f"< {_EDGES[-1]:.1f} px"
    return f"{_EDGES[i]:.1f}-{_EDGES[i - 1]:.1f} px"


def object_height_px(h_norm: float, img_h: int, img_w: int, imgsz: int, basis: str = "input") -> float:
    """Height in pixels of a box whose normalised height is `h_norm`.

    basis="input"  : pixels the detector actually sees, after the image is scaled to fit
                     `imgsz` (letterbox). This is the quantity MEASUREMENTS.md section 3
                     varied, because it fed frames to the detector at 640. Default.
    basis="stored" : pixels in the stored file. The Phase 1 hard-set rule and gate G12 use this.
    """
    stored = h_norm * img_h
    if basis == "stored":
        return stored
    if basis != "input":
        raise ValueError(f"basis must be 'input' or 'stored', got {basis!r}")
    scale = min(imgsz / float(img_h), imgsz / float(img_w))
    return stored * scale


# --------------------------------------------------------------------------------------------
# Modality slices. "day" is the visible-spectrum camera and "ir" is LWIR, the same convention
# datasets/ uses in gate G8 and docs/DATASET_SPEC.md section 2.4.
# --------------------------------------------------------------------------------------------

SLICE_DAY = "day"
SLICE_IR = "ir"
MODALITY_SLICES: tuple[str, ...] = (SLICE_DAY, SLICE_IR)

_MODALITY_TO_SLICE = {"visible": SLICE_DAY, "lwir": SLICE_IR}
_MODALITY_RE = re.compile(r"_(visible|lwir)$")
_KAIST_SET_RE = re.compile(r"(set\d{2})")
_KAIST_VIDEO_RE = re.compile(r"(V\d{3})")

LIGHTING_DAYLIGHT = "daylight"
LIGHTING_NIGHT = "night"
LIGHTING_UNRESOLVED = "unresolved"

# docs/DATASET_SPEC.md section 3.4: "IDD is daylight-only. KAIST is roughly half night. LLVIP is
# predominantly night." IDD and LLVIP lighting is therefore an assumption taken from the spec, not
# something measured per frame; KAIST lighting comes from datasets/config/splits.yaml, whose
# declaration 02_convert_kaist.py verifies against the frames at conversion time (OQ-1).
SOURCE_VISIBLE_LIGHTING = {"idd": LIGHTING_DAYLIGHT, "llvip": LIGHTING_NIGHT}
# Lighting as written into the split index by the converters. 'unknown' is deliberately absent:
# it stays unresolved rather than being guessed from the source.
_INDEX_LIGHTING = {"day": LIGHTING_DAYLIGHT, "night": LIGHTING_NIGHT}


def modality_of_stem(stem: str) -> str | None:
    """'visible' or 'lwir' from a final dataset filename stem, or None if it carries neither.

    09_build_yolo_ds.py names every file f"{source}_{raw_stem}_{modality}" (tiles included), so
    the modality is always the last token. None is never silently mapped to a slice.
    """
    match = _MODALITY_RE.search(stem)
    return match.group(1) if match else None


def slice_of_modality(modality: str) -> str:
    return _MODALITY_TO_SLICE[modality]


def kaist_lighting_table() -> dict[str, str]:
    """set -> 'daylight' | 'night' from datasets/config/splits.yaml. Empty if unavailable."""
    try:
        import yaml

        cfg = yaml.safe_load(SPLITS_YAML.read_text(encoding="utf-8"))
        table = {}
        for name, entry in cfg["kaist"]["sets"].items():
            table[name] = LIGHTING_DAYLIGHT if entry["lighting"] == "day" else LIGHTING_NIGHT
        return table
    except Exception:
        return {}


def llvip_prefix_length() -> int:
    try:
        import yaml

        cfg = yaml.safe_load(SPLITS_YAML.read_text(encoding="utf-8"))
        return int(cfg["llvip"]["scene_key"]["prefix_length"])
    except Exception:
        return 2


@dataclass(frozen=True)
class ImageMeta:
    name: str        # final file stem
    source: str      # idd | flir | kaist | llvip | unknown
    modality: str    # visible | lwir
    slice: str       # day | ir
    lighting: str    # daylight | night | unresolved   (visible frames only; lwir is 'n/a')
    cluster: str     # correlated-frame group, for the bootstrap


class MetaResolver:
    """Resolve modality, lighting and bootstrap cluster for a final dataset image.

    Modality always comes from the filename (guaranteed by 09_build_yolo_ds.py). Lighting and
    cluster use the split index (`processed/index/split.jsonl`) when one is supplied, because
    the raw stem that 09 keeps may not encode the KAIST set. Anything that cannot be resolved is
    labelled 'unresolved' and counted; it is never defaulted into the daylight slice.
    """

    def __init__(self, index_path: str | os.PathLike | None = None) -> None:
        self._kaist_lighting = kaist_lighting_table()
        self._llvip_prefix = llvip_prefix_length()
        self._index: dict[str, dict] = {}
        self.index_collisions = 0
        if index_path:
            self._load_index(Path(index_path))

    def _load_index(self, path: Path) -> None:
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("source") or not rec.get("modality") or not rec.get("image"):
                    continue
                # 09_build_yolo_ds.py records the placed file; older indexes only carry the
                # source image, so fall back to the rule 09 uses to name a placed file.
                if rec.get("final_image"):
                    stem = Path(rec["final_image"]).stem
                else:
                    stem = f"{rec['source']}_{Path(rec['image']).stem}_{rec['modality']}"
                if stem in self._index:
                    self.index_collisions += 1
                self._index[stem] = rec

    @property
    def has_index(self) -> bool:
        return bool(self._index)

    def resolve(self, stem: str) -> ImageMeta | None:
        modality = modality_of_stem(stem)
        if modality is None:
            return None
        rec = self._index.get(stem, {})
        source = rec.get("source") or stem.split("_", 1)[0]
        if source not in ("idd", "flir", "kaist", "llvip"):
            source = "unknown"

        if modality == "lwir":
            lighting = "n/a"
        elif rec.get("lighting") in _INDEX_LIGHTING:
            # Measured per frame at conversion time (FLIR has no per-source lighting).
            lighting = _INDEX_LIGHTING[rec["lighting"]]
        elif source in SOURCE_VISIBLE_LIGHTING:
            lighting = SOURCE_VISIBLE_LIGHTING[source]
        elif source == "kaist":
            set_name = rec.get("set")
            if not set_name:
                match = _KAIST_SET_RE.search(stem)
                set_name = match.group(1) if match else None
            lighting = self._kaist_lighting.get(set_name or "", LIGHTING_UNRESOLVED)
        else:
            lighting = LIGHTING_UNRESOLVED

        cluster = rec.get("sequence_key")
        if not cluster:
            if source == "kaist":
                s, v = _KAIST_SET_RE.search(stem), _KAIST_VIDEO_RE.search(stem)
                cluster = f"kaist/{s.group(1)}/{v.group(1)}" if s and v else None
            elif source == "llvip":
                digits = re.search(r"llvip_(\w+?)_(?:visible|lwir)$", stem)
                cluster = f"llvip/{digits.group(1)[: self._llvip_prefix]}" if digits else None
        return ImageMeta(
            name=stem,
            source=source,
            modality=modality,
            slice=slice_of_modality(modality),
            lighting=lighting,
            cluster=cluster or stem,  # no group known: each frame is its own cluster
        )


# --------------------------------------------------------------------------------------------
# Dataset access
# --------------------------------------------------------------------------------------------


def resolve_data_root(explicit: str | os.PathLike | None = None) -> Path:
    """Locate the Phase 1 dataset root (the directory holding images/, labels/, data.yaml).

    Order: --data-root, $TRUEWATCH_DATA_ROOT, a Kaggle input mount, datasets/processed/yolo.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("TRUEWATCH_DATA_ROOT", "").strip()
    if env:
        candidates.append(Path(env))
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for pattern in ("*/data.yaml", "*/*/data.yaml", "*/*/*/data.yaml"):
            candidates.extend(sorted(p.parent for p in kaggle.glob(pattern)))
    candidates.append(REPO_ROOT / "datasets" / "processed" / "yolo")

    for root in candidates:
        if (root / "images" / "val").is_dir() and (root / "labels" / "val").is_dir():
            return root.resolve()
    tried = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(
        "could not find the dataset root (needs images/val and labels/val). Tried:\n  "
        f"{tried}\nPass --data-root or set TRUEWATCH_DATA_ROOT."
    )


def list_split_images(root: Path, split: str, limit: int | None = None) -> list[Path]:
    folder = root / "images" / split
    if not folder.is_dir():
        raise SystemExit(f"no images/{split} under {root}")
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return images[:limit] if limit else images


def label_path_for(image: Path) -> Path:
    """images/<split>/x.jpg -> labels/<split>/x.txt, the Ultralytics convention."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        raise ValueError(f"{image} is not under an 'images' directory")
    return Path(*parts).with_suffix(".txt")


def read_yolo_label(path: Path):
    """(n, 5) float64 array [class, cx, cy, w, h]; empty (0, 5) for a background image."""
    import numpy as np

    if not path.exists():
        return np.zeros((0, 5), dtype=np.float64)
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) == 5:
            rows.append([float(p) for p in parts])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def load_class_names(root: Path | None = None) -> list[str]:
    """Class names in id order: the dataset's own data.yaml, else datasets/config/schema.yaml."""
    import yaml

    if root is not None and (root / "data.yaml").exists():
        data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
        names = data["names"]
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    schema = yaml.safe_load(SCHEMA_YAML.read_text(encoding="utf-8"))
    return [c["name"] for c in sorted(schema["classes"], key=lambda c: c["id"])]


# --------------------------------------------------------------------------------------------
# Files, hashes, host description
# --------------------------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _cpu_brand() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _cgroup_cpu_limit() -> float | None:
    """CPUs granted by a container quota (HF Spaces, Kaggle, Docker), or None if unlimited.

    os.cpu_count() reports the host's cores inside a container, which would make a 2-vCPU Space
    look like a 32-core machine in a benchmark record. This reads the actual quota.
    """
    try:
        v2 = Path("/sys/fs/cgroup/cpu.max")
        if v2.exists():
            quota, period = v2.read_text().split()
            return None if quota == "max" else int(quota) / int(period)
        q = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        p = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q.exists() and p.exists() and int(q.read_text()) > 0:
            return int(q.read_text()) / int(p.read_text())
    except Exception:
        pass
    return None


def host_info() -> dict:
    """Describe the machine a number was measured on. Every measured figure carries this."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_brand(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_usable": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "cgroup_cpu_limit": _cgroup_cpu_limit(),
        "python": platform.python_version(),
    }
    for module in ("numpy", "onnxruntime", "torch"):
        try:
            info[module] = __import__(module).__version__
        except Exception:
            info[module] = None
    return info


def fmt(value, digits: int = 3, missing: str = "n/a") -> str:
    """Format a metric for a table; None and NaN become the word for 'no data', never 0."""
    if value is None:
        return missing
    try:
        if isinstance(value, float) and math.isnan(value):
            return missing
    except TypeError:
        pass
    return f"{value:.{digits}f}"


def chunked(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
