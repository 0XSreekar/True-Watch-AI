"""Shared pieces of the tiled-inference evaluation (tiled_infer.py runs the network, tiled_eval.py scores).

The window layout and the merge are NOT implemented here: they are `edge/pipeline/tiling.py`, loaded by
file path, so the evaluator measures exactly the code the edge pipeline runs. This module imports no
Ultralytics code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent


def load_tiling():
    """edge/pipeline/tiling.py (or a copy beside this file, as shipped to Kaggle) as a module."""
    for candidate in (REPO_ROOT / "edge" / "pipeline" / "tiling.py", SCRIPTS_DIR / "tiling.py"):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("truewatch_tiling", candidate)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["truewatch_tiling"] = mod   # dataclasses resolve annotations through sys.modules
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("edge/pipeline/tiling.py not found")


T = load_tiling()

# Window layouts. Everything except the layout (score floors, merge IoU, truncation) is tuned later.
LAYOUTS = {
    "g2x2": T.TilingConfig(enabled=True, band=(0.0, 1.0), grid=(2, 2), overlap=0.2),
    "g3x2": T.TilingConfig(enabled=True, band=(0.0, 1.0), grid=(3, 2), overlap=0.2),
    "n640": T.TilingConfig(enabled=True, band=(0.0, 1.0), tile=640, overlap=0.0),   # original resolution on 1080p
    "strip": T.TilingConfig(enabled=True),                                         # the edge default: far-field band
}


def all_windows(w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Union of every layout's windows for a w x h frame, deduplicated, stable order."""
    seen, out = set(), []
    for cfg in LAYOUTS.values():
        for win in T.tile_windows(w, h, cfg):
            if win not in seen:
                seen.add(win)
                out.append(win)
    return out


def layout_windows(name: str, w: int, h: int) -> list[tuple[int, int, int, int]]:
    return T.tile_windows(w, h, LAYOUTS[name])


def variant(name: str, **overrides):
    return replace(LAYOUTS[name], **overrides)


def half_of(cluster: str, seed: str = "truewatch-tiling-v1") -> int:
    """0 = tuning half, 1 = report half. Hash of the correlated-frame cluster, so no sequence straddles the halves
    and each source is split about evenly (the split is within source, not between sources)."""
    return int(hashlib.sha256(f"{seed}:{cluster}".encode()).hexdigest(), 16) & 1


def passes_for(image_passes, w: int, h: int, layout: str | None):
    """[(window or None, boxes, conf, cls)] for one image under `layout` (None = full frame only).

    image_passes: dict window-or-None -> (boxes, conf, cls) as cached by tiled_infer.py.
    """
    out = [(None, *image_passes[None])]
    if layout is not None:
        for win in layout_windows(layout, w, h):
            if win in image_passes:
                out.append((win, *image_passes[win]))
    return out


def merge_image(image_passes, w: int, h: int, layout: str | None, cfg, *, full_conf: float, iou: float, max_det: int):
    """Merge one image's cached passes with the edge's merge_passes. Returns boxes, conf, cls."""
    boxes, conf, cls, _ = T.merge_passes(passes_for(image_passes, w, h, layout), w, h, cfg,
                                         full_conf=full_conf, iou=iou, max_det=max_det)
    return boxes.astype(np.float32), conf.astype(np.float32), cls.astype(np.int64)
