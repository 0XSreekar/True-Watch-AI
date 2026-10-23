"""Tiled inference: where the windows go, and how the per-window detections are merged.

Pure numpy, no OpenCV and no onnxruntime, so the SAME code runs on the edge (pipeline/appearance.py)
and in the offline evaluator (training/scripts/tiled_eval.py). What is measured is what ships.

Why: the v2 detector sees a 1080p frame at 640 px, so a person 30 px tall in the frame is 10 px at the
input and is mostly missed (training/results/DAY_GAP_AUDIT.md). Running the detector on overlapping
windows of the original frame (upscaling small windows) keeps those people large enough to find.

Merge rules (`merge_passes`)
----------------------------
A "pass" is one forward run: the full frame (window None) or one window. Boxes arrive in ORIGINAL frame
pixels. Then, in order:
  1. score threshold per scale: full-frame boxes must reach `full_conf`, window boxes `tile_conf`;
  2. truncation: a window box that touches an INTERIOR window edge (an edge that is not the frame
     border) may be a person cut in half by the window. It is dropped when an untruncated box of the
     same class covers at least `trunc_ioa` of it (intersection over the truncated box's own area);
     a truncated box with no such partner is kept;
  3. class-aware NMS at `merge_iou` across all passes, highest score first, capped at `max_det`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TilingConfig:
    """Tiling for one camera. Band fractions are of the original frame height."""

    enabled: bool = False
    band: tuple[float, float] = (0.0, 0.55)    # (top, bottom) of the band that is tiled; (0, 1) = whole frame
    tile: int = 640                            # window edge in ORIGINAL-frame pixels (square windows)
    overlap: float = 0.2
    grid: tuple[int, int] | None = None        # (cols, rows) over the band; overrides `tile`, windows are not square
    tile_conf: float | None = None             # score floor for window boxes; None = same as the full-frame floor
    merge_iou: float | None = None             # cross-window NMS IoU; None = the detector's own NMS IoU
    trunc_margin: float = 4.0                  # px: a box this close to an interior window edge is "truncated"
    trunc_ioa: float = 0.5                     # drop a truncated box covered this much by an untruncated one; 0 = off


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS; indices kept, highest score first."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ious = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[ious <= iou]
    return np.asarray(keep, dtype=np.int64)


def class_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float, max_det: int) -> np.ndarray:
    """Class-aware NMS by the coordinate-offset trick: boxes of different classes never overlap."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    offset = classes[:, None].astype(np.float32) * (float(boxes.max()) + 1.0)
    return nms(boxes + offset, scores, iou)[:max_det]


def tile_windows(width: int, height: int, cfg: TilingConfig) -> list[tuple[int, int, int, int]]:
    """Overlapping windows (x1, y1, x2, y2) covering the band. A window that is the whole frame is dropped."""
    top = int(round(cfg.band[0] * height))
    bottom = int(round(cfg.band[1] * height))
    bh = max(bottom - top, 1)
    if cfg.grid is not None:
        cols, rows = max(int(cfg.grid[0]), 1), max(int(cfg.grid[1]), 1)
        tw = int(round(width / (cols - (cols - 1) * cfg.overlap)))
        th = int(round(bh / (rows - (rows - 1) * cfg.overlap)))
        tw, th = min(tw, width), min(th, bh)
        xs = [int(round(i * (width - tw) / max(cols - 1, 1))) for i in range(cols)]
        ys = [top + int(round(j * (bh - th) / max(rows - 1, 1))) for j in range(rows)]
    else:
        tw = th = int(min(cfg.tile, width, bh))
        step_x = max(1, int(tw * (1.0 - cfg.overlap)))
        step_y = max(1, int(th * (1.0 - cfg.overlap)))
        xs = list(range(0, max(width - tw, 0) + 1, step_x))
        if xs[-1] + tw < width:
            xs.append(width - tw)
        ys = list(range(top, max(bottom - th, top) + 1, step_y))
        if ys[-1] + th < bottom:
            ys.append(bottom - th)
    wins = [(x, y, x + tw, y + th) for y in ys for x in xs]
    # a window that is (nearly) the whole frame repeats the full-frame pass and only adds duplicates
    return [w for w in wins if (w[2] - w[0]) * (w[3] - w[1]) < 0.9 * width * height]


def _ioa(boxes: np.ndarray, others: np.ndarray) -> np.ndarray:
    """(len(boxes), len(others)) intersection over the area of each of `boxes`."""
    ix1 = np.maximum(boxes[:, None, 0], others[None, :, 0]); iy1 = np.maximum(boxes[:, None, 1], others[None, :, 1])
    ix2 = np.minimum(boxes[:, None, 2], others[None, :, 2]); iy2 = np.minimum(boxes[:, None, 3], others[None, :, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    return inter / area[:, None]


def truncated_by_window(boxes: np.ndarray, window: tuple[int, int, int, int], width: int, height: int,
                        margin: float) -> np.ndarray:
    """Which boxes touch an interior edge of `window` (an edge that is not the frame border)."""
    wx1, wy1, wx2, wy2 = window
    t = np.zeros(len(boxes), dtype=bool)
    if wx1 > 0:
        t |= boxes[:, 0] <= wx1 + margin
    if wx2 < width:
        t |= boxes[:, 2] >= wx2 - margin
    if wy1 > 0:
        t |= boxes[:, 1] <= wy1 + margin
    if wy2 < height:
        t |= boxes[:, 3] >= wy2 - margin
    return t


def merge_passes(passes, width: int, height: int, cfg: TilingConfig, *, full_conf: float, iou: float,
                 max_det: int = 300):
    """Merge per-pass detections into one set.

    passes: iterable of (window or None, boxes (k,4) xyxy in frame px, scores (k,), classes (k,)).
    Returns (boxes, scores, classes, from_window) of the kept detections, highest score first.
    """
    tile_conf = full_conf if cfg.tile_conf is None else cfg.tile_conf
    merge_iou = iou if cfg.merge_iou is None else cfg.merge_iou
    bs, ss, cs, ts, tr = [], [], [], [], []
    for window, boxes, scores, classes in passes:
        if len(boxes) == 0:
            continue
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores)
        classes = np.asarray(classes)
        keep = scores >= (full_conf if window is None else tile_conf)
        if not keep.any():
            continue
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        bs.append(boxes); ss.append(scores); cs.append(classes)
        ts.append(np.full(len(boxes), window is not None))
        tr.append(np.zeros(len(boxes), bool) if window is None
                  else truncated_by_window(boxes, window, width, height, cfg.trunc_margin))
    if not bs:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64), np.zeros(0, bool))
    boxes = np.concatenate(bs)
    scores = np.concatenate(ss).astype(np.float32)
    classes = np.concatenate(cs).astype(np.int64)
    from_win = np.concatenate(ts)
    trunc = np.concatenate(tr)
    np.clip(boxes[:, [0, 2]], 0, width, out=boxes[:, [0, 2]])
    np.clip(boxes[:, [1, 3]], 0, height, out=boxes[:, [1, 3]])
    valid = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
    boxes, scores, classes, from_win, trunc = boxes[valid], scores[valid], classes[valid], from_win[valid], trunc[valid]
    if cfg.trunc_ioa > 0 and trunc.any() and (~trunc).any():
        t_idx, f_idx = np.flatnonzero(trunc), np.flatnonzero(~trunc)
        cover = _ioa(boxes[t_idx], boxes[f_idx])
        cover[classes[t_idx][:, None] != classes[f_idx][None, :]] = 0.0
        alive = np.ones(len(boxes), bool)
        alive[t_idx[cover.max(axis=1) >= cfg.trunc_ioa]] = False
        boxes, scores, classes, from_win = boxes[alive], scores[alive], classes[alive], from_win[alive]
    keep = class_nms(boxes, scores, classes, merge_iou, max_det)
    return boxes[keep], scores[keep], classes[keep], from_win[keep]


# Measured presets (training/results/TILING_EVAL.md). tile_conf 0.10 is the edge candidate floor (the detector
# never decodes below it); the evaluator's optimum was 0.02, worth under 0.01 day person AP50.
PRESETS: dict[str, TilingConfig] = {
    "off": TilingConfig(),
    "strip": TilingConfig(enabled=True, band=(0.0, 0.55), tile=640, overlap=0.2, tile_conf=0.10, merge_iou=0.5, trunc_ioa=0.5),
    "full": TilingConfig(enabled=True, band=(0.0, 1.0), tile=640, overlap=0.0, tile_conf=0.10, merge_iou=0.5, trunc_ioa=0.5),
}


def preset(name: str) -> TilingConfig:
    try:
        return PRESETS[name.strip().lower()]
    except KeyError:
        raise ValueError(f"tiling must be one of {sorted(PRESETS)}, got {name!r}") from None


__all__ = ["PRESETS", "preset", "TilingConfig","nms", "class_nms", "tile_windows", "truncated_by_window", "merge_passes"]
