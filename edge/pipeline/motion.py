"""The motion channel: dense Farneback optical flow, scored as object-to-background contrast.

How it is computed, and why it matches docs/MEASUREMENTS.md section 1
---------------------------------------------------------------------
Section 1 measured dense Farneback flow between CONSECUTIVE paired KAIST frames at their native
640 x 512, and compared the mean flow magnitude inside detected object boxes with the mean over the
background (every pixel outside those boxes):

    visible  5.17 px object / 2.74 px background = 1.9x
    LWIR     2.73 px object / 0.81 px background = 3.4x

The record does not list the Farneback parameters. This module pins OpenCV's reference parameters in
one constant, FARNEBACK_PARAMS, and uses them everywhere: here, in the live channel, and in
tools/reproduce_flow_contrast.py, which re-runs the section 1 method. Frames are converted to one grey
channel (LWIR in KAIST is stored as three identical channels) and resized to a working width of 640 px,
so a frame from a 1080p camera is judged on the same pixel scale the measurement was taken on and the
px/frame figures stay comparable. `object_background_contrast` is the section 1 measurement itself.

Absolute flow is lower on LWIR (whole-frame flow is only about 32% of visible, section 1). It is the
CONTRAST that rises, from 1.9x to 3.4x: a moving warm body stands out from a thermally flat background
more than a moving textured body stands out from a textured background. That, not any claim that flow
is untouched by the change of sensor, is why fusion.py weights motion higher on LWIR.

What the channel emits per frame
--------------------------------
MotionField   the flow magnitude map at working scale, per-cell background energy, the motion mask
              (magnitude above the camera's learnt per-cell threshold, calibration.py) and its connected
              regions as MotionRegion objects in original-frame pixels.
score_box()   the motion score m in [0, 1] for one candidate box:
                  E_in   mean magnitude inside the box
                  E_bg   mean magnitude in a ring around the box (other candidate boxes excluded):
                         the LOCAL background, so global camera shake raises both and cancels
                  C      E_in / max(E_bg, background floor learnt for this camera)
                  s_c    clip((C - 1) / (C_SAT - 1), 0, 1)                 contrast term
                  s_e    clip((E_in - tau_box) / tau_box, 0, 1)            energy above this camera's
                         learnt per-cell threshold tau (calibration.py), so sensor noise that happens to
                         have a high ratio over a near-zero background does not score
                  m      min(s_c, s_e): the box must move more than its surroundings AND more than this
                         camera's own background ever does
              plus the spatial-gate IoU: the IoU between the box and the flow-peak region, i.e. the
              connected motion-mask component inside the search window carrying the most flow energy
              inside the box (a component filling the whole window, as under shake, scores 0).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from pipeline.types import BBox, ChannelScore, MotionRegion

# OpenCV's reference Farneback parameters (the values in OpenCV's own optical-flow tutorial), pinned once.
FARNEBACK_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
WORKING_WIDTH = 640          # the native width of the KAIST frames section 1 was measured on
CELL_PX = 32                 # background grid cell edge, at working scale
C_SAT = 3.0                  # contrast at which the contrast term saturates
RING_SCALE = 1.0             # the background ring extends this many box sizes beyond the box (half each side)
MIN_REGION_PX = 24           # smallest motion-mask component kept, working-scale pixels
DEFAULT_TAU_PX = 0.6         # per-pixel motion threshold before a camera is calibrated, px/frame


# docs/MEASUREMENTS.md section 1: mean Farneback flow magnitude (px/frame) inside detected object boxes and
# over the background, 100 paired KAIST frames. Visible 5.17 / 2.74 = 1.9x; LWIR 2.73 / 0.81 = 3.4x.
MEASURED_FLOW_PX: dict[str, tuple[float, float]] = {"visible": (5.17, 2.74), "lwir": (2.73, 0.81)}


def measured_contrast(modality: str) -> float:
    obj, bg = MEASURED_FLOW_PX[modality]
    return obj / bg


def motion_reliability(modality: str) -> float:
    """Share of the in-box flow that belongs to the object rather than the background, 1 - bg/obj.

    From the section 1 numbers: visible 1 - 2.74/5.17 = 0.470; LWIR 1 - 0.81/2.73 = 0.703. It is the
    fraction of a moving object's flow signal that survives after the background level is subtracted,
    and fusion.py uses it directly as the motion weight for that modality.
    """
    obj, bg = MEASURED_FLOW_PX[modality]
    return 1.0 - bg / obj


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def working_size(width: int, height: int, working_width: int = WORKING_WIDTH) -> tuple[int, int]:
    if width <= working_width:
        return width, height
    return working_width, max(8, int(round(height * working_width / width)))


def farneback(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Dense flow (H, W, 2) with the pinned parameters."""
    return cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK_PARAMS)


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return cv2.magnitude(flow[..., 0], flow[..., 1])


def remove_global_shift(flow: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Subtract the frame's median flow vector: the translation every pixel shares.

    A pole swaying in the wind or a knocked mount translates the whole image; that shared vector is not
    object motion. The median over a 1-in-16 pixel subsample is robust to objects covering up to half
    the frame. On a steady camera the median is a few hundredths of a pixel and the subtraction is a
    no-op, so the live channel's magnitudes stay on the section 1 scale; the section 1 reproduction
    (tools/reproduce_flow_contrast.py) uses the raw flow, as the measurement did. Rotation and zoom are
    not translations and are not removed; the local-ring contrast in score_box absorbs what remains.
    """
    sub = flow[::4, ::4].reshape(-1, 2)
    dx, dy = float(np.median(sub[:, 0])), float(np.median(sub[:, 1]))
    if dx == 0.0 and dy == 0.0:
        return flow, (0.0, 0.0)
    return flow - np.array([dx, dy], dtype=flow.dtype), (dx, dy)


def boxes_mask(shape: tuple[int, int], boxes: Sequence[BBox]) -> np.ndarray:
    """Boolean mask of the union of `boxes` (already in the mask's pixel scale)."""
    m = np.zeros(shape, dtype=bool)
    h, w = shape
    for x1, y1, x2, y2 in boxes:
        xa, ya = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        xb, yb = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = True
    return m


def object_background_contrast(mag: np.ndarray, boxes: Sequence[BBox]) -> tuple[float, float, float] | None:
    """The docs/MEASUREMENTS.md section 1 measurement on one frame pair.

    Returns (mean magnitude inside the union of boxes, mean magnitude outside it, their ratio), or None
    when there is no box or no background. The section 1 table's contrast column equals the ratio of its
    two mean columns (5.17 / 2.74, 2.73 / 0.81), so over many pairs `aggregate_contrast` averages the object
    and background means separately and reports the ratio of those averages.
    """
    inside = boxes_mask(mag.shape, boxes)
    if not inside.any() or inside.all():
        return None
    obj, bg = float(mag[inside].mean()), float(mag[~inside].mean())
    return obj, bg, (obj / bg if bg > 0 else float("inf"))


def aggregate_contrast(samples: Sequence[tuple[float, float, float]]) -> dict:
    """Mean object flow, mean background flow and their ratio over many frame pairs."""
    if not samples:
        return {"pairs": 0, "object_px": float("nan"), "background_px": float("nan"), "contrast": float("nan")}
    obj = float(np.mean([s[0] for s in samples]))
    bg = float(np.mean([s[1] for s in samples]))
    return {"pairs": len(samples), "object_px": obj, "background_px": bg, "contrast": obj / bg if bg > 0 else float("inf")}


@dataclass
class MotionField:
    """The flow of one frame pair, at working scale, plus what the mask found."""

    mag: np.ndarray                  # float32 (h, w) flow magnitude, px/frame at working scale
    scale: tuple[float, float]       # (sx, sy): original px per working px
    cell_energy: np.ndarray          # float32 (gy, gx) mean magnitude per background cell
    tau_map: np.ndarray              # float32 (h, w) per-pixel threshold used for the mask
    mask: np.ndarray                 # bool (h, w) motion mask
    regions: list[MotionRegion]
    whole_frame: float               # mean magnitude over the whole frame
    moving_fraction: float           # share of the frame inside the motion mask
    timestamp: float
    elapsed_ms: float
    global_shift: tuple[float, float] = (0.0, 0.0)   # median flow vector removed (camera shake / pan), px

    def to_working(self, b: BBox) -> BBox:
        sx, sy = self.scale
        return b[0] / sx, b[1] / sy, b[2] / sx, b[3] / sy

    def to_original(self, b: BBox) -> BBox:
        sx, sy = self.scale
        return b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy


def cell_means(mag: np.ndarray, cell: int = CELL_PX) -> np.ndarray:
    """Mean magnitude per cell of a regular grid (ragged last row/column included)."""
    h, w = mag.shape
    gy, gx = max(1, int(np.ceil(h / cell))), max(1, int(np.ceil(w / cell)))
    return cv2.resize(mag, (gx, gy), interpolation=cv2.INTER_AREA)


def expand_to_map(cells: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Per-cell values -> per-pixel map (nearest cell)."""
    h, w = shape
    return cv2.resize(cells.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)


def _clip_box(b: BBox, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(np.clip(np.floor(b[0]), 0, w - 1)); y1 = int(np.clip(np.floor(b[1]), 0, h - 1))
    x2 = int(np.clip(np.ceil(b[2]), x1 + 1, w)); y2 = int(np.clip(np.ceil(b[3]), y1 + 1, h))
    return x1, y1, x2, y2


def _iou_int(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class MotionChannel:
    """Runs on EVERY frame. Holds the previous grey frame; nothing else is stateful."""

    def __init__(self, working_width: int = WORKING_WIDTH, cell_px: int = CELL_PX, compensate_global: bool = True) -> None:
        self.working_width = working_width
        self.cell_px = cell_px
        self.compensate_global = compensate_global
        self._prev: np.ndarray | None = None
        self._prev_shape: tuple[int, int] | None = None

    def reset(self) -> None:
        self._prev = None
        self._prev_shape = None

    def prepare(self, image: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        h, w = image.shape[:2]
        ww, wh = working_size(w, h, self.working_width)
        gray = to_gray(image)
        if (ww, wh) != (w, h):
            gray = cv2.resize(gray, (ww, wh), interpolation=cv2.INTER_AREA)
        return gray, (w / ww, h / wh)

    def update(self, image: np.ndarray, timestamp: float, tau_cells: np.ndarray | None = None) -> MotionField | None:
        """Flow from the previous frame to this one; None on the first frame (or after a size change).

        tau_cells  the camera's learnt per-cell threshold (calibration.py); DEFAULT_TAU_PX everywhere
                   before calibration exists.
        """
        t0 = time.perf_counter()
        gray, scale = self.prepare(image)
        prev = self._prev
        self._prev = gray
        if prev is None or prev.shape != gray.shape:
            return None
        flow = farneback(prev, gray)
        shift = (0.0, 0.0)
        if self.compensate_global:
            flow, shift = remove_global_shift(flow)
        mag = flow_magnitude(flow)
        cells = cell_means(mag, self.cell_px)
        if tau_cells is None or tau_cells.shape != cells.shape:
            tau_cells = np.full(cells.shape, DEFAULT_TAU_PX, dtype=np.float32)
        tau_map = expand_to_map(tau_cells, mag.shape)
        mask = mag > tau_map
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        regions = self._regions(mag, mask, scale)
        return MotionField(mag=mag, scale=scale, cell_energy=cells, tau_map=tau_map, mask=mask, regions=regions,
                           whole_frame=float(mag.mean()), moving_fraction=float(mask.mean()),
                           timestamp=timestamp, elapsed_ms=(time.perf_counter() - t0) * 1000, global_shift=shift)

    @staticmethod
    def _regions(mag: np.ndarray, mask: np.ndarray, scale: tuple[float, float]) -> list[MotionRegion]:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        sx, sy = scale
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < MIN_REGION_PX:
                continue
            sel = labels[y:y + h, x:x + w] == i
            vals = mag[y:y + h, x:x + w][sel]
            out.append(MotionRegion(bbox=(x * sx, y * sy, (x + w) * sx, (y + h) * sy), energy=float(vals.mean()),
                                    peak=float(vals.max()), area_px=float(area * sx * sy)))
        return out

    # ------------------------------------------------------------------------------ scoring

    @staticmethod
    def score_box(field: MotionField, bbox: BBox, others: Sequence[BBox] = (), bg_floor: float = 0.05,
                  c_sat: float = C_SAT) -> tuple[ChannelScore, float]:
        """(motion ChannelScore for `bbox`, spatial-gate IoU). `bbox` and `others` in original pixels."""
        h, w = field.mag.shape
        box = _clip_box(field.to_working(bbox), w, h)
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        inside = field.mag[y1:y2, x1:x2]
        e_in = float(inside.mean())

        # local background ring, other candidates excluded
        rx, ry = int(np.ceil(bw * RING_SCALE / 2)) + 2, int(np.ceil(bh * RING_SCALE / 2)) + 2
        wx1, wy1, wx2, wy2 = max(0, x1 - rx), max(0, y1 - ry), min(w, x2 + rx), min(h, y2 + ry)
        win = field.mag[wy1:wy2, wx1:wx2]
        ring = np.ones(win.shape, dtype=bool)
        ring[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1] = False
        for o in others:
            ox1, oy1, ox2, oy2 = _clip_box(field.to_working(o), w, h)
            ring[max(oy1, wy1) - wy1:max(min(oy2, wy2) - wy1, 0), max(ox1, wx1) - wx1:max(min(ox2, wx2) - wx1, 0)] = False
        e_bg = float(win[ring].mean()) if ring.sum() >= 16 else field.whole_frame
        contrast = e_in / max(e_bg, bg_floor)

        tau_box = float(field.tau_map[y1:y2, x1:x2].mean())
        s_c = float(np.clip((contrast - 1.0) / (c_sat - 1.0), 0.0, 1.0))
        s_e = float(np.clip((e_in - tau_box) / max(tau_box, 1e-6), 0.0, 1.0))
        m = min(s_c, s_e)

        # spatial gate: the flow-peak region is the motion-mask component (inside the search window) that
        # carries the most flow energy INSIDE the box; its bounding box must overlap the box by IoU. A
        # component that fills the whole window (touches all four sides) is not a peak but a field of
        # motion, as under camera shake, and scores IoU 0.
        iou = 0.0
        wmask = field.mask[wy1:wy2, wx1:wx2].astype(np.uint8)
        if wmask.any():
            n, labels, stats, _ = cv2.connectedComponentsWithStats(wmask, connectivity=8)
            if n > 1:
                lab_in = labels[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1]
                energy = np.bincount(lab_in.ravel(), weights=inside.ravel(), minlength=n)
                energy[0] = 0.0
                k = int(energy.argmax())
                if energy[k] > 0:
                    px, py, pw, ph, _ = stats[k]
                    fills_window = px == 0 and py == 0 and px + pw == wx2 - wx1 and py + ph == wy2 - wy1
                    if not fills_window:
                        iou = _iou_int((px + wx1, py + wy1, px + wx1 + pw, py + wy1 + ph), box)
        basis = f"E_in {e_in:.2f} px, E_bg {e_bg:.2f} px, C {contrast:.2f}x, tau {tau_box:.2f}"
        return ChannelScore("motion", round(m, 4), round(contrast, 4), basis), round(iou, 4)
