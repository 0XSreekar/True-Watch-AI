"""Perspective correction, upscaling and two-line splitting for plate crops.

Nepali plates are two lines — a zone+lot line over a vehicle-class+serial line
(datasets/plates/plates.yaml, DATASET_SPEC.md section 6.1: `बा १२` / `प १२३४`).
Bhutan plates (`BP-N-ANNNN`) are single line. This module never assumes the
line count: it reads it off the crop's own horizontal ink profile, so the same
code serves both grammars.

`datasets/plates/degrade.py` documents the corpus's hardest cases: a plate as
narrow as 60 px, seen at up to 35 degrees of yaw and 25 degrees of pitch. This
module targets exactly that: a perspective quad found from the crop's own
edges (falling back to the crop rectangle when no quad is confident), then
upscaling to a fixed working height before recognition.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

RECT_WIDTH = 520  # matches datasets/plates/plates.yaml layout.plate_width_px
RECT_HEIGHT = 300  # matches datasets/plates/plates.yaml layout.plate_height_px
MIN_WORKING_WIDTH = 240  # recognition below this width degrades sharply
LINE_VALLEY_MIN_GAP_FRACTION = 0.12  # a real inter-line gap is at least this tall
BORDER_TRIM_FRACTION = 0.06  # rows dropped at top and bottom before measuring ink (the printed border)
BORDER_TRIM_X_FRACTION = 0.05  # columns dropped at left and right, for the same reason
TWO_LINE_MAX_ASPECT = 2.6  # width/height at or below this is a two-line plate shape (Nepali 520x300 = 1.73)


@dataclass(frozen=True)
class RectifyResult:
    image: np.ndarray  # RECT_WIDTH x RECT_HEIGHT, perspective-corrected and upscaled
    used_quad: bool  # True if a 4-point contour was found; False if the crop rect was used


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left, by sum/difference of coordinates."""
    rect = np.zeros((4, 2), dtype="float32")
    total = points.sum(axis=1)
    rect[0] = points[np.argmin(total)]
    rect[2] = points[np.argmax(total)]
    diff = np.diff(points, axis=1).reshape(-1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def _find_quad(gray: np.ndarray) -> np.ndarray | None:
    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.dilate(edges, None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    crop_area = gray.shape[0] * gray.shape[1]
    best: np.ndarray | None = None
    best_area = 0.0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area < 0.35 * crop_area:  # too small to be the whole plate
            continue
        if area > best_area:
            best = approx.reshape(4, 2).astype("float32")
            best_area = area
    return best


def rectify(plate_bgr: np.ndarray) -> RectifyResult:
    """Warp `plate_bgr` onto a RECT_WIDTH x RECT_HEIGHT canonical plate."""
    if plate_bgr.size == 0:
        raise ValueError("rectify() received an empty crop")
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) if plate_bgr.ndim == 3 else plate_bgr
    quad = _find_quad(gray)
    height, width = gray.shape[:2]

    destination = np.float32([[0, 0], [RECT_WIDTH, 0], [RECT_WIDTH, RECT_HEIGHT], [0, RECT_HEIGHT]])
    if quad is not None:
        source = _order_corners(quad)
        used_quad = True
    else:
        source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        used_quad = False

    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        plate_bgr, matrix, (RECT_WIDTH, RECT_HEIGHT), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return RectifyResult(image=warped, used_quad=used_quad)


def upscale_for_recognition(image: np.ndarray, min_width: int = MIN_WORKING_WIDTH) -> np.ndarray:
    """Upscale small crops (a 60 px plate at range) before OCR, sharpening the result.

    A plain resize blurs small text further; an unsharp mask afterwards recovers
    stroke edges without inventing detail the sensor never captured.
    """
    height, width = image.shape[:2]
    if width >= min_width:
        return image
    scale = min_width / max(1, width)
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(resized, 1.6, blurred, -0.6, 0)
    return sharpened


def _ink_mask(image: np.ndarray) -> np.ndarray:
    """Binary ink mask (255 = ink) of a whole plate.

    Otsu on the inverted image makes dark strokes on a light plate ink; when more than half the
    pixels come out as ink the polarity was backwards (light text on a dark plate) and is flipped.
    The decision is made on the WHOLE plate, where the background dominates, and never on an
    inner crop, whose text can cover more than half of it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if binary.mean() > 255 * 0.5:
        binary = 255 - binary
    return binary


def _row_ink_profile(image: np.ndarray) -> np.ndarray:
    return _ink_mask(image).sum(axis=1).astype("float64")


def split_lines(image: np.ndarray) -> list[np.ndarray]:
    """Split a rectified plate into 1 or 2 line images at the gap between its text lines.

    Returns line crops top-to-bottom. A plate with no two-line structure (a Bhutan single-line
    plate, or a Nepali plate already cropped to one line) comes back as a single-element list.

    The ink profile is measured inside the plate's border: the printed border's thick top and
    bottom edges are the heaviest rows of the plate, and measuring against them made real text
    lines look too faint to count, which left 11.8% of the synthetic validation plates unsplit
    (and therefore unreadable). The valley is judged against the ink of the two text bands on
    either side of it, not against the whole-plate peak. When no confident valley is found but
    the plate has two-line proportions, it is split at the least-inked row near the middle.
    """
    height, width = image.shape[:2]
    if height < 12:
        return [image]

    y0, y1 = int(height * BORDER_TRIM_FRACTION), int(height * (1.0 - BORDER_TRIM_FRACTION))
    x0, x1 = int(width * BORDER_TRIM_X_FRACTION), int(width * (1.0 - BORDER_TRIM_X_FRACTION))
    core_mask = _ink_mask(image)[y0:y1, x0:x1]
    core_h = core_mask.shape[0]
    if core_h < 12:
        return [image]

    profile = core_mask.sum(axis=1).astype("float64")
    # Smooth so single-pixel noise (dust, compression) cannot create a false valley.
    size = max(3, (core_h // 60) | 1)
    smoothed = np.convolve(profile, np.ones(size) / size, mode="same")
    peak = float(smoothed.max())
    if peak <= 0:
        return [image]

    band_lo, band_hi = int(core_h * 0.30), int(core_h * 0.70)
    valley = band_lo + int(np.argmin(smoothed[band_lo:band_hi]))
    min_gap = int(core_h * LINE_VALLEY_MIN_GAP_FRACTION)
    # Each side's ink is measured away from the outer edge: a border thicker than the trim leaves
    # a dark row at the very top or bottom, which must not pass for a line of text.
    above = float(smoothed[min_gap:valley].max()) if valley > min_gap else 0.0
    below = float(smoothed[valley:core_h - min_gap].max()) if valley < core_h - min_gap else 0.0
    weaker_side = min(above, below)
    depth = 1.0 - float(smoothed[valley]) / weaker_side if weaker_side > 0 else 0.0

    confident = (
        weaker_side >= 0.2 * peak           # a real second band of ink, not a margin
        and depth >= 0.5                    # the gap is clearly emptier than both lines
        and valley >= min_gap
        and core_h - valley >= min_gap
    )
    two_line_shape = width / max(1, height) <= TWO_LINE_MAX_ASPECT
    if not confident and not (two_line_shape and depth >= 0.2):
        return [image]

    cut = y0 + valley
    top, bottom = image[:cut, :], image[cut:, :]
    if top.shape[0] < 4 or bottom.shape[0] < 4:
        return [image]
    return [top, bottom]


def prepare_lines(plate_bgr: np.ndarray) -> tuple[list[np.ndarray], bool]:
    """The full rectify -> split -> upscale pipeline. Returns (line images, used_quad)."""
    rectified = rectify(plate_bgr)
    lines = split_lines(rectified.image)
    upscaled = [upscale_for_recognition(line) for line in lines]
    return upscaled, rectified.used_quad
