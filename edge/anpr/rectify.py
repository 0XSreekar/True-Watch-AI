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


def _row_ink_profile(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Otsu on the *inverted* image so dark strokes on a light plate (and vice
    # versa, via the automatic threshold) both count as ink == high value.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink_mean = binary.mean()
    if ink_mean > 255 * 0.5:
        # More than half the pixels came out as "ink": the polarity was
        # backwards (light text on dark plate looked inverted the other way).
        binary = 255 - binary
    return binary.sum(axis=1).astype("float64")


def split_lines(image: np.ndarray) -> list[np.ndarray]:
    """Split a rectified plate into 1 or 2 line images, by its own ink valley.

    Returns line crops top-to-bottom. A plate with no clear valley (a Bhutan
    single-line plate, or a Nepali plate cropped to one line already) comes
    back as a single-element list — the caller never has to special-case the
    grammar here.
    """
    height = image.shape[0]
    if height < 12:
        return [image]

    profile = _row_ink_profile(image)
    # Smooth the profile so single-pixel noise doesn't create a false valley.
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(profile, kernel, mode="same")

    # Search the middle band for the darkest (least-ink) row: the gap between
    # two lines of text sits roughly at mid-height on a well-cropped plate.
    band_lo = int(height * 0.25)
    band_hi = int(height * 0.75)
    if band_hi <= band_lo:
        return [image]
    band = smoothed[band_lo:band_hi]
    valley_local = int(np.argmin(band))
    valley = band_lo + valley_local

    peak = float(smoothed.max()) if smoothed.max() > 0 else 1.0
    valley_depth = 1.0 - (smoothed[valley] / peak)
    min_gap = int(height * LINE_VALLEY_MIN_GAP_FRACTION)

    # A real inter-line gap sits BETWEEN two ink bands: both sides of the
    # candidate valley, within the search band, must carry substantial ink of
    # their own. Without this, a single line of text with generous top/bottom
    # margin produces a "valley" at the margin's edge — no second line, just
    # the same line's own boundary — and would otherwise be split in two.
    above_peak = float(smoothed[band_lo:valley].max()) if valley > band_lo else 0.0
    below_peak = float(smoothed[valley:band_hi].max()) if valley < band_hi else 0.0
    both_sides_have_ink = above_peak >= 0.25 * peak and below_peak >= 0.25 * peak

    if valley_depth < 0.35 or valley < min_gap or (height - valley) < min_gap or not both_sides_have_ink:
        # No confident two-line structure: treat as one line.
        return [image]

    top = image[: max(1, valley), :]
    bottom = image[valley:, :]
    if top.shape[0] < 4 or bottom.shape[0] < 4:
        return [image]
    return [top, bottom]


def prepare_lines(plate_bgr: np.ndarray) -> tuple[list[np.ndarray], bool]:
    """The full rectify -> split -> upscale pipeline. Returns (line images, used_quad)."""
    rectified = rectify(plate_bgr)
    lines = split_lines(rectified.image)
    upscaled = [upscale_for_recognition(line) for line in lines]
    return upscaled, rectified.used_quad
