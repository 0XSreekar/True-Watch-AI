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
# A plate whose source shape is at least this wide is a one-line plate ("बा.२० च ४६८०"). Measured on hand-transcribed
# Kathmandu crops: two-line plates 1.45-2.04, one-line plates 2.37-2.54 (tilted, so the box under-states them).
ONE_LINE_MIN_ASPECT = 2.3
TWO_LINE_MAX_ASPECT = 2.6  # width/height at or below this is a two-line plate shape (Nepali 520x300 = 1.73)


@dataclass(frozen=True)
class RectifyResult:
    image: np.ndarray  # RECT_WIDTH x RECT_HEIGHT (one-line plates: RECT_WIDTH x their own height), perspective-corrected
    used_quad: bool  # True if a 4-point contour was found; False if the crop rect was used
    one_line: bool = False  # the source shape is a one-line plate, so the warp kept its aspect


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


def _source_aspect(source: np.ndarray) -> float:
    """Width over height of the plate as photographed: mean of opposite edges of the ordered quad."""
    top, right, bottom, left = (np.linalg.norm(source[(i + 1) % 4] - source[i]) for i in range(4))
    return float((top + bottom) / max(1e-6, left + right))


DESKEW_MAX_DEGREES = 30  # parked motorbikes and cars photographed from the side tilt plates this far
DESKEW_MIN_DEGREES = 2   # below this the axis-aligned crop is already level enough to split


def deskew(plate_bgr: np.ndarray) -> np.ndarray:
    """Level a tilted plate crop when no plate outline was found, and trim it to its text block.

    On hand-transcribed real photographs 36 of 84 readings were much shorter than the plate: an axis-aligned crop of a
    tilted plate was cut into lines horizontally, across the text. The tilt is the angle at which the ink's row profile
    is sharpest (text lines give tall, narrow peaks only when level), searched over +/- DESKEW_MAX_DEGREES.
    """
    height, width = plate_bgr.shape[:2]
    if height < 16 or width < 16:
        return plate_bgr
    scale = 200.0 / max(width, height)
    small = cv2.resize(plate_bgr, (max(8, int(width * scale)), max(8, int(height * scale))), interpolation=cv2.INTER_AREA)
    mask = _ink_mask(small).astype("float32")
    centre = (mask.shape[1] / 2.0, mask.shape[0] / 2.0)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-DESKEW_MAX_DEGREES, DESKEW_MAX_DEGREES + 0.5, 1.0):
        turned = cv2.warpAffine(mask, cv2.getRotationMatrix2D(centre, float(angle), 1.0), (mask.shape[1], mask.shape[0]))
        score = float(np.var(turned.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    if abs(best_angle) < DESKEW_MIN_DEGREES:
        return plate_bgr
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), best_angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(height * sin + width * cos), int(height * cos + width * sin)
    matrix[0, 2] += new_w / 2.0 - width / 2.0
    matrix[1, 2] += new_h / 2.0 - height / 2.0
    turned = cv2.warpAffine(plate_bgr, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Keep the rotated rectangle of the original crop's text block: rows and columns holding most of the ink.
    ink = _ink_mask(turned) > 0
    rows, cols = np.where(ink.sum(axis=1) > 0.02 * ink.shape[1])[0], np.where(ink.sum(axis=0) > 0.02 * ink.shape[0])[0]
    if len(rows) < 8 or len(cols) < 8:
        return turned
    pad_y, pad_x = int(0.08 * (rows[-1] - rows[0])), int(0.04 * (cols[-1] - cols[0]))
    return turned[max(0, rows[0] - pad_y):rows[-1] + pad_y + 1, max(0, cols[0] - pad_x):cols[-1] + pad_x + 1]


def rectify(plate_bgr: np.ndarray, layout: str | None = None, level: bool = False) -> RectifyResult:
    """Warp `plate_bgr` onto a RECT_WIDTH x RECT_HEIGHT canonical plate.

    A one-line plate (source aspect >= ONE_LINE_MIN_ASPECT, or layout="one") keeps its own aspect instead: squeezing a
    4:1 plate into the 1.73:1 two-line canvas stretched its glyphs to over twice their height and made the line splitter
    cut 43% of synthetic one-line plates through the text. layout="two" forces the canonical canvas.
    """
    if plate_bgr.size == 0:
        raise ValueError("rectify() received an empty crop")
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) if plate_bgr.ndim == 3 else plate_bgr
    quad = _find_quad(gray)
    if quad is None and level:
        # Opt-in: levelling always misjudged some already-level plates (synthetic one-line exact-match fell from 57.5% to
        # 41.8% when it ran on every plate), so recognise.read_plate asks for it only after a plain reading failed.
        plate_bgr = deskew(plate_bgr)
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) if plate_bgr.ndim == 3 else plate_bgr
    height, width = gray.shape[:2]

    if quad is not None:
        source = _order_corners(quad)
        used_quad = True
    else:
        source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        used_quad = False
    one_line = layout == "one" or (layout is None and _source_aspect(source) >= ONE_LINE_MIN_ASPECT)
    out_height = max(24, int(round(RECT_WIDTH / max(ONE_LINE_MIN_ASPECT, _source_aspect(source))))) if one_line else RECT_HEIGHT
    destination = np.float32([[0, 0], [RECT_WIDTH, 0], [RECT_WIDTH, out_height], [0, out_height]])

    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        plate_bgr, matrix, (RECT_WIDTH, out_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return RectifyResult(image=warped, used_quad=used_quad, one_line=one_line)


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


def split_lines(image: np.ndarray, max_lines: int = 2) -> list[np.ndarray]:
    """Split a rectified plate into line images, top to bottom: 1 or 2 by default, up to 3 with max_lines=3.

    A province plate prints three lines ("बागमती प्रदेश-०२" / "०३१ प" / "२०५०"); max_lines=3 tries _split_three first.
    It is opt-in: forced onto synthetic two-line plates it cut 8% of them in three, so recognise.read_plate only asks
    for it when the two-line reading does not parse as a plate (it finds three lines on 80% of province plates).
    """
    if max_lines >= 3:
        three = _split_three(image)
        if three is not None:
            return three
    return _split_once(image, allow_shape_fallback=True)


def _split_three(image: np.ndarray) -> list[np.ndarray] | None:
    """Cut a province plate at the emptiest row of its upper half and of its lower half, or return None.

    Only reached through max_lines=3, i.e. when a label says three lines or when a two-line reading failed the plate
    grammar, so a wrong cut costs one rejected reading rather than a misread plate.
    """
    height, width = image.shape[:2]
    y0, y1 = int(height * BORDER_TRIM_FRACTION), int(height * (1.0 - BORDER_TRIM_FRACTION))
    x0, x1 = int(width * BORDER_TRIM_X_FRACTION), int(width * (1.0 - BORDER_TRIM_X_FRACTION))
    core = _ink_mask(image)[y0:y1, x0:x1]
    core_h = core.shape[0]
    if core_h < 36:
        return None
    profile = core.sum(axis=1).astype("float64")
    size = max(3, (core_h // 60) | 1)
    smoothed = np.convolve(profile, np.ones(size) / size, mode="same")
    peak = float(smoothed.max())
    if peak <= 0:
        return None
    first = int(core_h * 0.15) + int(np.argmin(smoothed[int(core_h * 0.15):int(core_h * 0.48)]))
    second = int(core_h * 0.48) + int(np.argmin(smoothed[int(core_h * 0.48):int(core_h * 0.85)]))
    bands = (smoothed[:first], smoothed[first:second], smoothed[second:])
    if min(len(b) for b in bands) < 4 or min(float(b.max()) for b in bands) < 0.05 * peak:
        return None
    for valley, left, right in ((first, bands[0], bands[1]), (second, bands[1], bands[2])):
        weaker = min(float(left.max()), float(right.max()))
        if 1.0 - float(smoothed[valley]) / weaker < 0.5:
            return None
    cuts = (y0 + first, y0 + second)
    return [image[:cuts[0], :], image[cuts[0]:cuts[1], :], image[cuts[1]:, :]]


def _split_once(image: np.ndarray, allow_shape_fallback: bool) -> list[np.ndarray]:
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
    if not confident and not (allow_shape_fallback and two_line_shape and depth >= 0.2):
        return [image]

    cut = y0 + valley
    top, bottom = image[:cut, :], image[cut:, :]
    if top.shape[0] < 4 or bottom.shape[0] < 4:
        return [image]
    return [top, bottom]


def prepare_lines(plate_bgr: np.ndarray, layout: str | None = None, level: bool = False) -> tuple[list[np.ndarray], bool]:
    """The full rectify -> split -> upscale pipeline. Returns (line images, used_quad).

    layout None decides one-line versus two-line from the plate's shape; "one" / "two" force either reading and
    "three" allows a province plate's third line. recognise.read_plate uses them as fallbacks when the first reading
    does not parse as a plate.
    """
    rectified = rectify(plate_bgr, "two" if layout == "three" else layout, level=level)
    if rectified.one_line:
        lines = [rectified.image]
    else:
        lines = split_lines(rectified.image, max_lines=3 if layout == "three" else 2)
    upscaled = [upscale_for_recognition(line) for line in lines]
    return upscaled, rectified.used_quad
