"""Locate plate regions inside vehicle detections from Phase 3.

Phase 3 (edge/pipeline/, on a concurrent branch) defines a `Detection` dataclass
carrying a pixel-space `bbox` (x1, y1, x2, y2), a class name, a score, a
`track_id`, a `camera_id` and a `modality`. This module never imports
`edge.pipeline` — the branches have not merged — so it accepts anything that
satisfies `VehicleDetection` below through structural typing
(`typing.Protocol`). Once Phase 3 lands, its `Detection` instances plug in here
unchanged, because a dataclass with a `.bbox` and a `.cls`/class-name attribute
already satisfies the protocol; no adapter is needed.

A plate is found with classical CV, not a second detector: gradient-based
blackhat + Sobel highlighting, morphological closing into candidate blobs,
then filtering by aspect ratio and position within the vehicle box. This is
deliberately cheap (no extra model download, no extra inference pass) because
Phase 3's own appearance channel already spent the detector budget on the
vehicle box; ANPR only has to find a plate-shaped region inside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import cv2
import numpy as np

# Classes that can carry a plate. Two-wheelers, cars, trucks and carts all can;
# a bare 'person' cannot, so it is never scanned.
PLATE_BEARING_CLASSES = {"car", "truck", "two_wheeler", "motorcycle", "cart", "bus"}

# A Nepali two-line plate is ~520x300 (datasets/plates/plates.yaml layout), so
# width/height ~= 1.73. A Bhutan single-line plate is wider and shorter. This
# window covers both without needing to know which one a given crop holds —
# recognise.py decides the script from the pixels, not from this ratio.
MIN_ASPECT = 1.15
MAX_ASPECT = 5.5
MIN_PLATE_WIDTH_PX = 18
MIN_PLATE_AREA_FRACTION = 0.0025  # of the vehicle crop
MAX_PLATE_AREA_FRACTION = 0.55


@runtime_checkable
class VehicleDetection(Protocol):
    """Structural contract with pipeline.types.Detection (Phase 3, concurrent branch).

    Anything with a pixel-space `.bbox` (x1, y1, x2, y2) and a `.cls` class name
    satisfies this without importing edge/pipeline/. Extra attributes (score,
    track_id, camera_id, modality) are ignored here; they are read by the
    fusion and rule modules Phase 3/4 own.
    """

    bbox: Sequence[float]
    cls: str


@dataclass(frozen=True)
class PlateCandidate:
    """A plate-shaped region, in the ORIGINAL frame's pixel coordinates."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float  # 0..1, how plate-like the region's gradient structure is
    source_track_id: int | None

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return frame[y1:y2, x1:x2]


def _class_name(detection: VehicleDetection) -> str:
    return str(getattr(detection, "cls", "") or "").lower()


def _track_id(detection: VehicleDetection) -> int | None:
    value = getattr(detection, "track_id", None)
    return int(value) if value is not None else None


def is_plate_bearing(detection: VehicleDetection) -> bool:
    return _class_name(detection) in PLATE_BEARING_CLASSES


def _clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return x1, y1, x2, y2


def vehicle_crop(frame: np.ndarray, detection: VehicleDetection) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """The vehicle's own crop, plus its clipped bbox in frame coordinates."""
    height, width = frame.shape[:2]
    box = tuple(float(v) for v in detection.bbox)
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = _clip_box(*box, width=width, height=height)
    if x2 - x1 < MIN_PLATE_WIDTH_PX or y2 - y1 < MIN_PLATE_WIDTH_PX:
        return None
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def _candidate_blobs(crop_gray: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """Blackhat + Sobel gradient candidate boxes (local coordinates, in `crop_gray`)."""
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    blackhat = cv2.morphologyEx(crop_gray, cv2.MORPH_BLACKHAT, kernel_rect)

    grad_x = cv2.Sobel(blackhat, cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    max_val = float(grad_x.max()) if grad_x.size else 0.0
    if max_val > 0:
        grad_x = (grad_x / max_val * 255.0).astype("uint8")
    else:
        grad_x = grad_x.astype("uint8")

    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel_rect)
    _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        area = cv2.contourArea(contour)
        rect_area = w * h
        fill = area / rect_area if rect_area > 0 else 0.0
        boxes.append((x, y, x + w, y + h, fill))
    return boxes


def _score_box(box: tuple[int, int, int, int, float], crop_area: int) -> float | None:
    x1, y1, x2, y2, fill = box
    width, height = x2 - x1, y2 - y1
    if width < MIN_PLATE_WIDTH_PX or height <= 0:
        return None
    aspect = width / height
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        return None
    area_fraction = (width * height) / crop_area if crop_area else 0.0
    if not (MIN_PLATE_AREA_FRACTION <= area_fraction <= MAX_PLATE_AREA_FRACTION):
        return None
    # A plate sits in the lower two-thirds of a vehicle box (bumper height), and
    # rectangularity (`fill`) rewards a filled rectangle over a scattered blob.
    aspect_mid = (MIN_ASPECT + MAX_ASPECT) / 2
    aspect_score = 1.0 - min(1.0, abs(aspect - aspect_mid) / aspect_mid)
    return max(0.0, min(1.0, 0.5 * fill + 0.5 * aspect_score))


def find_plates_in_crop(vehicle_bgr: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
    """Plate boxes local to `vehicle_bgr`, as (x1, y1, x2, y2, score), best first."""
    if vehicle_bgr.size == 0:
        return []
    gray = cv2.cvtColor(vehicle_bgr, cv2.COLOR_BGR2GRAY) if vehicle_bgr.ndim == 3 else vehicle_bgr
    # Bias the search to the lower 75% of the vehicle box (bumper/plate height);
    # a plate mounted higher (rare) is still reachable because the offset is
    # added back before returning frame-local coordinates.
    y_offset = int(gray.shape[0] * 0.15)
    search = gray[y_offset:, :]
    crop_area = search.shape[0] * search.shape[1]
    if crop_area == 0:
        return []

    scored: list[tuple[tuple[int, int, int, int], float]] = []
    for box in _candidate_blobs(search):
        score = _score_box(box, crop_area)
        if score is None:
            continue
        x1, y1, x2, y2, _fill = box
        scored.append(((x1, y1 + y_offset, x2, y2 + y_offset), score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def detect_plates(
    frame: np.ndarray,
    detections: Sequence[VehicleDetection],
    max_per_vehicle: int = 1,
) -> list[PlateCandidate]:
    """Plate candidates for every plate-bearing detection, in frame coordinates."""
    candidates: list[PlateCandidate] = []
    for detection in detections:
        if not is_plate_bearing(detection):
            continue
        cropped = vehicle_crop(frame, detection)
        if cropped is None:
            continue
        vehicle_bgr, (vx1, vy1, _vx2, _vy2) = cropped
        local_boxes = find_plates_in_crop(vehicle_bgr)[:max_per_vehicle]
        for (lx1, ly1, lx2, ly2), score in local_boxes:
            candidates.append(
                PlateCandidate(
                    bbox=(vx1 + lx1, vy1 + ly1, vx1 + lx2, vy1 + ly2),
                    score=score,
                    source_track_id=_track_id(detection),
                )
            )
    return candidates
