"""Face privacy: blur-by-default storage.

Implements the first half of slide 4's privacy commitment: "face data stays on
the post" starts with never storing a face unblurred. Every stored clip or
thumbnail is built from `blur_faces_in_frame`, which blurs every detected face
before the frame is written anywhere. Access-controlled unsealing of the
unblurred crop, its encryption at rest, the audit log, and the 30-day
retention purge are added next.
"""

from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------------------------------------------
# Blur-by-default
# --------------------------------------------------------------------------------------------


def blur_region(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Return `frame_bgr` with a strong Gaussian blur over `bbox`. Does not mutate the input."""
    x1, y1, x2, y2 = bbox
    out = frame_bgr.copy()
    region = out[y1:y2, x1:x2]
    if region.size == 0:
        return out
    # Kernel scales with the face box so a close-range (large) face is not left
    # partially legible by a fixed small kernel.
    k = max(15, (min(region.shape[0], region.shape[1]) // 2) | 1)  # odd, >= 15
    out[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
    return out


def blur_faces_in_frame(frame_bgr: np.ndarray, faces) -> np.ndarray:
    """Blur every detected face. This is what gets written to a stored clip/thumbnail."""
    out = frame_bgr
    for face in faces:
        out = blur_region(out, face.bbox)
    return out
