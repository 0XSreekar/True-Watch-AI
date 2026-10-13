"""YuNet face detection — pretrained, no fine-tuning (PDF slide 2, capability 3).

Close range at gates and check posts is exactly YuNet's designed regime (it is
trained for faces from roughly 10 px upward, and check-post cameras put a face
well inside that range at the barrier). `cv2.FaceDetectorYN` runs the ONNX
graph `yunet_weights.py` fetches; nothing here trains or adjusts its weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from face import yunet_weights

log = logging.getLogger("truewatch.edge.face")

DEFAULT_SCORE_THRESHOLD = 0.75
DEFAULT_NMS_THRESHOLD = 0.35
DEFAULT_TOP_K = 25


@dataclass(frozen=True)
class Face:
    """One detected face, in the ORIGINAL frame's pixel coordinates."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    landmarks: tuple[tuple[int, int], ...]  # right eye, left eye, nose, right mouth, left mouth

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return frame[y1:y2, x1:x2]


class FaceDetector:
    """A YuNet session bound to one input size; re-created only when the size changes.

    YuNet's ONNX graph takes a fixed input size per call
    (`cv2.FaceDetectorYN.setInputSize`), so this wrapper re-sets it per frame
    shape instead of rebuilding the detector — cheap, and avoids re-reading the
    ONNX file from disk on every frame.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        path = model_path or yunet_weights.fetch()
        self._detector = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), score_threshold, nms_threshold, top_k
        )
        self._size: tuple[int, int] | None = None
        log.info("YuNet loaded from %s (score_threshold=%.2f)", path, score_threshold)

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        height, width = frame_bgr.shape[:2]
        if (width, height) != self._size:
            self._detector.setInputSize((width, height))
            self._size = (width, height)

        _retval, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return []

        results: list[Face] = []
        for row in faces:
            x, y, w, h = row[0:4]
            score = float(row[14])
            landmark_pairs = row[4:14].reshape(5, 2)
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(width, int(round(x + w)))
            y2 = min(height, int(round(y + h)))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                Face(
                    bbox=(x1, y1, x2, y2),
                    score=score,
                    landmarks=tuple((int(round(px)), int(round(py))) for px, py in landmark_pairs),
                )
            )
        return results


_shared_detector: FaceDetector | None = None


def get_detector() -> FaceDetector:
    """A process-wide singleton, so the pipeline never reloads the ONNX graph per frame."""
    global _shared_detector
    if _shared_detector is None:
        _shared_detector = FaceDetector()
    return _shared_detector


def detect_faces(frame_bgr: np.ndarray) -> list[Face]:
    return get_detector().detect(frame_bgr)
