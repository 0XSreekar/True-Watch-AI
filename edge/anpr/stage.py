"""The ANPR step of the live pipeline: plate-bearing tracks in, plate readings out.

It is fed TRACKS, not raw detections. `pipeline.types.Detection` carries no track id; `Track` does, and
it has the same `class_name` and pixel (x1, y1, x2, y2) `bbox` that `detect_plate.VehicleDetection`
reads, so every `PlateCandidate` comes back with `source_track_id` set and the reading can be attached
to that track's events.

Each track gets a bounded number of attempts (`max_attempts_per_track`, on frames where the detector
actually saw it) and stops once a reading parses as a plate. Recognition (PP-OCRv5 through paddleocr,
recognise.read_plate) is loaded on first use; if it is not installed, candidates are still located and
counted and the stage reports the recogniser as unavailable instead of failing the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from . import detect_plate

log = logging.getLogger("truewatch.edge.anpr")


@dataclass(frozen=True)
class PlateRead:
    track_id: int | None
    bbox: tuple[int, int, int, int]          # plate box in frame pixels
    candidate_score: float
    text: str | None                         # None when the recogniser is unavailable or read nothing
    script: str | None
    line_confidences: tuple[float, ...] = ()
    valid: bool = False                      # the text parses as a Nepal / Bhutan plate

    def contract_block(self) -> dict | None:
        """The `anpr` block of truewatch.event.v1, or None when nothing was read."""
        if not self.text:
            return None
        return {"text": self.text, "script": self.script, "line_confidences": list(self.line_confidences)}


def _default_reader(plate_bgr: np.ndarray):
    from .recognise import read_plate

    return read_plate(plate_bgr)


def _validate(text: str, script: str) -> bool:
    from .postprocess import validate

    return validate(text, script).valid


@dataclass
class PlateStage:
    reader: Callable[[np.ndarray], object] = _default_reader
    max_attempts_per_track: int = 3
    min_vehicle_width_px: int = 48
    save_dir: Path | None = None             # write each plate crop here (debugging, offline OCR)
    attempts: int = 0                        # tracks x frames the plate locator ran on
    candidates: int = 0                      # plate-shaped regions found
    reads: int = 0                           # candidates the recogniser returned text for
    valid_reads: int = 0
    recogniser_error: str | None = None
    _per_track: dict = field(default_factory=dict)
    _done: set = field(default_factory=set)

    def eligible(self, track) -> bool:
        if not detect_plate.is_plate_bearing(track) or track.track_id in self._done:
            return False
        if self._per_track.get(track.track_id, 0) >= self.max_attempts_per_track:
            return False
        x1, _, x2, _ = track.bbox
        return (x2 - x1) >= self.min_vehicle_width_px

    def run(self, frame_bgr: np.ndarray, tracks: Iterable, frame_index: int = 0) -> list[PlateRead]:
        out: list[PlateRead] = []
        for track in tracks:
            if not self.eligible(track):
                continue
            self._per_track[track.track_id] = self._per_track.get(track.track_id, 0) + 1
            self.attempts += 1
            for cand in detect_plate.detect_plates(frame_bgr, [track]):
                self.candidates += 1
                crop = cand.crop(frame_bgr)
                if self.save_dir is not None:
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(self.save_dir / f"trk{cand.source_track_id}_f{frame_index}.png"), crop)
                read = self._recognise(crop, cand)
                out.append(read)
                if read.valid:
                    self._done.add(track.track_id)
        return out

    def _recognise(self, crop: np.ndarray, cand: detect_plate.PlateCandidate) -> PlateRead:
        base = dict(track_id=cand.source_track_id, bbox=cand.bbox, candidate_score=round(float(cand.score), 4))
        if self.recogniser_error is not None:
            return PlateRead(text=None, script=None, **base)
        try:
            result = self.reader(crop)
        except ImportError as exc:
            self.recogniser_error = f"plate recogniser unavailable: {exc}"
            log.warning("anpr: %s; plates are located but not read", self.recogniser_error)
            return PlateRead(text=None, script=None, **base)
        text = getattr(result, "text", "") or ""
        script = getattr(result, "script", None)
        if not text:
            return PlateRead(text=None, script=script, **base)
        self.reads += 1
        valid = _validate(text, script or "devanagari")
        self.valid_reads += int(valid)
        return PlateRead(text=text, script=script, valid=valid,
                         line_confidences=tuple(getattr(result, "line_confidences", ()) or ()), **base)

    def stats(self) -> dict:
        return {"attempts": self.attempts, "candidates": self.candidates, "reads": self.reads,
                "valid_reads": self.valid_reads, "recogniser_error": self.recogniser_error}
