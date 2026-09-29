"""Face detection on close-range person tracks, wired into the live pipeline (pipeline/service.py).

YuNet (face/detect.py) is only useful when a face is big enough to find, so this stage looks only at
PERSON tracks whose box is at least `min_height_frac` of the frame height (close range: a check-post
barrier, a gate). For each such track it runs YuNet on the head region of the person box, not on the
whole frame, which is both faster and keeps the search where a face can be.

What is stored, and what is not (face/privacy.py holds the rules; nothing here weakens them)
    * a BLURRED crop of the person: every detected face is Gaussian-blurred before the crop is written,
      so nothing on disk in the clear shows a face;
    * the UNBLURRED face crop only in sealed form (`privacy.store_sealed_crop`, Fernet key from
      FACE_UNSEAL_KEY). If no key is configured the sealed copy is skipped and said so in the stats; the
      clear crop is never written instead;
    * expired sealed crops are purged after `RETENTION_DAYS` (30) by `privacy.purge_expired`, run at start
      and once an hour, each purge audited. Unsealing stays Supervisor-only with an audit row per attempt
      (`privacy.unseal`).

One stored record per track: once a face is stored for a track it is not looked for again, and a track with
no face is retried only every `retry_every` observed frames, up to `max_attempts`, so a person walking
with their back to the camera costs a handful of detector calls, not one per frame.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from face import privacy
from face.detect import Face

log = logging.getLogger("truewatch.edge.face.stage")

HEAD_FRACTION = 0.45          # the top part of a standing person's box, where the head is
HEAD_MARGIN = 0.15            # extra room around the box, as a fraction of its size
PURGE_EVERY_S = 3600.0
DEFAULT_MIN_HEIGHT_FRAC = 0.25
DEFAULT_RETRY_EVERY = 3
DEFAULT_MAX_ATTEMPTS = 6


class FaceStage:
    def __init__(
        self,
        store_dir: Path,
        *,
        min_height_frac: float = DEFAULT_MIN_HEIGHT_FRAC,
        detect: Callable[[np.ndarray], Sequence[Face]] | None = None,
        retry_every: int = DEFAULT_RETRY_EVERY,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        audit_log: privacy.AuditLog | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.min_height_frac = float(min_height_frac)
        self._detect = detect
        self.retry_every = max(1, int(retry_every))
        self.max_attempts = max(1, int(max_attempts))
        self.audit_log = audit_log
        self._clock = clock
        self._attempts: dict[int, int] = {}
        self._seen_frames: dict[int, int] = {}
        self._done: set[int] = set()
        self._last_purge = 0.0
        self._sealed_warned = False
        self.considered = 0       # close-range person track-frames looked at
        self.detector_calls = 0
        self.tracks_with_face = 0
        self.faces_found = 0
        self.stored = 0
        self.sealed = 0
        self.sealed_skipped = 0   # no FACE_UNSEAL_KEY: the blurred crop was still stored
        self.purged = 0
        self.errors = 0
        self.total_ms = 0.0

    def warm_up(self) -> bool:
        """Load the detector now (fetching its pinned weights if they are missing) so the first close-range
        person does not stall the frame loop on a download. False if it cannot be loaded."""
        try:
            self._detector()
            return True
        except Exception:  # noqa: BLE001
            log.exception("face detector could not be loaded; face detection stays off")
            return False

    def _detector(self) -> Callable[[np.ndarray], Sequence[Face]]:
        if self._detect is None:
            from face.detect import detect_faces

            self._detect = detect_faces
        return self._detect

    def _maybe_purge(self) -> None:
        now = self._clock()
        if now - self._last_purge < PURGE_EVERY_S:
            return
        self._last_purge = now
        try:
            self.purged += len(privacy.purge_expired(self.store_dir, now=now, audit_log=self.audit_log))
        except Exception:  # noqa: BLE001 - housekeeping must not stop detection
            self.errors += 1
            log.exception("face crop retention purge failed")

    def _is_close_person(self, track, frame_h: int) -> bool:
        if track.class_name != "person":
            return False
        x1, y1, x2, y2 = track.bbox
        return (y2 - y1) >= self.min_height_frac * frame_h

    def run(self, image: np.ndarray, tracks: Sequence, *, frame_index: int, camera_id: str) -> list[dict]:
        """Look at each close-range person track once per its retry schedule; return new stored records."""
        self._maybe_purge()
        h, w = image.shape[:2]
        out: list[dict] = []
        for track in tracks:
            tid = track.track_id
            if tid in self._done or not self._is_close_person(track, h):
                continue
            self.considered += 1
            n = self._seen_frames[tid] = self._seen_frames.get(tid, -1) + 1
            if n % self.retry_every != 0 or self._attempts.get(tid, 0) >= self.max_attempts:
                continue
            self._attempts[tid] = self._attempts.get(tid, 0) + 1
            t0 = time.perf_counter()
            try:
                record = self._process(image, track, frame_index, camera_id, w, h)
            except Exception:  # noqa: BLE001 - a face failure must never stop the pipeline
                self.errors += 1
                log.exception("face stage failed on track %s", tid)
                record = None
            self.total_ms += (time.perf_counter() - t0) * 1000.0
            if record is not None:
                self._done.add(tid)
                out.append(record)
        return out

    def _process(self, image: np.ndarray, track, frame_index: int, camera_id: str, w: int, h: int) -> dict | None:
        x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * HEAD_MARGIN), int(bh * HEAD_MARGIN)
        px1, px2 = max(0, x1 - mx), min(w, x2 + mx)
        py1, py2 = max(0, y1 - my), min(h, y2 + my)
        head_y2 = min(h, y1 + int(bh * HEAD_FRACTION) + my)
        if px2 <= px1 or head_y2 <= py1:
            return None
        head = image[py1:head_y2, px1:px2]
        self.detector_calls += 1
        faces = list(self._detector()(head))
        if not faces:
            return None
        self.faces_found += len(faces)
        self.tracks_with_face += 1
        person = image[py1:py2, px1:px2].copy()
        # Faces are in head-crop coordinates, which share the person crop's origin, so they blur in place.
        blurred = privacy.blur_faces_in_frame(person, faces)
        clip_id = f"{camera_id}-t{track.track_id}-f{frame_index}"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        blurred_path = self.store_dir / f"{clip_id}.blurred.jpg"
        if not cv2.imwrite(str(blurred_path), blurred, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
            raise privacy.PrivacyError(f"could not write the blurred crop {blurred_path}")
        self.stored += 1
        best = max(faces, key=lambda fc: fc.score)
        sealed_path = None
        try:
            stored = privacy.store_sealed_crop(best.crop(head), clip_id, self.store_dir)
            sealed_path = str(stored.path)
            self.sealed += 1
        except privacy.PrivacyError as exc:
            self.sealed_skipped += 1
            if not self._sealed_warned:
                self._sealed_warned = True
                log.warning("unblurred face crops are NOT kept (%s); blurred crops are still stored", exc)
        return {
            "track_id": track.track_id,
            "clip_id": clip_id,
            "frame_index": frame_index,
            "faces": len(faces),
            "face_boxes_in_crop": [list(fc.bbox) for fc in faces],   # x1, y1, x2, y2 inside the blurred crop
            "best_score": round(float(best.score), 3),
            "blurred_crop": str(blurred_path),
            "sealed_crop": sealed_path,
        }

    def stats(self) -> dict:
        calls = max(1, self.detector_calls)
        return {
            "considered": self.considered,
            "detector_calls": self.detector_calls,
            "tracks_with_face": self.tracks_with_face,
            "faces_found": self.faces_found,
            "blurred_stored": self.stored,
            "sealed_stored": self.sealed,
            "sealed_skipped_no_key": self.sealed_skipped,
            "purged": self.purged,
            "errors": self.errors,
            "ms_per_detector_call": round(self.total_ms / calls, 2),
        }
