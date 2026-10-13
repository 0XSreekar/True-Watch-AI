"""Seal a clip for every event as it is recorded (slide 3, step 6; innovation claim (d)).

The service hands every processed frame to `ClipSealer.add`, which keeps the last few seconds as
JPEG bytes in a bounded ring. When an event is emitted, `seal` writes that window through
`EvidenceStore.open_clip` — the SHA-256 is computed while the bytes are written, not afterwards —
and appends the block to the hash chain. The returned `evidence` block travels to the backend as a
`truewatch.update.v1` for the event, so the alert carries the hash of the clip that shows it.

The clip is the window BEFORE and including the event frame: sealing never waits for later
frames, so an event is never held back by evidence.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

import cv2
import numpy as np

log = logging.getLogger("truewatch.edge.sealing")

JPEG_QUALITY = 80
MAX_FRAMES = 240          # hard bound on the ring, whatever the frame rate
HEARTBEAT_S = 5.0


class ClipSealer:
    def __init__(self, store, camera_id: str, clip_seconds: float = 4.0, max_frames: int = MAX_FRAMES) -> None:
        self.store = store
        self.camera_id = camera_id
        self.clip_seconds = max(0.5, float(clip_seconds))
        self._ring: deque[tuple[float, bytes]] = deque(maxlen=max(1, int(max_frames)))
        self._lock = threading.Lock()
        self.sealed = 0
        self.failed = 0
        self._beat_at = 0.0

    def add(self, image: np.ndarray, captured_at: float) -> None:
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return
        with self._lock:
            self._ring.append((float(captured_at), buf.tobytes()))
            while self._ring and self._ring[-1][0] - self._ring[0][0] > self.clip_seconds:
                self._ring.popleft()
        if captured_at - self._beat_at >= HEARTBEAT_S:
            self._beat_at = float(captured_at)
            self.store.heartbeat()    # keeps a later power-loss gap as short as the truth allows

    def seal(self) -> dict | None:
        """Seal the current window; return the event contract's `evidence` block, or None on failure."""
        with self._lock:
            frames = list(self._ring)
        if not frames:
            return None
        try:
            writer = self.store.open_clip(self.camera_id, frames[0][0])
            for _, data in frames:
                writer.write_frame(data)
            block = writer.seal(frames[-1][0])
        except Exception:  # evidence must never stop detection; the failure is counted and logged
            self.failed += 1
            log.exception("sealing a clip for camera %s failed", self.camera_id)
            return None
        self.sealed += 1
        return block.evidence_dict()

    def stats(self) -> dict:
        with self._lock:
            buffered = len(self._ring)
        return {"sealed": self.sealed, "failed": self.failed, "buffered_frames": buffered}
