"""Step 1 of slide 3: 'Existing IP camera — no new hardware on the pole'.

One decode path serves both sources. A live ONVIF/RTSP pull and a replayed file
differ only in the string handed to OpenCV, so the substitution documented in
docs/PHASE_MINUS1_SCOPE.md section 6 is a configuration change, not a code path.

Every frame carries the source label ('rtsp' or 'file') so nothing downstream
can present a replayed file as a live camera.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

# An RTSP pull that stalls should fail rather than hang the pipeline.
RTSP_OPEN_TIMEOUT_MS = 5000
RTSP_READ_TIMEOUT_MS = 5000


class IngestError(RuntimeError):
    """The source could not be opened or produced no frames."""


@dataclass(frozen=True)
class Frame:
    """One decoded frame and everything downstream needs to judge it."""

    index: int
    image: np.ndarray
    captured_at: float  # time.time() at decode
    monotonic: float  # time.monotonic() at decode, for latency arithmetic
    source: str  # 'rtsp' | 'file' — the honesty flag
    camera_id: str

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


def _open(source: str | Path, is_rtsp: bool) -> cv2.VideoCapture:
    if is_rtsp:
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        # Keep the buffer at one frame: on a live pull, a stale frame is worse
        # than a dropped one.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS)
    else:
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise IngestError(f"could not open source: {source}")
    return cap


def _decimation(cap: cv2.VideoCapture, target_fps: float) -> int:
    """How many decoded frames to advance per emitted frame, for a file.

    A file has its own timeline. Sampling it down to target_fps means keeping
    every Nth frame, not dropping by wall-clock — a fast machine would otherwise
    read a short clip to its end inside one target interval and emit almost
    nothing. When the file is slower than the target, every frame is kept.
    """
    declared = cap.get(cv2.CAP_PROP_FPS)
    if not declared or declared <= 0 or declared <= target_fps:
        return 1
    return max(1, round(declared / target_fps))


def frames(
    source: str | Path,
    *,
    is_rtsp: bool,
    camera_id: str,
    target_fps: float = 8.0,
    loop: bool = False,
    pace: bool = False,
    stop: threading.Event | None = None,
    max_frames: int | None = None,
) -> Iterator[Frame]:
    """Yield decoded frames from either source, reduced to target_fps.

    The two sources need different reductions, because they have different
    clocks:

    * A **file** is decimated against its own declared frame rate — keep every
      Nth frame. Its timeline is fixed, so dropping by wall-clock would empty a
      short clip on a fast machine.
    * A **live pull** is paced by the camera. A frame arriving sooner than the
      target interval is dropped rather than queued, because on a live source a
      stale frame is worse than a missing one.

    `pace=True` additionally sleeps so emissions are spaced at 1/target_fps,
    which is what a replay needs to look like a live feed. Tests leave it off.
    """
    if target_fps <= 0:
        raise IngestError(f"target_fps must be positive, got {target_fps}")

    label = "rtsp" if is_rtsp else "file"
    interval = 1.0 / target_fps
    emitted = 0
    read = 0
    last_emit = 0.0

    cap = _open(source, is_rtsp)
    step = 1 if is_rtsp else _decimation(cap, target_fps)
    if step > 1:
        log.info("decimating %s: keeping every %dth frame for %.1f fps", source, step, target_fps)
    try:
        while True:
            if stop is not None and stop.is_set():
                log.info("ingest stopped after %d frames", emitted)
                return
            if max_frames is not None and emitted >= max_frames:
                return

            ok, image = cap.read()
            if not ok:
                if not is_rtsp and loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    read = 0
                    continue
                if emitted == 0:
                    raise IngestError(f"source opened but produced no frames: {source}")
                log.info("source exhausted after %d frames", emitted)
                return

            now = time.monotonic()
            if is_rtsp:
                # Live: drop anything arriving inside the target interval.
                if last_emit and (now - last_emit) < interval:
                    continue
            else:
                # File: keep every step-th decoded frame.
                keep = read % step == 0
                read += 1
                if not keep:
                    continue
                if pace and last_emit:
                    delay = interval - (now - last_emit)
                    if delay > 0:
                        if stop is not None and stop.wait(delay):
                            return
                        elif stop is None:
                            time.sleep(delay)
                        now = time.monotonic()
            last_emit = now

            yield Frame(
                index=emitted,
                image=image,
                captured_at=time.time(),
                monotonic=now,
                source=label,
                camera_id=camera_id,
            )
            emitted += 1
    finally:
        cap.release()


def probe(source: str | Path, *, is_rtsp: bool) -> dict:
    """Open the source, read one frame, report what it is. Used by /healthz."""
    cap = _open(source, is_rtsp)
    try:
        ok, image = cap.read()
        if not ok:
            raise IngestError(f"source opened but produced no frames: {source}")
        h, w = image.shape[:2]
        declared = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return {
            "source": "rtsp" if is_rtsp else "file",
            "width": int(w),
            "height": int(h),
            "declared_fps": round(declared, 3) if declared and declared > 0 else None,
            "frame_count": int(count) if count and count > 0 else None,
        }
    finally:
        cap.release()
