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
    # A file's own clock: seconds into the recording (decoded frames / declared fps, continuing across
    # loops). None on a live pull, whose clock is `monotonic`. The pipeline measures flow per second
    # of scene time, so a file must be judged on its media time, not on how fast this machine decodes.
    media_time: float | None = None

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


def _file_fps(cap: cv2.VideoCapture) -> float | None:
    declared = cap.get(cv2.CAP_PROP_FPS)
    return float(declared) if declared and 0 < declared < 1000 else None


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

    * A **file** is decimated on its own timeline (decoded frames / declared
      fps): a frame is kept when its media time reaches the next 1/target_fps
      slot, so 20 fps sampled to 8 fps really is 8 fps (alternating gaps of 2
      and 3 source frames), not round(20/8) = every 2nd frame = 10 fps.
      Dropping by wall-clock would empty a short clip on a fast machine. Each
      frame carries its `media_time`; when the file is slower than the target,
      every frame is kept.
    * A **live pull** is paced by the camera. A frame arriving sooner than the
      target interval is dropped rather than queued, because on a live source a
      stale frame is worse than a missing one.

    `pace=True` additionally sleeps so emissions are spaced by the media time
    between them, which is what a replay needs to look like a live feed. Tests
    leave it off.
    """
    if target_fps <= 0:
        raise IngestError(f"target_fps must be positive, got {target_fps}")

    label = "rtsp" if is_rtsp else "file"
    interval = 1.0 / target_fps
    emitted = 0
    read = 0
    last_emit = 0.0

    cap = _open(source, is_rtsp)
    fps = None if is_rtsp else _file_fps(cap)
    media_dt = 1.0 / fps if fps else interval
    loop_offset = 0.0
    next_slot = 0.0
    last_media: float | None = None
    if fps and fps > target_fps * 1.01:
        log.info("decimating %s from %.2f to %.2f fps on its own timeline", source, fps, target_fps)
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
                    loop_offset += read * media_dt
                    read = 0
                    continue
                if emitted == 0:
                    raise IngestError(f"source opened but produced no frames: {source}")
                log.info("source exhausted after %d frames", emitted)
                return

            now = time.monotonic()
            media_time = None
            if is_rtsp:
                # Live: drop anything arriving inside the target interval.
                if last_emit and (now - last_emit) < interval:
                    continue
            else:
                # File: keep the frame that reaches the next 1/target_fps slot of media time.
                media_time = loop_offset + read * media_dt
                read += 1
                if media_time + 1e-6 < next_slot:
                    continue
                next_slot = max(next_slot + interval, media_time + interval / 2)
                if pace and last_emit and last_media is not None:
                    delay = (media_time - last_media) - (now - last_emit)
                    if delay > 0:
                        if stop is not None and stop.wait(delay):
                            return
                        elif stop is None:
                            time.sleep(delay)
                        now = time.monotonic()
            last_emit = now
            last_media = media_time

            yield Frame(
                index=emitted,
                image=image,
                captured_at=time.time(),
                monotonic=now,
                source=label,
                camera_id=camera_id,
                media_time=media_time,
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
