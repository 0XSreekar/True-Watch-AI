"""Frame sources for the pipeline: an RTSP URL, a local video file, an image folder or a webcam.

Every source yields `Frame` objects with a MONOTONIC `timestamp` in seconds, which is the only clock
tracking, fusion and calibration do arithmetic on:

  file / folder   media time, index / fps. A replay runs faster than real time in the demo, and dwell
                  times must still be the clip's seconds, not the laptop's.
  rtsp / webcam   time.monotonic() at decode. Never time.time(): a wall-clock step (NTP) must not
                  produce a negative dwell.
In both cases the timestamp is forced non-decreasing, so a decoder that reports a stale PTS cannot
move time backwards.

`Frame.source` is the honesty flag of docs/PHASE_MINUS1_SCOPE.md section 6: 'rtsp' for a live pull,
'file' for a replayed file or image folder (a folder is a replay too), 'webcam' for a development
camera. A replay is never labelled as live.

RTSP reconnect
--------------
A live pull that stops delivering frames (read failure, or the read timeout) is released and reopened
with exponential backoff, 0.5 s doubling to 30 s, forever unless `max_reconnects` is set. Each frame
carries the number of reconnects so far and the length of the gap before it, so camera_state.py can
raise SIGNAL_LOSS for the outage instead of the pipeline silently resuming.

Demo an "existing IP camera" without a camera (local RTSP loopback)
--------------------------------------------------------------------
Two free tools; nothing is installed by this repository.

  1. Start an RTSP server. mediamtx (MIT licence, single binary):
         brew install mediamtx          # or download a release from github.com/bluenviron/mediamtx
         mediamtx                       # listens on rtsp://localhost:8554
  2. Publish a clip to it in a loop, at the clip's own rate (-re), without re-encoding:
         ffmpeg -re -stream_loop -1 -i samples/day.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1
     (if the clip's codec is not RTSP-friendly, re-encode: -c:v libx264 -preset veryfast -tune zerolatency)
  3. Point the pipeline at it:
         cd edge && python -m pipeline.demo --source rtsp://localhost:8554/cam1 --show-scores
  4. Kill the ffmpeg process for a few seconds and start it again: the source logs the drop,
     reconnects with backoff, and camera_state.py raises SIGNAL_LOSS for the gap.
A frame pulled this way is labelled 'rtsp' because it genuinely crossed an RTSP session; the demo
must still say out loud that the camera is a loopback of a recorded clip.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

log = logging.getLogger("truewatch.edge.source")

RTSP_OPEN_TIMEOUT_MS = 5000
RTSP_READ_TIMEOUT_MS = 5000
BACKOFF_START_S = 0.5
BACKOFF_MAX_S = 30.0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_FOLDER_FPS = 20.0          # KAIST Multispectral is recorded at 20 fps


class SourceError(RuntimeError):
    """The source could not be opened, or stopped for good."""


@dataclass(frozen=True)
class Frame:
    index: int                     # 0-based count of frames this source has emitted
    image: np.ndarray              # BGR uint8, H x W x 3
    timestamp: float               # monotonic seconds, see module docstring
    captured_at: float             # wall clock time.time(), display only
    source: str                    # 'rtsp' | 'file' | 'webcam'
    camera_id: str
    reconnects: int = 0            # reconnects so far (live sources)
    gap_s: float = 0.0             # seconds without frames immediately before this one

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def classify(spec: str) -> str:
    """'rtsp', 'webcam', 'folder' or 'file' for a source string."""
    s = str(spec)
    if s.lower().startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")):
        return "rtsp"
    if s.isdigit() or s.lower().startswith("webcam"):
        return "webcam"
    if Path(s).is_dir():
        return "folder"
    return "file"


class FrameSource:
    """Iterate frames from one source. Use `open_source` rather than constructing this directly."""

    def __init__(
        self,
        spec: str,
        *,
        camera_id: str = "CAM-01",
        folder_fps: float = DEFAULT_FOLDER_FPS,
        max_frames: int | None = None,
        loop: bool = False,
        resize_to: tuple[int, int] | None = None,
        max_reconnects: int | None = None,
        stop: threading.Event | None = None,
        capture_factory: Callable[..., cv2.VideoCapture] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.spec = str(spec)
        self.kind = classify(self.spec)
        self.camera_id = camera_id
        self.folder_fps = folder_fps
        self.max_frames = max_frames
        self.loop = loop
        self.resize_to = resize_to
        self.max_reconnects = max_reconnects
        self.stop = stop or threading.Event()
        self._factory = capture_factory or cv2.VideoCapture
        self._sleep = sleep
        self._clock = clock
        self.fps: float | None = None
        self.reconnects = 0

    @property
    def label(self) -> str:
        """The honesty flag carried on every frame and event."""
        return {"rtsp": "rtsp", "webcam": "webcam"}.get(self.kind, "file")

    # ------------------------------------------------------------------ openers

    def _open_capture(self) -> cv2.VideoCapture:
        if self.kind == "rtsp":
            cap = self._factory(self.spec, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # a stale live frame is worse than a dropped one
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS)
        elif self.kind == "webcam":
            digits = re.sub(r"\D", "", self.spec) or "0"
            cap = self._factory(int(digits))
        else:
            cap = self._factory(self.spec)
        return cap

    def _prep(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if self.resize_to is not None and (image.shape[1], image.shape[0]) != self.resize_to:
            image = cv2.resize(image, self.resize_to, interpolation=cv2.INTER_LINEAR)
        return image

    # ------------------------------------------------------------------ iteration

    def __iter__(self) -> Iterator[Frame]:
        if self.kind == "folder":
            yield from self._iter_folder()
        elif self.kind == "file":
            yield from self._iter_file()
        else:
            yield from self._iter_live()

    def _iter_folder(self) -> Iterator[Frame]:
        files = sorted((p for p in Path(self.spec).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key)
        if not files:
            raise SourceError(f"no images in folder {self.spec}")
        self.fps = self.folder_fps
        emitted = 0
        while True:
            for path in files:
                if self.stop.is_set() or (self.max_frames is not None and emitted >= self.max_frames):
                    return
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    log.warning("unreadable image skipped: %s", path)
                    continue
                yield Frame(emitted, self._prep(image), emitted / self.folder_fps, time.time(), "file", self.camera_id)
                emitted += 1
            if not self.loop:
                return

    def _iter_file(self) -> Iterator[Frame]:
        if not Path(self.spec).is_file():
            raise SourceError(f"no such file: {self.spec}")
        cap = self._open_capture()
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"could not open {self.spec}")
        declared = cap.get(cv2.CAP_PROP_FPS)
        self.fps = declared if declared and 0 < declared < 1000 else 25.0
        emitted, last_ts, base = 0, -1.0, 0.0
        try:
            while True:
                if self.stop.is_set() or (self.max_frames is not None and emitted >= self.max_frames):
                    return
                ok, image = cap.read()
                if not ok:
                    if self.loop and emitted:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        base = last_ts + 1.0 / self.fps   # keep the clock monotonic across the loop point
                        continue
                    if emitted == 0:
                        raise SourceError(f"{self.spec} opened but produced no frames")
                    return
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = base + (pos_ms / 1000.0 if pos_ms and pos_ms > 0 else emitted / self.fps)
                ts = max(ts, last_ts + 1e-6) if last_ts >= 0 else ts
                last_ts = ts
                yield Frame(emitted, self._prep(image), ts, time.time(), "file", self.camera_id)
                emitted += 1
        finally:
            cap.release()

    def _wait(self, seconds: float) -> bool:
        """Sleep for a backoff interval; True when a stop was requested meanwhile."""
        if self._sleep is time.sleep:
            return self.stop.wait(seconds)
        self._sleep(seconds)
        return self.stop.is_set()

    def _iter_live(self) -> Iterator[Frame]:
        emitted, last_ts = 0, -1.0
        backoff = BACKOFF_START_S
        cap = None
        down_since: float | None = None      # set while the source is down
        try:
            while not self.stop.is_set():
                if self.max_frames is not None and emitted >= self.max_frames:
                    return
                if cap is None:
                    if down_since is not None:   # this open is a reconnect, not the first open
                        if self.max_reconnects is not None and self.reconnects >= self.max_reconnects:
                            raise SourceError(f"{self.spec}: gave up after {self.reconnects} reconnect attempts")
                        self.reconnects += 1
                    cap = self._open_capture()
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        if down_since is None:
                            if emitted == 0 and self.max_reconnects == 0:
                                raise SourceError(f"could not open {self.spec}")
                            down_since = self._clock()
                        log.warning("source %s unavailable; next attempt in %.1f s", self.spec, backoff)
                        if self._wait(backoff):
                            return
                        backoff = min(backoff * 2.0, BACKOFF_MAX_S)
                        continue
                    declared = cap.get(cv2.CAP_PROP_FPS)
                    self.fps = declared if declared and 0 < declared < 1000 else None
                ok, image = cap.read()
                if not ok or image is None:
                    log.warning("source %s dropped after %d frames; reconnecting", self.spec, emitted)
                    cap.release()
                    cap = None
                    if down_since is None:
                        down_since = self._clock()
                    continue
                now = self._clock()
                gap = 0.0
                if down_since is not None:
                    gap = now - down_since
                    log.info("source %s back after %.1f s (%d reconnects)", self.spec, gap, self.reconnects)
                    down_since = None
                    backoff = BACKOFF_START_S
                ts = max(now, last_ts + 1e-6) if last_ts >= 0 else now
                last_ts = ts
                yield Frame(emitted, self._prep(image), ts, time.time(), self.label, self.camera_id,
                            reconnects=self.reconnects, gap_s=gap)
                emitted += 1
        finally:
            if cap is not None:
                cap.release()


def open_source(spec: str, **kwargs) -> FrameSource:
    """A FrameSource for `spec`: rtsp://..., a video path, an image folder, or 'webcam'/'0'."""
    return FrameSource(spec, **kwargs)
