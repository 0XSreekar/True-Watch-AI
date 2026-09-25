"""One camera's per-frame pipeline: camera state -> motion -> appearance -> tracking -> calibration -> fusion.

Frame-skipping strategy
-----------------------
The motion channel runs on EVERY frame; the appearance channel runs on every `stride`-th frame
(default 2) and ByteTrack's Kalman filter carries each track's box across the skipped frames.

  - Motion must see consecutive frames. Farneback estimates displacement between the two frames it is
    given; the pyramid copes with a few pixels, not with a gap two or three times larger, and the
    section 1 contrast was measured on consecutive 20 fps frames. Skipping frames for motion would change
    the quantity being measured. It is also the cheap channel (a 640 px pair costs a fraction of one
    detector pass on the same CPU; the demo prints both).
  - Appearance is the expensive channel (one YOLO11-s pass dominates the frame budget) and its output
    changes slowly: a walking person moves a few pixels per frame, well inside what a constant-velocity
    Kalman prediction bridges over one or two frames. ByteTrack only needs boxes often enough to keep
    IoU association unambiguous.
  - Fusion treats an appearance score as fresh for `stride` frames (FusionConfig.appearance_max_age), so
    the temporal gate counts motion frames, and n_consecutive = 3 at stride 2 still spans two
    independent detector passes.
When a camera-state anomaly is active the detector is not run at all: its output would not be used.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pipeline.appearance import AppearanceChannel
from pipeline.calibration import DEFAULT_WARMUP_FRAMES, CalibrationStore, CameraCalibration, Calibrator
from pipeline.camera_state import CameraStateMonitor
from pipeline.fusion import Fusion, FusionConfig, FusionFrame
from pipeline.motion import CELL_PX, MotionChannel, MotionField
from pipeline.source import Frame
from pipeline.tracking import ByteTracker, TrackerConfig
from pipeline.types import CameraStateEvent, Detection, Track

log = logging.getLogger("truewatch.edge.runner")

MOTION_FIRE_AREA = 0.001          # a motion region covering >= 0.1% of the frame counts as "motion fired"


def detect_modality(image: np.ndarray, tolerance: float = 1.5) -> str:
    """'lwir' when the three colour channels are (near) identical, as thermal video is stored; else 'visible'.

    A monochrome visible camera would be misread as LWIR, so a deployment sets the modality explicitly;
    this is the fallback for a clip of unknown origin.
    """
    if image.ndim == 2 or image.shape[2] == 1:
        return "lwir"
    small = image[::8, ::8].astype(np.int16)
    spread = float(np.abs(small[..., 0] - small[..., 1]).mean() + np.abs(small[..., 1] - small[..., 2]).mean())
    return "lwir" if spread < tolerance else "visible"


@dataclass
class PipelineConfig:
    camera_id: str = "CAM-01"
    modality: str = "auto"                     # 'auto' | 'visible' | 'lwir'
    stride: int = 2
    warmup_frames: int = DEFAULT_WARMUP_FRAMES
    calibration_dir: Path = Path("var/calibration")
    recalibrate: bool = False
    source: str = "file"
    parallel: bool = True                      # run the detector concurrently with the flow of the same frame
    fusion: FusionConfig = field(default_factory=FusionConfig.from_env)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


@dataclass
class FrameResult:
    frame: Frame
    modality: str
    detections: list[Detection] | None         # None on frames the detector skipped
    tracks: list[Track]
    motion: MotionField | None
    fusion: FusionFrame
    camera_events: list[CameraStateEvent]
    appearance_fired: bool
    motion_fired: bool
    fused: bool
    timings_ms: dict
    calibration_saved: Path | None = None


class Pipeline:
    def __init__(self, appearance: AppearanceChannel | None, config: PipelineConfig | None = None) -> None:
        self.appearance = appearance
        self.config = config or PipelineConfig()
        c = self.config
        self.store = CalibrationStore(c.calibration_dir)
        if c.recalibrate:
            removed = self.store.reset(c.camera_id)
            if removed:
                log.info("calibration reset for %s: removed %s", c.camera_id, [p.name for p in removed])
        self.motion = MotionChannel()
        self.camera_state = CameraStateMonitor(c.camera_id, source=c.source)
        self.modality: str | None = None if c.modality == "auto" else c.modality
        self.tracker: ByteTracker | None = None
        self.fusion: Fusion | None = None
        self.calibration: CameraCalibration | None = None
        self.calibrator: Calibrator | None = None
        self._stride = max(1, int(c.stride))
        # onnxruntime and OpenCV both release the GIL, so the detector pass and the flow of the same frame
        # can overlap on separate cores; a frame with a detector pass then costs max(flow, detector)
        # rather than their sum. The two stages share nothing until tracking.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="appearance") if c.parallel else None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # ------------------------------------------------------------------ setup on the first frame

    def _init_for(self, frame: Frame) -> None:
        c = self.config
        if self.modality is None:
            self.modality = detect_modality(frame.image)
            log.info("camera %s: modality detected as %s", c.camera_id, self.modality)
        fcfg = c.fusion
        if fcfg.appearance_max_age != self._stride:
            fcfg = FusionConfig(**{**fcfg.__dict__, "appearance_max_age": self._stride})
        self.fusion = Fusion(c.camera_id, self.modality, fcfg, source=frame.source)
        self.tracker = ByteTracker(c.tracker, camera_id=c.camera_id, modality=self.modality)
        wsize = self.motion.prepare(frame.image)[0].shape[::-1]
        self.calibration = self.store.load(c.camera_id, self.modality, wsize)
        if self.calibration is not None:
            log.info("camera %s: using stored calibration %s", c.camera_id, self.store.path(c.camera_id, self.modality))
            self.camera_state.set_baseline(self.calibration.scene_brightness, self.calibration.scene_std,
                                           self.calibration.scene_detail)
        else:
            self.calibrator = Calibrator(c.camera_id, self.modality, fcfg.threshold, c.warmup_frames)
            log.info("camera %s: learning thresholds over %d warm-up frames", c.camera_id, c.warmup_frames)

    def reset_calibration(self) -> None:
        """Forget this camera's thresholds and learn them again from the next frames."""
        self.store.reset(self.config.camera_id)
        self.calibration = None
        if self.modality is not None and self.fusion is not None:
            self.calibrator = Calibrator(self.config.camera_id, self.modality, self.fusion.config.threshold,
                                         self.config.warmup_frames)

    # ------------------------------------------------------------------ per frame

    def process(self, frame: Frame) -> FrameResult:
        if self.fusion is None:
            self._init_for(frame)
        assert self.fusion is not None and self.tracker is not None
        t = {}
        t0 = time.perf_counter()
        cam_events = self.camera_state.update(frame.image, frame_index=frame.index, timestamp=frame.timestamp,
                                              captured_at=frame.captured_at, gap_s=frame.gap_s,
                                              reconnects=frame.reconnects)
        blocked = self.camera_state.blocked
        t["camera_state"] = (time.perf_counter() - t0) * 1000

        detections = None
        future = None
        run_detector = self.appearance is not None and not blocked and frame.index % self._stride == 0
        kwargs = dict(camera_id=self.config.camera_id, frame_index=frame.index, timestamp=frame.timestamp,
                      modality=self.modality)
        if run_detector and self._pool is not None:
            future = self._pool.submit(self.appearance.detect, frame.image, **kwargs)

        t1 = time.perf_counter()
        tau = self.calibration.tau_cells() if self.calibration is not None else None
        field_ = self.motion.update(frame.image, frame.timestamp, tau)
        t["motion"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        if future is not None:
            detections = future.result()
        elif run_detector:
            detections = self.appearance.detect(frame.image, **kwargs)
        t["appearance_wait"] = (time.perf_counter() - t2) * 1000
        t["appearance"] = self.appearance.last_ms if run_detector else 0.0

        t3 = time.perf_counter()
        if detections is not None:
            tracks = self.tracker.update(detections, frame.index, frame.timestamp)
        else:
            tracks = self.tracker.predict_only(frame.index, frame.timestamp)
        t["tracking"] = (time.perf_counter() - t3) * 1000

        saved = None
        if self.calibration is None and self.calibrator is not None and field_ is not None and not blocked:
            boxes = [field_.to_working(b.bbox) for b in (detections or [])] + [field_.to_working(tr.bbox) for tr in tracks]
            self.calibrator.observe(field_.cell_energy, field_.mag.shape[::-1], CELL_PX, boxes,
                                    self.camera_state.last_stats)
            if self.calibrator.ready:
                self.calibration = self.calibrator.finalize()
                saved = self.store.save(self.calibration)
                self.camera_state.set_baseline(self.calibration.scene_brightness, self.calibration.scene_std,
                                               self.calibration.scene_detail)
                self.calibrator = None
                log.info("camera %s: calibration learnt from %d frames, written to %s",
                         self.config.camera_id, self.calibration.frames, saved)

        t4 = time.perf_counter()
        status = "camera_state" if blocked else ("ok" if self.calibration is not None else "warming_up")
        appearance_of = {}
        for tr in tracks:
            got = self.tracker.last_score(tr.track_id)
            if got is not None:
                appearance_of[tr.track_id] = got
        h, w = frame.image.shape[:2]
        fused = self.fusion.evaluate(
            tracks, field_, appearance_of=appearance_of, frame_index=frame.index, timestamp=frame.timestamp,
            captured_at=frame.captured_at, frame_size=(w, h),
            bg_floor=self.calibration.bg_floor if self.calibration else 0.05,
            threshold=self.calibration.threshold if self.calibration else None, status=status,
        )
        t["fusion"] = (time.perf_counter() - t4) * 1000
        t["total"] = (time.perf_counter() - t0) * 1000

        theta = self.calibration.threshold if self.calibration else self.fusion.config.threshold
        appearance_fired = any(d.appearance.score >= theta for d in fused.decisions)
        motion_fired = bool(field_ is not None and status == "ok" and any(
            r.area_px >= MOTION_FIRE_AREA * w * h for r in field_.regions))
        return FrameResult(frame=frame, modality=self.modality, detections=detections, tracks=tracks, motion=field_,
                           fusion=fused, camera_events=cam_events, appearance_fired=appearance_fired,
                           motion_fired=motion_fired, fused=any(d.agree for d in fused.decisions), timings_ms=t,
                           calibration_saved=saved)
