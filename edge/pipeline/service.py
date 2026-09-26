"""The per-camera loop the edge service runs on every ingested frame.

`pipeline/demo.py` drives the same `Pipeline` (camera state, motion, appearance, tracking, fusion) over a
clip for inspection; this module drives it from the service's ingest loop and turns what fusion agrees
on into `truewatch.event.v1` payloads (pipeline/events.py). Only fusion-agreed tracks ever produce an
event: the contract sends no `agreed: false` events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.events import EventSink, build_event
from pipeline.runner import FrameResult, Pipeline, PipelineConfig
from pipeline.source import Frame
from pipeline.types import FusedEvent

log = logging.getLogger("truewatch.edge.service")


@dataclass
class ServiceStats:
    frames: int = 0
    detector_passes: int = 0
    fused_events: int = 0
    events_emitted: int = 0
    camera_state_events: int = 0
    last_timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frames": self.frames,
            "detector_passes": self.detector_passes,
            "fused_events": self.fused_events,
            "events_emitted": self.events_emitted,
            "camera_state_events": self.camera_state_events,
            "last_timings_ms": {k: round(v, 2) for k, v in self.last_timings_ms.items()},
        }


class CameraService:
    """One camera: frames in, provisional events out."""

    def __init__(
        self,
        appearance,
        *,
        camera_id: str,
        post_id: str,
        sink: EventSink,
        stride: int = 2,
        calibration_dir: Path = Path("var/calibration"),
        warmup_frames: int | None = None,
        source: str = "file",
    ) -> None:
        kwargs = dict(camera_id=camera_id, stride=max(1, int(stride)), calibration_dir=Path(calibration_dir),
                      source=source)
        if warmup_frames is not None:
            kwargs["warmup_frames"] = int(warmup_frames)
        self.pipeline = Pipeline(appearance, PipelineConfig(**kwargs))
        self.camera_id = camera_id
        self.post_id = post_id
        self.sink = sink
        self.stats = ServiceStats()
        self._t0: float | None = None
        self.last_fused: dict[int, FusedEvent] = {}   # track_id -> latest fusion agreement

    def close(self) -> None:
        self.pipeline.close()

    def to_frame(self, frame) -> Frame:
        """Adapt a pipeline.ingest frame (monotonic + captured_at) to the pipeline's Frame."""
        if isinstance(frame, Frame):
            return frame
        if self._t0 is None:
            self._t0 = frame.monotonic
        return Frame(index=frame.index, image=frame.image, timestamp=frame.monotonic - self._t0,
                     captured_at=frame.captured_at, source=frame.source, camera_id=self.camera_id)

    def process(self, frame) -> tuple[FrameResult, list[dict]]:
        f = self.to_frame(frame)
        result = self.pipeline.process(f)
        self.stats.frames += 1
        self.stats.detector_passes += int(result.detections is not None)
        self.stats.camera_state_events += len(result.camera_events)
        self.stats.last_timings_ms = dict(result.timings_ms)
        events = []
        for ev in result.fusion.events:
            self.stats.fused_events += 1
            self.last_fused[ev.track_id] = ev
            events.append(build_event(ev, post_id=self.post_id))
        for event in events:
            self.sink.emit(event)
        self.stats.events_emitted += len(events)
        return result, events
