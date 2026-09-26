"""The per-camera loop the edge service runs on every ingested frame.

`pipeline/demo.py` drives the same `Pipeline` (camera state, motion, appearance, tracking, fusion) over a
clip for inspection; this module drives it from the service's ingest loop and turns what fusion agrees
on into `truewatch.event.v1` payloads (pipeline/events.py). Only fusion-agreed tracks ever produce an
event: the contract sends no `agreed: false` events.

Rules (edge/rules/) are evaluated on EVERY frame, on the current snapshot of every track fusion has
agreed on at least once, so a fence crossed, a dwell reached or a wrong-way heading seen frames after the
fusion event is still caught ("rules run on fused events, never raw detections", ARCHITECTURE_V2
section 4). Each new (camera, track, rule) firing that clears the alert throttle becomes one event.
The per-camera direction baseline is re-learnt periodically from all confirmed tracks seen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.events import EventSink, build_event
from pipeline.rules_adapter import RuleTrack, dominant_direction, rule_tracks
from pipeline.runner import FrameResult, Pipeline, PipelineConfig
from pipeline.source import Frame
from pipeline.types import FusedEvent
from rules import baseline as baseline_mod
from rules import tracks as rtracks
from rules.baseline import CameraBaseline
from rules.config import CameraRuleConfig, ConfigStore, RuleConfigError
from rules.engine import evaluate_camera
from rules.throttle import AlertThrottle

BASELINE_EVERY_FRAMES = 100        # re-learn the direction baseline this often
BASELINE_MIN_TRACKS = 5            # ... once at least this many confirmed tracks have been seen
BASELINE_MAX_TRACKS = 500          # most recent tracks kept for learning

log = logging.getLogger("truewatch.edge.service")


@dataclass
class ServiceStats:
    frames: int = 0
    detector_passes: int = 0
    fused_events: int = 0
    events_emitted: int = 0
    camera_state_events: int = 0
    rule_evaluations: int = 0                               # frames on which the rule engine ran
    rule_fires: dict = field(default_factory=dict)          # rule name -> distinct (track, rule) firings
    rule_events: int = 0                                    # firings that cleared the throttle
    rule_suppressed: int = 0                                # firings the throttle refused (budget)
    last_timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frames": self.frames,
            "detector_passes": self.detector_passes,
            "fused_events": self.fused_events,
            "events_emitted": self.events_emitted,
            "camera_state_events": self.camera_state_events,
            "rule_evaluations": self.rule_evaluations,
            "rule_fires": dict(self.rule_fires),
            "rule_events": self.rule_events,
            "rule_suppressed": self.rule_suppressed,
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
        rules: CameraRuleConfig | ConfigStore | None = None,
        baseline: CameraBaseline | None = None,
        throttle: AlertThrottle | None = None,
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
        self.rules = rules
        self.baseline = baseline or CameraBaseline.empty(camera_id)
        self._baseline_fixed = baseline is not None
        self.throttle = throttle or AlertThrottle()
        self._epoch_offset: float | None = None
        self._seen_tracks: dict[int, RuleTrack] = {}      # for baseline learning
        self._fired: set[tuple[int, str]] = set()           # (track_id, rule) pairs already counted
        self._rules_error: str | None = None
        self.fired_by_track: dict[int, list[str]] = {}

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

    # ------------------------------------------------------------------ rules

    def rule_config(self) -> CameraRuleConfig | None:
        """This camera's rules; hot-reloaded when they come from a ConfigStore. None = rules off."""
        if self.rules is None or isinstance(self.rules, CameraRuleConfig):
            return self.rules
        try:
            cfg = self.rules.get(self.camera_id)
        except RuleConfigError as exc:
            if str(exc) != self._rules_error:
                log.error("camera %s: rules off: %s", self.camera_id, exc)
                self._rules_error = str(exc)
            return None
        self._rules_error = None
        return cfg

    def _learn_baseline(self, tracks: list[RuleTrack]) -> None:
        for t in tracks:
            self._seen_tracks[t.track_id] = t
        while len(self._seen_tracks) > BASELINE_MAX_TRACKS:
            self._seen_tracks.pop(next(iter(self._seen_tracks)))
        if (not self._baseline_fixed and self.stats.frames % BASELINE_EVERY_FRAMES == 0
                and len(self._seen_tracks) >= BASELINE_MIN_TRACKS):
            self.baseline = baseline_mod.learn(self.camera_id, list(self._seen_tracks.values()))

    def _baseline_block(self) -> dict | None:
        b = self.baseline
        if b.majority_heading_deg is None:
            return None
        return {"dominant_direction": dominant_direction(b.majority_heading_deg),
                "with": b.with_flow_count, "against": b.against_flow_count}

    def evaluate_rules(self, result: FrameResult, f: Frame) -> list[dict]:
        cfg = self.rule_config()
        all_tracks = rule_tracks(result.tracks, self._epoch_offset or 0.0)
        self._learn_baseline(all_tracks)
        if cfg is None:
            return []
        agreed = [t for t in all_tracks if t.track_id in self.last_fused]
        if not agreed:
            return []
        self.stats.rule_evaluations += 1
        verdicts = evaluate_camera(agreed, cfg, self.baseline)
        by_id = {t.track_id: t for t in agreed}
        events = []
        for verdict in verdicts:
            for rule in verdict.fired:
                key = (verdict.track_id, rule.rule_name)
                if key in self._fired:
                    self.throttle.offer(self.camera_id, verdict.track_id, rule.rule_name, f.captured_at)
                    continue
                self._fired.add(key)
                self.stats.rule_fires[rule.rule_name] = self.stats.rule_fires.get(rule.rule_name, 0) + 1
                self.fired_by_track.setdefault(verdict.track_id, []).append(rule.rule_name)
                alert = self.throttle.offer(self.camera_id, verdict.track_id, rule.rule_name, f.captured_at)
                if alert is None:
                    self.stats.rule_suppressed += 1
                    log.info("camera %s: %s on track %s suppressed by the alert budget", self.camera_id,
                             rule.rule_name, verdict.track_id)
                    continue
                track = by_id[verdict.track_id]
                metrics = {"dwell_s": round(rtracks.dwell_seconds(track), 2),
                           "net_px": round(rtracks.net_displacement_px(track), 1),
                           "path_px": round(rtracks.path_length_px(track), 1)}
                events.append(build_event(
                    self.last_fused[verdict.track_id], post_id=self.post_id, bbox=track.bbox,
                    captured_at=f.captured_at, rules_fired=[rule.rule_name], rule_baseline=self._baseline_block(),
                    rule_metrics=metrics,
                ))
                self.stats.rule_events += 1
        return events

    # ------------------------------------------------------------------ per frame

    def process(self, frame) -> tuple[FrameResult, list[dict]]:
        f = self.to_frame(frame)
        if self._epoch_offset is None:
            self._epoch_offset = f.captured_at - f.timestamp
        result = self.pipeline.process(f)
        self.stats.frames += 1
        self.stats.detector_passes += int(result.detections is not None)
        self.stats.camera_state_events += len(result.camera_events)
        self.stats.last_timings_ms = dict(result.timings_ms)
        events = []
        for ev in result.fusion.events:
            self.stats.fused_events += 1
            self.last_fused[ev.track_id] = ev
            events.append(build_event(ev, post_id=self.post_id,
                                      rules_fired=self.fired_by_track.get(ev.track_id, ())))
        events.extend(self.evaluate_rules(result, f))
        for event in events:
            self.sink.emit(event)
        self.stats.events_emitted += len(events)
        return result, events
