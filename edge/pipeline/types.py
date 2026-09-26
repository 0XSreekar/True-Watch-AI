"""The pipeline's data contract. Phases 4, 5 and 6 build on these types; change them additively only.

Every type is a frozen dataclass: a value produced by one stage cannot be mutated by the next, so a
FusedEvent always describes exactly the evidence it was decided on.

Conventions that hold for every field below
-------------------------------------------
bbox        (x1, y1, x2, y2) in PIXELS of the ORIGINAL decoded frame, origin top-left, x1 < x2,
            y1 < y2, floats. Never normalised here; normalisation to the wire format of
            docs/ARCHITECTURE_V2.md section 4.2 ([x, y, w, h] in 0..1) happens once, in
            `FusedEvent.bbox_normalised`, at the edge of the process.
class_name  one of the schema names in datasets/config/schema.yaml: 'person', 'two_wheeler', 'car',
            'truck', 'cart'. Model-specific names (COCO 'bicycle', 'bus', ...) never leave
            appearance.py.
score       float in [0, 1].
timestamp   seconds on a MONOTONIC clock (`Frame.timestamp` in source.py): media time for a file or
            image folder, time.monotonic() for a live pull. Durations (dwell, streaks, velocity) use
            this clock only. `captured_at` is wall-clock time.time(), for display and the event
            record, never for arithmetic.
camera_id   the configured camera id (CAMERA_ID), e.g. 'RXL-01'.
modality    'visible' or 'lwir' (long-wave infrared). Decides the fusion weights (fusion.py).
source      the honesty flag of docs/PHASE_MINUS1_SCOPE.md section 6: 'rtsp' for a live pull,
            'file' for a replayed file or image folder, 'webcam' for a local development camera.

The types
---------
Detection           one appearance-channel box on one frame (appearance.py).
MotionRegion        one connected region of the motion mask on one frame (motion.py).
ChannelScore        one channel's verdict on one candidate: the [0, 1] score fusion consumes, the raw
                    quantity it came from, and a short human-readable basis.
Track               a snapshot of one ByteTrack track, with the loitering signals of
                    docs/MEASUREMENTS.md section 2: dwell, net displacement and path length. Satisfies
                    the rule engine's TrackLike protocol (track_id, camera_id, class_name, history of
                    (timestamp_s, x, y) ground points); note `bbox` here is xyxy, not xywh.
FrameRef            a pointer to one decoded frame (the "best frame" of an event).
FusedEvent          an alert candidate on which BOTH channels agreed, past every gate (fusion.py).
                    Satisfies FusedEventLike (track, camera_id, timestamp_s).
CameraStateEvent    a CAMERA_STATE anomaly: signal loss, lens tampering or IR floodlight interference
                    (camera_state.py). Raised, never discarded as darkness (slide 4).
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

BBox = tuple[float, float, float, float]
Modality = Literal["visible", "lwir"]
SourceLabel = Literal["rtsp", "file", "webcam"]
CameraStateKind = Literal["SIGNAL_LOSS", "LENS_TAMPER", "IR_FLOOD"]

SCHEMA_CLASSES: tuple[str, ...] = ("person", "two_wheeler", "car", "truck", "cart")


def bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_center(b: BBox) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


@dataclass(frozen=True)
class Detection:
    """One appearance-channel box. `score` is the detector confidence after the class map."""

    bbox: BBox
    class_name: str
    score: float
    camera_id: str
    frame_index: int
    timestamp: float
    modality: Modality
    class_id: int = -1               # index of class_name in SCHEMA_CLASSES
    tiled: bool = False              # True when the box came from a far-field tile (appearance.py)


@dataclass(frozen=True)
class MotionRegion:
    """One connected region of the motion mask, in original-frame pixels."""

    bbox: BBox
    energy: float                    # mean flow magnitude inside the region, px/frame at working scale
    peak: float                      # maximum flow magnitude inside the region
    area_px: float                   # region area in original-frame pixels


@dataclass(frozen=True)
class ChannelScore:
    """One channel's judgement of one candidate box.

    channel  'appearance' or 'motion'
    score    in [0, 1]; the number fusion weighs
    raw      the quantity the score came from (detector confidence; motion contrast ratio)
    basis    a short string saying how `score` was derived, for logs and the operator
    """

    channel: Literal["appearance", "motion"]
    score: float
    raw: float
    basis: str = ""


@dataclass(frozen=True)
class Track:
    """An immutable snapshot of one track at `last_seen`.

    Loitering signals follow docs/MEASUREMENTS.md section 2 exactly:
      frames   number of frames on which a detection was associated to the track
      dwell_s  time elapsed from the first to the last associated frame, in seconds
               ((last_frame - first_frame) frame periods; 33 contiguous frames at 20 fps = 1.6 s, as in
               the section 2 table)
      net_px   straight-line distance between the first and last observed box centres
      path_px  sum of the distances between consecutive observed box centres
    A person pacing in place has a small net_px and a much larger path_px (track 6 in section 2).
    """

    track_id: int
    class_name: str
    bbox: BBox
    score: float
    camera_id: str
    modality: Modality
    state: Literal["tentative", "confirmed", "lost"]
    first_frame: int
    last_frame: int
    first_seen: float
    last_seen: float
    frames: int
    dwell_s: float
    net_px: float
    path_px: float
    velocity: tuple[float, float]    # px per second, from the Kalman state
    # (timestamp_s, x_px, y_px) ground-contact points (bottom-centre of each observed box), oldest first:
    # the shape the rule engine's TrackLike protocol (edge/rules/tracks.py) reads.
    history: tuple[tuple[float, float, float], ...] = ()
    # (frame_index, timestamp_s, cx, cy) box CENTRES of observed frames, oldest first: what the section 2
    # signals above (net_px, path_px) were computed from, matching how MEASUREMENTS.md measured them.
    observations: tuple[tuple[int, float, float, float], ...] = ()
    observed_now: bool = True        # False when `bbox` is a Kalman prediction on a frame with no detection


@dataclass(frozen=True)
class FrameRef:
    """Which frame an event is best seen on. Evidence (Phase 9) resolves it to pixels."""

    camera_id: str
    frame_index: int
    timestamp: float
    captured_at: float
    bbox: BBox
    fused_score: float


@dataclass(frozen=True)
class FusedEvent:
    """An agreed detection: both channels vouched, past the spatial gate, the per-camera threshold and
    the temporal gate (fusion.py documents the rule). Only agreed events exist; a disagreement is logged
    per frame, never emitted as an event.
    """

    camera_id: str
    track_id: int
    class_name: str
    bbox: BBox
    frame_size: tuple[int, int]      # (width, height) of the frame bbox is in
    appearance: ChannelScore
    motion: ChannelScore
    fused_score: float
    threshold: float                 # the per-camera fused threshold this was judged against
    weights: tuple[float, float]     # (w_appearance, w_motion) actually applied
    modality: Modality
    night: bool                      # True when the night (motion-heavier) weights were applied
    spatial_iou: float               # IoU of the motion peak region with the box (spatial gate)
    streak: int                      # consecutive agreeing frames when the event fired (temporal gate)
    first_agreed_at: float           # monotonic timestamp of the first frame of the streak
    timestamp: float                 # monotonic timestamp of the frame the gate was passed on
    captured_at: float
    frame_index: int
    best_frame: FrameRef
    source: SourceLabel
    track: Track | None = None       # the track snapshot the event was decided on (rule engine input)
    kind: Literal["DETECTION"] = "DETECTION"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def timestamp_s(self) -> float:
        """Alias of `timestamp`, the name the rule engine's FusedEventLike protocol reads."""
        return self.timestamp

    @property
    def appearance_score(self) -> float:
        return self.appearance.score

    @property
    def motion_score(self) -> float:
        return self.motion.score

    @property
    def bbox_normalised(self) -> tuple[float, float, float, float]:
        """[x, y, w, h] in 0..1, the wire format of ARCHITECTURE_V2 section 4.2."""
        w, h = self.frame_size
        x1, y1, x2, y2 = self.bbox
        return (round(x1 / w, 4), round(y1 / h, 4), round((x2 - x1) / w, 4), round((y2 - y1) / h, 4))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox_normalised"] = self.bbox_normalised
        return d


@dataclass(frozen=True)
class CameraStateEvent:
    """A camera-state anomaly (slide 4). `active` False marks the end of a previously raised anomaly."""

    camera_id: str
    anomaly: CameraStateKind
    active: bool
    frame_index: int
    timestamp: float
    captured_at: float
    since: float                     # monotonic timestamp the condition was first seen
    detail: str
    metrics: tuple[tuple[str, float], ...] = ()
    source: SourceLabel = "file"
    kind: Literal["CAMERA_STATE"] = "CAMERA_STATE"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metrics"] = dict(self.metrics)
        return d
