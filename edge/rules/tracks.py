"""Typing contracts the rule engine is written against — not Phase 3's runtime.

Phase 3 (`feat/phase3-fusion`, `edge/pipeline/types.py`) defines frozen
dataclasses `Detection`, `Track`, `ChannelScore`, `FusedEvent` concurrently
with this phase and is not merged yet. Every module under `edge/rules/` is
written against the two `typing.Protocol`s below instead of importing
anything from `edge/pipeline/`. Rules consume fused events/tracks, never raw
detections (docs/ARCHITECTURE_V2.md section 4: "rules run on fused events,
never raw detections").

ADAPTER CONTRACT for whoever wires Phase 3's real `Track` / `FusedEvent` into
this engine once it merges:

* `pipeline.types.Track` satisfies `TrackLike` as soon as it exposes:
    - `track_id`    (already on the frozen dataclass — any hashable)
    - `camera_id`   (str)
    - `class_name`  (str — Phase 3 may spell this `cls` or `label`; alias it)
    - `history`     (a sequence of `(timestamp_s, x_px, y_px)` ground-contact
                     points, oldest first. If Phase 3's `Track` instead holds
                     a sequence of `Detection`s or box centroids, the adapter
                     is one comprehension:
                     `[(d.timestamp_s, *ground_point(d.bbox)) for d in track.detections]`.
                     A ground-contact point is the bottom-centre of the
                     bounding box — `(x + w / 2, y + h)` — because that is
                     the point that sits on the ground plane a homography is
                     calibrated against; a box centroid is not.)
    - `bbox`        (current `(x, y, w, h)` in pixels, or `None`)
  No wrapper class is needed if the field names already match — Python's
  structural typing (`@runtime_checkable`) accepts `Track` as-is.
* `pipeline.types.FusedEvent` satisfies `FusedEventLike` as soon as it
  exposes `track` (a `TrackLike`), `camera_id` and `timestamp_s`. The engine
  only reads `.track` off it today; `appearance_score` / `motion_score` are
  read defensively via `getattr(event, "appearance_score", None)` so their
  absence never breaks a rule.

Keep this adapter surface small: a one-file shim in the orchestrator
(e.g. `edge/pipeline/rules_adapter.py`) that walks a list of Phase 3
`FusedEvent`s and rebuilds `history` is enough. Nothing in `edge/rules/`
should need to change when Phase 3 merges.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence, runtime_checkable

GroundPoint = tuple[float, float, float]  # (timestamp_s, x_px, y_px)
BBox = tuple[float, float, float, float]  # (x, y, w, h) in pixels


@runtime_checkable
class TrackLike(Protocol):
    """What every rule needs from a tracked object. Structural, not nominal."""

    track_id: int | str
    camera_id: str
    class_name: str
    history: Sequence[GroundPoint]
    bbox: BBox | None


@runtime_checkable
class FusedEventLike(Protocol):
    """A fusion-confirmed event the rule engine may evaluate.

    Rules read `.track` off this; they never see the per-channel detections
    that produced it.
    """

    track: TrackLike
    camera_id: str
    timestamp_s: float


def ground_point(bbox: BBox) -> tuple[float, float]:
    """Bottom-centre of a pixel bbox — the point that sits on the ground plane."""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h)


def dwell_seconds(track: TrackLike) -> float:
    """Wall-clock span the track has been alive, from its own history."""
    history = track.history
    if len(history) < 2:
        return 0.0
    return float(history[-1][0] - history[0][0])


def net_displacement_px(track: TrackLike) -> float:
    """Straight-line distance between the track's first and last ground point."""
    history = track.history
    if len(history) < 2:
        return 0.0
    _, x0, y0 = history[0]
    _, x1, y1 = history[-1]
    return math.hypot(x1 - x0, y1 - y0)


def path_length_px(track: TrackLike) -> float:
    """Total distance travelled along the track, summed segment by segment.

    Net displacement can be small while path length is large — that gap is
    exactly what marks loitering (moving back and forth in place) rather
    than a fast pass-through. See MEASUREMENTS.md section 2, track 6.
    """
    history = track.history
    if len(history) < 2:
        return 0.0
    total = 0.0
    for (_, x0, y0), (_, x1, y1) in zip(history, history[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def average_velocity_px_s(track: TrackLike) -> tuple[float, float]:
    """Mean (vx, vy) in px/s from first to last point. Zero if too short or instant."""
    history = track.history
    if len(history) < 2:
        return (0.0, 0.0)
    t0, x0, y0 = history[0]
    t1, x1, y1 = history[-1]
    dt = t1 - t0
    if dt <= 0:
        return (0.0, 0.0)
    return ((x1 - x0) / dt, (y1 - y0) / dt)


def last_point(track: TrackLike) -> GroundPoint | None:
    return track.history[-1] if track.history else None


def first_point(track: TrackLike) -> GroundPoint | None:
    return track.history[0] if track.history else None
