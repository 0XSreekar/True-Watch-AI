"""Hand the tracker's tracks to the rule engine (edge/rules/), every frame.

`pipeline.types.Track` already has the fields `rules.tracks.TrackLike` reads, with one difference in
meaning that matters: its `history` timestamps are the pipeline's monotonic seconds (pipeline/source.py),
while the rule engine reads them as wall-clock epoch seconds (rules/hours.py compares them with the
post's sanctioned hours; rules/baseline.py buckets footfall by hour). A monotonic 6.9 s read as epoch is
01:00 on 1 January 1970, which would make every track "out of hours". `to_rule_track` shifts the
timestamps by the frame's wall-clock offset; the shift is constant, so dwell, velocity and crossing
order are unchanged.

Boxes stay in the pipeline's pixel (x1, y1, x2, y2) form. The rules read positions only from `history`
(bottom-centre ground-contact points, which tracking.py already computes from x1..x2 and y2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pipeline.types import BBox, Track


@dataclass(frozen=True)
class RuleTrack:
    """A `rules.tracks.TrackLike` view of one pipeline track, with epoch-second history."""

    track_id: int
    camera_id: str
    class_name: str
    history: tuple[tuple[float, float, float], ...]
    bbox: BBox | None                  # pixel (x1, y1, x2, y2), as in pipeline.types
    state: str = "confirmed"


def to_rule_track(track: Track, epoch_offset_s: float) -> RuleTrack:
    """`epoch_offset_s` = frame.captured_at - frame.timestamp (wall clock minus pipeline clock)."""
    history = tuple((t + epoch_offset_s, x, y) for (t, x, y) in track.history)
    return RuleTrack(track_id=track.track_id, camera_id=track.camera_id, class_name=track.class_name,
                     history=history, bbox=track.bbox, state=track.state)


def rule_tracks(tracks: Iterable[Track], epoch_offset_s: float, only_ids: set[int] | None = None) -> list[RuleTrack]:
    """Confirmed or recently lost tracks (never tentative), optionally restricted to `only_ids`."""
    out = []
    for t in tracks:
        if t.state == "tentative" or len(t.history) < 1:
            continue
        if only_ids is not None and t.track_id not in only_ids:
            continue
        out.append(to_rule_track(t, epoch_offset_s))
    return out


def dominant_direction(heading_deg: float | None) -> str | None:
    """Majority heading (image coordinates, y down) as the contract's short label."""
    if heading_deg is None:
        return None
    h = heading_deg % 360.0
    if h < 45 or h >= 315:
        return "ltr"
    if h < 135:
        return "ttb"
    if h < 225:
        return "rtl"
    return "btt"
