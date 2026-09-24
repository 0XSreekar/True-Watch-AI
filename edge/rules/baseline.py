"""Per-camera unsupervised baseline: what "normal" looks like on this camera.

Slide 4: *"the operator dismisses one and that camera's baseline updates."*
No labelled data exists for "suspicious" (slide 4 concedes this outright), so
the baseline is learnt from the camera's own unlabelled tracks: the majority
direction of travel, typical dwell, and typical footfall by hour of day.
`direction.py` compares a track's heading against `majority_heading_deg`;
future rules can compare dwell or footfall the same way.

The baseline is a frozen, persisted snapshot — `learn()` builds one from a
batch of tracks, `update_on_dismissal()` nudges an existing one towards a
specific dismissed observation, and `save()` / `load()` round-trip it to
JSON so it survives a restart. Nothing here mutates in place; every function
returns a new `CameraBaseline`.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import tracks as trk
from .tracks import TrackLike

# How strongly one dismissal nudges the learnt baseline. Small on purpose: one
# operator override should not overwrite weeks of learning, but a run of
# dismissals on the same pattern should visibly shift it (see
# `test_rules.py::test_dismissal_measurably_shifts_the_baseline`).
DISMISSAL_LEARNING_RATE = 0.15
MIN_DISPLACEMENT_FOR_HEADING_PX = 10.0


@dataclass(frozen=True)
class CameraBaseline:
    camera_id: str
    majority_heading_deg: float | None = None
    with_flow_count: int = 0
    against_flow_count: int = 0
    typical_dwell_s: float = 0.0
    footfall_by_hour: dict[int, int] = field(default_factory=dict)
    dismissal_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["footfall_by_hour"] = {str(k): v for k, v in self.footfall_by_hour.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CameraBaseline":
        d = dict(d)
        d["footfall_by_hour"] = {int(k): v for k, v in d.get("footfall_by_hour", {}).items()}
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "CameraBaseline":
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def empty(cls, camera_id: str) -> "CameraBaseline":
        return cls(camera_id=camera_id)


def _heading_deg(track: TrackLike) -> float | None:
    if trk.net_displacement_px(track) < MIN_DISPLACEMENT_FOR_HEADING_PX:
        return None
    vx, vy = trk.average_velocity_px_s(track)
    if vx == 0.0 and vy == 0.0:
        return None
    return math.degrees(math.atan2(vy, vx)) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def learn(camera_id: str, tracks: list[TrackLike]) -> CameraBaseline:
    """Build a baseline from a batch of unlabelled tracks.

    Majority heading: the circular mean of all track headings that have a
    reliable displacement, expressed as with/against counts relative to that
    mean (MEASUREMENTS.md section 2: "majority flow left to right, 6 with,
    4 against").
    """
    headings = [h for h in (_heading_deg(t) for t in tracks) if h is not None]
    dwell_values = [trk.dwell_seconds(t) for t in tracks if trk.dwell_seconds(t) > 0]

    footfall: dict[int, int] = {}
    for t in tracks:
        first = trk.first_point(t)
        if first is None:
            continue
        from datetime import datetime, timezone

        hour = datetime.fromtimestamp(first[0], tz=timezone.utc).hour
        footfall[hour] = footfall.get(hour, 0) + 1

    if not headings:
        return CameraBaseline(
            camera_id=camera_id,
            typical_dwell_s=statistics.fmean(dwell_values) if dwell_values else 0.0,
            footfall_by_hour=footfall,
        )

    # Circular mean so 350 deg and 10 deg average to 0 deg, not 180 deg.
    sin_sum = sum(math.sin(math.radians(h)) for h in headings)
    cos_sum = sum(math.cos(math.radians(h)) for h in headings)
    majority = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0

    with_flow = sum(1 for h in headings if _angle_diff_deg(h, majority) < 90.0)
    against_flow = len(headings) - with_flow

    return CameraBaseline(
        camera_id=camera_id,
        majority_heading_deg=majority,
        with_flow_count=with_flow,
        against_flow_count=against_flow,
        typical_dwell_s=statistics.fmean(dwell_values) if dwell_values else 0.0,
        footfall_by_hour=footfall,
    )


def update_on_dismissal(
    baseline: CameraBaseline, track: TrackLike, *, rule_name: str
) -> CameraBaseline:
    """Fold one dismissed observation into the baseline.

    A dismissal means the operator judged this track's behaviour normal for
    this camera even though a rule fired. The direction rule is the one that
    responds to a heading: the baseline's majority heading is nudged towards
    the dismissed track's own heading, by `DISMISSAL_LEARNING_RATE`, and the
    with/against counters shift by one from "against" to "with" — a
    measurable move, not a cosmetic one. A loitering dismissal instead widens
    `typical_dwell_s` towards the dismissed track's dwell, since that is the
    figure the loitering rule reads.
    """
    heading = _heading_deg(track)
    majority = baseline.majority_heading_deg
    with_flow = baseline.with_flow_count
    against_flow = baseline.against_flow_count

    if rule_name in ("wrong direction",) and heading is not None:
        if majority is None:
            majority = heading
        else:
            sin_m = math.sin(math.radians(majority))
            cos_m = math.cos(math.radians(majority))
            sin_t = math.sin(math.radians(heading))
            cos_t = math.cos(math.radians(heading))
            rate = DISMISSAL_LEARNING_RATE
            sin_blend = (1 - rate) * sin_m + rate * sin_t
            cos_blend = (1 - rate) * cos_m + rate * cos_t
            majority = math.degrees(math.atan2(sin_blend, cos_blend)) % 360.0
        if against_flow > 0:
            against_flow -= 1
            with_flow += 1

    typical_dwell = baseline.typical_dwell_s
    if rule_name == "loitering":
        dwell = trk.dwell_seconds(track)
        typical_dwell = (
            (1 - DISMISSAL_LEARNING_RATE) * typical_dwell + DISMISSAL_LEARNING_RATE * dwell
        )

    return CameraBaseline(
        camera_id=baseline.camera_id,
        majority_heading_deg=majority,
        with_flow_count=with_flow,
        against_flow_count=against_flow,
        typical_dwell_s=typical_dwell,
        footfall_by_hour=dict(baseline.footfall_by_hour),
        dismissal_count=baseline.dismissal_count + 1,
    )
