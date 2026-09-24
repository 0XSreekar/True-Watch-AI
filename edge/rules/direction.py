"""Wrong direction: a track moving against the camera's learnt majority flow.

The learnt flow comes from `rules.baseline` — an unsupervised majority
direction computed from unlabelled tracks on that camera, per slide 4
("baseline learnt from the camera itself"). This module only compares a
track's own heading against that baseline; it does not learn anything.

A track with negligible net displacement has no reliable heading — a person
standing still and jittering 3 px between frames is not "moving against"
anything. `min_displacement_px` gates that out so short/noisy tracks (e.g.
MEASUREMENTS.md section 2 tracks 12, 26, 28: 3 px, 22 px, 34 px net) never
get a direction verdict at all, matching the fact that none of them fire any
rule in that table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from . import tracks as trk
from .tracks import TrackLike

DEFAULT_MIN_DISPLACEMENT_PX = 40.0
# A heading within this many degrees of the baseline's opposite is "against
# flow"; narrower than 180 so a track moving mostly sideways isn't scored
# either way.
DEFAULT_OPPOSING_ANGLE_DEG = 90.0


@runtime_checkable
class BaselineLike(Protocol):
    """The one field this module reads off a baseline.

    A structural Protocol rather than importing `rules.baseline.
    CameraBaseline` directly, so this module has no build-order dependency
    on the baseline module — `rules.baseline.CameraBaseline` satisfies this
    as-is, and so would any other camera-baseline representation that
    carries a learnt majority heading.
    """

    majority_heading_deg: float | None


@dataclass(frozen=True)
class DirectionConfig:
    min_displacement_px: float = DEFAULT_MIN_DISPLACEMENT_PX
    opposing_angle_deg: float = DEFAULT_OPPOSING_ANGLE_DEG


@dataclass(frozen=True)
class WrongDirectionEvent:
    track_id: int | str
    heading_deg: float
    baseline_heading_deg: float
    angle_from_baseline_deg: float
    net_displacement_px: float

    def reason(self) -> str:
        return (
            f"track {self.track_id} moved at {self.heading_deg:.0f}°, "
            f"{self.angle_from_baseline_deg:.0f}° off the learnt majority "
            f"flow of {self.baseline_heading_deg:.0f}°"
        )

    def evidence(self) -> dict:
        return {
            "heading_deg": self.heading_deg,
            "baseline_heading_deg": self.baseline_heading_deg,
            "angle_from_baseline_deg": self.angle_from_baseline_deg,
            "net_displacement_px": self.net_displacement_px,
        }


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angle between two headings in degrees, 0..180."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def evaluate(
    track: TrackLike,
    baseline: BaselineLike,
    config: DirectionConfig | None = None,
) -> WrongDirectionEvent | None:
    config = config or DirectionConfig()

    if baseline.majority_heading_deg is None:
        return None  # baseline hasn't learnt a flow yet

    net_px = trk.net_displacement_px(track)
    if net_px < config.min_displacement_px:
        return None

    vx, vy = trk.average_velocity_px_s(track)
    if vx == 0.0 and vy == 0.0:
        return None
    heading = math.degrees(math.atan2(vy, vx)) % 360.0

    angle_from_baseline = _angle_diff_deg(heading, baseline.majority_heading_deg)
    if angle_from_baseline < (180.0 - config.opposing_angle_deg):
        return None  # broadly with the flow

    return WrongDirectionEvent(
        track_id=track.track_id,
        heading_deg=heading,
        baseline_heading_deg=baseline.majority_heading_deg,
        angle_from_baseline_deg=angle_from_baseline,
        net_displacement_px=net_px,
    )
