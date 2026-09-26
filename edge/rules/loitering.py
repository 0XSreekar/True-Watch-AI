"""Loitering: dwell long enough, without actually going anywhere.

MEASUREMENTS.md section 2 defines it precisely: **dwell >= 3 s AND net
displacement < 90 px**. Track 6 is the canonical example — 98 frames, 5.0 s
dwell, only 70 px net displacement but 163 px of path travelled, i.e. it
moved back and forth in place rather than passing through. That gap between
net displacement and path length is what separates loitering from someone
standing still (both small) or someone walking a long, slow arc (both
large).

Thresholds are configurable in **metres** wherever a trustworthy homography
exists for the camera (slide 4: thresholds must transfer between cameras of
different zoom), and fall back to **pixels** where it does not — the exact
84/90 px case MEASUREMENTS.md measured had no homography, so the pixel path
is not a secondary/lesser mode, it's what reproduces that table exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import tracks as trk
from .homography import Homography, px_to_metres
from .tracks import TrackLike

DEFAULT_DWELL_S = 3.0
DEFAULT_NET_DISPLACEMENT_PX = 90.0


@dataclass(frozen=True)
class LoiteringConfig:
    dwell_s: float = DEFAULT_DWELL_S
    net_displacement_m: float | None = None  # used when a homography exists
    net_displacement_px: float = DEFAULT_NET_DISPLACEMENT_PX  # fallback


@dataclass(frozen=True)
class LoiteringEvent:
    track_id: int | str
    dwell_s: float
    net_displacement_px: float
    path_length_px: float
    net_displacement_m: float | None
    threshold_px_or_m: float
    unit: str  # "m" | "px"

    def reason(self) -> str:
        if self.unit == "m":
            return (
                f"track {self.track_id} loitered {self.dwell_s:.1f}s, net "
                f"{self.net_displacement_m:.1f}m (path {self.path_length_px:.0f}px) "
                f"under the {self.threshold_px_or_m:.1f}m threshold"
            )
        return (
            f"track {self.track_id} loitered {self.dwell_s:.1f}s, net "
            f"{self.net_displacement_px:.0f}px (path {self.path_length_px:.0f}px) "
            f"under the {self.threshold_px_or_m:.0f}px threshold"
        )

    def evidence(self) -> dict:
        return {
            "dwell_s": self.dwell_s,
            "net_displacement_px": self.net_displacement_px,
            "path_length_px": self.path_length_px,
            "net_displacement_m": self.net_displacement_m,
            "threshold": self.threshold_px_or_m,
            "unit": self.unit,
        }


def evaluate(
    track: TrackLike,
    config: LoiteringConfig | None = None,
    homography: Homography | None = None,
) -> LoiteringEvent | None:
    config = config or LoiteringConfig()

    dwell = trk.dwell_seconds(track)
    if dwell < config.dwell_s:
        return None

    net_px = trk.net_displacement_px(track)
    path_px = trk.path_length_px(track)

    anchor = trk.first_point(track)
    at_point = (anchor[1], anchor[2]) if anchor else (0.0, 0.0)
    net_m = None
    if homography is not None:
        net_m = px_to_metres(homography, net_px, at_point)

    if net_m is not None and config.net_displacement_m is not None:
        if net_m >= config.net_displacement_m:
            return None
        return LoiteringEvent(
            track_id=track.track_id,
            dwell_s=dwell,
            net_displacement_px=net_px,
            path_length_px=path_px,
            net_displacement_m=net_m,
            threshold_px_or_m=config.net_displacement_m,
            unit="m",
        )

    if net_px >= config.net_displacement_px:
        return None
    return LoiteringEvent(
        track_id=track.track_id,
        dwell_s=dwell,
        net_displacement_px=net_px,
        path_length_px=path_px,
        net_displacement_m=net_m,
        threshold_px_or_m=config.net_displacement_px,
        unit="px",
    )
