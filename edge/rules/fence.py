"""Virtual fence intrusion: a tracked object crosses an operator-drawn line.

Slide 2, capability 5. MEASUREMENTS.md section 2 draws the fence at a single
vertical line (x = 320) for the rule-sanity check; the operator config format
here generalises that to an arbitrary polyline (one segment is the x = 320
case), because a real post's fence rarely runs perfectly vertical in the
frame.

The crossing test is **segment intersection between the track's own path and
the fence's segments**, not "is the bbox centre inside a polygon this frame".
A centre-in-polygon test misses a track that crosses and re-crosses between
sampled frames, and it cannot report the sub-pixel point or instant of
crossing — both of which this module returns as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .tracks import TrackLike

Point = tuple[float, float]
Direction = Literal["in", "out", "both"]

_EPS = 1e-9


class FenceConfigError(ValueError):
    """A fence line was configured with fewer than two points."""


@dataclass(frozen=True)
class FenceLine:
    """An operator-drawn polyline. `direction` filters which crossing fires.

    `direction="in"` fires only when a track crosses from the line's right
    side to its left side (relative to each segment's own direction vector);
    `"out"` is the opposite; `"both"` fires on either. The convention is
    documented, not guessed: see `_side`.
    """

    name: str
    points: Sequence[Point]
    direction: Direction = "both"

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise FenceConfigError(f"fence {self.name!r} needs at least 2 points")

    def segments(self) -> list[tuple[Point, Point]]:
        return list(zip(self.points, self.points[1:]))


@dataclass(frozen=True)
class FenceCrossing:
    """Evidence for one fired `fence crossed` rule."""

    fence_name: str
    track_id: int | str
    crossed_at_s: float
    point_px: Point
    direction: Literal["in", "out"]

    def reason(self) -> str:
        return (
            f"track {self.track_id} crossed fence {self.fence_name!r} "
            f"({self.direction}bound) at ({self.point_px[0]:.0f}, {self.point_px[1]:.0f})"
        )

    def evidence(self) -> dict:
        return {
            "fence_name": self.fence_name,
            "crossed_at_s": self.crossed_at_s,
            "point_px": self.point_px,
            "direction": self.direction,
        }


def _side(a: Point, b: Point, p: Point) -> float:
    """Signed area of (a, b, p): >0 left of a->b, <0 right, 0 on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_intersect(
    p1: Point, p2: Point, p3: Point, p4: Point
) -> Point | None:
    """Proper 2-D segment intersection via the standard orientation test.

    Returns the intersection point if segments (p1, p2) and (p3, p4) cross,
    including a touch at an endpoint; None if they don't.
    """
    d1 = _side(p3, p4, p1)
    d2 = _side(p3, p4, p2)
    d3 = _side(p1, p2, p3)
    d4 = _side(p1, p2, p4)

    straddles_34 = (d1 > _EPS and d2 < -_EPS) or (d1 < -_EPS and d2 > _EPS)
    straddles_12 = (d3 > _EPS and d4 < -_EPS) or (d3 < -_EPS and d4 > _EPS)
    if not (straddles_34 and straddles_12):
        return None

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < _EPS:
        return None  # parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


def evaluate(track: TrackLike, fence: FenceLine) -> FenceCrossing | None:
    """Return the first fence crossing found along `track`'s path, or None.

    Only ground-contact points already in `track.history` are used — no new
    sampling is introduced, so a crossing that truly happened between two
    tracked frames is caught by testing the segment those two frames define,
    sub-pixel accurate, rather than only the frames themselves.
    """
    points = track.history
    if len(points) < 2:
        return None

    for (t0, x0, y0), (t1, x1, y1) in zip(points, points[1:]):
        track_seg = ((x0, y0), (x1, y1))
        for a, b in fence.segments():
            hit = _segments_intersect(track_seg[0], track_seg[1], a, b)
            if hit is None:
                continue
            side_before = _side(a, b, track_seg[0])
            side_after = _side(a, b, track_seg[1])
            if side_before >= 0 and side_after < 0:
                direction: Literal["in", "out"] = "in"
            elif side_before <= 0 and side_after > 0:
                direction = "out"
            else:
                continue  # started on the line with no net side change
            if fence.direction != "both" and direction != fence.direction:
                continue
            return FenceCrossing(
                fence_name=fence.name,
                track_id=track.track_id,
                crossed_at_s=t1,
                point_px=hit,
                direction=direction,
            )
    return None


def evaluate_all(track: TrackLike, fences: Sequence[FenceLine]) -> list[FenceCrossing]:
    """Evaluate every configured fence for one track."""
    crossings = []
    for fence in fences:
        hit = evaluate(track, fence)
        if hit is not None:
            crossings.append(hit)
    return crossings
