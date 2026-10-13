"""Group formation: N or more tracks within R metres of each other for T seconds.

Slide 2 capability 6 / the `FootageSearch` demo query ("group of more than
four people after midnight") needs a named rule behind it. Radius is
configured in metres where a homography exists (falling back to pixels
otherwise, same convention as `loitering.py`); persistence in seconds
guards against two unrelated people who happen to pass within R of each
other for one frame.

Grouping is evaluated over a *set* of simultaneous tracks on one camera, not
a single track in isolation — the only rule module in this package with that
shape, because a group is a property of several tracks together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .homography import Homography, px_to_metres
from .tracks import TrackLike

DEFAULT_MIN_GROUP_SIZE = 4
DEFAULT_RADIUS_M = 3.0
DEFAULT_RADIUS_PX = 150.0
DEFAULT_PERSIST_S = 5.0
# How finely to sample the union of track lifespans when checking persistence.
SAMPLE_STEP_S = 0.5


@dataclass(frozen=True)
class GroupingConfig:
    min_size: int = DEFAULT_MIN_GROUP_SIZE
    radius_m: float | None = None
    radius_px: float = DEFAULT_RADIUS_PX
    persist_s: float = DEFAULT_PERSIST_S


@dataclass(frozen=True)
class GroupEvent:
    track_ids: tuple
    size: int
    duration_s: float
    radius_used: float
    unit: str  # "m" | "px"

    def reason(self) -> str:
        ids = ", ".join(str(i) for i in self.track_ids)
        return (
            f"group of {self.size} ({ids}) held within {self.radius_used:.1f}{self.unit} "
            f"for {self.duration_s:.1f}s"
        )

    def evidence(self) -> dict:
        return {
            "track_ids": list(self.track_ids),
            "size": self.size,
            "duration_s": self.duration_s,
            "radius": self.radius_used,
            "unit": self.unit,
        }


def _position_at(track: TrackLike, t: float) -> tuple[float, float] | None:
    """Linearly interpolated ground position of `track` at time `t`, or None if absent."""
    history = track.history
    if not history:
        return None
    if t < history[0][0] or t > history[-1][0]:
        return None
    for (t0, x0, y0), (t1, x1, y1) in zip(history, history[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return (x0, y0)
            frac = (t - t0) / (t1 - t0)
            return (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))
    return (history[-1][1], history[-1][2])


def _distance(a: tuple[float, float], b: tuple[float, float], homography: Homography | None) -> tuple[float, str]:
    if homography is not None:
        m = px_to_metres(homography, 1.0, a)
        if m is not None:
            scale = m  # metres per pixel near `a`
            px_dist = math.hypot(a[0] - b[0], a[1] - b[1])
            return px_dist * scale, "m"
    return math.hypot(a[0] - b[0], a[1] - b[1]), "px"


def _clusters_at(
    tracks: list[TrackLike], t: float, radius: float, homography: Homography | None
) -> list[list[TrackLike]]:
    positions = {}
    for track in tracks:
        p = _position_at(track, t)
        if p is not None:
            positions[track.track_id] = (track, p)

    ids = list(positions.keys())
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            _, pa = positions[ids[i]]
            _, pb = positions[ids[j]]
            dist, _unit = _distance(pa, pb, homography)
            if dist <= radius:
                union(ids[i], ids[j])

    groups: dict = {}
    for i in ids:
        groups.setdefault(find(i), []).append(positions[i][0])
    return list(groups.values())


def evaluate(
    tracks: list[TrackLike],
    config: GroupingConfig | None = None,
    homography: Homography | None = None,
) -> list[GroupEvent]:
    """Find every group of >= min_size tracks held within radius for >= persist_s.

    Samples the union of all track lifespans on a fixed grid (`SAMPLE_STEP_S`)
    and unions tracks within the radius at each sample using union-find; a
    cluster of track ids that stays a qualifying-size cluster across samples
    spanning >= `persist_s` fires one `GroupEvent`.
    """
    config = config or GroupingConfig()
    if not tracks:
        return []

    radius = config.radius_px
    unit = "px"
    if homography is not None and config.radius_m is not None:
        radius = config.radius_m
        unit = "m"

    starts = [t.history[0][0] for t in tracks if t.history]
    ends = [t.history[-1][0] for t in tracks if t.history]
    if not starts:
        return []
    t0, t1 = min(starts), max(ends)

    # cluster_key (frozenset of ids) -> [first_seen_s, last_seen_s]
    spans: dict[frozenset, list[float]] = {}
    t = t0
    while t <= t1 + 1e-9:
        for cluster in _clusters_at(tracks, t, radius, homography):
            if len(cluster) < config.min_size:
                continue
            key = frozenset(tr.track_id for tr in cluster)
            if key not in spans:
                spans[key] = [t, t]
            else:
                spans[key][1] = t
        t += SAMPLE_STEP_S

    events = []
    for key, (first, last) in spans.items():
        duration = last - first
        if duration >= config.persist_s:
            events.append(
                GroupEvent(
                    track_ids=tuple(sorted(key, key=str)),
                    size=len(key),
                    duration_s=duration,
                    radius_used=radius,
                    unit=unit,
                )
            )
    return events
