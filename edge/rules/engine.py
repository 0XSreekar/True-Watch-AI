"""Evaluate every rule against every track, and return what fired.

This is the single entry point the rest of `edge/` (and, later, the Phase 3
orchestrator) calls. It knows about no detector, no ONNX model, no ByteTrack
internals — only `TrackLike` / `FusedEventLike` (`rules/tracks.py`),
`CameraRuleConfig` (`rules/config.py`) and `CameraBaseline`
(`rules/baseline.py`). Everything it needs is either on the track or in the
per-camera config.

Every `FiredRule` this module returns carries:

* `rule_name` — one of the canonical names below, used verbatim in
  `AlertQueue`'s `rule_fired` field and joined with `" + "` for a track that
  fires more than one rule in the same pass (docs/ARCHITECTURE_V2.md
  section 3.3, matching `event.rule.fired.join(' + ')` in
  `ingest.service.js`).
* `evidence` — a `dict` of the numeric values that caused it to fire (dwell,
  net px, path px, heading, …), never just the name.
* `reason` — a one-line human-readable string. This is the fallback
  `AlertQueue.reason` when no VLM explanation exists yet (Phase 6), and it is
  **never empty** — that is the one invariant every caller may rely on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from . import direction as direction_rule
from . import fence as fence_rule
from . import grouping as grouping_rule
from . import hours as hours_rule
from . import loitering as loitering_rule
from .baseline import CameraBaseline
from .config import CameraRuleConfig
from .throttle import AlertThrottle, ThrottledAlert
from .tracks import TrackLike

RULE_FENCE_CROSSED = "fence crossed"
RULE_WRONG_DIRECTION = "wrong direction"
RULE_LOITERING = "loitering"
RULE_OUT_OF_HOURS = "out of hours"
RULE_GROUP_FORMED = "group formed"

ALL_RULE_NAMES = (
    RULE_FENCE_CROSSED,
    RULE_WRONG_DIRECTION,
    RULE_LOITERING,
    RULE_OUT_OF_HOURS,
    RULE_GROUP_FORMED,
)


@dataclass(frozen=True)
class FiredRule:
    rule_name: str
    track_id: int | str
    evidence: dict
    reason: str

    def __post_init__(self) -> None:
        # The one invariant every downstream consumer (AlertQueue via the
        # ingest mapper) is allowed to assume. Fail loudly here rather than
        # ship a blank card.
        if not self.reason or not self.reason.strip():
            raise ValueError(f"rule {self.rule_name!r} produced an empty reason")


@dataclass(frozen=True)
class TrackVerdict:
    """Every rule that fired for one track, folded into one alert-worth of evidence."""

    track_id: int | str
    camera_id: str
    fired: tuple[FiredRule, ...]

    @property
    def rule_fired(self) -> str:
        """The `"+"`-joined string `AlertQueue`/`rule_fired` expects. Empty if nothing fired."""
        return " + ".join(f.rule_name for f in self.fired)

    @property
    def reason(self) -> str:
        """Every fired rule's reason, joined — always non-empty when anything fired."""
        return "; ".join(f.reason for f in self.fired)

    @property
    def evidence(self) -> dict:
        return {f.rule_name: f.evidence for f in self.fired}


def evaluate_track(
    track: TrackLike,
    config: CameraRuleConfig,
    baseline: CameraBaseline,
) -> TrackVerdict:
    """Run every applicable rule against one track. Never touches the throttle."""
    fired: list[FiredRule] = []

    for fence in config.fences:
        hit = fence_rule.evaluate(track, fence)
        if hit is not None:
            fired.append(
                FiredRule(
                    rule_name=RULE_FENCE_CROSSED,
                    track_id=track.track_id,
                    evidence=hit.evidence(),
                    reason=hit.reason(),
                )
            )

    loiter = loitering_rule.evaluate(track, config.loitering, config.homography)
    if loiter is not None:
        fired.append(
            FiredRule(
                rule_name=RULE_LOITERING,
                track_id=track.track_id,
                evidence=loiter.evidence(),
                reason=loiter.reason(),
            )
        )

    wrong_way = direction_rule.evaluate(track, baseline, config.direction)
    if wrong_way is not None:
        fired.append(
            FiredRule(
                rule_name=RULE_WRONG_DIRECTION,
                track_id=track.track_id,
                evidence=wrong_way.evidence(),
                reason=wrong_way.reason(),
            )
        )

    if config.hours is not None:
        out_of_hours = hours_rule.evaluate(track, config.hours)
        if out_of_hours is not None:
            fired.append(
                FiredRule(
                    rule_name=RULE_OUT_OF_HOURS,
                    track_id=track.track_id,
                    evidence=out_of_hours.evidence(),
                    reason=out_of_hours.reason(),
                )
            )

    return TrackVerdict(
        track_id=track.track_id, camera_id=track.camera_id, fired=tuple(fired)
    )


def evaluate_camera(
    tracks: Sequence[TrackLike],
    config: CameraRuleConfig,
    baseline: CameraBaseline,
) -> list[TrackVerdict]:
    """Per-track rules for every track on one camera, plus the group rule across all of them.

    Group formation is folded into each involved track's verdict as an
    additional `FiredRule`, so a track that is both loitering and part of a
    group still produces exactly one `TrackVerdict` — one alert candidate,
    per the de-dup contract in `throttle.py`.
    """
    verdicts = {t.track_id: evaluate_track(t, config, baseline) for t in tracks}

    groups = grouping_rule.evaluate(list(tracks), config.grouping, config.homography)
    for group in groups:
        for track_id in group.track_ids:
            existing = verdicts.get(track_id)
            if existing is None:
                continue
            new_fired = existing.fired + (
                FiredRule(
                    rule_name=RULE_GROUP_FORMED,
                    track_id=track_id,
                    evidence=group.evidence(),
                    reason=group.reason(),
                ),
            )
            verdicts[track_id] = TrackVerdict(
                track_id=existing.track_id, camera_id=existing.camera_id, fired=new_fired
            )

    return list(verdicts.values())


def throttled_alerts(
    verdicts: Sequence[TrackVerdict], throttle: AlertThrottle, now_s: float | None = None
) -> list[ThrottledAlert]:
    """Route every fired rule through the shared throttle. Verdicts with nothing fired are skipped.

    A track that fired more than one rule offers each rule separately — the
    dedup key is (camera, track, rule), so "fence crossed" and "loitering"
    on the same track are two independent alert slots, matching the fact
    MEASUREMENTS.md section 2 track 6 is reported as
    `"fence crossed + loitering"`, not folded into one anonymous rule name.
    """
    now_s = now_s if now_s is not None else time.time()
    out: list[ThrottledAlert] = []
    for verdict in verdicts:
        for rule in verdict.fired:
            alert = throttle.offer(verdict.camera_id, verdict.track_id, rule.rule_name, now_s)
            if alert is not None:
                out.append(alert)
    return out
