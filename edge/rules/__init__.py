"""Named rules on tracks, the per-camera baseline, and the ground-plane homography.

Rules consume fused events/tracks (`rules.tracks.TrackLike` /
`FusedEventLike`), never raw detections. See `rules.engine` for the entry
point.
"""

from .engine import (
    ALL_RULE_NAMES,
    RULE_FENCE_CROSSED,
    RULE_GROUP_FORMED,
    RULE_LOITERING,
    RULE_OUT_OF_HOURS,
    RULE_WRONG_DIRECTION,
    FiredRule,
    TrackVerdict,
    evaluate_camera,
    evaluate_track,
    throttled_alerts,
)

__all__ = [
    "ALL_RULE_NAMES",
    "RULE_FENCE_CROSSED",
    "RULE_GROUP_FORMED",
    "RULE_LOITERING",
    "RULE_OUT_OF_HOURS",
    "RULE_WRONG_DIRECTION",
    "FiredRule",
    "TrackVerdict",
    "evaluate_camera",
    "evaluate_track",
    "throttled_alerts",
]
