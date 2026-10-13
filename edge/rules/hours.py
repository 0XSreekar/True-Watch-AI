"""Out-of-sanctioned-hours: a track present when this camera's post says no one should be.

Per-camera sanctioned windows are wall-clock, local to the post, and every
post in scope for this project is in India — hence `Asia/Kolkata` via the
standard-library `zoneinfo`, not a hardcoded UTC offset (which would be
wrong for half the year everywhere DST applies, and Asia/Kolkata has none,
but the point stands: never assume the host machine's local time is the
post's local time).

A window may cross midnight (e.g. "22:00 to 05:00"), which is the common
case for a sanctioned-hours fence at a border post, so the check handles
wrap-around explicitly rather than assuming `start < end`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from .tracks import TrackLike

POST_TIMEZONE = ZoneInfo("Asia/Kolkata")


class HoursConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SanctionedWindow:
    """Movement inside [start, end) local time is expected and does not fire."""

    start: dt_time
    end: dt_time

    def contains(self, moment: dt_time) -> bool:
        if self.start == self.end:
            return True  # a zero-width window means "sanctioned all day"
        if self.start < self.end:
            return self.start <= moment < self.end
        # wraps past midnight, e.g. 22:00 -> 05:00
        return moment >= self.start or moment < self.end


@dataclass(frozen=True)
class HoursConfig:
    camera_id: str
    sanctioned: tuple[SanctionedWindow, ...] = ()

    def __post_init__(self) -> None:
        if not self.sanctioned:
            raise HoursConfigError(
                f"camera {self.camera_id!r} needs at least one sanctioned window "
                "(use 00:00-00:00 for 'sanctioned all day')"
            )


@dataclass(frozen=True)
class OutOfHoursEvent:
    track_id: int | str
    camera_id: str
    local_time: str  # HH:MM:SS, Asia/Kolkata

    def reason(self) -> str:
        return f"track {self.track_id} moved at {self.local_time} IST, outside sanctioned hours"

    def evidence(self) -> dict:
        return {"local_time": self.local_time, "timezone": "Asia/Kolkata"}


def local_time_at(epoch_s: float) -> datetime:
    return datetime.fromtimestamp(epoch_s, tz=POST_TIMEZONE)


def evaluate(track: TrackLike, config: HoursConfig) -> OutOfHoursEvent | None:
    """Check the track's most recent ground-contact timestamp against the config.

    `track.history` timestamps are epoch seconds (`time.time()`-style, the
    same convention `edge/pipeline/ingest.py`'s `Frame.captured_at` uses).
    """
    if not track.history:
        return None
    last_ts = track.history[-1][0]
    local_dt = local_time_at(last_ts)
    moment = local_dt.time()

    if any(window.contains(moment) for window in config.sanctioned):
        return None

    return OutOfHoursEvent(
        track_id=track.track_id,
        camera_id=config.camera_id,
        local_time=local_dt.strftime("%H:%M:%S"),
    )
