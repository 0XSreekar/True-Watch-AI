"""Out-of-hours rule at the edges: IST conversion, the UTC date boundary and windows past midnight."""

from __future__ import annotations

from datetime import datetime, time, timezone
from types import SimpleNamespace

import pytest

from rules.hours import HoursConfig, SanctionedWindow, evaluate


def _track_at(utc: datetime):
    return SimpleNamespace(track_id=7, history=[(utc.timestamp(), 0.0, 0.0)])


DAYTIME = HoursConfig("RXL-01", (SanctionedWindow(time(6, 0), time(22, 0)),))
NIGHT_SHIFT = HoursConfig("RXL-01", (SanctionedWindow(time(22, 0), time(5, 0)),))


@pytest.mark.parametrize(
    "utc, fires, local",
    [
        (datetime(2026, 9, 29, 16, 29, 59, tzinfo=timezone.utc), False, "21:59:59"),  # just inside
        (datetime(2026, 9, 29, 16, 30, 0, tzinfo=timezone.utc), True, "22:00:00"),    # end is exclusive
        (datetime(2026, 9, 29, 0, 30, 0, tzinfo=timezone.utc), False, "06:00:00"),    # start is inclusive
        # 19:00 UTC on the 29th is 00:30 IST on the 30th: the local date differs from the UTC date
        (datetime(2026, 9, 29, 19, 0, 0, tzinfo=timezone.utc), True, "00:30:00"),
    ],
)
def test_daytime_window_is_judged_in_ist(utc, fires, local):
    event = evaluate(_track_at(utc), DAYTIME)
    assert (event is not None) is fires
    if event:
        assert event.local_time == local
        assert "IST" in event.reason() and event.evidence()["timezone"] == "Asia/Kolkata"


@pytest.mark.parametrize(
    "utc, fires",
    [
        (datetime(2026, 9, 29, 16, 30, tzinfo=timezone.utc), False),  # 22:00 IST, window opens
        (datetime(2026, 9, 29, 20, 30, tzinfo=timezone.utc), False),  # 02:00 IST, past midnight, inside
        (datetime(2026, 9, 29, 23, 29, tzinfo=timezone.utc), False),  # 04:59 IST, inside
        (datetime(2026, 9, 29, 23, 30, tzinfo=timezone.utc), True),   # 05:00 IST, window closed
        (datetime(2026, 9, 29, 6, 30, tzinfo=timezone.utc), True),    # 12:00 IST
    ],
)
def test_window_that_wraps_past_midnight(utc, fires):
    assert (evaluate(_track_at(utc), NIGHT_SHIFT) is not None) is fires


def test_empty_history_never_fires():
    assert evaluate(SimpleNamespace(track_id=1, history=[]), DAYTIME) is None
