"""False-alert budget: rate limiting, de-duplication and the shift budget.

Slide 4 / slide 5 target: under 5 false alerts per camera per 24 h, so an
operator watching 8 cameras faces roughly 40 alerts a shift. This module
enforces the mechanical half of that (the rule engine's job is the other
half — firing only on real evidence):

* **de-duplication** — the same track firing the same rule repeatedly is ONE
  alert with a repeat count, never N alerts. A track that loiters for 30 s
  and gets re-evaluated every second should not produce 30 alerts.
* **per-camera rate limiting** — a rolling 24 h window, capped at
  `PER_CAMERA_24H_LIMIT` (5) *new* (non-duplicate) alerts.
* **shift budget** — a rolling window matching the console's
  `SHIFT_ALARM_BUDGET` (`backend/src/data/mockData.js`, currently 12),
  capping total new alerts across every camera an operator is watching.

Both caps only ever refuse a *new* alert (a fresh track+rule pair); a repeat
of an alert already inside the window is always folded into the existing
`ThrottledAlert.repeat_count` rather than being silently dropped, so the
operator never loses the fact that a dismissed pattern kept recurring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PER_CAMERA_24H_LIMIT = 5
SHIFT_ALARM_BUDGET = 12  # backend/src/data/mockData.js SHIFT_ALARM_BUDGET
ROLLING_WINDOW_S = 24 * 60 * 60
SHIFT_WINDOW_S = 12 * 60 * 60  # one operator shift, per docs/PHASE_MINUS1_SCOPE.md §2

DedupKey = tuple[str, object, str]  # (camera_id, track_id, rule_name)


@dataclass
class ThrottledAlert:
    camera_id: str
    track_id: object
    rule_name: str
    first_fired_at_s: float
    last_fired_at_s: float
    repeat_count: int = 1

    @property
    def key(self) -> DedupKey:
        return (self.camera_id, self.track_id, self.rule_name)


@dataclass
class AlertThrottle:
    """Stateful gate a rule engine call routes every fired rule through.

    Not thread-safe by itself — one instance is expected per edge process,
    matching the existing service's single-process model (see
    `edge/config.py`).
    """

    per_camera_limit: int = PER_CAMERA_24H_LIMIT
    shift_budget: int = SHIFT_ALARM_BUDGET
    rolling_window_s: float = ROLLING_WINDOW_S
    shift_window_s: float = SHIFT_WINDOW_S

    _alerts: dict[DedupKey, ThrottledAlert] = field(default_factory=dict)

    def _prune(self, now_s: float) -> None:
        cutoff = now_s - self.rolling_window_s
        stale = [k for k, a in self._alerts.items() if a.last_fired_at_s < cutoff]
        for k in stale:
            del self._alerts[k]

    def _camera_count(self, camera_id: str, now_s: float) -> int:
        cutoff = now_s - self.rolling_window_s
        return sum(
            1
            for a in self._alerts.values()
            if a.camera_id == camera_id and a.first_fired_at_s >= cutoff
        )

    def _shift_count(self, now_s: float) -> int:
        cutoff = now_s - self.shift_window_s
        return sum(1 for a in self._alerts.values() if a.first_fired_at_s >= cutoff)

    def offer(
        self, camera_id: str, track_id: object, rule_name: str, fired_at_s: float
    ) -> ThrottledAlert | None:
        """Register one rule firing. Returns the alert to surface, or None if suppressed.

        A repeat of an existing (camera, track, rule) is always accepted and
        folded in — repeats never count against either budget. A genuinely
        new alert is admitted only while both the per-camera 24 h count and
        the shift-wide count are under budget; over budget, it is dropped
        (not queued) so a flooded camera cannot starve the rest of the shift.
        """
        self._prune(fired_at_s)
        key = (camera_id, track_id, rule_name)

        existing = self._alerts.get(key)
        if existing is not None:
            existing.repeat_count += 1
            existing.last_fired_at_s = fired_at_s
            return existing

        if self._camera_count(camera_id, fired_at_s) >= self.per_camera_limit:
            return None
        if self._shift_count(fired_at_s) >= self.shift_budget:
            return None

        alert = ThrottledAlert(
            camera_id=camera_id,
            track_id=track_id,
            rule_name=rule_name,
            first_fired_at_s=fired_at_s,
            last_fired_at_s=fired_at_s,
        )
        self._alerts[key] = alert
        return alert

    def active_alerts(self) -> list[ThrottledAlert]:
        return list(self._alerts.values())

    def dismiss(self, camera_id: str, track_id: object, rule_name: str) -> ThrottledAlert | None:
        """Remove an alert from the active set, e.g. once an operator dismisses it.

        Returns the removed alert so a caller can feed it to
        `baseline.update_on_dismissal` before it's gone — this module has no
        opinion on the baseline; that wiring lives in `engine.py`.
        """
        return self._alerts.pop((camera_id, track_id, rule_name), None)
