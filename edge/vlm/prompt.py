"""The grounded prompt, and the deterministic sentence the prompt is modelled on.

Everything the explanation layer is allowed to say comes from one place: the
`FactSheet` built here from the Phase 4 rule evidence (`TrackVerdict.
evidence`, i.e. ``{rule_name: evidence_dict}``) plus the camera context. Three
consumers read the same `FactSheet`, so they cannot drift apart:

* `build_prompt`      what the vision-language model is shown and asked.
* `rule_sentence`     the deterministic sentence rendered purely from evidence.
                      It is the t=0 `reason` on a provisional alert and the
                      fallback whenever model output is rejected.
* `guardrails.check`  what a model sentence may claim (numbers, times,
                      directions, object classes, rules, place names).

Target register — backend/src/data/mockData.js REASONS: factual, one
sentence, carries the time, the direction and the reason the movement is
unusual, no speculation, no statement about who a person is.

Prompt design for a small VLM (Moondream 2, ~2B parameters)
-----------------------------------------------------------
* Short. Every extra instruction a 2B model reads dilutes the ones that
  matter, so the whole template is a dozen lines.
* Facts before instructions, as ``key: value`` pairs — small models copy
  concrete tokens far more reliably than they follow abstract rules, so the
  prompt hands them the exact tokens (``02:41``, ``north``, ``06:00-20:00``)
  they should reuse.
* Few-shot with three examples in exactly the ``Facts: … / Reason: …`` shape
  the real query ends with. The examples are rendered by `rule_sentence`
  from synthetic evidence (different places, times and numbers from any real
  event, and not copied from the mock fixtures), so the model learns the
  register rather than a sentence to parrot — and if it does parrot an
  example's number or place, the guardrail rejects it because that fact is
  not in this event's evidence.
* Instructions are phrased as short prohibitions the guardrail enforces
  anyway; the prompt reduces the rejection rate, the guardrail guarantees
  the result.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable
from zoneinfo import ZoneInfo

from rules.engine import (
    RULE_FENCE_CROSSED,
    RULE_GROUP_FORMED,
    RULE_LOITERING,
    RULE_OUT_OF_HOURS,
    RULE_WRONG_DIRECTION,
)

POST_TIMEZONE = ZoneInfo("Asia/Kolkata")

MAX_REASON_CHARS = 220

# Canonical detector classes (docs/ARCHITECTURE_V2.md section 4.2,
# `detection.class`) and the noun each renders as.
OBJECT_NOUNS: dict[str, str] = {
    "person": "person",
    "truck": "truck",
    "car": "car",
    "two_wheeler": "two-wheeler",
    "cart": "cart",
}
VEHICLE_CLASSES = frozenset({"truck", "car", "two_wheeler", "cart"})

COMPASS_8 = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")

NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}

# Local-time bands used both to phrase facts and to check time-of-day words.
NIGHT_START = dt_time(18, 30)
NIGHT_END = dt_time(5, 30)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@runtime_checkable
class FiredRuleLike(Protocol):
    """What the explanation layer reads off a Phase 4 `rules.engine.FiredRule`."""

    rule_name: str
    evidence: Mapping[str, Any]
    reason: str


@dataclass(frozen=True)
class CameraContext:
    """The camera row (frozen `{id, name, sector, ir, road}`) plus post configuration.

    `sanctioned` holds the post's sanctioned windows as ``("HH:MM", "HH:MM")``
    local-time pairs (Asia/Kolkata); an empty tuple means "not configured" and
    no window is ever quoted.

    `image_right_bearing_deg` is the per-camera calibration that turns an
    image heading into a compass direction: the compass bearing (degrees
    clockwise from north) of motion towards the right edge of the frame.
    Phase 4 headings are ``atan2(vy, vx)`` in image coordinates with y
    pointing down, i.e. already clockwise from +x, so
    ``bearing = (heading + image_right_bearing_deg) % 360``. When it is None
    the camera has no compass calibration and no compass word is ever
    rendered or accepted — only image-plane motion ("left to right across
    the frame", "up the frame", …).
    """

    camera_id: str
    name: str
    sector: str
    ir: bool = False
    sanctioned: tuple[tuple[str, str], ...] = ()
    image_right_bearing_deg: float | None = None

    @classmethod
    def from_camera_row(
        cls,
        row: Mapping[str, Any],
        *,
        sanctioned: Iterable[Any] = (),
        image_right_bearing_deg: float | None = None,
    ) -> "CameraContext":
        """Build from the frozen camera row; `sanctioned` accepts `rules.hours.SanctionedWindow`s too."""
        return cls(
            camera_id=str(row.get("id", "")),
            name=str(row.get("name", "") or row.get("id", "")),
            sector=str(row.get("sector", "")),
            ir=bool(row.get("ir", False)),
            sanctioned=tuple(_window_pair(w) for w in sanctioned),
            image_right_bearing_deg=image_right_bearing_deg,
        )


@dataclass(frozen=True)
class EventContext:
    """Everything about the event that is not rule evidence.

    `captured_at` is epoch seconds (the `Frame.captured_at` convention);
    it is rendered in Asia/Kolkata regardless of the host clock.
    `object_class` is the fused detection's canonical class.
    """

    camera: CameraContext
    captured_at: float
    object_class: str = "person"

    @property
    def local_dt(self) -> datetime:
        ts = self.captured_at if _finite(self.captured_at) else 0.0
        return datetime.fromtimestamp(ts, tz=POST_TIMEZONE)


def _window_pair(w: Any) -> tuple[str, str]:
    if isinstance(w, (tuple, list)) and len(w) == 2:
        return (_hhmm_str(w[0]), _hhmm_str(w[1]))
    start, end = getattr(w, "start", None), getattr(w, "end", None)
    return (_hhmm_str(start), _hhmm_str(end))


def _hhmm_str(v: Any) -> str:
    if isinstance(v, dt_time):
        return v.strftime("%H:%M")
    s = str(v).strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", s)
    if not m:
        raise ValueError(f"not a HH:MM time: {v!r}")
    return f"{int(m.group(1)):02d}:{m.group(2)}"


# --------------------------------------------------------------------------
# The fact sheet
# --------------------------------------------------------------------------


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _num(d: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(d, Mapping):
        return None
    v = d.get(key)
    return float(v) if _finite(v) else None


def _clean_label(v: Any) -> str | None:
    """An operator-supplied label (fence name) made safe to put in a sentence."""
    if not isinstance(v, str):
        return None
    s = re.sub(r"[^\w\s-]", " ", v)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:32] or None


def compass_name(bearing_deg: float) -> str:
    return COMPASS_8[int(((bearing_deg % 360.0) + 22.5) // 45.0) % 8]


def fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{round(seconds)} s"
    return f"{round(seconds / 60)} min"


def fmt_metres(m: float) -> str:
    return f"{m:.1f} m" if m < 10 else f"{round(m)} m"


def count_phrase(n: int) -> str:
    return NUMBER_WORDS.get(n, str(n))


def is_night(t: dt_time) -> bool:
    return t >= NIGHT_START or t < NIGHT_END


@dataclass(frozen=True)
class NumericFact:
    value: float
    kind: str  # "duration_s" | "distance_m" | "angle_deg" | "count" | "label" | "hour"


@dataclass
class FactSheet:
    """Every checkable fact about one event, in operator units."""

    object_class: str
    count: int
    local_hhmm: str
    local_hms: str
    night: bool
    site: str
    sector: str
    camera_name: str
    ir: bool
    sanctioned: tuple[tuple[str, str], ...]
    rules: frozenset[str]
    other_rules: tuple[str, ...] = ()
    fence_name: str | None = None
    fence_direction: str | None = None  # "inbound" | "outbound"
    dwell_s: float | None = None
    loiter_net_m: float | None = None
    heading_deg: float | None = None
    angle_off_deg: float | None = None
    compass: str | None = None
    bearing_deg: float | None = None
    # Image-plane motion for cameras without compass calibration:
    # "left to right across the frame" | "right to left across the frame" |
    # "up the frame" | "down the frame".
    frame_motion: str | None = None
    group_size: int | None = None
    group_duration_s: float | None = None
    numbers: list[NumericFact] = field(default_factory=list)
    proper_words: frozenset[str] = frozenset()

    @property
    def noun(self) -> str:
        return OBJECT_NOUNS.get(self.object_class, "object")

    @property
    def subject(self) -> str:
        if self.group_size and self.group_size > 1:
            return f"Group of {count_phrase(self.group_size)}"
        n = self.noun
        return n[:1].upper() + n[1:]

    def window_text(self, sep: str = "–") -> str:
        return " and ".join(f"{a}{sep}{b}" for a, b in self.sanctioned)

    def prompt_facts(self) -> list[str]:
        """The ``key: value`` facts shown to the model, in operator units only."""
        obj = (
            f"group of {self.group_size} people"
            if self.object_class == "person" and self.group_size and self.group_size > 1
            else f"group of {self.group_size} {self.noun}s"
            if self.group_size and self.group_size > 1
            else self.noun
        )
        facts = [
            f"object: {obj}",
            f"place: {self.site}" + (f", {self.sector} sector" if self.sector else ""),
            f"camera: {'infrared' if self.ir else 'day'}",
            f"time: {self.local_hhmm} IST",
        ]
        if self.sanctioned and RULE_OUT_OF_HOURS in self.rules:
            facts.append(f"sanctioned hours: {self.window_text('-')}")
        if RULE_OUT_OF_HOURS in self.rules:
            facts.append("rule: outside sanctioned hours")
        if RULE_WRONG_DIRECTION in self.rules:
            way = self.compass or self.frame_motion
            facts.append(
                f"rule: moving {way}, against the learnt flow" if way else "rule: against the learnt flow"
            )
        if RULE_FENCE_CROSSED in self.rules:
            facts.append(
                "rule: crossed the "
                + (f"{self.fence_name} " if self.fence_name else "")
                + "fence line"
                + (f" {self.fence_direction}" if self.fence_direction else "")
            )
        if RULE_LOITERING in self.rules:
            facts.append(
                "rule: halted"
                + (f" {fmt_duration(self.dwell_s)}" if self.dwell_s is not None else "")
                + (f" within {fmt_metres(self.loiter_net_m)}" if self.loiter_net_m is not None else "")
            )
        if RULE_GROUP_FORMED in self.rules:
            facts.append(
                "rule: held together"
                + (f" for {fmt_duration(self.group_duration_s)}" if self.group_duration_s is not None else "")
            )
        for name in self.other_rules:
            facts.append(f"rule: {name}")
        return facts


_PROPER_SPLIT = re.compile(r"[A-Za-z][A-Za-z'’]*")


def _words(*texts: str | None) -> set[str]:
    out: set[str] = set()
    for t in texts:
        if t:
            out.update(w.casefold() for w in _PROPER_SPLIT.findall(t))
    return out


def _site_of(name: str) -> str:
    """'BOP Raxaul — Pillar 42' -> 'Pillar 42' (the part an operator names a spot by)."""
    parts = re.split(r"\s+[—–-]\s+", name.strip())
    site = parts[-1].strip() if parts else ""
    return site or name.strip() or "this camera"


def build_facts(evidence: Mapping[str, Any] | None, context: EventContext) -> FactSheet:
    """Turn ``{rule_name: evidence_dict}`` + context into a `FactSheet`. Never raises on bad evidence."""
    ev: Mapping[str, Any] = evidence if isinstance(evidence, Mapping) else {}
    cam = context.camera
    local = context.local_dt
    obj = context.object_class if context.object_class in OBJECT_NOUNS else "person"

    known = {RULE_FENCE_CROSSED, RULE_GROUP_FORMED, RULE_LOITERING, RULE_OUT_OF_HOURS, RULE_WRONG_DIRECTION}
    rules = frozenset(str(k) for k in ev.keys() if str(k) in known)
    other = tuple(
        name
        for name in (_clean_label(str(k)) for k in ev.keys() if str(k) not in known)
        if name
    )

    sheet = FactSheet(
        object_class=obj,
        count=1,
        local_hhmm=local.strftime("%H:%M"),
        local_hms=local.strftime("%H:%M:%S"),
        night=is_night(local.time()),
        site=_site_of(cam.name or cam.camera_id),
        sector=cam.sector.strip(),
        camera_name=cam.name,
        ir=bool(cam.ir),
        sanctioned=tuple(cam.sanctioned),
        rules=rules,
        other_rules=other,
    )
    nums: list[NumericFact] = [NumericFact(1, "count")]

    fence = ev.get(RULE_FENCE_CROSSED)
    if RULE_FENCE_CROSSED in rules and isinstance(fence, Mapping):
        sheet.fence_name = _clean_label(fence.get("fence_name"))
        d = fence.get("direction")
        sheet.fence_direction = {"in": "inbound", "out": "outbound"}.get(d) if isinstance(d, str) else None

    loiter = ev.get(RULE_LOITERING)
    if RULE_LOITERING in rules:
        dwell = _num(loiter, "dwell_s")
        sheet.dwell_s = dwell if dwell is not None and dwell >= 0 else None
        if isinstance(loiter, Mapping) and loiter.get("unit") == "m":
            net_m = _num(loiter, "net_displacement_m")
            sheet.loiter_net_m = net_m if net_m is not None and net_m >= 0 else None

    wrong = ev.get(RULE_WRONG_DIRECTION)
    if RULE_WRONG_DIRECTION in rules:
        heading = _num(wrong, "heading_deg")
        sheet.angle_off_deg = _num(wrong, "angle_from_baseline_deg")
        if heading is not None:
            sheet.heading_deg = heading % 360.0
            c, s_ = math.cos(math.radians(heading)), math.sin(math.radians(heading))
            if abs(c) >= abs(s_):
                sheet.frame_motion = f"{'left to right' if c > 0 else 'right to left'} across the frame"
            else:  # image y points down
                sheet.frame_motion = "down the frame" if s_ > 0 else "up the frame"
            if _finite(cam.image_right_bearing_deg):
                sheet.bearing_deg = (heading + float(cam.image_right_bearing_deg)) % 360.0
                sheet.compass = compass_name(sheet.bearing_deg)

    group = ev.get(RULE_GROUP_FORMED)
    if RULE_GROUP_FORMED in rules:
        size = _num(group, "size")
        if size is None and isinstance(group, Mapping) and isinstance(group.get("track_ids"), (list, tuple)):
            size = float(len(group["track_ids"]))
        if size is not None and 1 < size < 1000:
            sheet.group_size = int(size)
            sheet.count = int(size)
        dur = _num(group, "duration_s")
        sheet.group_duration_s = dur if dur is not None and dur >= 0 else None

    for v in (sheet.dwell_s, sheet.group_duration_s):
        if v is not None:
            nums.append(NumericFact(v, "duration_s"))
    if sheet.loiter_net_m is not None:
        nums.append(NumericFact(sheet.loiter_net_m, "distance_m"))
    if sheet.angle_off_deg is not None:
        nums.append(NumericFact(sheet.angle_off_deg, "angle_deg"))
    if sheet.group_size is not None:
        nums.append(NumericFact(sheet.group_size, "count"))
    for label in (cam.name, cam.sector, sheet.fence_name):
        for n in re.findall(r"\d+(?:\.\d+)?", label or ""):
            nums.append(NumericFact(float(n), "label"))
    sheet.numbers = nums

    sheet.proper_words = frozenset(
        _words(cam.name, cam.sector, sheet.site, sheet.fence_name, *other)
        | {"ist", "ir"}
        | {w for c in COMPASS_8 for w in c.split("-")}
    )
    return sheet


# --------------------------------------------------------------------------
# The deterministic sentence
# --------------------------------------------------------------------------


def _render(sheet: FactSheet) -> str:
    rules = sheet.rules
    used: set[str] = set()

    if RULE_FENCE_CROSSED in rules:
        verb = "crossing the " + (f"{sheet.fence_name} " if sheet.fence_name else "") + "fence line"
        if sheet.fence_direction:
            verb += f" {sheet.fence_direction}"
        used.add(RULE_FENCE_CROSSED)
    elif RULE_LOITERING in rules:
        verb = "halted" + (f" {fmt_duration(sheet.dwell_s)}" if sheet.dwell_s is not None else "")
        if sheet.loiter_net_m is not None:
            verb += f" within {fmt_metres(sheet.loiter_net_m)}"
        used.add(RULE_LOITERING)
    elif RULE_WRONG_DIRECTION in rules and sheet.compass:
        verb = f"moving {sheet.compass}"
    elif RULE_WRONG_DIRECTION in rules and sheet.frame_motion:
        verb = f"moving {sheet.frame_motion}"
    elif RULE_GROUP_FORMED in rules:
        verb = "holding together" + (
            f" for {fmt_duration(sheet.group_duration_s)}" if sheet.group_duration_s is not None else ""
        )
        used.add(RULE_GROUP_FORMED)
    else:
        verb = "moving"

    head = f"{sheet.subject} {verb} near {sheet.site} at {sheet.local_hhmm}"

    clauses: list[str] = []
    if RULE_OUT_OF_HOURS in rules:
        clauses.append(
            f"outside sanctioned hours ({sheet.window_text()})" if sheet.sanctioned else "outside sanctioned hours"
        )
    if RULE_WRONG_DIRECTION in rules:
        clauses.append("against the learnt flow on this route")
    if RULE_LOITERING in rules and RULE_LOITERING not in used:
        clauses.append("after halting" + (f" {fmt_duration(sheet.dwell_s)}" if sheet.dwell_s is not None else ""))
    if RULE_FENCE_CROSSED in rules and RULE_FENCE_CROSSED not in used:
        clauses.append("after crossing the fence line")
    if RULE_GROUP_FORMED in rules and RULE_GROUP_FORMED not in used:
        clauses.append(
            "held together"
            + (f" for {fmt_duration(sheet.group_duration_s)}" if sheet.group_duration_s is not None else "")
        )
    for name in sheet.other_rules:
        clauses.append(f"{name} rule fired")
    if not rules and not sheet.other_rules:
        clauses.append("flagged by both detection channels")
    if sheet.ir:
        clauses.append("seen on IR")

    sentence = head
    for clause in clauses:
        candidate = f"{sentence}, {clause}"
        if len(candidate) + 1 > MAX_REASON_CHARS:
            break
        sentence = candidate
    if len(sentence) + 1 > MAX_REASON_CHARS:
        sentence = sentence[: MAX_REASON_CHARS - 1].rsplit(" ", 1)[0].rstrip(",;")
    return sentence + "."


def rule_sentence(evidence: Mapping[str, Any] | None, context: EventContext) -> str:
    """Render one REASONS-register sentence purely from rule evidence and camera context.

    Deterministic, no model involved. Always returns a non-empty sentence of
    at most `MAX_REASON_CHARS` characters, including when `evidence` is empty
    or malformed (an event only reaches here after both channels agreed, so
    "flagged by both detection channels" is itself a measured fact).
    """
    return _render(build_facts(evidence, context))


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------

PROMPT_HEADER = "Write the alert reason for a border camera operator."
PROMPT_RULES = (
    "Rules: one sentence, under {max_chars} characters. Say what moved, where, the time, "
    "and why it is unusual. Use only the facts given and what is plainly visible. "
    "Do not guess intent. Never describe a person's face, identity, gender, age or origin. "
    "Never mention AI or any model."
)


def _example_contexts() -> list[tuple[dict, EventContext]]:
    """Three synthetic events for the few-shot block. Invented places/times, never a real camera."""

    def ctx(name: str, sector: str, ir: bool, hhmm: str, cls: str, windows=(), bearing=None) -> EventContext:
        h, m = (int(x) for x in hhmm.split(":"))
        ts = datetime(2026, 1, 15, h, m, 0, tzinfo=POST_TIMEZONE).timestamp()
        cam = CameraContext("EX", name, sector, ir, tuple(windows), bearing)
        return EventContext(cam, ts, cls)

    return [
        (
            {
                RULE_WRONG_DIRECTION: {"heading_deg": 90.0, "angle_from_baseline_deg": 172.0},
                RULE_OUT_OF_HOURS: {"local_time": "23:52:10"},
            },
            ctx("Post 5 — Check Gate 3", "Eastfield", False, "23:52", "car", [("06:00", "20:00")], 90.0),
        ),
        (
            {
                RULE_FENCE_CROSSED: {"fence_name": "Stream", "direction": "in"},
                RULE_GROUP_FORMED: {"size": 4, "duration_s": 35.0},
            },
            ctx("Post 8 — Stream Ford", "Westbank", True, "03:18", "person"),
        ),
        (
            {RULE_LOITERING: {"dwell_s": 75.0, "unit": "px"}, RULE_OUT_OF_HOURS: {"local_time": "21:34:02"}},
            ctx("Post 2 — Pillar 118", "Northgate", False, "21:34", "two_wheeler", [("06:00", "21:00")]),
        ),
    ]


def few_shot_examples() -> list[tuple[str, str]]:
    """``(facts_line, reason)`` pairs, both rendered from synthetic evidence by this module."""
    out = []
    for ev, ctx in _example_contexts():
        sheet = build_facts(ev, ctx)
        out.append(("; ".join(sheet.prompt_facts()), _render(sheet)))
    return out


def build_prompt(evidence: Mapping[str, Any] | None, context: EventContext) -> str:
    """The full text prompt for one event. Pure function of evidence + context."""
    sheet = build_facts(evidence, context)
    lines = [PROMPT_HEADER, PROMPT_RULES.format(max_chars=MAX_REASON_CHARS), "Examples:"]
    for facts, reason in few_shot_examples():
        lines.append(f"Facts: {facts}")
        lines.append(f"Reason: {reason}")
    lines.append("Now this event.")
    lines.append(f"Facts: {'; '.join(sheet.prompt_facts())}")
    lines.append("Reason:")
    return "\n".join(lines)


__all__ = [
    "MAX_REASON_CHARS",
    "POST_TIMEZONE",
    "CameraContext",
    "EventContext",
    "FactSheet",
    "FiredRuleLike",
    "NumericFact",
    "build_facts",
    "build_prompt",
    "few_shot_examples",
    "rule_sentence",
]
