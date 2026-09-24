"""Reject or repair vision-language output before it becomes an alert's `reason`.

`check(raw, facts)` returns a `GuardrailResult`. It first applies the only
repairs that cannot change meaning, then runs every content check and
collects *all* violations (so a log line explains exactly why a sentence was
dropped). Anything that fails is replaced upstream (`judge.fallback_reason`)
by the deterministic rule sentence — the operator never sees a rejected
sentence and never sees an empty one.

Safe repairs (meaning-preserving only)
--------------------------------------
* whitespace collapsed, wrapping quotes / markdown bullets stripped;
* a leading answer label or image preamble removed ("Reason:", "The image
  shows …") and the first letter re-capitalised;
* a trailing unterminated fragment after one complete sentence dropped —
  that is what a token-limit cut looks like ("… hours. The tru");
* a missing final full stop added, unless the text ends mid-clause
  ("… at 02:41, outside"), which is treated as truncation and rejected.

Not repaired, rejected
----------------------
* two or more complete sentences — which one carries the grounded fact is a
  judgement, and a model that ignores "one sentence" is not trusted;
* more than `MAX_REASON_CHARS` characters — cutting a sentence mid-clause
  can silently drop the "why it is unusual" part;
* a question;
* speculation, person-identifying claims, self-reference, vendor names;
* any fact not supported by the `FactSheet`: a number, time, compass or
  left/right direction, object class, rule claim ("halted", "fence",
  "against the flow", "sanctioned", "group"), time-of-day word, IR claim,
  or capitalised name that the evidence does not contain;
* a sentence that omits the event time, or states none of the rules that
  fired — the REASONS register always says when, and why it is unusual.

The lexicons below are data: add a pattern to the relevant tuple and every
caller picks it up. Each entry is ``(regex, label)``; the label is what shows
up in the violation detail.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Iterable

from rules.engine import (
    RULE_FENCE_CROSSED,
    RULE_GROUP_FORMED,
    RULE_LOITERING,
    RULE_OUT_OF_HOURS,
    RULE_WRONG_DIRECTION,
)

from .prompt import MAX_REASON_CHARS, VEHICLE_CLASSES, FactSheet

MIN_REASON_CHARS = 20
MIN_REASON_WORDS = 4

# --------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------

SPECULATION_LEXICON: tuple[tuple[str, str], ...] = (
    (r"\bappears?\s+to\b", "appears to"),
    (r"\bseem(s|ed|ing|ingly)?\b", "seems"),
    (r"\bapparent(ly)?\b", "apparently"),
    (r"\b(un)?likely\b", "likely"),
    (r"\bprobabl[ey]\b", "probably"),
    (r"\bpossib(le|ly|ility)\b", "possibly"),
    (r"\bperhaps\b|\bmaybe\b", "perhaps"),
    (r"\bmight\b|\bmay\b|\bcould\b|\bwould\b", "modal hedge"),
    (r"\bpresumabl[ey]\b|\bsupposedly\b|\ballegedly\b", "presumably"),
    (r"\bsuspicio\w*|\bsuspect\w*", "suspicion"),
    (r"\bintent\w*|\bintend\w*|\bmotive\w*|\bpurpose\w*", "intent"),
    (r"\bplan(s|ned|ning)?\b|\bprepar\w*", "planning"),
    (r"\battempt\w*|\btry(ing)?\b|\btries\b|\btried\b|\bin order to\b", "attempting"),
    (r"\bsmuggl\w*|\bcontraband\b|\btraffick\w*|\bpoach\w*", "smuggling"),
    (r"\billegal\w*|\billicit\w*|\bunlawful\w*|\bcriminal\w*|\bunauthori[sz]ed\b", "illegality"),
    (r"\bterror\w*|\bmilitant\w*|\binfiltrat\w*|\bintruder\w*|\binsurgen\w*", "threat label"),
    (r"\bthreat\w*|\bdanger\w*|\bhostile\b|\baggressive\w*", "threat"),
    (r"\bsneak\w*|\bcovert\w*|\bstealth\w*|\bevad\w*|\bevasi\w*|\bhid(e|es|ing|den)\b|\bescap\w*", "evasion"),
    (r"\bi\s+(think|believe|guess)\b|\bguess\w*", "opinion"),
    (r"\bsuggest\w*|\bindicat\w*|\bconsistent\s+with\b|\bimpl(y|ies|ied)\b", "inference"),
)

PERSON_IDENTITY_LEXICON: tuple[tuple[str, str], ...] = (
    (r"\bfac(e|es|ial)\b|\bfeatures\b", "face"),
    (r"\bidentif\w*|\bidentity\b|\bnamed\b|\bname\s+is\b|\bknown\s+as\b|\brecogni[sz]\w*", "identity"),
    (r"\bm[ae]n\b|\bwom[ae]n\b|\bboys?\b|\bgirls?\b|\bmale\b|\bfemale\b|\blad(y|ies)\b|\bgentlem[ae]n\b", "gender"),
    (r"\bhe\b|\bshe\b|\bhis\b|\bher\b|\bhim\b|\bhers\b", "gendered pronoun"),
    (r"\bchild\w*|\bkids?\b|\bteen\w*|\belderly\b|\bold\b|\byoung\w*|\byouth\w*|\baged\b|\bage\b", "age"),
    (
        r"\bnepal[ei]\w*|\bindians?\b|\bbhutanese\b|\btibetan\w*|\bbangladeshi\w*|\bchinese\b|\bpakistani\w*"
        r"|\bmadhesi\w*|\btharu\w*|\bforeign\w*|\bnational(s|ity)?\b|\bcitizen\w*",
        "origin",
    ),
    (r"\bethnic\w*|\brac(e|es|ial)\b|\bcaste\w*|\btrib(e|es|al)\b", "ethnicity"),
    (r"\bmuslim\w*|\bhindu\w*|\bsikh\w*|\bchristian\w*|\bbuddhist\w*|\breligio\w*", "religion"),
    (r"\bskin\b|\bcomplexion\b|\bbeard\w*|\bmoustache\w*|\bmustache\w*|\bhair\w*|\btattoo\w*|\bscar(s|red)?\b", "physical trait"),
    (r"\bwearing\b|\bdressed\b|\bcloth(es|ing)\b|\bshirt\w*|\bjacket\w*|\bkurta\w*|\bsar(i|ee)s?\b|\bturban\w*|\bcap\b|\bhat\b|\buniform\w*", "clothing"),
    (r"\bvillager\w*|\bresident\w*|\blocals?\b|\bfarmer\w*|\bstudent\w*|\bporter\w*|\blabou?rer\w*|\bsoldier\w*|\bofficer\w*", "role"),
)

SELF_REFERENCE_LEXICON: tuple[tuple[str, str], ...] = (
    (r"\bai\b|\bartificial\s+intelligence\b", "ai"),
    (r"\b(language|vision|neural|generative)\s+(model|network)\w*|\bmodel\b|\bllms?\b|\bvlms?\b", "model"),
    (r"\bchat\s*bot\w*|\bassistant\b|\balgorithm\w*", "assistant"),
    (r"\bi\b|\bi'm\b|\bi’m\b|\bmy\b|\bsorry\b", "first person"),
)

# Vendor names. Stored only as salted SHA-256 prefixes so the names never
# appear in source; `_token_hash` is compared against each lower-cased word,
# and against each pair of adjacent words joined (so a split spelling is
# caught as well).
_VENDOR_SALT = "tw-vlm:"
VENDOR_HASHES: frozenset[str] = frozenset(
    {
        "f043a75d5a7d6dab", "745a8d4a57c616cb", "372987385311720e", "ba0834f340bab8fc",
        "39101224e3912d01", "db6b90364711ad69", "bde3ebdf8c2f3912", "45adabde773267f3",
        "55467996f894358b", "db6427c2a8f09487", "2040f02a7cd21f34", "c62f0939ea646123",
        "6e838f94797b78bd", "7bac1dd179aadfe6", "a44fc69d79dd0c1b", "b5d2cd0a7be381ea",
        "6a86eefc20782801", "51738ead487a5c48", "330dff445d60291c", "5fa8088ca1556094",
        "c775da20a5a222b7", "822481d673471799", "1e7233ced72f361f", "736fbb17f4d11a85",
        "efdea1e61504b839", "73e09ce1e0b9e0e5", "3239d5e295fdb6f0", "054b3f2f3e86f2a5",
        "f1cc1ac74e421c19", "f1ffe93650094ce0", "ea8a155fe5260b27", "c677e01e098e1558",
        "1f8cacd77d858ca7", "19da2d05745aa54b",
    }
)


def _token_hash(token: str) -> str:
    return hashlib.sha256((_VENDOR_SALT + token.casefold()).encode()).hexdigest()[:16]


# Claims that are only true when the named rule fired.
RULE_CLAIM_LEXICON: tuple[tuple[str, str, frozenset[str]], ...] = (
    (r"\bfenc\w*", "fence", frozenset({RULE_FENCE_CROSSED})),
    (
        r"\bloiter\w*|\blinger\w*|\bhalt\w*|\bstopp?(ed|ing|s)?\b|\bstationary\b|\bidl(e|ed|ing)\b"
        r"|\bwait\w*|\bdwell\w*|\bparked\b|\bpaused?\b",
        "halted",
        frozenset({RULE_LOITERING}),
    ),
    (
        r"\bagainst\b|\bwrong[\s-]+(way|direction)\b|\bopposite\s+direction\b|\bcontra(ry|flow)\b|\bcounter[\s-]?flow\b",
        "against the flow",
        frozenset({RULE_WRONG_DIRECTION}),
    ),
    (
        r"\bsanction\w*|\bout[\s-]+of[\s-]+hours\b|\bcurfew\b|\b(restricted|prohibited)\s+hours\b|\bafter[\s-]+hours\b",
        "outside hours",
        frozenset({RULE_OUT_OF_HOURS}),
    ),
    (
        r"\bgroup\w*|\bformation\b|\bsingle\s+file\b|\btogether\b|\bcrowd\w*|\bgather\w*|\bcluster\w*|\bconvoy\w*",
        "group",
        frozenset({RULE_GROUP_FORMED}),
    ),
)

# Object words -> canonical detector class ("vehicle" = any vehicle class).
OBJECT_LEXICON: dict[str, str] = {
    **{w: "person" for w in ("person", "persons", "people", "pedestrian", "pedestrians", "figure",
                            "figures", "individual", "individuals", "human", "humans", "walker", "walkers")},
    **{w: "truck" for w in ("truck", "trucks", "lorry", "lorries", "pickup", "pickups", "pick-up",
                           "tanker", "tankers", "tipper", "tippers", "tractor", "tractors")},
    **{w: "car" for w in ("car", "cars", "sedan", "sedans", "suv", "suvs", "jeep", "jeeps", "van", "vans",
                         "taxi", "taxis", "hatchback", "hatchbacks")},
    **{w: "two_wheeler" for w in ("motorcycle", "motorcycles", "motorbike", "motorbikes", "bike", "bikes",
                                 "scooter", "scooters", "moped", "mopeds", "bicycle", "bicycles", "cycle",
                                 "cycles", "two-wheeler", "two-wheelers")},
    **{w: "cart" for w in ("cart", "carts", "handcart", "handcarts", "rickshaw", "rickshaws")},
    **{w: "vehicle" for w in ("vehicle", "vehicles")},
    **{w: "animal" for w in ("animal", "animals", "cow", "cows", "cattle", "dog", "dogs", "goat", "goats",
                            "buffalo", "buffaloes", "elephant", "elephants", "deer", "horse", "horses")},
}
PLURAL_OBJECT_WORDS = frozenset({w for w in OBJECT_LEXICON if w.endswith("s")} | {"people", "cattle"})

NUMBER_WORD_VALUES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "dozen": 12, "dozens": 24,
    "hundred": 100, "hundreds": 200, "thousand": 1000, "pair": 2, "couple": 2, "several": 3, "few": 3,
    "multiple": 2, "many": 5, "numerous": 5,
}

UNIT_KINDS: dict[str, str] = {
    **{u: "s" for u in ("s", "sec", "secs", "second", "seconds")},
    **{u: "min" for u in ("min", "mins", "minute", "minutes")},
    **{u: "h" for u in ("h", "hr", "hrs", "hour", "hours")},
    **{u: "m" for u in ("m", "metre", "metres", "meter", "meters")},
    **{u: "km" for u in ("km", "kms", "kilometre", "kilometres", "kilometer", "kilometers")},
    **{u: "deg" for u in ("°", "deg", "degree", "degrees")},
    **{u: "ampm" for u in ("am", "pm", "a.m", "p.m")},
    **{u: "px" for u in ("px", "pixel", "pixels", "%", "percent", "kg", "kmph", "kph")},
}

# Compass words -> bearing.
COMPASS_BEARINGS: dict[str, float] = {
    "north": 0, "northeast": 45, "east": 90, "southeast": 135,
    "south": 180, "southwest": 225, "west": 270, "northwest": 315,
}
COMPASS_TOLERANCE_DEG = 67.5  # the word's own sector plus half of each neighbour

TIME_OF_DAY_LEXICON: tuple[tuple[str, str], ...] = (
    (r"\bnight\w*|\bmidnight\b|\bnocturnal\b|\bovernight\b|\bafter\s+dark\b|\bdarkness\b|\bpre[\s-]?dawn\b", "night"),
    (r"\bdaylight\b|\bdaytime\b|\bafternoon\b|\bnoon\b|\bmidday\b|\bsunlit\b|\bbroad\s+day\w*", "day"),
)

IR_LEXICON = r"\binfra[\s-]?red\b|\bthermal\b"

ABBREVIATIONS = ("approx.", "e.g.", "i.e.", "vs.", "no.", "nos.", "st.", "a.m.", "p.m.")
META_PREFIX = re.compile(
    r"^(?:(?:reason|answer|sentence|output|alert|response|explanation)\s*[:\-–—]\s*"
    r"|(?:in\s+)?(?:the|this)\s+(?:image|frame|picture|photo|clip|scene|footage)\s*(?:shows|depicts|contains)?\s*[:,]?\s*)",
    re.IGNORECASE,
)
DANGLING_ENDINGS = frozenset(
    {"a", "an", "the", "and", "or", "but", "of", "to", "at", "in", "on", "near", "with", "by", "from",
     "for", "into", "outside", "inside", "towards", "toward", "across", "while", "after", "before"}
)


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}" if self.detail else self.code


@dataclass(frozen=True)
class GuardrailResult:
    accepted: bool
    text: str  # the repaired candidate; only meaningful when accepted
    violations: tuple[Violation, ...] = ()
    repairs: tuple[str, ...] = field(default=())

    @property
    def codes(self) -> frozenset[str]:
        return frozenset(v.code for v in self.violations)


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------


def _split_sentences(text: str) -> tuple[list[str], str]:
    """Complete sentences, plus any unterminated tail."""
    masked = text
    for abbr in ABBREVIATIONS:
        masked = re.sub(re.escape(abbr), lambda m: m.group(0).replace(".", "\x00"), masked, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[.!?])\s+", masked)
    complete: list[str] = []
    tail = ""
    for i, p in enumerate(parts):
        p = p.replace("\x00", ".").strip()
        if not p:
            continue
        if p[-1] in ".!?":
            complete.append(p)
        elif i == len(parts) - 1:
            tail = p
        else:  # pragma: no cover - split only happens after a terminator
            complete.append(p)
    return complete, tail


def repair(raw: str | None) -> tuple[str, list[str], list[Violation]]:
    """Apply only meaning-preserving repairs. Returns (text, repairs, structural violations)."""
    repairs: list[str] = []
    if raw is None or not isinstance(raw, str):
        return "", repairs, [Violation("empty", "no output")]

    text = re.sub(r"\s+", " ", raw.replace("​", "")).strip()
    if text != raw:
        repairs.append("whitespace")

    stripped = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+|#+\s*)", "", text)
    stripped = stripped.strip().strip("`*_").strip()
    quoted = re.fullmatch(r"[\"'“‘](.*)[\"'”’](\.?)", stripped)
    if quoted:
        stripped = (quoted.group(1) + quoted.group(2)).strip()
    for _ in range(2):
        new = META_PREFIX.sub("", stripped, count=1).strip()
        if new == stripped:
            break
        stripped = new
    if stripped != text:
        repairs.append("prefix")
        text = stripped
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    if not text:
        return "", repairs, [Violation("empty", "nothing after repair")]

    text = text.replace("!", ".")
    if "?" in text:
        return text, repairs, [Violation("question", "output is a question")]

    complete, tail = _split_sentences(text)
    if len(complete) >= 2:
        return text, repairs, [Violation("multi_sentence", f"{len(complete)} sentences")]
    if len(complete) == 1 and tail:
        repairs.append("trim_truncated_tail")
        text = complete[0]
    elif not complete:
        last = re.findall(r"[\w'’-]+", tail.casefold())
        if not last or last[-1] in DANGLING_ENDINGS or tail.rstrip()[-1:] in ",;:-–—":
            return text, repairs, [Violation("truncated", "ends mid-clause")]
        repairs.append("add_full_stop")
        text = tail.rstrip() + "."
    else:
        text = complete[0]
    text = re.sub(r"\.{2,}$", ".", text)
    return text, repairs, []


# --------------------------------------------------------------------------
# Content checks
# --------------------------------------------------------------------------


def _lexicon_hits(text: str, lexicon: Iterable[tuple[str, str]]) -> list[str]:
    return [label for pattern, label in lexicon if re.search(pattern, text, re.IGNORECASE)]


def _vendor_hits(text: str) -> bool:
    tokens = re.findall(r"[a-z]+", text.casefold())
    candidates = set(tokens) | {a + b for a, b in zip(tokens, tokens[1:])}
    return any(_token_hash(t) in VENDOR_HASHES for t in candidates)


_TIME_RE = re.compile(
    r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(?:\s*(a\.?m\.?|p\.?m\.?))?(?![\w:])", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"(?<![\w.:])(\d+(?:\.\d+)?)(?:\s*(°|%|[A-Za-z]+(?:\.[a-z])?))?")


def _to_24h(h: int, suffix: str | None) -> int:
    if not suffix:
        return h
    s = suffix.casefold().replace(".", "")
    if s == "pm" and h < 12:
        return h + 12
    if s == "am" and h == 12:
        return 0
    return h


def _allowed_times(facts: FactSheet) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for t in [facts.local_hhmm, *(x for w in facts.sanctioned for x in w)]:
        h, m = t.split(":")[:2]
        out.add((int(h), int(m)))
    return out


def _check_times(text: str, facts: FactSheet) -> tuple[str, list[Violation]]:
    allowed = _allowed_times(facts)
    bad: list[Violation] = []
    for m in _TIME_RE.finditer(text):
        h = _to_24h(int(m.group(1)), m.group(4))
        hhmm = f"{h:02d}:{m.group(2)}"
        if (h, int(m.group(2))) not in allowed:
            bad.append(Violation("invented_time", m.group(0)))
        elif m.group(3) is not None and hhmm == facts.local_hhmm and f"{hhmm}:{m.group(3)}" != facts.local_hms:
            bad.append(Violation("invented_time", m.group(0)))  # right minute, invented seconds
    return _TIME_RE.sub(" ", text), bad


def _num_ok(x: float, unit: str | None, facts: FactSheet) -> bool:
    kind = UNIT_KINDS.get(unit.casefold().rstrip(".")) if unit else None
    nums = facts.numbers

    def any_of(kinds: set[str], pred) -> bool:
        return any(n.kind in kinds and pred(n.value) for n in nums)

    dur = lambda secs: any_of({"duration_s"}, lambda v: abs(secs - v) <= max(1.0, 0.05 * v))  # noqa: E731
    if kind == "s":
        return dur(x)
    if kind == "min":
        return any_of({"duration_s"}, lambda v: abs(x * 60 - v) <= max(30.0, 0.1 * v))
    if kind == "h":
        return any_of({"duration_s"}, lambda v: abs(x * 3600 - v) <= max(900.0, 0.1 * v))
    if kind == "m":
        return any_of({"distance_m"}, lambda v: abs(x - v) <= max(0.5, 0.1 * v))
    if kind == "km":
        return any_of({"distance_m"}, lambda v: abs(x * 1000 - v) <= max(50.0, 0.1 * v))
    if kind == "deg":
        return any_of({"angle_deg"}, lambda v: abs(x - v) <= 2.0)
    if kind == "ampm":
        h = int(x) % 12
        event_h = int(facts.local_hhmm.split(":")[0])
        return x == int(x) and h in {event_h % 12}
    if kind == "px":
        return False
    # no unit: a count or a label number (e.g. "Pillar 42") matches exactly;
    # a bare measurement must be close to some measured value.
    return any(
        (n.kind in ("count", "label") and x == n.value)
        or (n.kind in ("duration_s", "distance_m", "angle_deg") and abs(x - n.value) <= max(1.0, 0.05 * n.value))
        for n in nums
    )


def _check_numbers(text: str, facts: FactSheet) -> list[Violation]:
    bad = []
    for m in _NUMBER_RE.finditer(text):
        try:
            x = float(m.group(1))
        except ValueError:  # pragma: no cover
            continue
        unit = m.group(2)
        if unit and unit.casefold().rstrip(".") not in UNIT_KINDS:
            unit = None
        if not math.isfinite(x) or not _num_ok(x, unit, facts):
            bad.append(Violation("invented_number", m.group(0).strip()))
    counts = {n.value for n in facts.numbers if n.kind in ("count", "label")}
    for m in re.finditer(r"\b(" + "|".join(NUMBER_WORD_VALUES) + r")\b(?!-wheel)", text, re.IGNORECASE):
        word = m.group(1).casefold()
        if word == "one":
            continue  # "one" is always true of a single tracked object and too common to police
        if NUMBER_WORD_VALUES[word] not in counts:
            bad.append(Violation("invented_number", m.group(1)))
    return bad


def _check_directions(text: str, facts: FactSheet) -> list[Violation]:
    bad = []
    for m in re.finditer(
        r"\b(?:(north|south)(?:[\s-]?(east|west))?|(east|west))(?:ward|wards|bound|erly|ern)?\b",
        text,
        re.IGNORECASE,
    ):
        word = ((m.group(1) or "") + (m.group(2) or m.group(3) or "")).casefold()
        if facts.bearing_deg is None:
            bad.append(Violation("invented_direction", m.group(0).strip()))
            continue
        d = abs(COMPASS_BEARINGS[word] - facts.bearing_deg) % 360.0
        if min(d, 360.0 - d) > COMPASS_TOLERANCE_DEG:
            bad.append(Violation("invented_direction", m.group(0).strip()))
    motion = facts.frame_motion or ""
    for m in re.finditer(r"\b(left|right)\s+to\s+(left|right)\b|\b(up|down)\s+the\s+frame\b", text, re.IGNORECASE):
        phrase = " ".join(m.group(0).casefold().split())
        if phrase not in motion:
            bad.append(Violation("invented_direction", m.group(0)))
    return bad


def _check_objects(text: str, facts: FactSheet) -> list[Violation]:
    bad = []
    for m in re.finditer(r"[A-Za-z][A-Za-z-]*", text):
        w = m.group(0).casefold()
        cls = OBJECT_LEXICON.get(w)
        if cls is None:
            continue
        if cls == "vehicle":
            ok = facts.object_class in VEHICLE_CLASSES
        else:
            ok = cls == facts.object_class
        if ok and w in PLURAL_OBJECT_WORDS and facts.count <= 1:
            ok = False
        if not ok:
            bad.append(Violation("invented_object", m.group(0)))
    return bad


def _check_rule_claims(text: str, facts: FactSheet) -> list[Violation]:
    bad = []
    for pattern, label, needs in RULE_CLAIM_LEXICON:
        if re.search(pattern, text, re.IGNORECASE) and not (needs & facts.rules):
            if label == "group" and facts.count > 1:
                continue
            bad.append(Violation("invented_claim", label))
    for pattern, band in TIME_OF_DAY_LEXICON:
        if re.search(pattern, text, re.IGNORECASE) and (band == "night") != facts.night:
            bad.append(Violation("invented_claim", f"{band}-time word at {facts.local_hhmm}"))
    if (re.search(IR_LEXICON, text, re.IGNORECASE) or re.search(r"\bIR\b", text)) and not facts.ir:
        bad.append(Violation("invented_claim", "IR on a day camera"))
    return bad


def _check_completeness(text: str, facts: FactSheet) -> list[Violation]:
    """The register requires the time and at least one reason the movement is unusual."""
    bad = []
    event_h, event_m = (int(x) for x in facts.local_hhmm.split(":"))
    if not any(
        (_to_24h(int(m.group(1)), m.group(4)), int(m.group(2))) == (event_h, event_m)
        for m in _TIME_RE.finditer(text)
    ):
        bad.append(Violation("missing_time", f"event time {facts.local_hhmm} not stated"))
    if facts.rules:
        stated = any(
            re.search(pattern, text, re.IGNORECASE) and (needs & facts.rules)
            for pattern, _label, needs in RULE_CLAIM_LEXICON
        )
        if not stated and RULE_WRONG_DIRECTION in facts.rules:
            stated = bool(
                re.search(r"\b(north|south|east|west)\w*|\b(left|right)\s+to\s+(left|right)\b|\b(up|down)\s+the\s+frame\b",
                          text, re.IGNORECASE)
            )
        if not stated:
            bad.append(Violation("missing_reason", "no fired rule is stated"))
    return bad


def _check_names(text: str, facts: FactSheet) -> list[Violation]:
    """Any capitalised word after the first must be a known place/label word."""
    bad = []
    words = list(re.finditer(r"[A-Za-z][A-Za-z'’]*", text))
    for i, m in enumerate(words):
        if i == 0:
            continue
        w = m.group(0)
        if w[0].isupper() and w.casefold() not in facts.proper_words:
            bad.append(Violation("invented_name", w))
    return bad


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def check(raw: str | None, facts: FactSheet) -> GuardrailResult:
    """Repair what is safe, then accept only a sentence every fact of which the evidence supports."""
    text, repairs, violations = repair(raw)
    if violations and violations[0].code in ("empty",):
        return GuardrailResult(False, "", tuple(violations), tuple(repairs))

    if len(text) > MAX_REASON_CHARS:
        violations.append(Violation("too_long", f"{len(text)} > {MAX_REASON_CHARS} chars"))
    if len(text) < MIN_REASON_CHARS or len(re.findall(r"\w+", text)) < MIN_REASON_WORDS:
        violations.append(Violation("too_short", f"{len(text)} chars"))

    for label in _lexicon_hits(text, SPECULATION_LEXICON):
        violations.append(Violation("speculation", label))
    for label in _lexicon_hits(text, PERSON_IDENTITY_LEXICON):
        violations.append(Violation("person_identity", label))
    for label in _lexicon_hits(text, SELF_REFERENCE_LEXICON):
        violations.append(Violation("self_reference", label))
    if _vendor_hits(text):
        violations.append(Violation("vendor", "vendor name"))

    remaining, time_bad = _check_times(text, facts)
    violations += time_bad
    violations += _check_numbers(remaining, facts)
    violations += _check_directions(text, facts)
    violations += _check_objects(text, facts)
    violations += _check_rule_claims(text, facts)
    violations += _check_names(text, facts)
    violations += _check_completeness(text, facts)

    return GuardrailResult(not violations, text, tuple(violations), tuple(repairs))


__all__ = [
    "GuardrailResult",
    "MIN_REASON_CHARS",
    "PERSON_IDENTITY_LEXICON",
    "RULE_CLAIM_LEXICON",
    "SPECULATION_LEXICON",
    "SELF_REFERENCE_LEXICON",
    "Violation",
    "check",
    "repair",
]
