"""Grammar validation, per-character confidence, and confusion correction.

Two plate grammars are validated, matching slide 4's two post types:

- Nepal (Devanagari, two-line): zone (1-2 Devanagari letters) + lot (2
  Devanagari digits) + vehicle class (1 Devanagari letter) + serial (4
  Devanagari digits), concatenated with no separators — the same convention
  DATASET_SPEC.md section 6.4 uses for ground truth. The exact zone and
  vehicle-class letters are NOT hardcoded here (they are OQ-8/OQ-9 in
  DATASET_SPEC.md, config-owned in datasets/plates/plates.yaml); this module
  only checks Devanagari-letter-vs-Devanagari-digit *structure*, which is
  known with certainty from MEASUREMENTS.md section 4's confirmed example.
- Bhutan (Latin, single line): `BP-N-ANNNN` per slide 4.

The confusion table is DATA-DRIVEN: `evaluate.py` measures it from actual
substitution errors the recognition model makes against ground truth on the
synthetic validation set (datasets/plates/labels/rec_gt_val.txt) and writes it
to `edge/anpr/data/confusions.json`. This module only ever *reads* that file;
it never hardcodes a guess about which glyphs look alike.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFUSIONS_PATH = HERE / "data" / "confusions.json"

# Devanagari letters (vowels, consonants, vowel signs) vs digits, by Unicode block.
_DEVANAGARI_DIGIT = re.compile(r"[०-९]")
_DEVANAGARI_LETTER = re.compile(r"[ऀ-ॡॢ-॥॰-ॿ]")

# Zone: a zone or province code, up to 4 code points so conjuncts such as "प्र" (प + virama + र,
# the provincial format in backend/src/data/mockData.js "प्र १ ख २३४५") are accepted; lot: 1-2
# digits (provincial plates use one); class: one letter, optionally with a vowel sign; serial: 1-4
# digits. The synthetic corpus (2-code-point zones, 2-digit lots, 4-digit serials) is a subset.
NEPAL_PLATE_RE = re.compile(
    r"^(?P<zone>[ऀ-ॡ॰-ॿ]{1,4})"
    r"(?P<lot>[०-९]{1,2})"
    r"(?P<vehicle_class>[क-ह][ऀ-ःा-ौ]?)"
    r"(?P<serial>[०-९]{1,4})$"
)

BHUTAN_PLATE_RE = re.compile(r"^(?P<prefix>BP)-(?P<district>\d)-(?P<letter>[A-Z])(?P<serial>\d{4})$")

# Province plates as photographed in the Kathmandu valley (2026-09-26 transcriptions): a province header, a two-digit
# office code, a three-digit lot, a class letter and the serial, e.g. "बागमती प्रदेश-०२ / ०३७ प / १६४३" ->
# "बागमतीप्रदेश०२०३७प१६४३", or with the older numbered header "प्रदेश ३-०२ / ०१३ प / ७१७३".
NEPAL_PROVINCES = ("कोशी", "मधेश", "मधेस", "बागमती", "गण्डकी", "लुम्बिनी", "कर्णाली", "सुदूरपश्चिम")
NEPAL_PROVINCE_RE = re.compile(
    r"^(?P<province>(?:" + "|".join(NEPAL_PROVINCES) + r")?प्रदेश[०-९]?)"
    r"(?P<office>[०-९]{2})"
    r"(?P<lot>[०-९]{3})"
    r"(?P<vehicle_class>[क-ह][ऀ-ःा-ौ]?)"
    r"(?P<serial>[०-९]{1,4})$"
)

# An inline province plate prints the number on one line and the small header above its gap; read on its own, the
# number line is lot + class + serial, and the header crop is province name + "प्रदेश" (+ digit) + office code.
PROVINCE_CORE_RE = re.compile(r"^[०-९]{3}[क-ह][ऀ-ःा-ौ]?[०-९]{1,4}$")
PROVINCE_HEADER_RE = re.compile(r"^(?:" + "|".join(NEPAL_PROVINCES) + r")?प्रदेश[०-९]?[०-९]{2}$")


def province_header(text: str) -> str | None:
    """A header reading cleaned to "बागमतीप्रदेश०२" when it is one, else None (spaces, hyphens, dots dropped)."""
    text = re.sub(r"[\s.\-]", "", unicodedata.normalize("NFC", text))
    return text if PROVINCE_HEADER_RE.match(text) else None


# Embossed Nepali plates are Latin: a province letter, one or two class letters and four digits ("B AC 5763"), under a
# small printed province name ("BAGMATI") that is not part of the registration.
NEPAL_EMBOSSED_RE = re.compile(r"^(?P<province>[A-Z])(?P<letters>[A-Z]{1,2})(?P<serial>\d{4})$")
_EMBOSSED_HEADERS = ("SUDURPASHCHIM", "SUDURPASCHIM", "BAGMATI", "GANDAKI", "LUMBINI", "KARNALI", "MADHESH", "MADHES", "KOSHI")


def strip_latin_header(text: str) -> str:
    """Canonical Latin reading of an embossed plate: "BAGMATI B AC 5763" -> "BAC5763".

    Upper-cases, keeps letters and digits only, drops a leading printed province name, and when the rest still does
    not parse but ends in a registration (a stray bolt or edge read as "E" or ")" in "EBAB3985"), keeps that ending.
    Bhutan plates keep their hyphens: validate_bhutan is tried on the raw reading first.
    """
    text = re.sub(r"[^A-Z0-9]", "", unicodedata.normalize("NFC", text).upper())
    for header in _EMBOSSED_HEADERS:
        if text.startswith(header) and len(text) > len(header):
            text = text[len(header):]
            break
    if not NEPAL_EMBOSSED_RE.match(text):
        tail = re.search(r"[A-Z]{2,3}\d{4}$", text)
        if tail and NEPAL_EMBOSSED_RE.match(tail.group(0)):
            return tail.group(0)
    return text


@dataclass(frozen=True)
class GrammarResult:
    valid: bool
    script: str  # 'devanagari' | 'latin'
    fields: dict  # named groups on success, {} on failure


def validate_nepal(text: str) -> GrammarResult:
    text = unicodedata.normalize("NFC", text)
    match = NEPAL_PLATE_RE.match(text) or NEPAL_PROVINCE_RE.match(text)
    if not match:
        return GrammarResult(valid=False, script="devanagari", fields={})
    return GrammarResult(valid=True, script="devanagari", fields=match.groupdict())


def validate_bhutan(text: str) -> GrammarResult:
    text = unicodedata.normalize("NFC", text.upper().strip())
    match = BHUTAN_PLATE_RE.match(text)
    if not match:
        return GrammarResult(valid=False, script="latin", fields={})
    return GrammarResult(valid=True, script="latin", fields=match.groupdict())


# Zonal codes of old-style plates (datasets/plates/plates.yaml, confirmed and unverified) plus codes seen on photographs.
KNOWN_ZONES = frozenset(("बा", "मे", "को", "सा", "ज", "ना", "ग", "धौ", "लु", "रा", "भे", "से", "म", "क", "स", "प्र"))


def extract_nepal_plate(text: str) -> str | None:
    """The longest stretch of a Devanagari reading that is a whole plate, or None.

    A background strip or a hallucinated province header read as an extra line ("बतपरेश०४बा२च९५८५") leaves junk around
    a correct registration; a plate cannot contain that junk, so the grammar can cut it away.
    """
    text = unicodedata.normalize("NFC", text)
    best, best_rank = None, None
    for start in range(len(text)):
        for end in range(start + 1, len(text) + 1):
            piece = text[start:end]
            match = NEPAL_PROVINCE_RE.match(piece) or NEPAL_PLATE_RE.match(piece)
            if not match:
                continue
            # The loose grammar also accepts junk ("परेश०४बा२": a four-letter "zone", "बा" as the class), so a known
            # zone code and a full four-digit serial rank first and length only breaks ties.
            fields = match.groupdict()
            rank = ("province" in fields or fields.get("zone") in KNOWN_ZONES, len(fields["serial"]) == 4, len(piece))
            if best_rank is None or rank > best_rank:
                best, best_rank = piece, rank
    return best


def validate_latin(text: str) -> GrammarResult:
    """A Latin reading is a plate if it is a Bhutan plate or, header stripped, an embossed Nepali plate."""
    bhutan = validate_bhutan(text)
    if bhutan.valid:
        return bhutan
    match = NEPAL_EMBOSSED_RE.match(strip_latin_header(text))
    if not match:
        return GrammarResult(valid=False, script="latin", fields={})
    return GrammarResult(valid=True, script="latin", fields=match.groupdict())


def validate(text: str, script: str) -> GrammarResult:
    if script == "latin":
        return validate_latin(text)
    return validate_nepal(text)


def per_character_confidence(text: str, line_confidence: float) -> list[float]:
    """Per-character confidence, approximated from the line-level score.

    PP-OCRv5's `TextRecognition.predict` reports one confidence per LINE, not
    per character (`return_word_box` gives per-word boxes, not per-glyph
    scores). Broadcasting the line score is therefore an approximation, not a
    measured per-character value — every caller of this function receives the
    same list length as `len(text)` and treats every entry as that
    approximation, never as an independently measured probability.
    """
    return [round(float(line_confidence), 4) for _ in text]


def _load_confusions() -> dict[str, list[str]]:
    if not CONFUSIONS_PATH.is_file():
        return {}
    try:
        doc = json.loads(CONFUSIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    table = doc.get("confusions", {}) if isinstance(doc, dict) else {}
    return {str(k): [str(v) for v in vs] for k, vs in table.items() if isinstance(vs, list)}


@dataclass(frozen=True)
class Correction:
    index: int
    original: str
    corrected: str


def apply_corrections(
    text: str,
    char_confidences: list[float],
    expected_kind: list[str] | None = None,
    confidence_threshold: float = 0.90,
) -> tuple[str, list[Correction]]:
    """Replace low-confidence characters using the measured confusion table.

    `expected_kind`, when given, is a same-length list of `'letter'` or
    `'digit'` (the grammar's structural expectation at that position, e.g.
    from `NEPAL_PLATE_RE`'s field layout). A correction is only applied when:
      1. the character's approximate confidence is below `confidence_threshold`, and
      2. the confusion table (measured, not guessed) has a candidate for it, and
      3. applying it does not contradict `expected_kind` at that position, when known.
    This is deliberately conservative: it is not required to fix every low-
    confidence character, only ones the DATA says are commonly confused.
    """
    table = _load_confusions()
    if not table:
        return text, []

    chars = list(text)
    corrections: list[Correction] = []
    for index, char in enumerate(chars):
        if index >= len(char_confidences) or char_confidences[index] >= confidence_threshold:
            continue
        candidates = table.get(char)
        if not candidates:
            continue
        kind = expected_kind[index] if expected_kind and index < len(expected_kind) else None
        for candidate in candidates:
            candidate_kind = "digit" if _DEVANAGARI_DIGIT.match(candidate) else (
                "letter" if _DEVANAGARI_LETTER.match(candidate) else None
            )
            if kind is not None and candidate_kind is not None and candidate_kind != kind:
                continue
            if candidate != char:
                corrections.append(Correction(index=index, original=char, corrected=candidate))
                chars[index] = candidate
            break

    return "".join(chars), corrections


def expected_kinds_for_nepal(match_fields: dict) -> list[str] | None:
    """Structural kind ('letter'/'digit') per character position, from a NEPAL_PLATE_RE match.

    Returns None when `match_fields` is empty (grammar did not match, so no
    structural expectation is known — corrections then run without a kind
    filter).
    """
    if not match_fields:
        return None
    order = ("zone", "lot", "vehicle_class", "serial")
    kinds: list[str] = []
    field_kind = {"zone": "letter", "lot": "digit", "vehicle_class": "letter", "serial": "digit"}
    for field in order:
        value = match_fields.get(field, "")
        kinds.extend([field_kind[field]] * len(value))
    return kinds
