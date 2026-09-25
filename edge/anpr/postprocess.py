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


@dataclass(frozen=True)
class GrammarResult:
    valid: bool
    script: str  # 'devanagari' | 'latin'
    fields: dict  # named groups on success, {} on failure


def validate_nepal(text: str) -> GrammarResult:
    text = unicodedata.normalize("NFC", text)
    match = NEPAL_PLATE_RE.match(text)
    if not match:
        return GrammarResult(valid=False, script="devanagari", fields={})
    return GrammarResult(valid=True, script="devanagari", fields=match.groupdict())


def validate_bhutan(text: str) -> GrammarResult:
    text = unicodedata.normalize("NFC", text.upper().strip())
    match = BHUTAN_PLATE_RE.match(text)
    if not match:
        return GrammarResult(valid=False, script="latin", fields={})
    return GrammarResult(valid=True, script="latin", fields=match.groupdict())


def validate(text: str, script: str) -> GrammarResult:
    if script == "latin":
        return validate_bhutan(text)
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
