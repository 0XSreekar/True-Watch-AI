"""Dual-path plate recognition: Devanagari (Nepal) and Latin (Bhutan).

MEASUREMENTS.md section 4: on a synthetic plate, PP-OCRv5 Devanagari read
"बा १२ प १२३४" correctly (0.989 / 0.990); the Latin model returned "9 9238" —
wrong, because the digits are Devanagari glyphs that a Latin recognition head
was never trained to see as digits, not because the Latin model is bad at its
own job. Bhutan posts (slide 4) use Latin alphanumeric plates (`BP-N-ANNNN`),
so both recognition heads are real production paths here, not a comparison
fixture — `compare_latin.py` is the fixture; this module is what actually runs.

Both heads are PP-OCRv5 recognition models (Apache-2.0) served through
`paddleocr.TextRecognition`, a recognition-ONLY pipeline: detection and line
splitting already happened in `detect_plate.py` / `rectify.py`, so this module
never re-runs text detection.

Script routing runs the Devanagari head, then the Latin head, on every line,
and picks per-plate by which head's output actually looks like its own
script (Unicode range) combined with recognition confidence — never by
guessing the script from the image before either head has read it. The
chosen path is always reported (`RecognitionResult.script`), per the PDF's
"reports which path was used".
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

log = logging.getLogger("truewatch.edge.anpr")

DEVANAGARI_MODEL_NAME = "devanagari_PP-OCRv5_mobile_rec"
LATIN_MODEL_NAME = "en_PP-OCRv5_mobile_rec"

_DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")
_LATIN_ALNUM_RANGE = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class LineReading:
    text: str  # NFC-normalised
    confidence: float  # 0..1


@dataclass(frozen=True)
class RecognitionResult:
    text: str  # NFC-normalised, lines joined with no separator (DATASET_SPEC 6.4)
    script: str  # 'devanagari' | 'latin'
    line_confidences: list[float]
    lines: list[LineReading]


def _model_dir_env(var: str) -> str | None:
    """A fine-tuned checkpoint directory, if the Kaggle notebook's output was installed here.

    Unset by default: the pretrained PP-OCRv5 head is used until fine-tuning
    (edge/anpr/notebooks/kaggle_ocr_finetune.ipynb) has produced a checkpoint and
    an operator points this at it. This module never fails when it is unset.
    """
    value = os.environ.get(var, "").strip()
    return value or None


@lru_cache(maxsize=1)
def _devanagari_recogniser():
    from paddleocr import TextRecognition

    model_dir = _model_dir_env("ANPR_DEVANAGARI_MODEL_DIR")
    kwargs = {"model_name": DEVANAGARI_MODEL_NAME}
    if model_dir:
        kwargs["model_dir"] = model_dir
        log.info("anpr: loading fine-tuned Devanagari recognition model from %s", model_dir)
    else:
        log.info("anpr: loading pretrained %s (no fine-tune installed)", DEVANAGARI_MODEL_NAME)
    return TextRecognition(**kwargs)


@lru_cache(maxsize=1)
def _latin_recogniser():
    from paddleocr import TextRecognition

    model_dir = _model_dir_env("ANPR_LATIN_MODEL_DIR")
    kwargs = {"model_name": LATIN_MODEL_NAME}
    if model_dir:
        kwargs["model_dir"] = model_dir
    return TextRecognition(**kwargs)


def _run(recogniser, image: np.ndarray) -> LineReading:
    (result,) = list(recogniser.predict(image))
    text = unicodedata.normalize("NFC", str(result.get("rec_text", "")))
    score = float(result.get("rec_score", 0.0) or 0.0)
    return LineReading(text=text, confidence=score)


def recognise_line(image: np.ndarray) -> tuple[LineReading, LineReading]:
    """Run BOTH heads on one line image. Returns (devanagari, latin) readings."""
    devanagari = _run(_devanagari_recogniser(), image)
    latin = _run(_latin_recogniser(), image)
    return devanagari, latin


def _script_of(text: str) -> str:
    """Which script a recognised string is actually written in."""
    if _DEVANAGARI_RANGE.search(text):
        return "devanagari"
    if _LATIN_ALNUM_RANGE.search(text):
        return "latin"
    return "unknown"


def _choose_script(devanagari_lines: Sequence[LineReading], latin_lines: Sequence[LineReading]) -> str:
    """Pick the script whose OWN head actually produced text in that script, by confidence.

    Both heads always run (§ module docstring: routing never happens before
    either head has read the pixels). A head whose output contains no
    character of its own script did not recognise anything meaningful — e.g.
    the Latin head emitting only Devanagari-shaped confusions as punctuation
    is not evidence for the Latin path.
    """
    devanagari_hit = any(_script_of(line.text) == "devanagari" for line in devanagari_lines)
    latin_hit = any(_script_of(line.text) == "latin" for line in latin_lines)

    devanagari_conf = sum(l.confidence for l in devanagari_lines) / max(1, len(devanagari_lines))
    latin_conf = sum(l.confidence for l in latin_lines) / max(1, len(latin_lines))

    if devanagari_hit and not latin_hit:
        return "devanagari"
    if latin_hit and not devanagari_hit:
        return "latin"
    if devanagari_hit and latin_hit:
        return "devanagari" if devanagari_conf >= latin_conf else "latin"
    # Neither head produced a character of its own script (a badly degraded
    # plate): fall back to whichever head is more confident in its garbage,
    # which is still the best available signal.
    return "devanagari" if devanagari_conf >= latin_conf else "latin"


def _route_by_grammar(devanagari_lines: Sequence[LineReading], latin_lines: Sequence[LineReading]) -> str | None:
    """A reading that parses as a whole plate in its own script's grammar decides the route.

    Script characters and confidence alone sent 1.4% of synthetic Nepali plates to the Latin head
    (a degraded Devanagari line can look like Latin letters to it). A complete, valid Nepali plate
    from the Devanagari head, or a valid Bhutan BP-N-ANNNN plate from the Latin head, is much
    stronger evidence. Returns None when neither or both parse, leaving the decision to
    _choose_script.
    """
    try:  # imported as anpr.recognise (tests, the edge app) or as a top-level module (the anpr scripts)
        from .postprocess import validate_latin, validate_nepal
    except ImportError:
        from postprocess import validate_latin, validate_nepal

    def joined(lines: Sequence[LineReading]) -> str:
        return unicodedata.normalize("NFC", "".join(line.text.replace(" ", "") for line in lines))

    nepal = validate_nepal(joined(devanagari_lines)).valid
    bhutan = validate_latin("".join(line.text for line in latin_lines).replace(" ", "")).valid
    if nepal and not bhutan:
        return "devanagari"
    if bhutan and not nepal:
        return "latin"
    return None


def recognise_plate(line_images: Sequence[np.ndarray]) -> RecognitionResult:
    """Recognise a plate's line images (from rectify.prepare_lines) and route by script."""
    if not line_images:
        return RecognitionResult(text="", script="devanagari", line_confidences=[], lines=[])

    devanagari_lines: list[LineReading] = []
    latin_lines: list[LineReading] = []
    for image in line_images:
        devanagari, latin = recognise_line(image)
        devanagari_lines.append(devanagari)
        latin_lines.append(latin)

    script = _route_by_grammar(devanagari_lines, latin_lines) or _choose_script(devanagari_lines, latin_lines)
    chosen = devanagari_lines if script == "devanagari" else latin_lines

    # DATASET_SPEC.md section 6.4: the ground-truth string is the plate's
    # characters with inter-field spaces removed; joining lines with no
    # separator reproduces the same convention for a predicted string.
    text = unicodedata.normalize("NFC", "".join(line.text.replace(" ", "") for line in chosen))
    if script == "latin":
        try:
            from .postprocess import strip_latin_header
        except ImportError:
            from postprocess import strip_latin_header
        text = strip_latin_header(text)  # an embossed plate's printed "BAGMATI" is not part of the registration
    return RecognitionResult(
        text=text,
        script=script,
        line_confidences=[round(line.confidence, 4) for line in chosen],
        lines=chosen,
    )


def read_plate(plate_bgr: np.ndarray) -> RecognitionResult:
    """Rectify, split and recognise one plate crop, falling back to other readings when the first fails the grammar.

    The plate's shape picks one-line or two-line first (rectify.ONE_LINE_MIN_ASPECT). Tilted plates (levelled by
    rectify.deskew), province plates (three lines) and plates on the wrong side of the shape threshold are recovered by
    later attempts; a reading that parses as a plate is only ever replaced by nothing, never by a later attempt.
    """
    try:
        from . import rectify
        from .postprocess import validate_latin, validate_nepal
    except ImportError:  # run as a top-level module (tests, evaluate.py from edge/anpr)
        import rectify
        from postprocess import validate_latin, validate_nepal

    def parses(result: RecognitionResult) -> bool:
        check = validate_nepal if result.script == "devanagari" else validate_latin
        return check(result.text).valid

    # Every reading is tried in order until one parses as a plate: as photographed, then levelled (deskew), then the
    # other line layouts. Only when none parses is a reading cut down to a plate it contains (_rescue), so a clean
    # parse always beats a rescued one.
    attempts: list[RecognitionResult] = []
    first_lines, _ = rectify.prepare_lines(plate_bgr)
    first = recognise_plate(first_lines)
    attempts.append(first)
    if parses(first):
        return first
    other = ("three", "one") if len(first_lines) == 2 else ("two", "three")
    for layout, level in ((None, True),) + tuple((l, False) for l in other) + tuple((l, True) for l in other):
        lines = rectify.prepare_lines(plate_bgr, layout, level=level)[0]
        candidate = recognise_plate(lines)
        attempts.append(candidate)
        if parses(candidate):
            return candidate
    for candidate in attempts:
        rescued = _rescue(candidate)
        if rescued is not None:
            return rescued
    return first


def _rescue(result: RecognitionResult) -> RecognitionResult | None:
    """A reading that fails the grammar but holds a whole plate: keep the plate, drop the junk around it.

    First by dropping whole lines (a background strip read as an extra line), then by cutting the joined text down to
    its longest grammatical stretch.
    """
    if result.script != "devanagari" or not result.lines:
        return None
    try:
        from .postprocess import extract_nepal_plate, validate_nepal
    except ImportError:
        from postprocess import extract_nepal_plate, validate_nepal

    def joined(lines):
        return unicodedata.normalize("NFC", "".join(line.text.replace(" ", "") for line in lines))

    n = len(result.lines)
    for size in range(n - 1, 0, -1):  # contiguous runs of lines, longest first
        for start in range(0, n - size + 1):
            run = result.lines[start:start + size]
            if validate_nepal(joined(run)).valid:
                return RecognitionResult(text=joined(run), script="devanagari",
                                         line_confidences=[round(l.confidence, 4) for l in run], lines=list(run))
    piece = extract_nepal_plate(result.text)
    if piece is None:
        return None
    return RecognitionResult(text=piece, script="devanagari", line_confidences=result.line_confidences, lines=result.lines)

