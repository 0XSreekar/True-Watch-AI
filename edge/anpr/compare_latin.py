#!/usr/bin/env python3
"""Reproduce MEASUREMENTS.md section 4: the Latin-vs-Devanagari demonstration.

    | Model                      | Read           | Result           |
    |-----------------------------|----------------|------------------|
    | Latin OCR (off-the-shelf)   | "9 9238"       | wrong            |
    | PP-OCRv5 Devanagari         | बा १२ प १२३४   | correct, 0.989/0.990 |

This is the artefact that proves innovation claim (a) to a judge: the SAME
plate image, through the SAME PP-OCRv5 recognition family, read correctly by
the Devanagari head and misread by the Latin head — because a Nepali plate's
digits are Devanagari glyphs, and an off-the-shelf Latin-only ANPR stack has
no head that was ever trained to read them.

Without --image, this script renders its own sample plate with
`datasets/plates/gen_plates.py`'s composer and degradation pipeline (the same
code path the 20,000-sample training corpus comes from) using the exact
zone/lot/class/serial MEASUREMENTS.md section 4 confirms: zone "बा", lot 12,
class "प", serial 1234. The rendered image is written to samples/plate.png
(gitignored — this is a demo asset, not data) and is NOT committed.

    python edge/anpr/compare_latin.py
    python edge/anpr/compare_latin.py --image samples/plate.png
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import cv2  # noqa: E402

import recognise  # noqa: E402
import rectify  # noqa: E402

DATASETS_PLATES = HERE.parent.parent / "datasets" / "plates"
DEFAULT_SAMPLE = HERE / "samples" / "plate.png"

# The confirmed composition from MEASUREMENTS.md section 4.
CONFIRMED_ZONE = "बा"
CONFIRMED_LOT = "१२"
CONFIRMED_CLASS = "प"
CONFIRMED_SERIAL = "१२३४"
EXPECTED_TEXT = f"{CONFIRMED_ZONE}{CONFIRMED_LOT}{CONFIRMED_CLASS}{CONFIRMED_SERIAL}"


def render_confirmed_sample(out_path: Path) -> Path:
    sys.path.insert(0, str(DATASETS_PLATES))
    import gen_plates  # type: ignore

    font_path = gen_plates.find_font(None)
    if font_path is None:
        raise SystemExit(
            "No Devanagari font found. Fetch one per datasets/plates/fonts/README.md, "
            "then re-run this script."
        )
    config = gen_plates.load_config(DATASETS_PLATES / "plates.yaml")
    layout = config["layout"]
    colours = config["colour_series"][0]  # series_a: light plate, dark text — the common case

    fields = {
        "line1": f"{CONFIRMED_ZONE} {CONFIRMED_LOT}",
        "line2": f"{CONFIRMED_CLASS} {CONFIRMED_SERIAL}",
        "text": EXPECTED_TEXT,
    }
    import random

    rng = random.Random(42)
    plate = gen_plates.render_plate(fields, colours, layout, font_path, rng)
    if plate is None:
        raise SystemExit("Rendering the sample plate failed (font could not draw the glyphs).")

    # A light, honest degradation — the same pipeline the training corpus uses
    # — so the demo reflects a realistic check-post frame, not a studio shot.
    import degrade as degrade_module  # type: ignore

    degraded, applied = degrade_module.degrade(plate, random.Random(7))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), degraded)
    print(f"rendered sample plate -> {out_path} (degradation applied: {applied})")
    return out_path


def run_comparison(image_path: Path) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"could not read image: {image_path}")

    lines, used_quad = rectify.prepare_lines(image)
    print(f"lines found: {len(lines)} (perspective quad used: {used_quad})")

    devanagari_lines = []
    latin_lines = []
    for line in lines:
        devanagari, latin = recognise.recognise_line(line)
        devanagari_lines.append(devanagari)
        latin_lines.append(latin)

    devanagari_text = unicodedata.normalize(
        "NFC", "".join(line.text.replace(" ", "") for line in devanagari_lines)
    )
    latin_text = unicodedata.normalize("NFC", "".join(line.text.replace(" ", "") for line in latin_lines))

    print()
    print("=" * 64)
    print("MEASUREMENTS.md section 4 reproduction")
    print("=" * 64)
    print(f"{'Model':<28} {'Read':<20} {'Confidence'}")
    print(
        f"{'Latin OCR (PP-OCRv5 en)':<28} {latin_text!r:<20} "
        f"{[round(l.confidence, 3) for l in latin_lines]}"
    )
    print(
        f"{'PP-OCRv5 Devanagari':<28} {devanagari_text!r:<20} "
        f"{[round(l.confidence, 3) for l in devanagari_lines]}"
    )
    print("=" * 64)

    devanagari_correct = devanagari_text == EXPECTED_TEXT
    latin_wrong = latin_text != EXPECTED_TEXT
    if devanagari_correct and latin_wrong:
        print("RESULT: Devanagari correct, Latin wrong — reproduces MEASUREMENTS.md section 4.")
        return 0
    if not devanagari_correct:
        print(
            f"RESULT: Devanagari path did NOT read the expected text {EXPECTED_TEXT!r} "
            f"(got {devanagari_text!r}). This is the pretrained baseline; see "
            "edge/anpr/results/ANPR_METRICS.md and the fine-tune notebook."
        )
    if not latin_wrong:
        print(f"RESULT: Latin path unexpectedly matched {EXPECTED_TEXT!r}; the comparison did not reproduce.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None, help="an existing plate image; default renders one")
    args = ap.parse_args()

    image_path = Path(args.image) if args.image else DEFAULT_SAMPLE
    if not image_path.exists():
        if args.image:
            raise SystemExit(f"--image {image_path} does not exist")
        render_confirmed_sample(image_path)

    return run_comparison(image_path)


if __name__ == "__main__":
    raise SystemExit(main())
