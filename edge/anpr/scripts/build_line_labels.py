#!/usr/bin/env python3
"""Build per-line recognition training data from the synthetic plate corpus.

The recogniser reads ONE text line. At inference `rectify.prepare_lines()` splits a
two-line Nepali plate into its two lines before recognition, so training must use the
same crops: feeding whole two-line plates squashed into the 48x320 input teaches the
head nothing (the pretrained model scores exact-match 0.0 on that input).

For each row of `<split>_list.txt` (whole-plate image, joined plate text) this script:
  1. splits the joined text into line1 = "zone lot" and line2 = "class serial" using the
     field vocabulary in datasets/plates/plates.yaml (longest known zone and class
     prefix, fixed 2-digit lot, fixed-width serial) and checks the parts re-join to the
     original text exactly;
  2. runs `rectify.prepare_lines()` on the image; a plate that does not split into
     exactly two lines is skipped and counted, never paired with the wrong label;
  3. writes each line crop as a JPEG and `<split>/<name>\t<line text>` rows in the
     SimpleDataSet format PaddleOCR's tools/train.py reads.

Usage:
    python edge/anpr/scripts/build_line_labels.py \
        --lists-dir <dir with train_list.txt and val_list.txt> \
        --images-root <dir the list paths are relative to> \
        --out-root <output dir for line crops>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

HERE = Path(__file__).resolve().parent
ANPR_DIR = HERE.parent
PLATES_YAML = ANPR_DIR.parent.parent / "datasets" / "plates" / "plates.yaml"
sys.path.insert(0, str(ANPR_DIR))
import rectify  # noqa: E402


def log(level: str, message: str, **fields) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"{stamp} {level:<5} build_line_labels | {message}"
    if extra:
        line = f"{line} | {extra}"
    print(line, flush=True)


def load_plate_config() -> dict:
    with PLATES_YAML.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def split_text_into_fields(text: str, config: dict) -> tuple[str, str] | None:
    """Recover (line1_no_space, line2_no_space) from the joined ground-truth string.

    `text` = zone + lot(2 digits) + vehicle_class(1 char) + serial(N digits), built by
    gen_plates.compose(): zone is a variable-length known string from plates.yaml
    (1-3 Devanagari chars), lot is ALWAYS 2 digit-glyphs, vehicle_class is ALWAYS 1
    char, serial is a fixed width from plates.yaml `serial.digits`. Matching the
    longest known zone/class prefix against the actual config (not guessing) is what
    makes this exact, not approximate.
    """
    zones = config["zones"]["confirmed"] + config["zones"]["unverified"]
    classes = config["vehicle_classes"]["confirmed"] + config["vehicle_classes"]["unverified"]
    serial_digits = int(config["serial"]["digits"])

    zone_match = next((z for z in sorted(zones, key=len, reverse=True) if text.startswith(z)), None)
    if zone_match is None:
        return None
    rest = text[len(zone_match):]
    if len(rest) < 2 + 1 + serial_digits:
        return None
    lot_text = rest[:2]
    rest2 = rest[2:]
    class_match = next((c for c in sorted(classes, key=len, reverse=True) if rest2.startswith(c)), None)
    if class_match is None:
        return None
    serial_text = rest2[len(class_match):]
    if len(serial_text) != serial_digits:
        return None
    line1 = f"{zone_match} {lot_text}"
    line2 = f"{class_match} {serial_text}"
    if f"{zone_match}{lot_text}{class_match}{serial_text}" != text:
        return None
    return line1, line2


def process_split(split: str, config: dict, lists_dir: Path, images_root: Path, out_root: Path) -> tuple[int, int, int]:
    src = lists_dir / f"{split}_list.txt"
    out_images = out_root / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_label = lists_dir / f"{split}_lines_list.txt"

    rows = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    written = 0
    skipped_split = 0
    skipped_parse = 0
    with out_label.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            rel_path, _, rest = row.partition("\t")
            text, _, explicit = rest.partition("\t")
            # A third column "line1|line2" (real-layout synthetic plates, hand-transcribed real plates) states the
            # line texts outright; without it they are recovered from the joined string the old layout implies.
            # A third column without "|" is a one-line plate and must come back from the splitter unsplit.
            fields = tuple(explicit.split("|")) if explicit else split_text_into_fields(text, config)
            if fields is None:
                skipped_parse += 1
                continue
            image = cv2.imread(str(images_root / rel_path))
            if image is None:
                skipped_parse += 1
                continue
            layout = {1: "one", 2: "two", 3: "three"}.get(len(fields))  # the label says how many lines to cut
            lines, _used_quad = rectify.prepare_lines(image, layout)
            if len(lines) != len(fields):
                # A tilted photograph splits across its text; levelled it often splits cleanly.
                lines, _used_quad = rectify.prepare_lines(image, layout, level=True)
            if len(lines) != len(fields):
                skipped_split += 1
                continue
            stem = Path(rel_path).stem
            for line_index, (line_img, line_text) in enumerate(zip(lines, fields), start=1):
                out_name = f"{stem}_l{line_index}.jpg"
                cv2.imwrite(str(out_images / out_name), line_img)
                out.write(f"{split}/{out_name}\t{line_text}\n")
                written += 1
            if (index + 1) % 2000 == 0:
                log("INFO", "progress", split=split, done=index + 1, total=len(rows), written=written)
    log(
        "INFO",
        "done",
        split=split,
        rows=len(rows),
        written=written,
        skipped_line_count_mismatch=skipped_split,
        skipped_unparseable_text=skipped_parse,
    )
    return written, skipped_split, skipped_parse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lists-dir", required=True, type=Path, help="holds train_list.txt and val_list.txt")
    ap.add_argument("--images-root", required=True, type=Path, help="directory the list paths are relative to")
    ap.add_argument("--out-root", required=True, type=Path, help="where line crops are written")
    ap.add_argument("--splits", nargs="+", default=["val", "train"])
    args = ap.parse_args()
    config = load_plate_config()
    for split in args.splits:
        process_split(split, config, args.lists_dir, args.images_root, args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
