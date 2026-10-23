#!/usr/bin/env python3
"""Build round-G8 recogniser training lines from the double-transcribed street plates (data/street_plates_g8.json).

Each plate is cut the way inference cuts it: the recorded locator box on the full photograph, then
`rectify.prepare_lines` (same attempt order as build_g7_lines.cut_lines). A plate is dropped (and counted) when no cut
yields the label's line count. Lines the baseline reader gets wrong are listed so the trainer can repeat them more.

Usage (from edge/, with ANPR_DEVANAGARI_MODEL_DIR set to the baseline recogniser):
    python anpr/scripts/build_g8_lines.py --photos <rf/vehicles/train/images> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

EDGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EDGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anpr import recognise  # noqa: E402
from build_g7_lines import cut_lines, norm  # noqa: E402

RECORDS = EDGE / "anpr" / "data" / "street_plates_g8.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    doc = json.loads(RECORDS.read_text(encoding="utf-8"))
    (args.out / "lines8").mkdir(parents=True, exist_ok=True)
    stats, rows, wrong = Counter(), [], []
    for k, p in enumerate(doc["plates"]):
        img = cv2.imread(str(args.photos / p["file"]))
        if img is None or not p.get("lines"):
            stats["dropped_unreadable_or_no_line_split"] += 1
            continue
        x1, y1, x2, y2 = p["box"]
        lines = cut_lines(img[y1:y2, x1:x2], len(p["lines"]))
        if lines is None:
            stats["dropped_line_count_mismatch"] += 1
            continue
        stats["kept_plates"] += 1
        for j, (im, lab) in enumerate(zip(lines, p["lines"]), 1):
            rel = f"lines8/s{k:03d}_l{j}.jpg"
            cv2.imwrite(str(args.out / rel), im)
            bad = norm(recognise.recognise_line(im)[0].text) != norm(lab)
            stats["lines_wrong_under_baseline" if bad else "lines_right_under_baseline"] += 1
            rows.append((rel, lab))
            if bad:
                wrong.append(rel)
    (args.out / "lines8_list.txt").write_text("".join(f"{r}\t{t}\n" for r, t in rows), encoding="utf-8")
    (args.out / "lines8_wrong.txt").write_text("".join(f"{r}\n" for r in wrong), encoding="utf-8")
    print(json.dumps(dict(stats), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
