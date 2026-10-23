#!/usr/bin/env python3
"""End-to-end plate read from full photographs: locate (ONNX locator, top-1 box), crop, read_plate.

Labels are `crops/<id>.jpg<TAB>text` (the hand-labelled real sets); the full photograph for `<id>` is
`<photos>/<id>.jpg`. Set ANPR_PLATE_LOCATOR_MODEL to the .onnx and ANPR_DEVANAGARI_MODEL_DIR to the recogniser.

    cd edge && python anpr/scripts/eval_e2e_locator.py --labels L.txt --photos DIR --out out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

EDGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EDGE))
from anpr import detect_plate, recognise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [l.rstrip("\n").split("\t") for l in Path(args.labels).read_text(encoding="utf-8").splitlines() if l.strip()]
    samples, found, hit = [], 0, 0
    for rel, ref in rows:
        photo = Path(args.photos) / (Path(rel).stem + ".jpg")
        img = cv2.imread(str(photo))
        if img is None:
            continue
        boxes = detect_plate.locate_plates_in_crop(img)
        hyp = ""
        if boxes:
            found += 1
            (x1, y1, x2, y2), _ = boxes[0]
            hyp = recognise.read_plate(img[y1:y2, x1:x2]).text
        ok = hyp == ref
        hit += ok
        samples.append({"id": Path(rel).stem, "reference": ref, "hypothesis": hyp, "exact": ok, "located": bool(boxes)})
    n = len(samples)
    out = {"n": n, "plate_found": found, "exact_match": hit, "exact_match_rate": round(hit / n, 4) if n else 0.0, "samples": samples}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"n={n} found={found} exact={hit} ({out['exact_match_rate']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
