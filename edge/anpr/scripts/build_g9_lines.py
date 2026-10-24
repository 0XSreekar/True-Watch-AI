#!/usr/bin/env python3
"""Build round-G9 recogniser lines from Roboflow character-box sets and the hf double-pass plates.

Roboflow sets (YOLOv8 exports, CC BY 4.0): character boxes are mapped to Devanagari, sorted into lines (y clustering,
then x), joined, and kept only when the plate grammar accepts the text. Plates are then cut the way inference cuts them
(locator top-1 box, `build_g7_lines.cut_lines`). Images are dropped when they match a test photo or an earlier training
photo by file hash or 4-bit dHash, a test text, or an earlier image of the same set (augmented copies).

Usage (from edge/, ANPR_PLATE_LOCATOR_MODEL=locator 0.2.0, ANPR_DEVANAGARI_MODEL_DIR=baseline reader):
    python anpr/scripts/build_g9_lines.py --scratch-b <dir> --rf-g9 <roboflow_g9 dir> --hf-g9 <json> --hf-photos <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2

EDGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EDGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anpr import detect_plate, recognise  # noqa: E402
from anpr.evaluate import char_error_rate  # noqa: E402
from anpr.postprocess import validate_nepal  # noqa: E402
from build_g7_lines import cut_lines, dhash, near, norm, test_inventory  # noqa: E402

DIG = "०१२३४५६७८९"
MAP = {
    "BAGMATI": "बागमती", "PRADESH": "प्रदेश", "PRA": "प्र", "BA": "बा", "ba": "बा", "CHA": "च", "cha": "च", "CA": "च",
    "KO": "को", "ko": "को", "PA": "प", "pa": "प", "GA": "ग", "ga": "ग", "JA": "ज", "ja": "ज", "jha": "झ", "KA": "क",
    "ka": "क", "KHA": "ख", "kha": "ख", "LU": "लु", "lu": "लु", "NA": "ना", "RA": "रा", "BHE": "भे", "SU": "सु",
}
SETS = [
    ("nivkarki", "nivkarki_nepali-number-plate-r1vin_v4"),
    ("crimson", "crimson-dhnwj_number-plate-detection-mp1ux_v1"),
    ("xyz", "xyz-lydy7_number-plate-det_v1"),
]


def read_names(data: Path) -> list[str]:
    for l in (data / "data.yaml").read_text().splitlines():
        if l.startswith("names:"):
            return ast.literal_eval(l.split(":", 1)[1].strip())
    return []


def plate_lines(boxes):
    """Sort character boxes (cx, cy, w, h, glyph) into lines of text."""
    hs = sorted(b[3] for b in boxes)
    mh = hs[len(hs) // 2]
    rows: list[list] = []
    for b in sorted(boxes, key=lambda b: b[1]):
        if rows and abs(b[1] - sum(x[1] for x in rows[-1]) / len(rows[-1])) < 0.6 * mh:
            rows[-1].append(b)
        else:
            rows.append([b])
    return ["".join(g[4] for g in sorted(r, key=lambda b: b[0])) for r in rows]


def hdr_len(t: str) -> int:
    m = re.match(r"^(?:बागमती)?प्रदेश[०-९]?[०-९]{2}", t)
    return m.end() if m else 0


def split_text(t: str, n: int):
    if n == 1:
        return [t]
    if n == 2:
        h = hdr_len(t)
        if h:
            return [t[:h], t[h:]]
        m = re.match(r"^(.+?[०-९]{1,2}[क-ह][ा-ौ]?)([०-९]+)$", t)
        if m:
            return [m.group(1), m.group(2)]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-b", required=True, type=Path)
    ap.add_argument("--rf-g9", required=True, type=Path)
    ap.add_argument("--hf-g9", required=True, type=Path)
    ap.add_argument("--hf-photos", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-sim", type=float, default=0.35)
    args = ap.parse_args()
    b, out = args.scratch_b, args.out
    (out / "lines9").mkdir(parents=True, exist_ok=True)
    inv = test_inventory(b)
    prev_md5, prev_dh = set(), []
    for pth in list((b / "rf").glob("*/*/images/*.jpg")) + list((b / "rf" / "vehicles").glob("**/*.jpg")):
        prev_md5.add(hashlib.md5(pth.read_bytes()).hexdigest())
        im = cv2.imread(str(pth))
        if im is not None:
            prev_dh.append(dhash(im))
    print("earlier photos hashed", len(prev_md5), flush=True)
    per_set, plates, rows, wrong = {}, [], [], []
    seen_md5: set[str] = set()
    seen_dh: list[int] = []

    def emit(tag, stem, lines, label_lines, readings):
        for k, (im, lb, r) in enumerate(zip(lines, label_lines, readings), 1):
            rel = f"lines9/{tag}/{stem}_l{k}.jpg"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / rel), im)
            rows.append((rel, lb))
            if norm(r) != norm(lb):
                wrong.append(rel)

    for name, folder in SETS:
        data = args.rf_g9 / folder / "data"
        names = read_names(data)
        c = Counter()
        for img_path in sorted(data.glob("*/images/*")):
            c["images"] += 1
            lab = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            img = cv2.imread(str(img_path))
            if img is None or not lab.is_file():
                c["unreadable"] += 1
                continue
            digest = hashlib.md5(img_path.read_bytes()).hexdigest()
            dh = dhash(img)
            if digest in seen_md5 or near(dh, seen_dh, 3):
                c["dropped_augmented_or_duplicate_in_sets"] += 1
                continue
            seen_md5.add(digest)
            seen_dh.append(dh)
            if digest in inv["md5"] or near(dh, inv["dhash"]):
                c["dropped_test_image_match"] += 1
                continue
            if digest in prev_md5 or near(dh, prev_dh):
                c["dropped_earlier_training_image_match"] += 1
                continue
            bx, bad = [], False
            for l in lab.read_text().splitlines():
                p = l.split()
                if len(p) < 5:
                    continue
                cls = names[int(p[0])]
                g = DIG[int(cls)] if cls.isdigit() else MAP.get(cls)
                if g is None:
                    bad = True
                    break
                bx.append((float(p[1]), float(p[2]), float(p[3]), float(p[4]), g))
            if bad or not bx:
                c["dropped_unmapped_class"] += 1
                continue
            label_lines = plate_lines(bx)
            text = norm("".join(label_lines))
            if not validate_nepal(text).valid:
                c["dropped_grammar"] += 1
                continue
            if text in inv["texts"]:
                c["dropped_test_text_match"] += 1
                continue
            boxes = detect_plate.locate_plates_in_crop(img)
            if not boxes:
                c["dropped_no_plate_located"] += 1
                continue
            (x1, y1, x2, y2), _ = boxes[0]
            lines = cut_lines(img[y1:y2, x1:x2], len(label_lines))
            if lines is None:
                c["dropped_line_count_mismatch"] += 1
                continue
            rd = [recognise.recognise_line(im)[0].text for im in lines]
            if 1 - char_error_rate("".join(r.replace(" ", "") for r in rd), text) < args.min_sim:
                c["dropped_wrong_plate_suspected"] += 1
                continue
            c["kept"] += 1
            plates.append({"source": name, "file": img_path.name, "text": text, "lines": label_lines})
            emit(name, img_path.stem[:40], lines, label_lines, rd)
        per_set[name] = dict(c)

    doc = json.loads(args.hf_g9.read_text(encoding="utf-8"))
    c = Counter()
    for p in doc["plates"]:
        if p["script"] != "devanagari":
            continue
        c["plates"] += 1
        img = cv2.imread(str(args.hf_photos / p["file"]))
        x1, y1, x2, y2 = p["box"]
        crop = img[y1:y2, x1:x2]
        t = p["text"]
        done = False
        for n in ((1, 2) if (x2 - x1) / max(1, y2 - y1) >= 4.2 else (2, 1)):
            ll = split_text(t, n)
            lines = cut_lines(crop, n) if ll else None
            if lines is None:
                continue
            rd = [recognise.recognise_line(im)[0].text for im in lines]
            if 1 - char_error_rate("".join(r.replace(" ", "") for r in rd), t) < args.min_sim:
                continue
            c["kept"] += 1
            plates.append({"source": "hf_g9", "file": p["file"], "text": t, "lines": ll})
            emit("hf_g9", Path(p["file"]).stem, lines, ll, rd)
            done = True
            break
        if not done:
            c["dropped_cut_or_guard"] += 1
    per_set["hf_g9"] = dict(c)
    (out / "lines9_list.txt").write_text("".join(f"{r}\t{t}\n" for r, t in rows), encoding="utf-8")
    (out / "lines9_wrong.txt").write_text("".join(f"{r}\n" for r in wrong), encoding="utf-8")
    (out / "lines9_report.json").write_text(json.dumps({"per_set": per_set, "plates": plates}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(per_set, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
