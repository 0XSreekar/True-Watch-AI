#!/usr/bin/env python3
"""Build round-G7 recogniser training lines from REAL training photographs, cut the way inference cuts them.

Earlier rounds trained on hand/box crops and synthetic renders. At inference the crop comes from the ONNX plate locator
(`detect_plate.locate_plates_in_crop`, top-1 box) and is split into lines by `rectify.prepare_lines`. This script runs
exactly that on every verified training photograph and writes the resulting line crops with their verified text.

A plate is dropped (and counted) when: no plate is located, the label does not parse as a Nepali registration, the
number of lines cut differs from the number of label lines, the photograph or its text collides with any test set, or
the baseline reading is so far from the label that the locator evidently picked a different plate.

Usage (from edge/, with ANPR_PLATE_LOCATOR_MODEL and ANPR_DEVANAGARI_MODEL_DIR set):
    python anpr/scripts/build_g7_lines.py --scratch-b <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

EDGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EDGE))
from anpr import detect_plate, recognise, rectify  # noqa: E402
from anpr.evaluate import char_error_rate  # noqa: E402
from anpr.postprocess import validate_nepal  # noqa: E402


def norm(text: str) -> str:
    return text.replace(" ", "").replace("|", "").replace(".", "")


def dhash(img: np.ndarray) -> int:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[list[str]]:
    return [l.rstrip("\n").split("\t") for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_inventory(b: Path) -> dict:
    """Texts, file hashes and perceptual hashes of every evaluation image (hf photos, Roboflow clean)."""
    hf = b / "hfplates"
    texts, photos = set(), []
    for name in ("test2_labels_ext.txt", "test2_labels_strict.txt", "test_labels_ext.txt", "test_labels.txt"):
        for rel, text in read_rows(hf / name):
            texts.add(norm(text))
            photos.append(hf / "images" / (Path(rel).stem + ".jpg"))
    rf_train = set()
    for d in ("college", "prajwal", "sushant"):
        for sp in ("train", "valid"):
            rf_train.update(norm(r[1]) for r in read_rows(b / "rf" / f"{d}_out" / f"{sp}_plates.txt"))
    clean = 0
    rf_imgs = []
    for rel, text in read_rows(b / "rf" / "all_test_labels.txt"):
        if norm(text) not in rf_train:  # the 114 clean held-out plates
            clean += 1
            texts.add(norm(text))
        rf_imgs.append(b / "rf" / rel)  # every Roboflow test image is also excluded by hash, clean or not
    photos.extend(rf_imgs)
    md5s, dh = set(), []
    for p in photos:
        if p.is_file():
            md5s.add(md5(p))
            im = cv2.imread(str(p))
            if im is not None:
                dh.append(dhash(im))
    return {"texts": texts, "md5": md5s, "dhash": dh, "n_photos": len(photos), "rf_clean": clean}


def near(h: int, others: list[int], bits: int = 4) -> bool:
    return any(bin(h ^ o).count("1") <= bits for o in others)


def sources(b: Path):
    """Yield (source, id, photo_path, label_lines) for every verified real training plate with text."""
    rf = b / "rf"
    for d in ("college", "prajwal", "sushant"):
        for rel, full, lines in read_rows(rf / f"{d}_out" / "train_plates.txt"):
            yield f"rf_{d}", Path(rel).stem, rf / d / rel, lines.split("|")
    hf = b / "hfplates"
    ids = [Path(r[0]).stem for r in read_rows(hf / "tr2_labels.txt")]
    for idx, kind, text, conf in read_rows(hf / "trans2" / "train_verify.tsv"):
        if conf == "high" and kind in ("OLD1", "OLD2", "PROV"):
            yield "hf_train", ids[int(idx)], hf / "images" / (ids[int(idx)] + ".jpg"), text.split("|")
    rp = b / "realplates"
    for r in json.loads((rp / "labels_merged.json").read_text(encoding="utf-8")):
        if r.get("kind") in ("OLD1", "OLD2", "PROV"):
            yield "realplates", Path(r["crop"]).stem, rp / "crops" / r["crop"], r["text"].split("|")


def cut_lines(crop: np.ndarray, n_lines: int):
    """The first inference attempt whose line count matches the label (same order as recognise.read_plate)."""
    first = rectify.prepare_lines(crop)[0]
    if len(first) == n_lines:
        return first
    other = ("three", "one") if len(first) == 2 else ("two", "three")
    for layout, level in ((None, True),) + tuple((l, False) for l in other) + tuple((l, True) for l in other):
        lines = rectify.prepare_lines(crop, layout, level=level)[0]
        if len(lines) == n_lines:
            return lines
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-b", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-sim", type=float, default=0.35, help="drop a plate whose baseline CER exceeds 1 - this")
    args = ap.parse_args()
    b, out = args.scratch_b, args.out
    (out / "lines7").mkdir(parents=True, exist_ok=True)
    inv = test_inventory(b)
    stats = Counter()
    seen_md5: set[str] = set()
    rows, plates = [], []
    for src, pid, photo, label_lines in sources(b):
        stats["candidates"] += 1
        img = cv2.imread(str(photo))
        if img is None:
            stats["unreadable"] += 1
            continue
        digest = hashlib.md5(photo.read_bytes()).hexdigest()
        text = norm("".join(label_lines))
        if digest in inv["md5"] or near(dhash(img), inv["dhash"]):
            stats["dropped_test_image_match"] += 1
            continue
        if text in inv["texts"]:
            stats["dropped_test_text_match"] += 1
            continue
        if digest in seen_md5:
            stats["dropped_duplicate_image"] += 1
            continue
        seen_md5.add(digest)
        if not validate_nepal(text).valid:
            stats["dropped_grammar"] += 1
            continue
        boxes = detect_plate.locate_plates_in_crop(img)
        if not boxes:
            stats["dropped_no_plate_located"] += 1
            continue
        (x1, y1, x2, y2), _ = boxes[0]
        lines = cut_lines(img[y1:y2, x1:x2], len(label_lines))
        if lines is None:
            stats["dropped_line_count_mismatch"] += 1
            continue
        readings = [recognise.recognise_line(im)[0].text for im in lines]
        cer = char_error_rate("".join(r.replace(" ", "") for r in readings), text)
        if 1 - cer < args.min_sim:
            stats["dropped_wrong_plate_suspected"] += 1
            continue
        stats["kept_plates"] += 1
        plates.append({"source": src, "id": pid, "text": text, "baseline": readings, "exact": norm("".join(readings)) == text})
        for k, (im, lab, rd) in enumerate(zip(lines, label_lines, readings), 1):
            rel = f"lines7/{src}/{pid}_l{k}.jpg"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / rel), im)
            wrong = norm(rd) != norm(lab)
            stats["lines_wrong_under_baseline" if wrong else "lines_right_under_baseline"] += 1
            rows.append((rel, lab, wrong))
    (out / "lines7_list.txt").write_text("".join(f"{r}\t{t}\n" for r, t, _ in rows), encoding="utf-8")
    (out / "lines7_wrong.txt").write_text("".join(f"{r}\n" for r, _, w in rows if w), encoding="utf-8")
    report = {"stats": dict(stats), "test_inventory": {k: (len(v) if hasattr(v, "__len__") else v) for k, v in inv.items()},
              "plates": plates}
    (out / "lines7_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["stats"], indent=1), report["test_inventory"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
