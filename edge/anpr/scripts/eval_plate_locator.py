#!/usr/bin/env python3
"""Release evaluation of the ONNX plate locator: the evidence its model card cites.

Five checks, each written to one JSON report:
  1. held-out test split   recall / precision / AP50 with Wilson 95% intervals, at the threshold
                            chosen on the VAL split (never on test)
  2. threshold choice       val-split sweep; the operating point maximises F1
  3. stress                 the same test images darkened, blurred, downscaled, IR-like, JPEG-damaged
                            and noisy; recall/precision per condition
  4. negatives              false detections per image on scenes that contain no vehicle at all
  5. natural dark           test images with the lowest 15% mean luminance, no synthetic degradation
Optionally (--ood-images) it draws its detections on photos from a source it was never trained on,
as contact sheets for a visual check, because those photos carry no plate boxes.

Usage (from edge/):
  python -m anpr.scripts.eval_plate_locator --model <onnx> --data <platedet build dir> \\
      --negatives <dir of plate-free images> [--ood-images <dir>] --out <report.json>
The data dir uses the YOLO layout images/{val,test}, labels/{val,test} (class 0 = plate).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from anpr.plate_locator import PlateLocator

IOU_MATCH = 0.5
SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_split(data: Path, split: str) -> list[tuple[Path, list[list[float]]]]:
    items = []
    for img in sorted((data / "images" / split).glob("*.jpg")):
        lab = data / "labels" / split / (img.stem + ".txt")
        boxes = []
        if lab.exists():
            h, w = cv2.imread(str(img)).shape[:2]
            for line in lab.read_text().split("\n"):
                p = line.split()
                if len(p) == 5:
                    cx, cy, bw, bh = (float(v) for v in p[1:])
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        items.append((img, boxes))
    return items


def detect_all(loc: PlateLocator, items, transform=None) -> list[list[tuple[list[float], float]]]:
    out = []
    for img_path, _ in items:
        img = cv2.imread(str(img_path))
        if transform is not None:
            img = transform(img)
        out.append(loc.locate(img))
    return out


def score(items, dets, thr: float) -> dict:
    tp = fp = 0
    gt_total = sum(len(b) for _, b in items)
    for (_, gts), ds in zip(items, dets):
        used = set()
        for box, conf in sorted(ds, key=lambda d: -d[1]):
            if conf < thr:
                continue
            best, bi = 0.0, -1
            for i, g in enumerate(gts):
                if i not in used:
                    v = iou(box, g)
                    if v > best:
                        best, bi = v, i
            if best >= IOU_MATCH:
                used.add(bi)
                tp += 1
            else:
                fp += 1
    rec = tp / gt_total if gt_total else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    return {"threshold": thr, "plates": gt_total, "tp": tp, "fp": fp, "recall": round(rec, 4),
            "recall_ci95": wilson(tp, gt_total), "precision": round(prec, 4), "precision_ci95": wilson(tp, tp + fp),
            "f1": round(f1, 4)}


def ap50(items, dets) -> float:
    flat, gt_total = [], sum(len(b) for _, b in items)
    for k, ((_, gts), ds) in enumerate(zip(items, dets)):
        for box, conf in ds:
            flat.append((conf, k, box))
    flat.sort(key=lambda x: -x[0])
    used: dict[int, set] = {}
    tps = []
    for conf, k, box in flat:
        gts = items[k][1]
        best, bi = 0.0, -1
        for i, g in enumerate(gts):
            if i not in used.get(k, set()):
                v = iou(box, g)
                if v > best:
                    best, bi = v, i
        hit = best >= IOU_MATCH
        if hit:
            used.setdefault(k, set()).add(bi)
        tps.append(hit)
    tp = np.cumsum(tps)
    fp = np.cumsum([not t for t in tps])
    rec = tp / max(gt_total, 1)
    prec = tp / np.maximum(tp + fp, 1)
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = prec[rec >= r]
        ap += (p.max() if p.size else 0.0) / 101
    return round(float(ap), 4)


def _gamma(img, g):
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def _rescale(img, f):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _noise(img, sigma):
    rng = np.random.default_rng(0)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _ir_like(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return cv2.cvtColor(cv2.equalizeHist(g), cv2.COLOR_GRAY2BGR)


STRESS = {
    "night (gamma 2.5, dark)": lambda im: _gamma(im, 2.5),
    "motion/defocus blur (9 px)": lambda im: cv2.GaussianBlur(im, (9, 9), 0),
    "half resolution": lambda im: _rescale(im, 0.5),
    "quarter resolution (far range proxy)": lambda im: _rescale(im, 0.25),
    "IR-like (grey, smoothed, equalised)": _ir_like,
    "heavy JPEG (q=15)": lambda im: _jpeg(im, 15),
    "sensor noise (sigma 20)": lambda im: _noise(im, 20),
}


def contact_sheets(loc: PlateLocator, images: list[Path], thr: float, out_dir: Path, per_sheet: int = 12) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    for s in range(0, len(images), per_sheet):
        tiles = []
        for k, p in enumerate(images[s:s + per_sheet]):
            img = cv2.imread(str(p))
            for box, conf in loc.locate(img):
                if conf >= thr:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            tile = cv2.resize(img, (400, 400))
            cv2.putText(tile, str(s + k), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            tiles.append(tile)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        path = out_dir / f"ood_sheet_{s // per_sheet:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        sheets.append(str(path))
    return sheets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--negatives", type=Path)
    ap.add_argument("--ood-images", type=Path)
    ap.add_argument("--ood-count", type=int, default=48)
    ap.add_argument("--sheets-dir", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    loc = PlateLocator(args.model, conf=min(SWEEP))     # keep low-score boxes; thresholds applied here
    val, test = load_split(args.data, "val"), load_split(args.data, "test")
    val_dets = detect_all(loc, val)
    sweep = [score(val, val_dets, t) for t in SWEEP]
    chosen = max(sweep, key=lambda r: (r["f1"], r["threshold"]))["threshold"]
    test_dets = detect_all(loc, test)
    report = {
        "iou_match": IOU_MATCH,
        "threshold_sweep_val": sweep,
        "chosen_threshold": chosen,
        "test": {**score(test, test_dets, chosen), "images": len(test), "ap50": ap50(test, test_dets)},
        "test_at_0.25": score(test, test_dets, 0.25),
        "stress_test": {},
    }
    for name, fn in STRESS.items():
        report["stress_test"][name] = score(test, detect_all(loc, test, fn), chosen)
    # natural-dark subset: the 15% of test images with the lowest mean luminance, no synthetic degradation
    lum = [float(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).mean()) for p, _ in test]
    cut = int(len(test) * 0.15)
    idx = sorted(range(len(test)), key=lambda i: lum[i])[:cut]
    dark_items, dark_dets = [test[i] for i in idx], [test_dets[i] for i in idx]
    report["natural_dark_bottom15pct"] = {"images": cut, "max_mean_luminance": round(max(lum[i] for i in idx), 1),
                                          "at_chosen": score(dark_items, dark_dets, chosen),
                                          "at_0.25": score(dark_items, dark_dets, 0.25)}
    if args.negatives:
        negs = sorted(p for p in args.negatives.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        neg_dets = [loc.locate(cv2.imread(str(p))) for p in negs]
        for key, thr in (("negatives", chosen), ("negatives_at_0.25", 0.25)):
            counts = [sum(c >= thr for _, c in d) for d in neg_dets]
            report[key] = {"threshold": thr, "images": len(negs), "false_detections": int(sum(counts)),
                           "images_with_any": int(sum(c > 0 for c in counts)),
                           "images_with_any_ci95": wilson(int(sum(c > 0 for c in counts)), len(negs)),
                           "per_image": round(sum(counts) / max(len(negs), 1), 4)}
    if args.ood_images and args.sheets_dir:
        pool = sorted(p for p in args.ood_images.rglob("*.jpg"))
        random.Random(7).shuffle(pool)
        report["ood_sheets"] = contact_sheets(loc, pool[: args.ood_count], chosen, args.sheets_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    t = report["test"]
    print(f"threshold {chosen} (val F1) | test recall {t['recall']} {t['recall_ci95']} precision {t['precision']} "
          f"{t['precision_ci95']} AP50 {t['ap50']}")
    for name, r in report["stress_test"].items():
        print(f"  {name:40s} recall {r['recall']:.3f} precision {r['precision']:.3f}")
    if "negatives" in report:
        print("negatives:", report["negatives"])
        print("negatives@0.25:", report["negatives_at_0.25"])
    print("natural dark:", report["natural_dark_bottom15pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
