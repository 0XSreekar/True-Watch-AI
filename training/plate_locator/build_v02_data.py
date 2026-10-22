#!/usr/bin/env python3
"""Build the v0.2 plate-locator training set (low light / near-IR / noise / small far plates).

Run with the edge venv (needs opencv, numpy, onnxruntime; NOT ultralytics):
  python build_v02_data.py --platedet <plate-box build> --street <rf/vehicles/train> \
      --detds <detector dataset with val/ train/ images+labels> --v01 <plate_locator.onnx> \
      --eval-negatives-list <list.txt> --out <work dir>

Train set = platedet train split
  + offline photometric / geometric copies (labels unchanged, or remapped for the small-plate paste):
      night (gamma 1.8-3.5, dimmed, read noise)            p=0.25 of the chosen condition draws
      near-IR  (grey, vignetted illuminator, plates bright) p=0.25
      sensor noise (sigma 10-35)                            p=0.15
      blur (gaussian 5-11 / linear motion)                  p=0.10
      heavy JPEG (q 8-25)                                   p=0.10
      down-then-up (x0.2-0.5)                               p=0.15
    each train image gets one such copy with probability 0.6; independently 20% of train images with
    1-3 plates also get a small-plate copy (image shrunk onto a 640 canvas so plates are 10-30 px wide).
  + street pseudo-labels: v0.1 run on every upscaled labelled VEHICLE crop of the street set, boxes with
    conf >= 0.6 mapped back to full-image coordinates; the 48 review photos are excluded entirely.
  + plate-free negatives (~10% of images) from the detector dataset's train split (no vehicle label),
    disjoint from the evaluation negatives list.
val and test are symlinks to the platedet splits, untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "edge"))
from anpr.plate_locator import PlateLocator  # noqa: E402

P_COPY = 0.6
P_SMALL = 0.2
COND_P = [("night", 0.25), ("nir", 0.25), ("noise", 0.15), ("blur", 0.10), ("jpeg", 0.10), ("downup", 0.15)]
PSEUDO_CONF = 0.6
NEG_FRACTION = 0.10


def gamma(img, g):
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def add_noise(img, sigma, rng):
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def px_boxes(lines, w, h):
    out = []
    for ln in lines:
        p = ln.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def to_yolo(boxes, w, h):
    rows = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        rows.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return rows


def nir_like(img, boxes, rng):
    """Grey, active-illuminator look: bright centre falling off, local contrast, retroreflective plates."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g = g ** rng.uniform(1.0, 2.2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * rng.uniform(0.3, 0.7), h * rng.uniform(0.4, 0.8)
    r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    light = np.clip(1.25 - 1.6 * r, 0.15, 1.2)
    g = g * light
    for x1, y1, x2, y2 in boxes:                      # plates bright, soft edge
        m = np.zeros((h, w), np.float32)
        cv2.rectangle(m, (int(x1), int(y1)), (int(x2), int(y2)), 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, (x2 - x1) * 0.04))
        g = g + m * rng.uniform(0.15, 0.45) * (1 - g)
    g8 = np.clip(g * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=rng.uniform(1.5, 3.5), tileGridSize=(8, 8))
    g8 = clahe.apply(g8)
    g8 = np.clip(g8.astype(np.float32) + rng.normal(0, rng.uniform(2, 9), g8.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        g8 = cv2.GaussianBlur(g8, (3, 3), 0)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)


def apply_condition(cond, img, boxes, rng):
    if cond == "night":
        out = gamma(img, rng.uniform(1.8, 3.5))
        out = (out.astype(np.float32) * rng.uniform(0.5, 1.0))
        return add_noise(np.clip(out, 0, 255).astype(np.uint8), rng.uniform(2, 12), rng)
    if cond == "nir":
        return nir_like(img, boxes, rng)
    if cond == "noise":
        return add_noise(img, rng.uniform(10, 35), rng)
    if cond == "blur":
        if rng.random() < 0.5:
            k = int(rng.choice([5, 7, 9, 11]))
            return cv2.GaussianBlur(img, (k, k), 0)
        k = int(rng.choice([7, 11, 15]))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        ang = rng.uniform(0, 180)
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
        ker = cv2.warpAffine(ker, M, (k, k))
        return cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
    if cond == "jpeg":
        return jpeg(img, int(rng.integers(8, 26)))
    if cond == "downup":
        h, w = img.shape[:2]
        f = rng.uniform(0.2, 0.5)
        s = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(cond)


def small_plate_copy(img, boxes, rng):
    """Shrink the whole image onto a 640 canvas so the widest plate is 10-30 px wide."""
    h, w = img.shape[:2]
    target = rng.uniform(10, 30)
    wmax = max(b[2] - b[0] for b in boxes)
    s = min(target / wmax, 1.0)
    nw, nh = max(8, int(w * s)), max(8, int(h * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    C = 640
    if nw > C or nh > C:
        return None
    canvas = cv2.copyMakeBorder(cv2.resize(small, (C, C), interpolation=cv2.INTER_LINEAR), 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 20)         # smeared surround so the context is not a flat fill
    ox, oy = int(rng.integers(0, C - nw + 1)), int(rng.integers(0, C - nh + 1))
    canvas[oy:oy + nh, ox:ox + nw] = small
    return canvas, [[b[0] * s + ox, b[1] * s + oy, b[2] * s + ox, b[3] * s + oy] for b in boxes]


def nms(dets, thr=0.5):
    dets = sorted(dets, key=lambda d: -d[1])
    keep = []
    for b, c in dets:
        ok = True
        for k, _ in keep:
            ix = max(0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0, min(b[3], k[3]) - max(b[1], k[1]))
            inter = ix * iy
            u = (b[2] - b[0]) * (b[3] - b[1]) + (k[2] - k[0]) * (k[3] - k[1]) - inter
            if u > 0 and inter / u > thr:
                ok = False
                break
        if ok:
            keep.append((b, c))
    return keep


def pseudo_label(loc, img, vlines):
    h, w = img.shape[:2]
    dets = []
    for ln in vlines:
        p = ln.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        x1, y1 = max(0, int((cx - bw * 0.55) * w)), max(0, int((cy - bh * 0.55) * h))
        x2, y2 = min(w, int((cx + bw * 0.55) * w)), min(h, int((cy + bh * 0.55) * h))
        if x2 - x1 < 24 or y2 - y1 < 24:
            continue
        crop = img[y1:y2, x1:x2]
        f = min(4.0, 640.0 / max(crop.shape[:2]))
        if f > 1.0:
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        else:
            f = 1.0
        cw, ch = x2 - x1, y2 - y1
        for (bx1, by1, bx2, by2), c in loc.locate(crop):
            if c < PSEUDO_CONF:
                continue
            fb = [bx1 / f + x1, by1 / f + y1, bx2 / f + x1, by2 / f + y1]
            bwid, bhei = fb[2] - fb[0], fb[3] - fb[1]
            if bhei <= 0 or not (1.0 <= bwid / bhei <= 7.0) or bwid > 0.6 * cw or bhei > 0.5 * ch:
                continue
            dets.append((fb, c))
    return [b for b, _ in nms(dets)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platedet", required=True, type=Path)
    ap.add_argument("--street", required=True, type=Path)
    ap.add_argument("--detds", required=True, type=Path)
    ap.add_argument("--v01", required=True)
    ap.add_argument("--eval-negatives-list", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261017)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pyr = random.Random(a.seed)
    out = a.out
    for sp in ("train",):
        (out / "ds/images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "ds/labels" / sp).mkdir(parents=True, exist_ok=True)
    for sp in ("val", "test"):
        for k in ("images", "labels"):
            link = out / "ds" / k / sp
            if not link.exists():
                link.symlink_to((a.platedet / k / sp).resolve())
    (out / "ds/data.yaml").write_text(f"path: {out / 'ds'}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                      "nc: 1\nnames: [\"plate\"]\n")
    summary = {"seed": a.seed, "p_copy": P_COPY, "p_small": P_SMALL, "condition_p": dict(COND_P)}

    # 1. platedet train + augmented copies
    n_orig = n_aug = n_small = 0
    cond_counts = {c: 0 for c, _ in COND_P}
    names, probs = zip(*COND_P)
    for img_path in sorted((a.platedet / "images/train").glob("*.jpg")):
        lab = (a.platedet / "labels/train" / (img_path.stem + ".txt"))
        lines = [l for l in lab.read_text().split("\n") if l.strip()] if lab.exists() else []
        tgt_i, tgt_l = out / "ds/images/train" / img_path.name, out / "ds/labels/train" / (img_path.stem + ".txt")
        if not tgt_i.exists():
            tgt_i.symlink_to(img_path.resolve())
        tgt_l.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_orig += 1
        if not lines:
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        boxes = px_boxes(lines, w, h)
        if rng.random() < P_COPY:
            cond = str(rng.choice(names, p=probs))
            aug = apply_condition(cond, img, boxes, rng)
            cv2.imwrite(str(out / "ds/images/train" / f"{img_path.stem}_a{cond}.jpg"), aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "ds/labels/train" / f"{img_path.stem}_a{cond}.txt").write_text("\n".join(lines) + "\n")
            n_aug += 1
            cond_counts[cond] += 1
        if len(boxes) <= 3 and rng.random() < P_SMALL:
            r = small_plate_copy(img, boxes, rng)
            if r is not None:
                sim, sb = r
                if rng.random() < 0.5:
                    sim = apply_condition(str(rng.choice(["night", "nir", "noise"])), sim, sb, rng)
                cv2.imwrite(str(out / "ds/images/train" / f"{img_path.stem}_small.jpg"), sim, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "ds/labels/train" / f"{img_path.stem}_small.txt").write_text("\n".join(to_yolo(sb, 640, 640)) + "\n")
                n_small += 1
    summary.update(platedet_train_images=n_orig, photometric_copies=n_aug, photometric_by_condition=cond_counts,
                   small_plate_copies=n_small)

    # 2. street pseudo-labels, excluding the 48 review photos
    pool = sorted((a.street / "images").rglob("*.jpg"))
    shuf = list(pool)
    random.Random(7).shuffle(shuf)
    review = [p.name for p in shuf[:48]]
    summary["street_review_excluded"] = review
    loc = PlateLocator(a.v01, conf=0.25)
    kept_imgs = kept_boxes = scanned = 0
    for i, p in enumerate(pool):
        if p.name in review:
            continue
        lab = a.street / "labels" / (p.stem + ".txt")
        if not lab.exists():
            continue
        vlines = [l for l in lab.read_text().split("\n") if l.strip()]
        if not vlines:
            continue
        img = cv2.imread(str(p))
        scanned += 1
        boxes = pseudo_label(loc, img, vlines)
        if not boxes:
            continue
        h, w = img.shape[:2]
        cv2.imwrite(str(out / "ds/images/train" / f"st_{p.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "ds/labels/train" / f"st_{p.stem}.txt").write_text("\n".join(to_yolo(boxes, w, h)) + "\n")
        kept_imgs += 1
        kept_boxes += len(boxes)
    summary.update(street_images_scanned=scanned, pseudo_images_kept=kept_imgs, pseudo_boxes_kept=kept_boxes,
                   pseudo_conf=PSEUDO_CONF)

    # 3. plate-free negatives from the detector dataset train split (no vehicle label), disjoint from eval list
    evalneg = set(a.eval_negatives_list.read_text().split("\n"))
    cands = []
    for img in sorted((a.detds / "images/train").glob("*.jpg")):
        if img.name in evalneg:
            continue
        lab = a.detds / "labels/train" / (img.stem + ".txt")
        cls = {l.split()[0] for l in lab.read_text().split("\n") if l.strip()} if lab.exists() else set()
        if not cls & {"1", "2", "3"}:
            cands.append(img)
    total_pos = len(list((out / "ds/images/train").glob("*.jpg")))
    n_neg = int(total_pos * NEG_FRACTION / (1 - NEG_FRACTION))
    pyr.shuffle(cands)
    for img in cands[:n_neg]:
        (out / "ds/images/train" / f"neg_{img.name}").symlink_to(img.resolve())
        (out / "ds/labels/train" / f"neg_{img.stem}.txt").write_text("")
    summary.update(negative_candidates=len(cands), negatives_added=min(n_neg, len(cands)),
                   train_images_total=len(list((out / "ds/images/train").glob("*"))))
    (out / "build_v02_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print({k: v for k, v in summary.items() if k != "street_review_excluded"})


if __name__ == "__main__":
    main()
