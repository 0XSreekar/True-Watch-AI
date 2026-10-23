#!/usr/bin/env python3
"""Open Images V6/V7 -> daytime small and crowded PERSON images, unified YOLO labels (DATASET_SPEC.md section 1.8).

Why this source. The day person gap sits in small and crowded people (IDD 0.52, VisDrone 0.37 AP50). Open Images is
the one large corpus with a clean, citable licence that holds many street, plaza, market and event scenes with a
dozen or more small people: annotations CC BY 4.0, images listed as CC BY 2.0 (Open Images fact sheet,
https://storage.googleapis.com/openimages/web/factsfigures_v7.html). Only images whose own `License` column is the
CC BY 2.0 URL are used.

Selection (every rule is a number in this file, nothing is judged by eye):
  * person boxes (MID /m/01g317): at least --min-persons, at least --min-small of them shorter than --small-h of the image
  * no IsGroupOf and no IsDepiction person box: a group box is an ignore region, and the converters never turn an
    unlabelled person into a negative, so an image holding one is skipped whole
  * no box of a vehicle or cart class (the other four schema classes are not labelled here, so an image that shows them would
    teach them as background): Car, Bus, Truck, Van, Motorcycle, Bicycle, Taxi, Land vehicle, Vehicle, Ambulance,
    Limousine, Cart, Wheelchair, Tricycle-like OI classes are all excluded
  * Rotation field empty or 0 (boxes are in the stored orientation)
  * daytime: mean grey level of the downloaded frame >= --min-brightness (night frames are dropped, counted and logged)
Class mapping: Person -> person (0). Nothing else is written. All images are train-only; the held-out set is drawn
from Open Images' OWN validation split (different photos by construction) and kept in images/val_oi, which the
headline v2 val split never contains.

Output (self-contained, same layout as 09_build_yolo_ds.py): <out>/images/{train,val_oi}/openimages_<id>_visible.jpg,
labels/..., index/split_extra.jsonl, manifest_extra.tsv, EXTRA_REPORT.json, and tar shards under <out>/../extra_shards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PERSON = "/m/01g317"
EXCLUDE_NAMES = {"Car", "Bus", "Truck", "Van", "Motorcycle", "Bicycle", "Taxi", "Land vehicle", "Vehicle", "Ambulance",
                 "Limousine", "Cart", "Wheelchair", "Golf cart", "Tank", "Snowmobile", "Segway", "Unicycle",
                 "Tricycle", "Scooter", "Rickshaw", "Train", "Tram", "Aircraft", "Boat", "Watercraft"}
CC_BY = "https://creativecommons.org/licenses/by/2.0/"
S3 = "https://open-images-dataset.s3.amazonaws.com"
BBOX_BASE = "https://storage.googleapis.com/openimages"


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def exclude_mids(cache: Path) -> set[str]:
    path = fetch(f"{BBOX_BASE}/v6/oidv6-class-descriptions.csv", cache / "oidv6-class-descriptions.csv")
    with path.open(encoding="utf-8") as fh:
        return {row[0] for row in csv.reader(fh) if len(row) == 2 and row[1] in EXCLUDE_NAMES}


def scan(bbox_csv: Path, bad_mids: set[str]) -> dict[str, dict]:
    """ImageID -> {'p': [(xmin,xmax,ymin,ymax,occluded)], 'bad': bool}. Streams the 2 GB file."""
    out: dict[str, dict] = {}
    with bbox_csv.open(encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        ix = {n: i for i, n in enumerate(head)}
        for row in rd:
            mid = row[ix["LabelName"]]
            if mid != PERSON and mid not in bad_mids:
                continue
            rec = out.setdefault(row[ix["ImageID"]], {"p": [], "bad": False})
            if mid in bad_mids:
                rec["bad"] = True
                continue
            if row[ix["IsGroupOf"]] == "1" or row[ix["IsDepiction"]] == "1":
                rec["bad"] = True
                continue
            rec["p"].append((float(row[ix["XMin"]]), float(row[ix["XMax"]]), float(row[ix["YMin"]]), float(row[ix["YMax"]]),
                             row[ix["IsOccluded"]] == "1"))
    return out


def licensed(images_csv: Path) -> dict[str, dict]:
    ok: dict[str, dict] = {}
    with images_csv.open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            rot = (r.get("Rotation") or "").strip()
            if r["License"].strip() == CC_BY and rot in ("", "0", "0.0"):
                ok[r["ImageID"]] = {"author": r.get("Author", ""), "url": r.get("OriginalLandingURL", "")}
    return ok


def rank(cands: dict[str, dict], lic: dict[str, dict], min_persons: int, min_small: int, small_h: float) -> list[tuple[int, str]]:
    ranked = []
    for iid, rec in cands.items():
        if rec["bad"] or iid not in lic or len(rec["p"]) < min_persons:
            continue
        small = sum(1 for (_x0, _x1, y0, y1, _o) in rec["p"] if (y1 - y0) < small_h)
        if small >= min_small:
            ranked.append((small, iid))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    return ranked


def process(job: tuple) -> dict | None:
    iid, split_dir, root, rec, min_brightness, folder = job
    from PIL import Image
    import numpy as np
    stem = f"openimages_{iid}_visible"
    img_path = root / "images" / split_dir / f"{stem}.jpg"
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(f"{S3}/{folder}/{iid}.jpg", timeout=30).read()
            break
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    else:
        return {"image": iid, "dropped": "download"}
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(data)
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception:
        img_path.unlink(missing_ok=True)
        return {"image": iid, "dropped": "decode"}
    w, h = im.size
    small = im.resize((64, 64))
    if float(np.asarray(small.convert("L")).mean()) < min_brightness:
        img_path.unlink(missing_ok=True)
        return {"image": iid, "dropped": "dark"}
    lines = []
    for x0, x1, y0, y1, _occ in rec["p"]:
        bw, bh = x1 - x0, y1 - y0
        if bw * w < 2 or bh * h < 2:
            continue
        lines.append(f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {bw:.6f} {bh:.6f}")
    if not lines:
        img_path.unlink(missing_ok=True)
        return {"image": iid, "dropped": "no_boxes"}
    lab = root / "labels" / split_dir / f"{stem}.txt"
    lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "image": f"openimages/{iid}.jpg", "source": "openimages", "modality": "visible", "lighting": "day",
        "split": "train" if split_dir == "train" else "val_oi", "sequence_key": f"openimages/{iid}",
        "final_image": f"images/{split_dir}/{stem}.jpg", "final_label": f"labels/{split_dir}/{stem}.txt",
        "label_rows": len(lines), "img_w": w, "img_h": h, "tiled": False, "is_negative": False,
        "author": rec.get("author", ""), "landing": rec.get("url", ""),
    }


def build(ids: list[str], split_dir: str, folder: str, cands: dict, lic: dict, root: Path, args, workers: int) -> tuple[list[dict], dict]:
    jobs = [(i, split_dir, root, {**cands[i], **lic[i]}, args.min_brightness, folder) for i in ids]
    rows, drops = [], defaultdict(int)
    with ThreadPoolExecutor(workers) as pool:
        for n, res in enumerate(pool.map(process, jobs), 1):
            if res is None:
                continue
            if "dropped" in res:
                drops[res["dropped"]] += 1
            else:
                rows.append(res)
            if n % 1000 == 0:
                print(f"  {split_dir}: {n}/{len(jobs)} processed, {len(rows)} kept", flush=True)
    return rows, dict(drops)


def shard(root: Path, out_shards: Path, per_shard: int = 4000) -> list[dict]:
    out_shards.mkdir(parents=True, exist_ok=True)
    files = sorted(p for d in ("images", "labels") for p in (root / d).rglob("*") if p.is_file())
    files += sorted(p for p in (root / "index").glob("*") if p.is_file()) + [root / "manifest_extra.tsv", root / "EXTRA_REPORT.json"]
    info = []
    for n in range(0, len(files), per_shard * 2):
        name = f"truewatch_extra_{n // (per_shard * 2):02d}.tar"
        with tarfile.open(out_shards / name, "w") as tar:
            for p in files[n:n + per_shard * 2]:
                tar.add(p, arcname=f"truewatch_extra/{p.relative_to(root).as_posix()}")
        digest = hashlib.sha256((out_shards / name).read_bytes()).hexdigest()
        info.append({"name": name, "bytes": (out_shards / name).stat().st_size, "sha256": digest})
    (out_shards / "SHARDS.json").write_text(json.dumps({"format": "truewatch_extra tar shards v1", "extract_to_parent_of": "truewatch_extra",
                                                         "shards": info}, indent=2), encoding="utf-8")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="where the Open Images CSVs live or are downloaded")
    ap.add_argument("--out", required=True, help="output tree root (becomes truewatch_extra/)")
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--min-persons", type=int, default=6)
    ap.add_argument("--min-small", type=int, default=4)
    ap.add_argument("--small-h", type=float, default=0.15, help="a person shorter than this share of the image height is small")
    ap.add_argument("--min-brightness", type=float, default=70.0)
    ap.add_argument("--oversample", type=float, default=1.25, help="candidates fetched per kept image, to cover dark/failed drops")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cache, root = Path(args.cache), Path(args.out)
    bad = exclude_mids(cache)
    print(f"excluded classes: {len(bad)} MIDs")
    report = {"licence": {"annotations": "CC BY 4.0", "images": "CC BY 2.0 (per-image License column == " + CC_BY + ")"},
              "source": "Open Images V6/V7 boxes, image list 2018_04", "params": vars(args)}
    all_rows = []
    for split_dir, folder, bbox, imgs, n in (
        ("train", "train", "v6/oidv6-train-annotations-bbox.csv", "2018_04/train/train-images-boxable-with-rotation.csv", args.n_train),
        ("val_oi", "validation", "v5/validation-annotations-bbox.csv", "2018_04/validation/validation-images-with-rotation.csv", args.n_val),
    ):
        b = fetch(f"{BBOX_BASE}/{bbox}", cache / Path(bbox).name)
        i = fetch(f"{BBOX_BASE}/{imgs}", cache / Path(imgs).name)
        cands, lic = scan(b, bad), licensed(i)
        ranked = rank(cands, lic, args.min_persons, args.min_small, args.small_h)
        report[f"{split_dir}_candidates"] = len(ranked)
        # take the top of the ranking but shuffle within it so the set is not only the single densest scenes
        pool = [iid for _s, iid in ranked[: int(n * args.oversample * 2)]]
        random.Random(args.seed).shuffle(pool)
        pool = pool[: int(n * args.oversample)]
        print(f"{split_dir}: {len(ranked)} qualifying images, downloading {len(pool)}", flush=True)
        rows, drops = build(pool, split_dir, folder, cands, lic, root, args, args.workers)
        rows.sort(key=lambda r: r["image"])
        rows = rows[:n]
        keep = {r["final_image"] for r in rows}
        for p in list((root / "images" / split_dir).glob("*.jpg")):
            if f"images/{split_dir}/{p.name}" not in keep:        # trim the oversample, labels with them
                p.unlink()
                (root / "labels" / split_dir / (p.stem + ".txt")).unlink(missing_ok=True)
        report[split_dir] = {"images": len(rows), "boxes": sum(r["label_rows"] for r in rows), "drops": drops}
        all_rows += rows
    (root / "index").mkdir(parents=True, exist_ok=True)
    (root / "index" / "split_extra.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows), encoding="utf-8")
    (root / "manifest_extra.tsv").write_text("".join(f"{r['split']}\t{r['final_image']}\t{r['label_rows']}\n" for r in all_rows), encoding="utf-8")
    (root / "EXTRA_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    info = shard(root, root.parent / "extra_shards")
    print(json.dumps(report, indent=2))
    print(f"shards: {[s['name'] for s in info]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
