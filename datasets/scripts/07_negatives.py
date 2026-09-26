#!/usr/bin/env python3
"""Build the background/negative image pool. DATASET_SPEC.md sections 3.5 and 5.

Why this script matters more than its size suggests: slide 5's "under 5 false alerts
per camera per day" is a precision target, and precision on a 24-hour stream is decided
by how the model behaves on the 23 hours that contain nothing. A detector that has never
been shown an empty frame has no way to represent one.

Target: composition.negatives.ratio (10%) of TRAINING images - far-field tiles included,
because they are training images too - carry an empty label file (gate G9, +/- 2%). The tile
plan is computed with _lib.plan_object_tiles, the same function 08 uses, so this script sizes
the pool against exactly the tiles that will exist. Negatives go to train only; val and test
keep their natural composition.

Order of preference, each step only if the previous one is short:
  1. whole empty frames by category quota (section 5): empty_road, ir_hotspot, night_glare,
     foliage, weather, animal;
  2. any remaining whole empty frame (category kept, tagged redistributed);
  3. empty far-field tiles (composition.negatives.farfield_tile_fallback): a 640x640 native
     crop of the upper strip that NO annotated box of any native class touches, dropped
     classes included. That is precisely the input the edge tiler hands the detector on an
     empty approach road. At most farfield_tiles_per_frame per parent frame.

Train frames with no objects that are NOT selected are excluded from the build (their
number is reported), otherwise every empty frame would become a negative and the ratio
would be whatever the sources happen to contain.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    MANIFEST_DIR,
    Counters,
    Logger,
    base_parser,
    imread_gray,
    load_composition,
    parse_label_file,
    plan_object_tiles,
    read_jsonl,
    require_dirs,
    run,
    tile_windows,
    write_json,
    write_lines,
)

SCRIPT = "07_negatives"

QUOTAS = {
    "empty_road": 0.30,
    "ir_hotspot": 0.20,
    "night_glare": 0.20,
    "foliage": 0.15,
    "weather": 0.10,
    "animal": 0.05,
}

HOTSPOT_MIN_AREA = 100
HOTSPOT_MAX_AREA = 2000


def classify(record: dict, counters: Counters) -> str | None:
    """Assign a negative category. Returns None when the frame is not a usable negative."""
    if record.get("objects", 0) > 0:
        return None
    if record.get("source") == "llvip":
        # LLVIP carries unlabelled vehicles (section 1.4); an empty LLVIP frame is not
        # evidence of an empty scene.
        counters.bump("rejected.llvip_person_only")
        return None
    if not record.get("negative_worthy", False):
        counters.bump("rejected.unlabelled_object_in_frame")
        return None
    if record.get("animal_only"):
        return "animal"
    if record.get("modality") == "lwir":
        return "ir_hotspot"
    if record.get("lighting") in ("night", "dawn_dusk"):
        return "night_glare"
    return "empty_road"


def has_person_sized_hotspot(path: Path) -> bool:
    """A sun-warmed rock or an engine block at person temperature and person size."""
    import cv2
    import numpy as np

    gray = imread_gray(path)
    if gray is None:
        return False
    threshold = float(np.percentile(gray, 99.0))
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(HOTSPOT_MIN_AREA <= cv2.contourArea(c) <= HOTSPOT_MAX_AREA for c in contours)


def empty_tile_candidates(train: list[dict], tiles_cfg: dict, per_frame: int, rng: random.Random,
                          counters: Counters) -> list[dict]:
    """Far-field windows that no annotated box, kept or dropped, touches."""
    tile = int(tiles_cfg.get("size", 640))
    overlap = float(tiles_cfg.get("overlap", 0.20))
    top, bottom = float(tiles_cfg.get("strip_top", 0.0)), float(tiles_cfg.get("strip_bottom", 0.40))
    out: list[dict] = []
    for record in train:
        if record.get("source") == "llvip":
            continue  # unlabelled vehicles
        width, height = int(record.get("img_w") or 0), int(record.get("img_h") or 0)
        windows = tile_windows(width, height, tile, overlap, top, bottom)
        if not windows:
            continue
        rows, _errors = parse_label_file(Path(record["label"])) if record.get("objects") else ([], [])
        boxes = [((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)
                 for _c, cx, cy, w, h in rows]
        boxes += [(x1 * width, y1 * height, x2 * width, y2 * height) for x1, y1, x2, y2 in record.get("nonschema_boxes", [])]
        free = [(ox, oy) for ox, oy in windows
                if not any(b[0] < ox + tile and b[2] > ox and b[1] < oy + tile and b[3] > oy for b in boxes)]
        if not free:
            continue
        rng.shuffle(free)
        for ox, oy in free[:per_frame]:
            out.append({"parent": record["image"], "ox": ox, "oy": oy, "modality": record.get("modality"),
                        "source": record.get("source"), "sequence_key": record.get("sequence_key")})
            counters.bump("farfield_empty.candidate")
    return out


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--ratio", type=float, default=None, help="override composition.negatives.ratio")
    ap.add_argument("--verify-hotspots", action="store_true",
                    help="open each LWIR candidate and keep only those with a person-sized hot blob")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()
    rng = random.Random(args.seed)

    composition = load_composition()
    neg_cfg = composition.get("negatives", {})
    ratio = float(args.ratio if args.ratio is not None else neg_cfg.get("ratio", 0.10))

    processed = Path(args.processed).resolve()
    index_path = processed / "index" / "split.jsonl"
    if not index_path.exists():
        log.error("split index missing - run 06_split.py first", expected=str(index_path))
        return 1

    records = sorted(read_jsonl(index_path), key=lambda r: r["image"])
    train = [r for r in records if r.get("split") == "train"]
    positives = [r for r in train if r.get("objects", 0) > 0]
    planned_tiles = plan_object_tiles(train, args.seed)
    log.info("training records", total=len(train), positives=len(positives), planned_object_tiles=len(planned_tiles))
    if not positives:
        log.error("no positive training images - nothing to size the negative pool against")
        return 1

    base = len(positives) + len(planned_tiles)
    # n / (base + n) = ratio  =>  n = ratio * base / (1 - ratio)
    target = int(round(ratio * base / max(1e-9, 1.0 - ratio)))
    log.info("negative pool target", target=target, ratio=ratio, sized_against=base)

    candidates: dict[str, list[dict]] = {key: [] for key in QUOTAS}
    for record in train:
        category = classify(record, counters)
        if category is not None:
            candidates[category].append(record)
            counters.bump(f"candidate.{category}")

    if args.verify_hotspots and not args.dry_run:
        verified = [r for r in candidates["ir_hotspot"] if has_person_sized_hotspot(Path(r["image"]))]
        counters.bump("ir_hotspot.no_blob", len(candidates["ir_hotspot"]) - len(verified))
        candidates["ir_hotspot"] = verified

    chosen: list[dict] = []
    shortfalls: dict[str, int] = {}
    for category, share in sorted(QUOTAS.items()):
        want = int(round(target * share))
        pool = sorted(candidates[category], key=lambda r: r["image"])
        rng.shuffle(pool)
        take = pool[:want]
        if len(take) < want:
            shortfalls[category] = want - len(take)
            log.warn("category short of its quota", category=category, want=want, have=len(take))
        for record in take:
            record["negative_category"] = category
        chosen.extend(take)
        counters.bump(f"selected.{category}", len(take))

    missing = target - len(chosen)
    if missing > 0:
        already = {r["image"] for r in chosen}
        spare = sorted((r for pool in candidates.values() for r in pool if r["image"] not in already),
                       key=lambda r: r["image"])
        rng.shuffle(spare)
        for record in spare[:missing]:
            record["negative_category"] = "redistributed"
            chosen.append(record)
            counters.bump("selected.redistributed")

    tile_negatives: list[dict] = []
    missing = target - len(chosen)
    if missing > 0 and neg_cfg.get("farfield_tile_fallback", True):
        pool = empty_tile_candidates(train, composition.get("tiles", {}), int(neg_cfg.get("farfield_tiles_per_frame", 1)),
                                     random.Random(f"{args.seed}-empty-tiles"), counters)
        rng.shuffle(pool)
        tile_negatives = sorted(pool[:missing], key=lambda t: (t["parent"], t["oy"], t["ox"]))
        counters.bump("selected.farfield_empty_tile", len(tile_negatives))
        log.info("whole empty frames short - far-field empty tiles fill the pool",
                 needed=missing, available=len(pool), taken=len(tile_negatives))

    total_negatives = len(chosen) + len(tile_negatives)
    achieved = total_negatives / max(1, base + total_negatives)
    selected = {r["image"] for r in chosen}
    excluded = [r for r in train if r.get("objects", 0) == 0 and r["image"] not in selected]
    counters.bump("train_empty.not_selected_excluded", len(excluded))
    modality_mix = Counter(r.get("modality") for r in chosen) + Counter(t.get("modality") for t in tile_negatives)
    log.info("negative pool built", whole_frames=len(chosen), farfield_tiles=len(tile_negatives), target=target,
             achieved_ratio=round(achieved, 4), band="0.08-0.12 (gate G9)", modality=dict(modality_mix),
             empty_train_frames_excluded=len(excluded))

    by_category = Counter(r.get("negative_category") for r in chosen)
    if tile_negatives:
        by_category["farfield_empty_tile"] = len(tile_negatives)
    payload = {
        "target": target,
        "selected": total_negatives,
        "whole_frames": len(chosen),
        "farfield_empty_tiles": len(tile_negatives),
        "achieved_ratio": round(achieved, 6),
        "requested_ratio": ratio,
        "sized_against": {"train_positives": len(positives), "planned_object_tiles": len(planned_tiles)},
        "by_category": dict(sorted(by_category.items())),
        "by_modality": dict(sorted(modality_mix.items())),
        "shortfalls": shortfalls,
        "empty_train_frames_excluded": len(excluded),
        "counters": counters.as_dict(),
    }

    if args.dry_run:
        log.info("dry-run: nothing written")
        counters.report(log)
        return 0

    require_dirs(MANIFEST_DIR)
    write_json(MANIFEST_DIR / "negatives.json", payload)
    write_lines(MANIFEST_DIR / "negatives.txt",
                sorted(r["image"] for r in chosen) + [f"{t['parent']}#t{t['ox']}_{t['oy']}" for t in tile_negatives])

    excluded_images = {r["image"] for r in excluded}
    for record in records:
        record["is_negative"] = record["image"] in selected
        record["excluded_empty"] = record["image"] in excluded_images
    write_lines(processed / "index" / "with_negatives.jsonl", [json.dumps(r, sort_keys=True) for r in records])
    write_lines(processed / "index" / "negative_tiles.jsonl", [json.dumps(t, sort_keys=True) for t in tile_negatives])
    write_json(processed / "reports" / "negatives.json", payload)

    counters.report(log)
    if achieved < 0.08 or achieved > 0.12:
        log.error("negative ratio outside the gate G9 band", achieved=round(achieved, 4))
        return 1
    log.info("done", negatives=total_negatives)
    return 0


if __name__ == "__main__":
    run(main)
