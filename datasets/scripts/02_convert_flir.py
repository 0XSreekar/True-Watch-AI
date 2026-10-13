#!/usr/bin/env python3
"""Teledyne FLIR ADAS v2 -> unified YOLO labels. Replaces KAIST (DATASET_SPEC.md section 1.3).

Layout (verified 2026-09 on the Kaggle mirror samdazel/teledyne-flir-adas-thermal-dataset-v2,
which keeps the original FLIR_ADAS_v2 tree):

    <subset>/coco.json          COCO detection json; images[].file_name = data/<name>.jpg
    <subset>/data/*.jpg         video-<videoId>-frame-<n>-<hash>.jpg

    subsets: images_rgb_train, images_rgb_val, images_thermal_train, images_thermal_val,
             video_rgb_test, video_thermal_test          (sources.yaml flir.subsets)

The sequence key is the videoId (flir/<videoId>). RGB and thermal captures carry different
videoIds, and file names are unique across the whole release, so no staging is needed: the
record points at the source file directly and the source root may be read-only.

Class map (DATASET_SPEC.md section 1.3). FLIR's coco.json declares the 80 COCO-style names;
16 of them actually occur. Each one that occurs is mapped or dropped here, explicitly; a name
that occurs and is in neither table fails the run.

    person                     -> 0 person
    bike, motor, scooter       -> 1 two_wheeler
    car                        -> 2 car
    truck, bus                 -> 3 truck
    other vehicle              -> DROPPED. FLIR's catch-all (trailers, construction and farm
                                  vehicles). Merging it into truck would corrupt the truck size
                                  prior exactly as IDD's `vehicle fallback` would (section 1.5);
                                  its boxes are kept as unlabelled objects so the frame is
                                  never used as a negative and no negative tile contains one.
    light, sign, hydrant       -> DROPPED, static infrastructure
    train                      -> DROPPED, rail (unlabelled object)
    dog, deer                  -> DROPPED, animals (unlabelled objects; an animal-only frame
                                  is an `animal` hard negative, section 5)
    skateboard, stroller       -> DROPPED (unlabelled objects; the person using one is
                                  annotated separately as person)
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    Counters,
    Logger,
    base_parser,
    class_ids,
    format_label_line,
    load_sources,
    load_state,
    merge_jsonl,
    require_dirs,
    resolve_source_root,
    run,
    save_state,
    write_json,
    write_lines,
    xyxy_to_yolo,
)

SCRIPT = "02_convert_flir"

FLIR_MAP: dict[str, str | None] = {
    "person": "person",
    "rider": "person",
    "bike": "two_wheeler",
    "motor": "two_wheeler",
    "scooter": "two_wheeler",
    "car": "car",
    "truck": "truck",
    "bus": "truck",
    "other vehicle": None,
    "light": None,
    "sign": None,
    "hydrant": None,
    "train": None,
    "dog": None,
    "deer": None,
    "skateboard": None,
    "stroller": None,
}
UNLABELLED_OBJECTS = {"other vehicle", "train", "dog", "deer", "skateboard", "stroller"}
ANIMALS = {"dog", "deer"}
BLOCKS_NEGATIVE = UNLABELLED_OBJECTS - ANIMALS

NAME_RE = re.compile(r"video-([A-Za-z0-9]+)-frame-(\d+)")
LIGHTING = {"day": "day", "night": "night", "dawn/dusk": "dawn_dusk"}


def convert_subset(root: Path, subset: str, meta: dict, ids: dict[str, int], args, counters: Counters,
                   log: Logger, out_labels: Path, done: set[str], unknown: set[str]) -> tuple[list[str], int]:
    coco_path = root / subset / "coco.json"
    if not coco_path.exists():
        log.error("coco.json missing", subset=subset, expected=str(coco_path))
        counters.bump(f"{subset}.coco_missing")
        return [], 0
    with coco_path.open(encoding="utf-8") as fh:
        coco = json.load(fh)
    categories = {c["id"]: " ".join(str(c["name"]).lower().split()) for c in coco.get("categories", [])}
    per_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        per_image.setdefault(ann["image_id"], []).append(ann)

    modality = meta["modality"]
    official = meta["official_split"]
    images = sorted(coco.get("images", []), key=lambda im: im["file_name"])
    log.info("subset", subset=subset, images=len(images), annotations=len(coco.get("annotations", [])),
             modality=modality, official_split=official)
    if args.limit:
        images = images[: args.limit]

    records: list[str] = []
    missing = 0
    for im in images:
        image = root / subset / im["file_name"]
        key = f"{subset}/{im['file_name']}"
        if key in done and not args.force:
            counters.bump("image.already_done")
            continue
        if not image.exists():
            missing += 1
            counters.bump(f"{subset}.image_missing")
            continue
        match = NAME_RE.search(image.name)
        extra = im.get("extra_info") or {}
        video = extra.get("video_id") or (match.group(1) if match else None)
        if not video:
            counters.bump("image.no_video_id")
            log.warn("frame carries no video id - skipped and counted", image=key)
            continue
        width, height = int(im.get("width") or 0), int(im.get("height") or 0)
        if width <= 0 or height <= 0:
            counters.bump("image.no_size")
            log.warn("coco image has no size - skipped and counted", image=key)
            continue

        lines: list[str] = []
        natives: set[str] = set()
        nonschema: list[list[float]] = []
        for ann in per_image.get(im["id"], []):
            native = categories.get(ann["category_id"], f"id{ann['category_id']}")
            natives.add(native)
            if native not in FLIR_MAP:
                unknown.add(native)
                counters.bump(f"native.UNKNOWN.{native}")
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
            if native in UNLABELLED_OBJECTS:
                nonschema.append([round(x / width, 5), round(y / height, 5),
                                  round((x + w) / width, 5), round((y + h) / height, 5)])
            target = FLIR_MAP[native]
            if target is None:
                counters.bump(f"dropped.{native}")
                continue
            geometry = xyxy_to_yolo(x, y, x + w, y + h, width, height)
            if geometry is None:
                counters.bump("box.degenerate_dropped")
                continue
            lines.append(format_label_line(ids[target], *geometry))
            counters.bump(f"mapped.{target}")

        label_path = out_labels / modality / f"{image.stem}.txt"
        if not args.dry_run:
            write_lines(label_path, lines)
        empty = not lines
        negative_worthy = empty and not (natives & BLOCKS_NEGATIVE)
        records.append(
            json.dumps(
                {
                    "image": image.as_posix(),
                    "label": label_path.as_posix(),
                    "source": "flir",
                    "modality": modality,
                    "lighting": LIGHTING.get(str(extra.get("hours", "")).lower(), "unknown"),
                    "sequence_key": f"flir/{video}",
                    "video_id": video,
                    "frame_index": int(match.group(2)) if match else -1,
                    "flir_subset": subset,
                    "official_split": official,
                    "img_w": width,
                    "img_h": height,
                    "objects": len(lines),
                    "negative_worthy": bool(negative_worthy),
                    "animal_only": bool(negative_worthy and natives & ANIMALS),
                    "nonschema_boxes": nonschema,
                },
                sort_keys=True,
            )
        )
        counters.bump(f"{subset}.converted")
        if empty:
            counters.bump("image.zero_objects")
        done.add(key)
    return records, missing


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--source", default=None, help="FLIR_ADAS_v2 root, or any directory above it")
    ap.add_argument("--subset-ok", action="store_true",
                    help="the root is a deliberately partial fixture: coco entries with no file are counted, not fatal")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    root = resolve_source_root("flir", args, log)
    if root is None:
        log.error("FLIR root not found - run 00_fetch.py or pass --source")
        return 1
    processed = Path(args.processed).resolve()
    out_labels = processed / "labels" / "flir"
    out_index = processed / "index" / "flir.jsonl"
    require_dirs(out_labels, out_index.parent)

    ids = class_ids()
    subsets = load_sources()["sources"]["flir"]["subsets"]
    state = load_state(processed, SCRIPT)
    done = set(state.get("done", [])) if not args.force else set()

    records: list[str] = []
    unknown: set[str] = set()
    missing_total = 0
    for subset, meta in sorted(subsets.items()):
        subset_records, missing = convert_subset(root, subset, meta, ids, args, counters, log, out_labels, done, unknown)
        records.extend(subset_records)
        missing_total += missing
        if missing:
            (log.warn if args.subset_ok else log.error)("coco entries without an image file", subset=subset, missing=missing)

    # videoId -> official splits, reported here and ASSERTED in 06_split.py.
    video_splits: dict[str, set[str]] = {}
    for line in records:
        rec = json.loads(line)
        video_splits.setdefault(rec["video_id"], set()).add(rec["official_split"])
    shared = sorted(v for v, s in video_splits.items() if len(s) > 1)
    if shared:
        log.warn("videoIds present in more than one official split - 06_split.py will re-split them",
                 videos=len(shared), example=shared[0])
    counters.bump("videos", len(video_splits))

    if unknown:
        log.error("native classes not present in the class map - refusing to guess", classes=",".join(sorted(unknown)))

    if not args.dry_run:
        write_lines(out_index, merge_jsonl(out_index, records))
        state["done"] = sorted(done)
        save_state(processed, SCRIPT, state)
        by_split = Counter(json.loads(r)["official_split"] + "/" + json.loads(r)["modality"] for r in records)
        write_json(processed / "reports" / "convert_flir.json",
                   {"written": len(records), "by_split_modality": dict(sorted(by_split.items())),
                    "videos_in_two_official_splits": shared, "counters": counters.as_dict()})

    counters.report(log)
    log.info("done", labels_written=len(records), missing_images=missing_total)
    if unknown:
        return 1
    if missing_total and not args.subset_ok:
        return 1
    return 0


if __name__ == "__main__":
    run(main)
