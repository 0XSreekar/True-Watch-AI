#!/usr/bin/env python3
"""Elevated-view and thermal sources -> unified YOLO labels (DATASET_SPEC.md section 1.7).

The Phase 1 corpus is dashcam (IDD, FLIR) and street-level CCTV (LLVIP). Border cameras are
elevated and look down at small, distant people, so four sources are added for that view.
One script converts any of them; run it once per dataset with --dataset.

    hituav    HIT-UAV (CC BY 4.0). Thermal, UAV at 60-130 m, camera 30-90 deg, day and night.
              Mirror layout: hit-uav/{images,labels}/{train,val,test}/<name>.{jpg,txt}, YOLO ids
              0 Person, 1 Car, 2 Bicycle, 3 OtherVehicle, 4 DontCare. Name pattern
              <a>_<b>_<c>_<d>_<frame>: the first four fields identify the flight, which is the
              sequence key. The official split puts neighbouring frames of one flight into
              different splits, so 06_split.py re-splits HIT-UAV by flight.
    aaupdt    AAU-PD-T (Aalborg; "publicly available", no named licence: OQ-15). Thermal, fixed
              cameras about 9 m above sports fields. Mirror layout: Data/Train/<condition>/
              {Images,Annotations}/Image_<n>.{jpg,txt} and Data/Test/{Images,Annotations}. YOLO
              rows with class 0 = person. Image names repeat across condition folders, so every
              image is staged under a unique name. Sequence key: the condition folder for train;
              the official test frames carry no camera id and are keyed per frame.
    birdsai   BIRDSAI / Conservation Drones (CDLA-Permissive). Thermal UAV video at night over
              wildlife reserves. Layout: <TrainReal|TestReal>/annotations/<video>.csv and
              .../images/<video>/<video>_<n:010d>.jpg. CSV rows: frame, object_id, x, y, w, h,
              class, species, occlusion, noise; class 1 = human, 0 = animal. The CSV frame number
              is the position in the video's sorted image list, not the number in the file name
              (video 0000000011's images start at _0000000087 while its CSV starts at frame 0).
              Both facts were checked by drawing boxes on frames of videos 0000000011 (three
              humans) and 0000000058 (animals). Only frames inside a video's annotated range are
              used, every --frame-step-th of them; animals are kept
              as unlabelled objects so animal-only frames become hard negatives. Sequence key:
              the video; 06_split.py re-splits by video.
    visdrone  VisDrone2019-DET (CC BY-NC-SA 3.0, academic use). Visible, drone views of Asian
              streets. Layout: VisDrone2019-DET-{train,val,test-dev}/{images,annotations}; rows
              x, y, w, h, score, category, truncation, occlusion with categories 0 ignored,
              1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle,
              8 awning-tricycle, 9 bus, 10 motor, 11 others. Sequence key: the clip prefix of the
              file name; the official split is already by clip.

Every native class is mapped or dropped in the table for its dataset; an unmapped class fails
the run. Dropped road users are kept as unlabelled objects (never a negative frame).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

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
    stage_link,
    write_json,
    write_lines,
    xyxy_to_yolo,
)

SCRIPT = "12_convert_extra"
DATASETS = ("hituav", "aaupdt", "birdsai", "visdrone")

# Native class -> schema name, or None to drop. Each table lists every native class.
HITUAV_MAP = {0: "person", 1: "car", 2: "two_wheeler", 3: None, 4: None}
HITUAV_UNLABELLED = {3, 4}          # OtherVehicle, DontCare
AAUPDT_MAP = {0: "person"}
BIRDSAI_MAP = {1: "person", 0: None}
BIRDSAI_ANIMAL = 0
VISDRONE_MAP = {
    0: None,            # ignored region
    1: "person",        # pedestrian
    2: "person",        # people (sitting, lying, crouching)
    3: "two_wheeler",   # bicycle
    4: "car",
    5: "car",           # van
    6: "truck",
    7: "two_wheeler",   # tricycle (autorickshaw-like)
    8: "two_wheeler",   # awning-tricycle
    9: "truck",         # bus
    10: "two_wheeler",  # motor
    11: None,           # others
}
VISDRONE_UNLABELLED = {0, 11}

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
VISDRONE_SPLIT = {"train": "train", "val": "val", "test-dev": "test"}


class UnknownClass(Exception):
    pass


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as im:
            return im.size
    except (OSError, ValueError):
        return None


def norm_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> list[float]:
    return [round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5)]


def record(image: Path, label: Path, dataset: str, modality: str, sequence: str, official: str,
           size: tuple[int, int], lines: list[str], nonschema: list[list[float]], animal_only: bool,
           lighting: str, frame_index: int = -1) -> str:
    empty = not lines
    return json.dumps(
        {
            "image": image.as_posix(),
            "label": label.as_posix(),
            "source": dataset,
            "modality": modality,
            "lighting": lighting,
            "sequence_key": f"{dataset}/{sequence}",
            "official_split": official,
            "frame_index": frame_index,
            "img_w": size[0],
            "img_h": size[1],
            "objects": len(lines),
            # A frame with an unlabelled road user is never a negative; an animal-only frame is.
            "negative_worthy": bool(empty and (animal_only or not nonschema)),
            "animal_only": bool(empty and animal_only),
            "nonschema_boxes": nonschema,
        },
        sort_keys=True,
    )


def yolo_rows(path: Path, table: dict[int, str | None], unlabelled: set[int], ids: dict[str, int],
              counters: Counters) -> tuple[list[str], list[list[float]]]:
    """Re-map a YOLO label file; boxes are already normalised."""
    lines: list[str] = []
    nonschema: list[list[float]] = []
    if not path.exists():
        return lines, nonschema
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) != 5:
            counters.bump("label.malformed_row")
            continue
        native = int(float(parts[0]))
        if native not in table:
            raise UnknownClass(f"{path}: native class {native}")
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        if native in unlabelled:
            nonschema.append([round(cx - bw / 2, 5), round(cy - bh / 2, 5), round(cx + bw / 2, 5), round(cy + bh / 2, 5)])
        target = table[native]
        if target is None:
            counters.bump(f"dropped.{native}")
            continue
        if bw <= 0.0005 or bh <= 0.0005:
            counters.bump("box.degenerate_dropped")
            continue
        lines.append(format_label_line(ids[target], min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0), min(bw, 1.0), min(bh, 1.0)))
        counters.bump(f"mapped.{target}")
    return lines, nonschema


def convert_hituav(root: Path, ids, args, counters, log, out_labels: Path, done: set[str]) -> list[str]:
    records: list[str] = []
    for official in ("train", "val", "test"):
        images = sorted(p for p in (root / "images" / official).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        log.info("split", dataset="hituav", official=official, images=len(images))
        for image in images[: args.limit] if args.limit else images:
            key = f"hituav/{official}/{image.name}"
            if key in done and not args.force:
                counters.bump("image.already_done")
                continue
            parts = image.stem.split("_")
            if len(parts) < 5:
                counters.bump("image.unexpected_name")
                log.warn("name does not match <a>_<b>_<c>_<d>_<frame> - skipped and counted", image=image.name)
                continue
            size = image_size(image)
            if size is None:
                counters.bump("image.unreadable")
                continue
            lines, nonschema = yolo_rows(root / "labels" / official / f"{image.stem}.txt", HITUAV_MAP, HITUAV_UNLABELLED, ids, counters)
            label = out_labels / f"{image.stem}.txt"
            if not args.dry_run:
                write_lines(label, lines)
            records.append(record(image, label, "hituav", "lwir", "_".join(parts[:4]), official, size, lines, nonschema,
                                  False, "unknown", int(parts[4]) if parts[4].isdigit() else -1))
            counters.bump(f"{official}.converted")
            done.add(key)
    return records


def convert_aaupdt(root: Path, ids, args, counters, log, out_labels: Path, stage: Path, done: set[str]) -> list[str]:
    records: list[str] = []
    groups: list[tuple[str, str, Path]] = []
    for condition in sorted(p for p in (root / "Train").iterdir() if p.is_dir()):
        groups.append(("train", condition.name, condition))
    groups.append(("test", "Test", root / "Test"))
    for official, group, folder in groups:
        images = sorted(p for p in (folder / "Images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        log.info("group", dataset="aaupdt", official=official, group=group, images=len(images))
        for image in images[: args.limit] if args.limit else images:
            uid = f"{group}__{image.stem}"
            key = f"aaupdt/{uid}"
            if key in done and not args.force:
                counters.bump("image.already_done")
                continue
            size = image_size(image)
            if size is None:
                counters.bump("image.unreadable")
                continue
            staged = stage / f"{uid}{image.suffix.lower()}"
            if not stage_link(image, staged, counters, args.dry_run):
                counters.bump("image.stage_failed")
                continue
            lines, nonschema = yolo_rows(folder / "Annotations" / f"{image.stem}.txt", AAUPDT_MAP, set(), ids, counters)
            label = out_labels / f"{uid}.txt"
            if not args.dry_run:
                write_lines(label, lines)
            sequence = f"train/{group}" if official == "train" else f"test/{image.stem}"
            records.append(record(staged, label, "aaupdt", "lwir", sequence, official, size, lines, nonschema,
                                  False, "unknown"))
            counters.bump(f"{official}.converted")
            done.add(key)
    return records


def convert_birdsai(root: Path, ids, args, counters, log, out_labels: Path, done: set[str]) -> list[str]:
    records: list[str] = []
    csvs = sorted(p for p in root.rglob("*.csv") if p.parent.name == "annotations")
    log.info("videos", dataset="birdsai", annotation_files=len(csvs), frame_step=args.frame_step)
    for index, ann in enumerate(csvs[: args.limit] if args.limit else csvs):
        video = ann.stem
        official = "train" if "TrainReal" in ann.parts else ("test" if "TestReal" in ann.parts else "unknown")
        image_dir = ann.parent.parent / "images" / video
        if not image_dir.is_dir():
            counters.bump("video.images_missing")
            log.warn("annotation without an image folder - skipped and counted", video=video)
            continue
        by_frame: dict[int, list[list[str]]] = defaultdict(list)
        with ann.open(encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 7:
                    counters.bump("csv.short_row")
                    continue
                by_frame[int(float(row[0]))].append(row)
        if not by_frame:
            counters.bump("video.no_rows")
            continue
        frames_on_disk = sorted(p for p in image_dir.glob(f"{video}_*.jpg"))
        first, last = min(by_frame), max(by_frame)
        if last >= len(frames_on_disk):
            counters.bump("video.annotations_beyond_images")
            log.warn("annotated frames exceed the images on disk - trailing frames skipped",
                     video=video, last_frame=last, images=len(frames_on_disk))
        for frame in range(first, min(last, len(frames_on_disk) - 1) + 1, args.frame_step):
            image = frames_on_disk[frame]
            key = f"birdsai/{video}/{frame}"
            if key in done and not args.force:
                counters.bump("image.already_done")
                continue
            size = image_size(image)
            if size is None:
                counters.bump("image.unreadable")
                continue
            width, height = size
            lines: list[str] = []
            nonschema: list[list[float]] = []
            animals = 0
            for row in by_frame.get(frame, []):
                x, y, w, h = (float(v) for v in row[2:6])
                native = int(float(row[6]))
                if native not in BIRDSAI_MAP:
                    raise UnknownClass(f"{ann}: class {native}")
                if native == BIRDSAI_ANIMAL:
                    animals += 1
                    counters.bump("dropped.animal")
                    continue
                geometry = xyxy_to_yolo(x, y, x + w, y + h, width, height)
                if geometry is None:
                    counters.bump("box.degenerate_dropped")
                    continue
                lines.append(format_label_line(ids["person"], *geometry))
                counters.bump("mapped.person")
            label = out_labels / f"{image.stem}.txt"
            if not args.dry_run:
                write_lines(label, lines)
            records.append(record(image, label, "birdsai", "lwir", video, official, size, lines, nonschema,
                                  animals > 0, "night", frame))
            counters.bump(f"{official}.converted")
            done.add(key)
        if (index + 1) % 10 == 0:
            log.info("progress", videos=index + 1, of=len(csvs), records=len(records))
    return records


def convert_visdrone(root: Path, ids, args, counters, log, out_labels: Path, done: set[str]) -> list[str]:
    records: list[str] = []
    for native_split, official in VISDRONE_SPLIT.items():
        folders = sorted(p for p in root.rglob(f"VisDrone2019-DET-{native_split}") if (p / "images").is_dir())
        if not folders:
            log.warn("split folder not found", split=native_split)
            counters.bump(f"{native_split}.folder_missing")
            continue
        folder = folders[0]
        images = sorted(p for p in (folder / "images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        log.info("split", dataset="visdrone", official=native_split, images=len(images))
        for image in images[: args.limit] if args.limit else images:
            key = f"visdrone/{native_split}/{image.name}"
            if key in done and not args.force:
                counters.bump("image.already_done")
                continue
            size = image_size(image)
            if size is None:
                counters.bump("image.unreadable")
                continue
            width, height = size
            lines: list[str] = []
            nonschema: list[list[float]] = []
            ann = folder / "annotations" / f"{image.stem}.txt"
            if not ann.exists():
                counters.bump("image.no_annotation")
                continue
            for raw in ann.read_text(encoding="utf-8").splitlines():
                parts = [p for p in raw.strip().split(",") if p != ""]
                if len(parts) < 6:
                    continue
                x, y, w, h = (float(v) for v in parts[:4])
                native = int(parts[5])
                if native not in VISDRONE_MAP:
                    raise UnknownClass(f"{ann}: category {native}")
                if native in VISDRONE_UNLABELLED:
                    nonschema.append(norm_box(x, y, x + w, y + h, width, height))
                target = VISDRONE_MAP[native]
                if target is None:
                    counters.bump(f"dropped.{native}")
                    continue
                geometry = xyxy_to_yolo(x, y, x + w, y + h, width, height)
                if geometry is None:
                    counters.bump("box.degenerate_dropped")
                    continue
                lines.append(format_label_line(ids[target], *geometry))
                counters.bump(f"mapped.{target}")
            label = out_labels / f"{image.stem}.txt"
            if not args.dry_run:
                write_lines(label, lines)
            clip = re.match(r"(\d+_\d+)", image.stem)
            records.append(record(image, label, "visdrone", "visible", clip.group(1) if clip else image.stem, official,
                                  size, lines, nonschema, False, "day"))
            counters.bump(f"{official}.converted")
            done.add(key)
    return records


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--source", default=None, help="dataset root, or any directory above it")
    ap.add_argument("--frame-step", type=int, default=None,
                    help="birdsai: keep every n-th annotated frame (default: sources.yaml birdsai.frame_step)")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    spec = load_sources()["sources"].get(args.dataset, {})
    if not spec.get("enabled", True):
        log.info("source disabled in sources.yaml - nothing to do", dataset=args.dataset)
        return 0
    if args.frame_step is None:
        args.frame_step = int(spec.get("frame_step", 1))
    root = resolve_source_root(args.dataset, args, log)
    if root is None:
        log.error("source root not found - run 00_fetch.py or pass --source", dataset=args.dataset)
        return 1
    processed = Path(args.processed).resolve()
    out_labels = processed / "labels" / args.dataset
    out_index = processed / "index" / f"{args.dataset}.jsonl"
    stage = processed / "staged" / args.dataset
    require_dirs(out_labels, out_index.parent)

    ids = class_ids()
    state_name = f"{SCRIPT}_{args.dataset}"
    state = load_state(processed, state_name)
    done = set(state.get("done", [])) if not args.force else set()

    try:
        if args.dataset == "hituav":
            records = convert_hituav(root, ids, args, counters, log, out_labels, done)
        elif args.dataset == "aaupdt":
            records = convert_aaupdt(root, ids, args, counters, log, out_labels, stage, done)
        elif args.dataset == "birdsai":
            records = convert_birdsai(root, ids, args, counters, log, out_labels, done)
        else:
            records = convert_visdrone(root, ids, args, counters, log, out_labels, done)
    except UnknownClass as exc:
        log.error("native class not present in the class map - refusing to guess", detail=str(exc)[:300])
        return 1

    if not args.dry_run:
        write_lines(out_index, merge_jsonl(out_index, records))
        state["done"] = sorted(done)
        save_state(processed, state_name, state)
        by_split = Counter(json.loads(r)["official_split"] for r in records)
        write_json(processed / "reports" / f"convert_{args.dataset}.json",
                   {"written": len(records), "by_official_split": dict(sorted(by_split.items())),
                    "counters": counters.as_dict()})

    counters.report(log)
    log.info("done", dataset=args.dataset, labels_written=len(records))
    if not records and not counters.as_dict().get("image.already_done"):
        log.error("no images converted", dataset=args.dataset)
        return 1
    return 0


if __name__ == "__main__":
    run(main)
