#!/usr/bin/env python3
"""LLVIP -> unified YOLO labels.

LLVIP annotates one class, person, in registered visible/infrared pairs
(DATASET_SPEC.md section 1.4). Nothing is dropped.

Two hazards handled here:
  * LLVIP frames contain UNLABELLED vehicles. An unlabelled car teaches the model
    that cars are background, so every LLVIP record is flagged person_only=true and
    09_build_yolo_ds.py excludes them from the vehicle-class loss accounting.
  * Layout (verified 2026-09 on the Kaggle mirror afradhossain/llvip-dataset and on the
    upstream LLVIP.zip): Annotations/<id>.xml, visible/{train,test}/<id>.jpg,
    infrared/{train,test}/<id>.jpg; 12,025 train and 3,463 test pairs. A visible frame and its
    infrared partner share the id; the final file names differ by the _visible/_lwir suffix.
    The record points at the source file itself, so the root may be read-only.
  * Scene grouping (OQ-3). The filename prefix is tried first; if it does not
    partition the corpus usefully the record carries scene_key=null and 06_split.py
    falls back to perceptual-hash pseudo-sequences.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    IMAGE_SUFFIXES,
    Counters,
    Logger,
    base_parser,
    class_ids,
    format_label_line,
    imread_any,
    load_splits_config,
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

SCRIPT = "03_convert_llvip"

LLVIP_MAP = {"person": "person"}


def modality_of(path: Path) -> str:
    text = path.as_posix().lower()
    if "infrared" in text or "lwir" in text or "thermal" in text or "/ir/" in text:
        return "lwir"
    return "visible"


def official_split_of(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if "test" in parts:
        return "test"
    return "train"


def scene_key_from_name(stem: str, prefix_length: int) -> str | None:
    head = stem[:prefix_length]
    return head if head.isdigit() else None


def parse_voc(path: Path, counters: Counters, log: Logger):
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        log.error("unparseable annotation", path=str(path), detail=str(exc)[:120])
        counters.bump("annotation.unparseable")
        return None
    root = tree.getroot()
    size = root.find("size")
    width = height = 0
    if size is not None:
        width = int(float(size.findtext("width") or 0))
        height = int(float(size.findtext("height") or 0))
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        box = obj.find("bndbox")
        if box is None:
            counters.bump("object.no_bndbox")
            continue
        try:
            objects.append(
                (
                    name,
                    (
                        float(box.findtext("xmin")),
                        float(box.findtext("ymin")),
                        float(box.findtext("xmax")),
                        float(box.findtext("ymax")),
                    ),
                )
            )
        except (TypeError, ValueError):
            counters.bump("object.bad_coords")
    return width, height, objects


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--source", default=None, help="LLVIP root, or any directory above it")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    root = resolve_source_root("llvip", args, log)
    if root is None:
        log.error("LLVIP root not found - run 00_fetch.py or pass --source")
        return 1
    processed = Path(args.processed).resolve()
    out_labels = processed / "labels" / "llvip"
    out_index = processed / "index" / "llvip.jsonl"

    ids = class_ids()
    config = load_splits_config()["llvip"]
    prefix_length = int(config["scene_key"].get("prefix_length", 2))

    annotations = sorted((root / "Annotations").glob("*.xml")) or sorted(root.rglob("*.xml"))
    if not annotations:
        log.error("no VOC annotations found", root=str(root))
        return 1
    if args.limit:
        annotations = annotations[: args.limit]
    log.info("annotations found", annotations=len(annotations))

    image_index: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and "Annotations" not in path.parts:
            image_index.setdefault(path.stem, []).append(path)
    log.info("image index built", stems=len(image_index))

    state = load_state(processed, SCRIPT)
    done = set(state.get("done", [])) if not args.force else set()
    require_dirs(out_labels, out_index.parent)

    records: list[str] = []
    unknown_classes: set[str] = set()
    scene_keys: set[str] = set()
    written = 0

    for annotation in annotations:
        key = annotation.as_posix()
        scene = scene_key_from_name(annotation.stem, prefix_length)
        if scene:
            scene_keys.add(scene)
        if key in done:
            counters.bump("annotation.already_done")
            continue

        parsed = parse_voc(annotation, counters, log)
        if parsed is None:
            continue
        width, height, objects = parsed

        partners = image_index.get(annotation.stem, [])
        if not partners:
            counters.bump("annotation.no_image")
            log.warn("annotation has no image", annotation=annotation.name)
            continue

        geometry_rows: list[tuple[str, tuple[float, float, float, float]]] = []
        for name, coords in objects:
            if name not in LLVIP_MAP:
                unknown_classes.add(name)
                counters.bump(f"native.UNKNOWN.{name}")
                continue
            geometry_rows.append((LLVIP_MAP[name], coords))

        for image in sorted(partners):
            modality = modality_of(image)
            image_width, image_height = width, height
            if image_width <= 0 or image_height <= 0:
                probe = imread_any(image)
                if probe is None:
                    counters.bump("image.unreadable")
                    log.warn("unreadable image", image=image.as_posix())
                    continue
                image_height, image_width = probe.shape[:2]
                counters.bump("image.size_recovered")

            lines: list[str] = []
            for target, coords in geometry_rows:
                normalised = xyxy_to_yolo(*coords, image_width, image_height)
                if normalised is None:
                    counters.bump("box.degenerate_dropped")
                    continue
                lines.append(format_label_line(ids[target], *normalised))
                counters.bump(f"mapped.{target}")

            label_path = out_labels / f"{annotation.stem}_{modality}.txt"
            if not args.dry_run:
                write_lines(label_path, lines)
            written += 1

            records.append(
                json.dumps(
                    {
                        "image": image.as_posix(),
                        "label": label_path.as_posix(),
                        "source": "llvip",
                        "modality": modality,
                        "sequence_key": f"llvip/{scene}" if scene else None,
                        "pair_id": f"llvip/{annotation.stem}",
                        "official_split": official_split_of(image.relative_to(root)),
                        "img_w": int(image_width),
                        "img_h": int(image_height),
                        "objects": len(lines),
                        "person_only": True,
                        "lighting": "night",
                        "nonschema_boxes": [],
                        "negative_worthy": False,
                    },
                    sort_keys=True,
                )
            )
            if not lines:
                counters.bump("image.zero_objects")

        done.add(key)

    if len(scene_keys) < 2:
        log.warn(
            "filename prefix did not partition the corpus - 06_split.py will fall back to "
            "perceptual-hash pseudo-sequences (OQ-3)",
            distinct_prefixes=len(scene_keys),
        )
        counters.bump("scene_key.fallback_required")
    else:
        log.info("scene keys recovered from filenames", distinct=len(scene_keys))

    if unknown_classes:
        log.error("native classes not present in the class map", classes=",".join(sorted(unknown_classes)))

    if not args.dry_run:
        write_lines(out_index, merge_jsonl(out_index, records))
        state["done"] = sorted(done)
        save_state(processed, SCRIPT, state)
        write_json(
            processed / "reports" / "convert_llvip.json",
            {"written": written, "scene_keys": len(scene_keys), "counters": counters.as_dict()},
        )

    counters.report(log)
    log.info("done", labels_written=written, scene_keys=len(scene_keys))
    return 1 if unknown_classes else 0


if __name__ == "__main__":
    run(main)
