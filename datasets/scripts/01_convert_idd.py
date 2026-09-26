#!/usr/bin/env python3
"""IDD Detection -> unified YOLO labels.

All 15 IDD detection classes are handled explicitly (DATASET_SPEC.md section 1.2).
Nothing falls through to a default: a native class this script has never seen is a
FAILURE, not a silent drop, because an unmapped class is exactly how a taxonomy
quietly rots.

Layout (verified 2026-09 on the Kaggle mirror vinayak21574/idd-detection, which keeps the
original IDD_Detection tree):

    Annotations/<subset>/<drive>/<frame>.xml     subset = frontFar, frontNear, highquality_16k,
    JPEGImages/<subset>/<drive>/<frame>.jpg               rearNear, sideLeft, sideRight
    train.txt, val.txt, test.txt                 one <subset>/<drive>/<frame> per line

The official split comes from those lists, never from the path. Frame names repeat across
drives (2,694 stems occur in more than one drive), so every record is keyed by its full
<subset>/<drive>/<frame> path and its image is staged under a collision-proof name
(<subset>__<drive>__<frame>) as a symlink: no pixel is copied and the source root may be
read-only. Anything under Annotations/ that is not an .xml file (the mirror carries editor
swap files such as .001542_r.xml.swp) is counted and ignored.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    CART_SHEET_FIELDS,
    Counters,
    Logger,
    base_parser,
    cart_sheet_path,
    class_ids,
    format_label_line,
    imread_any,
    load_state,
    merge_jsonl,
    read_cart_sheet,
    require_dirs,
    resolve_source_root,
    run,
    save_state,
    stage_link,
    write_json,
    write_lines,
    xyxy_to_yolo,
)

SCRIPT = "01_convert_idd"

# DATASET_SPEC.md section 1.2. Every one of the 15 native classes appears here.
# None -> dropped from the label set (the image may still serve as a negative).
IDD_MAP: dict[str, str | None] = {
    "car": "car",
    "bus": "truck",
    "truck": "truck",
    "motorcycle": "two_wheeler",
    "bicycle": "two_wheeler",
    "autorickshaw": "two_wheeler",
    "person": "person",
    "rider": "person",
    "animal": None,
    "traffic light": None,
    "traffic sign": None,
    "caravan": "truck",
    "trailer": "truck",
    "train": None,
    "vehicle fallback": "__cart_review__",
}

# Dropped classes that are still OBJECTS in the frame. Their boxes are kept on the record as
# `nonschema_boxes` so a far-field negative tile never contains one, and a frame holding one
# of the vehicle-like ones is never a negative (an unlabelled vehicle is not an empty scene).
UNLABELLED_OBJECTS = {"animal", "train", "vehicle fallback"}
BLOCKS_NEGATIVE = {"train", "vehicle fallback"}


def normalise_native(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def read_split_lists(root: Path, counters: Counters) -> dict[str, str]:
    """<subset>/<drive>/<frame> -> official split, from train.txt / val.txt / test.txt."""
    listed: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.txt"
        if not path.exists():
            counters.bump(f"list.missing.{split}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            key = line.strip().strip("/")
            if not key:
                continue
            if key in listed and listed[key] != split:
                counters.bump("list.entry_in_two_lists")
            listed[key] = split
            counters.bump(f"list.{split}")
    return listed


def scan_annotations(root: Path, counters: Counters, log: Logger) -> list[str]:
    """Every .xml under Annotations/, as <subset>/<drive>/<frame>. Non-xml files are counted."""
    ann_root = root / "Annotations"
    found: list[str] = []
    for directory, _dirs, files in os.walk(ann_root):
        for name in files:
            if name.lower().endswith(".xml") and not name.startswith("."):
                rel = (Path(directory) / name).relative_to(ann_root).with_suffix("")
                found.append(rel.as_posix())
            else:
                counters.bump("annotation_dir.non_xml_ignored")
                if counters.get("annotation_dir.non_xml_ignored") <= 5:
                    log.warn("non-annotation file under Annotations/ ignored", file=str(Path(directory) / name))
    return sorted(found)


def find_image(root: Path, key: str) -> Path | None:
    base = root / "JPEGImages" / key
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = Path(str(base) + suffix)
        if candidate.exists():
            return candidate
    return None


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
        native = normalise_native(obj.findtext("name") or "")
        box = obj.find("bndbox")
        if box is None:
            counters.bump("object.no_bndbox")
            continue
        try:
            coords = tuple(float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            counters.bump("object.bad_coords")
            continue
        objects.append((native, coords))
    return width, height, objects


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--source", default=None, help="IDD root, or any directory above it")
    ap.add_argument(
        "--subset-ok",
        action="store_true",
        help="the root is a deliberately partial fixture: listed frames with no file are counted, not fatal",
    )
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    root = resolve_source_root("idd", args, log)
    if root is None:
        log.error("IDD root not found - run 00_fetch.py or pass --source")
        return 1
    processed = Path(args.processed).resolve()
    out_labels = processed / "labels" / "idd"
    out_stage = processed / "stage" / "idd"
    out_index = processed / "index" / "idd.jsonl"
    sheet = cart_sheet_path(processed)

    ids = class_ids()
    listed = read_split_lists(root, counters)
    annotations = scan_annotations(root, counters, log)
    log.info("layout", layout="IDD_Detection lists + Annotations/JPEGImages tree", listed=len(listed), xml=len(annotations))
    if not annotations:
        log.error("no annotation files under Annotations/", root=str(root))
        return 1

    present = set(annotations)
    listed_trainval = sorted(k for k, s in listed.items() if s in ("train", "val"))
    missing_xml = [k for k in listed_trainval if k not in present]
    if missing_xml:
        counters.bump("list.trainval_without_xml", len(missing_xml))
        (log.warn if args.subset_ok else log.error)(
            "listed train/val frames have no annotation file", count=len(missing_xml), example=missing_xml[0]
        )

    if args.limit:
        annotations = annotations[: args.limit]

    state = load_state(processed, SCRIPT)
    done = set(state.get("done", [])) if not args.force else set()
    require_dirs(out_labels, out_index.parent, sheet.parent)

    # The review sheet is a human's work. Verdicts already written are carried over, keyed by
    # frame and box, so a re-run can never erase a completed review.
    previous_verdicts = {
        (row.get("uid"), row.get("xmin"), row.get("ymin"), row.get("xmax"), row.get("ymax")): row.get("verdict", "")
        for row in read_cart_sheet(sheet)
    }
    cart_rows: list[dict] = []
    records: list[str] = []
    unknown_classes: set[str] = set()
    missing_images = 0
    written = 0

    for key in annotations:
        official = listed.get(key)
        if official is None:
            counters.bump("annotation.unlisted_skipped")
            continue
        if official == "test":
            counters.bump("annotation.official_test_unused")
            continue

        uid = key.replace("/", "__")
        parsed = parse_voc(root / "Annotations" / f"{key}.xml", counters, log)
        if parsed is None:
            continue
        width, height, objects = parsed

        image = find_image(root, key)
        if image is None:
            missing_images += 1
            counters.bump("annotation.no_image")
            if missing_images <= 5:
                log.warn("annotation has no image", annotation=key)
            continue

        if width <= 0 or height <= 0:
            probe = imread_any(image)
            if probe is None:
                log.warn("unreadable image", image=str(image))
                counters.bump("image.unreadable")
                continue
            height, width = probe.shape[:2]
            counters.bump("image.size_recovered")

        lines: list[str] = []
        natives: set[str] = set()
        nonschema: list[list[float]] = []
        for native, coords in objects:
            natives.add(native)
            if native not in IDD_MAP:
                unknown_classes.add(native)
                counters.bump(f"native.UNKNOWN.{native}")
                continue
            if native in UNLABELLED_OBJECTS:
                nonschema.append([round(coords[0] / width, 5), round(coords[1] / height, 5),
                                  round(coords[2] / width, 5), round(coords[3] / height, 5)])
            target = IDD_MAP[native]
            if target is None:
                counters.bump(f"dropped.{native}")
                continue
            if target == "__cart_review__":
                counters.bump("cart_review.queued")
                row = {"uid": uid, "image": key, "xmin": f"{coords[0]:.1f}", "ymin": f"{coords[1]:.1f}",
                       "xmax": f"{coords[2]:.1f}", "ymax": f"{coords[3]:.1f}"}
                row["verdict"] = previous_verdicts.get(
                    (uid, row["xmin"], row["ymin"], row["xmax"], row["ymax"]), ""
                ) or "UNVERIFIED"
                cart_rows.append(row)
                continue
            geometry = xyxy_to_yolo(*coords, width, height)
            if geometry is None:
                counters.bump("box.degenerate_dropped")
                continue
            lines.append(format_label_line(ids[target], *geometry))
            counters.bump(f"mapped.{target}")

        if key in done and not args.force:
            counters.bump("annotation.already_done")
            continue

        staged = out_stage / f"{uid}{image.suffix.lower()}"
        if not stage_link(image, staged, counters, args.dry_run):
            log.error("could not stage image", image=str(image))
            continue
        label_path = out_labels / f"{uid}.txt"
        if not args.dry_run:
            write_lines(label_path, lines)
        written += 1

        empty = not lines
        negative_worthy = empty and not (natives & BLOCKS_NEGATIVE)
        records.append(
            json.dumps(
                {
                    "image": staged.as_posix(),
                    "source_image": image.as_posix(),
                    "label": label_path.as_posix(),
                    "source": "idd",
                    "modality": "visible",
                    "lighting": "day",
                    "sequence_key": "idd/" + "/".join(key.split("/")[:2]),
                    "official_split": official,
                    "img_w": int(width),
                    "img_h": int(height),
                    "objects": len(lines),
                    "negative_worthy": bool(negative_worthy),
                    "animal_only": bool(negative_worthy and "animal" in natives),
                    "nonschema_boxes": nonschema,
                },
                sort_keys=True,
            )
        )
        if empty:
            counters.bump("image.zero_objects")
        done.add(key)

    if unknown_classes:
        log.error("native classes not present in the class map - refusing to guess",
                  classes=",".join(sorted(unknown_classes)))

    if not args.dry_run:
        write_lines(out_index, merge_jsonl(out_index, records))
        # Rows for frames outside this run (a --limit run) are preserved, never dropped.
        seen = {(r["uid"], r["xmin"], r["ymin"], r["xmax"], r["ymax"]) for r in cart_rows}
        for row in read_cart_sheet(sheet):
            k = (row.get("uid"), row.get("xmin"), row.get("ymin"), row.get("xmax"), row.get("ymax"))
            if k not in seen:
                cart_rows.append({f: row.get(f, "") for f in CART_SHEET_FIELDS})
        cart_rows.sort(key=lambda r: (r["uid"], float(r["xmin"]), float(r["ymin"])))
        tmp = sheet.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CART_SHEET_FIELDS)
            writer.writeheader()
            writer.writerows(cart_rows)
        tmp.replace(sheet)
        state["done"] = sorted(done)
        save_state(processed, SCRIPT, state)
        write_json(processed / "reports" / "convert_idd.json", {"written": written, "counters": counters.as_dict()})

    counters.report(log)
    log.info("done", labels_written=written, cart_candidates=len(cart_rows), review_sheet=str(sheet))
    log.info(
        "cart review",
        sheet=str(sheet),
        rule="mark each row cart or not_cart; class 4 ships only at >= 300 verified (DATASET_SPEC.md 1.5)",
    )
    if unknown_classes:
        return 1
    if (missing_xml or missing_images) and not args.subset_ok:
        log.error("listed frames are missing from the source root", missing_xml=len(missing_xml), missing_images=missing_images)
        return 1
    return 0


if __name__ == "__main__":
    run(main)
