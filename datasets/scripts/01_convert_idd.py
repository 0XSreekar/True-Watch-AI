#!/usr/bin/env python3
"""IDD Detection -> unified YOLO labels.

All 15 IDD detection classes are handled explicitly (DATASET_SPEC.md section 1.2).
Nothing falls through to a default: a native class this script has never seen is a
FAILURE, not a silent drop, because an unmapped class is exactly how a taxonomy
quietly rots.

IDD's on-disk layout is not assumed (OQ-2). The parser detects it at runtime and
logs which layout it found.
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
    load_state,
    merge_jsonl,
    require_dirs,
    run,
    save_state,
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

# Frames whose only content is one of these become hard negatives (section 5).
NEGATIVE_WORTHY = {"animal"}


def normalise_native(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def detect_layout(root: Path, log: Logger) -> tuple[str, list[Path]]:
    """Find the annotation files without assuming IDD's directory shape (OQ-2)."""
    xmls = list(root.rglob("*.xml"))
    if xmls:
        parallel = any("Annotations" in p.parts or "annotations" in p.parts for p in xmls[:50])
        layout = "voc_xml_parallel_tree" if parallel else "voc_xml_flat"
        log.info("layout detected", layout=layout, annotations=len(xmls))
        return layout, xmls
    jsons = [p for p in root.rglob("*.json") if p.name not in ("fetch_manifest.json",)]
    if jsons:
        log.info("layout detected", layout="json_index", annotations=len(jsons))
        return "json_index", jsons
    log.error("no annotation files found under the source root", root=str(root))
    return "unknown", []


def find_image_for(annotation: Path, root: Path, index: dict[str, Path]) -> Path | None:
    stem = annotation.stem
    direct = index.get(stem)
    if direct is not None:
        return direct
    for suffix in IMAGE_SUFFIXES:
        candidate = annotation.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def build_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem, path)
    return index


def sequence_key(image: Path, root: Path) -> str:
    """splits.yaml idd.sequence_key = drive_folder. The drive is the parent directory."""
    try:
        relative = image.relative_to(root)
    except ValueError:
        return f"idd/{image.parent.name}"
    parts = relative.parts
    drive = parts[-2] if len(parts) >= 2 else "root"
    return f"idd/{drive}"


def official_split_of(image: Path, root: Path) -> str:
    parts = {p.lower() for p in image.relative_to(root).parts} if root in image.parents else set()
    for candidate in ("train", "val", "test"):
        if candidate in parts:
            return candidate
    return "train"


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
            coords = (
                float(box.findtext("xmin")),
                float(box.findtext("ymin")),
                float(box.findtext("xmax")),
                float(box.findtext("ymax")),
            )
        except (TypeError, ValueError):
            counters.bump("object.bad_coords")
            continue
        objects.append((native, coords))
    return width, height, objects


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--source", default=None, help="IDD root (default: <raw>/idd)")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    root = Path(args.source).resolve() if args.source else Path(args.raw).resolve() / "idd"
    processed = Path(args.processed).resolve()
    out_labels = processed / "labels" / "idd"
    out_index = processed / "index" / "idd.jsonl"
    cart_review = processed / "cart_review" / "vehicle_fallback.csv"

    if not root.exists():
        log.error("source root does not exist - run 00_fetch.py first", root=str(root))
        return 1

    ids = class_ids()
    layout, annotations = detect_layout(root, log)
    if layout == "unknown" or not annotations:
        return 1
    if layout == "json_index":
        log.error(
            "this build supports IDD's VOC XML annotations; a JSON index was found instead",
            hint="report the layout so the parser can be extended rather than guessed at",
        )
        return 1

    if args.limit:
        annotations = annotations[: args.limit]

    state = load_state(processed, SCRIPT)
    done = set(state.get("done", [])) if not args.force else set()

    log.info("indexing images", root=str(root))
    index = build_image_index(root)
    log.info("image index built", images=len(index))

    require_dirs(out_labels, out_index.parent, cart_review.parent)

    records: list[str] = []
    cart_rows: list[str] = ["image,xmin,ymin,xmax,ymax,verdict"]
    unknown_classes: set[str] = set()
    written = 0

    for annotation in annotations:
        key = str(annotation.relative_to(root))
        if key in done:
            counters.bump("annotation.already_done")
            continue

        parsed = parse_voc(annotation, counters, log)
        if parsed is None:
            continue
        width, height, objects = parsed

        image = find_image_for(annotation, root, index)
        if image is None:
            log.warn("annotation has no image", annotation=key)
            counters.bump("annotation.no_image")
            continue

        if width <= 0 or height <= 0:
            from _lib import imread_any

            probe = imread_any(image)
            if probe is None:
                log.warn("unreadable image", image=str(image))
                counters.bump("image.unreadable")
                continue
            height, width = probe.shape[:2]
            counters.bump("image.size_recovered")

        lines: list[str] = []
        native_only: set[str] = set()
        for native, coords in objects:
            native_only.add(native)
            if native not in IDD_MAP:
                unknown_classes.add(native)
                counters.bump(f"native.UNKNOWN.{native}")
                continue
            target = IDD_MAP[native]
            if target is None:
                counters.bump(f"dropped.{native}")
                continue
            if target == "__cart_review__":
                counters.bump("cart_review.queued")
                cart_rows.append(
                    f"{image.relative_to(root)},{coords[0]:.1f},{coords[1]:.1f},"
                    f"{coords[2]:.1f},{coords[3]:.1f},UNVERIFIED"
                )
                continue
            geometry = xyxy_to_yolo(*coords, width, height)
            if geometry is None:
                counters.bump("box.degenerate_dropped")
                continue
            lines.append(format_label_line(ids[target], *geometry))
            counters.bump(f"mapped.{target}")

        label_path = out_labels / (annotation.stem + ".txt")
        if not args.dry_run:
            write_lines(label_path, lines)
        written += 1

        negative_worthy = bool(native_only) and native_only.issubset(NEGATIVE_WORTHY)
        records.append(
            json.dumps(
                {
                    "image": image.as_posix(),
                    "label": label_path.as_posix(),
                    "source": "idd",
                    "modality": "visible",
                    "sequence_key": sequence_key(image, root),
                    "official_split": official_split_of(image, root),
                    "img_w": int(width),
                    "img_h": int(height),
                    "objects": len(lines),
                    "negative_worthy": bool(negative_worthy or not native_only),
                },
                sort_keys=True,
            )
        )
        if not lines:
            counters.bump("image.zero_objects")
        done.add(key)

    if unknown_classes:
        log.error(
            "native classes not present in the class map - refusing to guess",
            classes=",".join(sorted(unknown_classes)),
        )

    if not args.dry_run:
        write_lines(out_index, merge_jsonl(out_index, records))
        write_lines(cart_review, cart_rows)
        state["done"] = sorted(done)
        state["layout"] = layout
        save_state(processed, SCRIPT, state)
        write_json(
            processed / "reports" / "convert_idd.json",
            {"layout": layout, "written": written, "counters": counters.as_dict()},
        )

    counters.report(log)
    log.info(
        "done",
        layout=layout,
        labels_written=written,
        cart_candidates=counters.get("cart_review.queued"),
        review_sheet=str(cart_review),
    )
    log.info(
        "NEXT: hand-verify the cart candidates",
        sheet=str(cart_review),
        gate="class 4 ships only at >= 300 verified instances (DATASET_SPEC.md section 1.5)",
    )
    return 1 if unknown_classes else 0


if __name__ == "__main__":
    run(main)
