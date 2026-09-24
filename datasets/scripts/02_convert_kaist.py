#!/usr/bin/env python3
"""KAIST Multispectral Pedestrian -> unified YOLO labels, visible/LWIR pairing preserved.

DISABLED in sources.yaml: KAIST is replaced by Teledyne FLIR ADAS v2 (02_convert_flir.py),
because no reachable copy of the KAIST annotations exists. This script is kept runnable.

Collision safety. KAIST frame names (I00000.jpg ...) restart in every set/video, so a lookup
keyed by the bare stem pairs a frame with another video's annotation, and an output named by
the bare stem overwrites another video's frame. Annotations are therefore indexed by
set/video/stem, and every frame is staged under set_video_stem (a symlink, no copy).

Four native tags, all handled (DATASET_SPEC.md section 1.3):
    person   -> class 0
    cyclist  -> class 0   (the KAIST box encloses the human on the bicycle)
    people   -> ignore region: zero-filled in the image, no label, frame NOT usable
                as a negative
    person?  -> ignore region: if it is the only annotation the frame is dropped

The visible and LWIR frames of one capture keep a shared pair_id so 04_ir_to_3ch.py
and 06_split.py can never separate them.

splits.yaml declares each set's lighting. This script VERIFIES that declaration from
the frames themselves (OQ-1) and fails if a set contradicts it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    IMAGE_SUFFIXES,
    Counters,
    Logger,
    base_parser,
    class_ids,
    format_label_line,
    imread_gray,
    imwrite,
    load_splits_config,
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

SCRIPT = "02_convert_kaist"

KAIST_MAP: dict[str, str | None] = {
    "person": "person",
    "cyclist": "person",
    "people": "__ignore__",
    "person?": "__ignore__",
    "person?a": "__ignore__",
}

SET_RE = re.compile(r"(set\d{2})")
VID_RE = re.compile(r"(V\d{3})")


def sequence_key(path: Path) -> str | None:
    text = path.as_posix()
    set_match = SET_RE.search(text)
    vid_match = VID_RE.search(text)
    if not set_match:
        return None
    video = vid_match.group(1) if vid_match else "V000"
    return f"kaist/{set_match.group(1)}/{video}"


def modality_of(path: Path) -> str | None:
    text = path.as_posix().lower()
    if "lwir" in text or "thermal" in text:
        return "lwir"
    if "visible" in text or "rgb" in text:
        return "visible"
    return None


def parse_kaist_annotation(path: Path, counters: Counters, log: Logger):
    """KAIST text annotations: '<label> x y w h occ xf yf wf hf ang'. Header lines start with %."""
    rows = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.error("unreadable annotation", path=str(path), detail=str(exc)[:120])
        counters.bump("annotation.unreadable")
        return None
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("%") or line.lower().startswith("class"):
            continue
        parts = line.split()
        if len(parts) < 5:
            counters.bump("annotation.short_row")
            log.warn("short annotation row", path=path.name, line=number)
            continue
        label = parts[0].strip().lower()
        try:
            x, y, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            counters.bump("annotation.bad_coords")
            log.warn("non-numeric box", path=path.name, line=number)
            continue
        rows.append((label, x, y, w, h))
    return rows


def frame_key(path: Path) -> str | None:
    """set/video/stem: the only key under which a KAIST frame name is unique."""
    sequence = sequence_key(path)
    if sequence is None:
        return None
    return f"{sequence.split('/', 1)[1]}/{path.stem}"


def find_annotation(image: Path, ann_index: dict[str, Path]) -> Path | None:
    key = frame_key(image)
    return ann_index.get(key) if key else None


def build_annotation_index(root: Path, counters: Counters | None = None) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*.txt"):
        if path.name.lower() in ("readme.txt", "license.txt"):
            continue
        key = frame_key(path)
        if key is None:
            if counters is not None:
                counters.bump("annotation.no_set_video_in_path")
            continue
        if key in index and counters is not None:
            counters.bump("annotation.duplicate_key")
        index.setdefault(key, path)
    return index


def verify_lighting(measured: dict[str, list[float]], declared: dict, log: Logger, counters: Counters) -> bool:
    """OQ-1: a set's declared lighting must match what its visible frames actually look like."""
    ok = True
    for set_name, values in sorted(measured.items()):
        if not values:
            continue
        mean = sum(values) / len(values)
        observed = "day" if mean >= 70.0 else "night"
        declared_lighting = declared.get(set_name, {}).get("lighting")
        if declared_lighting is None:
            log.warn("set not declared in splits.yaml", set=set_name, observed=observed)
            counters.bump("lighting.undeclared")
            continue
        if observed != declared_lighting:
            log.error(
                "declared lighting contradicts the frames",
                set=set_name,
                declared=declared_lighting,
                observed=observed,
                mean_intensity=f"{mean:.1f}",
            )
            counters.bump("lighting.mismatch")
            ok = False
        else:
            log.info("lighting verified", set=set_name, lighting=declared_lighting, mean_intensity=f"{mean:.1f}")
    return ok


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--source", default=None, help="KAIST root (default: the 00_fetch record, then <raw>/kaist)")
    ap.add_argument(
        "--lighting-sample",
        type=int,
        default=40,
        help="visible frames per set sampled to verify the declared lighting",
    )
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    root = resolve_source_root("kaist", args, log)
    if root is None:
        log.error("KAIST root not found; the source is disabled in sources.yaml (see disabled_reason)")
        return 1
    processed = Path(args.processed).resolve()
    out_labels = processed / "labels" / "kaist"
    out_masked = processed / "masked" / "kaist"
    out_index = processed / "index" / "kaist.jsonl"

    ids = class_ids()
    declared = load_splits_config()["kaist"]["sets"]
    state = load_state(processed, SCRIPT)
    done = set(state.get("done", [])) if not args.force else set()

    log.info("indexing annotations", root=str(root))
    ann_index = build_annotation_index(root, counters)
    log.info("annotation index built", annotations=len(ann_index))
    if not ann_index:
        log.error("no annotation files found", root=str(root))
        return 1

    images = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if args.limit:
        images = images[: args.limit]
    log.info("images found", images=len(images))

    require_dirs(out_labels, out_index.parent)

    records: list[str] = []
    lighting_samples: dict[str, list[float]] = {}
    unknown_tags: set[str] = set()
    written = 0

    for image in images:
        key = image.as_posix()
        if key in done:
            counters.bump("image.already_done")
            continue

        sequence = sequence_key(image)
        modality = modality_of(image)
        if sequence is None or modality is None:
            counters.bump("image.unrecognised_path")
            log.warn("path does not carry a set/modality - skipped and counted", image=key)
            continue
        set_name = sequence.split("/")[1]
        uid = f"{set_name}_{sequence.split('/')[2]}_{image.stem}"

        annotation = find_annotation(image, ann_index)
        if annotation is None:
            counters.bump("image.no_annotation")
            continue

        rows = parse_kaist_annotation(annotation, counters, log)
        if rows is None:
            continue

        gray = imread_gray(image)
        if gray is None:
            counters.bump("image.unreadable")
            log.warn("unreadable image", image=key)
            continue
        height, width = gray.shape[:2]

        if modality == "visible" and len(lighting_samples.get(set_name, [])) < args.lighting_sample:
            lighting_samples.setdefault(set_name, []).append(float(gray.mean()))

        lines: list[str] = []
        ignore_boxes: list[tuple[float, float, float, float]] = []
        uncertain_only = bool(rows)

        for label, x, y, w, h in rows:
            if label not in KAIST_MAP:
                unknown_tags.add(label)
                counters.bump(f"native.UNKNOWN.{label}")
                continue
            target = KAIST_MAP[label]
            if target == "__ignore__":
                ignore_boxes.append((x, y, x + w, y + h))
                counters.bump(f"ignore.{label}")
                continue
            geometry = xyxy_to_yolo(x, y, x + w, y + h, width, height)
            if geometry is None:
                counters.bump("box.degenerate_dropped")
                continue
            lines.append(format_label_line(ids[target], *geometry))
            counters.bump(f"mapped.{target}")
            uncertain_only = False

        if uncertain_only and ignore_boxes:
            counters.bump("image.dropped_uncertain_only")
            done.add(key)
            continue

        image_out = image
        if ignore_boxes:
            masked = gray.copy()
            for x1, y1, x2, y2 in ignore_boxes:
                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(width, int(x2)), min(height, int(y2))
                if xi2 > xi1 and yi2 > yi1:
                    masked[yi1:yi2, xi1:xi2] = 0
            image_out = out_masked / modality / f"{uid}.png"
            if not args.dry_run and not imwrite(image_out, masked):
                log.error("could not write masked frame", image=key)
                counters.bump("image.mask_write_failed")
                continue
            counters.bump("image.ignore_regions_filled")

        if image_out == image:
            image_out = processed / "stage" / "kaist" / modality / f"{uid}{image.suffix.lower()}"
            if not stage_link(image, image_out, counters, args.dry_run):
                log.error("could not stage frame", image=key)
                continue
        label_path = out_labels / modality / f"{uid}.txt"
        if not args.dry_run:
            write_lines(label_path, lines)
        written += 1

        records.append(
            json.dumps(
                {
                    "image": image_out.as_posix(),
                    "label": label_path.as_posix(),
                    "source": "kaist",
                    "modality": modality,
                    "sequence_key": sequence,
                    "pair_id": f"{sequence}/{image.stem}",
                    "lighting": declared.get(set_name, {}).get("lighting", "unknown"),
                    "set": set_name,
                    "img_w": int(width),
                    "img_h": int(height),
                    "objects": len(lines),
                    "negative_worthy": bool(not lines and not ignore_boxes),
                },
                sort_keys=True,
            )
        )
        if not lines:
            counters.bump("image.zero_objects")
        done.add(key)

    lighting_ok = verify_lighting(lighting_samples, declared, log, counters)

    if unknown_tags:
        log.error("native tags not present in the class map", tags=",".join(sorted(unknown_tags)))

    if not args.dry_run:
        write_lines(out_index, merge_jsonl(out_index, records))
        state["done"] = sorted(done)
        save_state(processed, SCRIPT, state)
        write_json(
            processed / "reports" / "convert_kaist.json",
            {
                "written": written,
                "lighting_verified": lighting_ok,
                "counters": counters.as_dict(),
            },
        )

    counters.report(log)
    log.info("done", labels_written=written, lighting_verified=lighting_ok)
    if unknown_tags or not lighting_ok:
        return 1
    return 0


if __name__ == "__main__":
    run(main)
