#!/usr/bin/env python3
"""SAHI-style tiling of the far-field strip. DATASET_SPEC.md section 3.3.

The arithmetic this implements, from MEASUREMENTS.md section 3:

  A 1920x1080 frame fed whole to a 640 px detector is downscaled 3.0x. A person who
  occupies 27 px in the native frame arrives at the detector as 9 px - the measured
  49% recall band. Tiling that strip into 640x640 tiles AT NATIVE RESOLUTION removes
  the downscale: the same person arrives as 27 px, the 100% recall band. A 19 px
  person arrives as 19 px (90%) instead of 6 px, which is below anything measured.

Tiles are produced for TRAINING as well as inference. Training untiled and inferring
tiled is a train/test mismatch and this pipeline refuses it.

A box sliced by a tile edge is kept only when at least 30% of its original area
survives, otherwise the model learns to fire on body fragments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    Counters,
    Logger,
    base_parser,
    format_label_line,
    imread_any,
    imwrite,
    load_state,
    merge_jsonl,
    parse_label_file,
    require_dirs,
    run,
    save_state,
    write_json,
    write_lines,
)

SCRIPT = "08_tile_farfield"

TILE = 640
OVERLAP = 0.20
STRIP_TOP = 0.0
STRIP_BOTTOM = 0.40  # upper 40% of frame height is the far field of a pole-mounted camera
MIN_BOX_AREA_RETAINED = 0.30


def tile_origins(extent: int, tile: int, overlap: float) -> list[int]:
    """Origins covering [0, extent) with the given overlap, last tile flush to the edge."""
    if extent <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    origins = list(range(0, max(1, extent - tile + 1), step))
    if origins[-1] != extent - tile:
        origins.append(extent - tile)
    return origins


def clip_box_to_tile(
    cx: float, cy: float, w: float, h: float, width: int, height: int, ox: int, oy: int, tile: int
):
    """Absolute-space clip of one normalised box into one tile. Returns tile-normalised geometry."""
    x1 = (cx - w / 2.0) * width
    y1 = (cy - h / 2.0) * height
    x2 = (cx + w / 2.0) * width
    y2 = (cy + h / 2.0) * height
    original_area = max(1e-9, (x2 - x1) * (y2 - y1))

    cx1, cy1 = max(x1, ox), max(y1, oy)
    cx2, cy2 = min(x2, ox + tile), min(y2, oy + tile)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    if ((cx2 - cx1) * (cy2 - cy1)) / original_area < MIN_BOX_AREA_RETAINED:
        return None

    return (
        ((cx1 + cx2) / 2.0 - ox) / tile,
        ((cy1 + cy2) / 2.0 - oy) / tile,
        (cx2 - cx1) / tile,
        (cy2 - cy1) / tile,
    )


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--tile", type=int, default=TILE, help="tile edge in pixels")
    ap.add_argument("--overlap", type=float, default=OVERLAP, help="tile overlap ratio")
    ap.add_argument("--strip-top", type=float, default=STRIP_TOP, help="far-field strip top, fraction of height")
    ap.add_argument(
        "--strip-bottom", type=float, default=STRIP_BOTTOM, help="far-field strip bottom, fraction of height"
    )
    ap.add_argument(
        "--only-with-objects",
        action="store_true",
        default=True,
        help="tile only frames whose strip actually contains an object",
    )
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    processed = Path(args.processed).resolve()
    index_path = processed / "index" / "with_negatives.jsonl"
    if not index_path.exists():
        index_path = processed / "index" / "split.jsonl"
    if not index_path.exists():
        log.error("input index missing - run 06_split.py and 07_negatives.py first")
        return 1

    out_images = processed / "tiles" / "images"
    out_labels = processed / "tiles" / "labels"
    require_dirs(out_images, out_labels)

    state = load_state(processed, SCRIPT)
    done = set(state.get("done", [])) if not args.force else set()

    records: list[dict] = []
    with index_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["image"])

    train = [r for r in records if r.get("split") == "train" and not r.get("is_negative")]
    if args.limit:
        train = train[: args.limit]
    log.info("candidate frames", frames=len(train), tile=args.tile, overlap=args.overlap)

    tile_records: list[str] = []
    tiles_written = 0

    for record in train:
        image_path = Path(record["image"])
        key = image_path.as_posix()
        if key in done:
            counters.bump("frame.already_done")
            continue

        width = int(record.get("img_w") or 0)
        height = int(record.get("img_h") or 0)
        if width <= 0 or height <= 0:
            probe = imread_any(image_path)
            if probe is None:
                counters.bump("frame.unreadable")
                log.warn("unreadable frame", image=key)
                continue
            height, width = probe.shape[:2]

        if width < args.tile or height < args.tile:
            counters.bump("frame.smaller_than_tile")
            continue

        strip_y1 = int(height * args.strip_top)
        strip_y2 = int(height * args.strip_bottom)
        if strip_y2 - strip_y1 < args.tile:
            # Strip thinner than a tile: anchor one tile row at the strip top.
            strip_y2 = min(height, strip_y1 + args.tile)
        if strip_y2 - strip_y1 < args.tile:
            counters.bump("frame.strip_too_thin")
            continue

        rows, errors = parse_label_file(Path(record["label"]))
        for message in errors:
            log.warn("label row rejected", label=record["label"], detail=message)
            counters.bump("label.row_rejected")

        origins_x = tile_origins(width, args.tile, args.overlap)
        origins_y = tile_origins(strip_y2 - strip_y1, args.tile, args.overlap)

        image = None
        for oy_rel in origins_y:
            oy = strip_y1 + oy_rel
            for ox in origins_x:
                tile_rows: list[str] = []
                for class_id, cx, cy, w, h in rows:
                    clipped = clip_box_to_tile(cx, cy, w, h, width, height, ox, oy, args.tile)
                    if clipped is None:
                        continue
                    tile_rows.append(format_label_line(class_id, *clipped))

                if args.only_with_objects and not tile_rows:
                    counters.bump("tile.empty_skipped")
                    continue

                # source and modality are part of the name because LLVIP's visible and
                # infrared frames share a stem, and a bare stem silently overwrites the pair.
                tile_name = (
                    f"{record.get('source', 'x')}_{image_path.stem}"
                    f"_{record.get('modality', 'v')}_t{ox}_{oy}"
                )
                tile_image = out_images / f"{tile_name}.jpg"
                tile_label = out_labels / f"{tile_name}.txt"

                if args.dry_run:
                    counters.bump("tile.would_write")
                    continue

                if image is None:
                    image = imread_any(image_path)
                    if image is None:
                        counters.bump("frame.unreadable")
                        log.warn("unreadable frame at crop time", image=key)
                        break

                crop = image[oy : oy + args.tile, ox : ox + args.tile]
                if crop.shape[0] != args.tile or crop.shape[1] != args.tile:
                    counters.bump("tile.wrong_shape_skipped")
                    continue
                if not imwrite(tile_image, crop):
                    log.error("tile write failed", tile=tile_image.as_posix())
                    counters.bump("tile.write_failed")
                    continue
                write_lines(tile_label, tile_rows)
                tiles_written += 1
                counters.bump("tile.written")

                tile_records.append(
                    json.dumps(
                        {
                            "image": tile_image.as_posix(),
                            "label": tile_label.as_posix(),
                            "source": record.get("source"),
                            "modality": record.get("modality"),
                            "sequence_key": record.get("sequence_key"),
                            "split": "train",
                            "objects": len(tile_rows),
                            "img_w": args.tile,
                            "img_h": args.tile,
                            "tiled": True,
                            "parent": key,
                        },
                        sort_keys=True,
                    )
                )
        done.add(key)

    if not args.dry_run:
        tiles_index = processed / "index" / "tiles.jsonl"
        write_lines(tiles_index, merge_jsonl(tiles_index, tile_records))
        state["done"] = sorted(done)
        save_state(processed, SCRIPT, state)
        write_json(
            processed / "reports" / "tiles.json",
            {
                "tiles": tiles_written,
                "tile": args.tile,
                "overlap": args.overlap,
                "strip": [args.strip_top, args.strip_bottom],
                "counters": counters.as_dict(),
            },
        )

    counters.report(log)
    log.info("done", tiles=tiles_written, gate="G13 requires >= 2000 tiles at exactly 640x640")
    return 0


if __name__ == "__main__":
    run(main)
