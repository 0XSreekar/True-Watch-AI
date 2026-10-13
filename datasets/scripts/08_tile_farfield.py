#!/usr/bin/env python3
"""SAHI-style tiling of the far-field strip. DATASET_SPEC.md section 3.3.

The arithmetic this implements, from MEASUREMENTS.md section 3:

  A 1920x1080 frame fed whole to a 640 px detector is downscaled 3.0x. A person who
  occupies 27 px in the native frame arrives at the detector as 9 px - the measured
  49% recall band. Tiling that strip into 640x640 tiles AT NATIVE RESOLUTION removes
  the downscale: the same person arrives as 27 px, the 100% recall band.

Tiles are produced for TRAINING as well as inference. Training untiled and inferring
tiled is a train/test mismatch and this pipeline refuses it.

What is cut, from the FULL-resolution source frame (09 caps untiled frames at 1280 px long
side afterwards; tiles are never resized, which is the whole point of them):
  * object tiles: _lib.plan_object_tiles, the same deterministic plan 07 sized the negative
    pool against. A box sliced by a tile edge is kept only when at least 30% of its area
    survives. The budget (composition.tiles.max_tiles) is split by modality at lwir_share,
    and tiles holding a small object are taken first.
  * empty far-field tiles that 07 chose as negatives (index/negative_tiles.jsonl).
LWIR tiles are cut from the converted B = G = R frame (_lib.load_for_output). Every tile is
written as JPEG at the build quality, so 09 copies it through unchanged in geometry.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    Counters,
    Logger,
    base_parser,
    format_label_line,
    ir_conversion_params,
    load_composition,
    parallel_map,
    plan_object_tiles,
    read_jsonl,
    require_dirs,
    run,
    tile_worker,
    write_json,
    write_lines,
)

SCRIPT = "08_tile_farfield"
JPEG_QUALITY = 92


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--quality", type=int, default=JPEG_QUALITY, help="JPEG quality of written tiles")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    processed = Path(args.processed).resolve()
    index_path = processed / "index" / "with_negatives.jsonl"
    if not index_path.exists():
        log.error("input index missing - run 07_negatives.py first", expected=str(index_path))
        return 1
    tiles_cfg = load_composition().get("tiles", {})
    tile = int(tiles_cfg.get("size", 640))

    records = sorted(read_jsonl(index_path), key=lambda r: r["image"])
    by_image = {r["image"]: r for r in records}
    train = [r for r in records if r.get("split") == "train"]
    plan = plan_object_tiles(train, args.seed)
    negative_tiles = read_jsonl(processed / "index" / "negative_tiles.jsonl")
    log.info("tile plan", object_tiles=len(plan), negative_tiles=len(negative_tiles), tile=tile,
             overlap=tiles_cfg.get("overlap"), strip=[tiles_cfg.get("strip_top"), tiles_cfg.get("strip_bottom")],
             modality=dict(Counter(c["modality"] for c in plan)))
    if args.limit:
        plan = plan[: args.limit]

    out_images = processed / "tiles" / "images"
    out_labels = processed / "tiles" / "labels"
    require_dirs(out_images, out_labels)

    jobs_by_parent: dict[str, list[dict]] = defaultdict(list)
    tile_records: list[dict] = []

    def add(parent: dict, ox: int, oy: int, lines: list[str], negative: bool) -> None:
        # The parent's stem is unique within its source (01/02/03 guarantee it), and the
        # modality is part of the name because LLVIP's visible and infrared frames share a stem.
        name = f"{Path(parent['image']).stem}_{parent.get('modality', 'visible')}_t{ox}_{oy}"
        image = out_images / parent.get("source", "x") / f"{name}.jpg"
        label = out_labels / parent.get("source", "x") / f"{name}.txt"
        jobs_by_parent[parent["image"]].append({"ox": ox, "oy": oy, "image": image.as_posix(),
                                                "label": label.as_posix(), "lines": lines})
        tile_records.append({
            "image": image.as_posix(),
            "label": label.as_posix(),
            "source": parent.get("source"),
            "modality": parent.get("modality"),
            "lighting": parent.get("lighting"),
            "sequence_key": parent.get("sequence_key"),
            "split": "train",
            "objects": len(lines),
            "img_w": tile,
            "img_h": tile,
            "tiled": True,
            "is_negative": negative,
            "negative_category": "farfield_empty_tile" if negative else None,
            "parent": parent["image"],
        })

    for spec in plan:
        parent = by_image.get(spec["parent"])
        if parent is None:
            counters.bump("plan.parent_missing")
            continue
        add(parent, spec["ox"], spec["oy"], [format_label_line(int(c), *g) for c, *g in spec["rows"]], False)
    for spec in negative_tiles:
        parent = by_image.get(spec["parent"])
        if parent is None:
            counters.bump("negative_tile.parent_missing")
            continue
        add(parent, int(spec["ox"]), int(spec["oy"]), [], True)

    if args.dry_run:
        log.info("dry-run: would write tiles", tiles=len(tile_records), parents=len(jobs_by_parent))
        return 0

    ir_params = list(ir_conversion_params())
    jobs = []
    for parent_image, specs in sorted(jobs_by_parent.items()):
        todo = [s for s in specs if args.force or not Path(s["image"]).exists()]
        counters.bump("tile.already_written", len(specs) - len(todo))
        if todo:
            jobs.append({"record": by_image[parent_image], "tiles": todo, "tile": tile,
                         "quality": args.quality, "ir_params": ir_params})
    log.info("cutting", parents=len(jobs), workers=args.workers)

    failed_parents: set[str] = set()
    for job, result in zip(jobs, parallel_map(tile_worker, jobs, args.workers, 8)):
        counters.bump("tile.written", result.get("written", 0))
        if not result.get("ok"):
            failed_parents.add(job["record"]["image"])
            counters.bump(f"tile.failed.{result.get('why')}")
            log.warn("tile cut failed", parent=job["record"]["image"], why=result.get("why"))

    kept = [r for r in tile_records if r["parent"] not in failed_parents or Path(r["image"]).exists()]
    counters.bump("tile.records", len(kept))
    write_lines(processed / "index" / "tiles.jsonl", [json.dumps(r, sort_keys=True) for r in kept])
    write_json(processed / "reports" / "tiles.json", {
        "object_tiles": sum(1 for r in kept if not r["is_negative"]),
        "negative_tiles": sum(1 for r in kept if r["is_negative"]),
        "by_modality": dict(Counter(r["modality"] for r in kept)),
        "tile": tile, "config": tiles_cfg, "counters": counters.as_dict(),
    })

    counters.report(log)
    log.info("done", tiles=len(kept), gate="G13 requires >= 2000 object tiles at exactly 640x640")
    return 1 if failed_parents else 0


if __name__ == "__main__":
    run(main)
