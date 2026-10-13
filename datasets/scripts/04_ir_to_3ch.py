#!/usr/bin/env python3
"""Single-channel LWIR -> 3-channel, per DATASET_SPEC.md section 3.4 and slide 4.

The invariant: B = G = R. No false colour. MEASUREMENTS.md section 1 measured what
false-colour thermal costs a COCO-pretrained detector: AP@50 = 0.133, precision 0.364,
recall 0.093. This pipeline replicates the channel instead.

The conversion itself (read single channel, 16-bit linearly rescaled to 8-bit, CLAHE clip 2.0
grid 8x8, np.repeat to three channels) lives in _lib.ir_to_three_channel and is applied by
_lib.load_for_output at the moment a pixel is written into the dataset, by 08 (tiles) and 09
(frames). Nothing 3-channel is stored in between: at Kaggle scale a lossless 3-channel
intermediate of ~31k LWIR frames would take ~36 GB, more than the whole output budget.

What this step does, for every LWIR record:
  * decodes the source frame and counts anything undecodable (never silently skipped);
  * runs the conversion and asserts B = G = R on the result;
  * records bit depth, colour-input and the raw intensity standard deviation (`ir_std`),
    which 06_split.py uses for the hard set's heavy-infrared bucket;
  * writes <processed>/index/combined.jsonl: every source index, LWIR records annotated.
Gate G10 then verifies the FINAL dataset files: 3 channels and B = G = R.
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
    ir_conversion_params,
    ir_probe_worker,
    load_sources,
    load_state,
    parallel_map,
    read_jsonl,
    run,
    save_state,
    write_json,
    write_lines,
)

SCRIPT = "04_ir_to_3ch"


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--max-failures", type=int, default=0,
                    help="undecodable LWIR frames tolerated (each is dropped, logged and counted)")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    processed = Path(args.processed).resolve()
    index_dir = processed / "index"
    clip, grid = ir_conversion_params()  # refuses a false-colour config outright

    # Only the per-source indexes of ENABLED sources. Globbing *.jsonl would re-read this
    # script's own output and multiply the corpus on every re-run.
    enabled = [name for name, spec in load_sources()["sources"].items() if spec.get("enabled", True)]
    indexes = [index_dir / f"{name}.jsonl" for name in sorted(enabled)]
    present = [p for p in indexes if p.exists()]
    for path in indexes:
        if not path.exists():
            log.error("source index missing - run its converter first", index=str(path))
    if not present or len(present) != len(indexes):
        return 1

    records: list[dict] = []
    for path in present:
        chunk = read_jsonl(path)
        log.info("index read", index=path.name, records=len(chunk))
        records.extend(chunk)

    state = load_state(processed, SCRIPT)
    cache: dict[str, dict] = {} if args.force else dict(state.get("probe", {}))
    lwir = [r for r in records if r.get("modality") == "lwir"]
    if args.limit:
        lwir = lwir[: args.limit]
    todo = sorted({r["image"] for r in lwir if r["image"] not in cache})
    log.info("lwir frames", total=len(lwir), to_probe=len(todo), cached=len(lwir) - len(todo),
             clahe_clip=clip, grid=grid, workers=args.workers)

    if args.dry_run:
        log.info("dry-run: would probe", frames=len(todo))
        return 0

    for position, (path, result) in enumerate(zip(todo, parallel_map(ir_probe_worker, todo, args.workers, 32)), 1):
        cache[path] = result
        if position % 5000 == 0:
            log.info("probing", done=position, total=len(todo))

    failures = 0
    for record in records:
        if record.get("modality") != "lwir":
            continue
        probe = cache.get(record["image"])
        if probe is None:
            continue  # beyond --limit
        if not probe.get("ok"):
            failures += 1
            counters.bump(f"lwir.failed.{probe.get('why', 'unknown')}")
            if failures <= 10:
                log.warn("LWIR frame failed conversion", image=record["image"], why=probe.get("why"))
            record["ir_failed"] = True
            continue
        record["channels"] = 3
        record["ir_conversion"] = f"clahe{clip}_g{grid}_replicate3"
        record["ir_std"] = probe["ir_std"]
        record["ir_bit16"] = probe["bit16"]
        counters.bump("lwir.verified_bgr_equal")
        if probe.get("bit16"):
            counters.bump("lwir.rescaled_from_16bit")
        if probe.get("colour_input"):
            counters.bump("lwir.colour_input_flattened")

    kept = [r for r in records if not r.get("ir_failed")]
    write_lines(index_dir / "combined.jsonl", [json.dumps(r, sort_keys=True) for r in kept])
    state["probe"] = cache
    save_state(processed, SCRIPT, state)
    write_json(processed / "reports" / "ir_to_3ch.json",
               {"lwir": len(lwir), "failures": failures, "records_out": len(kept), "counters": counters.as_dict()})

    counters.report(log)
    log.info("done", records=len(kept), lwir_failed=failures, index=str(index_dir / "combined.jsonl"))
    return 1 if failures > args.max_failures else 0


if __name__ == "__main__":
    run(main)
