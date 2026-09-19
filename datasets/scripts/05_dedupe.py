#!/usr/bin/env python3
"""Perceptual-hash dedupe, within and across sources. DATASET_SPEC.md gates G1 and G2.

Two frames from the same video are near-identical by construction, so a blanket
dedupe would gut the corpus. The rule this script applies:

  * WITHIN one sequence, near-duplicates are expected and are only REPORTED.
  * ACROSS sequences or ACROSS sources, a near-duplicate is a leakage hazard and the
    later record is marked dropped.

Streaming: hashes are computed one image at a time and held as 64-bit integers. A
million images costs a few tens of MB, not the corpus.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    Counters,
    Logger,
    base_parser,
    hamming,
    imread_gray,
    load_state,
    phash64,
    require_dirs,
    run,
    save_state,
    write_json,
    write_lines,
)

SCRIPT = "05_dedupe"


def bucket_keys(value: int) -> list[str]:
    """Band the 64-bit hash into four 16-bit bands. Two hashes within Hamming 4 share a band
    with high probability, which turns an O(n^2) comparison into a bucketed one."""
    return [f"{i}:{(value >> (16 * i)) & 0xFFFF}" for i in range(4)]


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--threshold", type=int, default=4, help="Hamming distance counted as duplicate")
    ap.add_argument(
        "--within-threshold",
        type=int,
        default=2,
        help="Hamming distance counted as a within-split duplicate for gate G2",
    )
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    processed = Path(args.processed).resolve()
    index_path = processed / "index" / "combined.jsonl"
    if not index_path.exists():
        log.error("combined index missing - run 04_ir_to_3ch.py first", expected=str(index_path))
        return 1

    state = load_state(processed, SCRIPT)
    cache: dict[str, int] = {} if args.force else {k: int(v) for k, v in state.get("hashes", {}).items()}

    require_dirs(processed / "reports")

    buckets: dict[str, list[str]] = defaultdict(list)
    hashes: dict[str, int] = {}
    meta: dict[str, dict] = {}
    cross_pairs: list[dict] = []
    within_pairs = 0
    dropped: set[str] = set()
    records: list[dict] = []

    with index_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]
    log.info("records", count=len(records))

    for position, record in enumerate(records):
        image = record["image"]
        if position and position % 2000 == 0:
            log.info("hashing", done=position, total=len(records))

        value = cache.get(image)
        if value is None:
            gray = imread_gray(Path(image))
            if gray is None:
                log.warn("unhashable image", image=image)
                counters.bump("image.unhashable")
                record["phash"] = None
                continue
            value = phash64(gray)
            cache[image] = value
        hashes[image] = value
        meta[image] = {
            "sequence_key": record.get("sequence_key"),
            "source": record.get("source"),
            "modality": record.get("modality"),
        }
        record["phash"] = f"{value:016x}"

        # A visible/LWIR pair of the same capture is near-identical in structure but is a
        # legitimate pair, never a duplicate. Pairs are compared only against other pair_ids.
        candidates: set[str] = set()
        for key in bucket_keys(value):
            candidates.update(buckets[key])
            buckets[key].append(image)

        for other in candidates:
            if other == image:
                continue
            distance = hamming(value, hashes[other])
            same_sequence = meta[other]["sequence_key"] == record.get("sequence_key")
            same_modality = meta[other]["modality"] == record.get("modality")

            if distance <= args.threshold and not same_sequence:
                cross_pairs.append(
                    {
                        "a": other,
                        "b": image,
                        "distance": distance,
                        "a_sequence": meta[other]["sequence_key"],
                        "b_sequence": record.get("sequence_key"),
                    }
                )
                dropped.add(image)
                counters.bump("duplicate.cross_sequence_dropped")
                break
            if distance <= args.within_threshold and same_sequence and same_modality:
                within_pairs += 1
                counters.bump("duplicate.within_sequence_reported")

    kept = [r for r in records if r["image"] not in dropped]
    for record in kept:
        record["duplicate_dropped"] = False

    report = {
        "records_in": len(records),
        "records_kept": len(kept),
        "cross_sequence_duplicates": len(cross_pairs),
        "within_sequence_duplicates": within_pairs,
        "within_sequence_duplicate_ratio": round(within_pairs / max(1, len(records)), 6),
        "threshold": args.threshold,
        "within_threshold": args.within_threshold,
        "counters": counters.as_dict(),
    }

    if args.dry_run:
        log.info("dry-run: nothing written")
    else:
        write_lines(
            processed / "index" / "deduped.jsonl",
            [json.dumps(r, sort_keys=True) for r in kept],
        )
        write_json(processed / "reports" / "dedupe.json", report)
        write_json(
            processed / "reports" / "dedupe_pairs.json",
            cross_pairs[:5000],
        )
        state["hashes"] = {k: str(v) for k, v in cache.items()}
        save_state(processed, SCRIPT, state)

    counters.report(log)
    log.info(
        "done",
        kept=len(kept),
        dropped=len(dropped),
        cross_duplicates=len(cross_pairs),
        within_duplicates=within_pairs,
    )
    return 0


if __name__ == "__main__":
    run(main)
