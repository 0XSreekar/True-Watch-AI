#!/usr/bin/env python3
"""Apply splits.yaml. DATASET_SPEC.md section 2.

The mechanism that makes frame-level leakage structurally impossible: this script
builds a mapping of sequence_key -> split and then assigns every image by looking up
its sequence_key. There is no code path that assigns a split to an image directly.
assert_sequence_separation() proves it afterwards by grouping the finished assignment
back by sequence_key and failing if any key resolves to more than one split.

Determinism: every random choice goes through one random.Random(seed) instance seeded
from --seed, and every iteration is over a sorted list, so the same seed produces a
byte-identical splits.json. Gate G16 checks this by sha256.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    MANIFEST_DIR,
    Counters,
    Logger,
    base_parser,
    hamming,
    load_splits_config,
    parse_label_file,
    require_dirs,
    run,
    sha256_file,
    write_json,
    write_lines,
)

SCRIPT = "06_split"


class SequenceSeparationError(AssertionError):
    """Raised when one sequence_key resolves to more than one split. Gate G3."""


def assert_sequence_separation(assignment: dict[str, str], records: list[dict]) -> None:
    """The named assertion. A sequence in two splits is a leak, and a leak is fatal here."""
    observed: dict[str, set[str]] = defaultdict(set)
    for record in records:
        key = record.get("sequence_key")
        split = record.get("split")
        if key is None or split is None:
            continue
        observed[key].add(split)
    offenders = {k: sorted(v) for k, v in observed.items() if len(v) > 1}
    if offenders:
        raise SequenceSeparationError(
            f"{len(offenders)} sequence keys resolve to more than one split: "
            + ", ".join(f"{k}->{v}" for k, v in sorted(offenders.items())[:10])
        )
    for key, split in assignment.items():
        if key in observed and split not in observed[key]:
            raise SequenceSeparationError(
                f"sequence {key} was assigned {split} but its images carry {sorted(observed[key])}"
            )


def phash_pseudo_sequences(records: list[dict], threshold: int, log: Logger) -> dict[str, str]:
    """OQ-3 fallback for LLVIP: cluster by pHash adjacency into conservative pseudo-sequences.

    Over-grouping costs split granularity but cannot leak, which is the correct direction
    to be wrong in."""
    assigned: dict[str, str] = {}
    centroids: list[tuple[int, str]] = []
    for record in records:
        digest = record.get("phash")
        if not digest:
            assigned[record["image"]] = f"llvip/orphan/{Path(record['image']).stem}"
            continue
        value = int(digest, 16)
        match = None
        for centroid, name in centroids:
            if hamming(value, centroid) <= threshold:
                match = name
                break
        if match is None:
            match = f"llvip/pseudo{len(centroids):04d}"
            centroids.append((value, match))
        assigned[record["image"]] = match
    log.info("pseudo-sequences formed", clusters=len(centroids), threshold=threshold)
    return assigned


def build_assignment(records: list[dict], config: dict, rng: random.Random, log: Logger, counters: Counters):
    kaist_sets = config["kaist"]["sets"]
    idd_config = config["idd"]
    llvip_config = config["llvip"]

    # LLVIP scene keys, with the OQ-3 fallback when filenames do not partition the corpus.
    llvip_records = sorted(
        (r for r in records if r.get("source") == "llvip"), key=lambda r: r["image"]
    )
    llvip_missing = [r for r in llvip_records if not r.get("sequence_key")]
    if llvip_missing:
        log.warn(
            "LLVIP scene keys absent - using perceptual-hash pseudo-sequences (OQ-3)",
            affected=len(llvip_missing),
        )
        counters.bump("llvip.pseudo_sequence_fallback")
        fallback = phash_pseudo_sequences(
            llvip_missing, int(llvip_config["scene_key"].get("fallback_hamming", 6)), log
        )
        for record in llvip_missing:
            record["sequence_key"] = fallback[record["image"]]

    assignment: dict[str, str] = {}

    # KAIST: entirely declarative. splits.yaml names the split of every set.
    for record in records:
        if record.get("source") != "kaist":
            continue
        set_name = record.get("set") or (record.get("sequence_key") or "//").split("/")[1]
        declared = kaist_sets.get(set_name)
        if declared is None:
            log.error("KAIST set is not declared in splits.yaml", set=set_name)
            counters.bump("kaist.undeclared_set")
            continue
        assignment[record["sequence_key"]] = declared["split"]

    # IDD: official train -> train; official val split by drive into our val and test.
    idd_val_drives = sorted(
        {
            r["sequence_key"]
            for r in records
            if r.get("source") == "idd" and r.get("official_split") == "val"
        }
    )
    rng.shuffle(idd_val_drives)
    half = int(round(len(idd_val_drives) * float(idd_config["official_val_split"]["val"])))
    for index, key in enumerate(idd_val_drives):
        assignment[key] = "val" if index < half else "test"
    for record in records:
        if record.get("source") != "idd":
            continue
        key = record["sequence_key"]
        if key in assignment:
            continue
        official = record.get("official_split", "train")
        if official == "test":
            counters.bump("idd.official_test_unused")
            continue
        assignment[key] = "train"

    # LLVIP: official test -> our test, untouched. Val carved from official train by scene key.
    llvip_train_scenes = sorted(
        {
            r["sequence_key"]
            for r in records
            if r.get("source") == "llvip" and r.get("official_split") != "test"
        }
    )
    rng.shuffle(llvip_train_scenes)
    val_count = int(round(len(llvip_train_scenes) * float(llvip_config["val_share_of_official_train"])))
    for index, key in enumerate(llvip_train_scenes):
        assignment[key] = "val" if index < val_count else "train"
    for record in records:
        if record.get("source") != "llvip":
            continue
        if record.get("official_split") == "test":
            assignment[record["sequence_key"]] = llvip_config.get("official_test_to", "test")

    return assignment



def enrich_for_hard_set(records: list[dict]) -> None:
    """Compute the two geometric properties section 2.6 selects on, from the label files.

    median_person_height_px and occluded_share are derived here rather than in the
    converters so that a change to the hard-set rule does not force a full reconversion."""
    for record in records:
        if record.get("split") != "val":
            continue
        label_path = Path(record.get("label", ""))
        height_px = int(record.get("img_h") or 0)
        rows, _errors = parse_label_file(label_path) if label_path.exists() else ([], [])
        persons = [r for r in rows if int(r[0]) == 0]
        if persons and height_px > 0:
            heights = sorted(r[4] * height_px for r in persons)
            middle = len(heights) // 2
            record["median_person_height_px"] = (
                heights[middle]
                if len(heights) % 2
                else (heights[middle - 1] + heights[middle]) / 2.0
            )
        occluded = 0
        for i, a in enumerate(persons):
            for b in persons[i + 1 :]:
                if _iou(a, b) > 0.3:
                    occluded += 2
                    break
        record["occluded_share"] = round(occluded / len(persons), 4) if persons else 0.0


def _iou(a, b) -> float:
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--index", default=None, help="input jsonl (default: <processed>/index/deduped.jsonl)")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    processed = Path(args.processed).resolve()
    index_path = Path(args.index) if args.index else processed / "index" / "deduped.jsonl"
    if not index_path.exists():
        log.error("input index missing - run 05_dedupe.py first", expected=str(index_path))
        return 1

    config = load_splits_config()
    rng = random.Random(args.seed)
    log.info("seeded", seed=args.seed)

    records: list[dict] = []
    with index_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["image"])  # determinism: never iterate an unsorted set
    log.info("records", count=len(records))

    assignment = build_assignment(records, config, rng, log, counters)

    unassigned = 0
    for record in records:
        key = record.get("sequence_key")
        split = assignment.get(key)
        if split is None:
            unassigned += 1
            counters.bump("record.unassigned")
            continue
        record["split"] = split
        counters.bump(f"split.{split}")

    assigned_records = [r for r in records if r.get("split")]

    try:
        assert_sequence_separation(assignment, assigned_records)
    except SequenceSeparationError as exc:
        log.fatal("sequence separation assertion failed", detail=str(exc)[:400])
        return 1
    log.info("assert_sequence_separation passed", sequences=len(assignment))

    counts = Counter(r["split"] for r in assigned_records)
    total = max(1, sum(counts.values()))
    proportions = {k: round(v / total, 4) for k, v in counts.items()}

    tolerance = float(config["proportions"]["tolerance"])
    for name in ("train", "val", "test"):
        target = float(config["proportions"][name])
        actual = proportions.get(name, 0.0)
        status = "ok" if abs(actual - target) <= tolerance else "OUT OF TOLERANCE"
        log.info("proportion", split=name, target=target, actual=actual, status=status)

    modality_by_split: dict[str, Counter] = defaultdict(Counter)
    for record in assigned_records:
        modality_by_split[record["split"]][record.get("modality", "unknown")] += 1
    for split, counter in sorted(modality_by_split.items()):
        visible = counter.get("visible", 0)
        split_total = max(1, sum(counter.values()))
        log.info("day/night", split=split, visible_share=round(visible / split_total, 4))

    payload = {
        "seed": args.seed,
        "sequence_to_split": dict(sorted(assignment.items())),
        "counts": dict(sorted(counts.items())),
        "proportions": dict(sorted(proportions.items())),
        "modality": {k: dict(sorted(v.items())) for k, v in sorted(modality_by_split.items())},
        "unassigned": unassigned,
    }

    splits_json = processed / "splits.json"
    if args.dry_run:
        log.info("dry-run: nothing written", would_write=str(splits_json))
        counters.report(log)
        return 0

    write_json(splits_json, payload)
    write_lines(
        processed / "index" / "split.jsonl",
        [json.dumps(r, sort_keys=True) for r in assigned_records],
    )

    require_dirs(MANIFEST_DIR)
    for name in ("train", "val", "test"):
        write_lines(
            MANIFEST_DIR / f"split_{name}.txt",
            sorted(r["image"] for r in assigned_records if r["split"] == name),
        )

    enrich_for_hard_set(assigned_records)
    build_hard_set(assigned_records, config, processed, log, counters)

    digest = sha256_file(splits_json)
    write_json(
        processed / "reports" / "split.json",
        {"sha256": digest, "seed": args.seed, "counters": counters.as_dict(), **payload["proportions"]},
    )
    log.info("splits written", path=str(splits_json), sha256=digest)

    counters.report(log)
    log.info("done", assigned=len(assigned_records), unassigned=unassigned)
    return 1 if unassigned and not assigned_records else 0


def build_hard_set(records: list[dict], config: dict, processed: Path, log: Logger, counters: Counters) -> None:
    """Section 2.6: 280 val images, four buckets, frozen to a manifest. Never drawn from test."""
    spec = config["hard_set"]
    val_records = sorted((r for r in records if r["split"] == "val"), key=lambda r: r["image"])
    chosen: list[str] = []
    seen: set[str] = set()

    def take(candidates: list[dict], count: int, bucket: str) -> None:
        for record in candidates:
            if len(chosen) >= spec["total"]:
                return
            if record["image"] in seen:
                continue
            chosen.append(record["image"])
            seen.add(record["image"])
            counters.bump(f"hard_set.{bucket}")
            count -= 1
            if count <= 0:
                return

    tiny = [r for r in val_records if r.get("median_person_height_px") is not None
            and r["median_person_height_px"] <= spec["buckets"]["tiny_objects"]["value"]]
    take(tiny, spec["buckets"]["tiny_objects"]["count"], "tiny_objects")

    infrared = [r for r in val_records if r.get("modality") == "lwir"]
    take(infrared, spec["buckets"]["heavy_infrared"]["count"], "heavy_infrared")

    crowds = [r for r in val_records if r.get("objects", 0) >= spec["buckets"]["crowds"]["value"]]
    take(crowds, spec["buckets"]["crowds"]["count"], "crowds")

    occluded = [r for r in val_records if r.get("occluded_share", 0.0) >= spec["buckets"]["occlusion"]["value"]]
    take(occluded, spec["buckets"]["occlusion"]["count"], "occlusion")

    if len(chosen) < spec["total"]:
        filler = [r for r in val_records if r["image"] not in seen]
        take(filler, spec["total"] - len(chosen), "filler")

    write_lines(MANIFEST_DIR / "hard_set.txt", chosen)
    log.info("hard set frozen", entries=len(chosen), target=spec["total"], path=str(MANIFEST_DIR / "hard_set.txt"))
    if len(chosen) != spec["total"]:
        log.warn("hard set is short - gate G15 will fail", have=len(chosen), want=spec["total"])


if __name__ == "__main__":
    run(main)
