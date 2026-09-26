#!/usr/bin/env python3
"""Apply splits.yaml. DATASET_SPEC.md sections 2 and 2.7.

The mechanism that makes frame-level leakage structurally impossible: this script
builds a mapping of sequence_key -> split and then assigns every image by looking up
its sequence_key. There is no code path that assigns a split to an image directly.
assert_sequence_separation() proves it afterwards by grouping the finished assignment
back by sequence_key and failing if any key resolves to more than one split.

Per source:
  IDD    official train -> train; official val drives carved frame-weighted into val / test.
  FLIR   official train -> train, val -> val, video test -> test, keyed by videoId.
         assert_flir_videos_disjoint() checks no videoId sits in two official splits; any
         that does is re-split as a whole video (majority split) and the move is counted.
         The video test set is temporally decimated (every Nth frame of each video).
  LLVIP  official test -> test; val carved from official train by scene, frame-weighted.
  KAIST  (disabled) explicit set table.

Then modality balancing (section 2.7): a split whose visible share exceeds its target loses
whole visible-only sequences, in a seeded order, until it does not. Infrared is never dropped.
Every dropped image is listed in manifests/balance_dropped.txt.

Determinism: every random choice goes through random.Random seeded from --seed (string
seeds per purpose, which Python hashes with sha512, independent of PYTHONHASHSEED), and
every iteration is over a sorted list, so the same seed produces a byte-identical
splits.json. Gate G16 checks this by sha256.
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
    detector_px,
    hamming,
    load_splits_config,
    parse_label_file,
    read_jsonl,
    require_dirs,
    run,
    sha256_file,
    write_json,
    write_lines,
)

SCRIPT = "06_split"
SPLITS = ("train", "val", "test")


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


def assert_flir_videos_disjoint(records: list[dict], log: Logger, counters: Counters) -> dict[str, str]:
    """videoId -> the one split its frames go to. A videoId found in two official splits is
    re-split as a whole: every frame follows the split that holds most of them (ties broken
    test, val, train, the conservative direction). Returns the final video -> split map, and
    the check is repeated on it so the function cannot return a leaking map."""
    by_video: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        if record.get("source") == "flir":
            by_video[record["sequence_key"]][record.get("official_split", "train")] += 1
    priority = {"test": 0, "val": 1, "train": 2}
    final: dict[str, str] = {}
    for video, splits in sorted(by_video.items()):
        if len(splits) == 1:
            final[video] = next(iter(splits))
            continue
        winner = sorted(splits.items(), key=lambda kv: (-kv[1], priority.get(kv[0], 9)))[0][0]
        final[video] = winner
        counters.bump("flir.video_resplit")
        log.warn("FLIR videoId in more than one official split - re-split as a whole video",
                 video=video, official=dict(splits), assigned=winner)
    for video, split in final.items():
        if split not in SPLITS:
            raise SequenceSeparationError(f"FLIR video {video} resolved to unknown split {split}")
    return final


def carve(sizes: dict[str, int], share: float, rng: random.Random) -> set[str]:
    """Take whole sequences, in a seeded order, until they hold `share` of the frames."""
    keys = sorted(sizes)
    rng.shuffle(keys)
    target = share * sum(sizes.values())
    taken: set[str] = set()
    held = 0
    for key in keys:
        if held >= target:
            break
        taken.add(key)
        held += sizes[key]
    return taken


def phash_pseudo_sequences(records: list[dict], threshold: int, log: Logger) -> dict[str, str]:
    """OQ-3 fallback for LLVIP: cluster by pHash adjacency into conservative pseudo-sequences."""
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


def build_assignment(records: list[dict], config: dict, seed: int, log: Logger, counters: Counters) -> dict[str, str]:
    llvip_records = [r for r in records if r.get("source") == "llvip"]
    llvip_missing = [r for r in llvip_records if not r.get("sequence_key")]
    if llvip_missing:
        log.warn("LLVIP scene keys absent - using perceptual-hash pseudo-sequences (OQ-3)", affected=len(llvip_missing))
        counters.bump("llvip.pseudo_sequence_fallback")
        fallback = phash_pseudo_sequences(
            llvip_missing, int(config["llvip"]["scene_key"].get("fallback_hamming", 6)), log
        )
        for record in llvip_missing:
            record["sequence_key"] = fallback[record["image"]]

    assignment: dict[str, str] = {}

    # KAIST (disabled in sources.yaml; kept so the script stays valid if it is re-enabled).
    kaist_sets = config.get("kaist", {}).get("sets", {})
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

    # IDD: official train -> train; official val drives -> val / test, frame-weighted.
    idd_val_sizes: Counter = Counter()
    for record in records:
        if record.get("source") != "idd":
            continue
        if record.get("official_split") == "val":
            idd_val_sizes[record["sequence_key"]] += 1
        elif record.get("official_split") == "train":
            assignment[record["sequence_key"]] = "train"
        else:
            counters.bump("idd.official_test_unused")
    to_val = carve(dict(idd_val_sizes), float(config["idd"]["official_val_split"]["val"]),
                   random.Random(f"{seed}-idd-val"))
    for key in idd_val_sizes:
        assignment[key] = "val" if key in to_val else "test"

    # FLIR: official partition by videoId, asserted disjoint.
    flir_cfg = config.get("flir", {})
    route = {"train": flir_cfg.get("official_train_to", "train"),
             "val": flir_cfg.get("official_val_to", "val"),
             "test": flir_cfg.get("official_test_to", "test")}
    for video, official in assert_flir_videos_disjoint(records, log, counters).items():
        assignment[video] = route[official]

    # LLVIP: official test -> test; val carved from official train by scene, frame-weighted.
    llvip_cfg = config["llvip"]
    train_scene_sizes: Counter = Counter()
    for record in llvip_records:
        if record.get("official_split") == "test":
            assignment[record["sequence_key"]] = llvip_cfg.get("official_test_to", "test")
        else:
            train_scene_sizes[record["sequence_key"]] += 1
    to_val = carve(dict(train_scene_sizes), float(llvip_cfg["val_share_of_official_train"]),
                   random.Random(f"{seed}-llvip-val"))
    for key in train_scene_sizes:
        if key in assignment and assignment[key] == "test":
            raise SequenceSeparationError(f"LLVIP scene {key} holds both official train and test frames")
        assignment[key] = "val" if key in to_val else "train"
    # Elevated-view and thermal sources (section 1.7), one generic rule per source.
    for name, rule in sorted((config.get("extra_sources") or {}).items()):
        assign_extra_source(name, rule, records, assignment, seed, counters)
    return assignment


def assign_extra_source(name: str, rule: dict, records: list[dict], assignment: dict[str, str], seed: int,
                        counters: Counters) -> None:
    """policy `official`: official_to maps each official split to ours; an optional `carve` moves a
    frame-weighted share of one official split's sequences to val (the rest keep official_to).
    policy `resplit`: the official split is ignored (it separates neighbouring frames of one
    sequence) and whole sequences are carved into train/val/test by `proportions`."""
    sizes: Counter = Counter()
    official_of: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.get("source") != name:
            continue
        sizes[record["sequence_key"]] += 1
        official_of[record["sequence_key"]].add(record.get("official_split", "unknown"))
    if not sizes:
        return
    rng = random.Random(f"{seed}-{name}")
    if rule.get("policy") == "resplit":
        shares = rule["proportions"]
        remaining = dict(sizes)
        to_val = carve(remaining, float(shares["val"]), rng)
        for key in to_val:
            remaining.pop(key)
        test_share = float(shares["test"]) / max(1e-9, 1.0 - float(shares["val"]))
        to_test = carve(remaining, test_share, rng)
        for key in sizes:
            assignment[key] = "val" if key in to_val else ("test" if key in to_test else "train")
        counters.bump(f"{name}.resplit_sequences", len(sizes))
        return
    route = rule.get("official_to", {})
    carve_rule = rule.get("carve") or {}
    carve_pool = {k: v for k, v in sizes.items() if official_of[k] == {carve_rule.get("official")}} if carve_rule else {}
    to_val = carve(carve_pool, float(carve_rule.get("val", 0.0)), rng) if carve_pool else set()
    for key in sizes:
        officials = official_of[key]
        if len(officials) != 1:
            raise SequenceSeparationError(f"{name} sequence {key} spans official splits {sorted(officials)}")
        official = next(iter(officials))
        if key in to_val:
            assignment[key] = "val"
        elif official in route:
            assignment[key] = route[official]
        else:
            counters.bump(f"{name}.official_{official}_unrouted")


def decimate_flir_test(records: list[dict], step: int, counters: Counters) -> tuple[list[dict], list[dict]]:
    """Keep every `step`-th frame of each FLIR test video. Within one sealed split this cannot
    leak; it removes near-identical consecutive frames that would swamp the 15% target."""
    if step <= 1:
        return records, []
    by_video: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("source") == "flir" and record.get("split") == "test":
            by_video[record["sequence_key"]].append(record)
    drop: set[str] = set()
    for _video, frames in sorted(by_video.items()):
        frames.sort(key=lambda r: (r.get("frame_index", -1), r["image"]))
        for position, record in enumerate(frames):
            if position % step:
                drop.add(record["image"])
    counters.bump("flir.test_decimated_dropped", len(drop))
    return [r for r in records if r["image"] not in drop], [r for r in records if r["image"] in drop]


def balance_split(records: list[dict], split: str, target: float, sources: list[str], seed: int,
                  log: Logger, counters: Counters) -> tuple[list[dict], list[dict]]:
    """Section 2.7. Drop whole visible-only sequences of `sources` until the split's visible
    share is <= target. Never drops infrared. Returns (kept, dropped)."""
    in_split = [r for r in records if r.get("split") == split]
    visible = sum(1 for r in in_split if r.get("modality") == "visible")
    total = len(in_split)
    if not total or visible / total <= target:
        return records, []

    seq_modalities: dict[str, set[str]] = defaultdict(set)
    seq_size: Counter = Counter()
    seq_source: dict[str, str] = {}
    for record in in_split:
        key = record["sequence_key"]
        seq_modalities[key].add(record.get("modality"))
        seq_size[key] += 1
        seq_source[key] = record.get("source")

    rng = random.Random(f"{seed}-balance-{split}")
    order: list[str] = []
    for source in sources:
        keys = sorted(k for k in seq_size if seq_source[k] == source and seq_modalities[k] == {"visible"})
        rng.shuffle(keys)
        order.extend(keys)

    dropped_keys: set[str] = set()
    for key in order:
        if visible / max(1, total) <= target:
            break
        dropped_keys.add(key)
        visible -= seq_size[key]
        total -= seq_size[key]
    share = visible / max(1, total)
    if share > target:
        log.warn("balance exhausted its candidate sequences", split=split, share=round(share, 4), target=target)
        counters.bump(f"balance.{split}.exhausted")
    dropped = [r for r in records if r.get("split") == split and r["sequence_key"] in dropped_keys]
    kept = [r for r in records if not (r.get("split") == split and r["sequence_key"] in dropped_keys)]
    counters.bump(f"balance.{split}.sequences_dropped", len(dropped_keys))
    counters.bump(f"balance.{split}.images_dropped", len(dropped))
    log.info("modality balance", split=split, target=target, visible_share_after=round(share, 4),
             sequences_dropped=len(dropped_keys), images_dropped=len(dropped))
    return kept, dropped


def enrich_for_hard_set(records: list[dict]) -> None:
    """The geometric properties section 2.6 selects on, from the label files, val only.
    Heights are at the detector input (long side 640), the scale of MEASUREMENTS.md section 3."""
    for record in records:
        if record.get("split") != "val":
            continue
        label_path = Path(record.get("label", ""))
        rows, _errors = parse_label_file(label_path) if label_path.exists() else ([], [])
        persons = [r for r in rows if int(r[0]) == 0]
        record["person_instances"] = len(persons)
        if persons:
            heights = sorted(detector_px(r[4], record.get("img_w", 0), record.get("img_h", 0)) for r in persons)
            middle = len(heights) // 2
            record["median_person_height_px"] = round(
                heights[middle] if len(heights) % 2 else (heights[middle - 1] + heights[middle]) / 2.0, 3
            )
        occluded = sum(1 for i, a in enumerate(persons) if any(_iou(a, b) > 0.3 for j, b in enumerate(persons) if i != j))
        record["occluded_share"] = round(occluded / len(persons), 4) if persons else 0.0


def _iou(a, b) -> float:
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def build_hard_set(records: list[dict], config: dict, log: Logger, counters: Counters) -> list[str]:
    """Section 2.6: 280 val images, four buckets, frozen to a manifest. Never drawn from test."""
    spec = config["hard_set"]
    buckets = spec["buckets"]
    val_records = sorted((r for r in records if r["split"] == "val"), key=lambda r: r["image"])
    chosen: list[str] = []
    seen: set[str] = set()

    def take(candidates: list[dict], count: int, bucket: str) -> None:
        for record in candidates:
            if len(chosen) >= spec["total"] or count <= 0:
                return
            if record["image"] in seen:
                continue
            chosen.append(record["image"])
            seen.add(record["image"])
            counters.bump(f"hard_set.{bucket}")
            count -= 1

    tiny = [r for r in val_records if r.get("median_person_height_px") is not None
            and r["median_person_height_px"] <= buckets["tiny_objects"]["value"]]
    take(tiny, buckets["tiny_objects"]["count"], "tiny_objects")

    infrared = sorted((r for r in val_records if r.get("modality") == "lwir" and r.get("ir_std") is not None),
                      key=lambda r: (r["ir_std"], r["image"]))
    decile = infrared[: max(1, len(infrared) // 10)] if infrared else []
    take(decile, buckets["heavy_infrared"]["count"], "heavy_infrared")

    crowds = [r for r in val_records if r.get("person_instances", 0) >= buckets["crowds"]["value"]]
    take(crowds, buckets["crowds"]["count"], "crowds")

    occluded = [r for r in val_records if r.get("occluded_share", 0.0) >= buckets["occlusion"]["value"]]
    take(occluded, buckets["occlusion"]["count"], "occlusion")

    if len(chosen) < spec["total"]:
        take([r for r in val_records if r["image"] not in seen], spec["total"] - len(chosen), "filler")

    if len(chosen) != spec["total"]:
        log.warn("hard set is short - gate G15 will fail", have=len(chosen), want=spec["total"])
    return chosen


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
    log.info("seeded", seed=args.seed)
    records = sorted(read_jsonl(index_path), key=lambda r: r["image"])
    log.info("records", count=len(records))

    try:
        assignment = build_assignment(records, config, args.seed, log, counters)
    except SequenceSeparationError as exc:
        log.fatal("sequence separation assertion failed", detail=str(exc)[:400])
        return 1

    unassigned = 0
    for record in records:
        split = assignment.get(record.get("sequence_key"))
        if split is None:
            unassigned += 1
            counters.bump(f"record.unassigned.{record.get('source')}")
            continue
        record["split"] = split
    assigned = [r for r in records if r.get("split")]

    assigned, decimated = decimate_flir_test(assigned, int(config.get("flir", {}).get("test_decimation", 1)), counters)

    balance_cfg = config.get("balance", {})
    targets = balance_cfg.get("target_visible_share", {})
    dropped_balance: list[dict] = []
    for split in SPLITS:
        if split in targets:
            assigned, dropped = balance_split(assigned, split, float(targets[split]),
                                              list(balance_cfg.get("subsample_sources", [])), args.seed, log, counters)
            dropped_balance.extend(dropped)
    removed_keys = {r["sequence_key"] for r in dropped_balance}
    final_assignment = {k: v for k, v in assignment.items() if k not in removed_keys}

    try:
        assert_sequence_separation(final_assignment, assigned)
    except SequenceSeparationError as exc:
        log.fatal("sequence separation assertion failed", detail=str(exc)[:400])
        return 1
    log.info("assert_sequence_separation passed", sequences=len(final_assignment))

    counts = Counter(r["split"] for r in assigned)
    total = max(1, sum(counts.values()))
    proportions = {k: round(v / total, 4) for k, v in counts.items()}
    tolerance = float(config["proportions"]["tolerance"])
    for name in SPLITS:
        target = float(config["proportions"][name])
        actual = proportions.get(name, 0.0)
        log.info("proportion (untiled, before negatives)", split=name, target=target, actual=actual,
                 status="ok" if abs(actual - target) <= tolerance else "OUT OF TOLERANCE")

    modality_by_split: dict[str, Counter] = defaultdict(Counter)
    for record in assigned:
        modality_by_split[record["split"]][record.get("modality", "unknown")] += 1
    for split, counter in sorted(modality_by_split.items()):
        log.info("day/ir", split=split, visible=counter.get("visible", 0), lwir=counter.get("lwir", 0),
                 visible_share=round(counter.get("visible", 0) / max(1, sum(counter.values())), 4))

    payload = {
        "seed": args.seed,
        "sequence_to_split": dict(sorted(final_assignment.items())),
        "counts": dict(sorted(counts.items())),
        "proportions": dict(sorted(proportions.items())),
        "modality": {k: dict(sorted(v.items())) for k, v in sorted(modality_by_split.items())},
        "balance_dropped_sequences": sorted(removed_keys),
        "decimated": len(decimated),
        "unassigned": unassigned,
    }

    splits_json = processed / "splits.json"
    if args.dry_run:
        log.info("dry-run: nothing written", would_write=str(splits_json))
        counters.report(log)
        return 0

    write_json(splits_json, payload)
    enrich_for_hard_set(assigned)
    write_lines(processed / "index" / "split.jsonl", [json.dumps(r, sort_keys=True) for r in assigned])

    require_dirs(MANIFEST_DIR)
    for name in SPLITS:
        write_lines(MANIFEST_DIR / f"split_{name}.txt", sorted(r["image"] for r in assigned if r["split"] == name))
    write_lines(MANIFEST_DIR / str(balance_cfg.get("dropped_manifest", "balance_dropped.txt")),
                sorted(f"{r['split']}\t{r['image']}" for r in dropped_balance)
                + sorted(f"decimated\t{r['image']}" for r in decimated))

    hard = build_hard_set(assigned, config, log, counters)
    write_lines(MANIFEST_DIR / "hard_set.txt", hard)
    log.info("hard set frozen", entries=len(hard), target=config["hard_set"]["total"], path=str(MANIFEST_DIR / "hard_set.txt"))

    digest = sha256_file(splits_json)
    write_json(
        processed / "reports" / "split.json",
        {"sha256": digest, "seed": args.seed, "counters": counters.as_dict(), **payload["proportions"]},
    )
    log.info("splits written", path=str(splits_json), sha256=digest)

    counters.report(log)
    log.info("done", assigned=len(assigned), unassigned=unassigned, balance_dropped=len(dropped_balance),
             decimated=len(decimated))
    if unassigned:
        log.error("records without a split - every record must resolve through its sequence key", unassigned=unassigned)
        return 1
    return 0


if __name__ == "__main__":
    run(main)
