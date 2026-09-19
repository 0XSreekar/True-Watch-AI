#!/usr/bin/env python3
"""Build the background/negative pool. DATASET_SPEC.md sections 3.5 and 5.

Why this script matters more than its size suggests: slide 5's "under 5 false alerts
per camera per day" is a precision target, and precision on a 24-hour stream is decided
by how the model behaves on the 23 hours that contain nothing. A detector that has never
been shown an empty frame has no way to represent one.

Target: 10% of TRAINING images carry an empty label file (gate G9, tolerance +/- 2%).
Negatives go to train only; val and test keep their natural composition so their metrics
stay comparable to published numbers on these datasets.

Six categories with fixed quotas (section 5) so the pool cannot become 90% empty road.
The highest-value category, IR hot spots, is detected here: an unannotated person-sized
hot blob in an LWIR frame is exactly the false positive a thermal camera generates for free.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    MANIFEST_DIR,
    Counters,
    Logger,
    base_parser,
    imread_gray,
    require_dirs,
    run,
    write_json,
    write_lines,
)

SCRIPT = "07_negatives"

QUOTAS = {
    "empty_road": 0.30,
    "ir_hotspot": 0.20,
    "night_glare": 0.20,
    "foliage": 0.15,
    "weather": 0.10,
    "animal": 0.05,
}

HOTSPOT_MIN_AREA = 100
HOTSPOT_MAX_AREA = 2000


def classify(record: dict, counters: Counters) -> str | None:
    """Assign a negative category. Returns None when the frame is not a usable negative."""
    if record.get("objects", 0) > 0:
        return None
    if record.get("source") == "llvip":
        # LLVIP carries unlabelled vehicles (section 1.4); an empty LLVIP frame is not
        # evidence of an empty scene.
        counters.bump("rejected.llvip_person_only")
        return None
    if not record.get("negative_worthy", False):
        counters.bump("rejected.not_negative_worthy")
        return None
    if record.get("animal_only"):
        return "animal"
    if record.get("modality") == "lwir":
        return "ir_hotspot"
    if record.get("set") in {"set03", "set04", "set05", "set09", "set10", "set11"}:
        return "night_glare"
    return "empty_road"


def has_person_sized_hotspot(path: Path) -> bool:
    """A sun-warmed rock or an engine block at person temperature and person size."""
    import cv2
    import numpy as np

    gray = imread_gray(path)
    if gray is None:
        return False
    threshold = float(np.percentile(gray, 99.0))
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if HOTSPOT_MIN_AREA <= area <= HOTSPOT_MAX_AREA:
            return True
    return False


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--ratio", type=float, default=0.10, help="share of training images with no objects")
    ap.add_argument(
        "--verify-hotspots",
        action="store_true",
        help="open each LWIR candidate and keep only those with a person-sized hot blob",
    )
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()
    rng = random.Random(args.seed)

    processed = Path(args.processed).resolve()
    index_path = processed / "index" / "split.jsonl"
    if not index_path.exists():
        log.error("split index missing - run 06_split.py first", expected=str(index_path))
        return 1

    records: list[dict] = []
    with index_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["image"])

    train = [r for r in records if r.get("split") == "train"]
    positives = [r for r in train if r.get("objects", 0) > 0]
    log.info("training records", total=len(train), positives=len(positives))

    if not positives:
        log.error("no positive training images - nothing to size the negative pool against")
        return 1

    # n / (positives + n) = ratio  =>  n = ratio * positives / (1 - ratio)
    target = int(round(args.ratio * len(positives) / max(1e-9, (1.0 - args.ratio))))
    log.info("negative pool target", target=target, ratio=args.ratio)

    candidates: dict[str, list[dict]] = {key: [] for key in QUOTAS}
    for record in train:
        category = classify(record, counters)
        if category is None:
            continue
        candidates[category].append(record)
        counters.bump(f"candidate.{category}")

    if args.verify_hotspots and not args.dry_run:
        verified = []
        for record in candidates["ir_hotspot"]:
            if has_person_sized_hotspot(Path(record["image"])):
                verified.append(record)
            else:
                counters.bump("ir_hotspot.no_blob")
        log.info("hotspot verification", before=len(candidates["ir_hotspot"]), after=len(verified))
        candidates["ir_hotspot"] = verified

    chosen: list[dict] = []
    shortfalls: dict[str, int] = {}
    for category, share in sorted(QUOTAS.items()):
        want = int(round(target * share))
        pool = sorted(candidates[category], key=lambda r: r["image"])
        rng.shuffle(pool)
        take = pool[:want]
        if len(take) < want:
            shortfalls[category] = want - len(take)
            log.warn("category short of its quota", category=category, want=want, have=len(take))
        for record in take:
            record["negative_category"] = category
        chosen.extend(take)
        counters.bump(f"selected.{category}", len(take))

    # Redistribute a shortfall across categories that still have candidates, so the total
    # ratio is met even when one source is thin. The quota is a shape preference; the
    # ratio is the gate.
    missing = target - len(chosen)
    if missing > 0:
        already = {r["image"] for r in chosen}
        spare = sorted(
            (r for pool in candidates.values() for r in pool if r["image"] not in already),
            key=lambda r: r["image"],
        )
        rng.shuffle(spare)
        for record in spare[:missing]:
            record["negative_category"] = record.get("negative_category") or "redistributed"
            chosen.append(record)
            counters.bump("selected.redistributed")

    achieved = len(chosen) / max(1, len(positives) + len(chosen))
    log.info(
        "negative pool built",
        selected=len(chosen),
        target=target,
        achieved_ratio=round(achieved, 4),
        tolerance="0.08-0.12 (gate G9)",
    )

    payload = {
        "target": target,
        "selected": len(chosen),
        "achieved_ratio": round(achieved, 6),
        "requested_ratio": args.ratio,
        "by_category": dict(sorted(Counter(r.get("negative_category") for r in chosen).items())),
        "shortfalls": shortfalls,
        "counters": counters.as_dict(),
    }

    if args.dry_run:
        log.info("dry-run: nothing written")
        counters.report(log)
        return 0

    require_dirs(MANIFEST_DIR)
    write_json(MANIFEST_DIR / "negatives.json", payload)
    write_lines(MANIFEST_DIR / "negatives.txt", sorted(r["image"] for r in chosen))

    selected = {r["image"] for r in chosen}
    for record in records:
        record["is_negative"] = record["image"] in selected
    write_lines(
        processed / "index" / "with_negatives.jsonl",
        [json.dumps(r, sort_keys=True) for r in records],
    )
    write_json(processed / "reports" / "negatives.json", payload)

    counters.report(log)
    if achieved < 0.08 or achieved > 0.12:
        log.error("negative ratio outside the gate G9 band", achieved=round(achieved, 4))
        return 1
    log.info("done", negatives=len(chosen))
    return 0


if __name__ == "__main__":
    run(main)
