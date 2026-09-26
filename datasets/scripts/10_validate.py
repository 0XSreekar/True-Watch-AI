#!/usr/bin/env python3
"""The gate. Every check in DATASET_SPEC.md section 9, with its numeric threshold.

Prints a PASS/FAIL table and exits 1 if any gate fails. Phase 2 does not start until
this exits 0.

It validates the BUILT dataset, not the pipeline's intermediate state: the records come from
<dataset>/index/split.jsonl, which 09 writes with one row per placed image, and every image
check opens the final file. So the gate runs anywhere the build is mounted (for example a
Kaggle input), with or without the processed/ directory that produced it.

G17 (plates) can be declared out of scope with --skip-plates, for a build that only makes the
detection corpus. A skipped gate prints SKIP, is recorded as skipped, and is never a PASS.

G19 is deliberately not automatable: a human opens 100 images with boxes drawn and
records a verdict. This script reads that verdict file and fails when it is missing,
so the build cannot pass by skipping the one check that catches coordinate-convention
errors no assertion catches.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    MANIFEST_DIR,
    REPORT_DIR,
    Logger,
    base_parser,
    class_names,
    detector_px,
    hamming,
    imread_gray,
    load_splits_config,
    parse_label_file,
    phash64,
    read_jsonl,
    require_dirs,
    run,
    sha256_file,
    write_json,
)

SCRIPT = "10_validate"
SPLITS = ("train", "val", "test")


@dataclass
class Gate:
    ident: str
    name: str
    threshold: str
    passed: bool | None  # None = skipped
    observed: str
    detail: str = ""

    @property
    def verdict(self) -> str:
        return "SKIP" if self.passed is None else ("PASS" if self.passed else "FAIL")


class Gates:
    def __init__(self) -> None:
        self.rows: list[Gate] = []

    def add(self, ident, name, threshold, passed, observed, detail="") -> None:
        self.rows.append(Gate(ident, name, threshold, None if passed is None else bool(passed), str(observed), detail))

    @property
    def failed(self) -> list[Gate]:
        return [g for g in self.rows if g.passed is False]

    @property
    def skipped(self) -> list[Gate]:
        return [g for g in self.rows if g.passed is None]

    def table(self) -> str:
        widths = [max(len(h), *(len(getattr(g, a)) for g in self.rows))
                  for h, a in (("GATE", "ident"), ("CHECK", "name"), ("THRESHOLD", "threshold"), ("OBSERVED", "observed"))]
        header = "  ".join(h.ljust(w) for h, w in zip(("GATE", "CHECK", "THRESHOLD", "OBSERVED"), widths)) + "  RESULT"
        lines = [header, "-" * len(header)]
        for g in self.rows:
            cells = (g.ident, g.name, g.threshold, g.observed)
            lines.append("  ".join(c.ljust(w) for c, w in zip(cells, widths)) + f"  {g.verdict}")
        return "\n".join(lines)


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def main() -> int:  # noqa: C901 - one block per gate keeps the table readable
    ap = base_parser(__doc__)
    ap.add_argument("--dataset", default=None, help="built dataset root (default: <processed>/yolo)")
    ap.add_argument("--hash-sample", type=int, default=6000, help="images hashed for G1 and G2")
    ap.add_argument("--ir-sample", type=int, default=500, help="IR images opened for G10")
    ap.add_argument("--plates", default=None, help="plate output dir (default: datasets/plates/out)")
    ap.add_argument("--skip-plates", action="store_true", help="G17 out of scope for this build: SKIP, never PASS")
    ap.add_argument("--verdict", default=None, help="G19 verdict file (default: <reports>/spotcheck/VERDICT.txt)")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    gates = Gates()
    rng = random.Random(args.seed)
    import cv2
    import numpy as np

    processed = Path(args.processed).resolve()
    dataset = Path(args.dataset).resolve() if args.dataset else processed / "yolo"
    plates_root = Path(args.plates).resolve() if args.plates else Path(__file__).resolve().parents[1] / "plates" / "out"
    manifests = [MANIFEST_DIR, dataset / "manifests"]
    config = load_splits_config()

    records = read_jsonl(dataset / "index" / "split.jsonl")
    if not records:
        log.error("built index missing - run 09_build_yolo_ds.py first", expected=str(dataset / "index" / "split.jsonl"))
        gates.add("G0", "built dataset present", "index exists", False, "missing")
        print(gates.table())
        return 1
    by_split: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    untiled = [r for r in records if not r.get("tiled")]

    # ---------------------------------------------------------------- G1, G2 perceptual hash
    sample = sorted(records, key=lambda r: r["image"])
    if len(sample) > args.hash_sample:
        rng.shuffle(sample)
        sample = sorted(sample[: args.hash_sample], key=lambda r: r["image"])
    hashes: dict[str, tuple[int, str, str]] = {}
    for record in sample:
        value = int(record["phash"], 16) if record.get("phash") else None
        if value is None:
            gray = imread_gray(dataset / record["final_image"])
            value = phash64(gray) if gray is not None else None
        if value is not None:
            hashes[record["final_image"]] = (value, record["split"], record.get("sequence_key") or "?")
    cross_split = within_split = 0
    buckets: dict[int, list[str]] = defaultdict(list)
    for image, (value, _s, _q) in sorted(hashes.items()):
        buckets[value >> 48].append(image)
    for members in buckets.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                distance = hamming(hashes[a][0], hashes[b][0])
                if distance <= 4 and hashes[a][1] != hashes[b][1]:
                    cross_split += 1
                elif distance <= 2 and hashes[a][1] == hashes[b][1] and hashes[a][2] != hashes[b][2]:
                    within_split += 1
    gates.add("G1", "no cross-split duplicates", "0 pairs @ d<=4", cross_split == 0, str(cross_split),
              f"{len(hashes)} images hashed")
    within_ratio = within_split / max(1, len(hashes))
    gates.add("G2", "within-split near-duplicates", "< 0.5%", within_ratio < 0.005, f"{within_ratio:.4%}")

    # ---------------------------------------------------------------- G3 sequence separation
    sequence_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.get("sequence_key"):
            sequence_splits[record["sequence_key"]].add(record["split"])
    leaked = {k: sorted(v) for k, v in sequence_splits.items() if len(v) > 1}
    gates.add("G3", "no leaked sequences", "0 multi-split keys", not leaked, str(len(leaked)),
              ", ".join(f"{k}->{v}" for k, v in sorted(leaked.items())[:5]))

    # ---------------------------------------------------------------- G4..G7, G11, G12 labels
    names = class_names()
    valid_ids = set(range(len(names)))
    dims = {r["final_label"]: (r.get("img_w", 0), r.get("img_h", 0)) for r in records}
    label_files = [p for s in SPLITS if (dataset / "labels" / s).exists() for p in sorted((dataset / "labels" / s).glob("*.txt"))]
    image_files = [p for s in SPLITS if (dataset / "images" / s).exists() for p in sorted((dataset / "images" / s).iterdir()) if p.is_file()]
    out_of_range = zero_area = bad_class = unreadable = 0
    class_counter: Counter = Counter()
    person_px: list[float] = []
    for label_path in label_files:
        rows, errors = parse_label_file(label_path)
        if errors:
            unreadable += 1
        width, height = dims.get(f"labels/{label_path.parent.name}/{label_path.name}", (0, 0))
        for class_id, cx, cy, w, h in rows:
            if class_id not in valid_ids:
                bad_class += 1
            else:
                class_counter[class_id] += 1
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                out_of_range += 1
            if w <= 0.0005 or h <= 0.0005:
                zero_area += 1
            if class_id == 0 and height:
                person_px.append(detector_px(h, width, height))
    gates.add("G4", "label values in [0,1]", "0 violations", out_of_range == 0, str(out_of_range))
    gates.add("G5", "no zero-area box", "0 boxes", zero_area == 0, str(zero_area))
    gates.add("G6", "class ids in schema", "0 out-of-schema", bad_class == 0, str(bad_class))
    label_keys = {(p.parent.name, p.stem) for p in label_files}
    image_keys = {(p.parent.name, p.stem) for p in image_files}
    orphans = len(label_keys ^ image_keys)
    gates.add("G7", "image/label parity", "0 orphans", orphans == 0 and unreadable == 0 and len(label_files) == len(records),
              f"{orphans} orphans, {unreadable} unreadable, {len(image_files)} images/{len(records)} records")

    # ---------------------------------------------------------------- G8 day/ir
    low = float(config["day_night"]["visible_share_min"])
    high = float(config["day_night"]["visible_share_max"])
    shares = {}
    for split in SPLITS:
        rows = by_split.get(split, [])
        if rows:
            shares[split] = sum(1 for r in rows if r.get("modality") == "visible") / len(rows)
    gates.add("G8", "day/IR ratio per split", f"{low:.2f}-{high:.2f} visible",
              bool(shares) and all(low <= v <= high for v in shares.values()) and len(shares) == 3,
              " ".join(f"{k}={v:.3f}" for k, v in shares.items()) or "n/a")

    # ---------------------------------------------------------------- G9 negatives
    train_labels = sorted((dataset / "labels" / "train").glob("*.txt")) if (dataset / "labels" / "train").exists() else []
    train_negatives = sum(1 for p in train_labels if p.stat().st_size == 0)
    negative_ratio = train_negatives / max(1, len(train_labels))
    gates.add("G9", "negative ratio in train", "0.08-0.12", 0.08 <= negative_ratio <= 0.12,
              f"{negative_ratio:.4f} ({train_negatives}/{len(train_labels)})")

    # ---------------------------------------------------------------- G10 IR is 3-channel, B=G=R
    ir_records = sorted((r for r in records if r.get("modality") == "lwir"), key=lambda r: r["final_image"])
    rng.shuffle(ir_records)
    not_three = checked = 0
    for record in ir_records[: args.ir_sample]:
        image = cv2.imread(str(dataset / record["final_image"]), cv2.IMREAD_UNCHANGED)
        checked += 1
        if (image is None or image.ndim != 3 or image.shape[2] != 3
                or not (np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(image[..., 1], image[..., 2]))):
            not_three += 1
    gates.add("G10", "IR frames 3-channel, B=G=R", "0 failures", not_three == 0 and checked > 0, f"{not_three} of {checked}")

    # ---------------------------------------------------------------- G11 class histogram
    exceptions_path = first_existing(*(m / "class_exceptions.txt" for m in manifests))
    exceptions = set()
    if exceptions_path:
        exceptions = {ln.strip() for ln in exceptions_path.read_text(encoding="utf-8").splitlines()
                      if ln.strip() and not ln.startswith("#")}
    total_instances = max(1, sum(class_counter.values()))
    starved = [names[i] for i in range(len(names))
               if class_counter.get(i, 0) / total_instances < 0.01 and names[i] not in exceptions]
    require_dirs(REPORT_DIR)
    write_json(REPORT_DIR / "class_histogram.json", {names[i]: class_counter.get(i, 0) for i in range(len(names))})
    gates.add("G11", "class histogram, no dead class", ">= 1% or documented", not starved,
              ",".join(starved) or "none starved", f"documented exceptions: {sorted(exceptions)}")

    # ---------------------------------------------------------------- G12 small-object tail
    small_share = sum(1 for p in person_px if p < 19) / len(person_px) if person_px else 0.0
    write_json(REPORT_DIR / "box_size_histogram.json",
               {"person_boxes": len(person_px), "share_below_19px_at_detector_input": round(small_share, 6)})
    gates.add("G12", "small-object tail present", ">= 5% below 19 px", small_share >= 0.05,
              f"{small_share:.2%} of {len(person_px)}")

    # ---------------------------------------------------------------- G13 tiles
    object_tiles = [r for r in records if r.get("tiled") and not r.get("is_negative")]
    wrong_size = 0
    for record in object_tiles[:200]:
        image = cv2.imread(str(dataset / record["final_image"]))
        if image is None or image.shape[:2] != (640, 640):
            wrong_size += 1
    gates.add("G13", "far-field tiles present", ">= 2000, all 640x640", len(object_tiles) >= 2000 and wrong_size == 0,
              f"{len(object_tiles)} tiles, {wrong_size} wrong size")

    # ---------------------------------------------------------------- G14 split proportions
    counts = Counter(r["split"] for r in untiled)
    total = max(1, sum(counts.values()))
    tolerance = float(config["proportions"]["tolerance"])
    proportion_ok = all(abs(counts.get(s, 0) / total - float(config["proportions"][s])) <= tolerance for s in SPLITS)
    gates.add("G14", "split proportions (untiled)", f"70/15/15 +/-{tolerance:.0%}", proportion_ok,
              "/".join(f"{counts.get(s, 0) / total:.3f}" for s in SPLITS))

    # ---------------------------------------------------------------- G15 hard set
    hard_path = first_existing(*(m / "hard_set.txt" for m in manifests))
    entries = [ln.strip() for ln in hard_path.read_text(encoding="utf-8").splitlines() if ln.strip()] if hard_path else []
    split_of = {r["image"]: r["split"] for r in records}
    contamination = sum(1 for e in entries if split_of.get(e) == "test")
    not_val = sum(1 for e in entries if split_of.get(e) != "val")
    expected_hard = int(config["hard_set"]["total"])
    gates.add("G15", "hard set frozen and clean", f"{expected_hard} val entries, 0 from test",
              len(entries) == expected_hard and contamination == 0 and not_val == 0,
              f"{len(entries)} entries, {contamination} from test, {not_val} not in built val")

    # ---------------------------------------------------------------- G16 determinism
    splits_json = first_existing(processed / "splits.json", dataset / "manifests" / "splits.json")
    recorded = first_existing(processed / "reports" / "split.json", dataset / "manifests" / "split_report.json")
    determinism_ok = False
    observed_digest = "missing"
    if splits_json and recorded:
        observed_digest = sha256_file(splits_json)[:16]
        determinism_ok = json.loads(recorded.read_text(encoding="utf-8")).get("sha256", "").startswith(observed_digest)
    gates.add("G16", "split determinism", "sha256 matches record", determinism_ok, observed_digest)

    # ---------------------------------------------------------------- G17 plates
    if args.skip_plates:
        gates.add("G17", "plate corpus", ">= 20000, labels match, 0 off-charset", None, "out of scope (--skip-plates)")
    else:
        plate_images = sorted((plates_root / "images").glob("*.jpg")) if (plates_root / "images").exists() else []
        labels_dir = Path(__file__).resolve().parents[1] / "plates" / "labels"
        label_lines = charset_violations = 0
        charset_path = labels_dir / "charset.txt"
        charset = set(charset_path.read_text(encoding="utf-8").split()) if charset_path.exists() else set()
        for name in ("rec_gt_train.txt", "rec_gt_val.txt"):
            path = labels_dir / name
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                label_lines += 1
                parts = line.split("\t")
                if len(parts) == 2 and charset:
                    charset_violations += sum(1 for ch in parts[1] if ch not in charset)
        gates.add("G17", "plate corpus", ">= 20000, labels match, 0 off-charset",
                  len(plate_images) >= 20000 and label_lines == len(plate_images) and charset_violations == 0,
                  f"{len(plate_images)} images, {label_lines} labels, {charset_violations} off-charset")

    # ---------------------------------------------------------------- G18 nothing staged
    data_suffixes = (".jpg", ".jpeg", ".png", ".mp4", ".zip", ".tar", ".pt", ".onnx")
    repo = Path(__file__).resolve().parents[2]
    staged_data = tracked_data = 0
    try:
        staged = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo), capture_output=True, text=True, check=False)
        for line in staged.stdout.splitlines():
            path = line[3:].strip().strip('"')
            if path.startswith("datasets/") and Path(path).suffix.lower() in data_suffixes:
                staged_data += 1
        tracked = subprocess.run(["git", "ls-files", "datasets"], cwd=str(repo), capture_output=True, text=True, check=False)
        tracked_data = sum(1 for p in tracked.stdout.splitlines() if Path(p).suffix.lower() in data_suffixes)
    except OSError:
        staged_data = -1
    gates.add("G18", "no data staged or tracked", "0 files", staged_data == 0 and tracked_data == 0,
              f"{staged_data} staged, {tracked_data} tracked")

    # ---------------------------------------------------------------- G19 manual spot check
    verdict_path = Path(args.verdict) if args.verdict else REPORT_DIR / "spotcheck" / "VERDICT.txt"
    verdict_text = verdict_path.read_text(encoding="utf-8").strip() if verdict_path.exists() else ""
    gates.add("G19", "manual 100-image spot check", "VERDICT.txt says PASS", verdict_text.upper().startswith("PASS"),
              verdict_text.splitlines()[0][:40] if verdict_text else "missing", f"write PASS or FAIL to {verdict_path}")

    # ---------------------------------------------------------------- report
    print()
    print(gates.table())
    print()
    write_json(REPORT_DIR / "validation.json", {
        "dataset": str(dataset),
        "gates": [{"id": g.ident, "check": g.name, "threshold": g.threshold, "observed": g.observed,
                   "result": g.verdict, "passed": g.passed, "detail": g.detail} for g in gates.rows],
        "failed": [g.ident for g in gates.failed],
        "skipped": [g.ident for g in gates.skipped],
    })
    for gate in gates.failed:
        log.error("gate failed", gate=gate.ident, check=gate.name, observed=gate.observed, detail=gate.detail)
    for gate in gates.skipped:
        log.warn("gate skipped - not a pass", gate=gate.ident, check=gate.name, reason=gate.observed)
    if gates.failed:
        log.error("VALIDATION FAILED", failed=len(gates.failed), of=len(gates.rows))
        return 1
    log.info("VALIDATION PASSED", gates=len(gates.rows), skipped=len(gates.skipped))
    return 0


if __name__ == "__main__":
    run(main)
