#!/usr/bin/env python3
"""The gate. Every check in DATASET_SPEC.md section 9, with its numeric threshold.

Prints a PASS/FAIL table and exits 1 if any gate fails. Phase 2 does not start until
this exits 0.

G19 is deliberately not automatable: a human opens 100 images with boxes drawn and
records a verdict. This script reads that verdict file and fails when it is missing,
so the build cannot pass by skipping the one check that catches coordinate-convention
errors no assertion catches.
"""

from __future__ import annotations

import json
import random
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
    hamming,
    imread_gray,
    load_splits_config,
    parse_label_file,
    phash64,
    require_dirs,
    run,
    sha256_file,
    write_json,
)

SCRIPT = "10_validate"


@dataclass
class Gate:
    ident: str
    name: str
    threshold: str
    passed: bool
    observed: str
    detail: str = ""


class Gates:
    def __init__(self) -> None:
        self.rows: list[Gate] = []

    def add(self, ident: str, name: str, threshold: str, passed: bool, observed: str, detail: str = "") -> None:
        self.rows.append(Gate(ident, name, threshold, bool(passed), str(observed), detail))

    @property
    def failed(self) -> list[Gate]:
        return [g for g in self.rows if not g.passed]

    def table(self) -> str:
        w_id = max(4, max(len(g.ident) for g in self.rows))
        w_name = max(4, max(len(g.name) for g in self.rows))
        w_thr = max(9, max(len(g.threshold) for g in self.rows))
        w_obs = max(8, max(len(g.observed) for g in self.rows))
        header = (
            f"{'GATE':<{w_id}}  {'CHECK':<{w_name}}  {'THRESHOLD':<{w_thr}}  "
            f"{'OBSERVED':<{w_obs}}  RESULT"
        )
        lines = [header, "-" * len(header)]
        for gate in self.rows:
            lines.append(
                f"{gate.ident:<{w_id}}  {gate.name:<{w_name}}  {gate.threshold:<{w_thr}}  "
                f"{gate.observed:<{w_obs}}  {'PASS' if gate.passed else 'FAIL'}"
            )
        return "\n".join(lines)


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:  # noqa: C901 - one function per gate would hide the table
    ap = base_parser(__doc__)
    ap.add_argument("--dataset", default=None, help="built dataset root (default: <processed>/yolo)")
    ap.add_argument("--hash-sample", type=int, default=6000, help="images hashed for G1 and G2")
    ap.add_argument("--ir-sample", type=int, default=500, help="IR images opened for G10")
    ap.add_argument("--plates", default=None, help="plate output dir (default: datasets/plates/out)")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    gates = Gates()
    rng = random.Random(args.seed)

    processed = Path(args.processed).resolve()
    dataset = Path(args.dataset).resolve() if args.dataset else processed / "yolo"
    plates_root = Path(args.plates).resolve() if args.plates else Path(__file__).resolve().parents[1] / "plates" / "out"
    config = load_splits_config()

    index_path = processed / "index" / "with_negatives.jsonl"
    records: list[dict] = load_records(index_path) if index_path.exists() else []
    tiles_path = processed / "index" / "tiles.jsonl"
    tiles: list[dict] = load_records(tiles_path) if tiles_path.exists() else []

    if not records:
        log.error("no pipeline index found - run the pipeline before the gate", expected=str(index_path))
        gates.add("G0", "pipeline output present", "index exists", False, "missing")
        print(gates.table())
        return 1

    all_records = records + tiles
    by_split: dict[str, list[dict]] = defaultdict(list)
    for record in all_records:
        if record.get("split"):
            by_split[record["split"]].append(record)

    # ---------------------------------------------------------------- G1, G2 perceptual hash
    sample = sorted(all_records, key=lambda r: r["image"])
    if len(sample) > args.hash_sample:
        rng.shuffle(sample)
        sample = sorted(sample[: args.hash_sample], key=lambda r: r["image"])

    hashes: dict[str, tuple[int, str, str]] = {}
    for record in sample:
        digest = record.get("phash")
        value = None
        if digest:
            value = int(digest, 16)
        else:
            gray = imread_gray(Path(record["image"]))
            if gray is not None:
                value = phash64(gray)
        if value is None:
            continue
        hashes[record["image"]] = (value, record.get("split", "?"), record.get("sequence_key") or "?")

    cross_split = 0
    within_split = 0
    items = sorted(hashes.items())
    buckets: dict[int, list[str]] = defaultdict(list)
    for image, (value, _split, _seq) in items:
        buckets[value >> 48].append(image)
    for _band, members in buckets.items():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                distance = hamming(hashes[a][0], hashes[b][0])
                if distance <= 4 and hashes[a][1] != hashes[b][1]:
                    cross_split += 1
                elif distance <= 2 and hashes[a][1] == hashes[b][1] and hashes[a][2] != hashes[b][2]:
                    within_split += 1

    gates.add("G1", "no cross-split duplicates", "0 pairs @ d<=4", cross_split == 0, str(cross_split))
    within_ratio = within_split / max(1, len(hashes))
    gates.add(
        "G2",
        "within-split near-duplicates",
        "< 0.5%",
        within_ratio < 0.005,
        f"{within_ratio:.4%}",
    )

    # ---------------------------------------------------------------- G3 sequence separation
    sequence_splits: dict[str, set[str]] = defaultdict(set)
    for record in all_records:
        key = record.get("sequence_key")
        if key and record.get("split"):
            sequence_splits[key].add(record["split"])
    leaked = {k: sorted(v) for k, v in sequence_splits.items() if len(v) > 1}
    gates.add(
        "G3",
        "no leaked sequences",
        "0 multi-split keys",
        len(leaked) == 0,
        str(len(leaked)),
        ", ".join(f"{k}->{v}" for k, v in sorted(leaked.items())[:5]),
    )

    # ---------------------------------------------------------------- G4..G7 label integrity
    label_dirs = [dataset / "labels" / s for s in ("train", "val", "test")]
    label_files = [p for d in label_dirs if d.exists() for p in sorted(d.glob("*.txt"))]
    image_dirs = [dataset / "images" / s for s in ("train", "val", "test")]
    image_files = [p for d in image_dirs if d.exists() for p in sorted(d.iterdir()) if p.is_file()]

    out_of_range = 0
    zero_area = 0
    bad_class = 0
    unreadable = 0
    class_counter: Counter = Counter()
    height_samples: list[float] = []
    valid_ids = set(range(len(class_names())))
    negatives = 0

    for label_path in label_files:
        rows, errors = parse_label_file(label_path)
        if errors:
            unreadable += 1
        if not rows:
            negatives += 1
        for class_id, cx, cy, w, h in rows:
            if class_id not in valid_ids:
                bad_class += 1
            else:
                class_counter[class_id] += 1
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                out_of_range += 1
            if w <= 0.0005 or h <= 0.0005:
                zero_area += 1
            if class_id == 0:
                height_samples.append(h)

    gates.add("G4", "label values in [0,1]", "0 violations", out_of_range == 0, str(out_of_range))
    gates.add("G5", "no zero-area box", "0 boxes", zero_area == 0, str(zero_area))
    gates.add("G6", "class ids in schema", "0 out-of-schema", bad_class == 0, str(bad_class))

    label_stems = {p.stem for p in label_files}
    image_stems = {p.stem for p in image_files}
    orphans = len(label_stems ^ image_stems)
    gates.add(
        "G7",
        "image/label parity",
        "0 orphans",
        orphans == 0 and unreadable == 0,
        f"{orphans} orphans, {unreadable} unreadable",
    )

    # ---------------------------------------------------------------- G8 day/night
    day_night_ok = True
    observed_ratios = []
    low = float(config["day_night"]["visible_share_min"])
    high = float(config["day_night"]["visible_share_max"])
    for split in ("train", "val", "test"):
        split_records = by_split.get(split, [])
        if not split_records:
            continue
        visible = sum(1 for r in split_records if r.get("modality") == "visible")
        share = visible / len(split_records)
        observed_ratios.append(f"{split}={share:.2f}")
        if not low <= share <= high:
            day_night_ok = False
    gates.add("G8", "day/night ratio", f"{low:.2f}-{high:.2f} visible", day_night_ok, " ".join(observed_ratios) or "n/a")

    # ---------------------------------------------------------------- G9 negatives
    train_labels = sorted((dataset / "labels" / "train").glob("*.txt")) if (dataset / "labels" / "train").exists() else []
    train_negatives = sum(1 for p in train_labels if p.stat().st_size == 0)
    negative_ratio = train_negatives / max(1, len(train_labels))
    gates.add(
        "G9",
        "negative ratio in train",
        "0.08-0.12",
        0.08 <= negative_ratio <= 0.12,
        f"{negative_ratio:.4f}",
    )

    # ---------------------------------------------------------------- G10 IR is 3-channel
    import cv2

    ir_records = [r for r in all_records if r.get("modality") == "lwir"]
    rng.shuffle(ir_records)
    ir_sample = ir_records[: args.ir_sample]
    not_three = 0
    checked = 0
    for record in ir_sample:
        path = Path(record["image"])
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        checked += 1
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            not_three += 1
    gates.add(
        "G10",
        "IR frames are 3-channel",
        "0 failures",
        not_three == 0 and checked > 0,
        f"{not_three} of {checked}",
    )

    # ---------------------------------------------------------------- G11 class histogram
    names = class_names()
    total_instances = max(1, sum(class_counter.values()))
    exceptions_path = MANIFEST_DIR / "class_exceptions.txt"
    exceptions = set()
    if exceptions_path.exists():
        exceptions = {line.strip() for line in exceptions_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    starved = [
        names[i]
        for i in range(len(names))
        if class_counter.get(i, 0) / total_instances < 0.01 and names[i] not in exceptions
    ]
    histogram_path = REPORT_DIR / "class_histogram.json"
    write_json(histogram_path, {names[i]: class_counter.get(i, 0) for i in range(len(names))})
    gates.add(
        "G11",
        "class histogram, no dead class",
        ">= 1% or documented",
        len(starved) == 0,
        ",".join(starved) or "none starved",
    )

    # ---------------------------------------------------------------- G12 small-object tail
    small_share = 0.0
    if height_samples:
        # h is normalised; the dataset caps the long side at 1280 and tiles are 640.
        small = sum(1 for h in height_samples if h * 640 < 19)
        small_share = small / len(height_samples)
    write_json(
        REPORT_DIR / "box_size_histogram.json",
        {"person_boxes": len(height_samples), "share_below_19px": round(small_share, 6)},
    )
    gates.add(
        "G12",
        "small-object tail present",
        ">= 5% below 19 px",
        small_share >= 0.05,
        f"{small_share:.2%}",
    )

    # ---------------------------------------------------------------- G13 tiles
    tile_images = sorted((processed / "tiles" / "images").glob("*")) if (processed / "tiles" / "images").exists() else []
    wrong_size = 0
    for path in tile_images[:200]:
        image = cv2.imread(str(path))
        if image is None or image.shape[0] != 640 or image.shape[1] != 640:
            wrong_size += 1
    gates.add(
        "G13",
        "far-field tiles present",
        ">= 2000, all 640x640",
        len(tile_images) >= 2000 and wrong_size == 0,
        f"{len(tile_images)} tiles, {wrong_size} wrong size",
    )

    # ---------------------------------------------------------------- G14 split proportions
    counts = {s: len(by_split.get(s, [])) for s in ("train", "val", "test")}
    total = max(1, sum(counts.values()))
    tolerance = float(config["proportions"]["tolerance"])
    proportion_ok = all(
        abs(counts[s] / total - float(config["proportions"][s])) <= tolerance for s in ("train", "val", "test")
    )
    gates.add(
        "G14",
        "split proportions",
        f"70/15/15 +/-{tolerance:.0%}",
        proportion_ok,
        "/".join(f"{counts[s] / total:.2f}" for s in ("train", "val", "test")),
    )

    # ---------------------------------------------------------------- G15 hard set
    hard_path = MANIFEST_DIR / "hard_set.txt"
    hard_entries = (
        [line.strip() for line in hard_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if hard_path.exists()
        else []
    )
    test_images = {r["image"] for r in by_split.get("test", [])}
    contamination = sum(1 for entry in hard_entries if entry in test_images)
    expected_hard = int(config["hard_set"]["total"])
    gates.add(
        "G15",
        "hard set frozen and clean",
        f"{expected_hard} entries, 0 from test",
        len(hard_entries) == expected_hard and contamination == 0,
        f"{len(hard_entries)} entries, {contamination} from test",
    )

    # ---------------------------------------------------------------- G16 determinism
    splits_json = processed / "splits.json"
    recorded = (processed / "reports" / "split.json")
    determinism_ok = False
    observed_digest = "missing"
    if splits_json.exists():
        observed_digest = sha256_file(splits_json)[:16]
        if recorded.exists():
            expected = json.loads(recorded.read_text(encoding="utf-8")).get("sha256", "")
            determinism_ok = expected.startswith(observed_digest)
    gates.add("G16", "split determinism", "sha256 matches record", determinism_ok, observed_digest)

    # ---------------------------------------------------------------- G17 plates
    plate_images = sorted((plates_root / "images").glob("*.jpg")) if (plates_root / "images").exists() else []
    labels_dir = Path(__file__).resolve().parents[1] / "plates" / "labels"
    label_lines = 0
    charset_violations = 0
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
    gates.add(
        "G17",
        "plate corpus",
        ">= 20000, labels match, 0 off-charset",
        len(plate_images) >= 20000 and label_lines == len(plate_images) and charset_violations == 0,
        f"{len(plate_images)} images, {label_lines} labels, {charset_violations} off-charset",
    )

    # ---------------------------------------------------------------- G18 nothing staged
    import subprocess

    # Staged AND tracked. Checking only the staging area misses the failure this gate
    # exists to catch: data that was committed in an earlier commit is invisible to
    # `git status` and stays in the history forever.
    data_suffixes = (".jpg", ".jpeg", ".png", ".mp4", ".zip", ".tar", ".pt", ".onnx")
    repo = Path(__file__).resolve().parents[2]
    staged_data = 0
    tracked_data = 0
    try:
        staged = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(repo), capture_output=True, text=True, check=False
        )
        for line in staged.stdout.splitlines():
            path = line[3:].strip().strip('"')
            if path.startswith("datasets/") and Path(path).suffix.lower() in data_suffixes:
                staged_data += 1
        tracked = subprocess.run(
            ["git", "ls-files", "datasets"], cwd=str(repo), capture_output=True, text=True, check=False
        )
        for path in tracked.stdout.splitlines():
            if Path(path).suffix.lower() in data_suffixes:
                tracked_data += 1
    except OSError:
        staged_data = -1
    gates.add(
        "G18",
        "no data staged or tracked",
        "0 files",
        staged_data == 0 and tracked_data == 0,
        f"{staged_data} staged, {tracked_data} tracked",
    )

    # ---------------------------------------------------------------- G19 manual spot check
    verdict_path = REPORT_DIR / "spotcheck" / "VERDICT.txt"
    verdict_text = verdict_path.read_text(encoding="utf-8").strip().upper() if verdict_path.exists() else ""
    gates.add(
        "G19",
        "manual 100-image spot check",
        "VERDICT.txt says PASS",
        verdict_text.startswith("PASS"),
        verdict_text.splitlines()[0] if verdict_text else "missing",
        f"write PASS or FAIL to {verdict_path}",
    )

    # ---------------------------------------------------------------- report
    require_dirs(REPORT_DIR)
    print()
    print(gates.table())
    print()

    write_json(
        REPORT_DIR / "validation.json",
        {
            "gates": [
                {
                    "id": g.ident,
                    "check": g.name,
                    "threshold": g.threshold,
                    "observed": g.observed,
                    "passed": g.passed,
                    "detail": g.detail,
                }
                for g in gates.rows
            ],
            "failed": len(gates.failed),
        },
    )

    for gate in gates.failed:
        log.error("gate failed", gate=gate.ident, check=gate.name, observed=gate.observed, detail=gate.detail)

    if gates.failed:
        log.error("VALIDATION FAILED", failed=len(gates.failed), of=len(gates.rows))
        return 1
    log.info("VALIDATION PASSED", gates=len(gates.rows))
    return 0


if __name__ == "__main__":
    run(main)
