#!/usr/bin/env python3
"""Corpus statistics and the figures the gate reads. DATASET_SPEC.md section 9.

Reads the BUILT dataset (<dataset>/index/split.jsonl and the final label files) and writes
<reports>/stats.json plus four PNG charts: images per split x modality, instances per
split x modality x class, visible share, negatives share, the person-height histogram at the
detector input, the cart-gate decision and the output size. The charts are drawn with
OpenCV rather than a plotting library so the dependency list stays at six packages
and nothing here needs a display.

Also writes the 100-image visual spot check that gate G19 requires a human to look at:
boxes drawn, class names written, one file per image under reports/spotcheck/.
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
    REPORT_DIR,
    Counters,
    Logger,
    base_parser,
    class_names,
    detector_px,
    imread_any,
    imwrite,
    parse_label_file,
    read_json,
    read_jsonl,
    require_dirs,
    run,
    write_json,
)

SCRIPT = "11_stats"

PALETTE = [
    (64, 188, 255),
    (255, 176, 64),
    (120, 220, 120),
    (200, 120, 255),
    (255, 120, 120),
]


def bar_chart(path: Path, title: str, labels: list[str], values: list[float], log: Logger) -> None:
    import cv2
    import numpy as np

    width, height = 1000, 520
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.putText(canvas, title, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)

    if not values or max(values) <= 0:
        cv2.putText(canvas, "no data", (24, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2, cv2.LINE_AA)
        imwrite(path, canvas)
        return

    top, bottom, left, right = 80, height - 70, 70, width - 30
    scale = (bottom - top) / float(max(values))
    slot = (right - left) / float(len(values))
    bar_width = max(6, int(slot * 0.62))

    cv2.line(canvas, (left, bottom), (right, bottom), (90, 90, 90), 1, cv2.LINE_AA)
    for index, (label, value) in enumerate(zip(labels, values)):
        x = int(left + slot * index + (slot - bar_width) / 2)
        bar_height = int(value * scale)
        colour = PALETTE[index % len(PALETTE)]
        cv2.rectangle(canvas, (x, bottom - bar_height), (x + bar_width, bottom), colour, -1)
        cv2.putText(
            canvas, label[:14], (x, bottom + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"{value:g}",
            (x, bottom - bar_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    imwrite(path, canvas)
    log.info("chart written", path=str(path))


def main() -> int:  # noqa: C901
    ap = base_parser(__doc__)
    ap.add_argument("--dataset", default=None, help="built dataset root (default: <processed>/yolo)")
    ap.add_argument("--spotcheck", type=int, default=100, help="images to render for the G19 manual check")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()
    rng = random.Random(args.seed)

    processed = Path(args.processed).resolve()
    dataset = Path(args.dataset).resolve() if args.dataset else processed / "yolo"
    records = read_jsonl(dataset / "index" / "split.jsonl")
    if not records:
        log.error("built index missing - run 09_build_yolo_ds.py first", expected=str(dataset / "index" / "split.jsonl"))
        return 1

    names = class_names()
    bin_edges = [0, 9, 14, 19, 27, 38, 54, 77, 10_000]
    size_bins: Counter = Counter()
    class_counter: Counter = Counter()
    cube: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))  # split -> modality -> class
    images: dict[str, Counter] = defaultdict(Counter)                                # split -> modality
    by_source: dict[str, Counter] = defaultdict(Counter)                             # split -> source
    negatives: Counter = Counter()
    tiles: Counter = Counter()

    for record in records:
        split, modality = record["split"], record.get("modality", "unknown")
        images[split][modality] += 1
        by_source[split][record.get("source", "?")] += 1
        if record.get("tiled"):
            tiles["negative" if record.get("is_negative") else "object"] += 1
        label_path = dataset / record["final_label"]
        if not label_path.exists():
            counters.bump("label.missing")
            continue
        rows, errors = parse_label_file(label_path)
        for message in errors:
            counters.bump("label.row_error")
            log.warn("label row error", label=label_path.name, detail=message)
        if not rows:
            negatives[split] += 1
            continue
        for class_id, _cx, _cy, _w, h in rows:
            name = names[class_id] if 0 <= class_id < len(names) else f"id{class_id}"
            class_counter[name] += 1
            cube[split][modality][name] += 1
            if class_id == 0:
                pixels = detector_px(h, record.get("img_w", 0), record.get("img_h", 0))
                for index in range(len(bin_edges) - 1):
                    if bin_edges[index] <= pixels < bin_edges[index + 1]:
                        size_bins[f"{bin_edges[index]}-{bin_edges[index + 1]}px"] += 1
                        break

    untiled = Counter(r["split"] for r in records if not r.get("tiled"))
    total_untiled = max(1, sum(untiled.values()))
    gate = read_json(first_manifest(dataset, "cart_gate.json"), default={}) or {}
    size_bytes = sum(p.stat().st_size for p in (dataset / "images").rglob("*") if p.is_file())
    stats = {
        "images_total": len(records),
        "by_split": {s: sum(images[s].values()) for s in sorted(images)},
        "by_split_modality": {s: dict(sorted(images[s].items())) for s in sorted(images)},
        "by_split_source": {s: dict(sorted(by_source[s].items())) for s in sorted(by_source)},
        "untiled_split_proportions": {k: round(v / total_untiled, 4) for k, v in sorted(untiled.items())},
        "instances_by_split_modality_class": {
            s: {m: {n: cube[s][m].get(n, 0) for n in names} for m in sorted(cube[s])} for s in sorted(cube)
        },
        "class_histogram": {n: class_counter.get(n, 0) for n in names},
        "class_share": {n: round(class_counter.get(n, 0) / max(1, sum(class_counter.values())), 4) for n in names},
        "person_box_height_histogram_px_at_640": dict(sorted(size_bins.items(), key=lambda kv: int(kv[0].split("-")[0]))),
        "visible_share": {s: round(images[s].get("visible", 0) / max(1, sum(images[s].values())), 4) for s in sorted(images)},
        "negatives_by_split": dict(sorted(negatives.items())),
        "negatives_share_train": round(negatives.get("train", 0) / max(1, sum(images["train"].values())), 4),
        "tiles": dict(tiles),
        "cart_gate": gate,
        "images_bytes": size_bytes,
        "counters": counters.as_dict(),
    }

    if args.dry_run:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    require_dirs(REPORT_DIR)
    write_json(REPORT_DIR / "stats.json", stats)
    bar_chart(REPORT_DIR / "class_histogram.png", "instances per class", names,
              [float(class_counter.get(n, 0)) for n in names], log)
    bar_chart(REPORT_DIR / "box_size_histogram.png", "person box height at the 640 px detector input",
              list(stats["person_box_height_histogram_px_at_640"].keys()),
              [float(v) for v in stats["person_box_height_histogram_px_at_640"].values()], log)
    bar_chart(REPORT_DIR / "day_night.png", "visible share per split", list(stats["visible_share"].keys()),
              [float(v) for v in stats["visible_share"].values()], log)
    bar_chart(REPORT_DIR / "splits.png", "images per split", list(stats["by_split"].keys()),
              [float(v) for v in stats["by_split"].values()], log)

    render_spotcheck(records, dataset, args.spotcheck, rng, names, log)

    print_summary(stats, names)
    counters.report(log)
    log.info("done", reports=str(REPORT_DIR))
    return 0


def first_manifest(dataset: Path, name: str) -> Path:
    for candidate in (MANIFEST_DIR / name, dataset / "manifests" / name):
        if candidate.exists():
            return candidate
    return MANIFEST_DIR / name


def print_summary(stats: dict, names: list[str]) -> None:
    print("\ninstances per split x modality x class", flush=True)
    header = f"  {'split':<6} {'modality':<8} {'images':>8} " + " ".join(f"{n[:11]:>11}" for n in names)
    print(header)
    for split, modalities in stats["by_split_modality"].items():
        for modality, count in modalities.items():
            row = stats["instances_by_split_modality_class"].get(split, {}).get(modality, {})
            print(f"  {split:<6} {modality:<8} {count:>8} " + " ".join(f"{row.get(n, 0):>11}" for n in names))
    print(f"\nvisible share: {stats['visible_share']}")
    print(f"negatives share (train): {stats['negatives_share_train']}  negatives by split: {stats['negatives_by_split']}")
    print(f"untiled split proportions: {stats['untiled_split_proportions']}  tiles: {stats['tiles']}")
    gate = stats.get("cart_gate") or {}
    print(f"cart gate: {gate.get('decision', 'unknown')} - {gate.get('reason', '')}")
    print(f"image bytes: {stats['images_bytes'] / 1024**3:.2f} GB\n", flush=True)


def render_spotcheck(records: list[dict], dataset: Path, count: int, rng: random.Random, names: list[str],
                     log: Logger) -> None:
    """Gate G19: draw boxes on 100 random FINAL training images (tiles included), read back from
    the built dataset, so the check covers every transform the build applied."""
    import cv2

    out_dir = REPORT_DIR / "spotcheck"
    require_dirs(out_dir)
    for old in out_dir.glob("spot_*.jpg"):
        old.unlink()
    train = sorted((r for r in records if r["split"] == "train" and r.get("label_rows", 0) > 0),
                   key=lambda r: r["final_image"])
    if not train:
        log.warn("no positive training frames to spot check")
        return
    rng.shuffle(train)
    drawn = 0
    listing = []
    for record in train:
        if drawn >= count:
            break
        image = imread_any(dataset / record["final_image"])
        if image is None:
            continue
        height, width = image.shape[:2]
        rows, _errors = parse_label_file(dataset / record["final_label"])
        for class_id, cx, cy, w, h in rows:
            x1, y1 = int((cx - w / 2) * width), int((cy - h / 2) * height)
            x2, y2 = int((cx + w / 2) * width), int((cy + h / 2) * height)
            colour = PALETTE[class_id % len(PALETTE)]
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
            label = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        imwrite(out_dir / f"spot_{drawn:03d}.jpg", image)
        listing.append(f"spot_{drawn:03d}.jpg\t{record['final_image']}\t{record.get('source')}\t{record.get('modality')}")
        drawn += 1
    (out_dir / "index.tsv").write_text("\n".join(listing) + "\n", encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "Gate G19, DATASET_SPEC.md section 9.\n\n"
        f"{drawn} final training images (index.tsv maps each to its dataset file) with their boxes\n"
        "drawn. Open them. Check that boxes sit on objects, that class names are right, and that\n"
        "nothing is shifted or mirrored - a coordinate-convention error looks exactly like this and\n"
        "no assertion catches it.\n\n"
        "Then write PASS or FAIL, plus a line saying what you saw, to VERDICT.txt in this\n"
        "directory. 10_validate.py reads that file and fails when it is missing.\n",
        encoding="utf-8",
    )
    log.info("spot check rendered", images=drawn, path=str(out_dir), next="write VERDICT.txt after looking")


if __name__ == "__main__":
    run(main)
