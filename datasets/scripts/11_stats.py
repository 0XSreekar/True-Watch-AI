#!/usr/bin/env python3
"""Corpus statistics and the figures the gate reads. DATASET_SPEC.md section 9.

Writes datasets/reports/stats.json plus four PNG charts. The charts are drawn with
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
    REPORT_DIR,
    Counters,
    Logger,
    base_parser,
    class_names,
    imread_any,
    imwrite,
    parse_label_file,
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


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--dataset", default=None, help="built dataset root (default: <processed>/yolo)")
    ap.add_argument("--spotcheck", type=int, default=100, help="images to render for the G19 manual check")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()
    rng = random.Random(args.seed)

    processed = Path(args.processed).resolve()
    dataset = Path(args.dataset).resolve() if args.dataset else processed / "yolo"
    index_path = processed / "index" / "with_negatives.jsonl"

    if not index_path.exists():
        log.error("pipeline index missing - run the pipeline first", expected=str(index_path))
        return 1

    records = []
    with index_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                records.append(json.loads(line))

    tiles_path = processed / "index" / "tiles.jsonl"
    if tiles_path.exists():
        with tiles_path.open(encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if line:
                    records.append(json.loads(line))

    names = class_names()
    class_counter: Counter = Counter()
    size_bins = Counter()
    bin_edges = [0, 9, 14, 19, 27, 38, 54, 77, 10_000]
    modality_by_split: dict[str, Counter] = defaultdict(Counter)
    split_counter: Counter = Counter()
    negatives = 0
    labels_read = 0

    for record in records:
        split = record.get("split")
        if not split:
            continue
        split_counter[split] += 1
        modality_by_split[split][record.get("modality", "unknown")] += 1

        label_path = Path(record.get("label", ""))
        if not label_path.exists():
            counters.bump("label.missing")
            continue
        rows, errors = parse_label_file(label_path)
        labels_read += 1
        for message in errors:
            counters.bump("label.row_error")
            log.warn("label row error", label=label_path.name, detail=message)
        if not rows:
            negatives += 1
            continue
        height_px = int(record.get("img_h") or 640)
        for class_id, _cx, _cy, _w, h in rows:
            if 0 <= class_id < len(names):
                class_counter[names[class_id]] += 1
            if class_id == 0:
                pixels = h * height_px
                for index in range(len(bin_edges) - 1):
                    if bin_edges[index] <= pixels < bin_edges[index + 1]:
                        size_bins[f"{bin_edges[index]}-{bin_edges[index + 1]}px"] += 1
                        break

    total_images = max(1, sum(split_counter.values()))
    train_total = max(1, split_counter.get("train", 0))
    stats = {
        "images_total": sum(split_counter.values()),
        "labels_read": labels_read,
        "by_split": dict(sorted(split_counter.items())),
        "split_proportions": {k: round(v / total_images, 4) for k, v in sorted(split_counter.items())},
        "class_histogram": dict(sorted(class_counter.items())),
        "class_share": {
            k: round(v / max(1, sum(class_counter.values())), 4) for k, v in sorted(class_counter.items())
        },
        "person_box_height_histogram_px": dict(sorted(size_bins.items())),
        "day_night": {
            split: {
                "visible": counter.get("visible", 0),
                "lwir": counter.get("lwir", 0),
                "visible_share": round(counter.get("visible", 0) / max(1, sum(counter.values())), 4),
            }
            for split, counter in sorted(modality_by_split.items())
        },
        "negatives": negatives,
        "negatives_ratio_overall": round(negatives / total_images, 4),
        "negatives_ratio_train_estimate": round(negatives / train_total, 4),
        "counters": counters.as_dict(),
    }

    if args.dry_run:
        log.info("dry-run: nothing written")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    require_dirs(REPORT_DIR)
    write_json(REPORT_DIR / "stats.json", stats)

    bar_chart(
        REPORT_DIR / "class_histogram.png",
        "instances per class",
        list(stats["class_histogram"].keys()),
        [float(v) for v in stats["class_histogram"].values()],
        log,
    )
    bar_chart(
        REPORT_DIR / "box_size_histogram.png",
        "person box height, px",
        list(stats["person_box_height_histogram_px"].keys()),
        [float(v) for v in stats["person_box_height_histogram_px"].values()],
        log,
    )
    bar_chart(
        REPORT_DIR / "day_night.png",
        "visible share per split",
        list(stats["day_night"].keys()),
        [float(v["visible_share"]) for v in stats["day_night"].values()],
        log,
    )
    bar_chart(
        REPORT_DIR / "splits.png",
        "images per split",
        list(stats["by_split"].keys()),
        [float(v) for v in stats["by_split"].values()],
        log,
    )

    render_spotcheck(records, args.spotcheck, rng, names, log)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    counters.report(log)
    log.info("done", reports=str(REPORT_DIR))
    return 0


def render_spotcheck(records: list[dict], count: int, rng: random.Random, names: list[str], log: Logger) -> None:
    """Gate G19: draw boxes on 100 random training images so a human can actually look."""
    import cv2

    out_dir = REPORT_DIR / "spotcheck"
    require_dirs(out_dir)
    train = [r for r in records if r.get("split") == "train" and r.get("objects", 0) > 0]
    if not train:
        log.warn("no positive training frames to spot check")
        return
    rng.shuffle(train)
    drawn = 0
    for record in train:
        if drawn >= count:
            break
        image = imread_any(Path(record["image"]))
        if image is None:
            continue
        height, width = image.shape[:2]
        rows, _errors = parse_label_file(Path(record["label"]))
        for class_id, cx, cy, w, h in rows:
            x1 = int((cx - w / 2) * width)
            y1 = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)
            colour = PALETTE[class_id % len(PALETTE)]
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
            label = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        imwrite(out_dir / f"spot_{drawn:03d}.jpg", image)
        drawn += 1

    readme = out_dir / "README.txt"
    readme.write_text(
        "Gate G19, DATASET_SPEC.md section 9.\n\n"
        f"{drawn} training images with their boxes drawn. Open them. Check that boxes sit on\n"
        "objects, that class names are right, and that nothing is shifted or mirrored - a\n"
        "coordinate-convention error looks exactly like this and no assertion catches it.\n\n"
        "Then write PASS or FAIL, plus a line saying what you saw, to VERDICT.txt in this\n"
        "directory. 10_validate.py reads that file and fails when it is missing.\n",
        encoding="utf-8",
    )
    log.info("spot check rendered", images=drawn, path=str(out_dir), next="write VERDICT.txt after looking")


if __name__ == "__main__":
    run(main)
