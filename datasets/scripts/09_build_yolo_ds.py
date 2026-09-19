#!/usr/bin/env python3
"""Emit the final Ultralytics-format dataset and data.yaml. DATASET_SPEC.md sections 2.5 and 8.

Two deliberate choices in this script:

  * data.yaml carries NO `test:` key. The test split is sealed until Phase 11, and the
    cheapest enforcement is that a trainer cannot address it by accident (section 2.5).
  * Visible frames are re-encoded to JPEG q90 with the long side capped at 1280 px to fit
    a free-tier Kaggle dataset. LWIR frames stay lossless PNG and TILES are never resized,
    because the resize is the exact thing tiling exists to avoid.

This file writes an Ultralytics-FORMAT dataset. It does not import Ultralytics: that
package is AGPL-3.0 and is confined to training/.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    MANIFEST_DIR,
    Counters,
    Logger,
    base_parser,
    class_names,
    imread_any,
    imwrite,
    load_state,
    require_dirs,
    run,
    save_state,
    sha256_file,
    write_json,
    write_lines,
)

SCRIPT = "09_build_yolo_ds"

JPEG_QUALITY = 90
LONG_SIDE_CAP = 1280


def place_image(source: Path, target: Path, record: dict, args, counters: Counters, log: Logger) -> bool:
    if target.exists() and not args.force:
        counters.bump("image.already_placed")
        return True
    if args.dry_run:
        counters.bump("image.would_place")
        return True
    if not source.exists():
        log.warn("source image missing", image=source.as_posix())
        counters.bump("image.missing")
        return False

    # LWIR stays lossless: the single channel is replicated three ways and JPEG ringing on a
    # low-contrast thermal frame is destructive. Tiles keep native resolution by definition.
    if record.get("modality") == "lwir" or record.get("tiled"):
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            log.error("copy failed", image=source.as_posix(), detail=str(exc)[:120])
            counters.bump("image.copy_failed")
            return False
        counters.bump("image.copied_lossless")
        return True

    image = imread_any(source)
    if image is None:
        log.warn("unreadable image", image=source.as_posix())
        counters.bump("image.unreadable")
        return False

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > LONG_SIDE_CAP:
        import cv2

        scale = LONG_SIDE_CAP / float(longest)
        image = cv2.resize(
            image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA
        )
        counters.bump("image.downscaled")

    import cv2

    target.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        log.error("encode failed", image=source.as_posix())
        counters.bump("image.encode_failed")
        return False
    buffer.tofile(str(target))
    counters.bump("image.reencoded")
    return True


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--out", default=None, help="dataset root (default: <processed>/yolo)")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    processed = Path(args.processed).resolve()
    out_root = Path(args.out).resolve() if args.out else processed / "yolo"

    index_path = processed / "index" / "with_negatives.jsonl"
    if not index_path.exists():
        log.error("input index missing - run 07_negatives.py first", expected=str(index_path))
        return 1

    records: list[dict] = []
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
        log.info("tiles merged", index=str(tiles_path))
    else:
        log.warn("no tile index found - gate G13 will fail", expected=str(tiles_path))

    records.sort(key=lambda r: r["image"])
    if args.limit:
        records = records[: args.limit]

    state = load_state(processed, SCRIPT)
    done = set(state.get("done", [])) if not args.force else set()

    per_split: Counter = Counter()
    manifest_rows: list[str] = []
    failures = 0

    for record in records:
        split = record.get("split")
        if split not in ("train", "val", "test"):
            counters.bump("record.no_split")
            continue

        source = Path(record["image"])
        stem = f"{record.get('source', 'x')}_{source.stem}_{record.get('modality', 'v')}"
        suffix = ".png" if (record.get("modality") == "lwir" or record.get("tiled")) else ".jpg"
        if record.get("tiled"):
            suffix = source.suffix
        target_image = out_root / "images" / split / f"{stem}{suffix}"
        target_label = out_root / "labels" / split / f"{stem}.txt"

        if record["image"] in done and target_image.exists() and not args.force:
            counters.bump("record.already_done")
            per_split[split] += 1
            # The manifest is rebuilt in full on every run, including rows for work that
            # was already done, so a resumed run cannot truncate it to the last increment.
            if target_label.exists():
                existing = sum(
                    1 for ln in target_label.read_text(encoding="utf-8").splitlines() if ln.strip()
                )
                manifest_rows.append(
                    f"{split}\t{target_image.relative_to(out_root).as_posix()}\t{existing}"
                )
            continue

        if not place_image(source, target_image, record, args, counters, log):
            failures += 1
            continue

        if record.get("is_negative"):
            lines: list[str] = []
            counters.bump("label.negative_empty")
        else:
            label_source = Path(record.get("label", ""))
            if label_source.exists():
                lines = [ln for ln in label_source.read_text(encoding="utf-8").splitlines() if ln.strip()]
            else:
                log.warn("label file missing", label=str(label_source))
                counters.bump("label.missing")
                lines = []

        if not args.dry_run:
            write_lines(target_label, lines)
            manifest_rows.append(f"{split}\t{target_image.relative_to(out_root).as_posix()}\t{len(lines)}")

        per_split[split] += 1
        counters.bump(f"placed.{split}")
        done.add(record["image"])

    names = class_names()
    data_yaml = out_root / "data.yaml"
    yaml_text = (
        "# TRUEWATCH unified detection dataset. Generated by datasets/scripts/09_build_yolo_ds.py.\n"
        "# There is deliberately NO `test:` key: the test split is sealed until Phase 11\n"
        "# (DATASET_SPEC.md section 2.5) and a trainer must not be able to address it by accident.\n"
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(names)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )

    if args.dry_run:
        log.info("dry-run: nothing written", would_write=str(data_yaml))
        counters.report(log)
        for split, count in sorted(per_split.items()):
            log.info("planned", split=split, images=count)
        return 0

    require_dirs(out_root, MANIFEST_DIR)
    data_yaml.write_text(yaml_text, encoding="utf-8")
    write_lines(out_root / "manifest.tsv", manifest_rows)
    write_lines(MANIFEST_DIR / "dataset_manifest.tsv", manifest_rows)

    digest = sha256_file(out_root / "manifest.tsv")
    write_json(
        MANIFEST_DIR / "dataset_manifest.json",
        {
            "images": sum(per_split.values()),
            "per_split": dict(sorted(per_split.items())),
            "classes": names,
            "manifest_sha256": digest,
            "jpeg_quality": JPEG_QUALITY,
            "long_side_cap": LONG_SIDE_CAP,
        },
    )
    state["done"] = sorted(done)
    save_state(processed, SCRIPT, state)
    write_json(
        processed / "reports" / "build.json",
        {"per_split": dict(sorted(per_split.items())), "failures": failures, "counters": counters.as_dict()},
    )

    counters.report(log)
    for split, count in sorted(per_split.items()):
        log.info("built", split=split, images=count)
    log.info("done", root=str(out_root), data_yaml=str(data_yaml), failures=failures, manifest_sha256=digest)
    return 1 if failures else 0


if __name__ == "__main__":
    run(main)
