#!/usr/bin/env python3
"""Emit the final Ultralytics-format dataset and data.yaml. DATASET_SPEC.md sections 2.5 and 8.

Deliberate choices in this script:

  * data.yaml carries NO `test:` key. The test split is sealed until Phase 11, and the
    cheapest enforcement is that a trainer cannot address it by accident (section 2.5).
    Its `path` is RELATIVE (".") so the directory works wherever it is mounted, e.g. when
    this build is attached as a Kaggle input to the training notebook; training/scripts
    resolves the root itself and never hands this file to the trainer unchanged.
  * Every image is written as JPEG, quality 92, and untiled frames are capped at 1280 px on
    the long side. Why 1280: the detector trains at 640, so 1280 keeps 2x headroom for the
    0.5-1.5x scale augmentation and for a later higher-resolution run, while it takes the
    largest sources (IDD 1920x1080, FLIR RGB 1800x1600) from ~0.5-1.2 MB to ~0.2-0.3 MB a
    frame and keeps the full build inside Kaggle's 20 GB output limit. The far field, where
    that downscale would hurt, is covered by 08's tiles, which are cut at native resolution
    BEFORE this cap and are never resized here. LWIR is written through the same encoder
    from the B = G = R conversion; JPEG keeps the three channels bit-identical (Cb = Cr = 128),
    and gate G10 checks it on the written files.
  * Final names are {source}_{stem of record image}_{modality}. training/scripts derive the
    same names from the split index, so the rule must not change. Every final name is
    asserted unique across ALL sources before a single file is written.
  * The cart gate (DATASET_SPEC.md section 1.5) is evaluated here: class 4 rows are added
    only if the review sheet holds >= 300 human-verified carts. Otherwise class 4 is
    withdrawn, id 4 stays reserved in data.yaml, and the decision is written to
    manifests/cart_gate.json and manifests/class_exceptions.txt for gate G11.

The build is self-contained: data.yaml, images/, labels/, manifest.tsv, index/split.jsonl
(one record per placed image: what training's evaluators read), manifests/ and reports/.

This file writes an Ultralytics-FORMAT dataset. It does not import Ultralytics: that
package is AGPL-3.0 and is confined to training/.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    MANIFEST_DIR,
    REPORT_DIR,
    Counters,
    Logger,
    assert_unique,
    base_parser,
    cart_gate_status,
    cart_sheet_path,
    class_ids,
    class_names,
    final_stem,
    format_label_line,
    ir_conversion_params,
    load_state,
    parallel_map,
    place_worker,
    read_cart_sheet,
    read_jsonl,
    require_dirs,
    run,
    save_state,
    sha256_file,
    write_json,
    write_lines,
    xyxy_to_yolo,
)

SCRIPT = "09_build_yolo_ds"
JPEG_QUALITY = 92
LONG_SIDE_CAP = 1280
SPLITS = ("train", "val", "test")


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def cart_lines_by_image(records: list[dict], processed: Path, counters: Counters) -> dict[str, list[str]]:
    """uid -> class-4 label rows, from the verified rows of the review sheet."""
    by_uid = {Path(r["image"]).stem: r for r in records if r.get("source") == "idd"}
    cart_id = class_ids()["cart"]
    out: dict[str, list[str]] = defaultdict(list)
    for row in read_cart_sheet(cart_sheet_path(processed)):
        if str(row.get("verdict", "")).strip().lower() != "cart":
            continue
        record = by_uid.get(row.get("uid", ""))
        if record is None:
            counters.bump("cart.verified_row_image_not_in_build")
            continue
        geometry = xyxy_to_yolo(float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]),
                                int(record["img_w"]), int(record["img_h"]))
        if geometry is None:
            counters.bump("cart.degenerate_box")
            continue
        out[record["image"]].append(format_label_line(cart_id, *geometry))
        counters.bump("cart.instances_added")
    return out


def main() -> int:  # noqa: C901 - one linear build
    ap = base_parser(__doc__)
    ap.add_argument("--out", default=None, help="dataset root (default: <processed>/yolo)")
    ap.add_argument("--quality", type=int, default=JPEG_QUALITY, help="JPEG quality of every written image")
    ap.add_argument("--long-side", type=int, default=LONG_SIDE_CAP, help="long-side cap for untiled frames")
    ap.add_argument("--budget-gb", type=float, default=15.0, help="warn when the projected output exceeds this")
    ap.add_argument("--hard-limit-gb", type=float, default=19.5,
                    help="stop when the projected output exceeds this (Kaggle /kaggle/working holds 20 GB)")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()
    started = time.time()

    processed = Path(args.processed).resolve()
    out_root = Path(args.out).resolve() if args.out else processed / "yolo"
    index_path = processed / "index" / "with_negatives.jsonl"
    if not index_path.exists():
        log.error("input index missing - run 07_negatives.py first", expected=str(index_path))
        return 1

    frames = read_jsonl(index_path)
    tiles = read_jsonl(processed / "index" / "tiles.jsonl")
    if not tiles:
        log.warn("no tile index found - gate G13 will fail", expected=str(processed / "index" / "tiles.jsonl"))
    excluded = [r for r in frames if r.get("excluded_empty")]
    counters.bump("record.excluded_empty_train_frame", len(excluded))
    records = sorted([r for r in frames if not r.get("excluded_empty")] + tiles, key=lambda r: r["image"])
    records = [r for r in records if r.get("split") in SPLITS]
    if args.limit:
        records = records[: args.limit]

    # The collision assertion. It runs on the whole plan, before any file is written.
    try:
        assert_unique((final_stem(r) for r in records), "final file names")
        assert_unique((r["image"] for r in records), "record image paths")
    except AssertionError as exc:
        log.fatal("final names are not unique - two records would overwrite one file", detail=str(exc)[:400])
        return 1
    log.info("final names unique", records=len(records))

    gate = cart_gate_status(processed)
    cart_rows = cart_lines_by_image(records, processed, counters) if gate["shipped"] else {}
    log.info("cart gate", decision=gate["decision"], verified=gate["verified_cart"], threshold=gate["threshold"],
             candidates=gate["candidates"], unreviewed=gate["unreviewed"])

    state = load_state(processed, SCRIPT)
    done = set(state.get("done", [])) if not args.force else set()
    ir_params = list(ir_conversion_params())

    jobs: list[dict] = []
    placed_records: list[dict] = []
    for record in records:
        split = record["split"]
        stem = final_stem(record)
        target_image = out_root / "images" / split / f"{stem}.jpg"
        target_label = out_root / "labels" / split / f"{stem}.txt"
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
            lines += cart_rows.get(record["image"], [])
        jobs.append({
            "record": record, "target_image": target_image.as_posix(), "target_label": target_label.as_posix(),
            "lines": lines, "quality": args.quality, "long_side_cap": args.long_side, "ir_params": ir_params,
            "skip_image": record["image"] in done and not args.force,
        })
        placed = dict(record)
        placed["final_image"] = f"images/{split}/{stem}.jpg"
        placed["final_label"] = f"labels/{split}/{stem}.txt"
        placed["label_rows"] = len(lines)
        placed_records.append(placed)

    per_split = Counter(j["record"]["split"] for j in jobs)
    log.info("plan", **{s: per_split.get(s, 0) for s in SPLITS}, total=len(jobs), workers=args.workers,
             quality=args.quality, long_side=args.long_side)
    if args.dry_run:
        log.info("dry-run: nothing written", root=str(out_root))
        return 0

    require_dirs(*(out_root / kind / s for kind in ("images", "labels") for s in SPLITS))
    written_bytes = 0
    failures = 0
    failed_images: set[str] = set()
    projected_reported = False
    sample_after = min(len(jobs), max(200, len(jobs) // 50))
    for position, (job, result) in enumerate(zip(jobs, parallel_map(place_worker, jobs, args.workers, 16)), 1):
        if not result.get("ok"):
            failures += 1
            failed_images.add(job["record"]["image"])
            counters.bump(f"image.failed.{result.get('why')}")
            if failures <= 10:
                log.warn("image not placed", image=job["record"]["image"], why=result.get("why"))
            continue
        written_bytes += int(result.get("bytes", 0))
        done.add(job["record"]["image"])
        counters.bump("image.already_placed" if result.get("why") == "already_placed" else "image.written")
        if result.get("downscaled"):
            counters.bump("image.downscaled_to_cap")
        if not projected_reported and position >= sample_after:
            projected_reported = True
            elapsed = time.time() - started
            projected = written_bytes / position * len(jobs)
            eta = elapsed / position * (len(jobs) - position)
            log.info("projection", sampled=position, projected_size=human_bytes(projected),
                     projected_remaining_min=round(eta / 60.0, 1), budget_gb=args.budget_gb)
            if projected > args.budget_gb * 1024**3:
                log.warn("projected output exceeds the budget", projected=human_bytes(projected), budget_gb=args.budget_gb)
            if projected > args.hard_limit_gb * 1024**3:
                log.fatal("projected output exceeds the hard limit - lower --quality or --long-side",
                          projected=human_bytes(projected), hard_limit_gb=args.hard_limit_gb)
                return 1
        if position % 10000 == 0:
            log.info("placing", done=position, total=len(jobs))

    placed_records = [r for r in placed_records if r["image"] not in failed_images]

    # Idempotency: a file this plan does not contain is stale from an earlier plan; remove it
    # so the directory always equals the manifest.
    planned = {r["final_image"] for r in placed_records} | {r["final_label"] for r in placed_records}
    for kind in ("images", "labels"):
        for split in SPLITS:
            for path in (out_root / kind / split).iterdir():
                if f"{kind}/{split}/{path.name}" not in planned:
                    path.unlink()
                    counters.bump("stale_file_removed")

    names = class_names()
    data_yaml = out_root / "data.yaml"
    withdrawn = "" if gate["shipped"] else (
        "# Class 4 (cart) is WITHDRAWN in this build: " + gate["reason"] + ".\n"
        "# Its id stays reserved so the wire format does not change (DATASET_SPEC.md 1.5).\n"
    )
    data_yaml.write_text(
        "# TRUEWATCH unified detection dataset, written by datasets/scripts/09_build_yolo_ds.py.\n"
        "# `path` is relative: this directory is self-contained and may be mounted anywhere.\n"
        "# There is deliberately NO `test:` key: the test split is sealed until Phase 11\n"
        "# (DATASET_SPEC.md section 2.5); images/test exists for Phase 11 only.\n"
        + withdrawn
        + "path: .\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8",
    )

    manifest_rows = sorted(f"{r['split']}\t{r['final_image']}\t{r['label_rows']}" for r in placed_records)
    write_lines(out_root / "manifest.tsv", manifest_rows)
    write_lines(out_root / "index" / "split.jsonl", [json.dumps(r, sort_keys=True) for r in placed_records])

    require_dirs(MANIFEST_DIR)
    exceptions_path = MANIFEST_DIR / "class_exceptions.txt"
    exceptions = [ln.strip() for ln in exceptions_path.read_text(encoding="utf-8").splitlines()] if exceptions_path.exists() else []
    exceptions = [e for e in exceptions if e and not e.startswith("#") and e != "cart"]
    if not gate["shipped"]:
        exceptions.append("cart")
    write_lines(exceptions_path, ["# classes allowed under 1% of instances (gate G11), with a documented decision",
                                  "# cart: DATASET_SPEC.md 1.5 gate, see cart_gate.json"] + sorted(set(exceptions)))
    write_json(MANIFEST_DIR / "cart_gate.json", gate)

    digest = sha256_file(out_root / "manifest.tsv")
    size_bytes = sum(p.stat().st_size for p in (out_root / "images").rglob("*") if p.is_file())
    per_split_placed = Counter(r["split"] for r in placed_records)
    summary = {
        "images": len(placed_records),
        "per_split": dict(sorted(per_split_placed.items())),
        "classes": names,
        "cart_gate": gate,
        "manifest_sha256": digest,
        "jpeg_quality": args.quality,
        "long_side_cap": args.long_side,
        "images_bytes": size_bytes,
        "excluded_empty_train_frames": len(excluded),
        "failures": failures,
        "seconds": round(time.time() - started, 1),
    }
    write_json(MANIFEST_DIR / "dataset_manifest.json", summary)
    write_lines(MANIFEST_DIR / "dataset_manifest.tsv", manifest_rows)

    # Self-contained output: when the manifests live elsewhere (a local build), copy them in.
    own_manifests = out_root / "manifests"
    if own_manifests.resolve() != MANIFEST_DIR.resolve():
        require_dirs(own_manifests)
        for path in sorted(MANIFEST_DIR.glob("*")):
            if path.is_file() and path.name != ".gitkeep":
                shutil.copy2(path, own_manifests / path.name)
    for name in ("splits.json",):
        if (processed / name).exists():
            shutil.copy2(processed / name, own_manifests / name)
    if (processed / "reports" / "split.json").exists():
        shutil.copy2(processed / "reports" / "split.json", own_manifests / "split_report.json")
    if cart_sheet_path(processed).exists():
        shutil.copy2(cart_sheet_path(processed), own_manifests / "vehicle_fallback.csv")

    state["done"] = sorted(done)
    save_state(processed, SCRIPT, state)
    write_json(processed / "reports" / "build.json", summary)
    write_json(REPORT_DIR / "build.json", summary)

    counters.report(log)
    for split in SPLITS:
        log.info("built", split=split, images=per_split_placed.get(split, 0))
    log.info("done", root=str(out_root), size=human_bytes(size_bytes), failures=failures,
             minutes=round((time.time() - started) / 60.0, 1), manifest_sha256=digest,
             cart=gate["decision"], workers=os.environ.get("TRUEWATCH_WORKERS", args.workers))
    return 1 if failures else 0


if __name__ == "__main__":
    run(main)
