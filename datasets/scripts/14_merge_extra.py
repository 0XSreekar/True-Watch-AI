#!/usr/bin/env python3
"""Merge the Open Images add-on (13_convert_openimages.py shards) into a built dataset tree, so it can be validated.

    python 14_merge_extra.py --shards <dir with SHARDS.json + truewatch_extra_*.tar> --dataset <truewatch_ds>

Only the TRAIN images are placed in the dataset (images/train, labels/train, index/split.jsonl, manifest.tsv), so the v2
val and test splits stay byte-identical and the headline comparison stays on the same images. The held-out Open Images
photos (images/val_oi) are unpacked next to the dataset, in <dataset>/../truewatch_extra/, and never enter it.
Idempotent: a second run finds the rows already present and adds nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    shards, ds = Path(args.shards), Path(args.dataset)
    index = json.loads((shards / "SHARDS.json").read_text())
    stage = ds.parent / "truewatch_extra_unpack"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for sh in index["shards"]:
        p = shards / sh["name"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != sh["sha256"]:
            raise SystemExit(f"{p.name}: sha256 mismatch against SHARDS.json")
        with tarfile.open(p) as tar:
            tar.extractall(stage, filter="data")
    extra = stage / "truewatch_extra"
    held = ds.parent / "truewatch_extra"
    if held.exists():
        shutil.rmtree(held)
    held.mkdir()
    for sub in ("images", "labels"):
        if (extra / sub / "val_oi").is_dir():
            shutil.copytree(extra / sub / "val_oi", held / sub / "val_oi")
    rows = [json.loads(ln) for ln in (extra / "index" / "split_extra.jsonl").read_text().splitlines() if ln.strip()]
    train_rows = [r for r in rows if r["split"] == "train"]
    split_jsonl = ds / "index" / "split.jsonl"
    present = {json.loads(ln).get("final_image") for ln in split_jsonl.read_text().splitlines() if ln.strip()}
    new = [r for r in train_rows if r["final_image"] not in present]
    for r in new:
        for key in ("final_image", "final_label"):
            dst = ds / r[key]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extra / r[key]), str(dst))
    with split_jsonl.open("a", encoding="utf-8") as fh:
        for r in new:
            fh.write(json.dumps({k: v for k, v in r.items() if k not in ("author", "landing")}) + "\n")
    with (ds / "manifest.tsv").open("a", encoding="utf-8") as fh:
        for r in new:
            fh.write(f"train\t{r['final_image']}\t{r['label_rows']}\n")
    shutil.copy2(extra / "EXTRA_REPORT.json", held / "EXTRA_REPORT.json")
    shutil.rmtree(stage)
    print(json.dumps({"added_train_images": len(new), "added_train_boxes": sum(r["label_rows"] for r in new),
                      "held_out_dir": str(held)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
