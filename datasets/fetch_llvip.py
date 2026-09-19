#!/usr/bin/env python3
"""Fetch the LLVIP visible/infrared pair set into a local, git-ignored folder.

Slide 4 names LLVIP for IR fine-tuning: "Fine-tune on LLVIP and KAIST, IR
replicated to 3 channels." No dataset file is ever committed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "jsonhash/LLVIP"
REPO_TYPE = "dataset"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./var/datasets/llvip", help="destination directory")
    ap.add_argument("--pattern", default=None, help="optional glob restricting the fetch")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed.\n  pip install huggingface_hub", file=sys.stderr)
        return 2

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"fetching {REPO_ID} into {out}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(out),
        allow_patterns=[args.pattern] if args.pattern else None,
        max_workers=4,
    )

    fetched = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"done: {fetched} files under {out}")
    print("This directory is git-ignored. Do not commit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
