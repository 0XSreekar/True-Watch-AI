#!/usr/bin/env python3
"""Fetch the KAIST Multispectral Pedestrian set into a local, git-ignored folder.

Every measured number in docs/MEASUREMENTS.md sections 1, 2 and 3 came from this
dataset. Fetching a subset is enough to reproduce them; --limit keeps the
download small.

No dataset file is ever committed. See datasets/README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset"
REPO_TYPE = "dataset"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./var/datasets/kaist", help="destination directory")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="fetch at most this many files (omit for the full 23,210)",
    )
    ap.add_argument(
        "--pattern",
        default="*set00*",
        help="glob restricting which files are fetched; MEASUREMENTS used set00/V007",
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import list_repo_files, snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed.\n"
            "  pip install huggingface_hub",
            file=sys.stderr,
        )
        return 2

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.limit is not None:
        files = [f for f in list_repo_files(REPO_ID, repo_type=REPO_TYPE)]
        print(f"repository holds {len(files)} files; fetching at most {args.limit}")

    print(f"fetching {REPO_ID} (pattern {args.pattern!r}) into {out}")
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
