#!/usr/bin/env python3
"""Prepare IDD Detection from a manually downloaded archive.

IIIT Hyderabad gates IDD behind an account and a licence agreement. This script
does NOT bypass that gate: it prints the registration URL, then verifies and
extracts an archive you downloaded yourself.

Slide 3 names IDD for the appearance channel's fine-tuning; capability 2's
vehicle classes come from it. docs/MEASUREMENTS.md records the set as 46,588
images (31,569 / 10,225 / 4,794).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path

REGISTRATION_URL = "https://idd.insaan.iiit.ac.in/"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", help="path to the archive you downloaded after registering")
    ap.add_argument("--out", default="./var/datasets/idd", help="destination directory")
    args = ap.parse_args()

    if not args.archive:
        print(__doc__)
        print(f"Register and download IDD Detection here:\n  {REGISTRATION_URL}\n")
        print("Then re-run:\n  python datasets/fetch_idd.py --archive /path/to/idd.tar.gz")
        return 1

    archive = Path(args.archive).expanduser().resolve()
    if not archive.is_file():
        print(f"no such archive: {archive}", file=sys.stderr)
        return 2

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"archive: {archive} ({archive.stat().st_size / 1e9:.2f} GB)")
    print(f"sha256:  {_digest(archive)}")
    print(f"extracting into {out}")

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(out, filter="data")
    else:
        print(f"unrecognised archive format: {archive.suffix}", file=sys.stderr)
        return 2

    images = sum(1 for p in out.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"done: {images} images under {out}")
    print("This directory is git-ignored. Do not commit it.")
    print("IDD's own licence governs this data; it is not covered by this repository's licence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
