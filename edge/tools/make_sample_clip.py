#!/usr/bin/env python3
"""Generate a synthetic clip so the ingest path can be exercised offline.

This is a development convenience, not data. It draws a figure walking left to
right across a dark frame — enough to open, decode and count, nothing more. Any
real evaluation uses the public datasets fetched by datasets/*.py.

    python edge/tools/make_sample_clip.py --out edge/var/samples/sample.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./var/samples/sample.mp4")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    count = int(args.fps * args.seconds)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        print(f"could not open a writer for {out}")
        return 2

    rng = np.random.default_rng(7)
    for i in range(count):
        frame = np.full((args.height, args.width, 3), 18, dtype=np.uint8)
        frame += rng.integers(0, 6, frame.shape, dtype=np.uint8)  # sensor noise
        # ground line
        frame[int(args.height * 0.72) : int(args.height * 0.72) + 2, :] = 60
        # a figure crossing
        x = int(i / max(1, count - 1) * (args.width - 24)) + 4
        y = int(args.height * 0.52)
        frame[y : y + 46, x : x + 16] = 210
        writer.write(frame)

    writer.release()
    print(f"wrote {out} — {count} frames, {args.width}x{args.height} at {args.fps} fps")
    print("This is synthetic development material, not evaluation data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
