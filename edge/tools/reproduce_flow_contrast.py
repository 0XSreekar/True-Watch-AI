#!/usr/bin/env python3
"""Re-run the docs/MEASUREMENTS.md section 1 flow-contrast measurement on paired visible / LWIR frames.

Section 1: dense Farneback flow over 100 paired KAIST frames, mean magnitude inside detected object boxes
versus the background: visible 5.17 / 2.74 px (1.9x), LWIR 2.73 / 0.81 px (3.4x); whole-frame LWIR flow
about 32% of visible. This script applies the same method with the parameters pinned in
pipeline.motion.FARNEBACK_PARAMS, RAW flow (no global-shift removal, as the measurement had none), at the
frames' native size (resized to 640 px wide only if larger):

  - boxes come from the appearance channel run on the VISIBLE frame and are applied to both streams
    (KAIST pairs are co-registered), so both streams are measured on the same objects; --boxes own runs
    the detector on each stream separately instead;
  - per pair: mean flow inside the union of boxes, mean flow outside it (pipeline.motion.
    object_background_contrast); over all pairs: the ratio of the two means (aggregate_contrast).

    cd edge
    python tools/reproduce_flow_contrast.py --pair-dir ../var/datasets/kaist/kaist_train/set00/V000 --start 1150

The section 1 set was KAIST set00/V007. A different sequence gives its own numbers; they are printed as
observed, next to the published ones, never substituted for them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

EDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDGE_ROOT))

from pipeline.motion import (FARNEBACK_PARAMS, MEASURED_FLOW_PX, WORKING_WIDTH, aggregate_contrast,  # noqa: E402
                             farneback, flow_magnitude, object_background_contrast, working_size)


def load_gray(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    h, w = img.shape[:2]
    ww, wh = working_size(w, h, WORKING_WIDTH)
    if (ww, wh) != (w, h):
        img = cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA)
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair-dir", type=Path, required=True, help="a folder holding visible/ and lwir/ frame folders")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default="person,two_wheeler,car,truck,cart")
    ap.add_argument("--boxes", choices=("visible", "own"), default="visible")
    args = ap.parse_args(argv)

    from config import load
    from models.detector_weights import load_at_boot
    from pipeline.appearance import AppearanceChannel, AppearanceConfig

    detector = load_at_boot(load())
    if detector is None:
        raise SystemExit("no detector configured: set YOLO_MODEL_URL and YOLO_MODEL_SHA256")
    ch = AppearanceChannel.from_detector(detector, AppearanceConfig(conf=args.conf))
    keep = set(args.classes.split(","))

    files = {s: sorted((args.pair_dir / s).glob("*.jpg")) for s in ("visible", "lwir")}
    n = min(len(files["visible"]), len(files["lwir"]))
    idx = range(args.start, min(args.start + args.pairs + 1, n))
    if len(idx) < 2:
        raise SystemExit(f"not enough paired frames in {args.pair_dir}")

    samples = {"visible": [], "lwir": []}
    whole = {"visible": [], "lwir": []}
    dets_count = {"visible": 0, "lwir": 0}
    prev = {}
    for k in idx:
        frames = {s: load_gray(files[s][k]) for s in ("visible", "lwir")}
        boxes = {}
        for s in ("visible", "lwir"):
            d = [x for x in ch.detect(frames[s][0], modality=s) if x.class_name in keep]
            dets_count[s] += sum(1 for x in d if x.class_name == "person")
            boxes[s] = [x.bbox for x in d]
        if args.boxes == "visible":
            boxes["lwir"] = boxes["visible"]
        for s in ("visible", "lwir"):
            if s in prev:
                mag = flow_magnitude(farneback(prev[s], frames[s][1]))
                whole[s].append(float(mag.mean()))
                got = object_background_contrast(mag, boxes[s])
                if got is not None:
                    samples[s].append(got)
            prev[s] = frames[s][1]

    print(f"Farneback {FARNEBACK_PARAMS}, raw flow, {len(idx) - 1} consecutive pairs from {args.pair_dir} "
          f"starting at frame {args.start}; boxes: {args.boxes}, classes {sorted(keep)}, conf >= {args.conf}\n")
    print("| stream | pairs with boxes | object px | background px | contrast | published (section 1) |")
    print("|---|---|---|---|---|---|")
    agg = {}
    for s in ("visible", "lwir"):
        a = agg[s] = aggregate_contrast(samples[s])
        po, pb = MEASURED_FLOW_PX[s]
        print(f"| {s} | {a['pairs']} | {a['object_px']:.2f} | {a['background_px']:.2f} | {a['contrast']:.2f}x | "
              f"{po} / {pb} px = {po / pb:.1f}x |")
    wv, wl = float(np.mean(whole["visible"])), float(np.mean(whole["lwir"]))
    print(f"\nwhole-frame flow: visible {wv:.2f} px, LWIR {wl:.2f} px, LWIR/visible {wl / wv:.0%} (published: 32%)")
    print(f"person boxes: visible {dets_count['visible']}, LWIR {dets_count['lwir']}")
    if agg["visible"]["pairs"] and agg["lwir"]["pairs"]:
        print("LWIR contrast " + ("EXCEEDS" if agg["lwir"]["contrast"] > agg["visible"]["contrast"] else "does NOT exceed")
              + " visible contrast on this sequence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
