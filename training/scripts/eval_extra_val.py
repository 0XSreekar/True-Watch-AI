#!/usr/bin/env python3
"""Person AP50 of one or more weights on the held-out Open Images photos (images/val_oi), 640 px.

These photos are from Open Images' own validation split, never in training and never in the headline v2 val split. Only the
person class is labelled there, so only person AP50 is reported (the other classes have no ground truth). Run it for the
previous detector and the new one and compare the two numbers; it is a side check, the headline stays on the v2 val split.

    python eval_extra_val.py --extra-root <truewatch_extra> --weights a.pt b.pt --out eval_extra_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra-root", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from ultralytics import YOLO

    root = Path(args.extra_root).resolve()
    images = root / "images" / "val_oi"
    if not images.is_dir():
        print(f"no held-out images at {images}", file=sys.stderr)
        return 1
    names = {0: "person", 1: "two_wheeler", 2: "car", 3: "truck", 4: "cart"}
    result: dict[str, dict] = {"images": len(list(images.glob("*.jpg")))}
    with tempfile.TemporaryDirectory() as tmp:
        yml = Path(tmp) / "val_oi.yaml"
        body = "\n".join(f"  {k}: {v}" for k, v in names.items())
        yml.write_text(f"path: {root.as_posix()}\ntrain: images/val_oi\nval: images/val_oi\nnames:\n{body}\n", encoding="utf-8")
        for w in args.weights:
            m = YOLO(w)
            r = m.val(data=str(yml), imgsz=args.imgsz, batch=args.batch, device=args.device, conf=0.001, iou=0.7,
                      plots=False, verbose=False)
            idx = list(r.box.ap_class_index).index(0)
            result[w] = {"person_ap50": round(float(r.box.ap50[idx]), 4), "person_ap50_95": round(float(r.box.ap[idx]), 4)}
            print(w, result[w], flush=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
