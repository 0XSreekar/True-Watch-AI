"""CPU cost of each tiling layout: forward passes per 1080p frame and ms per frame, onnxruntime, through the
edge AppearanceChannel (the code that ships). The frame is a synthetic 1920x1080 scene, so the detector
postprocess sees few candidates; the forward passes dominate and do not depend on content.

    python tiled_bench.py --onnx final.onnx --out cost.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "edge"))
import tiled_common as TC  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime intra-op threads, 0 = default")
    a = ap.parse_args(argv)

    import cv2
    import onnxruntime as ort
    from pipeline.appearance import AppearanceChannel, AppearanceConfig

    so = ort.SessionOptions()
    if a.threads:
        so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    frame = (rng.random((1080, 1920, 3)) * 255).astype(np.uint8)
    frame = cv2.GaussianBlur(frame, (0, 0), 3)
    out = {"onnx": Path(a.onnx).name, "ort": ort.__version__, "frame": "1920x1080", "threads": a.threads or "default", "layouts": {}}
    for name in [None, *TC.LAYOUTS]:
        tiling = TC.T.TilingConfig() if name is None else TC.variant(name)
        ch = AppearanceChannel(sess, 640, ["person", "two_wheeler", "car", "truck", "cart"], AppearanceConfig(tiling=tiling))
        passes = 1 + (len(TC.T.tile_windows(1920, 1080, tiling)) if tiling.enabled else 0)
        ch.detect(frame)   # warm-up
        ms = []
        for _ in range(a.frames):
            t0 = time.perf_counter()
            ch.detect(frame)
            ms.append((time.perf_counter() - t0) * 1000)
        out["layouts"][name or "full_frame"] = {"passes_per_1080p_frame": passes, "ms_median": statistics.median(ms), "ms_all": [round(m, 1) for m in ms]}
        print(name or "full_frame", out["layouts"][name or "full_frame"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
