"""Run the detector on the full frame and on every tiling window, cache the per-pass boxes (GPU, inference only).

Output (--out DIR), resumable per chunk:
  passes_NNNNN.npz   per chunk of images: for every pass (image, window or full frame) its boxes, scores, classes
  labels.jsonl.gz    one line per image: name, size, ground-truth rows, resolved metadata (source, slice, lighting, cluster)
  run.json           settings, timings and pass counts

Pass 0 (the full frame) goes through `_predict.predict_raw`, the exact path evaluate.py uses, so the
baseline can be compared with eval_final.json. Window passes feed the cropped region to the same model
through Ultralytics' own letterbox (a crop smaller than 640 is upscaled). Every pass is stored after
class-aware NMS at 0.7 (the evaluator's NMS), capped at 300 boxes; full frame floor 0.001, windows 0.005.

Windows are the union of tiled_common.LAYOUTS; the merge happens later, offline, in tiled_eval.py with
edge/pipeline/tiling.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
import _predict as P  # noqa: E402
import tiled_common as TC  # noqa: E402

WINDOW_FLOOR = 0.005
FULL_FLOOR = 0.001


def reduce_pass(boxes, conf, cls, nms_iou=0.7, max_det=300):
    idx = P.nms(boxes, conf, cls, nms_iou, max_det)
    return boxes[idx].astype(np.float32), conf[idx].astype(np.float32), cls[idx].astype(np.int16)


def save_chunk(path, records):
    """records: list of (img_idx, window or None, boxes, conf, cls)."""
    counts = np.array([len(r[2]) for r in records], dtype=np.int64)
    np.savez_compressed(
        path,
        img=np.array([r[0] for r in records], dtype=np.int32),
        win=np.array([(-1, -1, -1, -1) if r[1] is None else r[1] for r in records], dtype=np.int32).reshape(-1, 4),
        counts=counts,
        boxes=np.concatenate([r[2] for r in records]) if counts.sum() else np.zeros((0, 4), np.float32),
        conf=np.concatenate([r[3] for r in records]) if counts.sum() else np.zeros(0, np.float32),
        cls=np.concatenate([r[4] for r in records]) if counts.sum() else np.zeros(0, np.int16),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--index", default=None, help="split.jsonl (lighting and cluster)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--budget-s", type=float, default=None, help="stop cleanly after this many seconds (resume later)")
    ap.add_argument("--no-windows", action="store_true", help="full frame only (baseline check)")
    a = ap.parse_args(argv)

    from ultralytics import YOLO  # AGPL-3.0: training/ only
    import cv2
    from PIL import Image

    t_start = time.time()
    root = C.resolve_data_root(a.data_root)
    images = C.list_split_images(root, "val", a.limit)
    resolver = C.MetaResolver(a.index)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(a.weights), task="detect")
    print(f"{len(images)} val images under {root}; index={'yes' if resolver.has_index else 'no'}", flush=True)

    # labels + metadata (cheap, done once)
    lab_path = out / "labels.jsonl.gz"
    if not lab_path.exists():
        with gzip.open(lab_path.with_suffix(".tmp"), "wt") as fh:
            for p in images:
                meta = resolver.resolve(p.stem)
                if meta is None:
                    raise SystemExit(f"{p.name}: no modality suffix")
                lab = C.read_yolo_label(C.label_path_for(p))
                with Image.open(p) as im:
                    width, height = im.size
                fh.write(json.dumps({"name": p.stem, "path": p.name, "hw": [height, width], "label": lab.round(6).tolist(),
                                     "meta": [meta.name, meta.source, meta.modality, meta.slice, meta.lighting, meta.cluster]}) + "\n")
        lab_path.with_suffix(".tmp").rename(lab_path)

    pool = ThreadPoolExecutor(max_workers=4)
    n_pass = {"full": 0, "window": 0}
    gpu_s = 0.0
    done_chunks = 0
    for c0 in range(0, len(images), a.chunk):
        shard = out / f"passes_{c0 // a.chunk:05d}.npz"
        if shard.exists():
            continue
        if a.budget_s and time.time() - t_start > a.budget_s:
            print("time budget reached, stopping cleanly", flush=True)
            break
        chunk = images[c0:c0 + a.chunk]
        t0 = time.time()
        raw = P.predict_raw(a.weights, chunk, imgsz=640, batch=16, device=a.device, conf_floor=FULL_FLOOR, progress_every=0)
        records = []
        hws = {}
        for i, (b, c, k, hw) in enumerate(zip(raw.boxes, raw.conf, raw.cls, raw.hw)):
            records.append((c0 + i, None, *reduce_pass(b, c, k)))
            hws[c0 + i] = (int(hw[0]), int(hw[1]))
        n_pass["full"] += len(chunk)
        if not a.no_windows:
            imgs = list(pool.map(lambda p: cv2.imread(str(p)), chunk))
            jobs = []   # (img_idx, window, crop)
            for i, im in enumerate(imgs):
                h, w = im.shape[:2]
                assert (h, w) == hws[c0 + i], (chunk[i].name, (h, w), hws[c0 + i])
                for win in TC.all_windows(w, h):
                    x1, y1, x2, y2 = win
                    jobs.append((c0 + i, win, np.ascontiguousarray(im[y1:y2, x1:x2])))
            del imgs
            tg = time.time()
            with P.no_nms_time_limit():
                for j0 in range(0, len(jobs), a.batch):   # a list of arrays is ONE batch to Ultralytics: chunk it here
                    sub = jobs[j0:j0 + a.batch]
                    results = model.predict(source=[j[2] for j in sub], imgsz=640, conf=WINDOW_FLOOR, iou=1.0,
                                            max_det=P.RAW_MAX_DET, verbose=False, device=a.device)
                    for (img_idx, win, _), r in zip(sub, results):
                        b = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                        cf = r.boxes.conf.cpu().numpy().astype(np.float32)
                        cl = r.boxes.cls.cpu().numpy().astype(np.int16)
                        b, cf, cl = reduce_pass(b, cf, cl)
                        b[:, [0, 2]] += win[0]
                        b[:, [1, 3]] += win[1]
                        records.append((img_idx, win, b, cf, cl))
            gpu_s += time.time() - tg
            n_pass["window"] += len(jobs)
            del jobs
        save_chunk(shard, records)
        done_chunks += 1
        print(f"chunk {c0 // a.chunk} ({c0 + len(chunk)}/{len(images)}) {time.time() - t0:.0f}s, "
              f"elapsed {(time.time() - t_start) / 60:.1f} min, passes so far {n_pass}", flush=True)

    (out / "run.json").write_text(json.dumps({
        "weights": str(a.weights), "weights_sha256": C.sha256_file(Path(a.weights)), "n_images": len(images),
        "chunk": a.chunk, "window_floor": WINDOW_FLOOR, "full_floor": FULL_FLOOR, "layouts": list(TC.LAYOUTS),
        "passes_this_session": n_pass, "window_gpu_s": gpu_s, "wall_s": time.time() - t_start,
        "host": C.host_info(),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
