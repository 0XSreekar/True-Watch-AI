"""Score tiled inference from the per-pass cache written by tiled_infer.py (CPU only, no Ultralytics).

Merge = edge/pipeline/tiling.py (shared with the edge). Matching, AP and size buckets = _metrics.py via
evaluate.py's own scoring functions, so the definitions are the ones behind eval_final.json.

Protocol (no leakage)
---------------------
* Val is split by hash of the correlated-frame cluster into a TUNING half and a REPORT half; every source
  and both modalities are present in both halves.
* Merge parameters (window score floor, merge IoU, truncation rule) are chosen on the tuning half only:
  maximise day person AP50 subject to day person precision at 0.35 within --prec-drop of the full-frame
  baseline and IR person AP50 within --ir-drop of it.
* The chosen settings are then reported on the held-out half and on the full val.

Subcommands: baseline | tune | report
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
import _metrics as M  # noqa: E402
import evaluate as E  # noqa: E402
import tiled_common as TC  # noqa: E402

PERSON = 0
REPORT_CONF = 0.35
FULL_CONF = 0.001
IOU = 0.7
MAX_DET = 300
IMAGES: list[dict] = []     # filled by load(); module-level so forked workers share it


def load(cache: Path) -> list[dict]:
    labels = []
    with gzip.open(cache / "labels.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            n, src, mod, sl, lt, cl = r["meta"]
            h, w = r["hw"]
            lab = np.asarray(r["label"], dtype=np.float64).reshape(-1, 5)
            cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
            labels.append({
                "meta": C.ImageMeta(n, src, mod, sl, lt, cl), "hw": (h, w), "half": TC.half_of(cl),
                "gt_cls": lab[:, 0].astype(np.int64),
                "gt_box": np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) if len(lab) else np.zeros((0, 4)),
                "gt_h": np.array([C.object_height_px(v, h, w, 640, "input") for v in lab[:, 4]]) if len(lab) else np.zeros(0),
                "passes": {},
            })
    for shard in sorted(cache.glob("passes_*.npz")):
        z = np.load(shard)
        img, wins, counts = z["img"], z["win"], z["counts"]
        boxes, conf, cls = z["boxes"], z["conf"], z["cls"].astype(np.int64)   # npz members decompress on every access: once only
        off = np.concatenate(([0], np.cumsum(counts)))
        for k in range(len(img)):
            win = None if wins[k][0] < 0 else tuple(int(v) for v in wins[k])
            s = slice(off[k], off[k + 1])
            labels[int(img[k])]["passes"][win] = (boxes[s].copy(), conf[s].copy(), cls[s].copy())
    missing = [i for i, im in enumerate(labels) if None not in im["passes"]]
    if missing:
        raise SystemExit(f"{len(missing)} images have no full-frame pass in the cache (first: {missing[:3]})")
    return labels


def pr_h_of(box, h, w):
    return np.array([C.object_height_px((y2 - y1) / h, h, w, 640, "input") for _, y1, _, y2 in box]) if len(box) else np.zeros(0)


def to_image_data(im: dict, layout, tcfg, full_conf: float, iou: float) -> M.ImageData:
    h, w = im["hw"]
    if layout is None:
        b, c, k = im["passes"][None]
        b, c, k = b[c >= full_conf], c[c >= full_conf], k[c >= full_conf]
    else:
        b, c, k = TC.merge_image(im["passes"], w, h, layout, tcfg, full_conf=full_conf, iou=iou, max_det=MAX_DET)
    b = b.astype(np.float64)
    return M.ImageData(im["meta"], im["gt_cls"], im["gt_box"], im["gt_h"], k.astype(np.int64), b,
                       c.astype(np.float64), pr_h_of(b, h, w))


def passes_per_frame(ids, layout) -> float:
    if layout is None:
        return 1.0
    return float(np.mean([1 + sum(1 for _ in TC.layout_windows(layout, IMAGES[i]["hw"][1], IMAGES[i]["hw"][0])) for i in ids]))


def quick(ims: list[M.ImageData]) -> dict:
    m = M.collect_matches(ims, PERSON)
    p, r, _, _ = M.pr_at_confs(m, np.array([REPORT_CONF]))
    return {"n_gt": int(m.n_gt), "ap50": M.average_precision(m, 0), "precision": float(p[0]), "recall": float(r[0])}


def run_one(job):
    """Worker: one merge configuration over the given image ids; person AP50/precision/recall per slice."""
    layout, tcfg, ids = job
    ims = [to_image_data(IMAGES[i], layout, tcfg, FULL_CONF, IOU if tcfg is None or tcfg.merge_iou is None else tcfg.merge_iou)
           for i in ids]
    groups = {"day": [], "ir": [], "day/daylight": []}
    for im in ims:
        groups[im.meta.slice].append(im)
        if im.meta.slice == "day" and im.meta.lighting == "daylight":
            groups["day/daylight"].append(im)
    return {k: quick(v) for k, v in groups.items() if v}


def ids_of(half: int | None) -> list[int]:
    return [i for i, im in enumerate(IMAGES) if half is None or im["half"] == half]


def param_grid(layout):
    for tile_conf, merge_iou, trunc in itertools.product((0.02, 0.05, 0.1, 0.2, 0.3), (0.4, 0.5, 0.6, 0.7), (0.0, 0.5)):
        yield TC.variant(layout, tile_conf=tile_conf, merge_iou=merge_iou, trunc_ioa=trunc)


def cfg_key(layout, t):
    return f"{layout}|tc={t.tile_conf}|iou={t.merge_iou}|trunc={t.trunc_ioa}"


def baseline_cmd(a):
    ids = ids_of(None)
    res = run_one((None, None, ids))
    full = E.score_all([to_image_data(IMAGES[i], None, None, FULL_CONF, IOU) for i in ids], ["person", "two_wheeler", "car", "truck", "cart"], settings())
    ref = json.loads((C.RESULTS_DIR / "eval_final.json").read_text())["slices"]
    rows = []
    for sl in ("day", "day/daylight", "ir"):
        mine = full[sl]["classes"]["person"]["ap50"]
        off = ref[sl]["classes"]["person"]["ap50"]
        mm, om = full[sl]["mean_over_present_classes"]["ap50"], ref[sl]["mean_over_present_classes"]["ap50"]
        rows.append({"slice": sl, "person_ap50": mine, "official_person_ap50": off, "diff": mine - off,
                     "map50": mm, "official_map50": om, "map50_diff": mm - om})
    for r in rows:
        print(r)
    (Path(a.out) / "baseline_check.json").write_text(json.dumps(rows, indent=1))
    return rows


def settings():
    return E.Settings(weights=Path("-"), imgsz=640, batch=16, device=None, conf_floor=0.001, nms_iou=0.7, max_det=300,
                      report_conf=REPORT_CONF, n_boot=0, size_basis="input", size_classes=("person",))


def tune_cmd(a):
    tids = ids_of(0)
    base = run_one((None, None, tids))
    print("baseline (tuning half):", json.dumps(base), flush=True)
    jobs, keys = [], []
    for layout in a.layouts:
        for t in param_grid(layout):
            jobs.append((layout, t, [i for i in tids if IMAGES[i]["meta"].slice == "day"]))
            keys.append((layout, t))
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs) as pool:
        day_res = pool.map(run_one, jobs, chunksize=1)
    print(f"day grid {len(jobs)} configs in {time.time() - t0:.0f}s", flush=True)
    grid = []
    for (layout, t), r in zip(keys, day_res):
        grid.append({"layout": layout, "tile_conf": t.tile_conf, "merge_iou": t.merge_iou, "trunc_ioa": t.trunc_ioa,
                     "day_ap50": r["day"]["ap50"], "day_precision": r["day"]["precision"], "day_recall": r["day"]["recall"],
                     "daylight_ap50": r["day/daylight"]["ap50"]})
    # IR check for the top candidates per layout that pass the precision constraint
    ir_ids = [i for i in tids if IMAGES[i]["meta"].slice == "ir"]
    chosen = {}
    for layout in a.layouts:
        ok = [g for g in grid if g["layout"] == layout and g["day_precision"] >= base["day"]["precision"] - a.prec_drop]
        ok.sort(key=lambda g: -g["day_ap50"])
        cands = ok[:4]
        with mp.get_context("fork").Pool(min(a.procs, max(len(cands), 1))) as pool:
            irs = pool.map(run_one, [(layout, TC.variant(layout, tile_conf=g["tile_conf"], merge_iou=g["merge_iou"],
                                                          trunc_ioa=g["trunc_ioa"]), ir_ids) for g in cands])
        for g, r in zip(cands, irs):
            g["ir_ap50"] = r["ir"]["ap50"]
            g["ir_ok"] = r["ir"]["ap50"] >= base["ir"]["ap50"] - a.ir_drop
        passing = [g for g in cands if g["ir_ok"]]
        chosen[layout] = {"best_unconstrained_ir": cands[0] if cands else None, "selected": passing[0] if passing else None,
                          "note": None if passing else "no candidate keeps IR within the allowed drop when IR is tiled too"}
    out = Path(a.out)
    with (out / "tuning_grid.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for g in grid for k in g}))
        w.writeheader()
        w.writerows(grid)
    (out / "tuning_selected.json").write_text(json.dumps({"baseline_tuning_half": base, "prec_drop": a.prec_drop,
                                                          "ir_drop": a.ir_drop, "chosen": chosen}, indent=1))
    print(json.dumps(chosen, indent=1))


def report_cmd(a):
    sel = json.loads((Path(a.out) / "tuning_selected.json").read_text())["chosen"]
    out = {"layouts": {}}
    for scope, ids in (("held_out_half", ids_of(1)), ("full_val", ids_of(None)), ("tuning_half", ids_of(0))):
        out[scope] = {"baseline": score_all_slices(None, None, ids), "n_images": len(ids)}
        for layout in a.layouts:
            s = sel[layout]["selected"] or sel[layout]["best_unconstrained_ir"]
            if s is None:
                continue
            t = TC.variant(layout, tile_conf=s["tile_conf"], merge_iou=s["merge_iou"], trunc_ioa=s["trunc_ioa"])
            res = score_all_slices(layout, t, ids)
            res["passes_per_frame"] = {sl: passes_per_frame([i for i in ids if IMAGES[i]["meta"].slice == sl], layout) for sl in ("day", "ir")}
            res["params"] = {"tile_conf": t.tile_conf, "merge_iou": t.merge_iou, "trunc_ioa": t.trunc_ioa, "trunc_margin": t.trunc_margin}
            out[scope][layout] = res
    (Path(a.out) / "tiling_report.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote tiling_report.json")


def score_all_slices(layout, tcfg, ids):
    iou = IOU if tcfg is None or tcfg.merge_iou is None else tcfg.merge_iou
    ims = [to_image_data(IMAGES[i], layout, tcfg, FULL_CONF, iou) for i in ids]
    return E.score_all(ims, ["person", "two_wheeler", "car", "truck", "cart"], settings())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("baseline", "tune", "report"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layouts", nargs="+", default=list(TC.LAYOUTS))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--prec-drop", type=float, default=0.05)
    ap.add_argument("--ir-drop", type=float, default=0.01)
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    global IMAGES
    IMAGES = load(Path(a.cache))
    print(f"{len(IMAGES)} images, halves {np.bincount([im['half'] for im in IMAGES]).tolist()}", flush=True)
    {"baseline": baseline_cmd, "tune": tune_cmd, "report": report_cmd}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
