#!/usr/bin/env python3
"""Sweep confidence and NMS IoU and report two operating points per modality, as two numbers.

WHY this exists. A detector has no single "accuracy": every confidence threshold trades missed
people against false detections. Two questions have two different answers and this script keeps
them apart instead of averaging them into one flattering figure:

  * F1-OPTIMAL point: the threshold that balances precision and recall best. This is what a
    benchmark would call "the" operating point. It says nothing about how many false alerts an
    operator receives.
  * FALSE-ALERT-BUDGET point: the highest-recall threshold whose false-positive rate fits the
    project target of under N false alerts per camera per day (slide 5). At 8 fps that is a
    handful of false detections in about 700 thousand frames.

The second number is usually the uncomfortable one, and a validation split cannot even measure it:
with a few thousand frames, "zero false positives observed" is compatible with a false-positive
rate hundreds of times above the budget (rule of three: zero events in n trials bounds the rate
only to about 3/n at 95%). The budget status is therefore one of

  met           the 95% upper bound on the false-positive rate is within the budget;
  unresolvable  the observed rate is within the budget but the upper bound is not, so these frames
                cannot tell (the report states how many frames would be needed);
  unreachable   even at confidence 0.99 the observed rate exceeds the budget.

Only "met" is a pass. "unresolvable" is not one.

Inference runs once, with Ultralytics NMS switched off (see _predict.py), or the cached raw
predictions are loaded. NMS is then re-applied here at every IoU on the grid, and the confidence
sweep is a cumulative-count lookup (searchsorted over the sorted detections of _metrics.pr_at_confs),
so the cost is one matching pass per (NMS IoU, slice), not one per confidence.

Day (visible) and IR (LWIR replicated to 3 channels) are swept separately and never blended.
Optional all-negative background frames (--empty-dir) add frames and false-positive opportunities to
the ONE slice the user says they belong to. The test split stays sealed until Phase 11.

Outputs, under --out-dir:
    sweep_<tag>.json                 schema truewatch.sweep.v1
    sweep_<tag>_<slice>_grid.csv     one row per (NMS IoU, confidence); empty cell = no data
    cache/sweep_preds_<tag>.npz      raw predictions, when --weights was given (git-ignored)

Exit codes: 0 done; 2 refused or bad arguments; 3 the input data is unusable.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
import _metrics as M  # noqa: E402
import _predict as P  # noqa: E402

SCHEMA = "truewatch.sweep.v1"
SECONDS_PER_DAY = 86400
RULE_OF_THREE = 3.0          # 95% upper bound on a Poisson mean after observing zero events
CONF_CEILING = 0.99          # "unreachable" is judged at this confidence, so it is always swept
MATCH_IOU_INDEX = 0          # IoU 0.50 in _metrics.IOU_THRS
MATCH_THRS = M.IOU_THRS[:1]  # match once at 0.50 only; the sweep never reports mAP
MIN_CONF = 0.001             # the raw-prediction floor of _predict.py
SLICES = C.MODALITY_SLICES   # ("day", "ir"): the only two keys ever taken from group_by_slice
DEFAULT_CONF_GRID = "0.05:0.95:0.05"
DEFAULT_IOU_GRID = "0.5,0.6,0.7,0.8"
GENERIC_STEMS = {"best", "last", "weights"}
SPLITS = ("train", "val", "test")

# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def fail(message: str, code: int = 2):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _num(value) -> float | None:
    """A JSON-safe number: NaN and None both become None, never 0."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else value


def _display_path(path: Path) -> str:
    """Repo-relative when inside the repo, so committed JSON carries no home-directory prefix."""
    try:
        return str(Path(path).resolve().relative_to(C.REPO_ROOT))
    except ValueError:
        return str(path)


def parse_conf_grid(text: str) -> np.ndarray:
    """'start:stop:step' inclusive of both ends, e.g. 0.05:0.95:0.05 -> 19 values."""
    try:
        start, stop, step = (float(p) for p in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected start:stop:step such as 0.05:0.95:0.05, got {text!r}")
    if step <= 0 or start > stop:
        raise argparse.ArgumentTypeError("need step > 0 and start <= stop")
    if start < MIN_CONF or stop > 1.0:
        raise argparse.ArgumentTypeError(f"confidence must lie in [{MIN_CONF}, 1.0]")
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    return np.round(start + step * np.arange(count), 6)


def parse_iou_grid(text: str) -> np.ndarray:
    try:
        values = sorted({float(p) for p in text.split(",") if p.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers such as 0.5,0.7, got {text!r}")
    if not values or values[0] <= 0 or values[-1] > 1.0:
        raise argparse.ArgumentTypeError("NMS IoU values must lie in (0, 1]")
    return np.array(values)


# --------------------------------------------------------------------------------------------
# The false-alert budget and the statistics behind it
# --------------------------------------------------------------------------------------------


def budget_arithmetic(fps: float, alerts_per_camera_per_day: float, fp_to_alert: float) -> dict:
    """Frames per camera-day and the false positives per frame the target allows.

    fp_to_alert is the fraction of false detections that would surface as an alert. 1.0 is the
    worst case (no fusion, tracking or rule credit): every false detection is an alert.
    """
    frames_per_day = fps * SECONDS_PER_DAY
    budget = alerts_per_camera_per_day / fp_to_alert / frames_per_day
    return {
        "fps": fps,
        "alerts_per_camera_per_day": alerts_per_camera_per_day,
        "fp_to_alert": fp_to_alert,
        "frames_per_day": frames_per_day,
        "budget_fp_per_frame": budget,
        "rule_of_three_min_frames": math.ceil(RULE_OF_THREE / budget),
    }


def _poisson_cdf(k: int, lam: float, log_fact: np.ndarray) -> float:
    """P(X <= k) for X ~ Poisson(lam), summed in log space so large counts do not underflow."""
    i = np.arange(k + 1)
    log_terms = -lam + i * math.log(lam) - log_fact[: k + 1]
    peak = float(log_terms.max())
    return math.exp(peak) * float(np.exp(log_terms - peak).sum())


def poisson_upper95(k: int) -> float:
    """95% one-sided upper confidence limit on a Poisson mean after observing k events.

    k = 0 returns 3.0, the rule of three the brief specifies (the exact value is 2.996). For k > 0
    it is the exact limit: the mean at which seeing k or fewer events has probability 5%.
    """
    if k < 0:
        raise ValueError("count must be non-negative")
    if k == 0:
        return RULE_OF_THREE
    log_fact = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, k + 1, dtype=np.float64)))))
    lo, hi = float(k), float(k) + 10.0 * math.sqrt(k + 1.0) + 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _poisson_cdf(k, mid, log_fact) > 0.05:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fp_rate_upper95(fp_count: int, n_frames: int) -> float | None:
    """95% upper bound on false positives per frame. Frames are treated as independent."""
    if n_frames <= 0:
        return None
    return poisson_upper95(int(fp_count)) / n_frames


def frames_needed(fp_count: int, budget: float) -> int:
    """Fewest frames in which `fp_count` false positives would still bound the rate within budget.

    A lower limit: it assumes no further false positives appear as frames are added. For zero
    false positives it is 3 / budget, the smallest sample that could ever resolve the budget.
    """
    return math.ceil(poisson_upper95(int(fp_count)) / budget)


# --------------------------------------------------------------------------------------------
# The sweep itself
# --------------------------------------------------------------------------------------------


def count_grid(per_iou: dict[float, list[M.ImageData]], class_id: int, confs: np.ndarray):
    """(tp, fp) arrays of shape (n_iou, n_conf) and the slice's ground-truth count.

    One matching pass per NMS IoU (M.collect_matches, IoU 0.50); every confidence is then a
    binary-search lookup in the sorted cumulative TP/FP curve (M.pr_at_confs).
    """
    ious = sorted(per_iou)
    tp = np.zeros((len(ious), len(confs)))
    fp = np.zeros((len(ious), len(confs)))
    n_gt = 0
    for row, iou in enumerate(ious):
        matches = M.collect_matches(per_iou[iou], class_id, thrs=MATCH_THRS)
        _, _, tp[row], fp[row] = M.pr_at_confs(matches, confs, MATCH_IOU_INDEX)
        n_gt = matches.n_gt
    return np.array(ious), tp, fp, n_gt


def derived(tp: np.ndarray, fp: np.ndarray, n_gt: int, n_frames: int) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and FP per frame; NaN wherever the quantity has no definition."""
    fn = n_gt - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = tp / n_gt if n_gt > 0 else np.full(tp.shape, np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan) if n_gt > 0 else np.full(tp.shape, np.nan)
        fp_per_frame = fp / n_frames if n_frames > 0 else np.full(fp.shape, np.nan)
    return {"fn": fn, "precision": precision, "recall": recall, "f1": f1, "fp_per_frame": fp_per_frame}


def _pick(order_keys: list[np.ndarray], candidates: np.ndarray) -> tuple[int, int] | None:
    """Best (iou_index, conf_index) among the boolean mask `candidates`.

    order_keys are sort keys, most significant first, each the same shape as `candidates`.
    """
    idx = np.argwhere(candidates)
    if len(idx) == 0:
        return None
    keys = [k[candidates] for k in order_keys]
    best = np.lexsort(tuple(reversed(keys)))[0]
    return int(idx[best][0]), int(idx[best][1])


def f1_optimal_point(ious, confs, d, n_gt) -> dict:
    """Highest F1 over the (NMS IoU, confidence) grid. Ties go to the higher confidence."""
    empty = {"conf": None, "nms_iou": None, "f1": None, "precision": None, "recall": None,
             "fp_per_frame": None, "note": "no person ground truth in this slice, so F1 is undefined"}
    if n_gt == 0:
        return empty
    f1 = np.round(np.nan_to_num(d["f1"], nan=-1.0), 12)
    conf_grid = np.broadcast_to(confs[None, :], f1.shape)
    iou_grid = np.broadcast_to(ious[:, None], f1.shape)
    if not (f1 > 0).any():
        # Zero true positives everywhere: any "best" cell would be an arbitrary pick among ties.
        empty["note"] = (f"no person is detected at any confidence from {confs.min():g} up, so F1 is 0 "
                         "everywhere and there is no F1-optimal point")
        return empty
    pick = _pick([-f1, -conf_grid, iou_grid], f1 > 0)
    if pick is None:
        return empty
    r, c = pick
    return {
        "conf": float(confs[c]), "nms_iou": float(ious[r]), "f1": _num(d["f1"][r, c]),
        "precision": _num(d["precision"][r, c]), "recall": _num(d["recall"][r, c]),
        "fp_per_frame": _num(d["fp_per_frame"][r, c]),
    }


def false_alert_budget_point(ious, confs, tp, fp, d, n_gt, n_frames, budget: float) -> dict:
    """The highest-recall point whose false-positive rate fits the budget, with an honest status.

    Recall never rises with confidence, so the best point within budget is the lowest confidence
    that stays within it. "met" needs the 95% UPPER bound within budget, not the observed rate.
    """
    min_frames = math.ceil(RULE_OF_THREE / budget)
    base = {"status": "unresolvable", "vacuous": False, "conf": None, "nms_iou": None, "recall": None, "fp_count": None,
            "fp_per_frame": None, "fp_per_frame_upper95": None, "frames_needed_to_resolve": min_frames,
            "min_frames_any_resolution": min_frames, "n_frames": n_frames}
    if n_frames <= 0:
        base["note"] = "no frames in this slice, so the false-positive rate cannot be measured"
        return base

    fp_int = fp.astype(np.int64)
    recall0 = np.nan_to_num(d["recall"], nan=0.0)
    conf_grid = np.broadcast_to(confs[None, :], fp.shape)
    iou_grid = np.broadcast_to(ious[:, None], fp.shape)
    within = fp <= budget * n_frames + 1e-9

    # The upper bound depends only on the count, and only counts <= budget * n_frames can be
    # within budget, so this loop runs over a handful of distinct values, not the whole grid.
    passes = np.zeros(fp.shape, dtype=bool)
    for count in np.unique(fp_int[within]):
        if fp_rate_upper95(int(count), n_frames) <= budget:
            passes |= within & (fp_int == count)
    keys = [-recall0, fp, -conf_grid, iou_grid]

    pick = _pick(keys, passes)
    status = "met"
    if pick is None:
        pick = _pick(keys, within)
        status = "unresolvable"
    if pick is not None:
        r, c = pick
        count = int(fp_int[r, c])
        vacuous = n_gt > 0 and int(tp[r, c]) == 0
        point = {
            "status": status, "vacuous": vacuous, "conf": float(confs[c]), "nms_iou": float(ious[r]),
            "recall": _num(d["recall"][r, c]), "fp_count": count,
            "fp_per_frame": _num(d["fp_per_frame"][r, c]),
            "fp_per_frame_upper95": _num(fp_rate_upper95(count, n_frames)),
            "frames_needed_to_resolve": 0 if status == "met" else frames_needed(count, budget),
            "min_frames_any_resolution": min_frames, "n_frames": n_frames,
        }
        if status == "met":
            point["note"] = (
                f"{count} false positive(s) in {n_frames} frames; the 95% upper bound "
                f"{point['fp_per_frame_upper95']:.3g}/frame is within the budget {budget:.3g}/frame. "
                "Detector-level proxy only: alert-level rates are decided by fusion and tracking in Phases 4-5."
            )
        else:
            point["note"] = (
                f"{count} false positive(s) in {n_frames} frames is within the budget, but the 95% upper "
                f"bound {point['fp_per_frame_upper95']:.3g}/frame is not, so these frames cannot show the "
                f"budget is met. At least {point['frames_needed_to_resolve']:,} frames are needed (and only if "
                f"no false positive appears in them); the rule-of-three minimum is {min_frames:,}."
            )
        if vacuous:
            point["note"] += (" VACUOUS: recall is 0 here, the detector finds no person at all at this point, and "
                              "a zero false-positive rate is trivially reached by detecting nothing.")
        return point

    # No confidence up to the ceiling keeps the observed rate within budget: report the least bad.
    r, c = _pick([fp, -recall0, -conf_grid, iou_grid], np.ones(fp.shape, dtype=bool))
    count = int(fp_int[r, c])
    base.update({
        "status": "unreachable", "conf": float(confs[c]), "nms_iou": float(ious[r]),
        "recall": _num(d["recall"][r, c]), "fp_count": count,
        "fp_per_frame": _num(d["fp_per_frame"][r, c]),
        "fp_per_frame_upper95": _num(fp_rate_upper95(count, n_frames)),
        "frames_needed_to_resolve": None,
    })
    over = base["fp_per_frame"] / budget if budget > 0 else float("inf")
    base["note"] = (
        f"no confidence up to {confs.max():.2f} keeps the false-positive rate within the budget. The least bad "
        f"point (conf {base['conf']:.2f}, NMS IoU {base['nms_iou']:.2f}) still has {count} false positive(s) in "
        f"{n_frames} frames = {over:,.0f}x the budget. It does NOT meet it. More frames cannot fix this; the "
        "detector must produce fewer false positives."
    )
    return base


def sweep_slice(name: str, per_iou: dict[float, list[M.ImageData]], class_id: int,
                confs: np.ndarray, budget: float, out_csv: Path) -> dict:
    ious, tp, fp, n_gt = count_grid(per_iou, class_id, confs)
    n_frames = len(next(iter(per_iou.values())))
    d = derived(tp, fp, n_gt, n_frames)
    write_grid_csv(out_csv, name, ious, confs, tp, fp, d, n_gt, n_frames)
    return {
        "n_frames": n_frames,
        "n_gt": n_gt,
        "f1_optimal": f1_optimal_point(ious, confs, d, n_gt),
        "false_alert_budget": false_alert_budget_point(ious, confs, tp, fp, d, n_gt, n_frames, budget),
        "grid_csv": _display_path(out_csv),
    }


GRID_COLUMNS = ["slice", "nms_iou", "conf", "n_frames", "n_gt", "tp", "fp", "fn",
                "precision", "recall", "f1", "fp_per_frame"]


def write_grid_csv(path: Path, slice_name: str, ious, confs, tp, fp, d, n_gt: int, n_frames: int) -> None:
    """Every (NMS IoU, confidence) cell. An empty cell means the quantity has no definition there."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def cell(x):
        x = _num(x)
        return "" if x is None else f"{x:.6g}"

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(GRID_COLUMNS)
        for r, iou in enumerate(ious):
            for c, conf in enumerate(confs):
                writer.writerow([
                    slice_name, f"{iou:g}", f"{conf:g}", n_frames, n_gt, int(tp[r, c]), int(fp[r, c]),
                    int(d["fn"][r, c]), cell(d["precision"][r, c]), cell(d["recall"][r, c]),
                    cell(d["f1"][r, c]), cell(d["fp_per_frame"][r, c]),
                ])


# --------------------------------------------------------------------------------------------
# Inputs: predictions, background frames, sealing
# --------------------------------------------------------------------------------------------


def sealed_test_message(what: str) -> str:
    return (
        f"{what} would read images/test, and the test split is sealed until Phase 11 so that it stays a "
        "single, honest final measurement (DATASET_SPEC 2.5: every look at test while tuning makes the "
        "final number optimistic). Phase 2 measures on val. Pass --unseal-test only for the Phase 11 report."
    )


def _split_dirs(paths: list[str]) -> set[str]:
    return {Path(p).parent.name for p in paths}


def _reroot(raw: P.RawPreds, root: Path, split: str) -> int:
    """Point cached paths that no longer exist at root/images/<split>/<name>. Returns how many moved."""
    moved = 0
    for i, old in enumerate(raw.paths):
        if Path(old).exists():
            continue
        candidate = root / "images" / split / Path(old).name
        if candidate.exists():
            raw.paths[i] = str(candidate)
            moved += 1
    return moved


def _infer_root(paths: list[str]) -> Path | None:
    parts = Path(paths[0]).parts if paths else ()
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            return Path(*parts[:i])
    return None


def find_index(root: Path | None, explicit: str | None) -> Path | None:
    """The Phase 1 split index: --index, else next to the dataset root, else beside its parent."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            fail(f"--index {path} does not exist. Point it at processed/index/split.jsonl or omit the flag.")
        return path
    if root is not None:
        for candidate in (root / "index" / "split.jsonl", root.parent / "index" / "split.jsonl"):
            if candidate.is_file():
                return candidate
    return None


def is_synthetic(root: Path | None) -> bool:
    """True for the fixture _synth.py writes; its data.yaml says so in its first comment."""
    try:
        return root is not None and "synthetic fixture" in (root / "data.yaml").read_text(encoding="utf-8")
    except OSError:
        return False


def empty_frame_images(raw: P.RawPreds, class_id: int, nms_iou: float, modality: str,
                       imgsz: int, size_basis: str) -> list[M.ImageData]:
    """ImageData for background frames: no ground truth, so every detection is a false positive."""
    slice_name = C.slice_of_modality(modality)
    out = []
    for path, (h, w), boxes, conf, cls in zip(raw.paths, raw.hw, raw.boxes, raw.conf, raw.cls):
        keep = cls == class_id
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        idx = P.nms(boxes, conf, cls, nms_iou)
        pr_box = boxes[idx].astype(np.float64)
        pr_h = np.array([C.object_height_px((y2 - y1) / h, h, w, imgsz, size_basis)
                         for _, y1, _, y2 in pr_box]) if len(pr_box) else np.zeros(0)
        stem = Path(path).stem
        meta = C.ImageMeta(name=stem, source="empty", modality=modality, slice=slice_name,
                           lighting="n/a", cluster=stem)
        out.append(M.ImageData(meta, np.zeros(0, np.int64), np.zeros((0, 4)), np.zeros(0),
                               cls[idx].astype(np.int64), pr_box, conf[idx].astype(np.float64), pr_h))
    return out


def check_empty_dir(folder: Path, unseal: bool) -> list[Path]:
    if not folder.is_dir():
        fail(f"--empty-dir {folder} is not a directory. Point it at a folder of background frames.")
    if "test" in folder.parts and "images" in folder.parts and not unseal:
        fail(sealed_test_message(f"--empty-dir {folder}"))
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in C.IMAGE_SUFFIXES)
    if not images:
        fail(f"--empty-dir {folder} holds no images ({', '.join(C.IMAGE_SUFFIXES)}).")
    for image in images:
        try:
            label = C.label_path_for(image)
        except ValueError:
            continue
        if label.exists() and label.read_text(encoding="utf-8").strip():
            fail(f"{image.name} has a non-empty label file, so it is not a background frame. "
                 "--empty-dir must hold frames with nothing to detect; a labelled frame belongs in the split.")
    return images


def images_by_iou(raw: P.RawPreds, resolver: C.MetaResolver, ious: np.ndarray, imgsz: int, basis: str):
    """{nms_iou: ImageData list}, plus the composition report. Prints the label warning once."""
    out, report = {}, {}
    for n, iou in enumerate(ious):
        sink = io.StringIO()
        try:
            with (contextlib.nullcontext() if n == 0 else contextlib.redirect_stdout(sink)):
                out[float(iou)], report = P.image_data_from_raw(raw, resolver, float(iou), imgsz=imgsz, size_basis=basis)
        except SystemExit as exc:
            if isinstance(exc.code, str):
                fail(exc.code, 3)
            raise
    return out, report


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------


def _f(value, digits: int = 3) -> str:
    return C.fmt(value, digits)


def _sci(value) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def render_report(payload: dict) -> str:
    a = payload["assumptions"]
    lines = [
        f"TRUEWATCH confidence and NMS sweep   tag={payload['tag']}   split={payload['split']}   class={a['class']}",
        f"weights: {payload['weights'].get('path')}  sha256 {(payload['weights'].get('sha256') or 'n/a')[:12]}",
        f"host: {payload['host'].get('cpu')} / {payload['host'].get('platform')} "
        f"({payload['host'].get('cpu_count_usable')} usable cores)",
        f"grid: confidence {payload['grid']['conf_min']:g} to {payload['grid']['conf_max']:g} "
        f"({payload['grid']['n_conf']} values, including the {CONF_CEILING} ceiling), "
        f"NMS IoU {payload['grid']['nms_iou']}, matching at IoU 0.50",
        "",
        "False-alert budget arithmetic",
        f"  {a['fps']:g} fps x {SECONDS_PER_DAY:,} s = {a['frames_per_day']:,.0f} frames per camera per day",
        f"  {a['alerts_per_camera_per_day']:g} alerts per camera per day / {a['fp_to_alert']:g} "
        f"(fraction of false detections that become alerts) = "
        f"{a['alerts_per_camera_per_day'] / a['fp_to_alert']:g} false detections per day allowed",
        f"  budget = {a['budget_fp_per_frame']:.3e} false positives per frame",
        f"  rule of three: zero false positives in n frames bounds the rate only to 3/n at 95%, so a sample "
        f"needs at least {a['rule_of_three_min_frames']:,} frames before the budget can be resolved at all",
        "",
    ]
    for name in SLICES:
        s = payload["slices"][name]
        f1, bud = s["f1_optimal"], s["false_alert_budget"]
        modality = "visible camera" if name == "day" else "LWIR replicated to 3 channels"
        empties = f" + {s['n_empty_frames']} background" if s.get("n_empty_frames") else ""
        lines.append(f"Slice {name} ({modality}): {s['n_frames_split']} {payload['split']} frames{empties} = "
                     f"{s['n_frames']} frames, {s['n_gt']} person ground-truth boxes")
        if f1["f1"] is None:
            lines.append(f"  F1-optimal point:    n/a ({f1['note']})")
        else:
            lines.append(
                f"  F1-optimal point:    conf {f1['conf']:.2f}  NMS IoU {f1['nms_iou']:.2f}  F1 {_f(f1['f1'])}  "
                f"precision {_f(f1['precision'])}  recall {_f(f1['recall'])}  false positives per frame {_sci(f1['fp_per_frame'])}"
            )
            if f1["fp_per_frame"]:
                ratio = f1["fp_per_frame"] / a["budget_fp_per_frame"]
                lines.append(f"                       its false-positive rate is {ratio:,.0f}x the budget: the F1 point is not the budget point")
            else:
                lines.append("                       no false positive was observed at this point (see the budget line for what that can prove)")
        if bud["conf"] is None:
            lines.append(f"  False-alert budget:  {bud['status'].upper()}  ({bud['note']})")
        else:
            label = ("least-bad point, NOT within budget" if bud["status"] == "unreachable"
                     else "VACUOUS point (recall 0)" if bud.get("vacuous") else "point")
            lines.append(
                f"  False-alert budget:  {bud['status'].upper()}  {label}: conf {bud['conf']:.2f}  NMS IoU {bud['nms_iou']:.2f}  "
                f"recall {_f(bud['recall'])}  false positives {bud['fp_count']} in {bud['n_frames']} frames "
                f"(observed {_sci(bud['fp_per_frame'])}/frame, 95% upper bound {_sci(bud['fp_per_frame_upper95'])}/frame)"
            )
        can = {"met": "YES", "unresolvable": "NO", "unreachable": "NO"}[bud["status"]]
        if bud["status"] == "unreachable":
            need = "more frames cannot help; the false-positive rate itself exceeds the budget"
        elif bud["status"] == "met":
            need = "no more frames are needed at this point"
        else:
            need = (f"{bud['frames_needed_to_resolve']:,} frames are needed at the very least (rule-of-three minimum "
                    f"{bud['min_frames_any_resolution']:,}) against {bud['n_frames']:,} available"
                    + (f", {bud['frames_needed_to_resolve'] / max(bud['n_frames'], 1):,.0f}x more" if bud["n_frames"] else ""))
        lines.append(f"  Can the budget be resolved with these frames? {can}. {need}.")
        lines.append(f"  grid: {s['grid_csv']}")
        lines.append("")
    lines.append("Read with care")
    for note in payload["notes"]:
        lines.append(f"  - {note}")
    return "\n".join(lines)


def build_notes(has_empty: bool, split: str, synthetic: bool) -> list[str]:
    notes = [
        "The false-positive count is a detector-level proxy: every unmatched person detection at IoU 0.50 "
        "counts as a false alert (fp_to_alert as given). Fusion, tracking and rules (Phases 4-5) decide what "
        "becomes an alert; this is the rate before any of them, and nothing here shows how much they would remove.",
        "Frames are treated as independent trials. Frames of one sequence are correlated, so the effective "
        "sample is smaller than the frame count and the upper bound printed here is optimistic.",
        "Matching is done once at the confidence floor and then filtered by threshold, as Ultralytics and "
        "evaluate.py do, so these counts agree with the evaluation script's precision and recall.",
    ]
    if not has_empty:
        notes.append(
            f"The {split} split keeps its natural composition (DATASET_SPEC 3.5): most frames contain people. Its "
            "false-positive rate is not the rate on a border camera's empty hours. Pass --empty-dir with "
            "--empty-modality for background frames."
        )
    else:
        notes.append(
            "Background frames were added to ONE slice (the one named by --empty-modality). Their number and origin "
            "decide whether the rate resembles a real camera's empty hours; frames from one clip count as one trial each."
        )
    if split == "train":
        notes.append("train split: the detector fitted these images, so these numbers are optimistic and are not a result")
    if split == "test":
        notes.append("TEST SPLIT UNSEALED with --unseal-test. Every number here is a final-report number, not a tuning number.")
    if synthetic:
        notes.append("the dataset root is the synthetic fixture: these numbers prove the tooling runs and are NOT results")
    return notes


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_argument_group("predictions (one of the two is required)")
    src.add_argument("--weights", help="a .pt or .onnx detector; inference runs once and the raw predictions are cached")
    src.add_argument("--preds", help="a raw-prediction cache (.npz) written by evaluate.py or an earlier sweep")
    data = ap.add_argument_group("data")
    data.add_argument("--data-root", help="dataset root (images/, labels/, data.yaml); with --preds it re-roots moved paths")
    data.add_argument("--split", default="val", help="split to sweep; test is sealed until Phase 11")
    data.add_argument("--index", help="processed/index/split.jsonl for source and lighting metadata (default: found next to the dataset root; the numbers do not depend on it)")
    data.add_argument("--empty-dir", help="folder of all-negative background frames (needs --weights to score them)")
    data.add_argument("--empty-modality", choices=list(SLICES), help="which slice the --empty-dir frames belong to (required with --empty-dir)")
    inf = ap.add_argument_group("inference")
    inf.add_argument("--imgsz", type=int, default=640, help="inference size in pixels")
    inf.add_argument("--batch", type=int, default=16)
    inf.add_argument("--device", default=None, help="cpu, mps, or a CUDA index such as 0 (default: Ultralytics chooses)")
    inf.add_argument("--size-basis", choices=["input", "stored"], default="input", help="object-height basis recorded on each image (no effect on the sweep)")
    sw = ap.add_argument_group("sweep")
    sw.add_argument("--conf-grid", type=parse_conf_grid, default=DEFAULT_CONF_GRID, metavar="START:STOP:STEP", help="confidence thresholds; 0.99 is always added to judge reachability")
    sw.add_argument("--iou-grid", type=parse_iou_grid, default=DEFAULT_IOU_GRID, metavar="A,B,C", help="NMS IoU thresholds")
    sw.add_argument("--class", dest="class_name", default="person", help="class to sweep")
    bud = ap.add_argument_group("false-alert budget")
    bud.add_argument("--fps", type=float, default=8.0, help="frames per second the edge processes per camera (TARGET_FPS)")
    bud.add_argument("--alerts-per-camera-day", type=float, default=5.0, help="allowed false alerts per camera per day (slide 5 target: under 5)")
    bud.add_argument("--fp-to-alert", type=float, default=1.0, help="fraction of false detections that become an alert; 1.0 is the worst case")
    out = ap.add_argument_group("output")
    out.add_argument("--tag", help="name for the output files (default: derived from the weights or cache name)")
    out.add_argument("--out-dir", default=str(C.RESULTS_DIR), help="where sweep_<tag>.json, the grid CSVs and the cache go")
    out.add_argument("--unseal-test", action="store_true", help="allow reading images/test (Phase 11 final report only)")
    return ap


def default_tag(args) -> str:
    if args.preds:
        stem = Path(args.preds).stem
        return stem[len("preds_"):] if stem.startswith("preds_") else stem
    path = Path(args.weights)
    if path.stem in GENERIC_STEMS and path.parent.parent.name:
        return f"{path.parent.parent.name}_{path.stem}"
    return path.stem


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if not args.weights and not args.preds:
        fail("give --weights (run inference now) or --preds (reuse a cached raw-prediction .npz).")
    if args.empty_dir and not args.empty_modality:
        fail("--empty-dir needs --empty-modality day|ir: a background frame belongs to exactly one slice, and it "
             "is never added to both because that would attribute one camera's false positives to the other.")
    if args.empty_modality and not args.empty_dir:
        fail("--empty-modality has no effect without --empty-dir.")
    if args.empty_dir and not args.weights:
        fail("--empty-dir frames must be scored by the model: add --weights (a --preds cache holds only the split's frames).")
    if args.fps <= 0 or args.alerts_per_camera_day <= 0 or not 0 < args.fp_to_alert <= 1:
        fail("need --fps > 0, --alerts-per-camera-day > 0 and 0 < --fp-to-alert <= 1.")
    tag = args.tag or default_tag(args)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        fail(f"--tag {tag!r} may contain only letters, digits, underscore, dot and hyphen.")
    if args.split not in SPLITS:
        fail(f"--split must be one of {', '.join(SPLITS)} (got {args.split!r}). Phase 2 measures on val.")
    if args.split == "test" and not args.unseal_test:
        fail(sealed_test_message(f"--split {args.split}"))

    out_dir = Path(args.out_dir)
    imgsz = args.imgsz
    inference_notes: list[str] = []

    # ---- split predictions: from a cache or by running the model once
    if args.preds:
        cache_path = Path(args.preds)
        if not cache_path.exists():
            fail(f"{cache_path} does not exist. Run evaluate.py first, or pass --weights to run inference here.")
        raw = P.load_raw(cache_path)
        if not raw.paths:
            fail(f"{cache_path} holds no images.", 3)
        if args.data_root:
            moved = _reroot(raw, Path(args.data_root), args.split)
            if moved:
                inference_notes.append(f"{moved} cached image path(s) were re-rooted under --data-root")
        dirs = _split_dirs(raw.paths)
        if "test" in dirs and not args.unseal_test:
            fail(sealed_test_message(f"the cache {cache_path}"))
        if dirs != {args.split}:
            fail(f"{cache_path} holds frames from {sorted(dirs)} but --split is {args.split!r}. Pass the matching --split "
                 "(test also needs --unseal-test).", 3)
        meta = raw.meta
        imgsz = int(meta.get("imgsz", imgsz))
        root = Path(args.data_root) if args.data_root else _infer_root(raw.paths)
        weights_path = args.weights or meta.get("weights")
        weights_sha = meta.get("weights_sha256")
        if args.weights and weights_sha and C.sha256_file(Path(args.weights)) != weights_sha:
            fail(f"--weights {args.weights} is not the model that produced {cache_path} (sha256 differs), so its "
                 "background-frame predictions would not belong to the same detector.", 3)
        cache_record = {"path": _display_path(cache_path), "sha256": C.sha256_file(cache_path)}
    else:
        weights = Path(args.weights)
        if not weights.exists():
            fail(f"--weights {weights} does not exist.")
        root = C.resolve_data_root(args.data_root)
        images = C.list_split_images(root, args.split)
        print(f"running {weights.name} once over {len(images)} {args.split} images (imgsz {imgsz}) ...", flush=True)
        raw = P.predict_raw(weights, images, imgsz=imgsz, batch=args.batch, device=args.device)
        cache_path = out_dir / "cache" / f"sweep_preds_{tag}.npz"
        P.save_raw(cache_path, raw)
        meta = raw.meta
        weights_path, weights_sha = str(weights), meta.get("weights_sha256")
        cache_record = {"path": _display_path(cache_path), "sha256": C.sha256_file(cache_path)}

    names = C.load_class_names(root if root is not None and (root / "data.yaml").exists() else None)
    if args.class_name not in names:
        fail(f"class {args.class_name!r} is not one of {names}.")
    class_id = names.index(args.class_name)

    conf_floor = float(meta.get("conf_floor", MIN_CONF))
    confs = np.unique(np.round(np.append(args.conf_grid, CONF_CEILING), 6))
    if confs.min() < conf_floor - 1e-12:
        fail(f"--conf-grid starts at {confs.min()} but the raw predictions were floored at {conf_floor}; counts below the "
             "floor would be silently identical to the floor. Start the grid at or above it.")

    # ---- per-IoU ImageData for the split, and for the optional background frames
    resolver = C.MetaResolver(find_index(root, args.index))
    by_iou, report = images_by_iou(raw, resolver, args.iou_grid, imgsz, args.size_basis)
    if report.get("unlabelled_images") == len(raw.paths):
        fail("none of the images has a label file, so every detection would count as a false positive. The cached image "
             "paths probably moved: pass --data-root pointing at the dataset that holds images/<split> and labels/<split>.", 3)

    slice_frames: dict[str, dict[float, list[M.ImageData]]] = {s: {} for s in SLICES}
    n_split = {s: 0 for s in SLICES}
    for iou, imgs in by_iou.items():
        groups = P.group_by_slice(imgs)
        for s in SLICES:                  # only the two modality keys; sub-slices are not swept
            slice_frames[s][iou] = list(groups[s])
            n_split[s] = len(groups[s])

    n_empty = {s: 0 for s in SLICES}
    empty_record = None
    if args.empty_dir:
        folder = Path(args.empty_dir)
        empty_images = check_empty_dir(folder, args.unseal_test)
        print(f"scoring {len(empty_images)} background frames from {folder} as {args.empty_modality} ...", flush=True)
        empty_raw = P.predict_raw(Path(args.weights), empty_images, imgsz=imgsz, batch=args.batch, device=args.device)
        P.save_raw(out_dir / "cache" / f"sweep_empty_{tag}.npz", empty_raw)
        modality = "visible" if args.empty_modality == "day" else "lwir"
        for iou in by_iou:
            slice_frames[args.empty_modality][iou] += empty_frame_images(empty_raw, class_id, iou, modality, imgsz, args.size_basis)
        n_empty[args.empty_modality] = len(empty_images)
        empty_record = {"dir": _display_path(folder), "modality": args.empty_modality, "n_frames": len(empty_images)}

    # ---- sweep each modality slice on its own
    assumptions = budget_arithmetic(args.fps, args.alerts_per_camera_day, args.fp_to_alert)
    budget = assumptions["budget_fp_per_frame"]
    slices = {}
    for s in SLICES:
        slices[s] = sweep_slice(s, slice_frames[s], class_id, confs, budget, out_dir / f"sweep_{tag}_{s}_grid.csv")
        slices[s]["n_frames_split"] = n_split[s]
        slices[s]["n_empty_frames"] = n_empty[s]

    assumptions.update({
        "class": args.class_name,
        "note": ("false-alert proxy at detector level: every unmatched person detection at IoU 0.50 counts as a false "
                 "alert (fp_to_alert as given); frames are treated as independent trials, so the upper bound is optimistic"),
    })
    payload = {
        "schema": SCHEMA,
        "tag": tag,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": {"path": weights_path, "sha256": weights_sha},
        "split": args.split,
        "host": C.host_info(),
        "imgsz": imgsz,
        "size_basis": args.size_basis,
        "ultralytics": meta.get("ultralytics"),
        "raw_predictions": {**cache_record, "conf_floor": conf_floor, "meta": meta},
        "n_images": len(raw.paths),
        "grid": {"conf_min": float(confs.min()), "conf_max": float(confs.max()), "n_conf": int(len(confs)),
                 "conf_ceiling": CONF_CEILING, "nms_iou": [float(v) for v in args.iou_grid], "match_iou": 0.5},
        "empty_frames": empty_record,
        "index_used": resolver.has_index,
        "assumptions": assumptions,
        "slices": slices,
        "notes": build_notes(bool(args.empty_dir), args.split, is_synthetic(root)) + inference_notes + (
            ["With --preds the inference host is not recorded in the cache; `host` is where this sweep ran."] if args.preds else []),
    }
    out_json = out_dir / f"sweep_{tag}.json"
    C.write_json(out_json, payload)
    print(render_report(payload))
    print(f"\nwrote {_display_path(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
