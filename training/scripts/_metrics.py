"""Detection metrics with the slicing rules the Phase 2 brief makes non-negotiable.

Pure numpy. No Ultralytics, no torch. The matching and the AP integration follow the standard
COCO/Ultralytics conventions (greedy one-to-one matching by IoU, 101-point interpolated AP,
IoU 0.50:0.95 in steps of 0.05) so the numbers are comparable with `yolo val`; a test checks that
against Ultralytics' own `ap_per_class` when it is installed. One consequence worth knowing: this
interpolation caps a perfect detector at AP 0.995, exactly as `yolo val` does, so a slice that
reads 0.995 is a perfect slice, not a near miss.

What is different here, deliberately:

  * A metric is only ever computed over frames of ONE modality. `evaluate_slice` raises
    `BlendError` if it is handed both visible and LWIR frames. There is no code path that yields
    a number over both, so no report can contain one by accident.
  * Object-size slicing uses COCO's area-range protocol: ground truth outside the bucket is
    ignored (not a miss), a detection that matches it is ignored (not a false alarm), and an
    unmatched detection whose own height is outside the bucket is ignored. Without this, a
    large-object false positive would count against the 9 px bucket.
  * Every metric that has no ground truth is None, never 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from _common import ImageMeta, size_bucket

IOU_THRS = np.linspace(0.5, 0.95, 10)
CONF_GRID = np.linspace(0.0, 1.0, 1001)


class BlendError(ValueError):
    """Raised when a metric would be computed over frames of more than one modality."""


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


@dataclass
class ImageData:
    """One evaluated image. Boxes are xyxy in original-image pixels; heights in the size basis."""

    meta: ImageMeta
    gt_cls: np.ndarray   # (g,) int
    gt_box: np.ndarray   # (g, 4)
    gt_h: np.ndarray     # (g,) px
    pr_cls: np.ndarray   # (p,) int
    pr_box: np.ndarray   # (p, 4)
    pr_conf: np.ndarray  # (p,)
    pr_h: np.ndarray     # (p,) px


def assert_single_modality(images: Sequence[ImageData]) -> str | None:
    """Return the one modality present, or None if empty. Raise BlendError if there are two."""
    modalities = {im.meta.modality for im in images}
    if len(modalities) > 1:
        raise BlendError(
            f"refusing to compute a metric over {sorted(modalities)} together: day and IR are "
            "reported separately, never blended"
        )
    return next(iter(modalities), None)


# --------------------------------------------------------------------------------------------
# Geometry and matching
# --------------------------------------------------------------------------------------------


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 4) x (m, 4) xyxy -> (n, m) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-16)


def _greedy(iou: np.ndarray, thr: float, pred_free: np.ndarray, gt_cols: np.ndarray) -> np.ndarray:
    """One-to-one greedy assignment by descending IoU.

    Returns an array `assigned` of length p: the matched gt column for each prediction, or -1.
    Only predictions with pred_free[i] and gt columns in `gt_cols` may match.
    """
    p = iou.shape[0]
    assigned = np.full(p, -1, dtype=np.int64)
    if p == 0 or len(gt_cols) == 0:
        return assigned
    sub = iou[:, gt_cols]
    pi, gj = np.nonzero((sub >= thr) & pred_free[:, None])
    if len(pi) == 0:
        return assigned
    order = np.argsort(-sub[pi, gj], kind="stable")
    pi, gj = pi[order], gj[order]
    _, first_gt = np.unique(gj, return_index=True)   # each gt used once, by its best pred
    pi, gj = pi[first_gt], gj[first_gt]
    order = np.argsort(-sub[pi, gj], kind="stable")
    pi, gj = pi[order], gj[order]
    _, first_pred = np.unique(pi, return_index=True)  # each pred used once, by its best gt
    pi, gj = pi[first_pred], gj[first_pred]
    assigned[pi] = gt_cols[gj]
    return assigned


def match_image_class(
    gt_box: np.ndarray,
    gt_ignore: np.ndarray,
    pr_box: np.ndarray,
    pr_outside: np.ndarray,
    thrs: np.ndarray = IOU_THRS,
) -> tuple[np.ndarray, np.ndarray]:
    """Match one class's predictions in one image at every IoU threshold.

    `pr_box` must already be sorted by confidence, descending. Returns (tp, ignore), both
    (len(thrs), p) bool. `ignore` marks predictions that must not count as TP or FP: those
    matched to an ignored (out-of-slice) ground truth, and unmatched ones outside the slice.
    """
    p = len(pr_box)
    tp = np.zeros((len(thrs), p), dtype=bool)
    ignore = np.zeros((len(thrs), p), dtype=bool)
    if p == 0:
        return tp, ignore
    iou = box_iou(pr_box, gt_box)
    real_cols = np.nonzero(~gt_ignore)[0]
    ign_cols = np.nonzero(gt_ignore)[0]
    everyone = np.ones(p, dtype=bool)
    for t, thr in enumerate(thrs):
        assigned = _greedy(iou, thr, everyone, real_cols)
        matched = assigned >= 0
        tp[t] = matched
        if len(ign_cols):
            leftover = ~matched
            assigned_ign = _greedy(iou, thr, leftover, ign_cols)
            ignore[t] |= assigned_ign >= 0
        ignore[t] |= (~matched) & pr_outside
    return tp, ignore


# --------------------------------------------------------------------------------------------
# AP, precision, recall
# --------------------------------------------------------------------------------------------


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP over a recall/precision curve (COCO convention)."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


@dataclass
class ClassMatches:
    """Every prediction of one class across a slice, with its match outcome at each IoU."""

    conf: np.ndarray            # (N,)
    tp: np.ndarray              # (T, N) bool
    ignore: np.ndarray          # (T, N) bool
    image_id: np.ndarray        # (N,) int, index into the slice's image list
    n_gt_per_image: np.ndarray  # (I,) non-ignored ground truth per image

    @property
    def n_gt(self) -> int:
        return int(self.n_gt_per_image.sum())


def collect_matches(
    images: Sequence[ImageData],
    class_id: int,
    bucket: int | None = None,
    thrs: np.ndarray = IOU_THRS,
) -> ClassMatches:
    """Match `class_id` over `images`. `bucket` restricts scoring to one SIZE_BUCKETS_PX level."""
    conf_parts, tp_parts, ign_parts, id_parts = [], [], [], []
    n_gt = np.zeros(len(images), dtype=np.int64)
    for i, im in enumerate(images):
        g = im.gt_cls == class_id
        gt_box = im.gt_box[g]
        gt_ignore = np.zeros(len(gt_box), dtype=bool)
        if bucket is not None and len(gt_box):
            gt_ignore = np.array([size_bucket(h) != bucket for h in im.gt_h[g]], dtype=bool)
        n_gt[i] = int((~gt_ignore).sum())

        pm = im.pr_cls == class_id
        if not pm.any():
            continue
        conf = im.pr_conf[pm]
        order = np.argsort(-conf, kind="stable")
        conf = conf[order]
        pr_box = im.pr_box[pm][order]
        outside = np.zeros(len(pr_box), dtype=bool)
        if bucket is not None:
            outside = np.array([size_bucket(h) != bucket for h in im.pr_h[pm][order]], dtype=bool)
        tp, ign = match_image_class(gt_box, gt_ignore, pr_box, outside, thrs)
        conf_parts.append(conf)
        tp_parts.append(tp)
        ign_parts.append(ign)
        id_parts.append(np.full(len(conf), i, dtype=np.int64))

    t = len(thrs)
    if conf_parts:
        return ClassMatches(
            conf=np.concatenate(conf_parts),
            tp=np.concatenate(tp_parts, axis=1),
            ignore=np.concatenate(ign_parts, axis=1),
            image_id=np.concatenate(id_parts),
            n_gt_per_image=n_gt,
        )
    return ClassMatches(
        conf=np.zeros(0),
        tp=np.zeros((t, 0), dtype=bool),
        ignore=np.zeros((t, 0), dtype=bool),
        image_id=np.zeros(0, dtype=np.int64),
        n_gt_per_image=n_gt,
    )


def _curve(m: ClassMatches, t: int, weights: np.ndarray | None):
    """Sorted cumulative TP/FP for IoU index t. weights: per-image multiplicity (bootstrap)."""
    keep = ~m.ignore[t]
    conf = m.conf[keep]
    tp = m.tp[t][keep]
    w = np.ones(len(conf)) if weights is None else weights[m.image_id[keep]].astype(np.float64)
    order = np.argsort(-conf, kind="stable")
    conf, tp, w = conf[order], tp[order], w[order]
    tpc = np.cumsum(tp * w)
    fpc = np.cumsum((~tp) * w)
    n_gt = float(m.n_gt_per_image.sum() if weights is None else (m.n_gt_per_image * weights).sum())
    return conf, tpc, fpc, n_gt


def average_precision(m: ClassMatches, t: int = 0, weights: np.ndarray | None = None) -> float | None:
    """AP at IoU index t (0 -> 0.50). None when the slice holds no ground truth for the class."""
    conf, tpc, fpc, n_gt = _curve(m, t, weights)
    if n_gt <= 0:
        return None
    if len(conf) == 0:
        return 0.0
    recall = tpc / (n_gt + 1e-16)
    precision = tpc / (tpc + fpc + 1e-16)
    return compute_ap(recall, precision)


def ap50_95(m: ClassMatches) -> float | None:
    values = [average_precision(m, t) for t in range(m.tp.shape[0])]
    if values[0] is None:
        return None
    return float(np.mean(values))


def pr_at_confs(m: ClassMatches, confs: np.ndarray, t: int = 0):
    """Precision, recall, TP, FP at each confidence threshold in `confs` (IoU index t)."""
    conf, tpc, fpc, n_gt = _curve(m, t, None)
    k = np.searchsorted(-conf, -confs, side="right")  # detections with conf >= threshold
    tp = np.where(k > 0, tpc[np.maximum(k - 1, 0)] if len(tpc) else 0.0, 0.0)
    fp = np.where(k > 0, fpc[np.maximum(k - 1, 0)] if len(fpc) else 0.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = tp / n_gt if n_gt > 0 else np.full(len(confs), np.nan)
    return precision, recall, tp, fp


# --------------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------------


def bootstrap_ap50(
    m: ClassMatches,
    cluster_of_image: np.ndarray,
    n_boot: int = 200,
    seed: int = 42,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile interval for AP50, resampling whole clusters (sequences) with replacement.

    Frames of one video are strongly correlated, so resampling single frames would understate the
    uncertainty and flatter the result. Where no group is known a frame is its own cluster, which
    is optimistic for that data; the report says so.
    """
    if m.n_gt == 0 or n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    n_clusters = int(cluster_of_image.max()) + 1
    values = []
    for _ in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        counts = np.bincount(draw, minlength=n_clusters)
        ap = average_precision(m, 0, weights=counts[cluster_of_image])
        if ap is not None:
            values.append(ap)
    if not values:
        return None
    lo, hi = np.percentile(values, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def cluster_ids(images: Sequence[ImageData]) -> np.ndarray:
    lookup: dict[str, int] = {}
    return np.array([lookup.setdefault(im.meta.cluster, len(lookup)) for im in images], dtype=np.int64)


# --------------------------------------------------------------------------------------------
# Slice evaluation
# --------------------------------------------------------------------------------------------


def _class_row(ap50, ap, p, r, n_gt) -> dict:
    return {"n_gt": int(n_gt), "ap50": ap50, "ap50_95": ap, "precision": p, "recall": r}


def evaluate_slice(
    images: Sequence[ImageData],
    class_names: Sequence[str],
    report_conf: float = 0.35,
    n_boot: int = 200,
    seed: int = 42,
    bootstrap_classes: Sequence[str] = ("person",),
) -> dict:
    """Full per-class evaluation of one modality slice. Never mixes modalities."""
    modality = assert_single_modality(images)
    nc = len(class_names)
    matches = {c: collect_matches(images, c) for c in range(nc)}
    present = [c for c in range(nc) if matches[c].n_gt > 0]

    # Operating confidence: the one that maximises F1 averaged over the classes present in THIS
    # slice (Ultralytics' convention). Precision and recall are reported at it, per class.
    f1_mean = np.zeros(len(CONF_GRID))
    curves = {}
    for c in present:
        p, r, _, _ = pr_at_confs(matches[c], CONF_GRID)
        p0 = np.nan_to_num(p, nan=0.0)
        r0 = np.nan_to_num(r, nan=0.0)
        f1 = np.where(p0 + r0 > 0, 2 * p0 * r0 / (p0 + r0 + 1e-16), 0.0)
        curves[c] = (p, r)
        f1_mean += f1 / len(present)
    op_idx = int(np.argmax(f1_mean)) if present else 0
    op_conf = float(CONF_GRID[op_idx])
    rep_idx = int(np.argmin(np.abs(CONF_GRID - report_conf)))

    classes, fixed = {}, {}
    for c, name in enumerate(class_names):
        m = matches[c]
        if c not in present:
            classes[name] = _class_row(None, None, None, None, 0)
            fixed[name] = {"precision": None, "recall": None}
            continue
        p, r = curves[c]
        classes[name] = _class_row(
            average_precision(m, 0),
            ap50_95(m),
            _none_if_nan(p[op_idx]),
            _none_if_nan(r[op_idx]),
            m.n_gt,
        )
        fixed[name] = {"precision": _none_if_nan(p[rep_idx]), "recall": _none_if_nan(r[rep_idx])}

    def mean_of(key):
        vals = [classes[class_names[c]][key] for c in present if classes[class_names[c]][key] is not None]
        return float(np.mean(vals)) if vals else None

    boot = {}
    clusters = cluster_ids(images) if images else np.zeros(0, dtype=np.int64)
    for name in bootstrap_classes:
        if name in class_names and class_names.index(name) in present:
            ci = bootstrap_ap50(matches[class_names.index(name)], clusters, n_boot, seed)
            boot[name] = {
                "ap50_ci95": list(ci) if ci else None,
                "n_boot": n_boot,
                "n_clusters": int(clusters.max()) + 1 if len(clusters) else 0,
                "n_images": len(images),
                "resampling": "cluster (sequence) bootstrap; frames without a known group are their own cluster",
            }

    return {
        "modality": modality,
        "n_images": len(images),
        "operating_conf": op_conf,
        "operating_conf_rule": "argmax of F1 averaged over the classes present in this slice",
        "classes": classes,
        "mean_over_present_classes": {
            "ap50": mean_of("ap50"),
            "ap50_95": mean_of("ap50_95"),
            "precision": mean_of("precision"),
            "recall": mean_of("recall"),
        },
        "at_report_conf": {"conf": float(CONF_GRID[rep_idx]), "classes": fixed},
        "bootstrap": boot,
    }


def evaluate_size_buckets(
    images: Sequence[ImageData],
    class_names: Sequence[str],
    class_name: str,
    buckets: Sequence[int],
    report_conf: float = 0.35,
) -> list[dict]:
    """Per-height-bucket AP50, precision and recall for one class within one modality slice."""
    assert_single_modality(images)
    c = list(class_names).index(class_name)
    rep_idx = int(np.argmin(np.abs(CONF_GRID - report_conf)))
    rows = []
    for level in buckets:
        m = collect_matches(images, c, bucket=level)
        if m.n_gt == 0:
            rows.append({"bucket_px": level, "n_gt": 0, "ap50": None, "ap50_95": None,
                         "precision": None, "recall": None, "conf": float(CONF_GRID[rep_idx])})
            continue
        p, r, _, _ = pr_at_confs(m, CONF_GRID)
        rows.append({
            "bucket_px": level,
            "n_gt": m.n_gt,
            "ap50": average_precision(m, 0),
            "ap50_95": ap50_95(m),
            "precision": _none_if_nan(p[rep_idx]),
            "recall": _none_if_nan(r[rep_idx]),
            "conf": float(CONF_GRID[rep_idx]),
        })
    return rows


def _none_if_nan(x) -> float | None:
    x = float(x)
    return None if np.isnan(x) else x


# --------------------------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------------------------

MET = "MET"
NOT_MET = "NOT MET"
PARTIAL = "PARTIAL"
NOT_ASSESSED = "NOT ASSESSED"


def map_verdict(
    point: float | None,
    ci_low: float | None,
    target: float,
    narrower_point: float | None = None,
) -> tuple[str, str]:
    """MET / NOT MET / PARTIAL for an mAP@50 target, and the one-line reason.

    Deliberately conservative ("do not round in my favour"):
      * MET requires the point estimate to reach the target AND the lower 95% bound to reach it.
      * A point estimate at or above target with a lower bound below it is PARTIAL: reached, but
        not established.
      * A point estimate below target is PARTIAL only if a strictly narrower reading of the same
        target (for "day": daylight-only frames) does reach it; otherwise NOT MET.
      * No measurement -> NOT ASSESSED. It is never MET by default.
    """
    if point is None:
        return NOT_ASSESSED, "no ground truth in this slice"
    if point >= target:
        if ci_low is not None and ci_low >= target:
            return MET, f"{point:.3f} >= {target:.2f}, lower 95% bound {ci_low:.3f} also >= target"
        if ci_low is None:
            return PARTIAL, f"{point:.3f} >= {target:.2f} but no confidence interval was computed"
        return PARTIAL, f"{point:.3f} >= {target:.2f} but lower 95% bound {ci_low:.3f} < target"
    if narrower_point is not None and narrower_point >= target:
        return PARTIAL, (
            f"{point:.3f} < {target:.2f} on the full slice; only the narrower daylight-only "
            f"reading reaches it ({narrower_point:.3f})"
        )
    return NOT_MET, f"{point:.3f} < {target:.2f}"
