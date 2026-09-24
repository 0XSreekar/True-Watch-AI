"""Engine tests: known values, the no-blending guard, size-slice ignore semantics, verdicts."""

import numpy as np
import pytest

import _metrics as M
from _common import MetaResolver, size_bucket

R = MetaResolver()
NAMES = ["person", "two_wheeler", "car", "truck", "cart"]

# The 101-point interpolation used by COCO and Ultralytics tops out at 0.995 even for a perfect
# detector, because the last recall sample sits on the sentinel. `yolo val` shows the same 0.995.
PERFECT = 0.995


def box(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def img(stem, gts, preds):
    """gts: [(cls, cx, cy, w, h)], preds: [(cls, conf, cx, cy, w, h)] in pixels; height basis = h."""
    m = R.resolve(stem)
    gt_cls = np.array([g[0] for g in gts], dtype=np.int64)
    gt_box = np.array([box(*g[1:]) for g in gts], dtype=np.float64).reshape(-1, 4)
    gt_h = np.array([g[4] for g in gts], dtype=np.float64)
    pr_cls = np.array([p[0] for p in preds], dtype=np.int64)
    pr_conf = np.array([p[1] for p in preds], dtype=np.float64)
    pr_box = np.array([box(*p[2:]) for p in preds], dtype=np.float64).reshape(-1, 4)
    pr_h = np.array([p[5] for p in preds], dtype=np.float64)
    return M.ImageData(m, gt_cls, gt_box, gt_h, pr_cls, pr_box, pr_conf, pr_h)


def test_perfect_detections_give_ap_one():
    ims = [img("kaist_set06_V000_I1_visible", [(0, 50, 50, 20, 40)], [(0, 0.9, 50, 50, 20, 40)])]
    row = M.evaluate_slice(ims, NAMES, n_boot=0)
    assert row["classes"]["person"]["ap50"] == pytest.approx(PERFECT, abs=1e-6)
    assert row["classes"]["person"]["ap50_95"] == pytest.approx(PERFECT, abs=1e-6)
    assert row["classes"]["person"]["recall"] == pytest.approx(1.0)


def test_missing_ground_truth_is_none_not_zero():
    ims = [img("kaist_set06_V000_I1_visible", [(0, 50, 50, 20, 40)], [(0, 0.9, 50, 50, 20, 40)])]
    row = M.evaluate_slice(ims, NAMES, n_boot=0)
    for name in ("two_wheeler", "car", "truck", "cart"):
        assert row["classes"][name]["ap50"] is None
        assert row["classes"][name]["n_gt"] == 0
    # The mean is over the classes that have ground truth, so an empty class cannot drag it down.
    assert row["mean_over_present_classes"]["ap50"] == pytest.approx(PERFECT, abs=1e-6)


def test_no_detections_with_ground_truth_is_zero():
    ims = [img("kaist_set06_V000_I1_visible", [(0, 50, 50, 20, 40)], [])]
    assert M.evaluate_slice(ims, NAMES, n_boot=0)["classes"]["person"]["ap50"] == 0.0


def test_ap_matches_hand_computation():
    # Two GT. Detections by confidence: TP(0.9), FP(0.8), TP(0.7).
    ims = [img("kaist_set06_V000_I1_visible",
               [(0, 50, 50, 20, 40), (0, 200, 50, 20, 40)],
               [(0, 0.9, 50, 50, 20, 40), (0, 0.8, 120, 50, 20, 40), (0, 0.7, 200, 50, 20, 40)])]
    m = M.collect_matches(ims, 0)
    # recall = [.5, .5, 1], precision = [1, .5, 2/3]; envelope makes precision 1 up to r=.5, 2/3 after.
    x = np.linspace(0, 1, 101)
    env = np.where(x <= 0.5, 1.0, 2 / 3)
    expected = np.sum((env[1:] + env[:-1]) * np.diff(x)) / 2
    assert M.average_precision(m, 0) == pytest.approx(expected, abs=0.01)


def test_day_and_ir_are_never_blended():
    day = img("kaist_set06_V000_I1_visible", [(0, 50, 50, 20, 40)], [(0, 0.9, 50, 50, 20, 40)])
    ir = img("kaist_set06_V000_I1_lwir", [(0, 50, 50, 20, 40)], [(0, 0.9, 50, 50, 20, 40)])
    with pytest.raises(M.BlendError):
        M.evaluate_slice([day, ir], NAMES)
    with pytest.raises(M.BlendError):
        M.evaluate_size_buckets([day, ir], NAMES, "person", [77])
    # ...each alone is fine.
    assert M.evaluate_slice([day], NAMES, n_boot=0)["modality"] == "visible"
    assert M.evaluate_slice([ir], NAMES, n_boot=0)["modality"] == "lwir"


def test_size_bucket_ignores_out_of_slice_objects_and_their_detections():
    # One big person (80 px) found, one tiny person (9 px) missed, plus a big false positive.
    ims = [img("kaist_set06_V000_I1_visible",
               [(0, 50, 100, 30, 80), (0, 300, 100, 5, 9)],
               [(0, 0.9, 50, 100, 30, 80), (0, 0.8, 200, 100, 30, 80)])]
    big = M.evaluate_size_buckets(ims, NAMES, "person", [77, 9])
    b77, b9 = big
    assert b77["n_gt"] == 1 and b77["ap50"] == pytest.approx(PERFECT, abs=1e-6)
    # 9 px bucket: the tiny GT is a miss (recall 0), and the big false positive is NOT counted
    # against it (precision stays undefined rather than 0 because nothing in-bucket was predicted).
    assert b9["n_gt"] == 1 and b9["ap50"] == 0.0 and b9["recall"] == 0.0


def test_prediction_matched_to_out_of_slice_gt_is_ignored():
    # A detection on the big GT must not be a false positive in the 9 px bucket.
    ims = [img("kaist_set06_V000_I1_visible",
               [(0, 50, 100, 30, 80), (0, 300, 100, 5, 9)],
               [(0, 0.95, 50, 100, 30, 80), (0, 0.6, 300, 100, 5, 9)])]
    m = M.collect_matches(ims, 0, bucket=9)
    assert m.n_gt == 1
    assert M.average_precision(m, 0) == pytest.approx(PERFECT, abs=1e-6)
    assert m.ignore[0].sum() == 1  # the big-object detection


def test_operating_point_and_precision_recall():
    good = [img(f"kaist_set06_V000_I{i}_visible", [(0, 50, 50, 20, 40)], [(0, 0.9, 50, 50, 20, 40)]) for i in range(5)]
    noise = [img(f"kaist_set06_V001_I{i}_visible", [], [(0, 0.1, 50, 50, 20, 40)]) for i in range(5)]
    row = M.evaluate_slice(good + noise, NAMES, n_boot=0)
    assert row["operating_conf"] > 0.1                  # F1 is best above the noisy detections
    assert row["classes"]["person"]["precision"] == pytest.approx(1.0)
    assert row["classes"]["person"]["recall"] == pytest.approx(1.0)


def test_bootstrap_interval_brackets_the_estimate_and_is_seeded():
    ims = []
    for i in range(40):
        hit = i % 4 != 0  # 75% recall
        ims.append(img(f"kaist_set06_V{i % 8:03d}_I{i}_visible", [(0, 50, 50, 20, 40)],
                       [(0, 0.9, 50, 50, 20, 40)] if hit else []))
    a = M.evaluate_slice(ims, NAMES, n_boot=100, seed=1)
    b = M.evaluate_slice(ims, NAMES, n_boot=100, seed=1)
    lo, hi = a["bootstrap"]["person"]["ap50_ci95"]
    ap = a["classes"]["person"]["ap50"]
    assert lo <= ap <= hi and lo < hi
    assert a["bootstrap"]["person"]["ap50_ci95"] == b["bootstrap"]["person"]["ap50_ci95"]
    assert a["bootstrap"]["person"]["n_clusters"] == 8


def test_cross_check_ap_against_ultralytics_ap_per_class():
    ul = pytest.importorskip("ultralytics.utils.metrics")
    rng = np.random.default_rng(0)
    ims = []
    for i in range(60):
        n = rng.integers(0, 5)
        gts = [(0, rng.uniform(40, 600), rng.uniform(40, 400), rng.uniform(15, 60), rng.uniform(20, 120)) for _ in range(n)]
        preds = [(0, rng.uniform(0.02, 0.99), g[1] + rng.normal(0, 4), g[2] + rng.normal(0, 4), g[3], g[4]) for g in gts if rng.random() < 0.8]
        preds += [(0, rng.uniform(0.01, 0.6), rng.uniform(40, 600), rng.uniform(40, 400), 30, 60) for _ in range(rng.integers(0, 4))]
        ims.append(img(f"kaist_set06_V{i % 6:03d}_I{i}_visible", gts, preds))
    m = M.collect_matches(ims, 0)
    ap_ul = ul.ap_per_class(m.tp.T.copy(), m.conf, np.zeros(len(m.conf)), np.zeros(m.n_gt))[5]
    assert ap_ul[0, 0] == pytest.approx(M.average_precision(m, 0), abs=1e-6)
    assert ap_ul[0].mean() == pytest.approx(M.ap50_95(m), abs=1e-6)


@pytest.mark.parametrize(
    "point,low,narrow,expected",
    [
        (0.90, 0.87, None, M.MET),
        (0.86, 0.83, None, M.PARTIAL),       # reached, not established
        (0.85, 0.85, None, M.MET),           # boundary is inclusive
        (0.849, 0.80, None, M.NOT_MET),
        (0.80, 0.77, (0.87, 0.86, 400), M.PARTIAL),   # the narrower reading establishes it on its own
        (0.80, 0.77, (0.87, 0.80, 400), M.NOT_MET),   # narrower point reaches it, its bound does not
        (0.80, 0.77, (0.99, 0.95, 12), M.NOT_MET),    # a tiny daylight subset is an anecdote
        (0.80, 0.77, (0.87, None, 400), M.NOT_MET),   # no interval on the narrower reading
        (0.80, 0.77, (0.83, 0.80, 400), M.NOT_MET),
        (0.90, None, None, M.PARTIAL),       # no interval: cannot claim MET
        (None, None, None, M.NOT_ASSESSED),
    ],
)
def test_map_verdict(point, low, narrow, expected):
    n_point, n_low, n_gt = narrow if narrow else (None, None, None)
    verdict, reason = M.map_verdict(point, low, 0.85, n_point, n_low, n_gt)
    assert verdict == expected and reason


def test_verdict_text_never_rounds_a_miss_onto_the_target():
    verdict, reason = M.map_verdict(0.84962, 0.80, 0.85)
    assert verdict == M.NOT_MET
    assert "0.8496 < 0.85" in reason and "0.850" not in reason
    assert M.fmt_floor(0.84999999) == "0.8499" and M.fmt_floor(0.29) == "0.2900" and M.fmt_floor(None) == "n/a"
    _, reason = M.map_verdict(0.87, 0.84996, 0.85)
    assert "lower 95% bound 0.8499 < target" in reason


def test_crowd_matching_follows_ultralytics_prediction_first_order():
    """GT X and Y overlap. Pred A: IoU(X)=0.905, IoU(Y)=0.739; pred B: IoU(Y)=0.667 -> two TPs, not one.

    Taking each ground truth's best prediction first would hand both X and Y to A, keep only A->X,
    and leave B unmatched. Ultralytics keeps each prediction's best ground truth first.
    """
    iou = np.array([[0.905, 0.739], [0.0, 0.667]])
    assigned = M._greedy(iou, 0.5, np.ones(2, dtype=bool), np.arange(2))
    assert assigned.tolist() == [0, 1]
    # at IoU 0.70 B's only candidate is gone, and A still takes its best (X)
    assert M._greedy(iou, 0.70, np.ones(2, dtype=bool), np.arange(2)).tolist() == [0, -1]


def test_crowd_matching_with_real_boxes_scores_two_true_positives():
    # Unit-height boxes on a line, so IoU is interval overlap; they reproduce the IoUs of the test above.
    X = np.array([0.0, 0.0, 10.0, 1.0])
    Y = np.array([1.0, 0.0, 11.0, 1.0])
    A = np.array([-0.5, 0.0, 9.5, 1.0])       # IoU(A,X)=0.905, IoU(A,Y)=0.739
    B = np.array([4.3, 0.0, 11.05, 1.0])      # IoU(B,Y)=0.667 (and IoU(B,X)=0.516)
    iou = M.box_iou(np.stack([A, B]), np.stack([X, Y]))
    assert iou[0, 0] == pytest.approx(0.905, abs=1e-3) and iou[0, 1] == pytest.approx(0.739, abs=1e-3)
    assert iou[1, 1] == pytest.approx(0.667, abs=1e-3)
    tp, ignore = M.match_image_class(np.stack([X, Y]), np.zeros(2, bool), np.stack([A, B]), np.zeros(2, bool),
                                     np.array([0.5]))
    assert tp[0].tolist() == [True, True] and not ignore.any()


def test_greedy_matches_ultralytics_match_predictions_on_random_crowds():
    torch = pytest.importorskip("torch")
    validator = pytest.importorskip("ultralytics.engine.validator")
    rng = np.random.default_rng(3)
    stub = type("Stub", (), {"iouv": torch.linspace(0.5, 0.95, 10)})()
    for _ in range(200):
        g = rng.integers(1, 8)
        p = rng.integers(1, 10)
        gt = np.sort(rng.uniform(0, 60, (g, 2, 2)), axis=1).reshape(g, 4)[:, [0, 2, 1, 3]]
        pr = gt[rng.integers(0, g, p)] + rng.normal(0, 3, (p, 4))
        pr[:, 2:] = np.maximum(pr[:, 2:], pr[:, :2] + 1)
        iou = M.box_iou(pr, gt)
        ours = np.stack([M._greedy(iou, t, np.ones(p, bool), np.arange(g)) >= 0 for t in M.IOU_THRS], axis=1)
        theirs = validator.BaseValidator.match_predictions(
            stub, torch.zeros(p), torch.zeros(g), torch.from_numpy(iou.T.copy())).numpy()
        assert (ours == theirs).all()


def test_size_bucket_helper_agrees_with_engine():
    assert size_bucket(80) == 77 and size_bucket(9) == 9
