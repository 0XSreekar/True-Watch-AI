"""The locator release evaluation counts matches and intervals correctly."""

from __future__ import annotations

from anpr.scripts import eval_plate_locator as ev


def test_wilson_interval_brackets_the_rate():
    lo, hi = ev.wilson(29, 44)
    assert lo < 29 / 44 < hi and 0.50 < lo < 0.52 and 0.77 < hi < 0.79
    assert ev.wilson(0, 0) == [0.0, 0.0]


def test_iou_and_scoring_at_a_threshold():
    gt = [[0, 0, 10, 10]]
    items = [("a.jpg", gt), ("b.jpg", [])]
    dets = [[([0, 0, 10, 9], 0.9), ([50, 50, 60, 60], 0.3)], [([5, 5, 9, 9], 0.2)]]
    assert ev.iou([0, 0, 10, 10], [0, 0, 10, 9]) == 0.9
    hi = ev.score(items, dets, 0.5)
    assert (hi["tp"], hi["fp"], hi["recall"], hi["precision"]) == (1, 0, 1.0, 1.0)
    lo = ev.score(items, dets, 0.1)
    assert (lo["tp"], lo["fp"]) == (1, 2)


def test_ap50_is_one_for_a_perfect_ranking_and_lower_with_a_false_box_first():
    items = [("a.jpg", [[0, 0, 10, 10]])]
    assert ev.ap50(items, [[([0, 0, 10, 10], 0.9)]]) == 1.0
    assert ev.ap50(items, [[([50, 50, 60, 60], 0.95), ([0, 0, 10, 10], 0.5)]]) < 1.0
