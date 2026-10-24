"""Tiled-inference windows and merge rules (pipeline/tiling.py), shared by the edge and the evaluator."""

import numpy as np

from pipeline.tiling import TilingConfig, merge_passes, tile_windows, truncated_by_window


def _p(window, boxes, scores, cls):
    return (window, np.array(boxes, np.float32).reshape(-1, 4), np.array(scores, np.float32), np.array(cls, np.int64))


def test_grid_windows_cover_the_band_and_overlap():
    wins = tile_windows(1920, 1080, TilingConfig(enabled=True, band=(0.0, 1.0), grid=(3, 2), overlap=0.2))
    assert len(wins) == 6
    assert min(w[0] for w in wins) == 0 and max(w[2] for w in wins) == 1920
    assert min(w[1] for w in wins) == 0 and max(w[3] for w in wins) == 1080


def test_a_window_that_is_the_whole_frame_is_dropped():
    assert tile_windows(640, 640,TilingConfig(enabled=True, band=(0.0, 1.0), tile=640)) == []


def test_truncated_boxes_are_found_only_at_interior_edges():
    win = (640, 0, 1280, 640)
    boxes = np.array([[641, 100, 700, 200], [800, 100, 900, 200], [1100, 0, 1200, 100]], np.float32)
    assert truncated_by_window(boxes, win, 1920, 1080, 4.0).tolist() == [True, False, False]   # top edge is the frame border
    assert truncated_by_window(np.array([[10, 10, 50, 50]], np.float32), (0, 0, 640, 640), 640, 640, 4.0).tolist() == [False]


def test_truncated_window_box_is_dropped_when_an_untruncated_one_covers_it():
    cfg = TilingConfig(enabled=True)
    full = _p(None, [[600, 100, 700, 300]], [0.9], [0])
    win = (640, 0, 1280, 640)
    tile = _p(win, [[641, 100, 700, 300], [900, 100, 960, 220]], [0.8, 0.7], [0, 0])
    b, s, c, t = merge_passes([full, tile], 1920, 1080, cfg, full_conf=0.1, iou=0.9)
    assert len(b) == 2 and not t[0] and t[1]       # the cut-off half is gone, the new person stays


def test_truncated_box_with_no_partner_is_kept_and_trunc_can_be_switched_off():
    win = (640, 0, 1280, 640)
    tile = _p(win, [[641, 100, 700, 300]], [0.8], [0])
    assert len(merge_passes([tile], 1920, 1080, TilingConfig(), full_conf=0.1, iou=0.5)[0]) == 1
    full = _p(None, [[600, 100, 700, 300]], [0.5], [0])
    cfg = TilingConfig(trunc_ioa=0.0, merge_iou=0.9)
    assert len(merge_passes([full, tile], 1920, 1080, cfg, full_conf=0.1, iou=0.5)[0]) == 2


def test_per_scale_threshold_and_class_aware_nms():
    cfg = TilingConfig(tile_conf=0.4)
    full = _p(None, [[0, 0, 50, 50], [0, 0, 50, 50]], [0.2, 0.3], [0, 2])
    tile = _p((0, 0, 640, 640), [[300, 300, 340, 360]], [0.3], [0])
    b, s, c, t = merge_passes([full, tile], 1920, 1080, cfg, full_conf=0.1, iou=0.5)
    assert sorted(c.tolist()) == [0, 2] and not t.any()            # tile box is below its own floor; classes do not suppress


def test_merged_boxes_are_clipped_to_the_frame():
    # a box past the bottom-right corner (a person leaving the view) is clipped, so its ground point stays
    # on the image and a fence drawn to the image border can still be crossed
    full = _p(None, [[1900, 1000, 1935, 1083.5], [-6, -4, 40, 90]], [0.9, 0.8], [0, 0])
    b, _, _, _ = merge_passes([full], 1920, 1080, TilingConfig(), full_conf=0.1, iou=0.5)
    assert b.min() >= 0 and b[:, [0, 2]].max() <= 1920 and b[:, [1, 3]].max() <= 1080
    assert sorted(b[:, 3].tolist()) == [90.0, 1080.0]


def test_presets_and_config_switch():
    import pytest
    from config import Config, ConfigError
    from pipeline.tiling import preset
    assert not preset("off").enabled and preset("full").band == (0.0, 1.0) and preset("strip").band == (0.0, 0.55)
    with pytest.raises(ValueError):
        preset("everything")
    assert Config().detector_tiling == "off"
    with pytest.raises(ConfigError):
        Config(detector_tiling="all").validate()
