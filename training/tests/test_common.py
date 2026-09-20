"""Guards on the reporting rules in _common.py."""

import re
from pathlib import Path

import pytest

import _common as C

REPO = Path(__file__).resolve().parents[2]


def _measurements_rows():
    """Parse the section 3 table of docs/MEASUREMENTS.md: | 77 px | 51 | 100% |"""
    text = (REPO / "docs" / "MEASUREMENTS.md").read_text(encoding="utf-8")
    section = text.split("## 3.", 1)[1].split("## 4.", 1)[0]
    return [(int(h), int(pct)) for h, pct in re.findall(r"^\|\s*(\d+) px\s*\|\s*\d+\s*\|\s*(\d+)%\s*\|", section, re.M)]


def test_size_buckets_match_measurements_md_exactly():
    rows = _measurements_rows()
    assert [h for h, _ in rows] == [77, 54, 38, 27, 19, 14, 9]
    assert list(C.SIZE_BUCKETS_PX) == [h for h, _ in rows]


def test_pretrained_reference_recall_matches_measurements_md():
    rows = dict(_measurements_rows())
    assert {h: round(r * 100) for h, r in C.MEASURED_PRETRAINED_RECALL_VS_NATIVE.items()} == rows


@pytest.mark.parametrize(
    "height,level",
    [(200, 77), (77, 77), (64.6, 77), (64.4, 54), (54, 54), (45.4, 54), (45.2, 38), (38, 38),
     (32.1, 38), (31.9, 27), (27, 27), (22.7, 27), (22.5, 19), (19, 19), (16.4, 19), (16.2, 14),
     (14, 14), (11.3, 14), (11.1, 9), (9, 9), (1, 9)],
)
def test_size_bucket_is_nearest_level_on_a_log_scale(height, level):
    assert C.size_bucket(height) == level


def test_bucket_ranges_are_contiguous():
    assert C.bucket_range_label(77).startswith(">=")
    assert C.bucket_range_label(9).startswith("<")


def test_modality_comes_from_the_filename_suffix():
    assert C.modality_of_stem("kaist_I00019_visible") == "visible"
    assert C.modality_of_stem("llvip_010001_lwir") == "lwir"
    assert C.modality_of_stem("kaist_I00019_visible_t0_128_visible") == "visible"
    assert C.modality_of_stem("kaist_I00019") is None
    assert C.modality_of_stem("lwir_frame") is None


def test_object_height_bases():
    assert C.object_height_px(0.1, 1024, 1280, 640, "stored") == pytest.approx(102.4)
    assert C.object_height_px(0.1, 1024, 1280, 640, "input") == pytest.approx(51.2)  # 0.5x scale
    assert C.object_height_px(0.1, 512, 640, 640, "input") == pytest.approx(51.2)    # 1:1
    with pytest.raises(ValueError):
        C.object_height_px(0.1, 512, 640, 640, "nope")


def test_resolver_lighting_and_slices():
    r = C.MetaResolver()
    day = r.resolve("kaist_set06_V000_I00019_visible")
    night = r.resolve("kaist_set09_V001_I00003_visible")
    ir = r.resolve("kaist_set09_V001_I00003_lwir")
    assert (day.slice, day.lighting) == ("day", "daylight")
    assert (night.slice, night.lighting) == ("day", "night")
    assert (ir.slice, ir.lighting) == ("ir", "n/a")
    assert r.resolve("idd_0001_visible").lighting == "daylight"
    assert r.resolve("llvip_010001_visible").lighting == "night"
    # A KAIST frame whose stem does not carry its set is unresolved, never defaulted to day.
    assert r.resolve("kaist_I00019_visible").lighting == "unresolved"
    assert r.resolve("nonsense") is None


def test_resolver_cluster_keys():
    r = C.MetaResolver()
    assert r.resolve("kaist_set06_V000_I00019_visible").cluster == "kaist/set06/V000"
    assert r.resolve("llvip_010001_visible").cluster == "llvip/01"
    # No group known: the frame is its own cluster.
    assert r.resolve("idd_0001_visible").cluster == "idd_0001_visible"


def test_resolver_uses_the_split_index_when_given(tmp_path):
    idx = tmp_path / "split.jsonl"
    idx.write_text(
        '{"image": "/x/kaist/I00019.jpg", "source": "kaist", "modality": "visible", '
        '"set": "set09", "sequence_key": "kaist/set09/V001"}\n',
        encoding="utf-8",
    )
    r = C.MetaResolver(idx)
    meta = r.resolve("kaist_I00019_visible")
    assert meta.lighting == "night" and meta.cluster == "kaist/set09/V001"


def test_label_path_for():
    p = Path("/data/yolo/images/val/kaist_x_visible.jpg")
    assert C.label_path_for(p) == Path("/data/yolo/labels/val/kaist_x_visible.txt")
    with pytest.raises(ValueError):
        C.label_path_for(Path("/data/x.jpg"))


def test_read_yolo_label_handles_missing_and_empty(tmp_path):
    assert C.read_yolo_label(tmp_path / "nope.txt").shape == (0, 5)
    f = tmp_path / "a.txt"
    f.write_text("", encoding="utf-8")
    assert C.read_yolo_label(f).shape == (0, 5)
    f.write_text("0 0.5 0.5 0.1 0.2\n3 0.1 0.1 0.05 0.05\n", encoding="utf-8")
    assert C.read_yolo_label(f).shape == (2, 5)


def test_write_json_is_atomic_and_serialises_numpy(tmp_path):
    import numpy as np

    out = tmp_path / "d" / "x.json"
    C.write_json(out, {"a": np.float32(1.5), "b": np.arange(3), "p": tmp_path})
    assert C.read_json(out)["b"] == [0, 1, 2]
    assert not list(out.parent.glob("*.tmp"))


def test_fmt_never_prints_zero_for_missing():
    assert C.fmt(None) == "n/a"
    assert C.fmt(float("nan")) == "n/a"
    assert C.fmt(0.0) == "0.000"
