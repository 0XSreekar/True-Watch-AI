"""evaluate.py and eval_hardset.py: refusals, slicing, schema, cache reuse, mapping and the gate.

No network, no Ultralytics and no GPU. The detector is replaced by a stand-in that emits the ground
truth as detections (or a degraded version of it), so what is tested is everything around the
network: the sealed-test guard, the day/IR separation, the JSON and CSV schema, the size buckets, the
cache, the hard-set mapping rules and the regression gate. The engine itself is covered by
test_metrics.py, and the agreement with `yolo val` was measured separately against a real model.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import _common as C
import _predict as P
import _synth
import eval_hardset as H
import evaluate as E

cv2 = pytest.importorskip("cv2")

NAMES = _synth.NAMES
PERFECT = 0.995  # the 101-point interpolation caps a perfect detector here, as `yolo val` does


# --------------------------------------------------------------------------------------------
# Fixtures: a synthetic dataset and a stand-in detector
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth")
    _synth.make_dataset(root, n_train=10, n_val=30, seed=3, size=(160, 128))
    return root


@pytest.fixture()
def weights(tmp_path) -> Path:
    path = tmp_path / "detector.pt"
    path.write_bytes(b"not a real network; the predictor is replaced in these tests")
    return path


def make_predictor(calls: list, keep: float = 1.0):
    """A predict_raw stand-in: the label files as detections at confidence 0.9, keeping `keep` of them."""

    def fake(weights, images, imgsz=640, batch=16, device=None, conf_floor=0.001, half=False, progress_every=500,
             max_det=P.DEFAULT_MAX_DET):
        calls.append({"n": len(images), "imgsz": imgsz})
        hw, boxes, conf, cls = [], [], [], []
        for path in images:
            h, w = cv2.imread(str(path)).shape[:2]
            lab = C.read_yolo_label(C.label_path_for(Path(path)))
            n_keep = int(np.ceil(len(lab) * keep))
            lab = lab[:n_keep]
            xyxy = np.stack([(lab[:, 1] - lab[:, 3] / 2) * w, (lab[:, 2] - lab[:, 4] / 2) * h,
                             (lab[:, 1] + lab[:, 3] / 2) * w, (lab[:, 2] + lab[:, 4] / 2) * h], axis=1) if len(lab) else np.zeros((0, 4))
            hw.append((h, w))
            boxes.append(xyxy.astype(np.float32))
            conf.append(np.full(len(lab), 0.9, dtype=np.float32))
            cls.append(lab[:, 0].astype(np.int16))
        return P.RawPreds(
            paths=[str(p) for p in images],
            hw=np.asarray(hw, dtype=np.int32).reshape(-1, 2),
            boxes=boxes, conf=conf, cls=cls,
            meta={"weights": str(weights), "weights_sha256": C.sha256_file(Path(weights)), "imgsz": imgsz,
                  "conf_floor": conf_floor, "raw_max_det": P.RAW_MAX_DET, "raw_cap_hits": 0, "max_det": max_det,
                  "ultralytics": "test-double"},
        )

    return fake


@pytest.fixture()
def calls(monkeypatch) -> list:
    """Install the perfect stand-in detector; the list records every inference call."""
    recorded: list = []
    monkeypatch.setattr(P, "predict_raw", make_predictor(recorded))
    monkeypatch.setattr(E, "model_class_names", lambda weights: list(NAMES))
    return recorded


def run_eval(root: Path, weights: Path, out: Path, *extra: str) -> dict:
    tag = "t"
    # NMS at IoU 1.0 suppresses nothing: two overlapping ground-truth people would otherwise lose one
    # detection to NMS and the "perfect detector" would not be perfect.
    args = ["--weights", str(weights), "--data-root", str(root), "--out-dir", str(out), "--tag", tag,
            "--imgsz", "160", "--bootstrap", "20", "--nms-iou", "1.0", *extra]
    assert E.main(args) == 0
    return json.loads((out / f"eval_{tag}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# The sealed test split
# --------------------------------------------------------------------------------------------


def test_test_split_is_refused_without_unseal_flag_before_anything_is_read(monkeypatch, capsys, tmp_path):
    def boom(*a, **k):
        raise AssertionError("the data root must not even be resolved on a refusal")

    monkeypatch.setattr(C, "resolve_data_root", boom)
    with pytest.raises(SystemExit) as exc:
        E.main(["--weights", str(tmp_path / "missing.pt"), "--split", "test"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "sealed" in err and "Phase 11" in err and "--unseal-test" in err


def test_unknown_split_is_refused_with_exit_2(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        E.main(["--weights", str(tmp_path / "w.pt"), "--split", "holdout"])
    assert exc.value.code == 2
    assert "train, val, test" in capsys.readouterr().err


def test_unseal_flag_moves_past_the_guard(capsys, tmp_path):
    # With the flag the run proceeds to the next check (the weights file), so the refusal is different.
    with pytest.raises(SystemExit) as exc:
        E.main(["--weights", str(tmp_path / "missing.pt"), "--split", "test", "--unseal-test"])
    assert exc.value.code == 2
    assert "weights not found" in capsys.readouterr().err


def test_unsealed_test_run_is_labelled_loudly(calls, weights, synth_root, tmp_path, capsys):
    root = tmp_path / "with_test"
    shutil.copytree(synth_root, root)
    shutil.copytree(root / "images" / "val", root / "images" / "test")
    shutil.copytree(root / "labels" / "val", root / "labels" / "test")
    report = run_eval(root, weights, tmp_path / "out", "--split", "test", "--unseal-test")
    assert report["split"] == "test"
    assert any("UNSEALED" in n for n in report["notes"])
    assert "TEST SPLIT UNSEALED" in capsys.readouterr().out


def test_hardset_never_lists_the_test_folder(calls, weights, synth_root, tmp_path, monkeypatch):
    root = tmp_path / "with_test"
    shutil.copytree(synth_root, root)
    shutil.copytree(root / "images" / "val", root / "images" / "test")
    real_iterdir = Path.iterdir

    def guarded(self):
        if self.parts[-2:] == ("images", "test"):
            raise AssertionError("images/test must never be listed by the hard-set gate")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", guarded)
    code = H.main(["--weights", str(weights), "--data-root", str(root), "--hard-set", str(root / "hard_set.txt"),
                   "--out-dir", str(tmp_path / "out"), "--imgsz", "160", "--bootstrap", "10"])
    assert code == 0


# --------------------------------------------------------------------------------------------
# Refusals about what is being evaluated
# --------------------------------------------------------------------------------------------


def test_image_without_modality_suffix_stops_the_run(calls, weights, synth_root, tmp_path, capsys):
    root = tmp_path / "bad"
    shutil.copytree(synth_root, root)
    victim = next((root / "images" / "val").glob("idd_*"))
    victim.rename(victim.with_name("frame_0001.jpg"))
    with pytest.raises(SystemExit) as exc:
        E.main(["--weights", str(weights), "--data-root", str(root), "--out-dir", str(tmp_path / "o"), "--imgsz", "160"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "frame_0001.jpg" in err and "_visible" in err and "guess" in err
    assert calls == []          # refused before any inference was paid for
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("flag,value", [("--conf-floor", "0"), ("--nms-iou", "1.5"), ("--report-conf", "0.0001"),
                                        ("--bootstrap", "-1"), ("--imgsz", "8")])
def test_bad_numeric_flags_are_refused(flag, value, weights, synth_root, tmp_path):
    with pytest.raises(SystemExit) as exc:
        E.main(["--weights", str(weights), "--data-root", str(synth_root), "--out-dir", str(tmp_path), flag, value])
    assert exc.value.code == 2


def test_unknown_size_class_is_refused(calls, weights, synth_root, tmp_path):
    with pytest.raises(SystemExit) as exc:
        E.main(["--weights", str(weights), "--data-root", str(synth_root), "--out-dir", str(tmp_path), "--size-classes", "tank"])
    assert exc.value.code == 2


# --------------------------------------------------------------------------------------------
# The report: slices, schema, classes, buckets
# --------------------------------------------------------------------------------------------


@pytest.fixture()
def report(calls, weights, synth_root, tmp_path) -> dict:
    return run_eval(synth_root, weights, tmp_path / "out")


def test_json_carries_every_required_field(report, synth_root):
    for key in ("schema", "tag", "created", "weights", "split", "imgsz", "size_basis", "nms_iou", "conf_floor",
                "report_conf", "n_images", "host", "ultralytics", "class_names", "composition", "index_used",
                "slices", "notes"):
        assert key in report, key
    assert report["schema"] == "truewatch.eval.v1"
    assert report["split"] == "val" and report["size_basis"] == "input"
    assert report["weights"]["sha256"] and report["class_names"] == NAMES
    assert report["created"].endswith("Z") and report["index_used"] is True
    assert report["host"]["cpu"] and "platform" in report["host"]
    n_val = len(list((synth_root / "images" / "val").iterdir()))
    assert report["n_images"] == n_val == sum(report["composition"].values())


def test_day_and_ir_are_separate_and_no_slice_blends_them(report):
    slices = report["slices"]
    assert "day" in slices and "ir" in slices
    assert slices["day"]["modality"] == "visible" and slices["ir"]["modality"] == "lwir"
    assert slices["day"]["n_images"] + slices["ir"]["n_images"] == report["n_images"]
    assert set(slices) <= {"day", "ir", "day/daylight", "day/night", "day/unresolved"}
    assert not any(k in slices for k in ("all", "overall", "total", "combined"))
    inside_day = slices["day/daylight"]["n_images"] + slices["day/night"]["n_images"] + \
        slices.get("day/unresolved", {"n_images": 0})["n_images"]
    assert inside_day == slices["day"]["n_images"]


def test_ground_truth_counts_match_the_label_files_per_modality(report, synth_root):
    expected = {"day": np.zeros(len(NAMES), dtype=int), "ir": np.zeros(len(NAMES), dtype=int)}
    for img in (synth_root / "images" / "val").iterdir():
        slice_name = C.slice_of_modality(C.modality_of_stem(img.stem))
        for c in C.read_yolo_label(C.label_path_for(img))[:, 0].astype(int):
            expected[slice_name][c] += 1
    for slice_name in ("day", "ir"):
        got = [report["slices"][slice_name]["classes"][n]["n_gt"] for n in NAMES]
        assert got == expected[slice_name].tolist()


def test_every_class_is_present_and_missing_ground_truth_is_none_not_zero(report):
    for slice_name in ("day", "ir"):
        for name in NAMES:
            row = report["slices"][slice_name]["classes"][name]
            assert set(row) == {"n_gt", "ap50", "ap50_95", "precision", "recall"}
            if row["n_gt"] == 0:
                assert row["ap50"] is None and row["ap50_95"] is None and row["recall"] is None
            else:
                assert row["ap50"] == pytest.approx(PERFECT, abs=1e-6)
                assert row["recall"] == pytest.approx(1.0)
    assert report["slices"]["ir"]["classes"]["truck"]["n_gt"] == 0  # IR frames of the fixture hold only people


def test_size_buckets_use_the_measurements_md_levels_for_person_day_and_ir(report):
    for slice_name in ("day", "ir"):
        rows = report["slices"][slice_name]["size_buckets"]["person"]
        assert [r["bucket_px"] for r in rows] == list(C.SIZE_BUCKETS_PX)
        assert any(r["n_gt"] for r in rows)
        for r in rows:
            if r["n_gt"] == 0:
                assert r["ap50"] is None and r["recall"] is None
    total = sum(r["n_gt"] for r in report["slices"]["day"]["size_buckets"]["person"])
    assert total == report["slices"]["day"]["classes"]["person"]["n_gt"]


def test_by_source_blocks_sum_to_the_slice(report):
    for slice_name in ("day", "ir"):
        block = report["slices"][slice_name]["by_source"]
        assert sum(v["n_images"] for v in block.values()) == report["slices"][slice_name]["n_images"]
        assert sum(v["person_n_gt"] for v in block.values()) == report["slices"][slice_name]["classes"]["person"]["n_gt"]
        for v in block.values():
            assert (v["person_ap50"] is None) == (v["person_n_gt"] == 0)


def test_bootstrap_interval_is_recorded_for_person(report):
    boot = report["slices"]["day"]["bootstrap"]["person"]
    assert boot["n_boot"] == 20 and boot["ap50_ci95"][0] <= boot["ap50_ci95"][1]


def test_json_has_no_nan_and_notes_flag_the_synthetic_fixture(report, tmp_path):
    text = json.dumps(report)
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(f"non-finite {c} in JSON")))
    assert any("synthetic fixture" in n and "NOT results" in n for n in report["notes"])


def test_csv_files_have_the_documented_columns_and_shape(report, tmp_path):
    out = tmp_path / "out"
    with (out / "eval_t_classes.csv").open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == E.CLASSES_CSV_COLUMNS
    assert len(rows) - 1 == len(report["slices"]) * len(NAMES)
    truck_ir = next(r for r in rows if r[0] == "ir" and r[1] == "truck")
    assert truck_ir[2] == "0" and truck_ir[3] == "n/a" and truck_ir[5] == "n/a"
    with (out / "eval_t_size.csv").open(encoding="utf-8") as fh:
        size_rows = list(csv.reader(fh))
    assert tuple(size_rows[0]) == E.SIZE_CSV_COLUMNS
    assert len(size_rows) - 1 == len(report["slices"]) * len(C.SIZE_BUCKETS_PX)
    assert size_rows[1][3].startswith(">=")  # the 77 px bucket is open above


def test_empty_ir_slice_still_exists_with_null_metrics(calls, weights, synth_root, tmp_path):
    root = tmp_path / "day_only"
    shutil.copytree(synth_root, root)
    for p in list((root / "images" / "val").glob("*_lwir.png")):
        p.unlink()
        C.label_path_for(p).unlink()
    report = run_eval(root, weights, tmp_path / "out")
    ir = report["slices"]["ir"]
    assert ir["n_images"] == 0 and ir["mean_over_present_classes"]["ap50"] is None
    assert all(row["ap50"] is None and row["n_gt"] == 0 for row in ir["classes"].values())
    assert all(r["n_gt"] == 0 for r in ir["size_buckets"]["person"])


def test_printed_tables_show_day_then_ir_then_subslices_and_the_composition(report, capsys):
    text = E.render_report(report)
    assert text.index("=== DAY (visible camera)") < text.index("=== IR (LWIR replicated to 3 channels)") < text.index("day/daylight")
    assert "composition (images per source [lighting])" in text
    assert "kaist" in text and "llvip" in text
    assert "--- person by object height, DAY" in text and "--- person by object height, IR" in text
    ir_section = text.split("=== IR (LWIR replicated to 3 channels)")[1].split("\n===")[0]
    truck_line = next(ln for ln in ir_section.splitlines() if ln.strip().startswith("truck"))
    assert "n/a" in truck_line and "0.000" not in truck_line  # no IR truck ground truth: n/a, never 0


def test_default_tag_is_weights_stem_split_and_short_sha():
    assert E.default_tag(Path("runs/x/best.pt"), "val", "abcdef0123456789") == "best_val_abcdef01"


def test_class_name_mismatch_is_reported(calls, weights, synth_root, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(E, "model_class_names", lambda w: ["person", "bicycle", "car"])
    report = run_eval(synth_root, weights, tmp_path / "out")
    assert any(n.startswith("CLASS NAMES DIFFER") for n in report["notes"])
    assert "CLASS NAMES DIFFER" in capsys.readouterr().out


def test_limit_evaluates_a_subset_and_says_so(calls, weights, synth_root, tmp_path):
    report = run_eval(synth_root, weights, tmp_path / "out", "--limit", "6")
    assert report["n_images"] == 6
    assert any("--limit 6" in n for n in report["notes"])


# --------------------------------------------------------------------------------------------
# Prediction cache
# --------------------------------------------------------------------------------------------


def test_reuse_preds_skips_inference_when_the_cache_matches(calls, weights, synth_root, tmp_path):
    out = tmp_path / "out"
    first = run_eval(synth_root, weights, out, "--reuse-preds")
    assert len(calls) == 1 and (out / "cache" / "preds_t.npz").exists()
    second = run_eval(synth_root, weights, out, "--reuse-preds")
    assert len(calls) == 1
    assert any("reused cached raw predictions" in n for n in second["notes"])
    assert second["slices"]["day"]["classes"] == first["slices"]["day"]["classes"]


def test_cache_is_not_reused_when_imgsz_or_weights_change(calls, weights, synth_root, tmp_path):
    out = tmp_path / "out"
    run_eval(synth_root, weights, out, "--reuse-preds")
    run_eval(synth_root, weights, out, "--reuse-preds", "--imgsz", "192")
    assert len(calls) == 2
    weights.write_bytes(b"a different network")
    run_eval(synth_root, weights, out, "--reuse-preds", "--imgsz", "192")
    assert len(calls) == 3


def test_inference_always_runs_without_the_reuse_flag(calls, weights, synth_root, tmp_path):
    out = tmp_path / "out"
    run_eval(synth_root, weights, out)
    run_eval(synth_root, weights, out)
    assert len(calls) == 2


def test_cache_mismatch_reasons(weights, tmp_path):
    images = [tmp_path / "a.jpg"]
    raw = P.RawPreds(paths=[str(images[0])], hw=np.array([[4, 4]], np.int32), boxes=[np.zeros((0, 4), np.float32)],
                     conf=[np.zeros(0, np.float32)], cls=[np.zeros(0, np.int16)],
                     meta={"weights_sha256": "abc", "imgsz": 160, "conf_floor": 0.001, "raw_max_det": P.RAW_MAX_DET})
    s = E.Settings(weights, 160, 8, None, 0.001, 0.7, 300, 0.35, 0, "input", ("person",))
    assert E.cache_mismatch(raw, "abc", s, images) is None
    assert "sha256" in E.cache_mismatch(raw, "zzz", s, images)
    assert "image list" in E.cache_mismatch(raw, "abc", s, [tmp_path / "b.jpg"])
    assert "conf floor" in E.cache_mismatch(raw, "abc", E.Settings(**{**s.__dict__, "conf_floor": 0.01}), images)


# --------------------------------------------------------------------------------------------
# Hard set: mapping entries to final images
# --------------------------------------------------------------------------------------------


def _val(*stems):
    return {s: Path(f"/d/images/val/{s}.jpg") for s in stems}


def test_index_mapping_uses_the_final_name_rule():
    records = [{"image": "/proc/kaist/set06_V000_I00000.jpg", "source": "kaist", "modality": "visible", "split": "val"}]
    m = H.map_entries(["/proc/kaist/set06_V000_I00000.jpg"], records, _val("kaist_set06_V000_I00000_visible"), set())
    assert m.resolved_entries == 1 and [p.stem for p in m.images] == ["kaist_set06_V000_I00000_visible"]
    assert m.unresolved == [] and m.leaked == [] and m.index_used


def test_without_an_index_the_raw_stem_is_matched_against_final_stems():
    entries = ["/somewhere/else/0000005_leftImg8bit.jpg", "/x/nothing_here.jpg"]
    m = H.map_entries(entries, [], _val("idd_0000005_leftImg8bit_visible", "kaist_other_lwir"), set())
    assert m.resolved_entries == 1 and m.unresolved == ["/x/nothing_here.jpg"] and not m.index_used


def test_an_llvip_pair_maps_to_both_images_and_both_are_counted():
    val = _val("llvip_010001_visible", "llvip_010001_lwir")
    m = H.map_entries(["/raw/llvip/visible/010001.jpg"], [], val, set())
    assert m.resolved_entries == 1 and m.pair_entries == 1
    assert sorted(p.stem for p in m.images) == ["llvip_010001_lwir", "llvip_010001_visible"]


def test_same_modality_matches_are_ambiguous_not_guessed():
    val = _val("kaist_I00019_visible", "idd_I00019_visible")
    m = H.map_entries(["/raw/I00019.jpg"], [], val, set())
    assert m.unresolved == ["/raw/I00019.jpg"] and m.ambiguous == ["/raw/I00019.jpg"] and m.images == []


def test_an_index_hit_names_one_record_even_when_a_sibling_modality_exists():
    records = [
        {"image": "/p/llvip/visible/010001.jpg", "source": "llvip", "modality": "visible", "split": "val"},
        {"image": "/p/llvip/infrared/010001.jpg", "source": "llvip", "modality": "lwir", "split": "val"},
    ]
    m = H.map_entries(["/p/llvip/infrared/010001.jpg"], records, _val("llvip_010001_visible", "llvip_010001_lwir"), set())
    assert [p.stem for p in m.images] == ["llvip_010001_lwir"] and m.pair_entries == 0


def test_an_entry_missing_from_the_index_falls_back_to_the_raw_stem():
    records = [{"image": "/p/kaist/other.jpg", "source": "kaist", "modality": "visible", "split": "val"}]
    m = H.map_entries(["/elsewhere/010001.jpg"], records, _val("llvip_010001_visible", "llvip_010001_lwir"), set())
    assert m.resolved_entries == 1 and m.pair_entries == 1


def test_entries_in_train_or_test_are_leaks_never_unresolved():
    records = [
        {"image": "/p/a.jpg", "source": "idd", "modality": "visible", "split": "train"},
        {"image": "/p/b.jpg", "source": "idd", "modality": "visible", "split": "test"},
        {"image": "/p/c.jpg", "source": "idd", "modality": "visible", "split": "val"},
    ]
    m = H.map_entries(["/p/a.jpg", "/p/b.jpg", "/p/c.jpg"], records, _val("idd_c_visible"), set())
    assert {x["split"] for x in m.leaked} == {"train", "test"} and m.resolved_entries == 1
    # no index: a name that exists only in train is a leak as well
    m2 = H.map_entries(["/p/a.jpg"], [], {}, {"idd_a_visible"})
    assert m2.leaked and m2.leaked[0]["split"] == "train"


def test_a_name_present_in_val_and_train_without_an_index_is_ambiguous():
    m = H.map_entries(["/p/a.jpg"], [], _val("idd_a_visible"), {"idd_a_visible"})
    assert m.ambiguous == ["/p/a.jpg"] and m.images == [] and m.leaked == []


def test_index_split_beats_a_name_clash_and_val_wins_between_duplicate_records():
    records = [
        {"image": "/p/a.jpg", "source": "idd", "modality": "visible", "split": "train"},
        {"image": "/p/a.jpg", "source": "idd", "modality": "visible", "split": "val"},
    ]
    m = H.map_entries(["/p/a.jpg"], records, _val("idd_a_visible"), {"idd_a_visible"})
    assert m.resolved_entries == 1 and m.leaked == []


def test_assert_all_in_val_refuses_an_image_from_another_folder(tmp_path, capsys):
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "images" / "train").mkdir(parents=True)
    ok, bad = tmp_path / "images" / "val" / "a.jpg", tmp_path / "images" / "train" / "b.jpg"
    ok.write_bytes(b"x"), bad.write_bytes(b"x")
    H.assert_all_in_val([ok], tmp_path)
    with pytest.raises(SystemExit) as exc:
        H.assert_all_in_val([ok, bad], tmp_path)
    assert exc.value.code == 2 and "b.jpg" in capsys.readouterr().err


def test_image_set_identity_ignores_the_mount_point():
    a = [Path("/kaggle/input/x/images/val/idd_1_visible.jpg"), Path("/kaggle/input/x/images/val/idd_2_visible.jpg")]
    b = [Path("/tmp/y/images/val/idd_2_visible.jpg"), Path("/tmp/y/images/val/idd_1_visible.jpg")]
    assert H.image_set_sha256(a) == H.image_set_sha256(b)
    assert H.image_set_sha256(a) != H.image_set_sha256(a[:1])


def test_manifest_reader_skips_blanks_and_comments_and_refuses_an_empty_file(tmp_path, capsys):
    path = tmp_path / "hs.txt"
    path.write_text("# frozen\n/a/1.jpg\n\n  /a/2.jpg  \n", encoding="utf-8")
    assert H.read_manifest(path) == ["/a/1.jpg", "/a/2.jpg"]
    path.write_text("\n# only a comment\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        H.read_manifest(path)
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        H.read_manifest(tmp_path / "absent.txt")


# --------------------------------------------------------------------------------------------
# Hard set: the gate arithmetic
# --------------------------------------------------------------------------------------------

SETTINGS = {"imgsz": 640, "nms_iou": 0.7, "conf_floor": 0.001, "max_det": 300, "report_conf": 0.35, "size_basis": "input"}


def numbers(ap_day=0.8, ap_ir=0.7, recall=0.9):
    rec = {s: {str(b): recall for b in H.GATE_BUCKETS} for s in ("day", "ir")}
    n_gt = {s: {str(b): 10 for b in H.GATE_BUCKETS} for s in ("day", "ir")}
    return {"person_ap50": {"day": ap_day, "ir": ap_ir}, "person_recall_at_report_conf": rec, "person_n_gt_by_bucket": n_gt}


def baseline_from(nums, **over):
    doc = {"schema": H.BASELINE_SCHEMA, "settings": dict(SETTINGS), "hard_set": {"image_set_sha256": "s" * 64}, **nums}
    doc.update(over)
    return doc


def gate(current, baseline, tol=0.02, settings=None, image_set="s" * 64):
    return H.compare_to_baseline(current, baseline, settings or dict(SETTINGS), image_set, tol)


def test_gate_buckets_are_the_ones_at_or_below_19_px():
    assert H.GATE_BUCKETS == (19, 14, 9)


def test_identical_numbers_pass_and_every_check_is_listed():
    checks = gate(numbers(), baseline_from(numbers()))
    assert all(c["ok"] for c in checks)
    names = [c["name"] for c in checks]
    assert names[:2] == ["same_inference_settings", "same_image_set"]
    assert names.count("person_ap50") == 2 and sum(n.startswith("person_recall_bucket_") for n in names) == 6
    assert all({"name", "slice", "current", "baseline", "delta", "ok"} <= set(c) for c in checks)


def test_a_drop_of_exactly_the_tolerance_passes_and_a_larger_one_fails():
    base = baseline_from(numbers(ap_day=0.80))
    assert all(c["ok"] for c in gate(numbers(ap_day=0.78), base, tol=0.02))
    failing = [c for c in gate(numbers(ap_day=0.77), base, tol=0.02) if not c["ok"]]
    assert [(c["name"], c["slice"]) for c in failing] == [("person_ap50", "day")]
    assert failing[0]["delta"] == pytest.approx(-0.03)


def test_an_improvement_never_fails():
    assert all(c["ok"] for c in gate(numbers(ap_day=0.95, ap_ir=0.9, recall=1.0), baseline_from(numbers())))


def test_small_bucket_recall_is_gated_per_slice():
    cur = numbers()
    cur["person_recall_at_report_conf"]["ir"]["9"] = 0.5
    failing = [c for c in gate(cur, baseline_from(numbers())) if not c["ok"]]
    assert [(c["name"], c["slice"]) for c in failing] == [("person_recall_bucket_9px", "ir")]
    assert failing[0]["n_gt"] == 10


def test_a_metric_that_disappeared_fails_and_one_never_measured_does_not():
    lost = numbers()
    lost["person_ap50"]["ir"] = None
    assert [c["slice"] for c in gate(lost, baseline_from(numbers())) if not c["ok"]] == ["ir"]
    both_none = numbers(ap_ir=None)
    assert all(c["ok"] for c in gate(both_none, baseline_from(numbers(ap_ir=None))))


def test_different_settings_or_a_different_image_set_fail_the_gate():
    base = baseline_from(numbers())
    assert [c["name"] for c in gate(numbers(), base, settings={**SETTINGS, "imgsz": 320}) if not c["ok"]] == ["same_inference_settings"]
    assert [c["name"] for c in gate(numbers(), base, image_set="t" * 64) if not c["ok"]] == ["same_image_set"]


def test_gate_is_null_without_a_baseline_and_when_recording_one(tmp_path):
    g = H.build_gate(tmp_path / "none.json", None, numbers(), SETTINGS, "s" * 64, 0.02, writing_baseline=False)
    assert g["passed"] is None and g["baseline"] is None and g["checks"] == [] and g["tolerance"] == 0.02
    w = H.build_gate(tmp_path / "none.json", None, numbers(), SETTINGS, "s" * 64, 0.02, writing_baseline=True)
    assert w["passed"] is None
    ok = H.build_gate(tmp_path / "b.json", baseline_from(numbers()), numbers(), SETTINGS, "s" * 64, 0.02, False)
    assert ok["passed"] is True and ok["baseline"].endswith("b.json")


def test_unreadable_or_foreign_baseline_is_refused(tmp_path):
    path = tmp_path / "b.json"
    assert H.load_baseline(path) is None
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        H.load_baseline(path)
    assert exc.value.code == 2
    path.write_text(json.dumps({"schema": "something.else"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        H.load_baseline(path)


# --------------------------------------------------------------------------------------------
# Hard set: end to end with the stand-in detector
# --------------------------------------------------------------------------------------------


def hs_args(root: Path, weights: Path, out: Path, *extra: str) -> list[str]:
    return ["--weights", str(weights), "--data-root", str(root), "--hard-set", str(root / "hard_set.txt"),
            "--out-dir", str(out), "--imgsz", "160", "--bootstrap", "10", "--nms-iou", "1.0", "--tag", "h", *extra]


def test_hardset_lifecycle_write_baseline_pass_then_fail_on_a_degraded_model(monkeypatch, weights, synth_root, tmp_path, capsys):
    out = tmp_path / "out"
    recorded: list = []
    monkeypatch.setattr(E, "model_class_names", lambda w: list(NAMES))

    # 1. no baseline yet: the gate does not run, passed is null, exit 0
    monkeypatch.setattr(P, "predict_raw", make_predictor(recorded))
    assert H.main(hs_args(synth_root, weights, out)) == 0
    first = json.loads((out / "hardset_h.json").read_text(encoding="utf-8"))
    assert first["schema"] == "truewatch.hardset.v1" and first["gate"]["passed"] is None
    assert "NOT EVALUATED" in capsys.readouterr().out

    # 2. record the baseline
    assert H.main(hs_args(synth_root, weights, out, "--write-baseline")) == 0
    baseline = json.loads((out / "hardset_baseline.json").read_text(encoding="utf-8"))
    assert baseline["schema"] == "truewatch.hardset-baseline.v1" and baseline["split"] == "val"
    assert baseline["weights"]["sha256"] == C.sha256_file(weights) and baseline["date"]
    assert baseline["person_ap50"]["day"] == pytest.approx(PERFECT, abs=1e-6)
    assert set(baseline["person_recall_at_report_conf"]["ir"]) == {"19", "14", "9"}
    assert any("synthetic fixture" in n for n in baseline["notes"])

    # 3. the same detector passes
    assert H.main(hs_args(synth_root, weights, out)) == 0
    passed = json.loads((out / "hardset_h.json").read_text(encoding="utf-8"))
    assert passed["gate"]["passed"] is True and passed["gate"]["baseline"].endswith("hardset_baseline.json")
    assert all(c["ok"] for c in passed["gate"]["checks"])

    # 4. a degraded detector (half the detections gone) fails with exit code 1
    monkeypatch.setattr(P, "predict_raw", make_predictor(recorded, keep=0.4))
    assert H.main(hs_args(synth_root, weights, out)) == 1
    failed = json.loads((out / "hardset_h.json").read_text(encoding="utf-8"))
    assert failed["gate"]["passed"] is False
    assert any(not c["ok"] for c in failed["gate"]["checks"])
    assert "HARD-SET GATE: FAILED" in capsys.readouterr().out


def test_hardset_report_keeps_the_eval_fields_and_the_hard_set_block(calls, weights, synth_root, tmp_path):
    out = tmp_path / "out"
    assert H.main(hs_args(synth_root, weights, out)) == 0
    doc = json.loads((out / "hardset_h.json").read_text(encoding="utf-8"))
    for key in ("schema", "tag", "created", "weights", "split", "imgsz", "size_basis", "nms_iou", "conf_floor",
                "report_conf", "n_images", "host", "slices", "notes", "class_names", "composition", "index_used"):
        assert key in doc
    hs = doc["hard_set"]
    assert hs["entries"] == hs["resolved"] == 15 and hs["unresolved"] == [] and hs["index_used"] is True
    assert hs["partial"] is False and hs["allow_partial"] is False
    assert doc["split"] == "val" and doc["n_images"] == hs["images"] == 15
    assert set(doc["gate"]) >= {"baseline", "tolerance", "passed", "checks"}
    assert doc["slices"]["day"]["modality"] == "visible" and doc["slices"]["ir"]["modality"] == "lwir"


def test_hardset_baseline_is_never_overwritten_without_force(calls, weights, synth_root, tmp_path, capsys):
    out = tmp_path / "out"
    assert H.main(hs_args(synth_root, weights, out, "--write-baseline")) == 0
    before = (out / "hardset_baseline.json").read_bytes()
    n_calls = len(calls)
    with pytest.raises(SystemExit) as exc:
        H.main(hs_args(synth_root, weights, out, "--write-baseline"))
    assert exc.value.code == 2 and "--force-baseline" in capsys.readouterr().err
    assert (out / "hardset_baseline.json").read_bytes() == before and len(calls) == n_calls  # refused before inference
    assert H.main(hs_args(synth_root, weights, out, "--write-baseline", "--force-baseline")) == 0


def test_force_baseline_without_write_baseline_is_a_usage_error(calls, weights, synth_root, tmp_path):
    with pytest.raises(SystemExit) as exc:
        H.main(hs_args(synth_root, weights, tmp_path / "out", "--force-baseline"))
    assert exc.value.code == 2


def test_unresolved_entries_stop_the_run_unless_partial_is_allowed_and_then_it_is_recorded(calls, weights, synth_root, tmp_path, capsys):
    manifest = tmp_path / "hs.txt"
    manifest.write_text((synth_root / "hard_set.txt").read_text(encoding="utf-8") + "/synthetic/processed/kaist/set99_V9_I9.jpg\n",
                        encoding="utf-8")
    args = ["--weights", str(weights), "--data-root", str(synth_root), "--hard-set", str(manifest),
            "--out-dir", str(tmp_path / "out"), "--imgsz", "160", "--bootstrap", "10", "--tag", "p"]
    with pytest.raises(SystemExit) as exc:
        H.main(args)
    assert exc.value.code == 2
    assert "silently shrinks" in capsys.readouterr().err and calls == []
    assert H.main(args + ["--allow-partial"]) == 0
    out = capsys.readouterr().out
    assert "PARTIAL HARD SET" in out and "NOT a pass" in out
    doc = json.loads((tmp_path / "out" / "hardset_p.json").read_text(encoding="utf-8"))
    assert doc["hard_set"]["partial"] is True and doc["hard_set"]["allow_partial"] is True
    assert doc["hard_set"]["entries"] == 16 and doc["hard_set"]["resolved"] == 15
    assert doc["hard_set"]["unresolved"] == ["/synthetic/processed/kaist/set99_V9_I9.jpg"]
    assert any("PARTIAL" in n for n in doc["notes"])


def test_a_hard_set_entry_from_train_is_refused_even_with_allow_partial(calls, weights, synth_root, tmp_path, capsys):
    train_record = next(json.loads(ln) for ln in (synth_root / "index" / "split.jsonl").read_text(encoding="utf-8").splitlines()
                        if json.loads(ln)["split"] == "train" and json.loads(ln)["source"] == "idd")
    manifest = tmp_path / "hs.txt"
    manifest.write_text((synth_root / "hard_set.txt").read_text(encoding="utf-8") + train_record["image"] + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        H.main(["--weights", str(weights), "--data-root", str(synth_root), "--hard-set", str(manifest), "--allow-partial",
                "--out-dir", str(tmp_path / "out"), "--imgsz", "160", "--tag", "l"])
    assert exc.value.code == 2
    assert "outside val" in capsys.readouterr().err and calls == []


def test_hardset_evaluates_only_the_manifest_images(calls, weights, synth_root, tmp_path):
    assert H.main(hs_args(synth_root, weights, tmp_path / "out")) == 0
    assert calls[-1]["n"] == 15 < len(list((synth_root / "images" / "val").iterdir()))


# --------------------------------------------------------------------------------------------
# Raw prediction limits and the post-NMS cap shared with the sweep
# --------------------------------------------------------------------------------------------


def test_nms_time_limit_is_lifted_only_inside_the_block(monkeypatch):
    pytest.importorskip("ultralytics")
    from ultralytics.utils import nms as nms_module

    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return "ran"

    monkeypatch.setattr(nms_module, "non_max_suppression", spy)
    with P.no_nms_time_limit():
        assert nms_module.non_max_suppression("preds", 0.001, 1.0, max_det=P.RAW_MAX_DET) == "ran"
    assert seen[-1]["max_time_img"] == P.NMS_MAX_TIME_IMG and seen[-1]["max_det"] == P.RAW_MAX_DET
    assert nms_module.non_max_suppression is spy       # restored afterwards
    with pytest.raises(RuntimeError):
        with P.no_nms_time_limit():
            raise RuntimeError("boom")
    assert nms_module.non_max_suppression is spy       # restored on error too


def test_raw_cap_is_above_a_640_px_head_and_hits_are_reported(monkeypatch, weights, synth_root, tmp_path):
    assert P.RAW_MAX_DET > sum((640 // s) ** 2 for s in (8, 16, 32))
    recorded: list = []
    inner = make_predictor(recorded)

    def capped(*args, **kwargs):
        raw = inner(*args, **kwargs)
        raw.meta.update({"raw_cap_hits": 2, "raw_cap_hit_examples": ["a_visible.jpg", "b_lwir.jpg"]})
        return raw

    monkeypatch.setattr(P, "predict_raw", capped)
    monkeypatch.setattr(E, "model_class_names", lambda w: list(NAMES))
    report = run_eval(synth_root, weights, tmp_path / "out")
    assert any("2 image(s) reached the raw cap" in n for n in report["notes"])


def test_sweep_reuses_the_post_nms_cap_recorded_by_evaluate(calls, weights, synth_root, tmp_path):
    import sweep_conf as S

    out = tmp_path / "out"
    run_eval(synth_root, weights, out, "--max-det", "7")
    cache = out / "cache" / "preds_t.npz"
    assert P.load_raw(cache).meta["max_det"] == 7
    assert S.main(["--preds", str(cache), "--data-root", str(synth_root), "--out-dir", str(out), "--tag", "s"]) == 0
    sweep = json.loads((out / "sweep_s.json").read_text(encoding="utf-8"))
    assert sweep["max_det"] == 7 and sweep["grid"]["max_det"] == 7
    assert S.main(["--preds", str(cache), "--data-root", str(synth_root), "--out-dir", str(out), "--tag", "o", "--max-det", "9"]) == 0
    override = json.loads((out / "sweep_o.json").read_text(encoding="utf-8"))
    assert override["max_det"] == 9 and any("overrides the cap 7" in n for n in override["notes"])


def test_reusing_a_cache_with_another_max_det_updates_the_recorded_cap(calls, weights, synth_root, tmp_path):
    out = tmp_path / "out"
    run_eval(synth_root, weights, out, "--reuse-preds", "--max-det", "7")
    run_eval(synth_root, weights, out, "--reuse-preds", "--max-det", "11")
    assert len(calls) == 1 and P.load_raw(out / "cache" / "preds_t.npz").meta["max_det"] == 11


# --------------------------------------------------------------------------------------------
# make_metrics.py over a real evaluation JSON
# --------------------------------------------------------------------------------------------


def _metrics_inputs(eval_doc: dict, **extra):
    import make_metrics as MM

    return MM, MM.Inputs(eval=eval_doc, files={"eval": ["eval_t.json"], **{k: [f"{k}.json"] for k in extra}}, **extra)


def test_make_metrics_renders_the_class_list_of_the_eval_json(report):
    """The dataset build may withdraw the cart class; the report must follow the eval JSON, not a fixed list."""
    import copy

    four = copy.deepcopy(report)
    four["class_names"] = [n for n in NAMES if n != "cart"]
    for s in four["slices"].values():
        s["classes"].pop("cart", None)
        s["at_report_conf"]["classes"].pop("cart", None)
    MM, inputs = _metrics_inputs(four)
    text = MM.render_metrics(inputs)
    table = text.split("## 3.")[1].split("## 4.")[0]
    assert "| cart |" not in table and all(f"| {n} |" in table for n in four["class_names"])
    _, five = _metrics_inputs(report)
    assert "| cart |" in MM.render_metrics(five).split("## 3.")[1].split("## 4.")[0]


def _with_person(doc: dict, key: str, ap: float, lo: float, hi: float, n_gt: int) -> None:
    s = doc["slices"][key]
    s["n_images"] = max(s.get("n_images") or 0, 1)
    s["classes"]["person"].update({"ap50": ap, "n_gt": n_gt})
    s.setdefault("bootstrap", {})["person"] = {"ap50_ci95": [lo, hi], "n_boot": 200, "n_clusters": 40, "n_images": 100}


def test_a_tiny_daylight_subset_cannot_make_the_day_target_partial(report):
    import copy

    import _metrics as M

    doc = copy.deepcopy(report)
    _with_person(doc, "day", 0.80, 0.77, 0.83, 500)
    _with_person(doc, "day/daylight", 0.97, 0.90, 0.99, 12)          # 12 boxes: an anecdote
    MM, inputs = _metrics_inputs(doc)
    day = next(v for v in MM.compute_verdicts(inputs) if v.key == "day_map")
    assert day.verdict == M.NOT_MET and "fewer than 30" in day.reason
    _with_person(doc, "day/daylight", 0.90, 0.87, 0.93, 400)         # large and established on its own
    day = next(v for v in MM.compute_verdicts(inputs) if v.key == "day_map")
    assert day.verdict == M.PARTIAL


def test_verdict_rows_never_round_a_miss_onto_the_target(report):
    import copy

    import _metrics as M

    doc = copy.deepcopy(report)
    _with_person(doc, "day", 0.84962, 0.82, 0.87, 500)
    MM, inputs = _metrics_inputs(doc)
    day = next(v for v in MM.compute_verdicts(inputs) if v.key == "day_map")
    assert day.verdict == M.NOT_MET
    row = next(line for line in MM.render_targets(inputs, MM.compute_verdicts(inputs), False).splitlines()
               if line.startswith("| Person mAP@50, day"))
    assert "0.8496" in row and "0.850" not in row


def test_random_weight_benchmarks_show_inference_only():
    import make_metrics as MM

    committed = json.loads((C.RESULTS_DIR / "benchmark_mac-apple-m5-cpu.json").read_text(encoding="utf-8"))
    trained = json.loads(json.dumps(committed))
    trained["label"] = "trained-host"
    trained["model"].update({"weights_kind": "trained", "weights_provenance": "fine-tuned export"})
    trained["pipeline_representative"] = True
    text = MM.render_latency(MM.Inputs(benchmarks=[committed, trained]))
    random_row = next(line for line in text.splitlines() if line.startswith("| mac-apple-m5-cpu"))
    trained_row = next(line for line in text.splitlines() if line.startswith("| trained-host"))
    assert f"{committed['inference_ms']['p50']:.1f}" in random_row
    assert f"{committed['pipeline_ms']['p50']:.1f}" not in random_row and "suppressed: random weights" in random_row
    assert f"{trained['pipeline_ms']['p50']:.1f} / {trained['pipeline_ms']['p95']:.1f}" in trained_row
    assert "suppressed for mac-apple-m5-cpu" in text and "inflated" in text


def test_export_section_prints_raw_and_gated_parity_with_labels():
    import make_metrics as MM

    ex = {"onnx": {"file": "t.onnx", "opset": 17, "imgsz": 640, "size_bytes": 1, "sha256": "a" * 64},
          "parity": {"passed": True, "box_units": "grid", "max_abs_diff": 8.9e-5, "raw_max_abs_diff": 1.8e-3,
                     "tolerance": 1e-3, "by_batch": {"1": 8.9e-5}, "input_kind": "synthetic"},
          "hf": {"url": "https://huggingface.co/o/m", "revision": "b" * 40,
                 "resolve_url": f"https://huggingface.co/o/m/resolve/{'b' * 40}/t.onnx"}}
    text = MM.render_export(ex)
    assert "raw max abs diff (output0 as emitted, box rows in pixels): 0.0018" in text
    assert "box rows in grid units" in text and "revision `" + "b" * 40 in text
