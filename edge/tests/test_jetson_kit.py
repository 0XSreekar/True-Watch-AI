"""Tests for the Jetson kit (edge/tools/jetson). Everything here runs on a development machine.

NO number below is a measurement. The trtexec log and the nvpmodel / jetson_clocks texts are SYNTHETIC TEST
FIXTURES written from the documented output format, with obviously artificial values. They exist to test
parsers, never to stand in for the board.
"""

from __future__ import annotations

import json
import tarfile

import numpy as np
import pytest

from tools.jetson import accuracy_int8, benchmark, build_calib_set, build_engines, common

# ---- SYNTHETIC TEST FIXTURE (format from the trtexec documentation; values are artificial) ----------
SYNTHETIC_TRTEXEC_LOG = """\
&&&& RUNNING TensorRT.trtexec [TensorRT v100300] # trtexec --loadEngine=x.engine
[01/01/2000-00:00:00] [I] === Performance summary ===
[01/01/2000-00:00:00] [I] Throughput: 100.5 qps
[01/01/2000-00:00:00] [I] Latency: min = 1.5 ms, max = 9.5 ms, mean = 5.5 ms, median = 5.25 ms, percentile(90%) = 7.5 ms, percentile(95%) = 8.5 ms, percentile(99%) = 9.25 ms
[01/01/2000-00:00:00] [I] Enqueue Time: min = 0.5 ms, max = 0.75 ms, mean = 0.6 ms, median = 0.6 ms, percentile(90%) = 0.7 ms, percentile(95%) = 0.7 ms, percentile(99%) = 0.75 ms
[01/01/2000-00:00:00] [I] H2D Latency: min = 0.1 ms, max = 0.2 ms, mean = 0.15 ms, median = 0.15 ms, percentile(90%) = 0.2 ms, percentile(95%) = 0.2 ms, percentile(99%) = 0.2 ms
[01/01/2000-00:00:00] [I] GPU Compute Time: min = 1.0 ms, max = 9.0 ms, mean = 5.0 ms, median = 4.75 ms, percentile(90%) = 7.0 ms, percentile(95%) = 8.0 ms, percentile(99%) = 8.75 ms
[01/01/2000-00:00:00] [I] D2H Latency: min = 0.05 ms, max = 0.1 ms, mean = 0.07 ms, median = 0.07 ms, percentile(90%) = 0.1 ms, percentile(95%) = 0.1 ms, percentile(99%) = 0.1 ms
[01/01/2000-00:00:00] [I] Total Host Walltime: 3.0 s
&&&& PASSED TensorRT.trtexec [TensorRT v100300] # trtexec --loadEngine=x.engine
"""
SYNTHETIC_NVPMODEL = "NV Power Mode: SYNTH_MODE\n7\n"
SYNTHETIC_CLOCKS = """\
SOC family:tegra000  Machine:Synthetic Test Board
cpu0: Online=1 Governor=schedutil MinFreq=100 MaxFreq=2000 CurrentFreq=2000 IdleStates: WFI=1
GPU MinFreq=10 MaxFreq=1000 CurrentFreq=1000 IsBusy=0 Governor=nvhost_podgov
EMC MinFreq=5 MaxFreq=50 CurrentFreq=50 FreqOverride=0
"""


# ---- manifest and command lines -------------------------------------------------------------------


def test_manifest_loads_and_is_pinned():
    m = common.load_manifest()
    assert m["imgsz"] == 640 and len(m["sha256"]) == 64 and m["file"].endswith(".onnx")


def test_verify_onnx_rejects_wrong_bytes(tmp_path):
    f = tmp_path / "final.onnx"
    f.write_bytes(b"not the model")
    with pytest.raises(common.KitError):
        common.verify_onnx(f, common.load_manifest())


def test_build_cmd_fp16_and_int8(tmp_path):
    fp16 = common.build_cmd(tmp_path / "a.onnx", tmp_path / "a.engine", "fp16", 640, trtexec="trtexec")
    assert "--fp16" in fp16 and "--int8" not in fp16
    assert f"--shapes=images:1x3x640x640" in fp16 and "--skipInference" in fp16
    int8 = common.build_cmd(tmp_path / "a.onnx", tmp_path / "b.engine", "int8", 640, calib=tmp_path / "c.cache")
    assert "--int8" in int8 and "--fp16" in int8 and f"--calib={tmp_path / 'c.cache'}" in int8
    with pytest.raises(common.KitError):
        common.build_cmd(tmp_path / "a.onnx", tmp_path / "b.engine", "int8")
    with pytest.raises(common.KitError):
        common.build_cmd(tmp_path / "a.onnx", tmp_path / "b.engine", "int4")


def test_bench_cmd_excludes_warmup_via_trtexec_flag(tmp_path):
    cmd = common.bench_cmd(tmp_path / "e.engine", warmup_ms=1500, duration_s=20, times_json=tmp_path / "t.json")
    assert "--warmUp=1500" in cmd and "--duration=20" in cmd and f"--exportTimes={tmp_path / 't.json'}" in cmd


def test_build_engines_plan_names_and_dry_run(tmp_path, capsys):
    m = common.load_manifest()
    steps = build_engines.plan(tmp_path / "final.onnx", tmp_path, m, tmp_path / "calib.cache", "trtexec", ["fp16", "int8"], None)
    assert [s[0] for s in steps] == ["fp16", "int8"]
    assert steps[0][2].name == "final_fp16_640_b1.engine" and steps[1][2].name == "final_int8_640_b1.engine"
    assert build_engines.main(["--onnx", str(tmp_path / "final.onnx"), "--out-dir", str(tmp_path), "--dry-run"]) == 0
    assert "--int8" in capsys.readouterr().out


# ---- trtexec log parser ----------------------------------------------------------------------------


def test_parse_trtexec_log_synthetic():
    p = common.parse_trtexec_log(SYNTHETIC_TRTEXEC_LOG)
    gpu = p["sections"]["GPU Compute Time"]
    assert gpu == {"min": 1.0, "max": 9.0, "mean": 5.0, "median": 4.75, "p90": 7.0, "p95": 8.0, "p99": 8.75}
    assert p["sections"]["Latency"]["p99"] == 9.25
    assert p["throughput_qps"] == 100.5 and p["trt_version"] == "10.3.0" and p["passed"] is True


def test_parse_trtexec_log_missing_sections_are_absent():
    p = common.parse_trtexec_log("nothing useful here\n&&&& FAILED TensorRT.trtexec\n")
    assert p["sections"] == {} and p["throughput_qps"] is None and p["passed"] is False


def test_percentile_and_stats_and_export_times():
    assert common.percentile([1, 2, 3, 4], 50) == 2.5
    assert common.percentile([5], 99) == 5
    times = json.dumps([{"computeMs": float(i)} for i in range(1, 101)] + [{"other": 1}])
    s = common.latency_stats(common.parse_export_times(times))
    assert s["n"] == 100 and s["p50_ms"] == 50.5 and abs(s["p95_ms"] - 95.05) < 1e-9 and s["max_ms"] == 100


# ---- fingerprint parsers ---------------------------------------------------------------------------


def test_nvpmodel_and_clocks_parsers():
    assert common.parse_nvpmodel(SYNTHETIC_NVPMODEL) == {"name": "SYNTH_MODE", "id": 7}
    c = common.parse_jetson_clocks_show(SYNTHETIC_CLOCKS)
    assert c["gpu"]["MaxFreq"] == 1000 and c["cpu0"]["governor"] == "schedutil" and c["emc"]["CurrentFreq"] == 50
    assert common.clocks_pinned(c) is True
    c["gpu"]["CurrentFreq"] = 500
    assert common.clocks_pinned(c) is False
    assert common.clocks_pinned({}) is None


def test_is_jetson():
    assert common.is_jetson({"device_model": "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit"})
    assert not common.is_jetson({"device_model": None, "tegra_release": None})
    assert not common.is_jetson({"device_model": "Apple MacBook"})


def test_read_temps(tmp_path):
    z = tmp_path / "thermal_zone0"
    z.mkdir()
    (z / "type").write_text("gpu-thermal\n")
    (z / "temp").write_text("45500\n")
    assert benchmark.read_temps(str(tmp_path)) == {"thermal_zone0:gpu-thermal": 45.5}
    assert benchmark.read_temps(str(tmp_path / "missing")) == {}


def test_window_fps():
    stamps = [i * 0.5 for i in range(1, 41)]      # 2 fps for 20 s
    assert benchmark.window_fps(stamps, 5.0) == [2.0, 2.0, 2.0]
    assert benchmark.window_fps([], 5.0) == []


# ---- rendering -------------------------------------------------------------------------------------


def _record(on_jetson: bool) -> dict:
    return {"measured_on_jetson": on_jetson, "fingerprint": {"device_model": "SYNTHETIC", "tensorrt": "0.0.0"},
            "model": {"sha256": "ab" * 32},
            "detector_trt": {"synthetic_int8": {"n": 10, "p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0, "throughput_qps": 4.0}},
            "pipeline_e2e": {"off": {"n": 5, "p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0, "detector_p50_ms": None, "fps": 4.0}},
            "clip": {"name": "x.mp4", "width": 1920, "height": 1080}}


def test_render_markdown_banner_and_rows():
    assert "NOT A JETSON MEASUREMENT" in common.render_markdown(_record(False))
    md = common.render_markdown(_record(True))
    assert "NOT A JETSON" not in md and "| synthetic_int8 | 10 | 1.00 | 2.00 | 3.00 | 4.0 |" in md
    assert "| off | 5 | 1.00 | 2.00 | 3.00 | n/a | 4.0 |" in md and "not recorded" in md


def test_render_markdown_empty_record_makes_no_claim():
    md = common.render_markdown({"fingerprint": {}, "measured_on_jetson": True})
    assert common.WORDING_UNMEASURED in md


def test_scrub_paths():
    assert "/Users" not in common.scrub_paths("a /Users/someone/x b /pri" "vate/tmp/q/r c /home/nvidia/e")
    assert "/tmp" not in common.scrub_paths("x /tmp/abc y")


# ---- calibration list ------------------------------------------------------------------------------


def _names():
    out = []
    for src in ("idd", "flir", "llvip"):
        out += [f"{src}_{i:04d}_visible.jpg" for i in range(60)]
    for src in ("flir", "llvip"):
        out += [f"{src}_{i:04d}_lwir.jpg" for i in range(60)]
    return out + ["junk.jpg"]


def test_pick_is_deterministic_stratified_and_order_independent():
    names = _names()
    a = build_calib_set.pick(names, 40, 0.5, seed=3)
    b = build_calib_set.pick(list(reversed(names)), 40, 0.5, seed=3)
    assert a == b
    assert len(a) == 40 and sum(r["slice"] == "ir" for r in a) == 20
    assert {r["source"] for r in a if r["slice"] == "day"} == {"idd", "flir", "llvip"}
    assert all(r["name"] != "junk.jpg" for r in a)
    assert a != build_calib_set.pick(names, 40, 0.5, seed=4)


def test_pick_excludes_and_raises_when_short():
    names = _names()
    first = build_calib_set.pick(names, 20, seed=0)
    second = build_calib_set.pick(names, 20, seed=0, exclude=frozenset(r["name"] for r in first))
    assert not ({r["name"] for r in first} & {r["name"] for r in second})
    with pytest.raises(common.KitError):
        build_calib_set.pick(names, 1000, seed=0)


def test_manifest_hash_is_stable_and_licences_fall_back(tmp_path):
    rows = build_calib_set.pick(_names(), 20, seed=0)
    lic = build_calib_set.licences({"idd", "nowhere"}, tmp_path / "missing.yaml")
    assert lic == {"idd": "not recorded", "nowhere": "not recorded"}
    m1, m2 = (build_calib_set.build_manifest(rows, 0, 0.5, lic) for _ in range(2))
    assert m1 == m2 and m1["by_slice"] == {"day": 10, "ir": 10}


def test_real_sources_yaml_has_a_licence_for_every_source_in_the_dataset():
    lic = build_calib_set.licences({"idd", "flir", "llvip"})
    assert all(v != "not recorded" for v in lic.values())


def test_cli_refuses_output_inside_a_git_repo(tmp_path):
    root = tmp_path / "data"
    (root / "images" / "val").mkdir(parents=True)
    with pytest.raises(SystemExit):
        build_calib_set.main(["--data-root", str(root), "--out-dir", str(common.REPO_ROOT / "calib_out_never")])


def test_cli_writes_list_manifest_and_tarball_outside_git(tmp_path):
    root = tmp_path / "data"
    folder = root / "images" / "val"
    folder.mkdir(parents=True)
    for n in _names():
        (folder / n).write_bytes(b"x")
    out = tmp_path / "out"
    assert build_calib_set.main(["--data-root", str(root), "--out-dir", str(out), "--n", "20"]) == 0
    assert len((out / "calib_list.txt").read_text().splitlines()) == 20
    assert json.loads((out / "calib_manifest.json").read_text())["n"] == 20
    with tarfile.open(out / "calib_20.tar.gz") as t:
        assert len(t.getnames()) == 20


# ---- accuracy scoring ------------------------------------------------------------------------------


def _det(box, score, cls=0):
    from pipeline.types import Detection
    return Detection(bbox=box, class_name="person", score=score, camera_id="c", frame_index=0, timestamp=0.0,
                     modality="visible", class_id=cls)


def test_read_labels_xyxy(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.2 0.4\n")
    cls, box = accuracy_int8.read_labels_xyxy(f, 100, 50)
    assert cls.tolist() == [0] and np.allclose(box, [[40, 15, 60, 35]])
    assert accuracy_int8.read_labels_xyxy(tmp_path / "none.txt", 10, 10)[1].shape == (0, 4)


def test_score_reports_delta_per_slice_without_blending():
    gt_cls, gt_box = np.array([0]), np.array([[10.0, 10, 50, 90]])
    good, bad = [_det((10, 10, 50, 90), 0.9)], [_det((200, 200, 240, 280), 0.9)]
    fp32 = {"day": [accuracy_int8.image_data("flir_1_visible.jpg", gt_cls, gt_box, good)] * 30,
            "ir": [accuracy_int8.image_data("flir_2_lwir.jpg", gt_cls, gt_box, good)] * 30}
    int8 = {"day": [accuracy_int8.image_data("flir_1_visible.jpg", gt_cls, gt_box, good)] * 30,
            "ir": [accuracy_int8.image_data("flir_2_lwir.jpg", gt_cls, gt_box, bad)] * 30}
    r = accuracy_int8.score(fp32, int8, n_boot=30)
    assert r["day"]["delta_int8_minus_fp32"] == 0 and r["day"]["fp32_ap50"] > 0.99
    assert r["ir"]["int8_ap50"] == 0 and r["ir"]["delta_int8_minus_fp32"] < -0.9
    md = accuracy_int8.render_markdown({"measured_on_jetson": False, "n_images": 60, "model_sha256": "x", "slices": r})
    assert "NOT RUN ON A JETSON" in md and "| ir | 30 |" in md


def test_trt_session_metadata_helper():
    from tools.jetson.trt_session import meta_map
    assert json.loads(meta_map(["person", "car"])["class_names"]) == ["person", "car"]
    assert meta_map([]) == {}
