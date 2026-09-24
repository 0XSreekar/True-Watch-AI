"""Tests for export_onnx.py (parity, metadata, guards, upload path) and benchmark_cpu.py.

The export tests build a random-initialised YOLO11-n with five classes at 160 px, so a full
export plus parity run takes seconds and needs no dataset or weights download. The benchmark tests
use a hand-built ONNX graph and need only numpy, onnx and onnxruntime, because benchmark_cpu.py is
meant to run where torch and Ultralytics do not exist. No test touches the network: the Hugging Face
client is replaced by a recording fake, and a second fake that raises stands guard everywhere else.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import _common as C
import benchmark_cpu as B
import export_onnx as E

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
NAMES = ["person", "two_wheeler", "car", "truck", "cart"]
IMGSZ = 160
PARITY_LINE = re.compile(
    r"^PARITY raw_max_abs_diff=\d\.\d{3}e[-+]\d{2} gated_max_abs_diff=\d\.\d{3}e[-+]\d{2} "
    r"gated_box_units=(grid|normalized|pixels) tolerance=1e-03 -> (PASS|FAIL)$", re.M)


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------


def _random_checkpoint(path: Path, scale: str = "n", nc: int = 5, responsive: bool = True, head_std: float = 3e-4) -> Path:
    """A random-initialised YOLO11 checkpoint with `nc` classes, saved the way Ultralytics saves.

    A freshly initialised detector ignores its input: boxes sit on the anchor grid and scores on the head
    bias, so a wrong graph would still match it and a parity check on it proves little. With
    `responsive` the batch-norm statistics are set from one batch of data and the head is re-drawn with
    tiny weights, so the output varies with the input (by several pixels) while float32 noise stays
    far under the parity tolerance. A larger head (`head_std` around 0.05) makes noise exceed it.
    """
    pytest.importorskip("ultralytics")
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(0)
    yolo = YOLO(f"yolo11{scale}.yaml")
    detector = DetectionModel(f"yolo11{scale}.yaml", nc=nc, verbose=False)
    detector.names = dict(enumerate(NAMES[:nc]))
    detector.args = {"imgsz": IMGSZ}
    if responsive:
        for module in detector.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.momentum = 1.0  # one pass sets the running statistics to that batch's
        detector.train()
        with torch.no_grad():
            detector(torch.from_numpy(E.synthetic_inputs(8, IMGSZ, seed=5)))
        head = detector.model[-1]
        for branch in [*head.cv2, *head.cv3]:
            torch.nn.init.normal_(branch[-1].weight, 0.0, head_std)
            torch.nn.init.zeros_(branch[-1].bias)
        detector.eval()
    yolo.model = detector
    yolo.save(str(path))
    return path


@pytest.fixture(scope="module")
def export_stack(tmp_path_factory):
    """Skip unless the whole export toolchain is importable; return the pieces the tests share."""
    for module in ("torch", "ultralytics", "onnx", "onnxruntime"):
        pytest.importorskip(module)
    work = tmp_path_factory.mktemp("export")
    return {"weights": _random_checkpoint(work / "tiny.pt"), "work": work}


@pytest.fixture(scope="module")
def exported(export_stack, tmp_path_factory):
    """One real export shared by the read-only assertions: returns (onnx path, json record, stdout, exit code)."""
    out, results = tmp_path_factory.mktemp("out"), tmp_path_factory.mktemp("results")
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = E.main(["--weights", str(export_stack["weights"]), "--out", str(out), "--results-dir", str(results),
                       "--imgsz", str(IMGSZ), "--tag", "tiny"])
    return {
        "onnx": out / "tiny.onnx",
        "record": json.loads((results / "export_tiny.json").read_text(encoding="utf-8")),
        "stdout": buffer.getvalue(),
        "code": code,
        "results": results,
    }


@pytest.fixture()
def guarded_hub(monkeypatch):
    """Any construction of the Hugging Face client fails the test unless the test installs its own fake."""
    import huggingface_hub

    class Forbidden:
        def __init__(self, *args, **kwargs):
            raise AssertionError("the Hugging Face client was used but no upload was asked for")

    monkeypatch.setattr(huggingface_hub, "HfApi", Forbidden)
    return huggingface_hub


class FakeHfApi:
    """Records what would be sent to the Hub; each upload returns a CommitInfo-like object with a fresh sha."""

    instances: list["FakeHfApi"] = []
    private_repo = False

    def __init__(self, token=None, **kwargs):
        self.token, self.repos, self.uploads = token, [], []
        FakeHfApi.instances.append(self)

    def create_repo(self, repo_id, repo_type="model", private=False, exist_ok=False, **kwargs):
        self.repos.append({"repo_id": repo_id, "repo_type": repo_type, "private": private, "exist_ok": exist_ok})

    def repo_info(self, repo_id, repo_type="model", **kwargs):
        return type("Info", (), {"private": self.private_repo, "id": repo_id})()

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type="model", **kwargs):
        content = path_or_fileobj if isinstance(path_or_fileobj, (bytes, bytearray)) else Path(path_or_fileobj).read_bytes()
        oid = f"{len(self.uploads) + 1:x}".rjust(40, "a")
        self.uploads.append({"path_in_repo": path_in_repo, "repo_id": repo_id, "bytes": len(content), "content": content, "oid": oid})
        return type("CommitInfo", (), {"oid": oid, "commit_url": f"https://huggingface.co/{repo_id}/commit/{oid}"})()


# --------------------------------------------------------------------------------------------
# export_onnx.py: a real export
# --------------------------------------------------------------------------------------------


def test_real_export_passes_parity_and_prints_the_exact_line(exported):
    assert exported["code"] == E.EXIT_OK
    assert len(PARITY_LINE.findall(exported["stdout"])) == 1
    assert re.search(r"^PARITY raw_max_abs_diff=\S+ gated_max_abs_diff=\S+ gated_box_units=grid tolerance=1e-03 -> PASS$",
                     exported["stdout"], re.M)
    assert re.search(r"batch=1 .*raw_max_abs_diff=\S+ \(boxes px\) +gated: boxes\(grid\)=\S+ +scores=\S+", exported["stdout"])
    assert re.search(r"batch=3 .*raw_max_abs_diff=\S+ \(boxes px\) +gated: boxes\(grid\)=\S+ +scores=\S+", exported["stdout"])
    assert "raw   = " in exported["stdout"] and "gated = " in exported["stdout"]


def test_printed_numbers_are_the_recorded_raw_and_gated_differences(exported):
    parity = exported["record"]["parity"]
    line = PARITY_LINE.search(exported["stdout"]).group(0)
    assert f"raw_max_abs_diff={parity['raw_max_abs_diff']:.3e}" in line
    assert f"gated_max_abs_diff={parity['max_abs_diff']:.3e}" in line
    assert parity["box_units"] == "grid" and "stride" in parity["gated_metric"]
    assert set(parity["raw_by_batch"]) == {"1", "3"} and parity["raw_max_abs_diff"] == max(parity["raw_by_batch"].values())
    # grid units divide the box rows by 8, 16 or 32, so the gated number can never exceed the raw one
    assert parity["max_abs_diff"] <= parity["raw_max_abs_diff"] + 1e-12


def test_json_follows_the_export_schema(exported):
    rec = exported["record"]
    assert rec["schema"] == "truewatch.export.v1" and rec["tag"] == "tiny"
    assert set(rec["source_weights"]) == {"path", "sha256"}
    assert rec["onnx"]["file"] == "tiny.onnx" and "/" not in rec["onnx"]["file"]
    assert rec["onnx"]["sha256"] == C.sha256_file(exported["onnx"])
    assert rec["onnx"]["size_bytes"] == exported["onnx"].stat().st_size
    assert rec["onnx"]["opset"] == 17 and rec["onnx"]["imgsz"] == IMGSZ
    assert rec["onnx"]["dynamic_axes"] == {"images": {"0": "batch"}, "output0": {"0": "batch"}}
    assert rec["onnx"]["input_names"] == ["images"] and rec["onnx"]["output_names"] == ["output0"]
    assert rec["class_names"] == NAMES
    assert rec["hf"] is None
    assert set(rec["host"]) >= {"cpu", "platform", "python", "onnxruntime", "torch"}
    parity = rec["parity"]
    assert parity["passed"] is True and parity["tolerance"] == 1e-3
    assert set(parity["by_batch"]) == {"1", "3"}
    assert parity["max_abs_diff"] == max(parity["by_batch"].values()) < 1e-3
    assert parity["input_kind"] == "synthetic"
    # Sensitivity is measured in the same units as the tolerance (grid-unit boxes by default), so the
    # random-weight fixture may be flagged weak; the flag must agree with the number it is derived from.
    assert parity["input_sensitive"] is bool(parity["input_sensitivity"] >= 100 * parity["tolerance"])
    assert ("WEAK CHECK" in exported["stdout"]) is (not parity["input_sensitive"])


def test_only_batch_is_dynamic_and_spatial_size_is_fixed(exported):
    import onnxruntime as ort

    session = ort.InferenceSession(str(exported["onnx"]), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == ["batch", 3, IMGSZ, IMGSZ]
    anchors = sum((IMGSZ // s) ** 2 for s in (8, 16, 32))
    assert session.get_outputs()[0].shape == ["batch", 4 + len(NAMES), anchors]
    for batch in (1, 2, 5):
        out = session.run(None, {"images": np.zeros((batch, 3, IMGSZ, IMGSZ), np.float32)})[0]
        assert out.shape == (batch, 4 + len(NAMES), anchors)
    with pytest.raises(Exception):
        session.run(None, {"images": np.zeros((1, 3, IMGSZ * 2, IMGSZ * 2), np.float32)})


def test_metadata_lets_the_edge_read_names_and_size_without_ultralytics(exported):
    import onnx

    meta = B.read_onnx_header(exported["onnx"])["metadata"]
    assert json.loads(meta["class_names"]) == NAMES
    assert json.loads(meta["imgsz"]) == [IMGSZ, IMGSZ]
    assert meta["nc"] == "5" and meta["opset"] == "17" and meta["stride"] == "32" and meta["task"] == "detect"
    assert ast.literal_eval(meta["names"]) == dict(enumerate(NAMES))  # the form AutoBackend parses
    assert meta["source_weights_sha256"] == exported["record"]["source_weights"]["sha256"]
    assert "batch axis is dynamic" in meta["input"] and "No NMS" in meta["output"]
    proto = onnx.load(str(exported["onnx"]))
    assert {p.key: p.value for p in proto.metadata_props} == meta  # the dependency-free reader agrees with onnx
    assert proto.ir_version <= E.IR_VERSION_CAP
    assert [(o.domain, o.version) for o in proto.opset_import] == [("", 17)]
    assert B.read_onnx_header(exported["onnx"])["opsets"] == {"ai.onnx": 17}


def test_weights_never_land_in_results(exported):
    assert sorted(p.suffix for p in exported["results"].iterdir()) == [".json"]
    assert not [p for p in C.RESULTS_DIR.glob("*") if p.suffix in {".onnx", ".pt", ".engine"}]


def test_parity_detects_a_graph_that_differs_from_the_weights(exported, export_stack, tmp_path):
    import onnx
    from onnx import numpy_helper

    proto = onnx.load(str(exported["onnx"]))
    first_conv_weight = next(n for n in proto.graph.node if n.op_type == "Conv").input[1]
    for init in proto.graph.initializer:
        if init.name == first_conv_weight:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init) * 1.5, init.name))
    broken = tmp_path / "broken.onnx"
    onnx.save(proto, str(broken))
    parity = E.run_parity(broken, export_stack["weights"], IMGSZ, verbose=False)
    assert parity["passed"] is False and parity["max_abs_diff"] > 1e-3
    exact = parity["float64_reference"]   # the float64 run blames the graph, not float32 noise
    assert exact["graph_as_accurate_as_torch"] is False and exact["onnx_vs_float64"] > 10 * exact["torch_fp32_vs_float64"]


def test_an_input_insensitive_model_is_flagged_as_a_weak_check(export_stack, tmp_path, capsys, guarded_hub):
    inert = _random_checkpoint(tmp_path / "inert.pt", responsive=False)
    code = E.main(_args(tmp_path, inert, "--tag", "inert"))
    out = capsys.readouterr().out
    parity = json.loads((tmp_path / "r" / "export_inert.json").read_text(encoding="utf-8"))["parity"]
    assert code == E.EXIT_OK and parity["passed"] is True          # it still matches torch...
    assert parity["input_sensitive"] is False and "WEAK CHECK" in out  # ...but the record says the match proves little
    assert out.index("WEAK CHECK") < out.index("PARITY raw_max_abs_diff")  # and the PARITY line stays last


def test_float32_noise_is_told_apart_from_a_graph_error(tmp_path, capsys, guarded_hub):
    """A large head makes torch's own float32 output noisy in pixel units: the check fails, and says why."""
    noisy = _random_checkpoint(tmp_path / "noisy.pt", head_std=0.05)
    code = E.main(_args(tmp_path, noisy, "--tag", "noisy", "--box-units", "pixels"))
    captured = capsys.readouterr()
    record = json.loads((tmp_path / "r" / "export_noisy.json").read_text(encoding="utf-8"))
    exact = record["parity"]["float64_reference"]
    assert code == E.EXIT_PARITY_FAIL and record["parity"]["passed"] is False    # the spec'd tolerance is applied as written
    assert exact["graph_as_accurate_as_torch"] is True and exact["onnx_vs_float64"] < record["parity"]["max_abs_diff"]
    assert "float64 reference" in captured.out and "float32 rounding, not a graph error" in captured.out
    assert "below the float32 noise floor" in captured.err and "--box-units grid" in captured.err
    assert re.search(r"^PARITY raw_max_abs_diff=\S+ gated_max_abs_diff=\S+ gated_box_units=pixels tolerance=1e-03 -> FAIL$",
                     captured.out, re.M)
    assert record["parity"]["raw_max_abs_diff"] == record["parity"]["max_abs_diff"]  # in pixels, gated IS raw

    again = E.main(_args(tmp_path, noisy, "--tag", "noisy-grid"))  # grid is the default
    out = capsys.readouterr().out
    grid = json.loads((tmp_path / "r" / "export_noisy-grid.json").read_text(encoding="utf-8"))["parity"]
    assert again == E.EXIT_OK and grid["box_units"] == "grid" and grid["passed"] is True
    assert "boxes(grid)=" in out and "stride" in out
    assert grid["raw_max_abs_diff"] >= 1e-3 and "raw is NOT under the tolerance" in out  # raw is still printed honestly
    assert grid["max_abs_diff_scores"] == pytest.approx(record["parity"]["max_abs_diff_scores"], rel=0.5)  # scores are not rescaled


def test_failed_parity_exits_1_writes_evidence_and_never_uploads(export_stack, tmp_path, monkeypatch, capsys, guarded_hub):
    monkeypatch.setenv("HF_TOKEN", "dummy-token-for-test")
    code = E.main(["--weights", str(export_stack["weights"]), "--out", str(tmp_path / "o"), "--results-dir", str(tmp_path / "r"),
                   "--imgsz", str(IMGSZ), "--tag", "strict", "--parity-tol", "0", "--push-to-hub", "someone/model"])
    captured = capsys.readouterr()
    assert code == E.EXIT_PARITY_FAIL
    assert re.search(r"^PARITY raw_max_abs_diff=\S+ gated_max_abs_diff=\S+ gated_box_units=grid tolerance=0e\+00 -> FAIL$",
                     captured.out, re.M)
    record = json.loads((tmp_path / "r" / "export_strict.json").read_text(encoding="utf-8"))
    assert record["parity"]["passed"] is False and record["hf"] is None


def test_real_images_can_drive_the_parity_check(export_stack, tmp_path, capsys, guarded_hub):
    from _synth import make_dataset

    root = tmp_path / "ds"
    make_dataset(root, n_train=4, n_val=6, seed=1)
    code = E.main(["--weights", str(export_stack["weights"]), "--out", str(tmp_path / "o"), "--results-dir", str(tmp_path / "r"),
                   "--imgsz", str(IMGSZ), "--tag", "real", "--parity-images", str(root / "images" / "val"), "--parity-seed", "3"])
    assert code == E.EXIT_OK
    parity = json.loads((tmp_path / "r" / "export_real.json").read_text(encoding="utf-8"))["parity"]
    assert parity["input_kind"] == "images"
    assert len(parity["input_detail"]["files"]) == 3 and parity["input_detail"]["seed"] == 3
    assert "real images" in capsys.readouterr().out


def test_no_simplify_is_recorded_and_still_passes(export_stack, tmp_path, guarded_hub):
    code = E.main(["--weights", str(export_stack["weights"]), "--out", str(tmp_path / "o"), "--results-dir", str(tmp_path / "r"),
                   "--imgsz", str(IMGSZ), "--tag", "raw", "--no-simplify"])
    onnx_info = json.loads((tmp_path / "r" / "export_raw.json").read_text(encoding="utf-8"))["onnx"]
    assert code == E.EXIT_OK and onnx_info["simplified"] is False


# --------------------------------------------------------------------------------------------
# export_onnx.py: guards and the upload path
# --------------------------------------------------------------------------------------------


def _args(tmp_path: Path, weights: Path, *extra: str) -> list[str]:
    return ["--weights", str(weights), "--out", str(tmp_path / "o"), "--results-dir", str(tmp_path / "r"), "--imgsz", str(IMGSZ), *extra]


def test_push_to_hub_without_a_token_exits_2_before_exporting(export_stack, tmp_path, monkeypatch, capsys, guarded_hub):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    code = E.main(_args(tmp_path, export_stack["weights"], "--push-to-hub", "someone/model"))
    err = capsys.readouterr().err
    assert code == E.EXIT_USAGE
    assert "HF_TOKEN" in err and "never accepted as a flag" in err
    assert not (tmp_path / "o").exists()  # nothing was exported


def test_there_is_no_token_flag():
    assert "--token" not in E.build_parser().format_help()
    with pytest.raises(SystemExit):
        E.build_parser().parse_args(["--weights", "x.pt", "--token", "abc"])


def test_push_to_hub_uploads_onnx_and_card_and_pins_the_commit_for_the_edge(export_stack, tmp_path, monkeypatch, capsys):
    import huggingface_hub

    FakeHfApi.instances.clear()
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setenv("HF_TOKEN", "secret-token-value")
    code = E.main(_args(tmp_path, export_stack["weights"], "--tag", "hub", "--push-to-hub", "someone/truewatch-detector",
                        "--source-url", "https://example.org/src/truewatch"))
    captured = capsys.readouterr()
    assert code == E.EXIT_OK
    (api,) = FakeHfApi.instances
    assert api.token == "secret-token-value"
    assert api.repos == [{"repo_id": "someone/truewatch-detector", "repo_type": "model", "private": False, "exist_ok": True}]
    assert sorted(u["path_in_repo"] for u in api.uploads) == ["README.md", "hub.onnx"]
    onnx_upload = next(u for u in api.uploads if u["path_in_repo"] == "hub.onnx")
    onnx_file = tmp_path / "o" / "hub.onnx"
    assert onnx_upload["bytes"] == onnx_file.stat().st_size
    card = next(u for u in api.uploads if u["path_in_repo"] == "README.md")["content"].decode("utf-8")
    assert "https://example.org/src/truewatch" in card and "https://example.org/src/truewatch/blob/main/training/results/METRICS.md" in card

    record_text = (tmp_path / "r" / "export_hub.json").read_text(encoding="utf-8")
    hf = json.loads(record_text)["hf"]
    assert hf["repo_id"] == "someone/truewatch-detector" and hf["url"] == "https://huggingface.co/someone/truewatch-detector"
    assert hf["revision"] == onnx_upload["oid"]            # the commit that carries the .onnx, not a moving branch
    pinned = json.loads((tmp_path / "r" / "hf_model.json").read_text(encoding="utf-8"))
    assert pinned["schema"] == "truewatch.hf_model.v1"
    assert pinned["repo_id"] == "someone/truewatch-detector" and pinned["revision"] == onnx_upload["oid"] and pinned["file"] == "hub.onnx"
    assert pinned["resolve_url"] == f"https://huggingface.co/someone/truewatch-detector/resolve/{onnx_upload['oid']}/hub.onnx"
    assert pinned["sha256"] == C.sha256_file(onnx_file) and pinned["size"] == onnx_file.stat().st_size
    assert pinned["opset"] == 17 and pinned["imgsz"] == IMGSZ and pinned["class_names"] == NAMES
    assert pinned["source_url"] == "https://example.org/src/truewatch"
    assert "secret-token-value" not in record_text + json.dumps(pinned) + captured.out + captured.err
    assert "hf_model.json" in captured.out


def test_private_repo_is_refused_before_any_work(export_stack, tmp_path, monkeypatch, capsys, guarded_hub):
    monkeypatch.setenv("HF_TOKEN", "dummy-token-for-test")
    code = E.main(_args(tmp_path, export_stack["weights"], "--push-to-hub", "someone/model", "--private"))
    assert code == E.EXIT_USAGE and "without a token" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_an_existing_private_repo_is_warned_about(export_stack, tmp_path, monkeypatch, capsys):
    import huggingface_hub

    class PrivateRepo(FakeHfApi):
        private_repo = True

    monkeypatch.setattr(huggingface_hub, "HfApi", PrivateRepo)
    monkeypatch.setenv("HF_TOKEN", "dummy-token-for-test")
    code = E.main(_args(tmp_path, export_stack["weights"], "--tag", "priv", "--push-to-hub", "someone/model"))
    assert code == E.EXIT_OK and "PRIVATE" in capsys.readouterr().err
    assert json.loads((tmp_path / "r" / "export_priv.json").read_text(encoding="utf-8"))["hf"]["private"] is True


def test_an_upload_without_a_commit_sha_is_a_failed_upload(export_stack, tmp_path, monkeypatch, capsys):
    import huggingface_hub

    class NoSha(FakeHfApi):
        def upload_file(self, *args, **kwargs):
            super().upload_file(*args, **kwargs)
            return None

    monkeypatch.setattr(huggingface_hub, "HfApi", NoSha)
    monkeypatch.setenv("HF_TOKEN", "dummy-token-for-test")
    code = E.main(_args(tmp_path, export_stack["weights"], "--tag", "nosha", "--push-to-hub", "someone/model"))
    assert code == E.EXIT_UPLOAD_FAIL and "commit sha" in capsys.readouterr().err
    assert not (tmp_path / "r" / "hf_model.json").exists()


def test_upload_failure_exits_3_and_does_not_leak_the_token(export_stack, tmp_path, monkeypatch, capsys):
    import huggingface_hub

    class Failing(FakeHfApi):
        def create_repo(self, *args, **kwargs):
            raise RuntimeError("401 for token secret-token-value")

    monkeypatch.setattr(huggingface_hub, "HfApi", Failing)
    monkeypatch.setenv("HF_TOKEN", "secret-token-value")
    code = E.main(_args(tmp_path, export_stack["weights"], "--tag", "hubfail", "--push-to-hub", "someone/model"))
    captured = capsys.readouterr()
    assert code == E.EXIT_UPLOAD_FAIL
    assert "secret-token-value" not in captured.out + captured.err and "***" in captured.err
    assert json.loads((tmp_path / "r" / "export_hubfail.json").read_text(encoding="utf-8"))["hf"] is None


def test_model_card_states_the_licence_and_carries_no_accuracy_number():
    card = E.render_model_card(NAMES, 640, 17, "day", "a" * 64, "b" * 64)
    assert "AGPL-3.0" in card and "section 13" in card and "network" in card
    assert E.DEFAULT_SOURCE_URL in card and f"{E.DEFAULT_SOURCE_URL}/blob/main/training/results/METRICS.md" in card
    assert "not redistributed" in card and "IDD" in card and "LLVIP" in card and "KAIST" in card
    assert "METRICS.md" in card
    assert "person" in card and "cart" in card and "a" * 64 in card
    lowered = card.lower()
    assert "map" not in lowered.replace("mapping", "") and "ap50" not in lowered
    assert not re.search(r"\d+(\.\d+)?\s*%", card) and not re.search(r"\b0\.\d{2,}\b", card)
    assert "precision" not in lowered and "recall" not in lowered


def test_sealed_test_split_is_refused_unless_unsealed(tmp_path):
    sealed = tmp_path / "ds" / "images" / "test"
    sealed.mkdir(parents=True)
    assert E.points_at_test_split(sealed) and not E.points_at_test_split(tmp_path / "ds" / "images" / "val")
    with pytest.raises(SystemExit) as refused:
        E.refuse_sealed_test_split(sealed, unseal_test=False)
    assert "sealed until Phase 11" in str(refused.value) and "--unseal-test" in str(refused.value)
    E.refuse_sealed_test_split(sealed, unseal_test=True)  # allowed when the caller says so


def test_cli_refuses_the_sealed_split_before_doing_any_work(tmp_path, capsys):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"not a real checkpoint")
    sealed = tmp_path / "ds" / "images" / "test"
    sealed.mkdir(parents=True)
    code = E.main(["--weights", str(weights), "--out", str(tmp_path / "o"), "--results-dir", str(tmp_path / "r"), "--parity-images", str(sealed)])
    assert code == E.EXIT_USAGE and "sealed until Phase 11" in capsys.readouterr().err


def test_out_inside_results_is_refused(tmp_path, capsys):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    results = tmp_path / "results"
    code = E.main(["--weights", str(weights), "--out", str(results / "weights"), "--results-dir", str(results)])
    assert code == E.EXIT_USAGE and "inside the results directory" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra,needle",
    [(["--tag", "bad tag"], "--tag"), (["--imgsz", "100"], "multiple of 32"), (["--push-to-hub", "no-slash"], "owner/name"),
     (["--parity-tol", "-1"], "negative"), (["--source-url", "http://insecure.example/x"], "https://")],
)
def test_bad_arguments_exit_2_with_advice(tmp_path, capsys, extra, needle):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    code = E.main(["--weights", str(weights), "--out", str(tmp_path / "o"), *extra])
    assert code == E.EXIT_USAGE and needle in capsys.readouterr().err


def test_missing_or_wrong_weights_exit_2(tmp_path, capsys):
    assert E.main(["--weights", str(tmp_path / "nope.pt")]) == E.EXIT_USAGE
    (tmp_path / "w.onnx").write_bytes(b"x")
    assert E.main(["--weights", str(tmp_path / "w.onnx")]) == E.EXIT_USAGE
    assert ".pt" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# export_onnx.py: pure helpers
# --------------------------------------------------------------------------------------------


def test_parity_line_format_is_exact():
    assert E.parity_line(1.5e-3, 8.9e-5, "grid", 1e-3, True) == (
        "PARITY raw_max_abs_diff=1.500e-03 gated_max_abs_diff=8.900e-05 gated_box_units=grid tolerance=1e-03 -> PASS")
    assert E.parity_line(2.5e-3, 2.5e-3, "pixels", 1e-3, False) == (
        "PARITY raw_max_abs_diff=2.500e-03 gated_max_abs_diff=2.500e-03 gated_box_units=pixels tolerance=1e-03 -> FAIL")
    assert PARITY_LINE.match(E.parity_line(1.5e-3, 8.9e-5, "grid", 1e-3, True))


def test_grid_units_divide_each_anchor_by_its_own_stride():
    strides = E.anchor_strides([8, 16, 32], 64, 8 * 8 + 4 * 4 + 2 * 2)
    assert strides.tolist() == [8.0] * 64 + [16.0] * 16 + [32.0] * 4
    ref = np.zeros((1, 9, 84), np.float32)
    got = ref.copy()
    got[0, 0, 0], got[0, 1, 83] = 0.8, 3.2      # 0.8 px at stride 8 and 3.2 px at stride 32
    cmp = E.compare_outputs(ref, got, E.box_scale_for("grid", 64, [8, 16, 32], 84))
    assert cmp["max_abs_boxes"] == pytest.approx(0.1) and E.compare_outputs(ref, got)["max_abs_boxes"] == pytest.approx(3.2)
    with pytest.raises(SystemExit):
        E.anchor_strides([8, 16, 32], 64, 85)


def test_run_parity_defaults_match_the_cli():
    import inspect

    defaults = E.build_parser().parse_args(["--weights", "x.pt"])
    signature = inspect.signature(E.run_parity).parameters
    assert signature["box_units"].default == defaults.box_units == "grid"
    assert signature["tolerance"].default == defaults.parity_tol == 1e-3


def test_compare_outputs_splits_boxes_from_scores():
    ref = np.zeros((2, 9, 6), np.float32)
    got = ref.copy()
    got[0, 1, 2] = 5e-5       # a coordinate wobble
    got[1, 7, 3] = 2e-7       # a score wobble
    cmp = E.compare_outputs(ref, got)
    assert cmp["max_abs_boxes"] == pytest.approx(5e-5, rel=1e-3)
    assert cmp["max_abs_scores"] == pytest.approx(2e-7, rel=1e-3)
    assert cmp["max_abs"] == pytest.approx(5e-5, rel=1e-3)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_compare_outputs_fails_on_non_finite_values(bad):
    ref = np.ones((1, 9, 4), np.float32)
    got = ref.copy()
    got[0, 5, 0] = bad
    assert not np.isfinite(E.compare_outputs(ref, got)["max_abs"])


def test_compare_outputs_fails_on_shape_mismatch():
    cmp = E.compare_outputs(np.zeros((1, 9, 4), np.float32), np.zeros((1, 9, 5), np.float32))
    assert cmp["shape_ok"] is False and cmp["max_abs"] == float("inf")


def test_compare_outputs_box_scale_only_rescales_the_box_rows():
    ref = np.zeros((1, 9, 4), np.float32)
    got = ref.copy()
    got[0, 0, 0], got[0, 6, 0] = 0.64, 0.002
    cmp = E.compare_outputs(ref, got, box_scale=1 / 640)
    assert cmp["max_abs_boxes"] == pytest.approx(0.001) and cmp["max_abs_scores"] == pytest.approx(0.002)


def test_synthetic_inputs_are_deterministic_and_in_range():
    a, b, c = E.synthetic_inputs(3, 64, seed=0), E.synthetic_inputs(3, 64, seed=0), E.synthetic_inputs(3, 64, seed=1)
    assert a.shape == (3, 3, 64, 64) and a.dtype == np.float32
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.min() >= 0.0 and a.max() <= 1.0 and a.std() > 0.1
    assert np.array_equal(a[:1], E.synthetic_inputs(3, 64, seed=0)[:1])


def test_display_path_is_repo_relative_inside_the_repo():
    assert E.display_path(C.REPO_ROOT / "training" / "weights" / "best.pt") == str(Path("training") / "weights" / "best.pt")
    assert E.display_path(Path("/tmp/elsewhere/best.pt")) == "/tmp/elsewhere/best.pt"


# --------------------------------------------------------------------------------------------
# benchmark_cpu.py: dependency-free by construction
# --------------------------------------------------------------------------------------------


def _run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)


def test_benchmark_imports_and_reports_a_host_without_torch_ultralytics_or_yaml():
    code = (
        "import sys\n"
        "for name in ('torch', 'ultralytics', 'yaml', 'cv2', 'PIL'):\n"
        "    sys.modules[name] = None\n"          # any import of these now raises ImportError
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import benchmark_cpu as B\n"
        "host = B.collect_host()\n"
        "assert host['torch'] is None and host['effective_cpus'] >= 1\n"
        "print('ok')\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_collecting_the_host_does_not_import_torch():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "import benchmark_cpu as B\n"
        "B.collect_host()\n"
        "print('torch' in sys.modules and sys.modules['torch'] is not None)\n"
    )
    assert _run_python(code).stdout.strip() == "False"


def test_summarize_known_values():
    s = B.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s == {"n": 5, "p50": 3.0, "p95": 4.8, "mean": 3.0, "min": 1.0, "max": 5.0, "std": 1.5811}
    assert B.summarize([2.0])["std"] == 0.0
    with pytest.raises(ValueError):
        B.summarize([])


def test_time_calls_discards_warmup_and_returns_each_value():
    calls = []
    timings, values = B.time_calls(lambda: calls.append(1) or len(calls), runs=4, warmup=3)
    assert len(calls) == 7 and len(timings) == 4 and values == [4, 5, 6, 7]
    assert all(t >= 0 for t in timings)


def test_letterbox_geometry_and_padding():
    frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    lb = B.letterbox(frame, 640)
    assert lb.tensor.shape == (1, 3, 640, 640) and lb.tensor.dtype == np.float32
    assert lb.scale == 0.5 and lb.pad == (0, 140)
    assert lb.tensor.min() >= 0.0 and lb.tensor.max() <= 1.0
    assert np.all(lb.tensor[0, :, :140] == np.float32(114 / 255)) and np.all(lb.tensor[0, :, 500:] == np.float32(114 / 255))
    tall = B.letterbox(np.zeros((1000, 500, 3), np.uint8), 640)
    assert tall.scale == 0.64 and tall.pad[1] == 0 and tall.pad[0] == 160


def test_numpy_resize_matches_opencv_within_one_level():
    cv2 = pytest.importorskip("cv2")
    img = np.random.default_rng(1).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    for out_hw in ((45, 80), (60, 107), (180, 320)):
        ours = B.resize_bilinear(img, *out_hw)
        theirs = cv2.resize(img, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
        assert np.abs(ours.astype(int) - theirs.astype(int)).max() <= 1


def test_letterbox_matches_ultralytics_geometry():
    pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LetterBox as letterbox_cls
    frame = np.random.default_rng(2).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    theirs = letterbox_cls((640, 640), auto=False)(image=frame)
    ours = (B.letterbox(frame, 640).tensor[0].transpose(1, 2, 0) * 255).round().astype(int)
    assert theirs.shape == (640, 640, 3)
    assert np.abs(ours - theirs.astype(int)).max() <= 1


def test_nms_is_greedy_and_class_aware():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert B.nms(boxes, scores, 0.45).tolist() == [0, 2]
    assert B.nms(boxes, scores, 0.45, max_det=1).tolist() == [0]

    raw = np.zeros((1, 4 + 2, 3), np.float32)      # cx, cy, w, h then two class scores, 3 anchors
    raw[0, :4, 0] = [5, 5, 10, 10]
    raw[0, :4, 1] = [6, 6, 10, 10]
    raw[0, :4, 2] = [7, 7, 10, 10]
    raw[0, 4:, 0], raw[0, 4:, 1], raw[0, 4:, 2] = [0.9, 0.0], [0.0, 0.8], [0.6, 0.0]
    det = B.decode_and_nms(raw, conf=0.25, iou=0.45)
    # anchors 0 and 2 are the same class and overlap, so the weaker is dropped; anchor 1 is another class and stays
    assert det.candidates == 3 and det.classes.tolist() == [0, 1] and det.scores.tolist() == pytest.approx([0.9, 0.8])


def test_decode_maps_boxes_back_to_the_source_frame():
    raw = np.zeros((1, 4 + 1, 1), np.float32)
    raw[0, :, 0] = [320, 320, 100, 200, 0.9]       # cx, cy, w, h in input pixels, then one class score
    det = B.decode_and_nms(raw, conf=0.25, iou=0.45, scale=0.5, pad=(0, 140), frame_hw=(720, 1280))
    # input box (270, 220, 370, 420) minus the 140 px top pad, divided by the 0.5 scale
    assert det.boxes[0].tolist() == pytest.approx([540.0, 160.0, 740.0, 560.0])


def test_decode_with_nothing_above_the_threshold_is_empty_not_an_error():
    det = B.decode_and_nms(np.full((1, 9, 50), 0.01, np.float32), conf=0.25)
    assert det.candidates == 0 and det.boxes.shape == (0, 4)


def test_effective_cpus_and_the_quota_warning():
    space = {"cpu_count_usable": 32, "cpu_count_logical": 32, "cgroup_cpu_limit": 2.0}
    assert B.effective_cpus(space) == 2.0
    assert B.effective_cpus({"cpu_count_usable": 10, "cpu_count_logical": 10, "cgroup_cpu_limit": None}) == 10.0
    warning = B.thread_warning(space, 0)
    assert warning and "32 visible cores" in warning and "--threads 2" in warning
    assert B.thread_warning(space, 2) is None
    assert B.thread_warning({"cpu_count_usable": 10, "cgroup_cpu_limit": None}, 0) is None


# --------------------------------------------------------------------------------------------
# benchmark_cpu.py: end to end on a hand-built graph
# --------------------------------------------------------------------------------------------


def _tiny_detector(path: Path, imgsz: int = 64, nc: int = 5, input_type: str = "float", fixed_batch: bool = False) -> Path:
    """images (batch, 3, imgsz, imgsz) -> pool -> 1x1 conv -> (batch, 4+nc, cells) through a sigmoid."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    cells = (imgsz // 16) ** 2
    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(rng.normal(0, 3, (4 + nc, 3, 1, 1)).astype(np.float32), "W")
    shape = numpy_helper.from_array(np.array([0, 4 + nc, -1], np.int64), "shape")
    batch = 1 if fixed_batch else "batch"
    elem = TensorProto.FLOAT16 if input_type == "float16" else TensorProto.FLOAT
    nodes = []
    src = "images"
    if input_type == "float16":
        nodes.append(helper.make_node("Cast", ["images"], ["images32"], to=TensorProto.FLOAT))
        src = "images32"
    nodes += [
        helper.make_node("AveragePool", [src], ["pooled"], kernel_shape=[16, 16], strides=[16, 16]),
        helper.make_node("Conv", ["pooled", "W"], ["conv"]),
        helper.make_node("Reshape", ["conv", "shape"], ["flat"]),
        helper.make_node("Sigmoid", ["flat"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("images", elem, [batch, 3, imgsz, imgsz])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [batch, 4 + nc, cells])],
        initializer=[weight, shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 8
    entry = model.metadata_props.add()
    entry.key, entry.value = "class_names", json.dumps(NAMES)
    onnx.save(model, str(path))
    return path


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    return _tiny_detector(tmp_path_factory.mktemp("tiny") / "tiny.onnx")


def test_read_onnx_header_agrees_with_the_onnx_package(tiny_model):
    header = B.read_onnx_header(tiny_model)
    assert header["ir_version"] == 8 and header["opsets"] == {"ai.onnx": 17}
    assert json.loads(header["metadata"]["class_names"]) == NAMES


def test_run_benchmark_record_follows_the_schema(tiny_model):
    record = B.run_benchmark(tiny_model, "unit-host", imgsz=64, runs=6, warmup=2, threads=1, weights_provenance="hand-built graph")
    assert record["schema"] == "truewatch.benchmark.v1" and record["label"] == "unit-host"
    assert set(record) >= {"schema", "label", "created", "host", "model", "config", "inference_ms", "pipeline_ms", "caveat"}
    assert record["model"]["file"] == "tiny.onnx" and record["model"]["precision"] == "fp32" and record["model"]["opset"] == 17
    assert record["model"]["sha256"] == C.sha256_file(tiny_model) and record["model"]["weights_provenance"] == "hand-built graph"
    cfg = record["config"]
    assert (cfg["imgsz"], cfg["batch"], cfg["threads"], cfg["runs"], cfg["warmup"], cfg["provider"]) == (64, 1, 1, 6, 2, "CPUExecutionProvider")
    for key in ("inference_ms", "pipeline_ms"):
        assert set(record[key]) == {"n", "p50", "p95", "mean", "min", "max", "std"} and record[key]["n"] == 6
        assert record[key]["min"] <= record[key]["p50"] <= record[key]["p95"] <= record[key]["max"]
    assert record["host"]["effective_cpus"] >= 1 and "cgroup_cpu_limit" in record["host"] and "cpu_count_usable" in record["host"]
    detail = record["pipeline_detail"]
    assert detail["candidates_max"] > 0 and detail["detections_mean"] > 0    # the tiny graph does exercise decode and NMS


def test_weights_kind_marks_random_runs_and_only_trained_on_a_real_frame_is_representative(tiny_model, tmp_path):
    random_run = B.run_benchmark(tiny_model, "u", imgsz=64, runs=2, warmup=0, weights_provenance="randomly initialised YOLO11-s")
    assert random_run["model"]["weights_kind"] == "random" and random_run["pipeline_representative"] is False
    assert "inflated" in random_run["pipeline_note"]
    unknown = B.run_benchmark(tiny_model, "u", imgsz=64, runs=2, warmup=0)
    assert unknown["model"]["weights_kind"] == "unknown" and unknown["pipeline_representative"] is False
    synthetic = B.run_benchmark(tiny_model, "u", imgsz=64, runs=2, warmup=0, weights_kind="trained")
    assert synthetic["pipeline_representative"] is False and "--image" in synthetic["pipeline_note"]
    cv2 = pytest.importorskip("cv2")
    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), np.full((48, 80, 3), 90, np.uint8))
    real = B.run_benchmark(tiny_model, "u", imgsz=64, runs=2, warmup=0, weights_kind="trained", image=frame)
    assert real["pipeline_representative"] is True and real["pipeline_detail"]["frame_source"] == "file frame.png"


def test_onnx_flag_is_an_alias_for_model(tiny_model, tmp_path, capsys):
    out = tmp_path / "b.json"
    assert B.main(["--onnx", str(tiny_model), "--label", "unit", "--imgsz", "64", "--runs", "2", "--warmup", "0",
                   "--weights-kind", "trained", "--out", str(out)]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["model"]["file"] == "tiny.onnx" and record["model"]["weights_kind"] == "trained"
    assert "PIPELINE" in capsys.readouterr().out


def test_caveat_carries_every_required_statement():
    text = B.CAVEAT
    assert "FP32" in text and "not INT8" in text and "not the Jetson Orin Nano Super" in text
    assert "~30 ms TARGET" in text and "design budget" in text and "not comparable" in text
    assert "detector stage only" in text and "not end-to-end" in text
    assert re.findall(r"Jetson", text) and "TARGET" in text


def test_cli_prints_a_table_then_the_json_block_last(tiny_model, tmp_path, capsys):
    out = tmp_path / "benchmark_unit.json"
    assert B.main(["--model", str(tiny_model), "--label", "unit", "--imgsz", "64", "--runs", "5", "--warmup", "1", "--out", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "inference_ms" in stdout and "pipeline_ms" in stdout and "p50" in stdout and "p95" in stdout
    assert "CAVEAT" in stdout and "TARGET" in stdout
    assert stdout.rstrip().endswith(B.END_MARK)
    block = stdout.split(B.BEGIN_MARK, 1)[1].split(B.END_MARK, 1)[0]
    printed = json.loads(block)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert printed == written and printed["label"] == "unit"
    assert stdout.index("wrote ") < stdout.index(B.BEGIN_MARK)


def test_no_write_prints_only(tiny_model, tmp_path, capsys):
    out = tmp_path / "never.json"
    assert B.main(["--model", str(tiny_model), "--label", "unit", "--imgsz", "64", "--runs", "3", "--warmup", "0", "--out", str(out), "--no-write"]) == 0
    assert not out.exists() and B.BEGIN_MARK in capsys.readouterr().out


def test_default_output_path_is_results_benchmark_label(tiny_model, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    assert B.main(["--model", str(tiny_model), "--label", "lab.1", "--imgsz", "64", "--runs", "2", "--warmup", "0"]) == 0
    assert json.loads((tmp_path / "benchmark_lab.1.json").read_text(encoding="utf-8"))["schema"] == "truewatch.benchmark.v1"


def test_a_model_at_another_size_is_refused_with_advice(tiny_model, capsys):
    assert B.main(["--model", str(tiny_model), "--label", "unit", "--imgsz", "640", "--runs", "1", "--warmup", "0", "--no-write"]) == 2
    err = capsys.readouterr().err
    assert "fixed at 64" in err and "--imgsz 64" in err


def test_a_non_fp32_graph_is_refused(tmp_path, capsys):
    pytest.importorskip("onnx")
    half = _tiny_detector(tmp_path / "half.onnx", input_type="float16")
    assert B.main(["--model", str(half), "--label", "unit", "--imgsz", "64", "--runs", "1", "--warmup", "0", "--no-write"]) == 2
    assert "FP32" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra,needle",
    [(["--label", "bad label"], "--label"), (["--label", "unit", "--runs", "0"], "--runs"), (["--label", "unit", "--conf", "2"], "--conf")],
)
def test_bad_benchmark_arguments_exit_2(tiny_model, capsys, extra, needle):
    assert B.main(["--model", str(tiny_model), "--no-write", *extra]) == 2
    assert needle in capsys.readouterr().err


def test_missing_model_exits_2(tmp_path, capsys):
    assert B.main(["--model", str(tmp_path / "nope.onnx"), "--label", "unit", "--no-write"]) == 2
    assert "model not found" in capsys.readouterr().err
