"""Export a fine-tuned YOLO11 detector to ONNX and prove the graph computes what torch computes.

The parity check is the point of this file. An ONNX file that loads and runs is not evidence that
it is the same function as the weights it came from: a lost fusion, a constant baked at the wrong
batch size or a simplifier bug all still run. So after the export the same inputs go through the
fused FP32 torch model and through onnxruntime (CPUExecutionProvider) at batch 1 and batch 3, the
raw output tensors are compared, and a difference of 1e-3 or more exits non-zero. Batch 3 also
proves the batch axis is really dynamic, since a graph traced at batch 1 with a constant baked in
would fail there.

Why not `yolo export ... dynamic=True`. In Ultralytics 8.4.155 (engine/exporter.py, export_onnx)
that flag makes batch, height AND width dynamic on the input and batch and anchors dynamic on the
output, and switches the Detect head to rebuild its anchor grid inside the graph. The edge only
ever feeds 640x640 (a whole frame letterboxed, or a 640x640 SAHI tile), so this script keeps the
spatial size static (fixed 8400 anchors that fold into constants) and declares only the batch axis
dynamic. That graph is smaller, friendlier to onnxslim and to TensorRT later, and cannot silently
accept an input size the detector was never validated at. The model is prepared by Ultralytics'
own Exporter (deep copy, eval, FP32, Conv+BN fusion, split C2f, Detect in export mode): the
Exporter is run up to its on_export_start hook, the prepared module is captured, and the trace is
done here with torch2onnx and the batch-only dynamic_axes. Nothing of the preparation is copied
into this file, so it cannot drift from what Ultralytics does. The hook set is replaced with the
capture alone, which also keeps Ultralytics' default analytics callback out of an export.

The graph carries its own description as ONNX metadata_props (class names as JSON, input size,
the output layout) so the edge can read them with onnxruntime alone. The Ultralytics keys (names,
imgsz, stride, task) are written too, in the format its AutoBackend parses, so the .onnx can be
loaded through Ultralytics.

One Ultralytics 8.4.155 behaviour to know when doing that (nn/backends/onnx.py): the ONNX backend
sets `dynamic` from whether the session's output batch dimension is symbolic, after it has applied
the metadata. Any graph with a dynamic batch therefore counts as "dynamic" and `predict` letterboxes
to a minimal rectangle (for example 512x640), which this graph's fixed height and width reject.
Callers that run this .onnx through Ultralytics must pass `rect=False`, which gives the square
letterbox the edge uses. The parity check here does not go through Ultralytics, so it is unaffected.

The .onnx is written to --out (default training/weights/) and never to results/: metrics JSON is
committed, weights are not. Nothing is uploaded unless --push-to-hub is given, and then only the
.onnx and a generated model card; the token comes from $HF_TOKEN only.

Exit codes: 0 exported and parity passed, 1 parity FAILED, 2 bad input or environment (missing
weights, no HF_TOKEN with --push-to-hub, sealed test split, output inside results/), 3 the upload
failed after a good export.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np

import _common as C
from benchmark_cpu import letterbox, read_image_rgb, read_onnx_header  # numpy-only helpers shared with the benchmark

SCHEMA = "truewatch.export.v1"
EXIT_OK, EXIT_PARITY_FAIL, EXIT_USAGE, EXIT_UPLOAD_FAIL = 0, 1, 2, 3

DEFAULT_OPSET = 17
DEFAULT_IMGSZ = 640
DEFAULT_TOLERANCE = 1e-3
PARITY_BATCHES = (1, 3)
SENSITIVITY_FACTOR = 100  # outputs must move by this many tolerances between inputs for the check to mean anything
IR_VERSION_CAP = 10  # newer IR versions are refused by older onnxruntime builds on the edge
INPUT_NAME, OUTPUT_NAME = "images", "output0"
DYNAMIC_AXES = {INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}}

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

INPUT_DESCRIPTION = (
    "images: float32 NCHW RGB scaled to 0-1, letterboxed to imgsz x imgsz on 114-grey padding. "
    "The batch axis is dynamic; height and width are fixed."
)
OUTPUT_DESCRIPTION = (
    "output0: float32 (batch, 4+nc, anchors). Rows 0-3 are cx, cy, w, h in input pixels; rows 4 onward "
    "are per-class scores in 0-1 after the sigmoid. No NMS is applied: decode and suppress in the caller."
)


@contextlib.contextmanager
def _quiet_ultralytics() -> Iterator[None]:
    """Hold Ultralytics' logger at WARNING so its model summary and banner do not bury the parity output."""
    from ultralytics.utils import LOGGER

    previous = LOGGER.level
    LOGGER.setLevel(logging.WARNING)
    try:
        yield
    finally:
        LOGGER.setLevel(previous)


class _Prepared(Exception):
    """Raised by the capture hook once the Exporter has finished preparing the model."""


@dataclass
class PreparedModel:
    model: "object"            # torch.nn.Module: fused, FP32, eval, Detect in export mode
    example: "object"          # torch.Tensor (1, 3, imgsz, imgsz) the Exporter traces with
    class_names: list[str]
    stride: int
    imgsz: int
    ultralytics_version: str


# --------------------------------------------------------------------------------------------
# Preparing the torch model exactly as the Exporter does
# --------------------------------------------------------------------------------------------


def prepare_torch_model(weights: Path, imgsz: int) -> PreparedModel:
    """Load `weights` and return the module Ultralytics' Exporter would trace, plus its example input."""
    from ultralytics import YOLO, __version__
    from ultralytics.engine.exporter import Exporter

    with _quiet_ultralytics():
        yolo = YOLO(str(weights), task="detect")
    if yolo.task != "detect":
        raise SystemExit(f"{weights} is a {yolo.task!r} model; this exporter is for detection weights")

    captured: dict = {}

    def capture(exporter) -> None:
        captured["model"], captured["im"] = exporter.model, exporter.im
        captured["names"], captured["stride"] = exporter.metadata["names"], exporter.metadata["stride"]
        captured["imgsz"] = exporter.imgsz
        raise _Prepared

    exporter = Exporter(
        overrides={"mode": "export", "format": "onnx", "imgsz": imgsz, "batch": 1, "device": "cpu",
                   "dynamic": False, "simplify": False, "data": None, "verbose": False}
    )
    exporter.callbacks = {"on_export_start": [capture]}
    try:
        with _quiet_ultralytics():
            exporter(model=yolo.model)
    except _Prepared:
        pass
    else:  # pragma: no cover - would mean Ultralytics no longer fires the hook we rely on
        raise RuntimeError("Ultralytics did not reach on_export_start; this script targets ultralytics 8.4.x")

    names = captured["names"]
    ordered = [str(names[i]) for i in sorted(names)]
    if sorted(names) != list(range(len(ordered))):
        raise SystemExit(f"class ids in {weights} are not 0..n-1: {sorted(names)}")
    if captured["imgsz"][0] != captured["imgsz"][1]:
        raise SystemExit(f"non-square export size {captured['imgsz']}; the edge feeds square input only")
    return PreparedModel(captured["model"], captured["im"], ordered, int(captured["stride"]), int(captured["imgsz"][0]), __version__)


# --------------------------------------------------------------------------------------------
# The export
# --------------------------------------------------------------------------------------------


def build_metadata(prepared: PreparedModel, opset: int, source_sha256: str) -> dict[str, str]:
    """metadata_props for the graph. Values are strings, as ONNX requires."""
    names = dict(enumerate(prepared.class_names))
    return {
        "description": "TRUEWATCH appearance-channel detector, YOLO11 fine-tuned, exported by export_onnx.py",
        "task": "detect",
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "version": prepared.ultralytics_version,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "stride": str(prepared.stride),
        "batch": "1",
        "channels": "3",
        "end2end": "False",
        "imgsz": str([prepared.imgsz, prepared.imgsz]),   # a JSON-compatible list, and what AutoBackend expects
        "names": str(names),                              # Python-literal dict: the format AutoBackend parses
        "class_names": json.dumps(prepared.class_names),  # the same names as JSON, for a consumer without Ultralytics
        "nc": str(len(prepared.class_names)),
        "opset": str(opset),
        "input": INPUT_DESCRIPTION,
        "output": OUTPUT_DESCRIPTION,
        "source_weights_sha256": source_sha256,
    }


def _batch_is_dynamic(model_proto) -> bool:
    dim = model_proto.graph.input[0].type.tensor_type.shape.dim[0]
    return dim.HasField("dim_param") and bool(dim.dim_param)


def _slim(model_proto):
    """onnxslim the graph; keep the original if slimming fails or costs the dynamic batch axis."""
    try:
        import onnxslim

        slimmed = onnxslim.slim(model_proto)
    except Exception as exc:  # a simplifier failure must not block an export the parity check can still judge
        print(f"warning: onnxslim failed ({exc}); keeping the unsimplified graph", file=sys.stderr)
        return model_proto, False
    if slimmed is None or not _batch_is_dynamic(slimmed):
        print("warning: onnxslim removed the dynamic batch axis; keeping the unsimplified graph", file=sys.stderr)
        return model_proto, False
    return slimmed, True


def export_graph(prepared: PreparedModel, out_file: Path, opset: int, simplify: bool, source_sha256: str) -> dict:
    """Trace `prepared` with a batch-only dynamic axis, finish the graph and write `out_file`."""
    import onnx
    import torch
    from ultralytics.utils.export.engine import torch2onnx

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tw_export_") as tmp:
        raw = Path(tmp) / "raw.onnx"
        with torch.no_grad():
            torch2onnx(
                prepared.model, prepared.example, raw, opset=opset,
                input_names=[INPUT_NAME], output_names=[OUTPUT_NAME], dynamic=DYNAMIC_AXES,
            )
        model_proto = onnx.load(str(raw))

    simplified = False
    if simplify:
        model_proto, simplified = _slim(model_proto)

    for key, value in build_metadata(prepared, opset, source_sha256).items():
        entry = model_proto.metadata_props.add()  # slimming drops the props, so they are written last
        entry.key, entry.value = key, value
    if model_proto.ir_version > IR_VERSION_CAP:
        model_proto.ir_version = IR_VERSION_CAP
    onnx.checker.check_model(model_proto)

    staging = out_file.with_name(out_file.name + ".tmp")
    onnx.save(model_proto, str(staging))
    os.replace(staging, out_file)  # a killed run never leaves a half-written .onnx under the final name
    return {"simplified": simplified, "onnx_version": onnx.__version__, "ir_version": int(model_proto.ir_version)}


def _onnxslim_version() -> str | None:
    try:
        import onnxslim

        return onnxslim.__version__
    except ImportError:
        return None


def describe_graph(onnx_file: Path) -> dict:
    """Input and output names and shapes as onnxruntime reads them, batch shown by its symbolic name."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    shape = lambda spec: [d if isinstance(d, int) else str(d) for d in spec.shape]  # noqa: E731
    return {
        "input_names": [s.name for s in session.get_inputs()],
        "output_names": [s.name for s in session.get_outputs()],
        "input_shapes": {s.name: shape(s) for s in session.get_inputs()},
        "output_shapes": {s.name: shape(s) for s in session.get_outputs()},
    }


# --------------------------------------------------------------------------------------------
# Parity: torch versus onnxruntime on the same input
# --------------------------------------------------------------------------------------------


def synthetic_inputs(n: int, imgsz: int, seed: int) -> np.ndarray:
    """Fixed-seed float32 (n, 3, imgsz, imgsz) in 0..1: coarse random blocks plus fine noise.

    Blocks give the network edges and flat regions to respond to, as a frame does, where pure noise
    would leave most activations near zero and hide a numeric problem.
    """
    rng = np.random.default_rng(seed)
    cells = max(imgsz // 32, 1)
    coarse = rng.random((n, 3, cells, cells), dtype=np.float32)
    blocks = np.repeat(np.repeat(coarse, imgsz // cells, axis=2), imgsz // cells, axis=3)
    blocks = np.pad(blocks, ((0, 0), (0, 0), (0, imgsz - blocks.shape[2]), (0, imgsz - blocks.shape[3])), mode="edge")
    noise = rng.normal(0.0, 0.03, size=blocks.shape).astype(np.float32)
    return np.clip(blocks + noise, 0.0, 1.0).astype(np.float32)


def points_at_test_split(path: Path) -> bool:
    parts = Path(path).resolve().parts
    return any(a == "images" and b == "test" for a, b in zip(parts, parts[1:]))


def refuse_sealed_test_split(folder: Path, unseal_test: bool) -> None:
    """Exit with the reason when `folder` is an images/test directory and --unseal-test was not given."""
    if points_at_test_split(folder) and not unseal_test:
        raise SystemExit(
            f"refusing to read {folder}: the test split is sealed until Phase 11, so no earlier step may look at it. "
            "Use a folder under images/val, or pass --unseal-test only when Phase 11 has been opened on purpose."
        )


def image_inputs(folder: Path, n: int, imgsz: int, seed: int, unseal_test: bool) -> tuple[np.ndarray, list[str]]:
    """n letterboxed real images (RGB, 0..1) chosen with `seed` from the files directly inside `folder`."""
    refuse_sealed_test_split(folder, unseal_test)
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in C.IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"no images directly inside {folder}; point --parity-images at a folder such as <dataset>/images/val")
    order = np.random.default_rng(seed).permutation(len(files))
    chosen = [files[order[i % len(files)]] for i in range(n)]  # cycles when the folder holds fewer than n
    batch = np.concatenate([letterbox(read_image_rgb(p), imgsz).tensor for p in chosen], axis=0)
    return batch, [p.name for p in chosen]


def compare_outputs(reference: np.ndarray, candidate: np.ndarray, box_scale: float = 1.0) -> dict:
    """Max absolute differences over all channels, the four box rows and the score rows.

    Boxes are pixels (up to the input size) and scores are 0..1, so one number over both would let
    a score error hide under coordinate rounding; they are reported separately. `box_scale`
    multiplies the box rows first (1/imgsz turns pixels into fractions of the input). Non-finite
    values fail the comparison outright.
    """
    if reference.shape != candidate.shape:
        return {"shape_ok": False, "max_abs": float("inf"), "max_abs_boxes": float("inf"), "max_abs_scores": float("inf")}
    diff = np.abs((reference.astype(np.float64) - candidate.astype(np.float64)) * compare_scale(reference.shape[1], box_scale))
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    peak = lambda a: float(a.max()) if finite and a.size else float("inf")  # noqa: E731
    return {"shape_ok": True, "max_abs": peak(diff), "max_abs_boxes": peak(diff[:, :4]), "max_abs_scores": peak(diff[:, 4:])}


def parity_line(max_abs_diff: float, tolerance: float, passed: bool) -> str:
    return f"PARITY max_abs_diff={max_abs_diff:.3e} tolerance={tolerance:.0e} -> {'PASS' if passed else 'FAIL'}"


def float64_reference(model, x: np.ndarray, torch_out: np.ndarray, onnx_out: np.ndarray, box_scale: float) -> dict:
    """How far torch's float32 result and the ONNX result each are from the same weights run in float64.

    A torch-vs-ONNX difference cannot say which side is off. This can: the float64 run is the
    reference, and if the ONNX graph is no farther from it than torch's own float32 output is, the
    difference is float32 rounding (which grows with the stride the boxes are scaled by), not a
    graph error.
    """
    import copy

    import torch

    exact_model = copy.deepcopy(model).double()
    for module in exact_model.modules():
        if hasattr(module, "anchors") and hasattr(module, "shape"):
            module.shape = None  # rebuild the Detect anchor grid in float64
    with torch.inference_mode():
        out = exact_model(torch.from_numpy(x).double())
    exact = (out[0] if isinstance(out, (tuple, list)) else out).cpu().numpy()
    torch_gap = compare_outputs(exact, torch_out, box_scale)["max_abs"]
    onnx_gap = compare_outputs(exact, onnx_out, box_scale)["max_abs"]
    return {
        "batch": int(x.shape[0]),
        "torch_fp32_vs_float64": torch_gap,
        "onnx_vs_float64": onnx_gap,
        "graph_as_accurate_as_torch": bool(onnx_gap <= 1.5 * torch_gap),
    }


def run_parity(
    onnx_file: Path,
    weights: Path,
    imgsz: int,
    tolerance: float = DEFAULT_TOLERANCE,
    batches: Sequence[int] = PARITY_BATCHES,
    images_dir: Path | None = None,
    seed: int = 0,
    unseal_test: bool = False,
    box_units: str = "pixels",
    verbose: bool = True,
) -> dict:
    """Compare torch and onnxruntime raw outputs at each batch size; return the `parity` record.

    The torch side is prepared afresh from the .pt (not the module that was traced), so the check
    covers load, fuse and export as one chain, and the ONNX side is the file on disk. When the check
    fails, a float64 run of the same weights says whether the fault is the graph or float32 noise.
    """
    import onnxruntime as ort
    import torch

    box_scale = 1.0 / imgsz if box_units == "normalized" else 1.0
    box_label = "norm" if box_units == "normalized" else "px"
    reference = prepare_torch_model(weights, imgsz)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_file), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    n_inputs = max(batches)
    if images_dir is not None:
        pool, sources = image_inputs(images_dir, n_inputs, imgsz, seed, unseal_test)
        kind, detail = "images", {"files": sources, "seed": seed, "n": n_inputs}
    else:
        pool, kind, detail = synthetic_inputs(n_inputs, imgsz, seed), "synthetic", {"seed": seed, "n": n_inputs}

    by_batch, by_boxes, by_scores, shape_ok = {}, {}, {}, True
    for b in batches:
        x = np.ascontiguousarray(pool[:b])
        with torch.inference_mode():
            out = reference.model(torch.from_numpy(x))
        expected = (out[0] if isinstance(out, (tuple, list)) else out).cpu().numpy()
        got = session.run(None, {input_name: x})[0]
        cmp = compare_outputs(expected, got, box_scale)
        shape_ok &= cmp["shape_ok"]
        by_batch[str(b)], by_boxes[str(b)], by_scores[str(b)] = cmp["max_abs"], cmp["max_abs_boxes"], cmp["max_abs_scores"]
        if verbose:
            print(
                f"  batch={b}  output {tuple(got.shape)}  max_abs_diff={cmp['max_abs']:.3e}  "
                f"boxes({box_label})={cmp['max_abs_boxes']:.3e}  scores={cmp['max_abs_scores']:.3e}"
                + ("" if cmp["shape_ok"] else f"  SHAPE MISMATCH torch {tuple(expected.shape)}")
            )
    worst = max(by_batch.values())
    passed = bool(shape_ok and np.isfinite(worst) and worst < tolerance)

    # A network whose output barely depends on its input (random initialisation is the usual case: boxes
    # sit on the anchor grid and scores on the bias) passes any parity check, right or wrong. Measure how
    # far the torch output moves between different inputs so the record says whether PASS is informative.
    moved = np.abs((expected[:1] - expected[1:]) * compare_scale(expected.shape[1], box_scale)).max() if expected.shape[0] > 1 else None
    sensitivity = None if moved is None else float(moved)
    sensitive = None if sensitivity is None else bool(sensitivity >= SENSITIVITY_FACTOR * max(tolerance, 1e-9))

    exact = None
    if not passed and shape_ok and np.isfinite(worst):
        exact = float64_reference(reference.model, x, expected, got, box_scale)
        if verbose:
            print(
                f"  float64 reference (batch {exact['batch']}): onnx-vs-float64={exact['onnx_vs_float64']:.3e}  "
                f"torch-fp32-vs-float64={exact['torch_fp32_vs_float64']:.3e}"
            )
            print(
                "  reading: " + (
                    "the ONNX graph is no farther from exact than torch's own float32 result, so the difference is float32 "
                    "rounding, not a graph error."
                    if exact["graph_as_accurate_as_torch"] else
                    "the ONNX graph is farther from exact than torch's float32 result: treat the difference as a graph error."
                )
            )
    return {
        "tolerance": tolerance,
        "box_units": box_units,
        "max_abs_diff": worst,
        "max_abs_diff_boxes": max(by_boxes.values()),
        "max_abs_diff_scores": max(by_scores.values()),
        "by_batch": by_batch,
        "by_batch_boxes": by_boxes,
        "by_batch_scores": by_scores,
        "passed": passed,
        "input_kind": kind,
        "input_detail": detail,
        "input_sensitivity": sensitivity,
        "input_sensitive": sensitive,
        "float64_reference": exact,
    }


def compare_scale(channels: int, box_scale: float) -> np.ndarray:
    """(1, channels, 1) multiplier that puts the four box rows in the compared units."""
    scale = np.ones((1, channels, 1), dtype=np.float64)
    scale[:, :4] = box_scale
    return scale


# --------------------------------------------------------------------------------------------
# Optional upload
# --------------------------------------------------------------------------------------------


def render_model_card(class_names: Sequence[str], imgsz: int, opset: int, tag: str, source_sha256: str, onnx_sha256: str) -> str:
    """The README pushed with the weights. It states the licence terms and carries no accuracy figure."""
    classes = "\n".join(f"- {i}: {name}" for i, name in enumerate(class_names))
    return f"""---
license: agpl-3.0
library_name: onnx
pipeline_tag: object-detection
tags:
- object-detection
- yolo11
- onnx
---

# TRUEWATCH detector ({tag})

ONNX export of the single YOLO11-s detector behind the appearance channel of the TRUEWATCH
border-surveillance stack, fine-tuned on public research datasets and exported with `export_onnx.py`
from the TRUEWATCH source repository.

## Interface

- File: `{tag}.onnx`, ONNX opset {opset}, FP32.
- Input `images`: float32, shape (batch, 3, {imgsz}, {imgsz}), RGB scaled to 0-1, letterboxed onto
  114-grey padding. The batch axis is dynamic. Height and width are fixed.
- Output `output0`: float32, shape (batch, {4 + len(class_names)}, anchors). Rows 0-3 are cx, cy, w, h in input
  pixels, the remaining rows are per-class scores after the sigmoid. There is no NMS in the graph.
- The class names and input size are also embedded in the file as ONNX metadata_props.

Classes, by id:

{classes}

## Licence: read this before you use the model

- Ultralytics YOLO11 is licensed under AGPL-3.0. These weights are a derivative of it and are offered
  under the same licence. If you run this model, or a modified version, in a service that people reach over a
  network, AGPL-3.0 section 13 requires you to offer those users the corresponding source of that service.
  The licence text is at https://www.gnu.org/licenses/agpl-3.0.html. This card is not legal advice.
- The training data are public research datasets (IDD, LLVIP and KAIST). Each has its own terms, which may
  restrict some uses. The datasets are not redistributed here, and using this model grants no rights to them.
  Check the terms of each dataset before you rely on the model.

## Accuracy

This card carries no accuracy figures. Measured results, each with the split and the host it was measured on,
are published in `METRICS.md` in the source repository. Nothing measured on synthetic data is a result.

## Provenance

- Source weights SHA-256: `{source_sha256}`
- This file SHA-256: `{onnx_sha256}`
"""


def push_to_hub(onnx_file: Path, repo_id: str, private: bool, card: str, token: str) -> dict:
    """Create the repo if needed and upload the .onnx and the card. Returns {repo_id, url}."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(onnx_file), path_in_repo=onnx_file.name, repo_id=repo_id, repo_type="model",
        commit_message=f"Add {onnx_file.name}",
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id, repo_type="model",
        commit_message="Add model card",
    )
    return {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}"}


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export best.pt to ONNX (dynamic batch, fixed 640x640) and check torch/onnxruntime parity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Exit codes: 0 done, 1 parity FAILED, 2 bad input or environment, 3 upload failed. "
            "--push-to-hub reads the token from $HF_TOKEN only; an existing repo keeps its visibility and its README "
            "is replaced by the generated model card."
        ),
    )
    p.add_argument("--weights", required=True, type=Path, help="Ultralytics .pt checkpoint, e.g. runs/day/weights/best.pt")
    p.add_argument("--out", type=Path, default=C.TRAINING_ROOT / "weights", help="directory for the .onnx (never results/)")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="square input side; a multiple of the stride 32")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset")
    p.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True, help="run onnxslim on the graph")
    p.add_argument("--parity-tol", type=float, default=DEFAULT_TOLERANCE, help="max absolute difference torch vs onnxruntime that still passes")
    p.add_argument(
        "--box-units", choices=("pixels", "normalized"), default="normalized",
        help="units of the four box rows in the parity comparison: fractions of the input size (pixels / imgsz, the default: "
             "scale-invariant, so float32 rounding on 0-640 coordinates does not swamp the check), or raw input pixels",
    )
    p.add_argument("--parity-images", type=Path, default=None, help="folder of real images for the parity input (else a fixed-seed synthetic tensor)")
    p.add_argument("--parity-seed", type=int, default=0, help="seed for the synthetic tensor and for choosing images")
    p.add_argument("--unseal-test", action="store_true", help="allow --parity-images to point at images/test (sealed until Phase 11)")
    p.add_argument("--tag", default=None, help="name of this export, used for <tag>.onnx and export_<tag>.json; the weights file stem when omitted")
    p.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR, help="where export_<tag>.json is written")
    p.add_argument("--push-to-hub", metavar="REPO_ID", default=None, help="upload the .onnx and a model card to this Hugging Face repo, e.g. user/truewatch-detector")
    p.add_argument("--private", action="store_true", help="create the repo as private (only when it does not exist yet)")
    return p


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def display_path(path: Path) -> str:
    """Repo-relative when the file is inside the repository, else as given."""
    try:
        return str(Path(path).resolve().relative_to(C.REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def validate(args: argparse.Namespace) -> str:
    """Check the arguments before any slow work and return the tag. Raises SystemExit with advice."""
    tag = args.tag or args.weights.stem
    if not _NAME_RE.match(tag):
        raise SystemExit(f"--tag {tag!r} must match [A-Za-z0-9][A-Za-z0-9._-]* because it becomes a file name")
    if not args.weights.is_file() or args.weights.suffix != ".pt":
        raise SystemExit(f"--weights must be an existing .pt file, got {args.weights}")
    if args.imgsz < 32 or args.imgsz % 32:
        raise SystemExit(f"--imgsz {args.imgsz} must be a positive multiple of 32 (the detector's largest stride)")
    if args.opset < 12:
        raise SystemExit("--opset must be at least 12")
    if args.parity_tol < 0:
        raise SystemExit("--parity-tol must not be negative")
    if _inside(args.out, args.results_dir):
        raise SystemExit(
            f"--out {args.out} is inside the results directory {args.results_dir}. Weights are never written there: "
            "results/ is committed, weights are not. Use the default training/weights/ or a temp dir."
        )
    if args.parity_images is not None:
        if not args.parity_images.is_dir():
            raise SystemExit(f"--parity-images {args.parity_images} is not a directory")
        refuse_sealed_test_split(args.parity_images, args.unseal_test)
    if args.push_to_hub is not None:
        if not _REPO_ID_RE.match(args.push_to_hub):
            raise SystemExit(f"--push-to-hub expects owner/name, got {args.push_to_hub!r}")
        if not os.environ.get("HF_TOKEN", "").strip():
            raise SystemExit(
                "--push-to-hub needs a Hugging Face write token in the HF_TOKEN environment variable. "
                "Create one at https://huggingface.co/settings/tokens, then run `export HF_TOKEN=...` in your shell "
                "and repeat the command. The token is never accepted as a flag and never written to a file."
            )
    return tag


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tag = validate(args)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(f"error: {exc.code}", file=sys.stderr)
            return EXIT_USAGE
        raise

    out_file = args.out / f"{tag}.onnx"
    inside_repo = _inside(out_file, C.REPO_ROOT)
    if inside_repo and not (_inside(out_file, C.TRAINING_ROOT / "weights") or _inside(out_file, C.TRAINING_ROOT / "runs")):
        print(f"warning: {out_file} is inside the repository outside training/weights and training/runs; keep it out of git", file=sys.stderr)

    source_sha = C.sha256_file(args.weights)
    print(f"weights   {args.weights}  sha256 {source_sha}")
    try:
        prepared = prepare_torch_model(args.weights, args.imgsz)
        print(f"model     {len(prepared.class_names)} classes {prepared.class_names}, stride {prepared.stride}, imgsz {prepared.imgsz}")
        print(f"graph     dynamic axes {DYNAMIC_AXES}, static {prepared.imgsz}x{prepared.imgsz} spatial, opset {args.opset}")
        info = export_graph(prepared, out_file, args.opset, args.simplify, source_sha)
        graph = describe_graph(out_file)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(f"error: {exc.code}", file=sys.stderr)
            return EXIT_USAGE
        raise
    imgsz, ultralytics_version = prepared.imgsz, prepared.ultralytics_version
    del prepared

    onnx_sha = C.sha256_file(out_file)
    size = out_file.stat().st_size
    print(f"onnx      {out_file}  size {size} bytes ({size / 1e6:.1f} MB)  sha256 {onnx_sha}")
    print(f"          inputs {graph['input_shapes']} outputs {graph['output_shapes']}  simplified={info['simplified']}  ir_version={info['ir_version']}")

    kind = f"real images from {args.parity_images}" if args.parity_images else f"a fixed-seed synthetic tensor (seed {args.parity_seed})"
    print(f"parity    torch (fp32, cpu, eval, fused) vs onnxruntime CPUExecutionProvider on {kind}")
    try:
        parity = run_parity(out_file, args.weights, args.imgsz, args.parity_tol, PARITY_BATCHES,
                            args.parity_images, args.parity_seed, args.unseal_test, args.box_units)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(f"error: {exc.code}", file=sys.stderr)
            return EXIT_USAGE
        raise
    if args.box_units == "pixels":
        print("          box rows are pixels (0-640) where one float32 step at the top of the range is 6.1e-05; score rows are 0-1")
    else:
        print(f"          box rows are compared as fractions of the input size (pixels / {imgsz}); score rows are 0-1")
        print(f"          the same box difference in input pixels: {parity['max_abs_diff_boxes'] * imgsz:.3e} "
              f"(one float32 step at 640 is 6.1e-05; run with --box-units pixels to apply the tolerance to pixels)")
    if parity["input_sensitive"] is False:
        print(
            f"WEAK CHECK: the torch outputs for different inputs differ by at most {parity['input_sensitivity']:.3e}, "
            f"under {SENSITIVITY_FACTOR} x the tolerance, so this input cannot tell a correct graph from one that "
            "ignores its input. Expected for random-initialised weights; repeat on trained weights, ideally with --parity-images."
        )
    print(parity_line(parity["max_abs_diff"], parity["tolerance"], parity["passed"]))

    record = {
        "schema": SCHEMA,
        "tag": tag,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": C.host_info(),
        "source_weights": {"path": display_path(args.weights), "sha256": source_sha},
        "onnx": {
            "file": out_file.name,
            "sha256": onnx_sha,
            "size_bytes": size,
            "opset": args.opset,
            "imgsz": imgsz,
            "dynamic_axes": {name: {str(k): v for k, v in axes.items()} for name, axes in DYNAMIC_AXES.items()},
            "input_names": graph["input_names"],
            "output_names": graph["output_names"],
            "input_shapes": graph["input_shapes"],
            "output_shapes": graph["output_shapes"],
            "simplified": info["simplified"],
            "ir_version": info["ir_version"],
            "onnx_version": info["onnx_version"],
        },
        "tools": {"ultralytics": ultralytics_version, "onnx": info["onnx_version"], "onnxslim": _onnxslim_version() if info["simplified"] else None},
        "class_names": json.loads(read_onnx_header(out_file)["metadata"]["class_names"]),  # what the file itself says
        "parity": parity,
        "hf": None,
    }
    results_json = args.results_dir / f"export_{tag}.json"

    if not parity["passed"]:
        C.write_json(results_json, record)
        print(f"wrote {results_json}")
        exact = parity["float64_reference"]
        if exact and exact["graph_as_accurate_as_torch"]:
            advice = (
                "The float64 reference says the graph is as accurate as torch itself, so the tolerance is below the float32 "
                "noise floor of this model. If you accept that, choose it on purpose (--parity-tol, or --box-units normalized "
                "to compare boxes as fractions of the input) and record why."
            )
        else:
            advice = "Do not use this file. Re-run with --no-simplify to rule out the simplifier, and keep the JSON as evidence."
        print(f"FAILED: {out_file} does not match the torch model within {args.parity_tol:.0e}. {advice}", file=sys.stderr)
        return EXIT_PARITY_FAIL

    C.write_json(results_json, record)
    print(f"wrote {results_json}")

    if args.push_to_hub:
        card = render_model_card(record["class_names"], imgsz, args.opset, tag, source_sha, onnx_sha)
        token = os.environ["HF_TOKEN"].strip()
        try:
            record["hf"] = push_to_hub(out_file, args.push_to_hub, args.private, card, token)
        except Exception as exc:
            print(f"error: upload to {args.push_to_hub} failed: {str(exc).replace(token, '***')}", file=sys.stderr)
            return EXIT_UPLOAD_FAIL
        C.write_json(results_json, record)
        print(f"uploaded  {record['hf']['url']}  (updated {results_json})")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
