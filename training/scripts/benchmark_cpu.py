"""Honest CPU latency for an exported detector, using only numpy and onnxruntime.

WHY this file is dependency-free. The number the project needs is the latency a plain CPU gives
the ONNX graph, and the second place it has to be measured is a free Hugging Face Space, which
has neither torch nor Ultralytics. Nothing here imports either (tests/test_export_benchmark.py
proves it by blocking both at import time), and the pure-numpy letterbox, decode and
class-aware NMS mean the timed path does not lean on OpenCV or Pillow either. The image readers
below are optional and used only when --image is given.

WHAT is measured, per run, batch 1, with time.perf_counter after --warmup discarded runs:

  inference_ms  onnxruntime session.run on a prepared tensor. Nothing else.
  pipeline_ms   the detector stage as the edge would run it: letterbox a source frame to a
                square input, run, decode the raw (1, 4+nc, anchors) tensor, confidence filter,
                class-aware NMS, map boxes back to the frame. Detector stage only. Capture,
                decode of the video stream, tracking and fusion are not in it, so it is NOT an
                end-to-end latency.

WHICH weights. inference_ms follows the graph and the host, so a randomly initialised export of the
same architecture times the same. pipeline_ms does not: NMS cost grows with the candidate boxes,
and a random head scores nearly every anchor alike, so it hands NMS thousands of boxes a trained
detector never produces and inflates the figure. The record therefore carries `weights_kind`
(trained / random / unknown; "random" is also inferred from --weights-provenance) and
`pipeline_representative`, true only for trained weights on a real --image. Measure the trained
export with `--onnx training/weights/<tag>.onnx --weights-kind trained --image <frame>`.
make_metrics.py shows only inference_ms for a random-weight record.

WHAT the number is not. It is FP32 on whatever CPU ran it. It is not INT8 and it is not the
Jetson Orin Nano Super. The ~30 ms TARGET on slide 3 is a design budget for INT8 inference at
640 px on that board, hardware this project does not own; it is not comparable to anything this
script prints. The caveat is written into the JSON and the printout so a copied number cannot
lose it.

Host honesty. A container reports the host's core count, not its quota, so a 2-vCPU Space would
look like a 32-core box. The host record carries the cgroup CPU limit and `effective_cpus`, and a
warning is printed when onnxruntime's default pool would be larger than the quota.

The run ends with the full JSON between two marker lines so a Space's log can be copied out.

Exit codes: 0 done, 2 bad input (missing model, bad label, not an FP32 graph, shape mismatch).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, NamedTuple, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np

try:
    import _common as C
except ModuleNotFoundError as exc:  # a Space that received this one file only
    raise SystemExit(
        "benchmark_cpu.py needs _common.py from the same folder (training/scripts). Copy both files "
        "next to each other; _common.py itself imports nothing beyond the standard library."
    ) from exc

SCHEMA = "truewatch.benchmark.v1"
BEGIN_MARK = "--- BEGIN BENCHMARK JSON ---"
END_MARK = "--- END BENCHMARK JSON ---"

LETTERBOX_PAD = 114            # the grey Ultralytics pads with, so the detector sees what it trained on
DEFAULT_FRAME_HW = (720, 1280)  # a 720p camera frame; the resize cost depends on it, so it is recorded
DEFAULT_CONF = 0.25
DEFAULT_NMS_IOU = 0.45
MAX_DET = 300
PRE_NMS_TOPK = 3000            # candidates handed to NMS, highest score first
FRAME_SEED = 0

CAVEAT = (
    "FP32 on CPU through onnxruntime at batch 1; not INT8 and not the Jetson Orin Nano Super. "
    "The ~30 ms TARGET on slide 3 (YOLO11-s INT8 inference at 640 px on that board, hardware this "
    "project does not own) is a design budget, not a measurement, and is not comparable to any "
    "number in this file. pipeline_ms covers the detector stage only (letterbox, inference, "
    "decode, class-aware NMS in numpy) and is not end-to-end. Latency follows the graph and the "
    "host, not the weights, except that NMS cost grows with the number of candidate boxes; see "
    "pipeline_detail."
)

_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# --------------------------------------------------------------------------------------------
# Image handling in numpy
# --------------------------------------------------------------------------------------------


def read_image_rgb(path: str | Path) -> np.ndarray:
    """uint8 (H, W, 3) RGB from disk, using OpenCV or Pillow if either is importable."""
    try:
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"could not read image {path}")
        return np.ascontiguousarray(bgr[..., ::-1])
    except ImportError:
        pass
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit(
            "reading an image needs OpenCV or Pillow, and neither is installed. Drop --image to "
            "use the synthetic frame, which needs numpy only."
        ) from None
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _axis_weights(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source indices and blend weight per output position, half-pixel centres like cv2.INTER_LINEAR."""
    src = (np.arange(n_out, dtype=np.float32) + 0.5) * (n_in / n_out) - 0.5
    src = np.clip(src, 0.0, n_in - 1)
    lo = np.floor(src).astype(np.intp)
    hi = np.minimum(lo + 1, n_in - 1)
    return lo, hi, (src - lo).astype(np.float32)


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of a uint8 (H, W, 3) image in numpy. No antialiasing, as in OpenCV's default."""
    in_h, in_w = img.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return img
    y0, y1, wy = _axis_weights(in_h, out_h)
    x0, x1, wx = _axis_weights(in_w, out_w)
    top = img[y0].astype(np.float32)
    rows = top + (img[y1].astype(np.float32) - top) * wy[:, None, None]
    left = rows[:, x0]
    out = left + (rows[:, x1] - left) * wx[None, :, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class Letterboxed(NamedTuple):
    tensor: np.ndarray                 # (1, 3, size, size) float32, RGB, 0..1
    scale: float                       # input pixels per source pixel
    pad: tuple[int, int]               # (left, top) in input pixels


def letterbox(img_rgb: np.ndarray, size: int) -> Letterboxed:
    """Scale to fit `size` x `size` keeping aspect, centre on grey padding, return an NCHW tensor.

    Same geometry as Ultralytics' LetterBox for a square target, so a graph fed by this sees the
    input distribution it was trained on.
    """
    h, w = img_rgb.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    dh, dw = (size - new_h) / 2.0, (size - new_w) / 2.0
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    canvas = np.full((size, size, 3), LETTERBOX_PAD, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resize_bilinear(img_rgb, new_h, new_w)
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)[None] / np.float32(255.0)
    return Letterboxed(tensor, float(scale), (left, top))


def synthetic_frame(hw: tuple[int, int] = DEFAULT_FRAME_HW, seed: int = FRAME_SEED) -> np.ndarray:
    """A fixed-seed uint8 RGB frame. Its content does not change how long a convolution takes."""
    return np.random.default_rng(seed).integers(0, 256, size=(hw[0], hw[1], 3), dtype=np.uint8)


# --------------------------------------------------------------------------------------------
# Decode and class-aware NMS in numpy
# --------------------------------------------------------------------------------------------


def _iou_one_to_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_det: int = MAX_DET) -> np.ndarray:
    """Greedy NMS, indices into `boxes` in descending score order."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size and len(keep) < max_det:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[_iou_one_to_many(boxes[best], boxes[rest]) <= iou_thr]
    return np.asarray(keep, dtype=np.intp)


class Detections(NamedTuple):
    boxes: np.ndarray       # (k, 4) xyxy in the coordinates asked for
    scores: np.ndarray      # (k,)
    classes: np.ndarray     # (k,)
    candidates: int         # boxes that passed the confidence filter, before NMS


def decode_and_nms(
    raw: np.ndarray,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_NMS_IOU,
    max_det: int = MAX_DET,
    scale: float = 1.0,
    pad: tuple[int, int] = (0, 0),
    frame_hw: tuple[int, int] | None = None,
) -> Detections:
    """Turn the raw (1, 4+nc, anchors) tensor into detections.

    Rows 0-3 are cx, cy, w, h in input pixels and the rest are class scores already through a
    sigmoid, which is what the exporter's head emits. One class per anchor (the best), then NMS
    per class by shifting each class's boxes apart so different classes never suppress each other.
    """
    pred = raw[0].T                               # (anchors, 4+nc)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(axis=1)
    best = scores_all[np.arange(scores_all.shape[0]), cls]
    keep = np.flatnonzero(best >= conf)
    candidates = int(keep.size)
    if candidates == 0:
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.intp), 0)
    if candidates > PRE_NMS_TOPK:
        keep = keep[np.argpartition(-best[keep], PRE_NMS_TOPK - 1)[:PRE_NMS_TOPK]]

    cx, cy, w, h = (pred[keep, i] for i in range(4))
    xyxy = np.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), axis=1)
    sel_scores, sel_cls = best[keep], cls[keep]
    shifted = xyxy + (sel_cls.astype(xyxy.dtype) * 7680.0)[:, None]  # class-aware: 7680 exceeds any input side
    picked = nms(shifted, sel_scores, iou, max_det)
    out = xyxy[picked].astype(np.float32)
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / scale
    if frame_hw is not None:
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, frame_hw[1])
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, frame_hw[0])
    return Detections(out, sel_scores[picked], sel_cls[picked], candidates)


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------


def summarize(samples_ms: Sequence[float]) -> dict:
    """n, p50, p95, mean, min, max, std of a list of millisecond timings (std is the sample std)."""
    a = np.asarray(samples_ms, dtype=np.float64)
    if a.size == 0:
        raise ValueError("no timing samples")
    return {
        "n": int(a.size),
        "p50": round(float(np.percentile(a, 50)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "mean": round(float(a.mean()), 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
    }


def time_calls(fn: Callable[[], object], runs: int, warmup: int) -> tuple[list[float], list[object]]:
    """Wall-clock each call of `fn` in milliseconds after `warmup` discarded calls.

    The collector is paused while timing so a generation-2 sweep is not billed to one unlucky run.
    Returns the timings and the value each call returned, so the caller can inspect them outside
    the timed region.
    """
    for _ in range(warmup):
        fn()
    timings: list[float] = []
    results: list[object] = []
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(runs):
            t0 = time.perf_counter()
            value = fn()
            timings.append((time.perf_counter() - t0) * 1000.0)
            results.append(value)
    finally:
        if was_enabled:
            gc.enable()
    return timings, results


# --------------------------------------------------------------------------------------------
# Model file facts without the onnx package
# --------------------------------------------------------------------------------------------


def _varint(buf: memoryview, pos: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _fields(buf: memoryview) -> Iterator[tuple[int, int | memoryview]]:
    """(field number, varint value or length-delimited payload) for one protobuf message."""
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
            yield field, value
        elif wire == 2:
            length, pos = _varint(buf, pos)
            yield field, buf[pos : pos + length]
            pos += length
        elif wire == 1:
            pos += 8
        elif wire == 5:
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def read_onnx_header(path: str | Path) -> dict:
    """ir_version, opset per domain and metadata_props of an .onnx file, via a small protobuf walk.

    The `onnx` package is not installed on a free Space, and onnxruntime does not expose the opset,
    so the ModelProto header (fields 1, 8 and 14) is read directly. The graph bytes are skipped.
    """
    data = memoryview(Path(path).read_bytes())
    ir_version, opsets, metadata = None, {}, {}
    for field, value in _fields(data):
        if field == 1 and isinstance(value, int):
            ir_version = value
        elif field == 8 and not isinstance(value, int):
            domain, version = "", None
            for f, v in _fields(value):
                if f == 1:
                    domain = bytes(v).decode("utf-8")
                elif f == 2:
                    version = v
            opsets[domain or "ai.onnx"] = version
        elif field == 14 and not isinstance(value, int):
            key = val = ""
            for f, v in _fields(value):
                if f == 1:
                    key = bytes(v).decode("utf-8")
                elif f == 2:
                    val = bytes(v).decode("utf-8")
            metadata[key] = val
    return {"ir_version": ir_version, "opsets": opsets, "metadata": metadata}


# --------------------------------------------------------------------------------------------
# Host
# --------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _torch_not_imported() -> Iterator[None]:
    """Stop _common.host_info() from importing torch just to read its version.

    host_info() calls __import__('torch') for the version string. Importing torch loads its own
    OpenMP runtime and thread pools into the process being timed, which is the opposite of what a
    dependency-free benchmark wants. Blocking the import records torch as null, which is true: this
    process does not use it. If something already imported torch, it is left alone.
    """
    added = "torch" not in sys.modules
    if added:
        sys.modules["torch"] = None  # type: ignore[assignment]  # makes `import torch` raise ImportError
    try:
        yield
    finally:
        if added:
            sys.modules.pop("torch", None)


def effective_cpus(host: dict) -> float:
    """CPUs this process can really use: the affinity count, capped by any cgroup quota."""
    usable = host.get("cpu_count_usable") or host.get("cpu_count_logical") or 1
    limit = host.get("cgroup_cpu_limit")
    return float(min(usable, limit)) if limit else float(usable)


def collect_host() -> dict:
    with _torch_not_imported():
        host = C.host_info()
    host["effective_cpus"] = effective_cpus(host)
    return host


def thread_warning(host: dict, threads: int) -> str | None:
    """A note when the default thread pool is larger than the CPU quota the container grants."""
    limit = host.get("cgroup_cpu_limit")
    usable = host.get("cpu_count_usable") or 1
    if threads == 0 and limit and usable > math.ceil(limit):
        return (
            f"onnxruntime sizes its default pool from the {usable} visible cores, but this container is "
            f"limited to {limit:g} CPUs. Latency is likely to be worse than a --threads {math.ceil(limit)} run; "
            "report that one for a quota-limited host."
        )
    return None


# --------------------------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------------------------


def open_session(model: Path, threads: int):
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("onnxruntime is not installed. Install it with: python -m pip install onnxruntime numpy") from None
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if threads > 0:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"]), ort


def check_input(session, imgsz: int) -> tuple[str, str]:
    """(input name, precision label) after confirming the graph is FP32 and accepts imgsz x imgsz."""
    spec = session.get_inputs()[0]
    if spec.type != "tensor(float)":
        raise SystemExit(
            f"the model input is {spec.type}; this benchmark is defined for FP32 graphs (input tensor(float)) "
            "so a number cannot be mistaken for another precision. Export FP32 with export_onnx.py."
        )
    shape = spec.shape
    if len(shape) != 4:
        raise SystemExit(f"expected an NCHW input, the model declares {shape}")
    for axis, name in ((2, "height"), (3, "width")):
        dim = shape[axis]
        if isinstance(dim, int) and dim != imgsz:
            raise SystemExit(
                f"the model's input {name} is fixed at {dim} but --imgsz is {imgsz}. "
                f"Re-run with --imgsz {dim}, or re-export at {imgsz}."
            )
    return spec.name, "fp32"


def run_benchmark(
    model: Path,
    label: str,
    imgsz: int = 640,
    runs: int = 200,
    warmup: int = 20,
    threads: int = 0,
    weights_provenance: str = "not stated",
    image: Path | None = None,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_NMS_IOU,
    weights_kind: str | None = None,
) -> dict:
    """Time the graph and return the benchmark record (schema truewatch.benchmark.v1)."""
    kind = infer_weights_kind(weights_kind, weights_provenance)
    session, ort = open_session(model, threads)
    input_name, precision = check_input(session, imgsz)
    header = read_onnx_header(model)
    opset = header["opsets"].get("ai.onnx")

    if image is not None:
        frame, frame_source = read_image_rgb(image), f"file {Path(image).name}"
    else:
        frame, frame_source = synthetic_frame(), f"synthetic uint8 noise, seed {FRAME_SEED}"
    frame_hw = (int(frame.shape[0]), int(frame.shape[1]))

    fixed = letterbox(frame, imgsz)
    feed_fixed = {input_name: fixed.tensor}

    def inference_only():
        return session.run(None, feed_fixed)

    def pipeline():
        lb = letterbox(frame, imgsz)
        raw = session.run(None, {input_name: lb.tensor})[0]
        return decode_and_nms(raw, conf, iou, MAX_DET, lb.scale, lb.pad, frame_hw)

    inference_t, _ = time_calls(inference_only, runs, warmup)
    pipeline_t, dets = time_calls(pipeline, runs, warmup)
    candidates = np.asarray([d.candidates for d in dets], dtype=np.float64)
    detections = np.asarray([len(d.boxes) for d in dets], dtype=np.float64)

    host = collect_host()
    return {
        "schema": SCHEMA,
        "label": label,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "host": host,
        "model": {
            "file": Path(model).name,
            "sha256": C.sha256_file(Path(model)),
            "size_bytes": Path(model).stat().st_size,
            "opset": opset,
            "precision": precision,
            "weights_provenance": weights_provenance,
            "weights_kind": kind,
        },
        "config": {
            "imgsz": imgsz,
            "batch": 1,
            "threads": threads,
            "threads_meaning": "0 = onnxruntime default pool sized from the visible cores, not from a cgroup quota",
            "runs": runs,
            "warmup": warmup,
            "provider": session.get_providers()[0],
            "providers_available": ort.get_available_providers(),
            "onnxruntime": ort.__version__,
            "input_shape": [str(d) if not isinstance(d, int) else d for d in session.get_inputs()[0].shape],
        },
        "inference_ms": summarize(inference_t),
        "pipeline_ms": summarize(pipeline_t),
        "pipeline_detail": {
            "letterbox": "numpy bilinear resize onto 114-grey padding",
            "frame_hw": list(frame_hw),
            "frame_source": frame_source,
            "conf": conf,
            "nms_iou": iou,
            "nms": "class-aware, greedy, numpy",
            "max_det": MAX_DET,
            "pre_nms_topk": PRE_NMS_TOPK,
            "candidates_mean": round(float(candidates.mean()), 2),
            "candidates_max": int(candidates.max()),
            "detections_mean": round(float(detections.mean()), 2),
        },
        "pipeline_representative": bool(kind == "trained" and image is not None),
        "pipeline_note": pipeline_note(kind, image is not None),
        "caveat": CAVEAT,
    }


WEIGHTS_KINDS = ("trained", "random", "unknown")


def infer_weights_kind(explicit: str | None, provenance: str) -> str:
    """The stated kind, else 'random' when the provenance says so, else 'unknown'. Never guesses 'trained'."""
    if explicit:
        if explicit not in WEIGHTS_KINDS:
            raise SystemExit(f"--weights-kind must be one of {', '.join(WEIGHTS_KINDS)}, got {explicit!r}")
        return explicit
    return "random" if "random" in (provenance or "").lower() else "unknown"


def pipeline_note(kind: str, real_image: bool) -> str:
    if kind == "random":
        return ("random weights: every anchor scores near the head bias, so NMS receives thousands of candidates a trained "
                "detector never produces; pipeline_ms is inflated and is not a detector-stage latency. Use inference_ms only.")
    if kind != "trained":
        return "weights of unknown kind: pipeline_ms depends on how many boxes the head proposes; pass --weights-kind."
    if not real_image:
        return "trained weights on a synthetic noise frame: pipeline_ms reflects the boxes proposed on noise; pass --image."
    return "trained weights on a real frame: pipeline_ms is the detector-stage latency for that frame."


# --------------------------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------------------------


def render_table(record: dict) -> str:
    def row(name: str, s: dict) -> str:
        return (
            f"{name:<14}{s['n']:>6}{s['p50']:>10.2f}{s['p95']:>10.2f}{s['mean']:>10.2f}"
            f"{s['min']:>10.2f}{s['max']:>10.2f}{s['std']:>10.2f}"
        )

    m, c, h = record["model"], record["config"], record["host"]
    quota = f"cgroup limit {h['cgroup_cpu_limit']:g} CPUs" if h.get("cgroup_cpu_limit") else "no cgroup CPU limit"
    threads = "default" if c["threads"] == 0 else str(c["threads"])
    d = record["pipeline_detail"]
    lines = [
        f"Benchmark {record['label']}",
        f"Model     {m['file']}  {m['size_bytes'] / 1e6:.1f} MB  opset {m['opset']}  {m['precision']}  sha256 {m['sha256'][:16]}",
        f"Weights   {m['weights_provenance']} (kind: {m.get('weights_kind', 'unknown')})",
        f"Host      {h['cpu']} | {h['platform']}",
        f"CPUs      {h['cpu_count_logical']} logical, {h['cpu_count_usable']} usable, {quota}, effective {h['effective_cpus']:g}",
        f"Runtime   onnxruntime {c['onnxruntime']}, provider {c['provider']}, threads {threads}",
        f"Config    imgsz {c['imgsz']}, batch 1, warmup {c['warmup']}, runs {c['runs']}, "
        f"frame {d['frame_hw'][1]}x{d['frame_hw'][0]} ({d['frame_source']})",
        "",
        f"{'ms':<14}{'n':>6}{'p50':>10}{'p95':>10}{'mean':>10}{'min':>10}{'max':>10}{'std':>10}",
        row("inference_ms", record["inference_ms"]),
        row("pipeline_ms", record["pipeline_ms"]),
        "",
        f"NMS input   {d['candidates_mean']} candidate boxes per frame on average (max {d['candidates_max']}) "
        f"at conf {d['conf']}; pipeline_ms grows with this number.",
        f"PIPELINE    {record.get('pipeline_note', '')}",
        "",
        f"CAVEAT      {record['caveat']}",
    ]
    return "\n".join(lines)


def print_report(record: dict, written: Path | None) -> None:
    print(render_table(record))
    print()
    if written is not None:
        print(f"wrote {written}")
    print(BEGIN_MARK)
    print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False))
    print(END_MARK)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "CPU latency of an ONNX detector at batch 1 with numpy and onnxruntime only. FP32 on this host; "
            "not INT8 and not a Jetson number."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Exit codes: 0 done, 2 bad input.",
    )
    p.add_argument("--model", "--onnx", dest="model", required=True, type=Path,
                   help="the exported .onnx file, e.g. training/weights/<tag>.onnx from export_onnx.py")
    p.add_argument("--label", required=True, help="host label used in the output name, e.g. mac-apple-m5-cpu or hf-space-cpu-basic")
    p.add_argument("--imgsz", type=int, default=640, help="square input side in pixels")
    p.add_argument("--runs", type=int, default=200, help="timed iterations per measurement")
    p.add_argument("--warmup", type=int, default=20, help="untimed iterations before each measurement")
    p.add_argument("--threads", type=int, default=0, help="onnxruntime intra-op threads; 0 keeps the default")
    p.add_argument("--weights-provenance", default="not stated", help="one line on what the weights are, stored in the JSON")
    p.add_argument("--weights-kind", choices=WEIGHTS_KINDS, default=None,
                   help="trained, random or unknown; default: 'random' if --weights-provenance says so, else unknown. "
                        "Only trained weights on a real --image make pipeline_ms representative")
    p.add_argument("--image", type=Path, default=None, help="real frame for the pipeline timing (needs OpenCV or Pillow)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold for the NMS input")
    p.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU, help="IoU threshold for class-aware NMS")
    p.add_argument("--out", type=Path, default=None, help="result JSON path (default training/results/benchmark_<label>.json)")
    p.add_argument("--no-write", action="store_true", help="print only; do not write a JSON file (for a Space)")
    return p


def validate(args: argparse.Namespace) -> None:
    if not _LABEL_RE.match(args.label):
        raise SystemExit(f"--label {args.label!r} must match [A-Za-z0-9][A-Za-z0-9._-]* because it becomes a file name")
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if args.model.suffix.lower() != ".onnx":
        raise SystemExit(f"{args.model} is not an .onnx file")
    if args.runs < 1 or args.warmup < 0 or args.threads < 0 or args.imgsz < 32:
        raise SystemExit("--runs must be >= 1, --warmup and --threads >= 0, --imgsz >= 32")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.nms_iou <= 1.0:
        raise SystemExit("--conf and --nms-iou must lie in [0, 1]")
    if args.image is not None and not args.image.is_file():
        raise SystemExit(f"image not found: {args.image}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate(args)
        record = run_benchmark(
            args.model, args.label, args.imgsz, args.runs, args.warmup, args.threads,
            args.weights_provenance, args.image, args.conf, args.nms_iou, args.weights_kind,
        )
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(f"error: {exc.code}", file=sys.stderr)
            return 2
        raise
    warning = thread_warning(record["host"], args.threads)
    if warning:
        print(f"warning: {warning}", file=sys.stderr)

    written = None
    if not args.no_write:
        written = args.out or (C.RESULTS_DIR / f"benchmark_{args.label}.json")
        C.write_json(written, record)
    print_report(record, written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
