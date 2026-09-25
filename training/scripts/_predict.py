"""Run a detector over a dataset split and turn the output into `ImageData` for _metrics.py.

This is the only module besides train.py and export_onnx.py that imports Ultralytics, and it does
so lazily, inside functions. Ultralytics is AGPL-3.0 and is confined to training/.

Inference is run ONCE per weights file with Ultralytics' NMS set to IoU 1.0 (which suppresses
nothing) and a low confidence floor. The raw boxes are cached. Non-maximum suppression is then
applied here, at whatever IoU the caller wants, so `sweep_conf.py` can scan NMS IoU without
running the network again, and `evaluate.py` and the sweep agree on what a "detection" is.

Two Ultralytics limits would otherwise cut into the cached curve without saying so:

  * max_det. Ultralytics keeps at most `max_det` boxes per image after its NMS. With NMS off and a
    0.001 floor a crowded frame can propose more candidates than a small cap, and the ones dropped
    are the low-confidence tail that AP integrates over. The raw cap here (RAW_MAX_DET) is above the
    8400 anchors a 640 px YOLO11 head has, so at 640 it never binds; every image that reaches it
    anyway is counted in the cache metadata (`raw_cap_hits`) and reported.
  * the NMS time limit. Ultralytics' non_max_suppression stops after 2 s + 0.05 s per image in the
    batch and returns EMPTY results for every image it had not reached, with only a log warning. On a
    slow CPU with thousands of candidates per image that silently zeroes whole batches. predict_raw
    lifts the limit (NMS_MAX_TIME_IMG) for the duration of the run; the predictor does not expose it.

The post-NMS cap the evaluation applies (`max_det`, default 300) is recorded in the cache metadata
too, so sweep_conf.py reusing the cache applies the same cap evaluate.py did.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from _common import (
    MetaResolver,
    label_path_for,
    object_height_px,
    read_yolo_label,
    sha256_file,
)
from _metrics import ImageData

RAW_MAX_DET = 10000  # boxes kept per image before our own NMS; above the 8400 anchors of a 640 px head
NMS_MAX_TIME_IMG = 3600.0  # seconds per image Ultralytics' NMS may take before it gives up (default 0.05)
DEFAULT_CONF_FLOOR = 0.001  # Ultralytics' own validation floor; AP needs the whole curve
DEFAULT_MAX_DET = 300       # detections kept per image after our own NMS (Ultralytics' default)
PREDICT_CHUNK = 256         # images per predict() call; bounds memory (see predict_raw)


@dataclass
class RawPreds:
    """Pre-NMS predictions for a list of images."""

    paths: list[str]
    hw: np.ndarray                      # (n, 2) original (height, width)
    boxes: list[np.ndarray]             # each (k, 4) xyxy float32, original pixels
    conf: list[np.ndarray]
    cls: list[np.ndarray]
    meta: dict = field(default_factory=dict)


@contextlib.contextmanager
def no_nms_time_limit(max_time_img: float = NMS_MAX_TIME_IMG):
    """Run Ultralytics' predictor with its NMS time limit lifted.

    The predictor calls `ultralytics.utils.nms.non_max_suppression` through the module attribute and
    never passes `max_time_img`, so the default 0.05 s per image applies; past it NMS breaks out of
    its loop and every image not yet processed comes back with no boxes. The attribute is wrapped for
    the duration of the block and restored afterwards, even on error.
    """
    from ultralytics.utils import nms as nms_module  # AGPL-3.0: training/ only

    original = nms_module.non_max_suppression

    def unlimited(*args, **kwargs):
        kwargs.setdefault("max_time_img", max_time_img)
        return original(*args, **kwargs)

    nms_module.non_max_suppression = unlimited
    try:
        yield
    finally:
        nms_module.non_max_suppression = original


def predict_raw(
    weights: str | Path,
    images: Sequence[Path],
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    conf_floor: float = DEFAULT_CONF_FLOOR,
    half: bool = False,
    progress_every: int = 500,
    max_det: int = DEFAULT_MAX_DET,
) -> RawPreds:
    """Run `weights` (.pt or .onnx) over `images`. Boxes come back in original-image pixels.

    `max_det` is not applied here (the raw boxes are cached before any NMS); it is the post-NMS cap
    the caller will apply, recorded in the metadata so every reader of the cache applies the same one.
    """
    from ultralytics import YOLO  # AGPL-3.0: training/ only

    model = YOLO(str(weights), task="detect")
    kwargs = dict(
        imgsz=imgsz,
        conf=conf_floor,
        iou=1.0,
        max_det=RAW_MAX_DET,
        batch=batch,
        verbose=False,
        stream=True,
    )
    if half:  # only passed when wanted: Ultralytics warns on every call that `half` is deprecated
        kwargs["half"] = True
    if device not in (None, ""):
        kwargs["device"] = device

    paths, hw, boxes, conf, cls = [], [], [], [], []
    cap_hits: list[str] = []

    def results():
        # One predict() call over the whole list keeps every decoded frame alive until it returns
        # (about 11 MB per image on this corpus: 12,218 validation images peaked above 60 GB and
        # were killed). Bounded chunks release them; the boxes are identical either way.
        for start in range(0, len(images), PREDICT_CHUNK):
            chunk = [str(p) for p in images[start:start + PREDICT_CHUNK]]
            yield from model.predict(source=chunk, **kwargs)

    with no_nms_time_limit():
        for i, result in enumerate(results()):
            h, w = result.orig_shape
            paths.append(str(result.path))
            hw.append((h, w))
            b = result.boxes
            boxes.append(b.xyxy.cpu().numpy().astype(np.float32))
            conf.append(b.conf.cpu().numpy().astype(np.float32))
            cls.append(b.cls.cpu().numpy().astype(np.int16))
            if len(boxes[-1]) >= RAW_MAX_DET:
                cap_hits.append(Path(str(result.path)).name)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  predicted {i + 1}/{len(images)} images", flush=True)
    if cap_hits:
        print(f"warning: {len(cap_hits)} image(s) reached the raw cap of {RAW_MAX_DET} boxes before NMS, e.g. "
              f"{cap_hits[:3]}; their lowest-confidence candidates were dropped", flush=True)

    try:
        import ultralytics

        version = ultralytics.__version__
    except Exception:  # pragma: no cover
        version = None
    return RawPreds(
        paths=paths,
        hw=np.asarray(hw, dtype=np.int32).reshape(-1, 2),
        boxes=boxes,
        conf=conf,
        cls=cls,
        meta={
            "weights": str(weights),
            "weights_sha256": sha256_file(Path(weights)) if Path(weights).exists() else None,
            "imgsz": imgsz,
            "conf_floor": conf_floor,
            "raw_max_det": RAW_MAX_DET,
            "raw_cap_hits": len(cap_hits),
            "raw_cap_hit_examples": cap_hits[:10],
            "nms_max_time_img": NMS_MAX_TIME_IMG,
            "max_det": int(max_det),
            "ultralytics": version,
        },
    )


def save_raw(path: Path, raw: RawPreds) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.array([len(b) for b in raw.boxes], dtype=np.int64)
    np.savez_compressed(
        path,
        paths=np.array(raw.paths),
        hw=raw.hw,
        counts=counts,
        boxes=np.concatenate(raw.boxes) if len(counts) and counts.sum() else np.zeros((0, 4), np.float32),
        conf=np.concatenate(raw.conf) if len(counts) and counts.sum() else np.zeros(0, np.float32),
        cls=np.concatenate(raw.cls) if len(counts) and counts.sum() else np.zeros(0, np.int16),
        meta=np.array(json.dumps(raw.meta)),
    )


def load_raw(path: Path) -> RawPreds:
    z = np.load(path, allow_pickle=False)
    offsets = np.concatenate(([0], np.cumsum(z["counts"])))
    sl = lambda arr: [arr[offsets[i]:offsets[i + 1]] for i in range(len(z["counts"]))]  # noqa: E731
    return RawPreds(
        paths=[str(p) for p in z["paths"]],
        hw=z["hw"],
        boxes=sl(z["boxes"]),
        conf=sl(z["conf"]),
        cls=sl(z["cls"]),
        meta=json.loads(str(z["meta"])),
    )


# --------------------------------------------------------------------------------------------
# NMS
# --------------------------------------------------------------------------------------------


def nms(boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray, iou_thr: float, max_det: int = DEFAULT_MAX_DET):
    """Class-aware NMS. Returns indices into the inputs, highest confidence first."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    try:
        import torch
        from torchvision.ops import batched_nms

        keep = batched_nms(
            torch.from_numpy(boxes.astype(np.float32)),
            torch.from_numpy(conf.astype(np.float32)),
            torch.from_numpy(cls.astype(np.int64)),
            float(iou_thr),
        ).numpy()
        return keep[:max_det]
    except ImportError:
        return _nms_numpy(boxes, conf, cls, iou_thr)[:max_det]


def _nms_numpy(boxes, conf, cls, iou_thr):
    from _metrics import box_iou

    order = np.argsort(-conf, kind="stable")
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for a_pos in range(len(order)):
        if suppressed[a_pos]:
            continue
        a = order[a_pos]
        keep.append(a)
        rest = order[a_pos + 1:]
        if len(rest) == 0:
            break
        same = cls[rest] == cls[a]
        ious = box_iou(boxes[a:a + 1], boxes[rest])[0]
        suppressed[a_pos + 1:] |= same & (ious > iou_thr)
    return np.asarray(keep, dtype=np.int64)


# --------------------------------------------------------------------------------------------
# Ground truth + predictions -> ImageData
# --------------------------------------------------------------------------------------------


def image_data_from_raw(
    raw: RawPreds,
    resolver: MetaResolver,
    nms_iou: float = 0.7,
    max_det: int = DEFAULT_MAX_DET,
    conf_min: float = 0.0,
    imgsz: int = 640,
    size_basis: str = "input",
) -> tuple[list[ImageData], dict]:
    """Apply NMS, load the labels beside each image, resolve modality. Fails on an unclassified image.

    Returns (images, report) where report counts what was seen so a reader can audit composition.
    """
    out: list[ImageData] = []
    unclassified = []
    composition: dict[str, int] = {}
    unlabelled = 0
    for path, (h, w), b, c, k in zip(raw.paths, raw.hw, raw.boxes, raw.conf, raw.cls):
        p = Path(path)
        meta = resolver.resolve(p.stem)
        if meta is None:
            unclassified.append(p.name)
            continue

        keep_mask = c >= conf_min
        b, c, k = b[keep_mask], c[keep_mask], k[keep_mask]
        idx = nms(b, c, k, nms_iou, max_det)
        pr_box, pr_conf, pr_cls = b[idx].astype(np.float64), c[idx].astype(np.float64), k[idx].astype(np.int64)
        pr_h = np.array(
            [object_height_px((y2 - y1) / h, h, w, imgsz, size_basis) for _, y1, _, y2 in pr_box]
        ) if len(pr_box) else np.zeros(0)

        label_file = label_path_for(p)
        if not label_file.exists():
            unlabelled += 1
        lab = read_yolo_label(label_file)
        gt_cls = lab[:, 0].astype(np.int64)
        cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
        gt_box = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1) if len(lab) else np.zeros((0, 4))
        gt_h = np.array([object_height_px(v, h, w, imgsz, size_basis) for v in lab[:, 4]]) if len(lab) else np.zeros(0)

        out.append(ImageData(meta, gt_cls, gt_box, gt_h, pr_cls, pr_box, pr_conf, pr_h))
        key = f"{meta.slice}/{meta.source}/{meta.lighting}"
        composition[key] = composition.get(key, 0) + 1

    if unclassified:
        raise SystemExit(
            f"{len(unclassified)} image(s) carry neither a '_visible' nor a '_lwir' suffix, e.g. "
            f"{unclassified[:3]}. Refusing to guess a slice; every image must be day or IR."
        )
    if unlabelled:
        print(f"warning: {unlabelled} image(s) had no label file and were scored as background", flush=True)
    return out, {"composition": dict(sorted(composition.items())), "unlabelled_images": unlabelled}


def group_by_slice(images: Sequence[ImageData]) -> dict[str, list[ImageData]]:
    """{'day': [...], 'ir': [...]} plus the visible lighting sub-slices, keyed 'day/daylight' etc.

    There is deliberately no entry holding both modalities.
    """
    groups: dict[str, list[ImageData]] = {"day": [], "ir": []}
    for im in images:
        groups[im.meta.slice].append(im)
        if im.meta.slice == "day":
            groups.setdefault(f"day/{im.meta.lighting}", []).append(im)
    return groups


def cap_note(meta: dict) -> str | None:
    """A sentence for the report when any image reached the raw box cap, else None."""
    hits = int(meta.get("raw_cap_hits") or 0)
    if not hits:
        return None
    return (f"{hits} image(s) reached the raw cap of {meta.get('raw_max_det')} boxes per image before NMS, e.g. "
            f"{list(meta.get('raw_cap_hit_examples') or [])[:3]}; their lowest-confidence candidates were dropped, "
            "which can only lower AP and recall for those images")


def iter_batches(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
