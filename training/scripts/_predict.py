"""Run a detector over a dataset split and turn the output into `ImageData` for _metrics.py.

This is the only module besides train.py and export_onnx.py that imports Ultralytics, and it does
so lazily, inside functions. Ultralytics is AGPL-3.0 and is confined to training/.

Inference is run ONCE per weights file with Ultralytics' NMS set to IoU 1.0 (which suppresses
nothing) and a low confidence floor. The raw boxes are cached. Non-maximum suppression is then
applied here, at whatever IoU the caller wants, so `sweep_conf.py` can scan NMS IoU without
running the network again, and `evaluate.py` and the sweep agree on what a "detection" is.
"""

from __future__ import annotations

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

RAW_MAX_DET = 1000   # boxes kept per image before our own NMS
DEFAULT_CONF_FLOOR = 0.001  # Ultralytics' own validation floor; AP needs the whole curve


@dataclass
class RawPreds:
    """Pre-NMS predictions for a list of images."""

    paths: list[str]
    hw: np.ndarray                      # (n, 2) original (height, width)
    boxes: list[np.ndarray]             # each (k, 4) xyxy float32, original pixels
    conf: list[np.ndarray]
    cls: list[np.ndarray]
    meta: dict = field(default_factory=dict)


def predict_raw(
    weights: str | Path,
    images: Sequence[Path],
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    conf_floor: float = DEFAULT_CONF_FLOOR,
    half: bool = False,
    progress_every: int = 500,
) -> RawPreds:
    """Run `weights` (.pt or .onnx) over `images`. Boxes come back in original-image pixels."""
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
        half=half,
    )
    if device not in (None, ""):
        kwargs["device"] = device

    paths, hw, boxes, conf, cls = [], [], [], [], []
    for i, result in enumerate(model.predict(source=[str(p) for p in images], **kwargs)):
        h, w = result.orig_shape
        paths.append(str(result.path))
        hw.append((h, w))
        b = result.boxes
        boxes.append(b.xyxy.cpu().numpy().astype(np.float32))
        conf.append(b.conf.cpu().numpy().astype(np.float32))
        cls.append(b.cls.cpu().numpy().astype(np.int16))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  predicted {i + 1}/{len(images)} images", flush=True)

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


def nms(boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray, iou_thr: float, max_det: int = 300):
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
    max_det: int = 300,
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


def iter_batches(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
