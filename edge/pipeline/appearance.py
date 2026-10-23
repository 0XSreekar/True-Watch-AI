"""The appearance channel: YOLO11-s exported to ONNX, run with onnxruntime on CPU.

No Ultralytics code is imported here or anywhere under edge/. The graph is the one
training/scripts/export_onnx.py writes (or, until the fine-tuned model is published, a COCO-pretrained
YOLO11-s exported the same way); it is resolved, downloaded and SHA-256-verified by
models/detector_weights.py, and this module only feeds it.

Graph contract (both exports)
-----------------------------
input   float32 (batch, 3, H, W), RGB, 0..1, letterboxed with grey (114) padding, stride 32
output  float32 (batch, 4 + nc, anchors): rows 0-3 are cx, cy, w, h in input pixels, rows 4.. are
        per-class scores (already sigmoid). No NMS in the graph; NMS is done here.

Class map, by NAME, never by index
----------------------------------
The pipeline speaks the schema of datasets/config/schema.yaml. The model's own names come from the
graph metadata ('class_names' JSON written by export_onnx.py, else the 'names' dict every YOLO export
carries), or from the manifest. They are mapped by name:

    COCO person                 -> person
    COCO bicycle, motorcycle    -> two_wheeler
    COCO car                    -> car
    COCO bus, truck             -> truck
    fine-tuned person, two_wheeler, car, truck, cart -> themselves
    everything else             -> dropped (the column is never read)

Class 4 (cart) may be withdrawn by the cart gate (DATASET_SPEC 1.5) and left untrained in the head;
`drop_classes` removes it explicitly, and any id with no mapping is ignored either way. Several model
classes that map to one schema class are merged by taking the maximum score per box, and NMS runs per
SCHEMA class, so a box scored as both 'bus' and 'truck' becomes one 'truck'.

Far-field tiling (SAHI-style), per camera
-----------------------------------------
docs/MEASUREMENTS.md section 3 measured the small-object cliff: recall 100% down to 27 px person height,
90% at 19 px, 49% at 9 px. A 1080p frame letterboxed to 640 shrinks everything by 3x, so a 40 px person
on the horizon arrives at 13 px. `TilingConfig` slices a horizontal band of the ORIGINAL frame (the far
field, by default the top 55%) into overlapping tiles near the model's input size, runs them as one
batch, maps the boxes back and merges them with the full-frame boxes by class-aware NMS. It is off by
default and toggled per camera, because it multiplies detector cost by the number of tiles.
"""

from __future__ import annotations

import ast
import json
import logging
import time as _time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np

from pipeline.tiling import TilingConfig, class_nms, merge_passes, nms, tile_windows
from pipeline.types import SCHEMA_CLASSES, Detection

log = logging.getLogger("truewatch.edge.appearance")

# Model class name -> schema class name. Identity entries cover the fine-tuned head.
NAME_MAP: dict[str, str] = {
    "person": "person",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
    "motorbike": "two_wheeler",
    "car": "car",
    "bus": "truck",
    "truck": "truck",
    "two_wheeler": "two_wheeler",
    "cart": "cart",
}

LETTERBOX_FILL = 114


class AppearanceError(RuntimeError):
    """The detector graph cannot be used as the appearance channel."""


@dataclass(frozen=True)
class AppearanceConfig:
    conf: float = 0.10          # candidate floor. Low on purpose: weak boxes are kept for motion to confirm
    iou: float = 0.5            # NMS IoU within one schema class
    max_det: int = 300
    # The v2 detector (training/results/hf_model.json) was trained with 0 cart instances: the cart gate
    # (DATASET_SPEC 1.5) withdrew class 4 ("0 hand-verified instances < 300" in the dataset's
    # manifests/cart_gate.json) and training/results/METRICS.md lists cart with 0 boxes. The column is
    # still in the 5-output head but untrained, so any score it emits is noise; it is dropped here.
    drop_classes: tuple[str, ...] = ("cart",)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    # Local contrast equalisation (CLAHE) of LWIR frames before the detector. Off by default: the fine-tuned
    # model is trained on LWIR replicated to three channels without it, and its input must match its
    # training. It helps the COCO-pretrained interim model, which never saw thermal imagery (on the
    # set00/V001 demo clip it roughly triples person boxes at conf >= 0.4); enable it only for that model.
    lwir_clahe: bool = False


def model_class_names(session, fallback: Sequence[str] | None = None) -> list[str]:
    """The graph's class names, index-ordered: metadata 'class_names' (JSON list), else 'names' (dict)."""
    meta = session.get_modelmeta().custom_metadata_map or {}
    if meta.get("class_names"):
        try:
            return [str(n) for n in json.loads(meta["class_names"])]
        except ValueError:
            pass
    if meta.get("names"):
        try:
            names = ast.literal_eval(meta["names"])
        except (ValueError, SyntaxError):
            names = None
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
    if fallback:
        return list(fallback)
    raise AppearanceError("the detector graph carries no class names (metadata 'class_names' or 'names') and none were given")


def build_class_map(names: Sequence[str], drop: Iterable[str] = ()) -> tuple[np.ndarray, list[str]]:
    """(column index per model class -> schema index or -1, schema names). Unmapped ids are -1."""
    dropped = set(drop)
    schema = list(SCHEMA_CLASSES)
    out = np.full(len(names), -1, dtype=np.int64)
    for i, n in enumerate(names):
        target = NAME_MAP.get(str(n).strip().lower())
        if target is not None and target not in dropped:
            out[i] = schema.index(target)
    if not (out >= 0).any():
        raise AppearanceError(f"none of the model's classes {list(names)[:12]}... maps to the schema {schema}")
    return out, schema


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad to size x size. Returns (img, scale, (pad_x, pad_y))."""
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else image
    px, py = (size - nw) / 2.0, (size - nh) / 2.0
    top, left = int(round(py - 0.1)), int(round(px - 0.1))
    out = cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT,
                             value=(LETTERBOX_FILL,) * 3)
    return out, r, (float(left), float(top))


def to_blob(images: Sequence[np.ndarray]) -> np.ndarray:
    """BGR uint8 HxWx3 list -> float32 (N, 3, H, W) RGB 0..1."""
    arr = np.stack(images)[..., ::-1]
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


class AppearanceChannel:
    """Run the detector on a frame and return schema-mapped `Detection`s in original-frame pixels."""

    def __init__(self, session, imgsz: int = 640, class_names: Sequence[str] | None = None,
                 config: AppearanceConfig | None = None) -> None:
        self.session = session
        self.imgsz = int(imgsz)
        self.config = config or AppearanceConfig()
        self.input_name = session.get_inputs()[0].name
        shape = session.get_inputs()[0].shape
        self.dynamic_batch = not isinstance(shape[0], int)
        self.model_names = model_class_names(session, class_names)
        self.col_map, self.schema = build_class_map(self.model_names, self.config.drop_classes)
        self.used_cols = np.flatnonzero(self.col_map >= 0)
        self.last_ms: float = 0.0
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @classmethod
    def from_detector(cls, detector, config: AppearanceConfig | None = None) -> "AppearanceChannel":
        """Wrap a models.detector_weights.Detector (what load_at_boot returns)."""
        return cls(detector.session, detector.imgsz, detector.class_names or None, config)

    def describe(self) -> dict:
        mapped = {self.model_names[i]: self.schema[self.col_map[i]] for i in self.used_cols}
        return {"imgsz": self.imgsz, "mapped_classes": mapped, "conf_floor": self.config.conf,
                "tiling": self.config.tiling.enabled, "lwir_clahe": self.config.lwir_clahe}

    # -------------------------------------------------------------------------------------- core

    def _infer(self, crops: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Letterbox + run + decode each crop. Returns per crop (boxes xyxy in crop px, scores, schema ids)."""
        prepared = [letterbox(c, self.imgsz) for c in crops]
        # One crop per call, even when the graph accepts a batch: on CPU onnxruntime a batch of 4 640x640
        # inputs costs about 2.5x the same four calls made one after another (measured on the M-series Mac,
        # training/results/TILING_EVAL.md), so tiles run sequentially.
        preds = [self.session.run(None, {self.input_name: to_blob([p[0]])})[0][0] for p in prepared]
        results = []
        for (_, r, (px, py)), pred in zip(prepared, preds):
            results.append(self._decode(pred, r, px, py))
        return results

    def _decode(self, pred: np.ndarray, r: float, px: float, py: float):
        cls_scores = pred[4:][self.used_cols]                      # (n_used, anchors)
        schema_ids = self.col_map[self.used_cols]
        # merge model classes that map to one schema class by max, per anchor
        n_schema = len(self.schema)
        merged = np.zeros((n_schema, cls_scores.shape[1]), dtype=np.float32)
        for row, sid in zip(cls_scores, schema_ids):
            np.maximum(merged[sid], row, out=merged[sid])
        best = merged.argmax(0)
        score = merged[best, np.arange(merged.shape[1])]
        keep = score >= self.config.conf
        if not keep.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
        cx, cy, w, h = pred[0, keep], pred[1, keep], pred[2, keep], pred[3, keep]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes[:, [0, 2]] -= px
        boxes[:, [1, 3]] -= py
        boxes /= r
        return boxes.astype(np.float32), score[keep].astype(np.float32), best[keep].astype(np.int64)

    def detect(self, image: np.ndarray, *, camera_id: str = "CAM-01", frame_index: int = 0,
               timestamp: float = 0.0, modality: str = "visible") -> list[Detection]:
        """All schema-class detections on one BGR frame, NMS applied, in original-frame pixels."""
        t0 = _time.perf_counter()
        if self.config.lwir_clahe and modality == "lwir":
            g = self._clahe.apply(image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            image = cv2.merge([g, g, g])
        h, w = image.shape[:2]
        crops, origins, windows = [image], [(0, 0)], [None]
        tiling = self.config.tiling
        if tiling.enabled:
            for win in tile_windows(w, h, tiling):
                x1, y1, x2, y2 = win
                crops.append(image[y1:y2, x1:x2])
                origins.append((x1, y1))
                windows.append(win)
        passes = []
        for (boxes, scores, cls), (ox, oy), win in zip(self._infer(crops), origins, windows):
            boxes = boxes.copy()
            if len(boxes):
                boxes[:, [0, 2]] += ox
                boxes[:, [1, 3]] += oy
            passes.append((win, boxes, scores, cls))
        boxes, scores, cls, tiled = merge_passes(passes, w, h, tiling, full_conf=self.config.conf,
                                                 iou=self.config.iou, max_det=self.config.max_det)
        self.last_ms = (_time.perf_counter() - t0) * 1000
        keep = range(len(boxes))
        return [
            Detection(bbox=tuple(float(v) for v in boxes[i]), class_name=self.schema[int(cls[i])],
                      score=float(scores[i]), camera_id=camera_id, frame_index=frame_index,
                      timestamp=timestamp, modality=modality, class_id=int(cls[i]), tiled=bool(tiled[i]))
            for i in keep
        ]


def expected_person_px_after_letterbox(person_px: float, frame_w: int, frame_h: int, imgsz: int) -> float:
    """How tall a person of `person_px` in the original frame is at the detector input (MEASUREMENTS 3)."""
    return person_px * min(imgsz / frame_w, imgsz / frame_h)


def tiles_for(frame_w: int, frame_h: int, cfg: TilingConfig) -> int:
    return len(tile_windows(frame_w, frame_h, cfg)) if cfg.enabled else 0


__all__ = [
    "AppearanceChannel", "AppearanceConfig", "AppearanceError", "TilingConfig", "NAME_MAP",
    "build_class_map", "model_class_names", "letterbox", "nms", "class_nms", "tile_windows",
    "expected_person_px_after_letterbox", "tiles_for",
]
