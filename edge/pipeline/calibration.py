"""Per-camera thresholds learnt from an unlabelled warm-up period (slide 3: "fusion against a per-camera
threshold"; this is the "threshold learnt for that camera").

What is learnt, with no labels
------------------------------
For the first `warmup_frames` motion frames of a camera (default 100, five seconds at 20 fps), the
calibrator records the flow magnitude of the BACKGROUND: every grid cell (32 x 32 working-scale px) that
no appearance-channel box covers on that frame. Detector boxes are used only to keep candidate objects
out of the background statistics; nothing is labelled and no human input is involved. From those samples
`learn_camera_thresholds` computes:

  cell_tau     per cell, the MOTION THRESHOLD tau: max(MIN_TAU_PX, median + K_SIGMA robust sigmas (1.4826
               x MAD) of that cell's background energy over the warm-up). A robust statistic, so a person
               the detector missed who crosses a cell for a few frames does not raise its threshold, while
               a cell over swaying foliage, rippling water or a flickering sign learns a high tau, so its routine motion stops reaching the motion mask
               there while the rest of the frame stays sensitive. Cells never seen as background (always
               covered by a box) take the median tau of the frame.
  bg_floor     median background energy over the frame: the denominator floor of the contrast ratio, so
               a static LWIR background near 0 px cannot turn sensor noise into a huge ratio.
  threshold    the fused-score threshold for this camera (FUSION_THRESHOLD_DEFAULT unless set per camera).
  scene_*      median brightness, contrast and detail (Laplacian variance / variance) of the scene: the
               baseline camera_state.py judges lens tampering and IR flooding against.

Storage and reset
-----------------
One JSON file per (camera, modality) under the calibration directory (default edge/var/calibration,
which git ignores): `<camera_id>.<modality>.json`. A camera that switches between a visible day image
and an LWIR night image keeps two calibrations, because the two backgrounds are different distributions.
A stored file is reused only when its working size matches; otherwise the camera re-learns.
`CalibrationStore.reset(camera_id)` deletes the files so the next run re-learns; call it after the camera
is moved, re-aimed or re-focused. While a camera is warming up, fusion raises no DETECTION event for it
(the motion score has no per-camera scale yet); camera-state anomalies are still raised.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from pipeline.types import BBox

log = logging.getLogger("truewatch.edge.calibration")

CALIBRATION_SCHEMA = "truewatch.calibration.v1"
DEFAULT_WARMUP_FRAMES = 100
K_SIGMA = 4.0               # tau = median + K_SIGMA robust sigmas of the cell's background energy
MIN_TAU_PX = 0.35           # never call sub-0.35 px/frame "motion", however quiet the camera
MIN_BG_FLOOR_PX = 0.05


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    modality: str
    working_size: tuple[int, int]         # (w, h) of the motion channel's working frame
    grid: tuple[int, int]                 # (gy, gx) cells
    frames: int                           # warm-up frames the statistics came from
    cell_tau: tuple[tuple[float, ...], ...]
    bg_floor: float
    bg_p95: float
    threshold: float
    scene_brightness: float
    scene_std: float
    scene_detail: float
    created_at: float = field(default_factory=time.time)
    schema: str = CALIBRATION_SCHEMA

    def tau_cells(self) -> np.ndarray:
        return np.asarray(self.cell_tau, dtype=np.float32)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibration":
        return cls(
            camera_id=str(d["camera_id"]), modality=str(d["modality"]),
            working_size=tuple(int(v) for v in d["working_size"]), grid=tuple(int(v) for v in d["grid"]),
            frames=int(d["frames"]), cell_tau=tuple(tuple(float(v) for v in row) for row in d["cell_tau"]),
            bg_floor=float(d["bg_floor"]), bg_p95=float(d["bg_p95"]), threshold=float(d["threshold"]),
            scene_brightness=float(d["scene_brightness"]), scene_std=float(d["scene_std"]),
            scene_detail=float(d["scene_detail"]), created_at=float(d.get("created_at", 0.0)),
            schema=str(d.get("schema", CALIBRATION_SCHEMA)),
        )


def learn_camera_thresholds(
    camera_id: str,
    modality: str,
    working_size: tuple[int, int],
    cell_samples: Sequence[np.ndarray],
    background_masks: Sequence[np.ndarray],
    scene_stats: Sequence[dict],
    threshold: float,
) -> CameraCalibration:
    """Turn unlabelled warm-up samples into one camera's thresholds (see the module docstring).

    cell_samples      per frame, the (gy, gx) mean flow magnitude per cell
    background_masks  per frame, (gy, gx) bool: True where the cell was background on that frame
    scene_stats       per frame, camera_state.frame_stats() output
    """
    if not cell_samples:
        raise ValueError("no warm-up samples")
    stack = np.stack(cell_samples).astype(np.float32)                   # (n, gy, gx)
    bgm = np.stack(background_masks).astype(bool)
    gy, gx = stack.shape[1:]
    tau = np.full((gy, gx), np.nan, dtype=np.float32)
    for y in range(gy):
        for x in range(gx):
            vals = stack[bgm[:, y, x], y, x]
            if len(vals) >= max(5, len(stack) // 10):
                med = float(np.median(vals))
                sigma = 1.4826 * float(np.median(np.abs(vals - med)))
                tau[y, x] = max(MIN_TAU_PX, med + K_SIGMA * sigma)
    if np.isnan(tau).all():
        tau[:] = MIN_TAU_PX
    else:
        tau[np.isnan(tau)] = float(np.nanmedian(tau))
    bg_vals = stack[bgm]
    bg_floor = max(MIN_BG_FLOOR_PX, float(np.median(bg_vals)) if bg_vals.size else MIN_BG_FLOOR_PX)
    bg_p95 = float(np.percentile(bg_vals, 95)) if bg_vals.size else bg_floor

    def med(key: str) -> float:
        vals = [s[key] for s in scene_stats if key in s]
        return float(np.median(vals)) if vals else float("nan")

    return CameraCalibration(
        camera_id=camera_id, modality=modality, working_size=(int(working_size[0]), int(working_size[1])),
        grid=(int(gy), int(gx)), frames=len(stack),
        cell_tau=tuple(tuple(round(float(v), 4) for v in row) for row in tau),
        bg_floor=round(bg_floor, 4), bg_p95=round(bg_p95, 4), threshold=float(threshold),
        scene_brightness=round(med("brightness"), 3), scene_std=round(med("std"), 3),
        scene_detail=round(med("detail"), 5),
    )


class Calibrator:
    """Accumulates one camera's warm-up samples; `finalize()` learns and returns the calibration."""

    def __init__(self, camera_id: str, modality: str, threshold: float,
                 warmup_frames: int = DEFAULT_WARMUP_FRAMES) -> None:
        self.camera_id = camera_id
        self.modality = modality
        self.threshold = threshold
        self.warmup_frames = warmup_frames
        self._cells: list[np.ndarray] = []
        self._bg: list[np.ndarray] = []
        self._scene: list[dict] = []
        self._working_size: tuple[int, int] | None = None
        self._cell_px: int | None = None

    @property
    def count(self) -> int:
        return len(self._cells)

    @property
    def ready(self) -> bool:
        return self.count >= self.warmup_frames

    def observe(self, cell_energy: np.ndarray, working_size: tuple[int, int], cell_px: int,
                exclude_boxes_working: Sequence[BBox], scene: dict | None = None) -> None:
        """One motion frame. `exclude_boxes_working` are candidate boxes at WORKING scale."""
        if self._working_size is not None and self._working_size != tuple(working_size):
            self._cells.clear(); self._bg.clear(); self._scene.clear()
        self._working_size = tuple(working_size)
        self._cell_px = cell_px
        gy, gx = cell_energy.shape
        bg = np.ones((gy, gx), dtype=bool)
        for x1, y1, x2, y2 in exclude_boxes_working:
            cx1, cy1 = max(0, int(x1 // cell_px)), max(0, int(y1 // cell_px))
            cx2, cy2 = min(gx, int(np.ceil(x2 / cell_px))), min(gy, int(np.ceil(y2 / cell_px)))
            bg[cy1:cy2, cx1:cx2] = False
        self._cells.append(cell_energy.astype(np.float32).copy())
        self._bg.append(bg)
        if scene:
            self._scene.append(scene)

    def finalize(self) -> CameraCalibration:
        assert self._working_size is not None
        return learn_camera_thresholds(self.camera_id, self.modality, self._working_size, self._cells,
                                       self._bg, self._scene, self.threshold)


def _safe_name(camera_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", camera_id)


class CalibrationStore:
    """JSON files, one per (camera, modality). Writes are atomic (temp file + rename)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, camera_id: str, modality: str) -> Path:
        return self.root / f"{_safe_name(camera_id)}.{modality}.json"

    def load(self, camera_id: str, modality: str, working_size: tuple[int, int] | None = None) -> CameraCalibration | None:
        p = self.path(camera_id, modality)
        if not p.is_file():
            return None
        try:
            cal = CameraCalibration.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("calibration %s unreadable (%s); the camera will re-learn", p, exc)
            return None
        if cal.schema != CALIBRATION_SCHEMA:
            return None
        if working_size is not None and tuple(cal.working_size) != tuple(working_size):
            log.info("calibration %s was learnt at %s, frames are now %s; re-learning", p.name, cal.working_size, working_size)
            return None
        return cal

    def save(self, cal: CameraCalibration) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.path(cal.camera_id, cal.modality)
        fd, tmp = tempfile.mkstemp(prefix=".cal-", suffix=".json", dir=self.root)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(cal.to_json())
        os.replace(tmp, target)
        return target

    def reset(self, camera_id: str) -> list[Path]:
        """Delete every stored calibration of `camera_id`; returns the paths removed."""
        removed = []
        if self.root.is_dir():
            for p in self.root.glob(f"{_safe_name(camera_id)}.*.json"):
                p.unlink(missing_ok=True)
                removed.append(p)
        return removed
