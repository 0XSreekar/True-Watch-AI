"""Shared plumbing for the two field-measurement tools (measure_false_alerts, measure_fence_misses).

Both run a recording through the SAME path the service runs in production: pipeline.ingest.frames (file
decode, decimation to TARGET_FPS on the file's own clock) into pipeline.service.CameraService (camera
state, motion, appearance, tracking, calibration, fusion, rules), and count what comes out. Nothing is
paced: a file is processed as fast as the machine can, and every time below is MEDIA time (seconds into
the recording), which is also the pipeline's own clock.

Also here: the scripted-footage generator the tools' --self-check modes and the tests use, and the two
interval estimators the reports quote (Wilson score for a proportion, exact Poisson for a count).
"""

from __future__ import annotations

import logging
import math
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

EDGE_ROOT = Path(__file__).resolve().parents[1]
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

from pipeline.events import EventSink  # noqa: E402
from pipeline.ingest import frames  # noqa: E402
from pipeline.service import CameraService  # noqa: E402
from pipeline.types import Detection  # noqa: E402
from rules.config import CameraRuleConfig  # noqa: E402
from rules.engine import RULE_FENCE_CROSSED  # noqa: E402

Z95 = 1.959963984540054


def quiet_reason_logs() -> None:
    """The reason text is not what these tools measure; its per-event warnings would bury the report.
    Called from the command-line entry points only, never on import."""
    logging.getLogger("truewatch.vlm").setLevel(logging.ERROR)


# ------------------------------------------------------------------------------------------ statistics


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (95% by default). (0, 1) when n == 0."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def poisson_interval(k: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Garwood) two-sided interval for a Poisson count k."""
    from scipy.stats import chi2

    lo = 0.0 if k == 0 else float(chi2.ppf(alpha / 2, 2 * k) / 2)
    hi = float(chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2)
    return lo, hi


# ------------------------------------------------------------------------------------------ running


@dataclass
class RunResult:
    recording: str
    camera_id: str
    frames: int = 0
    media_s: float = 0.0                      # media time covered (last frame time + one frame interval)
    warmup_s: float | None = None             # media time at which the camera's calibration was ready
    events: list[dict] = field(default_factory=list)          # {"t": media s, "kind": ..., "track_id": ...}
    fence_crossings: list[dict] = field(default_factory=list)  # {"t": media s, "track_id", "direction"}
    fused_events: int = 0
    rule_suppressed: int = 0

    @property
    def alertable_s(self) -> float:
        """Media time during which an alert could be raised (after the calibration warm-up)."""
        if self.warmup_s is None:
            return 0.0
        return max(0.0, self.media_s - self.warmup_s)


def _kind(event: dict) -> str:
    if (event.get("rule") or {}).get("fired"):
        return "rule"
    return "anpr" if event.get("anpr") else "detection"


def run_recording(
    path: str | Path,
    appearance,
    *,
    camera_id: str,
    target_fps: float,
    stride: int = 2,
    rules: CameraRuleConfig | None = None,
    warmup_seconds: float | None = None,
    calibration_dir: Path | None = None,
    max_seconds: float | None = None,
) -> RunResult:
    """One recording through ingest + CameraService. A fresh calibration is learnt for each run unless
    `calibration_dir` already holds one for `camera_id`."""
    res = RunResult(recording=str(path), camera_id=camera_id)
    with tempfile.TemporaryDirectory() as tmp:
        cal_dir = Path(calibration_dir) if calibration_dir is not None else Path(tmp) / "cal"
        svc = CameraService(appearance, camera_id=camera_id, post_id="MEASURE", sink=EventSink(maxlen=10),
                            stride=stride, calibration_dir=cal_dir, warmup_seconds=warmup_seconds, rules=rules,
                            source="file")
        last_t, dt = 0.0, 1.0 / target_fps
        try:
            for frame in frames(path, is_rtsp=False, camera_id=camera_id, target_fps=target_fps):
                t = frame.media_time if frame.media_time is not None else frame.index / target_fps
                if max_seconds is not None and t > max_seconds:
                    break
                if res.frames:
                    dt = max(1e-6, t - last_t)
                last_t = t
                result, events = svc.process(frame)
                res.frames += 1
                if res.warmup_s is None and svc.pipeline.calibration is not None:
                    res.warmup_s = t
                for ev in events:
                    res.events.append({"t": round(t, 3), "kind": _kind(ev),
                                       "track_id": (ev.get("detection") or {}).get("track_id"),
                                       "class": (ev.get("detection") or {}).get("class"),
                                       "bbox_norm_xywh": (ev.get("detection") or {}).get("bbox"),
                                       "rules": list(((ev.get("rule") or {}).get("fired") or []))})
            offset = svc._epoch_offset or 0.0
            for track_id, fired in svc.fired_rules.items():
                for rule in fired:
                    if rule.rule_name == RULE_FENCE_CROSSED:
                        res.fence_crossings.append({"t": round(float(rule.evidence["crossed_at_s"]) - offset, 3),
                                                    "track_id": track_id,
                                                    "direction": rule.evidence.get("direction"),
                                                    "point_px": rule.evidence.get("point_px")})
            res.fused_events = svc.stats.fused_events
            res.rule_suppressed = svc.stats.rule_suppressed
        finally:
            svc.close()
        res.media_s = round(last_t + dt, 3) if res.frames else 0.0
    res.fence_crossings.sort(key=lambda c: c["t"])
    return res


def real_appearance():
    """The production appearance channel: the detector the service would load (YOLO_MODEL_URL/SHA256 or the
    committed manifest), wrapped exactly as app.build_service wraps it."""
    from config import load
    from models.detector_weights import load_at_boot
    from pipeline.appearance import AppearanceChannel

    det = load_at_boot(load())
    if det is None:
        raise SystemExit("no detector configured: set YOLO_MODEL_URL/YOLO_MODEL_SHA256 or the manifest")
    return AppearanceChannel.from_detector(det)


# ------------------------------------------------------------------------------------------ scripted footage


class ScriptedAppearance:
    """An appearance channel that returns the boxes the footage script drew (an oracle detector).

    Used only to prove the measurement harness on generated footage, where no trained detector would
    recognise a textured rectangle as a person. `boxes_at(t)` -> [(bbox, score, class_name)].
    """

    def __init__(self, boxes_at: Callable[[float], Iterable[tuple]], fps: float) -> None:
        self.boxes_at, self.fps, self.last_ms, self.calls = boxes_at, fps, 0.0, 0

    def detect(self, image, *, camera_id, frame_index, timestamp, modality):
        self.calls += 1
        return [Detection(bbox=tuple(float(v) for v in b), class_name=c, score=float(s), camera_id=camera_id,
                          frame_index=frame_index, timestamp=timestamp, modality=modality, class_id=0)
                for b, s, c in self.boxes_at(timestamp)]


def _texture(h: int, w: int, seed: int, amp: float, base: float, blur: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), blur)
    t = (t - t.mean()) / (t.std() + 1e-6)
    return np.clip(base + amp * t, 0, 255).astype(np.float32)


@dataclass
class Script:
    """A generated recording and what is in it."""

    path: Path
    fps: float
    size: tuple[int, int]
    seconds: float
    crossings: list[dict]                       # ground truth: {"t": media s, "direction", "blind"}
    boxes: dict[int, list[tuple]]               # source frame index -> [(bbox, score, class)]
    fence: dict | None = None                   # rules config dict for the camera

    def boxes_at(self, t: float) -> list[tuple]:
        return self.boxes.get(int(round(t * self.fps)), [])

    def appearance(self) -> ScriptedAppearance:
        return ScriptedAppearance(self.boxes_at, self.fps)


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not vw.isOpened():
        raise RuntimeError(f"cannot write {path}")
    return vw


def render_crossings(path: Path, n: int, *, fps: float = 20.0, size: tuple[int, int] = (320, 256),
                     speed_px_s: float = 100.0, gap_s: float = 2.5, lead_s: float = 6.0,
                     blind: Iterable[int] = (), seed: int = 7) -> Script:
    """`n` scripted walks across a vertical fence at x = width/2, alternating direction.

    Each walk is one object entering at one side and leaving at the other, then `gap_s` of empty scene (so
    the tracker ends the track and the next walk is a new track). `lead_s` of empty scene first covers the
    calibration warm-up. Walks whose index is in `blind` are drawn but the scripted detector does not
    report them: a crossing the system cannot have seen, so the harness must count it as missed.
    """
    w, h = size
    fx = w / 2.0
    bw, bh = 24, 48
    bg = _texture(h, w, seed, 60, 110, 5)
    obj = _texture(bh, bw, seed + 1, 70, 170, 2)
    rng = np.random.default_rng(seed + 2)
    blind = set(blind)
    x_start, x_end = 12.0, w - 12.0 - bw
    walk_frames = int(math.ceil((x_end - x_start) / speed_px_s * fps))
    gap_frames = int(round(gap_s * fps))
    lead_frames = int(round(lead_s * fps))
    total = lead_frames + n * (walk_frames + gap_frames)
    y = (h - bh) / 2.0
    boxes: dict[int, list[tuple]] = {}
    crossings: list[dict] = []
    vw = _writer(path, fps, size)
    try:
        for k in range(total):
            img = bg.copy()
            j = k - lead_frames
            if j >= 0:
                walk, step = divmod(j, walk_frames + gap_frames)
                if step < walk_frames:
                    frac = step / max(1, walk_frames - 1)
                    rightward = walk % 2 == 0
                    x = x_start + frac * (x_end - x_start) if rightward else x_end - frac * (x_end - x_start)
                    xi, yi = int(round(x)), int(round(y))
                    img[yi:yi + bh, xi:xi + bw] = obj
                    if walk not in blind:
                        boxes[k] = [((float(xi), float(yi), float(xi + bw), float(yi + bh)), 0.8, "person")]
                    if step == 0:
                        # the bottom-centre reaches the fence at this media time
                        cx0 = x_start + bw / 2 if rightward else x_end + bw / 2
                        frames_to_fence = abs(fx - cx0) / (x_end - x_start) * max(1, walk_frames - 1)
                        crossings.append({"t": round((k + frames_to_fence) / fps, 3),
                                          "direction": "right" if rightward else "left",
                                          "blind": walk in blind})
            img = img + rng.normal(0, 1.5, img.shape).astype(np.float32)
            g = np.clip(img, 0, 255).astype(np.uint8)
            vw.write(cv2.merge([g, g, g]))
    finally:
        vw.release()
    fence = {"camera_id": "SYN-FENCE",
             "fences": [{"name": "scripted-line", "points": [[fx, -10000.0], [fx, 10000.0]], "direction": "both"}]}
    return Script(path=path, fps=fps, size=size, seconds=total / fps, crossings=crossings, boxes=boxes, fence=fence)


def render_quiet(path: Path, seconds: float, *, fps: float = 20.0, size: tuple[int, int] = (320, 256),
                 intrusions: int = 0, seed: int = 11) -> Script:
    """Event-free footage: textured ground, sensor noise, a patch of swaying 'foliage' (texture that moves
    back and forth a few px every frame) and a parked object the scripted detector reports on every frame
    at 0.9. Nothing in it should raise an alert.

    `intrusions` > 0 adds that many walks across the frame (scripted detector boxes included), the
    positive control that proves the counter counts.
    """
    w, h = size
    bg = _texture(h, w, seed, 60, 110, 5)
    leaves = _texture(70, 90, seed + 1, 80, 120, 1.5)
    parked = _texture(40, 40, seed + 2, 70, 60, 2)
    obj = _texture(48, 24, seed + 3, 70, 170, 2)
    rng = np.random.default_rng(seed + 4)
    total = int(round(seconds * fps))
    park_box = (230.0, 170.0, 270.0, 210.0)
    boxes: dict[int, list[tuple]] = {}
    walk_frames = int(3.0 * fps)
    starts = [int((i + 1) * total / (intrusions + 1)) for i in range(intrusions)]
    crossings = []
    vw = _writer(path, fps, size)
    try:
        for k in range(total):
            img = bg.copy()
            sway = int(round(3 * math.sin(2 * math.pi * 0.7 * k / fps)))
            img[20:90, 20 + 3:110 - 3] = leaves[:, 3 + sway:87 + sway]
            img[170:210, 230:270] = parked
            boxes[k] = [(park_box, 0.9, "car")]
            for s in starts:
                if s <= k < s + walk_frames:
                    x = 10 + (k - s) * (w - 44) / walk_frames
                    xi, yi = int(round(x)), 110
                    img[yi:yi + 48, xi:xi + 24] = obj
                    boxes[k].append(((float(xi), float(yi), float(xi + 24), float(yi + 48)), 0.8, "person"))
                    if k == s:
                        crossings.append({"t": round(s / fps, 3), "direction": "right", "blind": False})
            img = img + rng.normal(0, 1.5, img.shape).astype(np.float32)
            g = np.clip(img, 0, 255).astype(np.uint8)
            vw.write(cv2.merge([g, g, g]))
    finally:
        vw.release()
    return Script(path=path, fps=fps, size=size, seconds=total / fps, crossings=crossings, boxes=boxes)
