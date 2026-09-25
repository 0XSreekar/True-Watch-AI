"""Camera-state anomalies (slide 4): signal loss, lens tampering and IR floodlight interference.

A dark or blank frame at a border post is not "nothing happening". Someone who wants a camera blind will
cover it, spray it, shine an IR floodlight into it, or cut it; each of those looks like darkness or
emptiness to a detector. This module raises each as a CAMERA_STATE anomaly (types.CameraStateEvent)
instead of letting it pass as a quiet night, and while one is active it tells fusion to stand down,
because a verdict of "no motion, no person" on a blinded camera is not evidence of anything.

Per-frame statistics (frame_stats) are taken on a 160 px wide grey thumbnail, so they cost about a
millisecond: mean brightness, standard deviation, sharpness (variance of the Laplacian), detail (sharpness
divided by variance, which does not change when the whole image gets darker or brighter), the share of
saturated pixels (>= 250) and the mean absolute difference from the previous thumbnail.

The conditions, each of which must hold for `persist_frames` consecutive frames to raise, and must be
absent for `clear_frames` consecutive frames to clear (hysteresis, so a flicker does not chatter):

SIGNAL_LOSS  - no picture: the frame is digitally flat (std < 1 grey level; a real sensor always shows
               noise), e.g. an all-black frame from a cut feed, a blue "no signal" screen, a decoder
               emitting a flat frame; or
             - a frozen picture: the thumbnail is bit-for-bit static (mean abs diff < 0.02) for
               >= freeze_s seconds (a hung encoder repeating one frame; real sensors always show noise); or
             - no frames at all: the source reported a gap of >= gap_s seconds before this frame
               (source.py reconnect). Raised and cleared on the same frame, with the outage length.
LENS_TAMPER  - a sensor is there (std >= 1) but the scene is gone: detail below TAMPER_DETAIL_RATIO (15%)
               of the camera's learnt scene detail, or contrast below TAMPER_STD_RATIO (8%) of the learnt
               scene std, or both partly gone (detail < 40% and contrast < 35%). A hand over the lens, tape, spray or a knocked focus ring remove the high
               frequencies, and a cover leaves only sensor noise. Darkness does not: detail is a ratio,
               so a scene that simply gets darker (dusk, a lower gain) keeps it, and an LWIR image keeps
               its thermal edges. Without a learnt baseline only an extreme collapse counts
               (detail < 0.005 or std < 3).
IR_FLOOD     - brightness at least 60 levels above the learnt scene brightness, with at least 25% of the
               frame saturated or the mean itself above 220: an IR illuminator aimed at the camera washes
               it out. Without a baseline: mean above 220 and (25% saturated or std below 20). A washed-out
               frame also loses detail, so IR_FLOOD takes precedence over LENS_TAMPER and SIGNAL_LOSS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline.types import CameraStateEvent

THUMB_WIDTH = 160
UNIFORM_STD = 1.0
FROZEN_DIFF = 0.02
TAMPER_DETAIL_RATIO = 0.15
TAMPER_DETAIL_ABS = 0.005
TAMPER_STD_RATIO = 0.08
TAMPER_BOTH = (0.40, 0.35)     # or both partly collapsed: detail < 40% AND contrast < 35% of the learnt scene
TAMPER_STD_ABS = 3.0
FLOOD_SATURATED = 0.25
FLOOD_BRIGHTNESS_DELTA = 60.0
FLOOD_BRIGHTNESS_ABS = 220.0


def thumbnail(image: np.ndarray) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    th = max(8, int(round(h * THUMB_WIDTH / w)))
    return cv2.resize(gray, (THUMB_WIDTH, th), interpolation=cv2.INTER_AREA)


def frame_stats(thumb: np.ndarray, prev_thumb: np.ndarray | None = None) -> dict:
    t = thumb.astype(np.float32)
    stats = {
        "brightness": float(t.mean()),
        "std": float(t.std()),
        "sharpness": float(cv2.Laplacian(thumb, cv2.CV_32F).var()),
        "saturated": float((thumb >= 250).mean()),
    }
    # detail: high-frequency energy relative to overall contrast. Scaling the image (a darker scene, a
    # lower gain) scales both by the same factor, so this is brightness-invariant; blurring (a hand, tape,
    # spray, lost focus) removes the high frequencies and collapses it.
    stats["detail"] = stats["sharpness"] / max(stats["std"] ** 2, 1.0)
    if prev_thumb is not None and prev_thumb.shape == thumb.shape:
        stats["diff"] = float(np.abs(t - prev_thumb.astype(np.float32)).mean())
    return stats


@dataclass
class _Cond:
    on: int = 0
    off: int = 0
    active: bool = False
    since: float = 0.0
    first_seen: float | None = None


@dataclass
class CameraStateMonitor:
    camera_id: str
    source: str = "file"
    baseline: dict | None = None          # {'brightness', 'std', 'detail'} from calibration
    persist_frames: int = 3
    clear_frames: int = 5
    freeze_s: float = 2.0
    gap_s: float = 2.0
    _prev: np.ndarray | None = None
    _frozen_since: float | None = None
    _conds: dict = field(default_factory=lambda: {k: _Cond() for k in ("SIGNAL_LOSS", "LENS_TAMPER", "IR_FLOOD")})
    last_stats: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        """True while any anomaly is active: fusion must not treat this camera's frames as evidence."""
        return any(c.active for c in self._conds.values())

    @property
    def active(self) -> list[str]:
        return [k for k, c in self._conds.items() if c.active]

    def set_baseline(self, brightness: float, std: float, detail: float) -> None:
        if all(np.isfinite([brightness, std, detail])):
            self.baseline = {"brightness": brightness, "std": std, "detail": detail}

    # --------------------------------------------------------------------------------------------

    def _evaluate(self, s: dict, timestamp: float) -> dict[str, tuple[bool, str]]:
        b = self.baseline
        uniform = s["std"] < UNIFORM_STD          # a digitally flat frame: no sensor behind it
        diff = s.get("diff")
        if diff is not None and diff < FROZEN_DIFF and not uniform:
            self._frozen_since = self._frozen_since if self._frozen_since is not None else timestamp
        else:
            self._frozen_since = None
        frozen = self._frozen_since is not None and timestamp - self._frozen_since >= self.freeze_s
        loss = uniform or frozen
        loss_detail = (f"uniform frame (std {s['std']:.1f}, brightness {s['brightness']:.0f})" if uniform
                       else f"picture frozen for {timestamp - (self._frozen_since or timestamp):.1f} s")

        if b:
            d_ratio, s_ratio = s["detail"] / max(b["detail"], 1e-9), s["std"] / max(b["std"], 1e-9)
            tamper = not uniform and (d_ratio < TAMPER_DETAIL_RATIO or s_ratio < TAMPER_STD_RATIO
                                      or (d_ratio < TAMPER_BOTH[0] and s_ratio < TAMPER_BOTH[1]))
            tamper_detail = (f"detail collapsed: {s['detail']:.4f} vs learnt {b['detail']:.4f} "
                             f"(sharpness {s['sharpness']:.1f}, std {s['std']:.1f})")
            flood = (s["brightness"] >= b["brightness"] + FLOOD_BRIGHTNESS_DELTA
                     and (s["saturated"] >= FLOOD_SATURATED or s["brightness"] >= FLOOD_BRIGHTNESS_ABS))
            flood_detail = (f"{s['saturated']:.0%} saturated, brightness {s['brightness']:.0f} vs learnt {b['brightness']:.0f}")
        else:
            tamper = not uniform and (s["detail"] < TAMPER_DETAIL_ABS or s["std"] < TAMPER_STD_ABS)
            tamper_detail = f"detail collapsed: {s['detail']:.4f} (no baseline yet; sharpness {s['sharpness']:.1f}, std {s['std']:.1f})"
            flood = s["brightness"] >= FLOOD_BRIGHTNESS_ABS and (s["saturated"] >= FLOOD_SATURATED or s["std"] < 20.0)
            flood_detail = f"{s['saturated']:.0%} saturated, brightness {s['brightness']:.0f} (no baseline yet)"
        if flood:                      # a washed-out frame is also uniform-ish; report the cause, not the symptom
            loss = False
        return {"SIGNAL_LOSS": (loss, loss_detail), "LENS_TAMPER": (tamper and not loss and not flood, tamper_detail),
                "IR_FLOOD": (flood, flood_detail)}

    def update(self, image: np.ndarray, *, frame_index: int, timestamp: float, captured_at: float,
               gap_s: float = 0.0, reconnects: int = 0) -> list[CameraStateEvent]:
        thumb = thumbnail(image)
        s = frame_stats(thumb, self._prev)
        self._prev = thumb
        self.last_stats = s
        events: list[CameraStateEvent] = []
        metrics = tuple((k, round(v, 4)) for k, v in s.items())

        def emit(kind: str, active: bool, since: float, detail: str) -> None:
            events.append(CameraStateEvent(camera_id=self.camera_id, anomaly=kind, active=active,
                                           frame_index=frame_index, timestamp=timestamp, captured_at=captured_at,
                                           since=since, detail=detail, metrics=metrics, source=self.source))

        if gap_s >= self.gap_s:
            emit("SIGNAL_LOSS", True, timestamp - gap_s,
                 f"no frames for {gap_s:.1f} s; source reconnected ({reconnects} reconnects so far)")
            emit("SIGNAL_LOSS", False, timestamp - gap_s, "frames resumed")

        for kind, (hit, detail) in self._evaluate(s, timestamp).items():
            c = self._conds[kind]
            if hit:
                c.off = 0
                c.on += 1
                if c.first_seen is None:
                    c.first_seen = timestamp
                if not c.active and c.on >= self.persist_frames:
                    c.active, c.since = True, c.first_seen
                    emit(kind, True, c.since, detail)
            else:
                c.on = 0
                c.first_seen = None
                if c.active:
                    c.off += 1
                    if c.off >= self.clear_frames:
                        c.active = False
                        emit(kind, False, c.since, f"cleared after {timestamp - c.since:.1f} s")
        return events
