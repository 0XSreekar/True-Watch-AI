"""Synthetic scenes shared by the pipeline tests. Deterministic; no dataset needed."""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.types import Track

W, H = 320, 256


def texture(h: int, w: int, seed: int, amp: float = 60.0, base: float = 110.0, blur: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.normal(0, 1, (h, w)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), blur)
    t = (t - t.mean()) / (t.std() + 1e-6)
    return np.clip(base + amp * t, 0, 255).astype(np.float32)


class Scene:
    """A static textured background, an optional parked (static) textured object, and a moving object."""

    def __init__(self, seed: int = 1, obj_size: tuple[int, int] = (24, 48), speed: float = 3.0,
                 start: tuple[float, float] = (40.0, 110.0), parked_at: tuple[int, int] | None = (230, 60)) -> None:
        self.bg = texture(H, W, seed)
        self.obj = texture(obj_size[1], obj_size[0], seed + 1, amp=70, base=160, blur=2)
        self.parked = texture(40, 40, seed + 2, amp=70, base=60, blur=2)
        self.parked_at = parked_at
        self.speed = speed
        self.start = start
        self.size = obj_size
        self.rng = np.random.default_rng(seed + 3)

    def obj_box(self, k: int, moving_until: int | None = None) -> tuple[float, float, float, float]:
        kk = k if moving_until is None else min(k, moving_until)
        x = self.start[0] + self.speed * kk
        y = self.start[1]
        return (x, y, x + self.size[0], y + self.size[1])

    def parked_box(self) -> tuple[float, float, float, float]:
        x, y = self.parked_at
        return (float(x), float(y), float(x + 40), float(y + 40))

    def frame(self, k: int, moving_until: int | None = None, with_object: bool = True) -> np.ndarray:
        img = self.bg.copy()
        if self.parked_at is not None:
            x, y = self.parked_at
            img[y:y + 40, x:x + 40] = self.parked
        if with_object:
            x1, y1, x2, y2 = (int(round(v)) for v in self.obj_box(k, moving_until))
            img[y1:y2, x1:x2] = self.obj[: y2 - y1, : x2 - x1]
        img = img + self.rng.normal(0, 1.5, img.shape).astype(np.float32)
        g = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.merge([g, g, g])


def make_track(track_id: int, bbox, class_name: str = "person", frame: int = 0, t: float = 0.0,
               modality: str = "visible") -> Track:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return Track(track_id=track_id, class_name=class_name, bbox=tuple(float(v) for v in bbox), score=0.9,
                 camera_id="TEST", modality=modality, state="confirmed", first_frame=frame, last_frame=frame,
                 first_seen=t, last_seen=t, frames=1, dwell_s=0.05, net_px=0.0, path_px=0.0, velocity=(0.0, 0.0),
                 history=((t, cx, float(bbox[3])),), observations=((frame, t, cx, cy),))
