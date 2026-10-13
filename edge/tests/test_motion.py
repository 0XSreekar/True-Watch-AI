"""The motion channel reproduces the MEASUREMENTS.md section 1 measurement and its direction.

Section 1: dense Farneback flow inside object boxes versus background, visible 5.17 / 2.74 px (1.9x),
LWIR 2.73 / 0.81 px (3.4x); whole-frame flow on LWIR about 32% of visible. The synthetic pair below
models why: the visible background is textured, so any camera vibration shows up as background flow,
while the LWIR background is thermally flat and gives the flow nothing to lock on to; the warm body is
the one strong edge in the LWIR frame. The test asserts the same ordering the measurement found.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.motion import (FARNEBACK_PARAMS, MotionChannel, aggregate_contrast, farneback, flow_magnitude,
                             object_background_contrast, remove_global_shift)
from tests.synth import H, W, texture


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def synthetic_pair(modality: str, k: int, rng: np.random.Generator):
    """Frame k and k+1 of a scene with a 1 px camera vibration and one object moving right."""
    if modality == "visible":
        bg = texture(H, W, 11, amp=55, base=110, blur=2)                   # textured street
        obj = texture(60, 26, 12, amp=60, base=120, blur=1)                 # textured clothing
        speed, noise = 4.0, 3.0
    else:
        bg = cv2.GaussianBlur(np.full((H, W), 60, np.float32) + texture(H, W, 13, amp=2, base=0, blur=25), (0, 0), 3)
        yy, xx = np.mgrid[0:60, 0:26]
        obj = (170 + 30 * np.exp(-(((xx - 13) / 9.0) ** 2 + ((yy - 30) / 22.0) ** 2))).astype(np.float32)  # warm body
        speed, noise = 2.5, 1.0                                             # thermal: slower apparent motion
    frames, boxes = [], []
    for j in (k, k + 1):
        jitter = (0.8 * np.sin(1.7 * j), 0.8 * np.cos(1.3 * j))              # the pole vibrates
        img = _shift(bg, *jitter)
        x, y = int(40 + speed * j), 100
        img[y:y + 60, x:x + 26] = obj
        img = img + rng.normal(0, noise, img.shape)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
        boxes.append((x, y, x + 26, y + 60))
    return frames, boxes[1]


def measure(modality: str, pairs: int = 12) -> tuple[dict, float]:
    rng = np.random.default_rng(5)
    samples, whole = [], []
    for k in range(pairs):
        (a, b), box = synthetic_pair(modality, k, rng)
        mag = flow_magnitude(farneback(a, b))          # raw flow, as section 1 measured it
        s = object_background_contrast(mag, [box])
        assert s is not None
        samples.append(s)
        whole.append(float(mag.mean()))
    return aggregate_contrast(samples), float(np.mean(whole))


def test_parameters_are_pinned():
    assert FARNEBACK_PARAMS == dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def test_contrast_measurement_matches_the_section_1_definition():
    mag = np.full((100, 100), 2.0, np.float32)
    mag[10:30, 10:30] = 6.0
    obj, bg, c = object_background_contrast(mag, [(10, 10, 30, 30)])
    assert obj == 6.0 and bg == 2.0 and c == 3.0
    agg = aggregate_contrast([(5.17, 2.74, 5.17 / 2.74), (5.17, 2.74, 5.17 / 2.74)])
    assert round(agg["contrast"], 1) == 1.9
    agg = aggregate_contrast([(2.73, 0.81, 2.73 / 0.81)])
    assert round(agg["contrast"], 1) == 3.4


def test_ir_contrast_exceeds_visible_on_a_synthetic_pair():
    vis, vis_whole = measure("visible")
    ir, ir_whole = measure("lwir")
    assert vis["contrast"] > 1.0 and ir["contrast"] > 1.0
    assert ir["contrast"] > vis["contrast"], (vis, ir)
    # and, as section 1 also found, absolute flow is LOWER on LWIR: the rise is in contrast, not in flow
    assert ir_whole < vis_whole
    assert ir["background_px"] < vis["background_px"]


def test_global_shift_is_removed_and_shake_does_not_look_like_an_object():
    bg = np.clip(texture(H, W, 21, amp=50, base=110, blur=2), 0, 255).astype(np.uint8)
    a, b = bg, _shift(bg, 6.0, -4.0)                                        # the whole image jumps: shake
    flow = farneback(a, b)
    comp, (dx, dy) = remove_global_shift(flow)
    assert abs(dx - 6.0) < 1.0 and abs(dy + 4.0) < 1.0
    assert float(flow_magnitude(comp)[20:-20, 20:-20].mean()) < 0.25 * float(flow_magnitude(flow)[20:-20, 20:-20].mean())


def test_score_box_separates_a_mover_from_a_static_object():
    from tests.synth import Scene
    scene = Scene()
    mc = MotionChannel()
    mc.update(scene.frame(0), 0.0)
    field = mc.update(scene.frame(1), 0.05)
    mover, iou_m = MotionChannel.score_box(field, scene.obj_box(1), [scene.parked_box()])
    parked, iou_p = MotionChannel.score_box(field, scene.parked_box(), [scene.obj_box(1)])
    assert mover.score > 0.5 and iou_m > 0.2
    assert parked.score == 0.0
    assert mover.raw > parked.raw
