#!/usr/bin/env python3
"""Degradation pipeline for synthetic plates. DATASET_SPEC.md section 6.3.

Ranges are chosen so the hardest sample is about as bad as the check-post frame in
MEASUREMENTS.md section 4, and no worse: a corpus of illegible plates teaches nothing.

Every stage takes the shared random.Random instance so a run with --seed 42 is
reproducible end to end.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

STAGES = (
    "perspective",
    "scale",
    "motion_blur",
    "defocus",
    "low_light",
    "infrared",
    "rain",
    "dirt",
    "glare",
    "noise",
    "jpeg",
)


def perspective(image: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    yaw = np.deg2rad(rng.uniform(-35, 35))
    pitch = np.deg2rad(rng.uniform(-25, 25))
    roll = np.deg2rad(rng.uniform(-8, 8))

    dx = np.tan(yaw) * width * 0.18
    dy = np.tan(pitch) * height * 0.18
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32(
        [
            [0 + max(0.0, dx), 0 + max(0.0, dy)],
            [width - max(0.0, -dx), 0 + max(0.0, -dy)],
            [width - max(0.0, dx), height - max(0.0, dy)],
            [0 + max(0.0, -dx), height - max(0.0, -dy)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), np.rad2deg(roll), 1.0)
    return cv2.warpAffine(warped, rotation, (width, height), borderMode=cv2.BORDER_REPLICATE)


def scale_to_target(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """A plate at a barrier is 260 px wide; a plate at 40 m is 60 px. Both must be in the corpus."""
    height, width = image.shape[:2]
    target_width = rng.randint(60, 260)
    factor = target_width / float(width)
    small = cv2.resize(
        image, (target_width, max(8, int(height * factor))), interpolation=cv2.INTER_AREA
    )
    return small


def motion_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    # The kernel is proportional to plate width. A 15 px kernel on a 60 px plate is not a
    # hard sample, it is a deleted one, and the spec is explicit that illegible plates
    # teach nothing.
    width = image.shape[1]
    largest = int(max(3, min(15, width // 12)))
    choices = [k for k in (3, 5, 7, 9, 11, 13, 15) if k <= largest] or [3]
    size = rng.choice(choices)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))
    total = kernel.sum()
    if total <= 0:
        return image
    return cv2.filter2D(image, -1, kernel / total)


def defocus(image: np.ndarray, rng: random.Random) -> np.ndarray:
    ceiling = 0.5 + 2.0 * min(1.0, image.shape[1] / 260.0)
    sigma = rng.uniform(0.5, ceiling)
    return cv2.GaussianBlur(image, (0, 0), sigma)


def low_light(image: np.ndarray, rng: random.Random) -> np.ndarray:
    gamma = rng.uniform(1.4, 3.0)
    brightness = rng.uniform(0.25, 0.8)
    table = np.array([((i / 255.0) ** gamma) * 255.0 * brightness for i in range(256)], dtype=np.float32)
    return cv2.LUT(image, np.clip(table, 0, 255).astype(np.uint8))


def infrared(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """IR wash-out, then replicate to three channels - the same invariant as section 3.4."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    span = rng.uniform(0.35, 0.70)
    low = rng.uniform(0.0, 1.0 - span)
    compressed = (gray.astype(np.float32) / 255.0 * span + low) * 255.0
    single = np.clip(compressed, 0, 255).astype(np.uint8)
    return np.repeat(single[:, :, None], 3, axis=2)


def rain(image: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    layer = np.zeros((height, width), dtype=np.uint8)
    for _ in range(rng.randint(40, 200)):
        x = rng.randint(0, max(1, width - 1))
        y = rng.randint(0, max(1, height - 1))
        length = rng.randint(8, 30)
        angle = np.deg2rad(rng.uniform(70, 110))
        x2 = int(x + length * np.cos(angle))
        y2 = int(y + length * np.sin(angle))
        cv2.line(layer, (x, y), (x2, y2), 255, 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    opacity = rng.uniform(0.10, 0.45)
    coloured = cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 1.0, coloured, opacity, 0)


def dirt(image: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    plate_area = height * width
    covered = 0.0
    out = image.copy()
    for _ in range(rng.randint(1, 6)):
        if covered >= 0.25:
            break
        share = rng.uniform(0.02, 0.12)
        radius = int(np.sqrt(share * plate_area / np.pi))
        if radius < 1:
            continue
        centre = (rng.randint(0, max(1, width - 1)), rng.randint(0, max(1, height - 1)))
        colour = (rng.randint(40, 110),) * 3
        overlay = out.copy()
        cv2.circle(overlay, centre, radius, colour, -1)
        out = cv2.addWeighted(overlay, rng.uniform(0.5, 0.9), out, 0.5, 0)
        covered += share
    return out


def glare(image: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    layer = np.zeros((height, width), dtype=np.float32)
    axes = (int(width * rng.uniform(0.12, 0.35)), int(height * rng.uniform(0.12, 0.35)))
    if axes[0] < 1 or axes[1] < 1:
        return image
    centre = (rng.randint(0, max(1, width - 1)), rng.randint(0, max(1, height - 1)))
    cv2.ellipse(layer, centre, axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), max(axes) * 0.4)
    strength = rng.uniform(40, 120)
    return np.clip(image.astype(np.float32) + layer[:, :, None] * strength, 0, 255).astype(np.uint8)


def noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    sigma = rng.uniform(2, 12)
    generator = np.random.default_rng(rng.randint(0, 2**31 - 1))
    noisy = image.astype(np.float32) + generator.normal(0, sigma, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    amount = rng.uniform(0.0005, 0.003)
    count = int(amount * image.shape[0] * image.shape[1])
    for _ in range(count):
        y = rng.randint(0, image.shape[0] - 1)
        x = rng.randint(0, image.shape[1] - 1)
        noisy[y, x] = 255 if rng.random() < 0.5 else 0
    return noisy


def jpeg(image: np.ndarray, rng: random.Random) -> np.ndarray:
    quality = rng.randint(25, 85)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


PROBABILITIES = {
    "perspective": 0.90,
    "scale": 1.00,
    "motion_blur": 0.45,
    "defocus": 0.30,
    "low_light": 0.40,
    "infrared": 0.15,
    "rain": 0.15,
    "dirt": 0.35,
    "glare": 0.25,
    "noise": 0.60,
    "jpeg": 0.85,
}

FUNCTIONS = {
    "perspective": perspective,
    "scale": scale_to_target,
    "motion_blur": motion_blur,
    "defocus": defocus,
    "low_light": low_light,
    "infrared": infrared,
    "rain": rain,
    "dirt": dirt,
    "glare": glare,
    "noise": noise,
    "jpeg": jpeg,
}


# Stages that each remove real information. At most SEVERITY_BUDGET of them may apply to
# one sample, so the corpus keeps a hard tail without accumulating a pile of plates no
# human and no recogniser could read.
HEAVY = ("motion_blur", "defocus", "low_light", "rain", "dirt", "glare")
SEVERITY_BUDGET = 3


def degrade(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, list[str]]:
    """Apply the pipeline in the fixed order of STAGES. Returns the image and what was applied."""
    applied: list[str] = []
    out = image
    spent = 0
    for stage in STAGES:
        if rng.random() > PROBABILITIES[stage]:
            continue
        if stage in HEAVY:
            if spent >= SEVERITY_BUDGET:
                continue
            spent += 1
        try:
            out = FUNCTIONS[stage](out, rng)
        except cv2.error:
            # A stage that cannot run on this sample is counted, never silently ignored.
            applied.append(f"{stage}:failed")
            continue
        applied.append(stage)
    return out, applied
