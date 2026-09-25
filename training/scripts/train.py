#!/usr/bin/env python3
"""Train the TRUEWATCH appearance detector: one YOLO11-s, one head, two stages of one lineage.

Why this file exists instead of a bare `yolo train` call. The training runs on free Kaggle sessions
that are capped at 12 h and can die at any instant, on a 30 GPU-hour weekly budget, so the parts
that matter are not the Ultralytics call but what surrounds it:

  * CHECKPOINT SAFETY. Ultralytics writes last.pt with a plain write_bytes(), so a session killed
    mid-write leaves a truncated file, and its own end-of-run step strips the optimiser out of
    last.pt (epoch -1), which makes a stopped-for-time run un-resumable. train.py therefore copies
    every finished last.pt to last_good.pt atomically and, on resume, picks the newest checkpoint
    that actually loads AND still carries optimiser state, repairing last.pt from last_good.pt.
  * A WALL-CLOCK DEADLINE (--max-hours). After every epoch the run predicts whether one more epoch
    (measured in this session, or taken from earlier sessions' train_log.csv) still ends inside the
    budget, and stops cleanly at the epoch boundary when it would not, so the next session can
    --resume instead of Kaggle killing the process in the middle of an epoch. The notebook passes
    the time REMAINING in its session, so two stages in one session share one 12 h clock.
  * A LOG THAT SPANS SESSIONS. train_log.csv is appended, never truncated, and reconciled against
    the checkpoint on resume so a kill leaves neither a gap nor a duplicate epoch. It carries a
    per-class AP50 column for every class; early-stopping patience is restored from it on resume.
  * THE AUGMENTATION CONTRACT. Every key of datasets/config/augment.yaml is either applied or
    reported, and an unknown key stops the run (exit 3) rather than being silently ignored:
      - mosaic, scale, translate, fliplr, hsv_* and mosaic_close_epochs become Ultralytics arguments;
      - downscale_upscale, jpeg and motion_blur become the Albumentations list Ultralytics applies to
        every training sample (replacing its built-in defaults, which are never used);
      - the infrared_only block (CLAHE, sensor noise, brightness/contrast, thermal wash-out) is
        applied per SOURCE image, before mosaic, and only to LWIR images, by a dataset hook, because
        the modality is a property of one image and a mosaic mixes day and IR;
      - the forbidden block (DATASET_SPEC 4.3) is asserted against the final resolved arguments.
    Albumentations is therefore required: without it the run fails loudly, listing the keys that
    would go unapplied.
  * IMBALANCE WITHOUT LOSS REWEIGHTING. An image holding a truck or a cart enters the epoch list
    twice; stage 2 also repeats LWIR images; nothing is repeated more than twice. The list is a
    train_list.txt in the run dir and the resolved data yaml points at it.
  * DATASET PREFLIGHT. Class-id range, orphan and empty labels, and duplicate rows in manifest.tsv
    (two source frames written to one final filename silently pair an image with the wrong labels)
    fail the run with exit code 3 before any GPU time is spent.
  * THE CART GATE. DATASET_SPEC 1.5 withdraws class 4 (cart) when fewer than 300 hand-verified
    instances exist. A dataset may say so by listing only the first four names in its data.yaml, or
    by keeping all five with no cart instance. Either way the head keeps five outputs (id 4 stays
    reserved, so the wire names of ARCHITECTURE_V2 4.2 and the ONNX output shape do not change) and
    class 4 simply receives no positives; a withdrawn class that still has label rows is an error.

ONE lineage, two stages: `--stage day` trains from COCO weights on the full mixed corpus, `--stage
ir` continues from the day stage's best.pt with LWIR images repeated and a lower learning rate.
The reasoning is at the top of configs/yolo11s_day.yaml.

The per-epoch validation numbers Ultralytics computes are over the whole val set, day and IR
together. They exist to pick checkpoints and to stop the run; they are not reported results.
evaluate.py reports day and IR separately, on val, with the host recorded.

Ultralytics is AGPL-3.0 and is imported only inside training/, and only inside functions here, so
the pure parts of this file (list building, preflight, checkpoint picking) import without it.

Exit codes: 0 done, already complete, or stopped for session time; 2 usage error or a missing
checkpoint or init weights; 3 dataset preflight failure or a violated augmentation assertion.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DATASET = 3

STAGES = ("day", "ir")
DEFAULT_CONFIGS = {stage: C.CONFIG_DIR / f"yolo11s_{stage}.yaml" for stage in STAGES}
DATA_YAML = C.CONFIG_DIR / "data.yaml"
WEIGHTS_DIR = C.TRAINING_ROOT / "weights"
RUNS_DIR = C.TRAINING_ROOT / "runs"
DEFAULT_MAX_HOURS = 9.5  # standalone default; the notebook always passes the time left in its session
EPOCH_ESTIMATE_SAFETY = 1.10  # the next epoch is assumed to take 10% longer than the slowest recent one

# The five wire names of ARCHITECTURE_V2 4.2 / datasets/config/schema.yaml, in id order. A test pins this
# tuple to the schema; it exists so the log columns are fixed even before the schema has been read.
SCHEMA_CLASS_NAMES = ("person", "two_wheeler", "car", "truck", "cart")
CART_ID = 4
AP_COLUMNS = [f"ap50_{name}" for name in SCHEMA_CLASS_NAMES]
LOG_COLUMNS = [
    "epoch", "session", "elapsed_s", "epoch_s", "box_loss", "cls_loss", "dfl_loss",
    "precision", "recall", "map50", "map50_95", "person_ap50", "lr", "fitness", *AP_COLUMNS,
]
VAL_SCOPE = (
    "per-epoch validation is over the whole val set, day and IR together; it selects checkpoints and "
    "stops the run, it is not a reported result (evaluate.py reports day and IR separately)"
)

# Arguments train.py takes from datasets/config/augment.yaml. A training config may not set them.
INJECTED_KEYS = ("mosaic", "scale", "translate", "fliplr", "hsv_h", "hsv_s", "hsv_v", "close_mosaic", "augmentations")
# Ultralytics arguments that DATASET_SPEC 4.3 forbids being non-zero. `degrees` is bounded, not zero, see below.
ZERO_ARGS = ("flipud", "perspective", "mixup", "copy_paste", "shear", "cutmix", "bgr")
# Arguments train.py sets itself; a config that carries them is a mistake, not a preference.
OWNED_KEYS = ("data", "model", "task", "mode", "project", "name", "save_dir", "exist_ok", "resume", "device", "fraction", "time")
# The only arguments Ultralytics lets a resume override; everything else comes from the checkpoint.
RESUME_OVERRIDABLE = ("imgsz", "batch", "close_mosaic", "augmentations", "save_period", "workers", "cache", "patience", "freeze", "val", "plots")
DRIFT_WATCH = ("epochs", "lr0", "lrf", "optimizer", "momentum", "weight_decay", "cos_lr", "warmup_epochs", "box", "cls", "dfl", "seed", "nbs")


class UsageError(Exception):
    """Bad command line, config or missing checkpoint. Exit 2."""


class DatasetError(Exception):
    """The dataset failed preflight. Exit 3."""


class AugmentationViolation(Exception):
    """A forbidden augmentation is active, or the Albumentations decision did not take effect. Exit 3."""


def say(message: str = "") -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr, flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------
# Configuration: the stage yaml, the augmentation contract, the data yaml
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingRules:
    rare_classes: frozenset[int]
    rare_repeat: int
    lwir_repeat: int
    max_repeat: int


@dataclass(frozen=True)
class StageConfig:
    stage: str
    model: str | None
    unfreeze_epoch: int | None
    sampling: SamplingRules
    train: dict[str, Any]
    source: Path
    close_mosaic_epochs: int | None = None  # the one stage-specific exception to "augmentation comes only from augment.yaml"


@dataclass(frozen=True)
class IRAugmentSpec:
    """The infrared_only block of augment.yaml (DATASET_SPEC 4.2), as numbers."""

    clahe_p: float
    clahe_clip: tuple[float, float]
    clahe_grid: int
    noise_p: float
    noise_sigma: tuple[float, float]       # 8-bit grey levels
    bc_p: float
    brightness: float                      # +/- fraction of full scale
    contrast: float                        # +/- fraction
    washout_p: float
    washout_range: tuple[float, float]     # the dynamic range kept, as a fraction of the original


@dataclass(frozen=True)
class AugmentPolicy:
    inject: dict[str, float]
    mosaic_close_epochs: int
    degrees: float
    degrees_hard_max: float
    source: Path
    albumentations: tuple[dict, ...] = ()   # always-on pixel transforms, in the A.to_dict form Ultralytics accepts
    ir: IRAugmentSpec | None = None
    plan: dict[str, str] = field(default_factory=dict)  # "block.key" -> how it is applied (or why it cannot be)


def _read_yaml(path: Path) -> dict:
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UsageError(f"{path} does not exist") from exc
    except yaml.YAMLError as exc:
        raise UsageError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_stage_config(path: Path, stage: str) -> StageConfig:
    raw = _read_yaml(path)
    if raw.get("stage") != stage:
        raise UsageError(f"{path} is a config for stage {raw.get('stage')!r}, but --stage {stage} was requested")
    train = raw.get("train")
    if not isinstance(train, dict) or not train:
        raise UsageError(f"{path} needs a non-empty `train:` mapping of Ultralytics arguments")

    duplicated = sorted(k for k in train if k in INJECTED_KEYS)
    if duplicated:
        raise UsageError(
            f"{path} sets {duplicated} under `train:`. Augmentation values come only from "
            f"datasets/config/augment.yaml (train.py injects them); remove them here so the two cannot drift."
        )
    owned = sorted(k for k in train if k in OWNED_KEYS)
    if owned:
        raise UsageError(f"{path} sets {owned} under `train:`, but train.py owns those (use the command-line flags)")
    if int(train.get("epochs", 0)) < 1:
        raise UsageError(f"{path}: train.epochs must be a positive integer")

    model = raw.get("model")
    if stage == "day" and not model:
        raise UsageError(f"{path}: stage day needs `model:` (the COCO weights to start from)")

    unfreeze = (raw.get("schedule") or {}).get("unfreeze_epoch")
    frozen = train.get("freeze") not in (None, 0, [])
    if frozen and unfreeze is None:
        raise UsageError(f"{path}: train.freeze is set but schedule.unfreeze_epoch is null, so the backbone would never be released")
    if not frozen and unfreeze is not None:
        raise UsageError(f"{path}: schedule.unfreeze_epoch is set but train.freeze is empty, so there is nothing to release")

    s = raw.get("sampling") or {}
    try:
        sampling = SamplingRules(
            rare_classes=frozenset(int(c) for c in s["rare_classes"]),
            rare_repeat=int(s["rare_repeat"]),
            lwir_repeat=int(s["lwir_repeat"]),
            max_repeat=int(s["max_repeat"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise UsageError(f"{path}: `sampling:` needs rare_classes, rare_repeat, lwir_repeat and max_repeat ({exc})") from exc
    if min(sampling.rare_repeat, sampling.lwir_repeat, sampling.max_repeat) < 1:
        raise UsageError(f"{path}: every sampling repeat must be at least 1")

    close_mosaic = raw.get("close_mosaic_epochs")
    if close_mosaic is not None and (isinstance(close_mosaic, bool) or not isinstance(close_mosaic, int) or close_mosaic < 0):
        raise UsageError(f"{path}: close_mosaic_epochs must be null or a non-negative integer, got {close_mosaic!r}")

    return StageConfig(stage, model, None if unfreeze is None else int(unfreeze), sampling, dict(train), Path(path), close_mosaic)


# How every key of augment.yaml reaches training. A key missing from this table is an unapplied key and
# stops the run: a policy entry that nobody applies is exactly the silent drift this contract prevents.
ULTRALYTICS_KEYS = ("mosaic", "scale", "translate", "fliplr", "hsv_h", "hsv_s", "hsv_v")
ALBUMENTATIONS_KEYS = ("downscale_upscale", "jpeg", "motion_blur")
IR_KEYS = ("clahe", "gaussian_noise", "brightness_contrast", "thermal_washout")
FORBIDDEN_ZERO_KEYS = ("flipud", "perspective", "mixup", "copy_paste", "channel_shuffle", "false_colour")
POLICY_ROUTES: dict[str, dict[str, str]] = {
    "always_on": {
        **{k: f"Ultralytics argument `{k}`" for k in ULTRALYTICS_KEYS},
        "mosaic_close_epochs": "Ultralytics argument `close_mosaic` (clamped to the stage length; a stage config may override it)",
        "downscale_upscale": "Albumentations Downscale (INTER_AREA down, INTER_LINEAR up) on every training sample",
        "jpeg": "Albumentations ImageCompression (JPEG) on every training sample",
        "motion_blur": "Albumentations MotionBlur (random angle) on every training sample",
    },
    "infrared_only": {k: "train.py LWIR hook, per source image before mosaic, LWIR images only" for k in IR_KEYS},
    "forbidden": {
        "flipud": "asserted 0 (Ultralytics `flipud`)",
        "perspective": "asserted 0 (Ultralytics `perspective`; `shear` is also held at 0)",
        "mixup": "asserted 0 (Ultralytics `mixup`; `cutmix` is also held at 0)",
        "copy_paste": "asserted 0 (Ultralytics `copy_paste`)",
        "degrees": "Ultralytics `degrees`, asserted within +/- degrees_hard_max",
        "degrees_hard_max": "the bound the `degrees` assertion uses",
        "channel_shuffle": "asserted 0; Ultralytics `bgr` held at 0 and the LWIR hook writes B = G = R",
        "false_colour": "asserted 0; nothing in the training pipeline maps grey to colour",
        "erase_max_box_fraction": "not applicable: the Ultralytics detection pipeline has no random erase (`erasing` is classification-only)",
    },
    "infrared_conversion": {
        "*": "offline conversion invariant applied by datasets/scripts/04_ir_to_3ch.py, not a training augmentation",
    },
}

CV2_INTER_LINEAR = 1  # cv2.INTER_LINEAR; a test pins both values to cv2 so no cv2 import is needed here
CV2_INTER_AREA = 3    # cv2.INTER_AREA


def _prob(block: Mapping[str, Any], key: str) -> float:
    p = float(block["p"])
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{key}.p={p} is not a probability")
    return p


def albumentations_specs(always: Mapping[str, Any]) -> tuple[dict, ...]:
    """The always-on pixel transforms as serialised Albumentations dicts (the `A.to_dict()` form).

    Ultralytics accepts this form in `augmentations` directly and restores it with `A.from_dict`,
    which keeps args.yaml and the checkpoint's train_args plain data, and needs no albumentations
    import here. Values come from augment.yaml only.
    """
    d, j, m = always["downscale_upscale"], always["jpeg"], always["motion_blur"]
    fmin, fmax = float(d["factor_min"]), float(d["factor_max"])
    if not 1.0 <= fmin <= fmax:
        raise ValueError(f"downscale_upscale factors {fmin}..{fmax} must satisfy 1 <= min <= max")
    qmin, qmax = int(j["quality_min"]), int(j["quality_max"])
    if not 1 <= qmin <= qmax <= 100:
        raise ValueError(f"jpeg quality {qmin}..{qmax} must lie in 1..100")
    kmin, kmax = int(m["kernel_min"]), int(m["kernel_max"])
    if kmin < 3 or kmax < kmin or kmin % 2 == 0 or kmax % 2 == 0:
        raise ValueError(f"motion_blur kernel {kmin}..{kmax} must be odd and at least 3")
    return (
        {"transform": {"__class_fullname__": "Downscale", "p": _prob(d, "downscale_upscale"),
                       "scale_range": [1.0 / fmax, 1.0 / fmin],
                       "interpolation_pair": {"downscale": CV2_INTER_AREA, "upscale": CV2_INTER_LINEAR}}},
        {"transform": {"__class_fullname__": "ImageCompression", "p": _prob(j, "jpeg"),
                       "compression_type": "jpeg", "quality_range": [qmin, qmax]}},
        {"transform": {"__class_fullname__": "MotionBlur", "p": _prob(m, "motion_blur"),
                       "blur_limit": [kmin, kmax], "angle_range": [0.0, 360.0]}},
    )


def ir_spec(block: Mapping[str, Any]) -> IRAugmentSpec:
    c, n, b, w = block["clahe"], block["gaussian_noise"], block["brightness_contrast"], block["thermal_washout"]
    spec = IRAugmentSpec(
        clahe_p=_prob(c, "clahe"), clahe_clip=(float(c["clip_min"]), float(c["clip_max"])), clahe_grid=int(c["tile_grid"]),
        noise_p=_prob(n, "gaussian_noise"), noise_sigma=(float(n["sigma_min"]), float(n["sigma_max"])),
        bc_p=_prob(b, "brightness_contrast"), brightness=float(b["brightness"]), contrast=float(b["contrast"]),
        washout_p=_prob(w, "thermal_washout"), washout_range=(float(w["range_min"]), float(w["range_max"])),
    )
    for name, (lo, hi) in (("clahe clip", spec.clahe_clip), ("noise sigma", spec.noise_sigma), ("washout range", spec.washout_range)):
        if not 0.0 <= lo <= hi:
            raise ValueError(f"infrared_only {name} {lo}..{hi} must satisfy 0 <= min <= max")
    if spec.washout_range[1] > 1.0 or spec.clahe_grid < 1:
        raise ValueError("thermal_washout range must be <= 1 and clahe tile_grid >= 1")
    return spec


def policy_plan(raw: Mapping[str, Any], path: Path) -> dict[str, str]:
    """"block.key" -> how it is applied, for every key of augment.yaml. Raises on any key with no route."""
    plan: dict[str, str] = {}
    unapplied: list[str] = []
    for block, value in raw.items():
        routes = POLICY_ROUTES.get(block)
        if routes is None or not isinstance(value, Mapping):
            unapplied.append(str(block))
            continue
        for key in value:
            route = routes.get(key) or routes.get("*")
            if route is None:
                unapplied.append(f"{block}.{key}")
            else:
                plan[f"{block}.{key}"] = route
    if unapplied:
        raise AugmentationViolation(
            f"{path} has key(s) that train.py does not apply: {unapplied}. Every policy entry must be applied or explicitly "
            f"reported (DATASET_SPEC 4); add a route for it in train.py POLICY_ROUTES and an implementation, or remove it."
        )
    return plan


def load_augment_policy(path: Path = C.AUGMENT_YAML) -> AugmentPolicy:
    raw = _read_yaml(path)
    plan = policy_plan(raw, Path(path))
    try:
        always = raw["always_on"]
        forbidden = raw["forbidden"]
        inject = {k: float(always[k]) for k in ULTRALYTICS_KEYS}
        policy = AugmentPolicy(
            inject=inject,
            mosaic_close_epochs=int(always["mosaic_close_epochs"]),
            degrees=float(forbidden["degrees"]),
            degrees_hard_max=float(forbidden["degrees_hard_max"]),
            source=Path(path),
            albumentations=albumentations_specs(always),
            ir=ir_spec(raw["infrared_only"]),
            plan=plan,
        )
        zeros = {k: float(forbidden[k]) for k in FORBIDDEN_ZERO_KEYS}
    except (KeyError, TypeError, ValueError) as exc:
        raise UsageError(f"{path} is missing or has a malformed entry the trainer depends on: {exc!r}") from exc
    # The file's own comment says any non-zero value in the forbidden block is a bug.
    bad = {k: v for k, v in zeros.items() if v != 0.0}
    if bad or policy.degrees > policy.degrees_hard_max:
        raise AugmentationViolation(f"{path} lists non-zero forbidden values {bad} (degrees {policy.degrees}); fix the policy file")
    return policy


def expected_albumentations(policy: AugmentPolicy) -> list[str]:
    """Class names of the always-on transforms, in order, as the dataset must report them."""
    return [spec["transform"]["__class_fullname__"] for spec in policy.albumentations]


def check_albumentations(policy: AugmentPolicy) -> str:
    """Albumentations must import and rebuild every always-on transform; returns its version. Exit 3 otherwise.

    Ultralytics swallows any Albumentations failure (a missing package, or an old release without
    `scale_range`/`quality_range`) and trains on without the transforms, so this is checked here,
    before any GPU time, and again against the live dataset in on_pretrain_routine_end.
    """
    keys = [f"always_on.{k}" for k in ALBUMENTATIONS_KEYS]
    try:
        import albumentations as A
    except ImportError as exc:
        raise AugmentationViolation(
            f"albumentations is not installed, so these augment.yaml keys would go unapplied: {keys}. "
            f"Install it (pip install -r training/requirements.txt)."
        ) from exc
    rebuilt = []
    for spec in policy.albumentations:
        try:
            rebuilt.append(type(A.from_dict(spec)).__name__)
        except Exception as exc:  # any failure means the transform would silently not run
            raise AugmentationViolation(
                f"albumentations {A.__version__} cannot build {spec['transform']['__class_fullname__']} ({type(exc).__name__}: {exc}); "
                f"these augment.yaml keys would go unapplied: {keys}. training/requirements.txt names the minimum version."
            ) from exc
    if rebuilt != expected_albumentations(policy):
        raise AugmentationViolation(f"albumentations rebuilt {rebuilt}, expected {expected_albumentations(policy)}")
    return str(A.__version__)


def augmentation_report(policy: AugmentPolicy) -> list[str]:
    """One line per augment.yaml key, saying how it is applied. Printed at start and stored in run_state.json."""
    lines = [f"augmentation policy {policy.source}: every key applied or reported"]
    lines += [f"  {key:<40} {route}" for key, route in policy.plan.items()]
    return lines


# --------------------------------------------------------------------------------------------
# The infrared-only block: applied per LWIR source image by a dataset hook
# --------------------------------------------------------------------------------------------


def thermal_washout(grey: Any, keep: float) -> Any:
    """Compress the dynamic range around the frame mean to `keep` (0..1) of the original: ground and body converge."""
    import numpy as np

    g = grey.astype(np.float32)
    mean = float(g.mean())
    return np.clip(mean + (g - mean) * float(keep), 0, 255).astype(np.uint8)


class IRAugment:
    """The DATASET_SPEC 4.2 infrared-only transforms, for one LWIR image.

    Order follows the signal chain: the scene washes out (thermal_washout), the microbolometer adds
    noise (gaussian_noise), auto-gain moves brightness and contrast (brightness_contrast), then the
    display-side CLAHE. All four act on ONE grey channel, which is replicated back to three at the
    end, so the B = G = R invariant of DATASET_SPEC 3.4 holds by construction (no channel shuffle,
    no false colour). Randomness comes from Python's `random`, which Ultralytics reseeds per dataloader
    worker, so workers do not repeat each other. Holds only numbers, so it pickles into workers.
    """

    def __init__(self, spec: IRAugmentSpec) -> None:
        self.spec = spec

    def __call__(self, img: Any, rng: random.Random | None = None) -> Any:
        import cv2
        import numpy as np

        r = rng or random
        s = self.spec
        if img.ndim == 3 and img.shape[2] == 3:
            same = np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2])
            grey = img[..., 0] if same else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            grey = img.reshape(img.shape[0], img.shape[1])
        out = np.ascontiguousarray(grey, dtype=np.uint8)
        if r.random() < s.washout_p:
            out = thermal_washout(out, r.uniform(*s.washout_range))
        if r.random() < s.noise_p:
            sigma = r.uniform(*s.noise_sigma)
            noise = np.random.default_rng(r.getrandbits(32)).normal(0.0, sigma, out.shape)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if r.random() < s.bc_p:
            alpha = 1.0 + r.uniform(-s.contrast, s.contrast)
            beta = r.uniform(-s.brightness, s.brightness) * 255.0   # brightness as a fraction of full scale
            out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if r.random() < s.clahe_p:
            clahe = cv2.createCLAHE(clipLimit=r.uniform(*s.clahe_clip), tileGridSize=(s.clahe_grid, s.clahe_grid))
            out = clahe.apply(out)
        if img.ndim == 3:
            return np.repeat(out[:, :, None], img.shape[2], axis=2)
        return out.reshape(img.shape)


_DATASET_CLASS: Any = None


def modality_dataset_class() -> Any:
    """YOLODataset with the LWIR hook: `get_image_and_label` applies `ir_augment` to LWIR images when augmenting.

    Ultralytics' Mosaic calls `dataset.get_image_and_label` for every image it tiles, so the hook
    runs per source image, before mosaic, and a day tile next to an IR tile is never touched. Created
    lazily so this module imports without Ultralytics; published as a module attribute (see
    __getattr__ below) so a spawned dataloader worker can unpickle it.
    """
    global _DATASET_CLASS
    if _DATASET_CLASS is None:
        from ultralytics.data.dataset import YOLODataset

        class ModalityAwareYOLODataset(YOLODataset):
            ir_augment: IRAugment | None = None

            def get_image_and_label(self, index: int) -> dict:
                label = super().get_image_and_label(index)
                if self.augment and self.ir_augment is not None and C.modality_of_stem(Path(label["im_file"]).stem) == "lwir":
                    label["img"] = self.ir_augment(label["img"])  # a new array: the RAM buffer's copy is never modified
                return label

        ModalityAwareYOLODataset.__module__ = __name__
        ModalityAwareYOLODataset.__qualname__ = "ModalityAwareYOLODataset"
        _DATASET_CLASS = ModalityAwareYOLODataset
        globals()["ModalityAwareYOLODataset"] = ModalityAwareYOLODataset
    return _DATASET_CLASS


def __getattr__(name: str) -> Any:
    if name == "ModalityAwareYOLODataset":
        return modality_dataset_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_trainer_class(ir_augment: IRAugment | None) -> Any:
    """A DetectionTrainer whose TRAIN dataset carries the LWIR hook. Validation data is never augmented."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    dataset_cls = modality_dataset_class()

    class TrueWatchDetectionTrainer(DetectionTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                if type(dataset) is not dataset_cls.__mro__[1]:
                    raise AugmentationViolation(
                        f"Ultralytics built a {type(dataset).__name__}, not a YOLODataset; the infrared_only block cannot be attached"
                    )
                dataset.__class__ = dataset_cls
                dataset.ir_augment = ir_augment
            return dataset

    return TrueWatchDetectionTrainer



def effective_close_mosaic(policy_epochs: int, epochs: int) -> int:
    """`close_mosaic` for a stage of `epochs` epochs.

    Ultralytics closes mosaic when epoch == epochs - close_mosaic and, on a resume, whenever
    start_epoch > epochs - close_mosaic. With close_mosaic larger than the stage the first is never
    true, so a fresh run would keep mosaic on while a resumed run switched it off. Clamping to the
    stage length makes both behave the same. (The ir stage sets its own close_mosaic_epochs so the
    clamp is a safety net there, not the mechanism.)
    """
    return max(0, min(int(policy_epochs), int(epochs)))


def resolve_train_args(cfg: StageConfig, policy: AugmentPolicy, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The Ultralytics arguments for this stage: the yaml, the CLI overrides, the injected augmentation.

    Forbidden arguments are given an explicit zero unless the config already carries a value, so a
    config that tries to enable one reaches assert_forbidden_augmentation instead of being masked.
    """
    args = dict(cfg.train)
    args.update({k: v for k, v in (overrides or {}).items() if v is not None})
    args = {k: v for k, v in args.items() if v is not None}
    args.update(policy.inject)
    close = cfg.close_mosaic_epochs if cfg.close_mosaic_epochs is not None else policy.mosaic_close_epochs
    args["close_mosaic"] = effective_close_mosaic(close, int(args["epochs"]))
    # augment.yaml's always-on pixel transforms. A non-empty list also REPLACES Ultralytics' built-in defaults
    # (Blur, MedianBlur, ToGray, CLAHE at p=0.01), which the policy does not contain; None would keep them.
    args["augmentations"] = [copy.deepcopy(spec) for spec in policy.albumentations]
    for key in ZERO_ARGS:
        args.setdefault(key, 0.0)
    args.setdefault("degrees", policy.degrees)
    return args


def assert_forbidden_augmentation(resolved: Any, policy: AugmentPolicy) -> None:
    """Raise AugmentationViolation if the resolved Ultralytics arguments enable anything DATASET_SPEC 4.3 forbids."""
    get = (lambda k: resolved.get(k, 0.0)) if isinstance(resolved, Mapping) else (lambda k: getattr(resolved, k, 0.0))
    problems = []
    for key in ZERO_ARGS:
        value = get(key)
        if value is None or float(value) != 0.0:
            problems.append(f"{key}={value} (must be 0)")
    degrees = get("degrees")
    if degrees is None or abs(float(degrees)) > policy.degrees_hard_max:
        problems.append(f"degrees={degrees} (must be within +/-{policy.degrees_hard_max:g})")
    if problems:
        raise AugmentationViolation(
            "forbidden augmentation active in the resolved training arguments: " + "; ".join(problems)
            + f". DATASET_SPEC 4.3 forbids these; the policy is {policy.source}."
        )


def load_data_yaml(path: Path = DATA_YAML) -> dict:
    data = _read_yaml(path)
    for key in ("path", "train", "val", "nc", "names"):
        if key not in data:
            raise UsageError(f"{path} has no `{key}` key")
    return data


def normalise_names(names: Any) -> list[str]:
    """Class names in id order from either yaml form (a list, or an {id: name} mapping)."""
    if isinstance(names, Mapping):
        return [str(names[i]) for i in sorted(names, key=int)]
    return [str(n) for n in names]


def assert_names_match_schema(names: Any, schema_names: Sequence[str], where: str) -> None:
    got = normalise_names(names)
    if got != list(schema_names):
        raise DatasetError(f"{where} class names {got} differ from datasets/config/schema.yaml {list(schema_names)}; every component must share one taxonomy")


def dataset_withdrawn_classes(dataset_names: Any, schema_names: Sequence[str], where: str) -> tuple[int, ...]:
    """Class ids the dataset's own data.yaml withdraws. () when it lists the full schema.

    DATASET_SPEC 1.5 lets exactly one class be withdrawn, cart (id 4, the last), by the hand-verification
    gate (schema.yaml cart_gate.on_failure: withdraw_class_4). A dataset may express that by listing only
    ids 0..3. Any other difference from the schema is a taxonomy mismatch and stops the run.
    """
    got = normalise_names(dataset_names)
    schema = list(schema_names)
    if got == schema:
        return ()
    if len(schema) == CART_ID + 1 and schema[CART_ID] == "cart" and got == schema[:CART_ID]:
        return (CART_ID,)
    raise DatasetError(
        f"{where} class names {got} differ from datasets/config/schema.yaml {schema}; every component must share one taxonomy "
        f"(the only allowed difference is the cart gate's withdrawal of the last class, id {CART_ID})"
    )


def resolved_data_dict(data_cfg: Mapping[str, Any], root: Path, train_list: Path) -> dict[str, Any]:
    """The data yaml Ultralytics actually reads: absolute path, the balanced list for train, no test key."""
    root = Path(root).resolve()
    return {
        "path": root.as_posix(),
        "train": Path(train_list).resolve().as_posix(),
        "val": (root / str(data_cfg["val"])).as_posix(),
        "nc": int(data_cfg["nc"]),
        "names": {i: n for i, n in enumerate(normalise_names(data_cfg["names"]))},
    }


def assert_resolved_data(resolved: Mapping[str, Any], schema_names: Sequence[str]) -> None:
    if "test" in resolved:
        raise DatasetError("the resolved data yaml has a `test` key; the test split is sealed until Phase 11 (DATASET_SPEC 2.5)")
    if not Path(str(resolved["path"])).is_absolute():
        raise DatasetError(f"resolved data path {resolved['path']!r} is not absolute; Ultralytics would resolve it against its own datasets dir")
    if int(resolved["nc"]) != len(schema_names):
        raise DatasetError(f"resolved nc={resolved['nc']} but the schema has {len(schema_names)} classes")
    assert_names_match_schema(resolved["names"], schema_names, "the resolved data yaml")


def write_yaml_atomic(path: Path, payload: Mapping[str, Any], header: str = "") -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(header + yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------
# Dataset preflight
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageEntry:
    path: str
    classes: frozenset[int]
    modality: str  # "visible" | "lwir"


@dataclass
class SplitScan:
    split: str
    entries: list[ImageEntry] = field(default_factory=list)
    instances: Counter = field(default_factory=Counter)
    images_with: Counter = field(default_factory=Counter)
    n_empty: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    bad_class_rows: list[str] = field(default_factory=list)
    bad_rows: list[str] = field(default_factory=list)
    unclassified: list[str] = field(default_factory=list)
    duplicate_stems: list[str] = field(default_factory=list)
    modality_counts: Counter = field(default_factory=Counter)
    images_with_bad_rows: int = 0


def parse_label_text(text: str, nc: int) -> tuple[list[int], list[str], list[str]]:
    """(class ids, bad-class rows, other malformed rows) for one YOLO detection label file's text."""
    classes: list[int] = []
    bad_class: list[str] = []
    bad_rows: list[str] = []
    for number, raw in enumerate(text.splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 5:
            bad_rows.append(f"line {number}: {len(parts)} fields, expected 5")
            continue
        try:
            cls_value = float(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            bad_class.append(f"line {number}: non-numeric field in {raw.strip()!r}")
            continue
        if cls_value != int(cls_value) or not 0 <= int(cls_value) < nc:
            bad_class.append(f"line {number}: class id {parts[0]} is outside 0..{nc - 1}")
            continue
        classes.append(int(cls_value))
        if any(not math.isfinite(c) or c < 0 or c > 1.000001 for c in coords) or coords[2] <= 0 or coords[3] <= 0:
            bad_rows.append(f"line {number}: box {coords} is not normalised (0..1) or has no area")
    return classes, bad_class, bad_rows


def scan_split(root: Path, split: str, nc: int) -> SplitScan:
    """Read every image and label of one split. Reads only; writes nothing."""
    scan = SplitScan(split)
    images = C.list_split_images(root, split)
    label_dir = root / "labels" / split
    stem_seen: Counter = Counter(p.stem for p in images)
    scan.duplicate_stems = sorted(s for s, n in stem_seen.items() if n > 1)
    label_stems = {p.stem for p in label_dir.glob("*.txt")} if label_dir.is_dir() else set()
    scan.orphan_labels = sorted(label_stems - set(stem_seen))

    for image in images:
        modality = C.modality_of_stem(image.stem)
        if modality is None:
            scan.unclassified.append(image.name)
            modality = "unknown"
        scan.modality_counts[modality] += 1

        label = C.label_path_for(image)
        if not label.exists():
            scan.missing_labels.append(image.name)
            classes: list[int] = []
        else:
            classes, bad_class, bad_rows = parse_label_text(label.read_text(encoding="utf-8", errors="replace"), nc)
            scan.bad_class_rows.extend(f"{label.name} {r}" for r in bad_class)
            if bad_rows or bad_class:
                scan.images_with_bad_rows += 1
            scan.bad_rows.extend(f"{label.name} {r}" for r in bad_rows)
            if not classes:
                scan.n_empty += 1
        scan.instances.update(classes)
        scan.images_with.update(set(classes))
        scan.entries.append(ImageEntry(str(image), frozenset(classes), modality))
    return scan


def read_manifest(path: Path) -> list[tuple[str, str, str]]:
    """Rows of manifest.tsv as (split, relative image path, label count)."""
    rows = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.strip():
            parts = raw.split("\t")
            rows.append((parts[0], parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else ""))
    return rows


def manifest_duplicates(rows: Sequence[tuple[str, str, str]]) -> list[tuple[str, int]]:
    counts = Counter(rel for _, rel, _ in rows)
    return sorted(((rel, n) for rel, n in counts.items() if n > 1), key=lambda t: (-t[1], t[0]))


@dataclass
class Preflight:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _histogram_lines(scan: SplitScan, names: Sequence[str]) -> list[str]:
    total = sum(scan.instances.values())
    out = [f"  {scan.split}: {len(scan.entries)} images, {total} instances, {scan.n_empty} empty-label images"]
    for cid, name in enumerate(names):
        n = scan.instances.get(cid, 0)
        share = 100.0 * n / total if total else 0.0
        out.append(f"    class {cid} {name:<12} {n:>9} instances {share:6.2f}%   in {scan.images_with.get(cid, 0):>7} images")
    return out


def run_preflight(
    root: Path, names: Sequence[str], train: SplitScan, val: SplitScan, manifest: Path | None, cart_gate: int,
    withdrawn: Sequence[int] = (),
) -> Preflight:
    """Judge the scans. Errors abort the run (exit 3); warnings are printed and the run goes on.

    `withdrawn` holds the ids the dataset's data.yaml withdraws (the cart gate). Such a class must have
    no label row in any split; it stays in the head as a reserved output that receives no positives.
    """
    report = Preflight()
    nc = len(names)
    withdrawn = tuple(int(c) for c in withdrawn)
    for cid in withdrawn:
        for scan in (train, val):
            n = scan.instances.get(cid, 0)
            if n:
                report.errors.append(
                    f"class {cid} ({names[cid]}) is withdrawn by the dataset's data.yaml (the DATASET_SPEC 1.5 cart gate) but {scan.split} "
                    f"still has {n} instance(s) of it; the build is inconsistent, rebuild it"
                )

    for scan in (train, val):
        if not scan.entries:
            report.errors.append(f"images/{scan.split} holds no images under {root}")
            continue
        if scan.bad_class_rows:
            report.errors.append(
                f"{len(scan.bad_class_rows)} label row(s) in {scan.split} have a class id outside 0..{nc - 1} or a non-numeric field, "
                f"e.g. {scan.bad_class_rows[:3]}; Ultralytics would crash or train on the wrong class"
            )
        if scan.missing_labels:
            report.errors.append(
                f"{len(scan.missing_labels)} image(s) in {scan.split} have no label file, e.g. {scan.missing_labels[:3]}; Phase 1 writes a "
                f"label file (possibly empty) for every image, and Ultralytics would train on these as empty background"
            )
        if scan.unclassified:
            report.errors.append(
                f"{len(scan.unclassified)} image(s) in {scan.split} carry neither a _visible nor a _lwir suffix, e.g. {scan.unclassified[:3]}; "
                f"every image must be day or IR"
            )
        if scan.duplicate_stems:
            report.errors.append(
                f"{len(scan.duplicate_stems)} file stem(s) in {scan.split} exist with two suffixes, e.g. {scan.duplicate_stems[:3]}; both "
                f"would read the same label file"
            )
        if scan.bad_rows:
            share = 100.0 * scan.images_with_bad_rows / len(scan.entries)
            text = (f"{len(scan.bad_rows)} malformed label row(s) in {scan.images_with_bad_rows} {scan.split} image(s) ({share:.2f}%), "
                    f"e.g. {scan.bad_rows[:3]}; Ultralytics drops those images")
            (report.errors if share > 1.0 else report.warnings).append(text)
        if scan.orphan_labels:
            report.warnings.append(f"{len(scan.orphan_labels)} label file(s) in {scan.split} have no image, e.g. {scan.orphan_labels[:3]}; they are ignored")

    if train.entries:
        total = sum(train.instances.values())
        for cid, name in enumerate(names):
            n = train.instances.get(cid, 0)
            if cid in withdrawn:
                continue  # reported above (error if it has rows) and in the plan lines below
            if cid == nc - 1 and name == "cart":
                if n == 0:
                    report.warnings.append(
                        f"class {cid} (cart) has no train instances: treated as WITHDRAWN by the DATASET_SPEC 1.5 cart gate. Id {cid} stays "
                        f"reserved in the 5-output head and receives no positives; its AP is n/a."
                    )
                elif n < cart_gate:
                    report.warnings.append(
                        f"class {cid} (cart) has {n} train instances, below the {cart_gate} gate of DATASET_SPEC 1.5; that section withdraws "
                        f"the class unless {cart_gate} hand-verified instances exist. Id {cid} stays reserved either way."
                    )
            elif n == 0:
                report.errors.append(f"class {cid} ({name}) has no train instances; the head would train a dead class")
            elif total and 100.0 * n / total < 1.0:
                report.warnings.append(f"class {cid} ({name}) is {100.0 * n / total:.2f}% of train instances, under the 1% line of DATASET_SPEC 3.1 gate G11")
        if train.modality_counts.get("lwir", 0) == 0:
            report.warnings.append("the train split has no LWIR images; the IR stage would repeat nothing")
        if train.modality_counts.get("visible", 0) == 0:
            report.warnings.append("the train split has no visible images")

    if manifest is None or not manifest.exists():
        report.warnings.append("manifest.tsv not found beside the dataset; the duplicate-row check was skipped")
    else:
        rows = read_manifest(manifest)
        dups = manifest_duplicates(rows)
        if dups:
            report.errors.append(
                f"manifest.tsv lists {len(dups)} image path(s) more than once, e.g. {[f'{p} x{n}' for p, n in dups[:3]]}. Duplicate rows mean two "
                f"source frames were written to the same final filename, so one overwrote the other and an image is now paired with the wrong "
                f"labels. datasets/scripts/09_build_yolo_ds.py names files {{source}}_{{raw_stem}}_{{modality}}; find the colliding records, fix "
                f"the naming upstream and rebuild before training."
            )
        by_split = Counter(s for s, _, _ in rows)
        for scan in (train, val):
            if by_split.get(scan.split, 0) != len(scan.entries):
                report.warnings.append(f"manifest.tsv lists {by_split.get(scan.split, 0)} {scan.split} rows but {len(scan.entries)} images are on disk")
        train_stems = {Path(rel).stem for s, rel, _ in rows if s == "train"}
        both = sorted(train_stems & {Path(rel).stem for s, rel, _ in rows if s == "val"})
        if both:
            report.warnings.append(
                f"{len(both)} file stem(s) appear in both train and val in manifest.tsv, e.g. {both[:3]}; on real data that is the same source "
                f"frame in two splits (leakage). Only a warning here, because the Phase 1 leakage gates own that check."
            )

    report.lines = ["dataset preflight"] + _histogram_lines(train, names) + _histogram_lines(val, names)
    report.lines.append(f"  train modalities: {dict(train.modality_counts)}; val modalities: {dict(val.modality_counts)}")
    for cid in withdrawn:
        report.lines.append(f"  class {cid} ({names[cid]}) WITHDRAWN by the dataset (cart gate): id reserved, head keeps {nc} outputs, no positives")
    return report


def effective_withdrawn(names: Sequence[str], train: SplitScan, declared: Sequence[int]) -> tuple[int, ...]:
    """Declared withdrawals plus a cart class the dataset kept by name but gave no train instance."""
    out = set(int(c) for c in declared)
    if len(names) == CART_ID + 1 and names[CART_ID] == "cart" and train.entries and train.instances.get(CART_ID, 0) == 0:
        out.add(CART_ID)
    return tuple(sorted(out))


def cart_gate_from_schema() -> int:
    try:
        return int(_read_yaml(C.SCHEMA_YAML)["cart_gate"]["min_verified_instances"])
    except (KeyError, TypeError, ValueError, UsageError):
        return 300


# --------------------------------------------------------------------------------------------
# The balanced epoch list
# --------------------------------------------------------------------------------------------


def repeat_count(entry: ImageEntry, rules: SamplingRules) -> int:
    """How many times an image enters the epoch list: 1, plus each applicable extra, capped."""
    copies = 1
    if entry.classes & rules.rare_classes:
        copies += rules.rare_repeat - 1
    if entry.modality == "lwir":
        copies += rules.lwir_repeat - 1
    return min(copies, rules.max_repeat)


def build_train_list(entries: Iterable[ImageEntry], rules: SamplingRules) -> list[str]:
    """Absolute image paths, duplicates allowed. Ultralytics reads this as the epoch's image list."""
    out: list[str] = []
    for entry in sorted(entries, key=lambda e: e.path):
        out.extend([entry.path] * repeat_count(entry, rules))
    return out


def subsample_entries(entries: Sequence[ImageEntry], fraction: float, seed: int) -> list[ImageEntry]:
    """A seeded random subset. Ultralytics' own `fraction` takes the first images alphabetically, which is a single source."""
    if fraction >= 1.0:
        return list(entries)
    k = max(1, round(len(entries) * fraction))
    return sorted(random.Random(seed).sample(list(entries), k), key=lambda e: e.path)


# --------------------------------------------------------------------------------------------
# Run directory, checkpoints, resume
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @property
    def weights(self) -> Path:
        return self.run_dir / "weights"

    @property
    def last(self) -> Path:
        return self.weights / "last.pt"

    @property
    def best(self) -> Path:
        return self.weights / "best.pt"

    @property
    def last_good(self) -> Path:
        return self.weights / "last_good.pt"

    @property
    def train_list(self) -> Path:
        return self.run_dir / "train_list.txt"

    @property
    def data_yaml(self) -> Path:
        return self.run_dir / "data_resolved.yaml"

    @property
    def log_csv(self) -> Path:
        return self.run_dir / "train_log.csv"

    @property
    def state_json(self) -> Path:
        return self.run_dir / "run_state.json"


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    valid: bool
    epoch: int | None
    epochs: int | None
    reason: str


def torch_load_checkpoint(path: Path) -> Any:
    """Load a full Ultralytics checkpoint. Needs ultralytics importable, because the file pickles its model classes."""
    import torch

    try:
        import ultralytics  # noqa: F401  (registers the classes the pickle refers to)
    except ImportError:
        pass
    return torch.load(str(path), map_location="cpu", weights_only=False)


def inspect_checkpoint(path: Path, loader: Callable[[Path], Any] = torch_load_checkpoint) -> CheckpointInfo:
    """Can `path` be resumed from? It must load, and it must still carry epoch, EMA, optimiser state and train_args."""
    path = Path(path)
    if not path.exists():
        return CheckpointInfo(path, False, None, None, "missing")
    if not zipfile.is_zipfile(path):
        return CheckpointInfo(path, False, None, None, f"not a complete torch archive ({path.stat().st_size} bytes; truncated?)")
    try:
        ckpt = loader(path)
    except Exception as exc:  # any unpickling or read failure means the file is unusable
        return CheckpointInfo(path, False, None, None, f"does not load: {type(exc).__name__}: {str(exc)[:100]}")
    if not isinstance(ckpt, dict):
        return CheckpointInfo(path, False, None, None, "not a checkpoint dict")
    epoch = ckpt.get("epoch")
    train_args = ckpt.get("train_args")
    epochs = train_args.get("epochs") if isinstance(train_args, dict) else None
    if not isinstance(epoch, int) or epoch < 0:
        return CheckpointInfo(path, False, epoch if isinstance(epoch, int) else None, epochs, "finished checkpoint (epoch -1, optimiser stripped); not resumable")
    if ckpt.get("ema") is None or ckpt.get("optimizer") is None:
        return CheckpointInfo(path, False, epoch, epochs, "no EMA or optimiser state; not resumable")
    if not isinstance(train_args, dict) or not isinstance(epochs, int):
        return CheckpointInfo(path, False, epoch, None, "no train_args")
    return CheckpointInfo(path, True, epoch, epochs, "ok")


def pick_resume_checkpoint(paths: RunPaths, loader: Callable[[Path], Any] = torch_load_checkpoint) -> tuple[CheckpointInfo | None, list[CheckpointInfo]]:
    """The newest resumable checkpoint among last.pt and last_good.pt, and what was found for each."""
    infos = [inspect_checkpoint(p, loader) for p in (paths.last, paths.last_good)]
    valid = [i for i in infos if i.valid]
    if not valid:
        return None, infos
    # Highest epoch wins; on a tie last.pt is preferred because it is the file Ultralytics itself wrote.
    best = max(valid, key=lambda i: (i.epoch, i.path == paths.last))
    return best, infos


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_copy(src: Path, dst: Path) -> None:
    """Copy src to dst so that dst is always either the old complete file or the new complete file."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    with tmp.open("rb+") as fh:
        os.fsync(fh.fileno())
    if not zipfile.is_zipfile(tmp):
        tmp.unlink()
        raise OSError(f"{src} is not a complete torch archive; refusing to make it the last good checkpoint")
    os.replace(tmp, dst)
    _fsync_dir(dst.parent)


def has_history(paths: RunPaths) -> bool:
    if any(p.exists() for p in (paths.last, paths.last_good, paths.best)):
        return True
    if paths.log_csv.exists() and read_log(paths.log_csv):
        return True
    return (C.read_json(paths.state_json, default={}) or {}).get("status") in ("complete", "session_stop")


@dataclass(frozen=True)
class Decision:
    kind: str  # "fresh" | "resume" | "complete"
    checkpoint: CheckpointInfo | None
    note: str


def _describe(infos: Sequence[CheckpointInfo]) -> str:
    return "; ".join(f"{i.path.name}: {i.reason}" for i in infos)


def decide_action(paths: RunPaths, resume: bool, auto_resume: bool, loader: Callable[[Path], Any] = torch_load_checkpoint) -> Decision:
    """Fresh start, resume, or nothing to do. Raises UsageError for a request that cannot be honoured."""
    if not (resume or auto_resume):
        if has_history(paths):
            raise UsageError(
                f"{paths.run_dir} already holds a run (checkpoints, a log or a finished marker). Pass --resume or --auto-resume to continue it, "
                f"or choose a different --run-dir or --name; starting fresh here would overwrite it."
            )
        return Decision("fresh", None, "fresh run")

    if (C.read_json(paths.state_json, default={}) or {}).get("status") == "complete":
        return Decision("complete", None, f"{paths.state_json.name} says this run is complete")

    chosen, infos = pick_resume_checkpoint(paths, loader)
    if chosen is None:
        if resume:
            raise UsageError(f"--resume needs a valid checkpoint in {paths.weights}, and there is none ({_describe(infos)}). Use --auto-resume to start fresh when none exists.")
        if any(i.reason != "missing" for i in infos):
            raise UsageError(
                f"--auto-resume found checkpoint files in {paths.weights} but none can be resumed ({_describe(infos)}). Refusing to start "
                f"over them; inspect the directory, or choose a different --run-dir."
            )
        if has_history(paths):
            raise UsageError(f"{paths.run_dir} has a best.pt or a log but no resumable checkpoint; choose a different --run-dir")
        return Decision("fresh", None, "no checkpoint found; starting fresh")

    if chosen.epoch + 1 >= chosen.epochs:
        return Decision("complete", chosen, f"{chosen.path.name} is at epoch {chosen.epoch + 1} of {chosen.epochs}: the run is finished")
    return Decision("resume", chosen, f"resuming from {chosen.path.name} (epoch {chosen.epoch + 1} of {chosen.epochs} done)")


# --------------------------------------------------------------------------------------------
# train_log.csv: appended across sessions, reconciled against the checkpoint on resume
# --------------------------------------------------------------------------------------------


def _cell(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return ""
    return str(int(number)) if isinstance(value, int) else f"{number:.6g}"


def read_log(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if row.get("epoch")]


def _log_header(path: Path) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh), [])


def append_log_row(path: Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    if not new_file and _log_header(path) != LOG_COLUMNS:
        # A log written by an older train.py (fewer columns): rewrite it under the current header first, so a
        # resumed run never appends rows whose cells sit under the wrong column names. Old cells are kept.
        _rewrite_log(path, read_log(path))
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        if new_file:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([_cell(row.get(c)) for c in LOG_COLUMNS])
        fh.flush()
        os.fsync(fh.fileno())


def _rewrite_log(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(LOG_COLUMNS)
        for row in rows:
            writer.writerow([_cell(row.get(c)) for c in LOG_COLUMNS])
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def prior_epoch_seconds(rows: Sequence[Mapping[str, str]], last_n: int = 3) -> float | None:
    """The slowest of the last `last_n` logged epoch durations, or None when none is logged."""
    values = [v for v in (_num(r.get("epoch_s")) for r in rows) if v is not None and v > 0]
    return max(values[-last_n:]) if values else None


def best_fitness_row(rows: Sequence[Mapping[str, str]], up_to_epoch: int) -> tuple[int, float] | None:
    """(epoch, fitness) of the best logged epoch <= up_to_epoch; the earliest one on a tie, as EarlyStopping keeps it."""
    best: tuple[int, float] | None = None
    for row in rows:
        epoch, fit = _num(row.get("epoch")), _num(row.get("fitness"))
        if epoch is None or fit is None or int(epoch) > up_to_epoch:
            continue
        if best is None or fit > best[1]:
            best = (int(epoch), fit)
    return best


def restore_early_stopping(stopper: Any, rows: Sequence[Mapping[str, str]], completed_epoch: int) -> tuple[int, float] | None:
    """Put the patience counter back where the killed session left it.

    Ultralytics creates a fresh EarlyStopping on every start, resumed or not (best_fitness 0, best_epoch
    0), so without this a run that resumes every session could never early-stop across sessions. Its
    epochs are 1-based, like the log's. Returns what was restored, or None when the log has no fitness.
    """
    best = best_fitness_row(rows, completed_epoch)
    if best is None or stopper is None:
        return None
    stopper.best_epoch, stopper.best_fitness = best
    patience = getattr(stopper, "patience", float("inf"))
    stopper.possible_stop = (completed_epoch - best[0]) >= (patience - 1)
    return best


def next_session_number(path: Path) -> int:
    sessions = []
    for row in read_log(path):
        try:
            sessions.append(int(float(row.get("session") or "")))
        except ValueError:
            continue
    return max(sessions, default=0) + 1


def backfill_row_from_checkpoint(ckpt: Mapping[str, Any]) -> dict[str, Any]:
    """A log row rebuilt from what the checkpoint itself stored. Anything it did not store stays blank, never zero."""
    epoch = int(ckpt["epoch"]) + 1
    metrics = ckpt.get("train_metrics") or {}
    results = ckpt.get("train_results") or {}

    def last(column: str) -> Any:
        values = results.get(column)
        return values[-1] if values else None

    same_epoch = last("epoch") is not None and int(float(last("epoch"))) == epoch
    pick = (lambda column: last(column)) if same_epoch else (lambda column: None)
    return {
        "epoch": epoch, "session": "", "elapsed_s": None, "epoch_s": None,
        "box_loss": pick("train/box_loss"), "cls_loss": pick("train/cls_loss"), "dfl_loss": pick("train/dfl_loss"),
        "precision": metrics.get("metrics/precision(B)"), "recall": metrics.get("metrics/recall(B)"),
        "map50": metrics.get("metrics/mAP50(B)"), "map50_95": metrics.get("metrics/mAP50-95(B)"),
        "person_ap50": None, "lr": pick("lr/pg1"), "fitness": metrics.get("fitness"),
    }


@dataclass(frozen=True)
class LogReconciliation:
    dropped: int
    backfilled: bool
    gaps: list[int]


def reconcile_log(path: Path, completed_epoch: int, backfill: Mapping[str, Any] | None) -> LogReconciliation:
    """Make train_log.csv agree with a checkpoint that finished `completed_epoch` (1-based) epochs.

    A kill can leave rows for epochs the checkpoint does not contain (they will be re-run, so keeping
    them would duplicate) or, in the instant between the checkpoint write and the row append, miss the
    epoch the checkpoint does contain. Rows past the checkpoint are dropped; a missing last epoch is
    rebuilt from the checkpoint's own stored metrics; any other hole is reported, never invented.
    """
    path = Path(path)
    rows = read_log(path)
    by_epoch: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            epoch = int(float(row["epoch"]))
        except ValueError:
            continue
        if epoch <= completed_epoch:
            by_epoch[epoch] = row  # a repeated epoch keeps its last row, the one consistent with the checkpoint lineage
    dropped = len(rows) - len(by_epoch)
    backfilled = False
    if completed_epoch >= 1 and completed_epoch not in by_epoch and backfill is not None:
        by_epoch[completed_epoch] = {k: v for k, v in backfill.items()}
        backfilled = True
    gaps = [e for e in range(1, completed_epoch + 1) if e not in by_epoch]
    if dropped or backfilled or (not path.exists() and by_epoch):
        _rewrite_log(path, [by_epoch[e] for e in sorted(by_epoch)])
    return LogReconciliation(dropped, backfilled, gaps)


# --------------------------------------------------------------------------------------------
# Training callbacks
# --------------------------------------------------------------------------------------------


@dataclass
class TrainState:
    paths: RunPaths
    session: int
    session_start: float
    max_hours: float
    unfreeze_epoch: int | None
    policy: AugmentPolicy
    expected_entries: int | None = None
    epoch_started: float = 0.0
    session_stopped: bool = False
    stage: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    prior_epoch_s: float | None = None                    # from earlier sessions' train_log.csv
    epoch_durations: list[float] = field(default_factory=list)  # this session's epochs, train + val
    ir_images: int | None = None                          # LWIR images in the epoch list, for the start-up report


def count_params(model: Any) -> tuple[int, int]:
    """(trainable, frozen) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def release_backbone(trainer: Any) -> int:
    """Set requires_grad=True on every parameter except `.dfl`, and reset the trainer's frozen-layer names.

    Ultralytics has no native unfreeze. Resetting `freeze_layer_names` matters: `_model_train` keeps
    every BatchNorm whose name matches it in eval mode, so leaving the old list would keep the
    unfrozen backbone's BatchNorm statistics frozen. Returns how many parameters changed, so a second
    call in the same epoch, or a call after a resume, is a harmless 0.
    """
    changed = 0
    for name, param in trainer.model.named_parameters():
        want = ".dfl" not in name
        if param.dtype.is_floating_point and param.requires_grad != want:
            param.requires_grad = want
            changed += 1
    trainer.freeze_layer_names = [".dfl"]
    return changed


def effective_albumentations(dataset: Any) -> list[str] | None:
    """The Albumentations transforms this dataset will really apply, or None when the package is not in use."""
    pipeline = getattr(getattr(dataset, "transforms", None), "transforms", None) or []
    for transform in pipeline:
        if type(transform).__name__ == "Albumentations":
            composed = getattr(transform, "transform", None)
            if composed is None:
                return None
            return [repr(t) for t in getattr(composed, "transforms", [])]
    return None


def assert_live_augmentation(dataset: Any, policy: AugmentPolicy) -> None:
    """The live train dataset must apply exactly the policy: both halves, or AugmentationViolation (exit 3).

    Ultralytics logs and swallows an Albumentations failure and trains on without it, so the effective
    transform list is read back from the dataset itself rather than trusted.
    """
    unapplied = []
    active = effective_albumentations(dataset)
    names = [re.split(r"[(\s]", t, maxsplit=1)[0] for t in (active or [])]
    if names != expected_albumentations(policy):
        unapplied += [f"always_on.{k}" for k in ALBUMENTATIONS_KEYS]
    if getattr(dataset, "ir_augment", None) is None or type(dataset).__name__ != "ModalityAwareYOLODataset":
        unapplied += [f"infrared_only.{k}" for k in IR_KEYS]
    if unapplied:
        raise AugmentationViolation(
            f"the live training dataset does not apply augment.yaml keys {unapplied} (Albumentations transforms in effect: {active}; "
            f"expected {expected_albumentations(policy)} and the LWIR hook)"
        )


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def class_ap50(trainer: Any) -> dict[int, float]:
    """Per-class AP50 from the last validation: {class id: AP50} for the classes val holds ground truth for.

    `box.ap50` is aligned with `box.ap_class_index` (Ultralytics 8.4.155 utils/metrics.py), so a class
    absent from val is simply missing here and is logged blank, never 0.
    """
    box = getattr(getattr(getattr(trainer, "validator", None), "metrics", None), "box", None)
    if box is None:
        return {}
    index = [int(i) for i in getattr(box, "ap_class_index", [])]
    ap50 = list(getattr(box, "ap50", []))
    out = {}
    for position, cid in enumerate(index):
        if position < len(ap50):
            value = _num(ap50[position])
            if value is not None:
                out[cid] = value
    return out


def person_ap50(trainer: Any) -> float | None:
    """Person (class 0) AP50 from the last validation, or None when val held no person (never 0)."""
    return class_ap50(trainer).get(0)


def should_stop_for_time(elapsed_s: float, budget_s: float, next_epoch_s: float | None) -> bool:
    """Stop now if the budget is spent, or if one more epoch (with the safety factor) would end past it."""
    if elapsed_s >= budget_s:
        return True
    return next_epoch_s is not None and elapsed_s + next_epoch_s * EPOCH_ESTIMATE_SAFETY > budget_s


def next_epoch_estimate(state: TrainState) -> float | None:
    """The slowest of this session's last three epochs, else the slowest recent epoch of earlier sessions."""
    recent = state.epoch_durations[-3:]
    return max(recent) if recent else state.prior_epoch_s


def weight_group_lr(trainer: Any) -> float | None:
    """The LR the weights see. Group 0 is the bias group, which is at a different rate during warmup."""
    groups = getattr(getattr(trainer, "optimizer", None), "param_groups", None) or []
    for group in groups:
        if group.get("param_group") == "weight":
            return _num(group.get("lr"))
    return _num(groups[0].get("lr")) if groups else None


def make_epoch_row(trainer: Any, state: TrainState, now: float) -> dict[str, Any]:
    losses = {}
    if getattr(trainer, "tloss", None) is not None:
        losses = {k.split("/", 1)[-1]: v for k, v in trainer.label_loss_items(trainer.tloss).items()}
    metrics = getattr(trainer, "metrics", None) or {}
    per_class = class_ap50(trainer)
    return {
        **{column: per_class.get(cid) for cid, column in enumerate(AP_COLUMNS)},
        "epoch": trainer.epoch + 1,
        "session": state.session,
        "elapsed_s": round(now - state.session_start, 1),
        "epoch_s": round(now - state.epoch_started, 1),
        "box_loss": _num(losses.get("box_loss")),
        "cls_loss": _num(losses.get("cls_loss")),
        "dfl_loss": _num(losses.get("dfl_loss")),
        "precision": _num(metrics.get("metrics/precision(B)")),
        "recall": _num(metrics.get("metrics/recall(B)")),
        "map50": _num(metrics.get("metrics/mAP50(B)")),
        "map50_95": _num(metrics.get("metrics/mAP50-95(B)")),
        "person_ap50": per_class.get(0),
        "lr": weight_group_lr(trainer),
        "fitness": _num(getattr(trainer, "fitness", None)),
    }


def write_run_state(
    state: TrainState,
    status: str,
    epochs_done: int | None = None,
    epochs_total: int | None = None,
    stopped_for_time: bool = False,
    early_stopped: bool = False,
    **extra: Any,
) -> None:
    """run_state.json. `complete` (bool) is what the notebook and smoke_test.py read; `status` is the readable form of it."""
    payload = {
        "schema": "truewatch.train_state.v1",
        "stage": state.stage,
        "status": status,
        "complete": status == "complete",
        "epochs_done": epochs_done,
        "epochs_total": epochs_total,
        "stopped_for_time": stopped_for_time,
        "early_stopped": early_stopped,
        "session": state.session,
        "updated": _now_iso(),
        "val_scope": VAL_SCOPE,
        "host": C.host_info(),
    }
    payload.update(state.extra)
    payload.update(extra)
    C.write_json(state.paths.run_dir / "run_state.json", payload)


def build_callbacks(state: TrainState) -> dict[str, Callable[[Any], None]]:
    """The Ultralytics callbacks. Each is idempotent per epoch so a resume at any epoch is correct."""

    def on_pretrain_routine_start(trainer: Any) -> None:
        assert_forbidden_augmentation(trainer.args, state.policy)

    def on_pretrain_routine_end(trainer: Any) -> None:
        assert_forbidden_augmentation(trainer.args, state.policy)
        if trainer.data.get("test"):
            raise DatasetError("the trainer's dataset has a test split; the test split is sealed until Phase 11")
        dataset = trainer.train_loader.dataset
        args = trainer.args
        say(f"augmentation: mosaic={args.mosaic} close_mosaic={args.close_mosaic} scale={args.scale} translate={args.translate} fliplr={args.fliplr} "
            f"hsv=({args.hsv_h}, {args.hsv_s}, {args.hsv_v}); zero: {', '.join(f'{k}={getattr(args, k)}' for k in ZERO_ARGS)}, degrees={args.degrees}")
        assert_live_augmentation(dataset, state.policy)
        say(f"augmentation: Albumentations on every sample: {effective_albumentations(dataset)}")
        say(f"augmentation: infrared_only block attached to the train dataset, applied per LWIR source image"
            + (f" ({state.ir_images} LWIR images in the epoch list)" if state.ir_images is not None else ""))
        entries = len(dataset)
        say(f"train dataset: {entries} entries per epoch, val dataset: {len(trainer.test_loader.dataset)} images")
        if state.expected_entries is not None and entries != state.expected_entries:
            warn(f"the train list has {state.expected_entries} entries but Ultralytics built {entries}; duplicates or invalid images were dropped")

    def on_train_start(trainer: Any) -> None:
        trainable, frozen = count_params(trainer.model)
        say(f"parameters at start: {trainable} trainable, {frozen} frozen; freeze={trainer.args.freeze}; unfreeze at epoch index {state.unfreeze_epoch}")
        say(f"session {state.session}: deadline {state.max_hours:g} h of wall clock from process start; stops at the epoch boundary where "
            f"one more epoch would cross it (estimate now: "
            + (f"{next_epoch_estimate(state):.0f} s)" if next_epoch_estimate(state) else "none until the first epoch ends)"))
        start_epoch = int(getattr(trainer, "start_epoch", 0) or 0)
        if start_epoch > 0:
            restored = restore_early_stopping(getattr(trainer, "stopper", None), read_log(state.paths.log_csv), start_epoch)
            if restored:
                stopper = trainer.stopper
                say(f"early stopping restored from train_log.csv: best fitness {restored[1]:.5g} at epoch {restored[0]}, "
                    f"{start_epoch - restored[0]} epoch(s) without improvement of patience {stopper.patience}")
            else:
                warn("resumed, but train_log.csv holds no fitness to restore early stopping from; patience restarts at 0")

    def on_train_epoch_start(trainer: Any) -> None:
        state.epoch_started = time.time()
        if state.unfreeze_epoch is not None and trainer.epoch >= state.unfreeze_epoch:
            changed = release_backbone(trainer)
            if changed:
                trainable, frozen = count_params(trainer.model)
                say(f"epoch index {trainer.epoch}: backbone released ({changed} parameter tensors); {trainable} trainable, {frozen} frozen")

    def on_model_save(trainer: Any) -> None:
        # Runs only after save_model() has finished writing last.pt, so the copy is of a complete file.
        try:
            atomic_copy(state.paths.last, state.paths.last_good)
        except OSError as exc:
            warn(f"last_good.pt was not updated after epoch {trainer.epoch + 1}: {exc}")
        now = time.time()
        append_log_row(state.paths.log_csv, make_epoch_row(trainer, state, now))
        if state.epoch_started > 0:
            state.epoch_durations.append(now - state.epoch_started)
        elapsed = now - state.session_start
        estimate = next_epoch_estimate(state)
        if trainer.epoch + 1 < trainer.epochs and not trainer.stop and should_stop_for_time(elapsed, state.max_hours * 3600.0, estimate):
            trainer.stop = True
            state.session_stopped = True
            say(f"session deadline: {elapsed / 3600.0:.2f} h used of {state.max_hours:g} h after epoch {trainer.epoch + 1}, and one more epoch "
                f"(~{(estimate or 0) / 3600.0:.2f} h) would not finish inside it; stopping cleanly. "
                f"Re-run the same command with --resume (or --auto-resume) in the next session.")

    def on_train_end(trainer: Any) -> None:
        done, total = trainer.epoch + 1, trainer.epochs
        if state.session_stopped:
            write_run_state(state, "session_stop", epochs_done=done, epochs_total=total, stopped_for_time=True)
        else:
            write_run_state(state, "complete", epochs_done=done, epochs_total=total, early_stopped=done < total)

    return {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_start": on_train_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_model_save": on_model_save,
        "on_train_end": on_train_end,
    }


# --------------------------------------------------------------------------------------------
# Command line and the run itself
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Exit codes: 0 done / already complete / stopped for session time; 2 usage error or missing checkpoint; 3 dataset preflight failure or forbidden augmentation.",
    )
    ap.add_argument("--stage", required=True, choices=STAGES, help="day: COCO weights on the full corpus. ir: continue from the day stage's best.pt with LWIR emphasis")
    ap.add_argument("--config", default=None, help="stage config (default: configs/yolo11s_<stage>.yaml)")
    ap.add_argument("--data-root", default=None, help="Phase 1 dataset root holding images/ and labels/ (default: $TRUEWATCH_DATA_ROOT, a Kaggle input mount, then datasets/processed/yolo)")
    ap.add_argument("--run-dir", default=None, help="where checkpoints, logs and the epoch list live (default: training/runs/<name or stage>)")
    ap.add_argument("--model", default=None, help="init weights for a fresh run (default: the config's `model` for day, <run-dir parent>/day/weights/best.pt for ir)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="continue from the newest valid checkpoint; exit 2 when there is none")
    group.add_argument("--auto-resume", action="store_true", help="resume when a checkpoint exists, otherwise start fresh (what the notebook uses)")
    ap.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS,
                    help="wall-clock deadline for this process, in hours from its start: training stops at the epoch boundary where one "
                         "more epoch would cross it (the notebook passes the time left in its 12 h session)")
    ap.add_argument("--epochs", type=int, default=None, help="override the config's epochs (ignored on resume: the checkpoint's value wins)")
    ap.add_argument("--batch", type=int, default=None, help="override the config's batch size")
    ap.add_argument("--imgsz", type=int, default=None, help="override the config's image size")
    ap.add_argument("--workers", type=int, default=None, help="override the config's dataloader workers")
    ap.add_argument("--device", default=None,
                    help="Ultralytics device, e.g. cpu or 0; a single device only, multi-GPU (DDP) is refused (default: Ultralytics picks)")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on a seeded random fraction of the train images (smoke runs); val is always whole")
    ap.add_argument("--unfreeze-epoch", type=int, default=None, help="override schedule.unfreeze_epoch (0-based epoch at which the backbone is released)")
    ap.add_argument("--name", default=None, help="run name; sets the default run dir")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the dataset preflight, write nothing, exit with the code a real run would")
    ap.add_argument("--skip-preflight", action="store_true", help="print preflight findings but do not abort on them (the labels are still read to build the epoch list)")
    return ap


def validate_cli(args: argparse.Namespace) -> None:
    if not args.max_hours > 0:
        raise UsageError("--max-hours must be greater than 0")
    if not 0 < args.fraction <= 1:
        raise UsageError("--fraction must be in (0, 1]")
    for flag in ("epochs", "batch", "imgsz"):
        value = getattr(args, flag)
        if value is not None and value < 1:
            raise UsageError(f"--{flag} must be a positive integer")
    if args.workers is not None and args.workers < 0:
        raise UsageError("--workers must be 0 or more")
    if args.unfreeze_epoch is not None and args.unfreeze_epoch < 0:
        raise UsageError("--unfreeze-epoch must be 0 or more")
    device = str(args.device or "")
    if "," in device or device.lower() in ("-1,-1",):
        raise UsageError(
            f"--device {device!r} names several GPUs. Ultralytics runs multi-GPU training in spawned child processes built from a "
            f"generated script, which carry neither this script's callbacks (freeze schedule, last_good.pt, train_log.csv, the session "
            f"deadline) nor its trainer (the infrared_only augmentation hook). Use a single device, e.g. --device 0."
        )


def resolve_init_weights(args: argparse.Namespace, cfg: StageConfig, paths: RunPaths, dry_run: bool) -> tuple[str, str]:
    """(path or name, description) of the weights a fresh run starts from."""
    if args.model:
        path = Path(args.model)
        if path.exists():
            return str(path.resolve()), f"--model {path}"
        if path.parent == Path(".") and ULTRALYTICS_ASSET_RE.fullmatch(path.name):
            # A bare release asset name (e.g. yolo11n.pt) on a fresh machine: fetch it like the config model.
            local = WEIGHTS_DIR / path.name
            if local.exists():
                return str(local), f"--model {path.name} (training/weights)"
            if dry_run:
                return str(local), f"--model {path.name} (not on disk; a real run would download it to {local})"
            return download_release_weights(path.name, local), f"--model {path.name} (downloaded)"
        raise UsageError(f"--model {path} does not exist")
    if cfg.stage == "ir":
        best = paths.run_dir.parent / "day" / "weights" / "best.pt"
        if not best.exists():
            raise UsageError(
                f"stage ir starts from the day stage's best weights, and {best} does not exist. Run --stage day first, or pass --model PATH."
            )
        day_state = C.read_json(best.parent.parent / "run_state.json", default={}) or {}
        if day_state.get("status") != "complete":
            warn(f"the day stage is not marked complete (status: {day_state.get('status', 'unknown')}); {best} is the best checkpoint so far")
        return str(best.resolve()), f"stage day best.pt ({best})"
    name = str(cfg.model)
    candidate = Path(name)
    if candidate.exists():
        return str(candidate.resolve()), f"config model {name}"
    local = WEIGHTS_DIR / candidate.name
    if local.exists():
        return str(local), f"config model {name} (training/weights)"
    if dry_run:
        return str(local), f"config model {name} (not on disk; a real run would download it to {local})"
    return download_release_weights(name, local), f"config model {name} (downloaded)"


# Ultralytics release asset names, e.g. yolo11n.pt, yolo11s.pt, yolov8m.pt: the only bare names fetched automatically.
ULTRALYTICS_ASSET_RE = re.compile(r"yolo(v?\d+)[nsmlx]\.pt")


def download_release_weights(name: str, local: Path) -> str:
    """Download an Ultralytics release asset to `local`; any failure is a usage error naming the manual fix."""
    say(f"{name} is not on disk; downloading it to {local} (needs internet)")
    from ultralytics.utils.downloads import attempt_download_asset

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        attempt_download_asset(str(local))
    except Exception as exc:
        raise UsageError(f"could not download {name}: {exc}. Place the file at {local} or pass --model PATH.") from exc
    if not local.exists():
        raise UsageError(f"{name} could not be fetched; place it at {local} or pass --model PATH")
    return str(local)


def ultralytics_args(train_args: Mapping[str, Any]) -> Any:
    """Ultralytics' own resolution of the arguments: rejects unknown keys and returns the namespace the trainer will hold."""
    from ultralytics.cfg import DEFAULT_CFG, get_cfg

    try:
        return get_cfg(DEFAULT_CFG, dict(train_args))
    except Exception as exc:  # Ultralytics raises SyntaxError, TypeError and ValueError for bad arguments
        raise UsageError(f"Ultralytics rejected the training arguments: {type(exc).__name__}: {exc}") from exc


def _ultralytics_version() -> str | None:
    try:
        import ultralytics

        return ultralytics.__version__
    except ImportError:
        return None


def run(args: argparse.Namespace) -> int:
    session_start = time.time()
    validate_cli(args)
    stage = args.stage
    cfg = load_stage_config(Path(args.config) if args.config else DEFAULT_CONFIGS[stage], stage)
    policy = load_augment_policy()
    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / (args.name or stage)
    run_dir = run_dir.expanduser().resolve()
    paths = RunPaths(run_dir)

    if args.unfreeze_epoch is not None and cfg.unfreeze_epoch is None:
        raise UsageError(f"--unfreeze-epoch was given but {cfg.source.name} freezes nothing")
    unfreeze_epoch = args.unfreeze_epoch if args.unfreeze_epoch is not None else cfg.unfreeze_epoch

    cli = {"epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz, "workers": args.workers}
    train_args = resolve_train_args(cfg, policy, cli)
    resolved = ultralytics_args(train_args)
    assert_forbidden_augmentation(resolved, policy)

    for line in augmentation_report(policy):
        say(line)
    albumentations_version = check_albumentations(policy)

    schema_names = C.load_class_names(None)
    data_cfg = load_data_yaml()
    assert_names_match_schema(data_cfg["names"], schema_names, str(DATA_YAML))

    decision = decide_action(paths, args.resume, args.auto_resume)
    if decision.kind == "complete":
        say(f"{decision.note}. Nothing to do.")
        return EXIT_OK
    say(f"run: {decision.note}")

    try:
        root = C.resolve_data_root(args.data_root)
    except SystemExit as exc:
        raise DatasetError(str(exc)) from exc
    declared_withdrawn = dataset_withdrawn_classes(C.load_class_names(root), schema_names, f"{root}/data.yaml")

    init_desc = "the checkpoint being resumed"
    init_path = None
    if decision.kind == "fresh":
        init_path, init_desc = resolve_init_weights(args, cfg, paths, args.dry_run)
    elif args.epochs is not None and decision.checkpoint and args.epochs != decision.checkpoint.epochs:
        warn(f"--epochs {args.epochs} is ignored on resume; the checkpoint's {decision.checkpoint.epochs} wins (Ultralytics restores it)")

    train_scan = scan_split(root, "train", len(schema_names))
    val_scan = scan_split(root, "val", len(schema_names))
    preflight = run_preflight(root, schema_names, train_scan, val_scan, root / "manifest.tsv", cart_gate_from_schema(), declared_withdrawn)
    withdrawn = effective_withdrawn(schema_names, train_scan, declared_withdrawn)
    for line in preflight.lines:
        say(line)
    for text in preflight.warnings:
        warn(text)
    if preflight.errors:
        for text in preflight.errors:
            print(f"preflight error: {text}", file=sys.stderr, flush=True)
        if not args.skip_preflight:
            raise DatasetError(f"{len(preflight.errors)} preflight failure(s); fix the dataset, or pass --skip-preflight to train anyway")
        warn(f"--skip-preflight: continuing past {len(preflight.errors)} preflight failure(s)")

    seed = int(train_args.get("seed", 42))
    entries = subsample_entries(train_scan.entries, args.fraction, seed)
    rules = cfg.sampling
    listing = build_train_list(entries, rules)
    if not listing:
        raise DatasetError("the balanced epoch list is empty")
    copies = Counter(repeat_count(e, rules) for e in entries)
    say(f"balanced epoch list ({stage}): {len(entries)} images -> {len(listing)} entries "
        f"({', '.join(f'{v} image(s) x{k}' for k, v in sorted(copies.items()))}); "
        f"day images {sum(1 for e in entries if e.modality == 'visible')}, IR images {sum(1 for e in entries if e.modality == 'lwir')}")
    effective = Counter()
    for e in entries:
        n = repeat_count(e, rules)
        for cid in e.classes:
            effective[cid] += n
    say("  effective images per class in the epoch list: " + ", ".join(f"{schema_names[c]}={effective.get(c, 0)}" for c in range(len(schema_names))))

    data_dict = resolved_data_dict(data_cfg, root, paths.train_list)
    assert_resolved_data(data_dict, schema_names)

    say("plan")
    say(f"  stage {stage}, config {cfg.source}, run dir {run_dir}")
    say(f"  init: {init_desc}")
    say(f"  data root {root}; resolved yaml {paths.data_yaml} (no test key)")
    say(f"  epochs {train_args['epochs']}, imgsz {train_args['imgsz']}, batch {train_args['batch']}, workers {train_args.get('workers')}, device {args.device or 'auto'}, "
        f"optimizer {train_args['optimizer']}, lr0 {train_args['lr0']}, lrf {train_args['lrf']}, cos_lr {train_args['cos_lr']}, patience {train_args['patience']}")
    say(f"  freeze {train_args.get('freeze')} until epoch index {unfreeze_epoch}; close_mosaic {train_args['close_mosaic']}; session deadline {args.max_hours:g} h")
    say(f"  Albumentations {albumentations_version}: {expected_albumentations(policy)} on every sample; infrared_only block on LWIR images")
    if withdrawn:
        say(f"  withdrawn class id(s) {list(withdrawn)} ({', '.join(schema_names[c] for c in withdrawn)}): reserved in the "
            f"{len(schema_names)}-output head, no positives, AP n/a (DATASET_SPEC 1.5)")
    say(f"  {VAL_SCOPE}")

    prior_epoch_s = prior_epoch_seconds(read_log(paths.log_csv)) if decision.kind == "resume" else None
    if prior_epoch_s is not None:
        say(f"  earlier sessions: slowest recent epoch {prior_epoch_s / 60.0:.1f} min")
        if prior_epoch_s * EPOCH_ESTIMATE_SAFETY > args.max_hours * 3600.0:
            say(f"not enough time: one epoch takes about {prior_epoch_s / 3600.0:.2f} h and the deadline is {args.max_hours:g} h away; "
                f"nothing trained in this session (exit 0). Resume in a session with more time left.")
            if not args.dry_run:
                note_state = TrainState(paths=paths, session=next_session_number(paths.log_csv), session_start=session_start,
                                        max_hours=args.max_hours, unfreeze_epoch=unfreeze_epoch, policy=policy, stage=stage)
                done = len(read_log(paths.log_csv))
                write_run_state(note_state, "session_stop", epochs_done=done, epochs_total=decision.checkpoint.epochs if decision.checkpoint else None,
                                stopped_for_time=True, note="insufficient session time for one epoch")
            return EXIT_OK

    if args.dry_run:
        say("dry run: nothing was written; the run dir was not created")
        return EXIT_OK

    paths.weights.mkdir(parents=True, exist_ok=True)
    C.write_json(paths.run_dir / "resolved_args.json", {"train": train_args, "cli": {k: v for k, v in vars(args).items()}})
    tmp = paths.train_list.with_name(paths.train_list.name + ".tmp")
    tmp.write_text("".join(f"{p}\n" for p in listing), encoding="utf-8")
    os.replace(tmp, paths.train_list)
    write_yaml_atomic(paths.data_yaml, data_dict, header="# resolved by train.py: absolute path, balanced train list, no test key\n")

    session = next_session_number(paths.log_csv)
    state = TrainState(
        paths=paths, session=session, session_start=session_start, max_hours=args.max_hours, unfreeze_epoch=unfreeze_epoch,
        policy=policy, expected_entries=len(listing), stage=stage, prior_epoch_s=prior_epoch_s,
        ir_images=sum(repeat_count(e, rules) for e in entries if e.modality == "lwir"),
        extra={
            "config": str(cfg.source), "config_sha256": C.sha256_file(cfg.source),
            "augment_policy": str(policy.source), "augment_sha256": C.sha256_file(policy.source),
            "augment_plan": dict(policy.plan), "albumentations": albumentations_version,
            "data_root": str(root), "train_images": len(entries), "train_list_entries": len(listing),
            "withdrawn_classes": [schema_names[c] for c in withdrawn],
            "ultralytics": _ultralytics_version(), "split": "val", "seed": seed,
        },
    )

    from ultralytics import YOLO

    common = dict(train_args)
    common.update(data=str(paths.data_yaml), project=str(run_dir.parent), name=run_dir.name, exist_ok=True, save_dir=str(run_dir))
    if args.device:
        common["device"] = args.device

    if decision.kind == "resume":
        checkpoint = decision.checkpoint
        assert checkpoint is not None
        if checkpoint.path != paths.last:
            atomic_copy(checkpoint.path, paths.last)
            say(f"repaired {paths.last.name} from {checkpoint.path.name}")
        ckpt = torch_load_checkpoint(paths.last)
        report = reconcile_log(paths.log_csv, int(ckpt["epoch"]) + 1, backfill_row_from_checkpoint(ckpt))
        say(f"train_log.csv: {report.dropped} row(s) past the checkpoint dropped, last epoch {'rebuilt from the checkpoint' if report.backfilled else 'present'}"
            + (f"; epochs {report.gaps} have no row (not invented)" if report.gaps else ""))
        drift = {k: (ckpt["train_args"].get(k), train_args.get(k)) for k in DRIFT_WATCH if k in train_args and ckpt["train_args"].get(k) != train_args.get(k)}
        if drift:
            warn("the checkpoint's saved settings differ from the current config and the checkpoint wins: "
                 + ", ".join(f"{k}: checkpoint {a} vs config {b}" for k, (a, b) in drift.items()))
        del ckpt
        model = YOLO(str(paths.last))
        kwargs = {k: common[k] for k in RESUME_OVERRIDABLE if k in common}
        kwargs.update(resume=True, data=common["data"], save_dir=common["save_dir"])
        if args.device:
            kwargs["device"] = args.device
    else:
        model = YOLO(str(init_path))
        kwargs = common

    write_run_state(state, "running", epochs_done=len(read_log(paths.log_csv)), epochs_total=int(train_args["epochs"]), action=decision.kind, init=init_desc)
    for event, callback in build_callbacks(state).items():
        model.add_callback(event, callback)

    say(f"starting session {session}")
    model.train(trainer=make_trainer_class(IRAugment(policy.ir)), **kwargs)

    rows = read_log(paths.log_csv)
    say(f"session {session} finished; train_log.csv has {len(rows)} epoch row(s); best {paths.best}, last {paths.last}")
    if state.session_stopped:
        say("stopped for session time: this is not an error (exit 0). Run the same command with --resume in the next session.")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return EXIT_USAGE
    except (DatasetError, AugmentationViolation) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return EXIT_DATASET


if __name__ == "__main__":
    sys.exit(main())
