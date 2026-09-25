"""Tests for train.py: the parts that decide whether a killed Kaggle session costs one epoch or the whole run.

Everything here is GPU-free and fast. The tests that need Ultralytics or torch skip cleanly without
them; the rest exercise pure functions (list building, preflight, checkpoint choice, log
reconciliation, the forbidden-augmentation assertion) on the synthetic fixture only. Nothing measured
on that fixture is a result.
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import _common as C
import train as T

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def entry(name: str, classes=(), modality: str = "visible") -> T.ImageEntry:
    return T.ImageEntry(f"/data/images/train/{name}", frozenset(classes), modality)


DAY_RULES = T.SamplingRules(frozenset({3, 4}), 2, 1, 2)
IR_RULES = T.SamplingRules(frozenset({3, 4}), 2, 2, 2)


def fake_checkpoint(path: Path, epoch: int = 1, epochs: int = 5, optimizer: bool = True) -> Path:
    """A small file with the keys Ultralytics' checkpoint has; torch.save writes the same zip container."""
    torch = pytest.importorskip("torch")
    payload = {
        "epoch": epoch,
        "ema": {"w": torch.zeros(2)},
        "optimizer": {"state": {}} if optimizer else None,
        "train_args": {"epochs": epochs},
        "train_metrics": {"metrics/mAP50(B)": 0.1, "fitness": 0.05},
        "train_results": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory) -> Path:
    import _synth

    root = tmp_path_factory.mktemp("synth")
    _synth.make_dataset(root, n_train=30, n_val=20, seed=1, size=(160, 128))
    return root


@pytest.fixture()
def dataset(tmp_path, synth_root) -> Path:
    """A private copy of the fixture, so tests that damage it do not affect each other."""
    import shutil

    dst = tmp_path / "data"
    shutil.copytree(synth_root, dst)
    return dst


# --------------------------------------------------------------------------------------------
# the balanced epoch list
# --------------------------------------------------------------------------------------------


def test_rare_class_images_appear_twice_and_others_once():
    entries = [entry("a_visible.jpg", [0]), entry("b_visible.jpg", [3]), entry("c_visible.jpg", [4, 0]), entry("d_visible.jpg")]
    listing = T.build_train_list(entries, DAY_RULES)
    counts = {Path(p).name: listing.count(p) for p in listing}
    assert counts == {"a_visible.jpg": 1, "b_visible.jpg": 2, "c_visible.jpg": 2, "d_visible.jpg": 1}


def test_stage_two_repeats_lwir_and_the_cap_is_two():
    entries = [
        entry("v_visible.jpg", [0], "visible"),
        entry("i_lwir.png", [0], "lwir"),
        entry("t_lwir.png", [3], "lwir"),          # rare AND lwir: 1 + 1 + 1 = 3, capped to 2
        entry("tv_visible.jpg", [3], "visible"),
    ]
    listing = T.build_train_list(entries, IR_RULES)
    counts = {Path(p).name: listing.count(p) for p in set(listing)}
    assert counts == {"v_visible.jpg": 1, "i_lwir.png": 2, "t_lwir.png": 2, "tv_visible.jpg": 2}
    assert max(counts.values()) <= 2


def test_stage_one_does_not_repeat_lwir():
    assert T.repeat_count(entry("i_lwir.png", [0], "lwir"), DAY_RULES) == 1


def test_no_image_exceeds_max_repeat_whatever_the_rules():
    rules = T.SamplingRules(frozenset({3}), 5, 5, 2)
    assert T.repeat_count(entry("x_lwir.png", [3], "lwir"), rules) == 2


def test_list_is_sorted_and_deterministic():
    entries = [entry("z_visible.jpg"), entry("a_visible.jpg", [3])]
    assert T.build_train_list(entries, DAY_RULES) == T.build_train_list(list(reversed(entries)), DAY_RULES)


def test_fraction_subsample_is_seeded_and_never_empty():
    entries = [entry(f"{i:03d}_visible.jpg") for i in range(100)]
    a = T.subsample_entries(entries, 0.2, seed=42)
    assert a == T.subsample_entries(entries, 0.2, seed=42)
    assert len(a) == 20
    assert a != T.subsample_entries(entries, 0.2, seed=7)
    assert len(T.subsample_entries(entries, 0.001, seed=1)) == 1
    assert T.subsample_entries(entries, 1.0, seed=1) == entries


# --------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stage,epochs,close,freeze", [("day", 20, 10, 10), ("ir", 6, 2, None)])
def test_shipped_configs_load_and_resolve(stage, epochs, close, freeze):
    cfg = T.load_stage_config(T.DEFAULT_CONFIGS[stage], stage)
    args = T.resolve_train_args(cfg, T.load_augment_policy())
    assert args["epochs"] == epochs
    assert args["close_mosaic"] == close
    assert args.get("freeze") == freeze
    assert args["seed"] == 42 and args["optimizer"] == "SGD" and args["deterministic"] is False
    names = [spec["transform"]["__class_fullname__"] for spec in args["augmentations"]]
    assert names == ["Downscale", "ImageCompression", "MotionBlur"]  # augment.yaml, replacing Ultralytics' built-in defaults
    assert args["imgsz"] == 640 and args["batch"] == 32 and args["nbs"] == 64


def test_stage_configs_carry_the_documented_differences():
    day = T.load_stage_config(T.DEFAULT_CONFIGS["day"], "day")
    ir = T.load_stage_config(T.DEFAULT_CONFIGS["ir"], "ir")
    assert (day.train["lr0"], day.train["lrf"], day.train["warmup_epochs"], day.train["patience"]) == (0.005, 0.01, 3, 6)
    assert (ir.train["lr0"], ir.train["lrf"], ir.train["warmup_epochs"], ir.train["patience"]) == (0.001, 0.1, 1, 3)
    assert day.sampling.lwir_repeat == 1 and ir.sampling.lwir_repeat == 2
    assert day.unfreeze_epoch == 3 and ir.unfreeze_epoch is None
    assert day.model == "yolo11s.pt" and ir.model is None


@pytest.mark.parametrize("stage", ["day", "ir"])
def test_every_train_argument_line_has_a_reason_comment(stage):
    lines = T.DEFAULT_CONFIGS[stage].read_text(encoding="utf-8").splitlines()
    start = lines.index("train:") if "train:" in lines else next(i for i, ln in enumerate(lines) if ln.startswith("train:"))
    missing = []
    for ln in lines[start + 1:]:
        if not ln.startswith("  ") or not ln.strip():
            break
        if "#" not in ln:
            missing.append(ln.strip())
    assert not missing, f"hyperparameter lines without a reason comment: {missing}"


def test_day_config_states_the_one_model_argument_at_the_top():
    head = "\n".join(T.DEFAULT_CONFIGS["day"].read_text(encoding="utf-8").splitlines()[:5])
    assert "ONE detector" in head and "slide 4" in head and "6-day" in head


def test_augmentation_keys_in_a_stage_config_are_rejected(tmp_path):
    text = T.DEFAULT_CONFIGS["day"].read_text(encoding="utf-8").replace("  epochs: 20", "  epochs: 20\n  mosaic: 1.0")
    bad = tmp_path / "yolo11s_day.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(T.UsageError, match="augment.yaml"):
        T.load_stage_config(bad, "day")


def test_config_for_the_wrong_stage_is_rejected():
    with pytest.raises(T.UsageError, match="stage"):
        T.load_stage_config(T.DEFAULT_CONFIGS["ir"], "day")


def test_freeze_without_unfreeze_epoch_is_rejected(tmp_path):
    text = T.DEFAULT_CONFIGS["day"].read_text(encoding="utf-8").replace("unfreeze_epoch: 3", "unfreeze_epoch: null")
    bad = tmp_path / "c.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(T.UsageError, match="never be released"):
        T.load_stage_config(bad, "day")


def test_close_mosaic_is_clamped_to_the_stage_length():
    assert T.effective_close_mosaic(10, 30) == 10
    assert T.effective_close_mosaic(10, 8) == 8
    assert T.effective_close_mosaic(3, 8) == 3
    assert T.effective_close_mosaic(0, 8) == 0


def test_close_mosaic_epochs_key_only_overrides_when_set():
    day = T.load_stage_config(T.DEFAULT_CONFIGS["day"], "day")
    ir = T.load_stage_config(T.DEFAULT_CONFIGS["ir"], "ir")
    assert day.close_mosaic_epochs is None and ir.close_mosaic_epochs == 2
    policy = T.load_augment_policy()
    assert policy.mosaic_close_epochs == 10
    assert T.resolve_train_args(day, policy)["close_mosaic"] == 10
    assert T.resolve_train_args(ir, policy)["close_mosaic"] == 2


def test_augmentation_values_come_from_augment_yaml():
    policy = T.load_augment_policy()
    args = T.resolve_train_args(T.load_stage_config(T.DEFAULT_CONFIGS["day"], "day"), policy)
    import yaml

    always = yaml.safe_load(C.AUGMENT_YAML.read_text(encoding="utf-8"))["always_on"]
    for key in ("mosaic", "scale", "translate", "fliplr", "hsv_h", "hsv_s", "hsv_v"):
        assert args[key] == always[key]


# --------------------------------------------------------------------------------------------
# the forbidden-augmentation assertion
# --------------------------------------------------------------------------------------------


def clean_args(**changes):
    base = dict(flipud=0.0, perspective=0.0, mixup=0.0, copy_paste=0.0, shear=0.0, cutmix=0.0, bgr=0.0, degrees=0.0)
    base.update(changes)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("key,value", [("flipud", 0.5), ("perspective", 0.0005), ("mixup", 0.1), ("copy_paste", 0.1),
                                       ("shear", 2.0), ("cutmix", 0.2), ("bgr", 0.1), ("degrees", 15.0), ("degrees", -11.0)])
def test_forbidden_augmentation_trips(key, value):
    with pytest.raises(T.AugmentationViolation, match=key):
        T.assert_forbidden_augmentation(clean_args(**{key: value}), T.load_augment_policy())


def test_small_rotation_within_the_hard_max_is_allowed():
    T.assert_forbidden_augmentation(clean_args(degrees=10.0), T.load_augment_policy())
    T.assert_forbidden_augmentation(clean_args(), T.load_augment_policy())


def test_a_config_that_enables_a_forbidden_augmentation_reaches_the_assertion():
    cfg = T.load_stage_config(T.DEFAULT_CONFIGS["day"], "day")
    bad = T.StageConfig(**{**cfg.__dict__, "train": {**cfg.train, "mixup": 0.3}})
    args = T.resolve_train_args(bad, T.load_augment_policy())
    assert args["mixup"] == 0.3                      # not masked by the defaults train.py fills in
    with pytest.raises(T.AugmentationViolation, match="mixup"):
        T.assert_forbidden_augmentation(args, T.load_augment_policy())


def test_a_policy_file_with_a_non_zero_forbidden_value_is_refused(tmp_path):
    import yaml

    raw = yaml.safe_load(C.AUGMENT_YAML.read_text(encoding="utf-8"))
    raw["forbidden"]["flipud"] = 0.5
    bad = tmp_path / "augment.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(T.AugmentationViolation):
        T.load_augment_policy(bad)


def test_ultralytics_resolution_of_our_arguments_passes_the_assertion():
    pytest.importorskip("ultralytics")
    policy = T.load_augment_policy()
    args = T.resolve_train_args(T.load_stage_config(T.DEFAULT_CONFIGS["day"], "day"), policy)
    resolved = T.ultralytics_args(args)
    T.assert_forbidden_augmentation(resolved, policy)
    assert resolved.close_mosaic == 10 and resolved.augmentations == args["augmentations"] == list(policy.albumentations)


# --------------------------------------------------------------------------------------------
# augment.yaml: every key applied or reported
# --------------------------------------------------------------------------------------------


def raw_policy() -> dict:
    import yaml

    return yaml.safe_load(C.AUGMENT_YAML.read_text(encoding="utf-8"))


def test_every_policy_key_has_a_route_and_the_plan_lists_them_all():
    raw = raw_policy()
    policy = T.load_augment_policy()
    expected = {f"{block}.{key}" for block, keys in raw.items() for key in keys}
    assert set(policy.plan) == expected
    for key in ("always_on.downscale_upscale", "always_on.jpeg", "always_on.motion_blur"):
        assert "Albumentations" in policy.plan[key]
    for key in ("infrared_only.clahe", "infrared_only.gaussian_noise", "infrared_only.brightness_contrast", "infrared_only.thermal_washout"):
        assert "LWIR" in policy.plan[key]
    assert "not applicable" in policy.plan["forbidden.erase_max_box_fraction"]
    lines = "\n".join(T.augmentation_report(policy))
    assert all(key in lines for key in expected)


@pytest.mark.parametrize("block,key", [("always_on", "gaussian_blur"), ("infrared_only", "polarity_inversion"), ("brand_new_block", "x")])
def test_an_unapplied_policy_key_stops_the_run_and_is_named(tmp_path, block, key):
    import yaml

    raw = raw_policy()
    raw.setdefault(block, {})[key] = {"p": 0.5}
    bad = tmp_path / "augment.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(T.AugmentationViolation, match=key if block != "brand_new_block" else block):
        T.load_augment_policy(bad)


@pytest.mark.parametrize("key", ["channel_shuffle", "false_colour"])
def test_a_non_zero_channel_shuffle_or_false_colour_is_refused(tmp_path, key):
    import yaml

    raw = raw_policy()
    raw["forbidden"][key] = 0.2
    bad = tmp_path / "augment.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(T.AugmentationViolation, match=key):
        T.load_augment_policy(bad)


def test_albumentations_specs_carry_the_policy_values():
    always = raw_policy()["always_on"]
    down, jpeg, blur = (s["transform"] for s in T.load_augment_policy().albumentations)
    d, j, m = always["downscale_upscale"], always["jpeg"], always["motion_blur"]
    assert down["__class_fullname__"] == "Downscale" and down["p"] == d["p"]
    assert down["scale_range"] == [1 / d["factor_max"], 1 / d["factor_min"]] == [0.25, 0.5]
    assert jpeg["__class_fullname__"] == "ImageCompression" and jpeg["p"] == j["p"] and jpeg["quality_range"] == [j["quality_min"], j["quality_max"]]
    assert blur["__class_fullname__"] == "MotionBlur" and blur["p"] == m["p"] and blur["blur_limit"] == [m["kernel_min"], m["kernel_max"]]
    assert blur["angle_range"] == [0.0, 360.0]          # DATASET_SPEC 4.1: random angle


def test_interpolation_constants_are_the_cv2_values():
    cv2 = pytest.importorskip("cv2")
    assert (T.CV2_INTER_AREA, T.CV2_INTER_LINEAR) == (cv2.INTER_AREA, cv2.INTER_LINEAR)
    assert T.load_augment_policy().albumentations[0]["transform"]["interpolation_pair"] == {"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}


def test_albumentations_rebuilds_the_policy_and_replaces_the_ultralytics_defaults():
    pytest.importorskip("ultralytics")
    pytest.importorskip("albumentations")
    from ultralytics.data.augment import Albumentations

    policy = T.load_augment_policy()
    assert T.check_albumentations(policy)
    default = Albumentations(p=1.0, transforms=None)
    assert len(default.transform.transforms) == 7                                    # what Ultralytics would apply by itself
    ours = Albumentations(p=1.0, transforms=[dict(s) for s in policy.albumentations])  # the serialised form train.py passes
    assert [type(x).__name__ for x in ours.transform.transforms] == T.expected_albumentations(policy)
    dataset = SimpleNamespace(transforms=SimpleNamespace(transforms=[ours]), ir_augment=object())
    assert [r.split("(")[0] for r in T.effective_albumentations(dataset)] == ["Downscale", "ImageCompression", "MotionBlur"]


def test_the_policy_transforms_really_change_pixels_when_forced_on():
    pytest.importorskip("albumentations")
    import albumentations as A
    import numpy as np

    img = np.random.default_rng(0).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    for spec in T.load_augment_policy().albumentations:
        forced = {"transform": {**spec["transform"], "p": 1.0}}
        out = A.from_dict(forced)(image=img)["image"]
        assert out.shape == img.shape and not np.array_equal(out, img), spec["transform"]["__class_fullname__"]


def test_missing_albumentations_fails_loudly_listing_the_unapplied_keys(monkeypatch):
    monkeypatch.setitem(sys.modules, "albumentations", None)   # makes `import albumentations` raise ImportError
    with pytest.raises(T.AugmentationViolation) as info:
        T.check_albumentations(T.load_augment_policy())
    for key in ("always_on.downscale_upscale", "always_on.jpeg", "always_on.motion_blur"):
        assert key in str(info.value)


def test_a_live_dataset_without_the_transforms_or_the_ir_hook_is_refused():
    policy = T.load_augment_policy()
    silent = SimpleNamespace(transforms=SimpleNamespace(transforms=[]))              # Ultralytics swallowed a failure
    with pytest.raises(T.AugmentationViolation, match="always_on.jpeg") as info:
        T.assert_live_augmentation(silent, policy)
    assert "infrared_only.clahe" in str(info.value)


def test_effective_albumentations_is_none_without_the_stage():
    assert T.effective_albumentations(SimpleNamespace(transforms=SimpleNamespace(transforms=[]))) is None


# --------------------------------------------------------------------------------------------
# the infrared-only block
# --------------------------------------------------------------------------------------------


def grey_bgr(seed: int = 0, shape=(64, 80)):
    import numpy as np

    g = np.random.default_rng(seed).integers(40, 200, shape, dtype=np.uint8)
    return np.repeat(g[:, :, None], 3, axis=2)


def forced_ir(**changes) -> T.IRAugment:
    spec = T.load_augment_policy().ir
    fields = {"clahe_p": 1.0, "noise_p": 1.0, "bc_p": 1.0, "washout_p": 1.0}
    fields.update(changes)
    return T.IRAugment(T.IRAugmentSpec(**{**spec.__dict__, **fields}))


def test_ir_spec_is_read_from_the_policy():
    spec = T.load_augment_policy().ir
    ir = raw_policy()["infrared_only"]
    assert spec.clahe_p == ir["clahe"]["p"] and spec.clahe_clip == (ir["clahe"]["clip_min"], ir["clahe"]["clip_max"])
    assert spec.clahe_grid == ir["clahe"]["tile_grid"] and spec.noise_sigma == (ir["gaussian_noise"]["sigma_min"], ir["gaussian_noise"]["sigma_max"])
    assert (spec.brightness, spec.contrast) == (ir["brightness_contrast"]["brightness"], ir["brightness_contrast"]["contrast"])
    assert spec.washout_p == ir["thermal_washout"]["p"] and spec.washout_range == (ir["thermal_washout"]["range_min"], ir["thermal_washout"]["range_max"])


def test_ir_augment_changes_the_frame_and_keeps_b_equal_g_equal_r():
    import random

    import numpy as np

    img = grey_bgr()
    out = forced_ir()(img, random.Random(3))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])   # DATASET_SPEC 3.4 invariant
    assert not np.array_equal(out, img)
    assert np.array_equal(forced_ir()(img, random.Random(3)), out)                                # seeded: reproducible
    assert img.max() < 200                                                                         # the input is never modified in place


def test_ir_augment_with_every_probability_zero_is_the_identity():
    import random

    import numpy as np

    img = grey_bgr(1)
    assert np.array_equal(forced_ir(clahe_p=0.0, noise_p=0.0, bc_p=0.0, washout_p=0.0)(img, random.Random(0)), img)


def test_thermal_washout_compresses_the_range_around_the_mean():
    import numpy as np

    g = np.array([[0, 100, 200]], dtype=np.uint8)
    out = T.thermal_washout(g, 0.5)
    assert out.tolist() == [[50, 100, 150]]
    assert np.ptp(T.thermal_washout(grey_bgr()[..., 0], 0.4)) < np.ptp(grey_bgr()[..., 0])


def test_ir_augment_pickles_for_dataloader_workers():
    import pickle
    import random

    import numpy as np

    aug = forced_ir()
    clone = pickle.loads(pickle.dumps(aug))
    img = grey_bgr(2)
    assert np.array_equal(clone(img, random.Random(9)), aug(img, random.Random(9)))


def test_the_dataset_hook_touches_lwir_images_only_and_pickles(synth_root):
    pytest.importorskip("ultralytics")
    import pickle

    import numpy as np
    from ultralytics.cfg import get_cfg

    from ultralytics.data.dataset import YOLODataset

    hyp = get_cfg(overrides={"mosaic": 0.0})
    names = dict(enumerate(C.load_class_names(None)))
    dataset = YOLODataset(img_path=str(synth_root / "images" / "train"), imgsz=160, augment=True, hyp=hyp,
                          data={"names": names, "nc": 5, "channels": 3}, task="detect")
    plain = [dataset.get_image_and_label(i)["img"].copy() for i in range(len(dataset))]
    dataset.__class__ = T.modality_dataset_class()
    dataset.ir_augment = forced_ir()
    touched = {"lwir": 0, "visible": 0}
    for i in range(len(dataset)):
        label = dataset.get_image_and_label(i)
        modality = C.modality_of_stem(Path(label["im_file"]).stem)
        changed = not np.array_equal(label["img"], plain[i])
        touched[modality] += changed
        if modality == "visible":
            assert not changed
        else:
            img = label["img"]
            assert np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2])
    assert touched["lwir"] > 0 and touched["visible"] == 0
    dataset.augment = False                                               # validation-style datasets are never touched
    assert all(np.array_equal(dataset.get_image_and_label(i)["img"], plain[i]) for i in range(len(dataset)))
    clone = pickle.loads(pickle.dumps(dataset))                          # what a spawned dataloader worker receives
    assert type(clone).__name__ == "ModalityAwareYOLODataset" and clone.ir_augment is not None
    assert T.ModalityAwareYOLODataset is T.modality_dataset_class()


def test_the_trainer_class_attaches_the_hook_to_the_train_dataset_only():
    pytest.importorskip("ultralytics")
    from ultralytics.models.yolo.detect import DetectionTrainer

    cls = T.make_trainer_class(forced_ir())
    assert issubclass(cls, DetectionTrainer) and "build_dataset" in cls.__dict__


# --------------------------------------------------------------------------------------------
# freeze and unfreeze
# --------------------------------------------------------------------------------------------


def frozen_trainer(freeze: int = 10):
    """A trainer-shaped object around a real 5-class YOLO11-n, frozen the way BaseTrainer._setup_train does it."""
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel("yolo11n.yaml", nc=5, verbose=False)
    names = [f"model.{i}." for i in range(freeze)] + [".dfl"]
    for name, param in model.named_parameters():
        param.requires_grad = not any(x in name for x in names)
    return SimpleNamespace(model=model, freeze_layer_names=names, epoch=0)


def test_release_backbone_counts_and_resets_the_frozen_names():
    trainer = frozen_trainer()
    before = T.count_params(trainer.model)
    assert before[1] > 0 and before[0] > 0
    changed = T.release_backbone(trainer)
    trainable, frozen = T.count_params(trainer.model)
    assert changed > 0
    assert trainable == before[0] + before[1] - frozen
    dfl = sum(p.numel() for n, p in trainer.model.named_parameters() if ".dfl" in n)
    assert frozen == dfl > 0                       # only the DFL projection stays frozen
    assert all(p.requires_grad for n, p in trainer.model.named_parameters() if ".dfl" not in n)
    assert trainer.freeze_layer_names == [".dfl"]  # otherwise _model_train keeps the backbone's BatchNorm in eval mode
    assert T.release_backbone(trainer) == 0        # idempotent


def test_release_lets_the_backbone_batchnorm_train_again():
    pytest.importorskip("ultralytics")
    from ultralytics.engine.trainer import BaseTrainer

    trainer = frozen_trainer()
    bn = [m for n, m in trainer.model.named_modules() if n.startswith("model.3.") and isinstance(m, __import__("torch").nn.BatchNorm2d)]
    assert bn
    BaseTrainer._model_train(trainer)
    assert all(not m.training for m in bn)         # frozen: statistics held
    T.release_backbone(trainer)
    BaseTrainer._model_train(trainer)
    assert all(m.training for m in bn)             # released: statistics update again


def make_state(tmp_path: Path, unfreeze_epoch: int | None = 3, max_hours: float = 9.5) -> T.TrainState:
    return T.TrainState(
        paths=T.RunPaths(tmp_path / "run"), session=1, session_start=0.0, max_hours=max_hours,
        unfreeze_epoch=unfreeze_epoch, policy=T.load_augment_policy(), stage="day",
    )


@pytest.mark.parametrize("epoch,released", [(0, False), (2, False), (3, True), (7, True)])
def test_unfreeze_callback_is_idempotent_by_epoch(tmp_path, epoch, released):
    trainer = frozen_trainer()
    trainer.epoch = epoch
    callback = T.build_callbacks(make_state(tmp_path))["on_train_epoch_start"]
    callback(trainer)
    assert (trainer.freeze_layer_names == [".dfl"]) is released
    trainable_first = T.count_params(trainer.model)
    callback(trainer)                               # a second call in the same epoch changes nothing
    assert T.count_params(trainer.model) == trainable_first


def test_a_resume_after_the_unfreeze_epoch_releases_a_freshly_frozen_model(tmp_path):
    trainer = frozen_trainer()                      # Ultralytics re-freezes layers 0-9 at every start, resumed or not
    trainer.epoch = 12                              # the resumed run's first epoch
    T.build_callbacks(make_state(tmp_path))["on_train_epoch_start"](trainer)
    assert T.count_params(trainer.model)[1] == sum(p.numel() for n, p in trainer.model.named_parameters() if ".dfl" in n)


def test_a_stage_with_no_unfreeze_epoch_never_touches_the_model(tmp_path):
    trainer = frozen_trainer(freeze=0)
    before = T.count_params(trainer.model)
    trainer.epoch = 50
    T.build_callbacks(make_state(tmp_path, unfreeze_epoch=None))["on_train_epoch_start"](trainer)
    assert T.count_params(trainer.model) == before


# --------------------------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------------------------


def test_picker_takes_the_newest_valid_checkpoint(tmp_path):
    paths = T.RunPaths(tmp_path)
    fake_checkpoint(paths.last, epoch=4)
    fake_checkpoint(paths.last_good, epoch=3)
    chosen, _ = T.pick_resume_checkpoint(paths)
    assert chosen.path == paths.last and chosen.epoch == 4


def test_picker_falls_back_to_last_good_when_last_is_truncated(tmp_path):
    paths = T.RunPaths(tmp_path)
    fake_checkpoint(paths.last, epoch=4)
    fake_checkpoint(paths.last_good, epoch=3)
    raw = paths.last.read_bytes()
    paths.last.write_bytes(raw[: len(raw) // 2])
    chosen, infos = T.pick_resume_checkpoint(paths)
    assert chosen.path == paths.last_good and chosen.epoch == 3
    assert "truncated" in next(i for i in infos if i.path == paths.last).reason


def test_picker_rejects_a_finished_checkpoint_whose_optimiser_was_stripped(tmp_path):
    # Ultralytics strips last.pt at the end of a run (epoch -1, no optimiser); that file cannot be resumed.
    paths = T.RunPaths(tmp_path)
    fake_checkpoint(paths.last, epoch=-1, optimizer=False)
    fake_checkpoint(paths.last_good, epoch=6, epochs=8)
    chosen, infos = T.pick_resume_checkpoint(paths)
    assert chosen.path == paths.last_good
    assert "not resumable" in next(i for i in infos if i.path == paths.last).reason


def test_picker_returns_none_when_nothing_loads(tmp_path):
    paths = T.RunPaths(tmp_path)
    paths.weights.mkdir(parents=True)
    paths.last.write_bytes(b"not a checkpoint")
    chosen, infos = T.pick_resume_checkpoint(paths)
    assert chosen is None and {i.path.name for i in infos} == {"last.pt", "last_good.pt"}


def test_a_corrupt_zip_that_is_not_a_checkpoint_is_invalid(tmp_path):
    bad = tmp_path / "x.pt"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("data.pkl", b"garbage")
    bad.write_bytes(buffer.getvalue())
    assert not T.inspect_checkpoint(bad).valid


def test_atomic_copy_replaces_only_with_a_complete_file(tmp_path):
    good = fake_checkpoint(tmp_path / "last.pt", epoch=2)
    dst = tmp_path / "last_good.pt"
    T.atomic_copy(good, dst)
    assert dst.read_bytes() == good.read_bytes() and not (tmp_path / "last_good.pt.tmp").exists()
    torn = tmp_path / "torn.pt"
    torn.write_bytes(good.read_bytes()[:100])
    with pytest.raises(OSError):
        T.atomic_copy(torn, dst)
    assert dst.read_bytes() == good.read_bytes()   # the previous good copy survives a refused overwrite
    assert not (tmp_path / "last_good.pt.tmp").exists()


def test_decide_action_fresh_resume_complete_and_errors(tmp_path):
    paths = T.RunPaths(tmp_path / "run")
    assert T.decide_action(paths, False, False).kind == "fresh"
    assert T.decide_action(paths, False, True).kind == "fresh"                 # --auto-resume with nothing to resume
    with pytest.raises(T.UsageError, match="--resume needs a valid checkpoint"):
        T.decide_action(paths, True, False)

    fake_checkpoint(paths.last, epoch=2, epochs=5)
    assert T.decide_action(paths, True, False).kind == "resume"
    assert T.decide_action(paths, False, True).kind == "resume"
    with pytest.raises(T.UsageError, match="already holds a run"):
        T.decide_action(paths, False, False)                                   # never silently overwrite a run

    fake_checkpoint(paths.last, epoch=4, epochs=5)                             # epoch 5 of 5 done
    fake_checkpoint(paths.last_good, epoch=4, epochs=5)
    decision = T.decide_action(paths, True, False)
    assert decision.kind == "complete" and "finished" in decision.note


def test_auto_resume_refuses_to_start_over_unreadable_checkpoints(tmp_path):
    paths = T.RunPaths(tmp_path / "run")
    paths.weights.mkdir(parents=True)
    paths.last.write_bytes(b"torn")
    with pytest.raises(T.UsageError, match="none can be resumed"):
        T.decide_action(paths, False, True)


def test_a_run_marked_complete_is_not_resumed(tmp_path):
    paths = T.RunPaths(tmp_path / "run")
    fake_checkpoint(paths.last_good, epoch=2, epochs=30)                       # early-stopped: epochs remain but the run is over
    C.write_json(paths.state_json, {"status": "complete", "complete": True})
    assert T.decide_action(paths, False, True).kind == "complete"


# --------------------------------------------------------------------------------------------
# train_log.csv
# --------------------------------------------------------------------------------------------


def log_row(epoch: int, session: int = 1, **extra) -> dict:
    return {"epoch": epoch, "session": session, "elapsed_s": 1.0, "epoch_s": 1.0, "map50": 0.1, **extra}


def epochs_of(path: Path) -> list[int]:
    return [int(r["epoch"]) for r in T.read_log(path)]


def test_log_appends_across_sessions_with_the_documented_columns(tmp_path):
    path = tmp_path / "train_log.csv"
    T.append_log_row(path, log_row(1))
    T.append_log_row(path, log_row(2))
    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        "epoch,session,elapsed_s,epoch_s,box_loss,cls_loss,dfl_loss,precision,recall,map50,map50_95,person_ap50,lr,fitness,"
        "ap50_person,ap50_two_wheeler,ap50_car,ap50_truck,ap50_cart"
    )
    assert T.next_session_number(path) == 2
    T.append_log_row(path, log_row(3, session=2))
    assert epochs_of(path) == [1, 2, 3] and T.next_session_number(path) == 3


def test_missing_metrics_are_blank_not_zero(tmp_path):
    path = tmp_path / "train_log.csv"
    T.append_log_row(path, log_row(1, person_ap50=None, precision=float("nan")))
    row = T.read_log(path)[0]
    assert row["person_ap50"] == "" and row["precision"] == ""


def test_reconcile_drops_rows_past_the_checkpoint_so_no_epoch_is_duplicated(tmp_path):
    path = tmp_path / "train_log.csv"
    for e in (1, 2, 3):
        T.append_log_row(path, log_row(e))
    report = T.reconcile_log(path, completed_epoch=2, backfill=None)
    assert epochs_of(path) == [1, 2] and report.dropped == 1 and not report.backfilled and report.gaps == []
    T.append_log_row(path, log_row(3, session=2))                              # the resumed run re-does epoch 3
    assert epochs_of(path) == [1, 2, 3]


def test_reconcile_rebuilds_an_epoch_the_kill_lost_from_the_checkpoint(tmp_path):
    path = tmp_path / "train_log.csv"
    T.append_log_row(path, log_row(1))
    ckpt = {"epoch": 1, "train_metrics": {"metrics/mAP50(B)": 0.25, "fitness": 0.1}, "train_results": {}}
    report = T.reconcile_log(path, completed_epoch=2, backfill=T.backfill_row_from_checkpoint(ckpt))
    assert epochs_of(path) == [1, 2] and report.backfilled
    rebuilt = T.read_log(path)[1]
    assert rebuilt["map50"] == "0.25" and rebuilt["box_loss"] == "" and rebuilt["person_ap50"] == ""


def test_reconcile_reports_a_hole_instead_of_inventing_a_row(tmp_path):
    path = tmp_path / "train_log.csv"
    T.append_log_row(path, log_row(1))
    T.append_log_row(path, log_row(4))
    report = T.reconcile_log(path, completed_epoch=4, backfill=None)
    assert report.gaps == [2, 3] and epochs_of(path) == [1, 4]


def test_reconcile_keeps_the_last_row_of_a_repeated_epoch(tmp_path):
    path = tmp_path / "train_log.csv"
    T.append_log_row(path, log_row(1, map50=0.1))
    T.append_log_row(path, log_row(2, map50=0.2))
    T.append_log_row(path, log_row(2, session=2, map50=0.3))
    T.reconcile_log(path, completed_epoch=2, backfill=None)
    assert [(r["epoch"], r["map50"]) for r in T.read_log(path)] == [("1", "0.1"), ("2", "0.3")]


# --------------------------------------------------------------------------------------------
# epoch rows, person AP, the session budget
# --------------------------------------------------------------------------------------------


class FakeTrainer(SimpleNamespace):
    def label_loss_items(self, loss_items=None, prefix="train"):
        return {f"{prefix}/{k}": round(float(v), 5) for k, v in loss_items.items()}


def fake_trainer(tmp_path: Path, epoch: int = 0, epochs: int = 10, ap_index=(0, 1), ap50=(0.5, 0.25)) -> FakeTrainer:
    box = SimpleNamespace(ap_class_index=list(ap_index), ap50=list(ap50))
    return FakeTrainer(
        epoch=epoch, epochs=epochs, stop=False,
        tloss={"box_loss": 1.5, "cls_loss": 2.5, "dfl_loss": 1.0},
        metrics={"metrics/precision(B)": 0.3, "metrics/recall(B)": 0.4, "metrics/mAP50(B)": 0.35, "metrics/mAP50-95(B)": 0.2},
        fitness=0.2,
        validator=SimpleNamespace(metrics=SimpleNamespace(box=box)),
        optimizer=SimpleNamespace(param_groups=[{"param_group": "bias", "lr": 0.1}, {"param_group": "weight", "lr": 0.004}]),
    )


def test_epoch_row_carries_person_ap_and_the_weight_group_lr(tmp_path):
    row = T.make_epoch_row(fake_trainer(tmp_path, epoch=2), make_state(tmp_path), now=100.0)
    assert row["epoch"] == 3 and row["person_ap50"] == 0.5 and row["lr"] == 0.004
    assert row["box_loss"] == 1.5 and row["fitness"] == 0.2 and row["map50_95"] == 0.2


def test_person_ap_is_none_not_zero_when_val_has_no_person(tmp_path):
    trainer = fake_trainer(tmp_path, ap_index=(1, 2), ap50=(0.4, 0.3))
    assert T.person_ap50(trainer) is None
    assert T.make_epoch_row(trainer, make_state(tmp_path), now=1.0)["person_ap50"] is None


def test_model_save_rolls_last_good_and_stops_at_the_session_budget(tmp_path):
    state = make_state(tmp_path, max_hours=0.0001)
    fake_checkpoint(state.paths.last, epoch=1, epochs=10)
    state.paths.weights.mkdir(parents=True, exist_ok=True)
    trainer = fake_trainer(tmp_path, epoch=1, epochs=10)
    T.build_callbacks(state)["on_model_save"](trainer)
    assert state.paths.last_good.read_bytes() == state.paths.last.read_bytes()
    assert epochs_of(state.paths.log_csv) == [2]
    assert trainer.stop is True and state.session_stopped is True


def test_model_save_does_not_stop_a_session_within_budget_or_on_the_last_epoch(tmp_path):
    state = make_state(tmp_path, max_hours=9.5)
    state.session_start = __import__("time").time()
    fake_checkpoint(state.paths.last, epoch=1, epochs=10)
    trainer = fake_trainer(tmp_path, epoch=1, epochs=10)
    T.build_callbacks(state)["on_model_save"](trainer)
    assert trainer.stop is False

    over = make_state(tmp_path, max_hours=0.0001)
    last_epoch = fake_trainer(tmp_path, epoch=9, epochs=10)
    T.build_callbacks(over)["on_model_save"](last_epoch)
    assert last_epoch.stop is False                 # the final epoch ends the run anyway; that is completion, not a session stop


def test_train_end_writes_the_state_the_notebook_reads(tmp_path):
    state = make_state(tmp_path)
    state.paths.run_dir.mkdir(parents=True, exist_ok=True)
    callbacks = T.build_callbacks(state)

    state.session_stopped = True
    callbacks["on_train_end"](fake_trainer(tmp_path, epoch=3, epochs=30))
    stopped = C.read_json(state.paths.state_json)
    assert stopped["complete"] is False and stopped["status"] == "session_stop" and stopped["stopped_for_time"] is True
    assert (stopped["epochs_done"], stopped["epochs_total"], stopped["early_stopped"], stopped["session"]) == (4, 30, False, 1)

    state.session_stopped = False
    callbacks["on_train_end"](fake_trainer(tmp_path, epoch=29, epochs=30))
    done = C.read_json(state.paths.state_json)
    assert done["complete"] is True and done["early_stopped"] is False and done["epochs_done"] == 30

    callbacks["on_train_end"](fake_trainer(tmp_path, epoch=11, epochs=30))
    early = C.read_json(state.paths.state_json)
    assert early["complete"] is True and early["early_stopped"] is True and early["epochs_done"] == 12


# --------------------------------------------------------------------------------------------
# data yaml
# --------------------------------------------------------------------------------------------


def test_data_yaml_points_at_the_phase_1_output_and_has_no_test_key():
    data = T.load_data_yaml()
    assert "test" not in data and data["nc"] == 5
    assert (C.CONFIG_DIR / data["path"]).resolve() == (REPO / "datasets" / "processed" / "yolo").resolve()
    assert T.normalise_names(data["names"]) == C.load_class_names(None)


def test_resolved_data_yaml_is_absolute_with_the_balanced_list_and_no_test(tmp_path, dataset):
    resolved = T.resolved_data_dict(T.load_data_yaml(), dataset, tmp_path / "run" / "train_list.txt")
    T.assert_resolved_data(resolved, C.load_class_names(None))
    assert Path(resolved["path"]).is_absolute() and resolved["train"].endswith("train_list.txt")
    assert Path(resolved["val"]) == (dataset / "images" / "val").resolve() or resolved["val"] == (dataset.resolve() / "images/val").as_posix()
    assert "test" not in resolved and list(resolved["names"].values()) == C.load_class_names(None)


def test_resolved_data_assertions_trip():
    names = C.load_class_names(None)
    good = {"path": "/abs", "train": "/abs/t.txt", "val": "/abs/v", "nc": 5, "names": dict(enumerate(names))}
    T.assert_resolved_data(good, names)
    with pytest.raises(T.DatasetError, match="test"):
        T.assert_resolved_data({**good, "test": "images/test"}, names)
    with pytest.raises(T.DatasetError, match="absolute"):
        T.assert_resolved_data({**good, "path": "relative/dir"}, names)
    with pytest.raises(T.DatasetError, match="class names"):
        T.assert_resolved_data({**good, "names": {0: "person", 1: "bike", 2: "car", 3: "truck", 4: "cart"}}, names)
    with pytest.raises(T.DatasetError, match="nc"):
        T.assert_resolved_data({**good, "nc": 4}, names)


# --------------------------------------------------------------------------------------------
# dataset preflight
# --------------------------------------------------------------------------------------------


def preflight(root: Path, manifest: bool = True) -> T.Preflight:
    names = C.load_class_names(None)
    train, val = T.scan_split(root, "train", 5), T.scan_split(root, "val", 5)
    return T.run_preflight(root, names, train, val, root / "manifest.tsv" if manifest else None, cart_gate=300)


def test_preflight_passes_on_the_clean_fixture_and_reports_the_histogram(dataset):
    report = preflight(dataset)
    assert report.ok, report.errors
    text = "\n".join(report.lines)
    assert "class 0 person" in text and "class 4 cart" in text and "empty-label images" in text
    assert any("class 4 (cart)" in w and "WITHDRAWN" in w and "1.5" in w for w in report.warnings)  # no carts: the gate withdraws it


def test_preflight_fails_on_duplicate_manifest_rows(dataset):
    manifest = dataset / "manifest.tsv"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join(rows + [rows[0]]) + "\n", encoding="utf-8")
    report = preflight(dataset)
    assert not report.ok
    message = next(e for e in report.errors if "manifest.tsv" in e)
    assert "more than once" in message and "same final filename" in message and "wrong labels" in message
    assert T.manifest_duplicates(T.read_manifest(manifest))[0][1] == 2


def test_manifest_duplicates_ignores_a_clean_manifest(dataset):
    assert T.manifest_duplicates(T.read_manifest(dataset / "manifest.tsv")) == []


def test_preflight_fails_on_a_class_id_out_of_range(dataset):
    label = next((dataset / "labels" / "train").glob("*.txt"))
    label.write_text(label.read_text(encoding="utf-8") + "7 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    report = preflight(dataset)
    assert any("outside 0..4" in e for e in report.errors)


def test_preflight_counts_a_missing_label_and_an_orphan_label(dataset):
    victim = next(iter(sorted((dataset / "images" / "train").iterdir())))
    (dataset / "labels" / "train" / f"{victim.stem}.txt").unlink()
    (dataset / "labels" / "train" / "ghost_visible.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    report = preflight(dataset)
    assert any("no label file" in e for e in report.errors)
    assert any("no image" in w for w in report.warnings)


def test_preflight_fails_on_an_image_that_is_neither_day_nor_ir(dataset):
    victim = next(iter(sorted((dataset / "images" / "train").iterdir())))
    victim.rename(victim.with_name("mystery_frame" + victim.suffix))
    assert any("neither a _visible nor a _lwir" in e for e in preflight(dataset).errors)


def test_preflight_fails_when_a_shipped_class_has_no_instances(dataset):
    for label in (dataset / "labels" / "train").glob("*.txt"):
        label.write_text("".join(ln + "\n" for ln in label.read_text(encoding="utf-8").splitlines() if not ln.startswith("3 ")), encoding="utf-8")
    assert any("class 3 (truck)" in e for e in preflight(dataset).errors)


def test_preflight_without_a_manifest_only_warns(dataset):
    (dataset / "manifest.tsv").unlink()
    report = preflight(dataset)
    assert report.ok and any("duplicate-row check was skipped" in w for w in report.warnings)


def test_label_parser_reports_bad_rows():
    classes, bad_class, bad_rows = T.parse_label_text("0 0.5 0.5 0.2 0.2\n5 0.5 0.5 0.2 0.2\n1 0.5 0.5\n2 0.5 0.5 0 0.1\nx 1 2 3 4\n", 5)
    assert classes == [0, 2] and len(bad_class) == 2 and len(bad_rows) == 2


# --------------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------------


def run_cli(*argv: str) -> int:
    return T.main(list(argv))


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, dataset, capsys):
    pytest.importorskip("ultralytics")
    pytest.importorskip("albumentations")
    run_dir = tmp_path / "runs" / "dry"
    code = run_cli("--stage", "day", "--dry-run", "--data-root", str(dataset), "--run-dir", str(run_dir),
                   "--model", str(C.TRAINING_ROOT / "weights" / "yolo11n.pt") if (C.TRAINING_ROOT / "weights" / "yolo11n.pt").exists() else str(dataset / "data.yaml"),
                   "--epochs", "2", "--imgsz", "160", "--batch", "8", "--workers", "0", "--device", "cpu")
    out = capsys.readouterr().out
    assert code == 0 and "plan" in out and "dry run: nothing was written" in out and "balanced epoch list" in out
    assert not run_dir.exists() and not (tmp_path / "runs").exists()


def test_dry_run_exits_3_on_a_duplicate_manifest_row(tmp_path, dataset, capsys):
    pytest.importorskip("ultralytics")
    manifest = dataset / "manifest.tsv"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join(rows + [rows[0]]) + "\n", encoding="utf-8")
    code = run_cli("--stage", "day", "--dry-run", "--data-root", str(dataset), "--run-dir", str(tmp_path / "r"), "--model", str(dataset / "data.yaml"))
    assert code == 3 and "more than once" in capsys.readouterr().err


def test_resume_without_a_checkpoint_exits_2(tmp_path, dataset, capsys):
    pytest.importorskip("ultralytics")
    code = run_cli("--stage", "day", "--resume", "--data-root", str(dataset), "--run-dir", str(tmp_path / "empty"))
    assert code == 2 and "--resume needs a valid checkpoint" in capsys.readouterr().err


def test_stage_ir_without_day_weights_exits_2_and_says_what_to_do(tmp_path, dataset, capsys):
    pytest.importorskip("ultralytics")
    code = run_cli("--stage", "ir", "--dry-run", "--data-root", str(dataset), "--run-dir", str(tmp_path / "runs" / "ir"))
    assert code == 2 and "Run --stage day first" in capsys.readouterr().err


def test_a_finished_run_exits_0_with_nothing_to_do(tmp_path, dataset, capsys):
    pytest.importorskip("ultralytics")
    paths = T.RunPaths(tmp_path / "runs" / "day")
    fake_checkpoint(paths.last_good, epoch=1, epochs=2)
    code = run_cli("--stage", "day", "--auto-resume", "--data-root", str(dataset), "--run-dir", str(paths.run_dir))
    assert code == 0 and "Nothing to do" in capsys.readouterr().out


def test_usage_errors_exit_2(tmp_path, dataset):
    pytest.importorskip("ultralytics")
    common = ["--stage", "day", "--dry-run", "--data-root", str(dataset), "--run-dir", str(tmp_path / "r")]
    assert run_cli(*common, "--max-hours", "0") == 2
    assert run_cli(*common, "--fraction", "2") == 2
    assert run_cli(*common, "--device", "0,1") == 2
    assert run_cli(*common, "--unfreeze-epoch", "-1") == 2
    assert run_cli("--stage", "ir", "--dry-run", "--unfreeze-epoch", "2", "--data-root", str(dataset), "--model", str(dataset / "data.yaml")) == 2


def test_resume_and_auto_resume_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        T.build_parser().parse_args(["--stage", "day", "--resume", "--auto-resume"])


def test_cli_defaults_match_the_contract():
    args = T.build_parser().parse_args(["--stage", "day"])
    assert args.max_hours == 9.5 and args.fraction == 1.0 and args.resume is False and args.auto_resume is False
    assert args.dry_run is False and args.skip_preflight is False


# --------------------------------------------------------------------------------------------
# per-class AP, early stopping across sessions, the session deadline
# --------------------------------------------------------------------------------------------


def test_schema_class_names_constant_matches_the_schema():
    assert list(T.SCHEMA_CLASS_NAMES) == C.load_class_names(None)
    assert T.AP_COLUMNS == [f"ap50_{n}" for n in C.load_class_names(None)]


def test_epoch_row_carries_ap50_for_every_class_present_and_blank_for_the_rest(tmp_path):
    trainer = fake_trainer(tmp_path, ap_index=(0, 2, 3), ap50=(0.6, 0.4, 0.2))
    row = T.make_epoch_row(trainer, make_state(tmp_path), now=5.0)
    assert (row["ap50_person"], row["ap50_car"], row["ap50_truck"]) == (0.6, 0.4, 0.2)
    assert row["ap50_two_wheeler"] is None and row["ap50_cart"] is None      # no ground truth in val: blank, never 0
    assert row["person_ap50"] == 0.6
    path = tmp_path / "train_log.csv"
    T.append_log_row(path, row)
    logged = T.read_log(path)[0]
    assert logged["ap50_car"] == "0.4" and logged["ap50_cart"] == ""


def test_an_old_format_log_is_migrated_before_a_new_row_is_appended(tmp_path):
    path = tmp_path / "train_log.csv"
    old = ["epoch", "session", "elapsed_s", "epoch_s", "box_loss", "cls_loss", "dfl_loss", "precision", "recall", "map50",
           "map50_95", "person_ap50", "lr", "fitness"]
    path.write_text(",".join(old) + "\n" + "1,1,10,10,1,1,1,0.1,0.2,0.3,0.2,0.25,0.001,0.2\n", encoding="utf-8")
    T.append_log_row(path, log_row(2, fitness=0.3, ap50_person=0.5))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == T.LOG_COLUMNS
    rows = T.read_log(path)
    assert [r["epoch"] for r in rows] == ["1", "2"] and rows[0]["fitness"] == "0.2" and rows[1]["ap50_person"] == "0.5"


def test_early_stopping_is_restored_from_the_log_on_resume(tmp_path):
    pytest.importorskip("ultralytics")
    from ultralytics.utils.torch_utils import EarlyStopping

    rows = [{"epoch": "1", "fitness": "0.10"}, {"epoch": "2", "fitness": "0.30"}, {"epoch": "3", "fitness": "0.30"},
            {"epoch": "4", "fitness": "0.25"}, {"epoch": "5", "fitness": "0.28"}, {"epoch": "6", "fitness": "0.99"}]
    stopper = EarlyStopping(patience=4)
    assert T.restore_early_stopping(stopper, rows, completed_epoch=5) == (2, 0.30)   # epoch 6 is past the checkpoint
    assert stopper.best_epoch == 2 and stopper.best_fitness == 0.30 and stopper.possible_stop is True
    assert stopper(6, 0.29) is True                         # 4 epochs without improvement across the session boundary
    fresh = EarlyStopping(patience=4)
    assert fresh(6, 0.29) is False                          # what the old behaviour did: patience silently restarted
    assert T.restore_early_stopping(EarlyStopping(patience=4), [{"epoch": "1", "fitness": ""}], 1) is None


def test_on_train_start_restores_the_stopper_only_when_resuming(tmp_path):
    state = make_state(tmp_path)
    for e, f in ((1, 0.1), (2, 0.4), (3, 0.3)):
        T.append_log_row(state.paths.log_csv, log_row(e, fitness=f))
    model = SimpleNamespace(parameters=lambda: [])
    stopper = SimpleNamespace(best_fitness=0.0, best_epoch=0, patience=5, possible_stop=False)
    trainer = SimpleNamespace(model=model, args=SimpleNamespace(freeze=None), start_epoch=3, stopper=stopper)
    T.build_callbacks(state)["on_train_start"](trainer)
    assert (stopper.best_epoch, stopper.best_fitness) == (2, 0.4)
    fresh = SimpleNamespace(best_fitness=0.0, best_epoch=0, patience=5, possible_stop=False)
    T.build_callbacks(state)["on_train_start"](SimpleNamespace(model=model, args=SimpleNamespace(freeze=None), start_epoch=0, stopper=fresh))
    assert (fresh.best_epoch, fresh.best_fitness) == (0, 0.0)


def test_should_stop_for_time_predicts_the_next_epoch():
    assert T.should_stop_for_time(100.0, 100.0, None) is True           # budget spent
    assert T.should_stop_for_time(50.0, 100.0, None) is False           # no estimate yet: run on
    assert T.should_stop_for_time(50.0, 100.0, 40.0) is False           # 50 + 44 fits
    assert T.should_stop_for_time(60.0, 100.0, 40.0) is True            # 60 + 44 would cross the deadline


def test_prior_epoch_seconds_uses_the_slowest_recent_epoch():
    rows = [{"epoch_s": "100"}, {"epoch_s": ""}, {"epoch_s": "300"}, {"epoch_s": "120"}, {"epoch_s": "110"}]
    assert T.prior_epoch_seconds(rows) == 300.0
    assert T.prior_epoch_seconds([{"epoch_s": ""}]) is None


def test_model_save_stops_before_an_epoch_that_would_cross_the_deadline(tmp_path):
    import time

    state = make_state(tmp_path, max_hours=1.0)
    now = time.time()
    state.session_start = now - 0.6 * 3600          # 0.6 h used
    state.epoch_started = now - 0.3 * 3600          # this epoch took 0.3 h; the next would end at ~0.93 h: fits
    fake_checkpoint(state.paths.last, epoch=1, epochs=10)
    trainer = fake_trainer(tmp_path, epoch=1, epochs=10)
    T.build_callbacks(state)["on_model_save"](trainer)
    assert trainer.stop is False
    state.session_start = now - 0.75 * 3600         # 0.75 h used: 0.75 + 0.33 > 1.0
    state.epoch_started = now - 0.3 * 3600
    trainer = fake_trainer(tmp_path, epoch=2, epochs=10)
    T.build_callbacks(state)["on_model_save"](trainer)
    assert trainer.stop is True and state.session_stopped is True


def test_an_early_stop_is_not_reported_as_a_session_stop(tmp_path):
    state = make_state(tmp_path, max_hours=0.0001)
    fake_checkpoint(state.paths.last, epoch=1, epochs=10)
    trainer = fake_trainer(tmp_path, epoch=1, epochs=10)
    trainer.stop = True                              # EarlyStopping already ended the run this epoch
    T.build_callbacks(state)["on_model_save"](trainer)
    assert state.session_stopped is False


# --------------------------------------------------------------------------------------------
# the cart gate: a dataset that withdraws class 4
# --------------------------------------------------------------------------------------------


def drop_cart_from_data_yaml(root: Path) -> None:
    import yaml

    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    data["nc"] = 4
    data["names"] = {i: n for i, n in enumerate(C.load_class_names(None)[:4])}
    (root / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def test_withdrawn_classes_from_the_dataset_names():
    schema = C.load_class_names(None)
    assert T.dataset_withdrawn_classes(schema, schema, "x") == ()
    assert T.dataset_withdrawn_classes(schema[:4], schema, "x") == (4,)
    assert T.dataset_withdrawn_classes({i: n for i, n in enumerate(schema[:4])}, schema, "x") == (4,)
    for bad in (schema[:3], ["person", "bike", "car", "truck", "cart"], schema[1:]):
        with pytest.raises(T.DatasetError, match="taxonomy"):
            T.dataset_withdrawn_classes(bad, schema, "x")


def test_preflight_accepts_a_withdrawn_cart_with_no_rows(dataset):
    names = C.load_class_names(None)
    train, val = T.scan_split(dataset, "train", 5), T.scan_split(dataset, "val", 5)
    report = T.run_preflight(dataset, names, train, val, dataset / "manifest.tsv", cart_gate=300, withdrawn=(4,))
    assert report.ok, report.errors
    assert any("WITHDRAWN" in line for line in report.lines)
    assert T.effective_withdrawn(names, train, ()) == (4,)          # five names, no cart label: withdrawn in effect


def test_preflight_fails_when_a_withdrawn_class_still_has_rows(dataset):
    label = next((dataset / "labels" / "train").glob("*.txt"))
    label.write_text(label.read_text(encoding="utf-8") + "4 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    names = C.load_class_names(None)
    train, val = T.scan_split(dataset, "train", 5), T.scan_split(dataset, "val", 5)
    report = T.run_preflight(dataset, names, train, val, dataset / "manifest.tsv", cart_gate=300, withdrawn=(4,))
    assert any("withdrawn" in e and "cart" in e for e in report.errors)
    assert T.effective_withdrawn(names, train, ()) == ()             # one cart row: not withdrawn in effect


def test_dry_run_accepts_a_four_class_dataset_and_keeps_five_outputs(tmp_path, dataset, capsys):
    pytest.importorskip("ultralytics")
    pytest.importorskip("albumentations")
    drop_cart_from_data_yaml(dataset)
    code = run_cli("--stage", "day", "--dry-run", "--data-root", str(dataset), "--run-dir", str(tmp_path / "r"), "--model", str(dataset / "data.yaml"))
    out = capsys.readouterr().out
    assert code == 0 and "withdrawn class id(s) [4] (cart)" in out and "5-output head" in out


def test_a_resume_with_less_time_than_one_epoch_trains_nothing_and_exits_0(tmp_path, dataset, capsys):
    pytest.importorskip("ultralytics")
    pytest.importorskip("albumentations")
    paths = T.RunPaths(tmp_path / "runs" / "day")
    fake_checkpoint(paths.last, epoch=1, epochs=10)
    for e in (1, 2):
        T.append_log_row(paths.log_csv, {**log_row(e), "epoch_s": 3600.0})   # one epoch took an hour
    code = run_cli("--stage", "day", "--auto-resume", "--data-root", str(dataset), "--run-dir", str(paths.run_dir), "--max-hours", "0.5")
    assert code == 0 and "not enough time" in capsys.readouterr().out
    state = C.read_json(paths.state_json)
    assert state["status"] == "session_stop" and state["complete"] is False and "insufficient" in state["note"]


# --------------------------------------------------------------------------------------------
# the Kaggle notebook
# --------------------------------------------------------------------------------------------

NOTEBOOK = C.TRAINING_ROOT / "notebooks" / "kaggle_train.ipynb"


def notebook_code_cells() -> list[str]:
    import json

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_nbformat_and_every_code_cell_parses():
    import ast

    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(str(NOTEBOOK), as_version=4))
    for source in notebook_code_cells():
        ast.parse(source)


def test_notebook_shares_one_session_clock_and_never_ignores_a_training_failure():
    cells = notebook_code_cells()
    assert "NB_START = time.time()" in cells[0]
    assert 'GIT_REF = "fix/phase1-2-complete"' in cells[0]
    train_calls = [c for c in cells if "training/scripts/train.py --stage" in c and "--dry-run" not in c]
    assert len(train_calls) == 2
    for cell in train_calls:
        assert "--max-hours {budget" in cell and "remaining_train_hours()" in cell
        call = cell[cell.index('sh(f"python training/scripts/train.py'):]
        assert "check=False" not in call[: call.index("\n\n") if "\n\n" in call else len(call)]
    assert 'DEVICE = "0" ' in cells[0]                                             # one GPU: train.py refuses DDP
    assert "RESERVE_HOURS = 0.75" in cells[0]


def test_notebook_evaluates_only_a_finished_final_stage_and_never_self_baselines():
    text = "\n".join(notebook_code_cells())
    assert "READY = STAGE1_DONE and (STAGE2_DONE or not RUN_STAGE_2)" in text
    write_baseline = [ln for ln in text.splitlines() if "--write-baseline" in ln]
    assert write_baseline and all("{coco}" in ln and "{WEIGHTS}" not in ln for ln in write_baseline)
    assert '"baseline": "pretrained"' in text
    assert 'secret("HF_TOKEN")' in text and "UserSecretsClient" in text
    assert "truewatch_ds/data.yaml" in text and "truewatch_ds/manifests/hard_set.txt" in text


def test_notebook_helpers_find_inputs_at_any_depth_and_budget_the_session(tmp_path):
    import time

    namespace: dict = {}
    exec(notebook_code_cells()[0], namespace)                    # definitions only: touches nothing under /kaggle
    namespace["INPUT_ROOT"] = tmp_path / "input"
    namespace["RUNS"] = tmp_path / "runs"
    deep = tmp_path / "input" / "datasets" / "someone" / "prev-output" / "runs" / "day"
    deep.mkdir(parents=True)
    (deep / "train_log.csv").write_text("epoch,epoch_s\n1,1800\n2,2000\n3,1900\n", encoding="utf-8")
    smoke = tmp_path / "input" / "prev" / "smoke" / "runs" / "day"
    smoke.mkdir(parents=True)
    (smoke / "train_log.csv").write_text("epoch,epoch_s\n1,1\n", encoding="utf-8")
    assert namespace["earlier_run_logs"]("day") == [deep / "train_log.csv"]
    (tmp_path / "runs" / "day").mkdir(parents=True)
    (tmp_path / "runs" / "day" / "train_log.csv").write_text((deep / "train_log.csv").read_text(), encoding="utf-8")
    assert abs(namespace["epoch_hours"]("day") - 2000 / 3600) < 1e-9
    assert namespace["epoch_hours"]("ir") is None
    namespace["NB_START"] = time.time() - 3 * 3600               # three hours into the session
    left = namespace["remaining_train_hours"]()
    assert abs(left - (12.0 - 3.0 - 0.75 - 0.25)) < 0.01


def test_notebook_detects_kaggle_and_colab_and_keeps_both_secret_paths():
    text = "\n".join(notebook_code_cells())
    assert 'ON_KAGGLE = Path("/kaggle/input").exists()' in text
    assert "import google.colab" in text and "ON_COLAB = True" in text
    assert "UserSecretsClient" in text          # Kaggle secrets, unchanged in the parameters cell
    assert "from google.colab import userdata" in text   # Colab secrets, in the platform cell


def test_notebook_colab_branch_redirects_every_path_the_kaggle_branch_sets():
    cells = notebook_code_cells()
    platform_cell = next(c for c in cells if "ON_KAGGLE and ON_COLAB" in c)
    colab_branch = platform_cell[platform_cell.index("if ON_COLAB:"):]
    for name in ("INPUT_ROOT", "WORK", "CODE", "RUNS", "OUT", "SESSION_HOURS", "PERSIST"):
        assert f"{name} = " in colab_branch, f"platform cell never reassigns {name} for Colab"
    assert "drive.mount(" in colab_branch
    assert "def secret(name):" in colab_branch    # overrides the Kaggle-secrets version for this branch only
    assert "WORKERS = os.cpu_count()" in colab_branch


def test_notebook_fetches_kaggle_inputs_on_colab_only_after_the_clone():
    cells = notebook_code_cells()
    clone_idx = next(i for i, c in enumerate(cells) if "git clone --depth 1" in c)
    fetch_idx = next(i for i, c in enumerate(cells) if "fetch_kaggle_outputs" in c)
    assert fetch_idx > clone_idx, "the fetch cell must run after the repo (and fetch_kaggle_outputs.py) is cloned"
    fetch_cell = cells[fetch_idx]
    assert fetch_cell.strip().startswith("if ON_COLAB:")
    assert "FK.fetch_dataset_shards(KAGGLE_BUILD_KERNEL" in fetch_cell
    assert "FK.fetch_prev_run_outputs(kernel" in fetch_cell
    assert "KAGGLE_API_TOKEN" in fetch_cell


def test_notebook_train_calls_use_workers_flag_and_sync_to_drive_after_each_stage():
    cells = notebook_code_cells()
    train_calls = [c for c in cells if "training/scripts/train.py --stage" in c and "--dry-run" not in c]
    assert len(train_calls) == 2
    for cell in train_calls:
        assert "{WORKERS_FLAG}" in cell
        assert "sync_runs_to_drive()" in cell


def test_notebook_packaging_copies_outputs_to_drive_on_colab():
    text = "\n".join(notebook_code_cells())
    assert "if ON_COLAB:" in text and "persist_out = PERSIST / \"outputs\"" in text
    assert "shutil.copytree(OUT, persist_out)" in text


def test_notebook_unpacking_frees_shard_disk_space_on_colab_only():
    cells = notebook_code_cells()
    unpack_cell = next(c for c in cells if "def unpack_shards" in c)
    assert "if ON_COLAB:" in unpack_cell and "path.unlink(missing_ok=True)" in unpack_cell


def test_notebook_top_markdown_names_the_colab_url_and_required_secret():
    import json

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    top_markdown = "".join(nb["cells"][0]["source"])
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert "colab.research.google.com/github/0XSreekar/True-Watch-AI/blob/fix/phase1-2-complete/training/notebooks/kaggle_train.ipynb" in top_markdown
    assert "KAGGLE_API_TOKEN" in top_markdown


# --------------------------------------------------------------------------------------------
# reporting rules for this file itself
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    C.TRAINING_ROOT / "scripts" / "train.py",
    C.TRAINING_ROOT / "tests" / "test_train.py",
    C.TRAINING_ROOT / "configs" / "yolo11s_day.yaml",
    C.TRAINING_ROOT / "configs" / "yolo11s_ir.yaml",
    C.TRAINING_ROOT / "configs" / "data.yaml",
    C.TRAINING_ROOT / "requirements.txt",
    C.TRAINING_ROOT / "notebooks" / "kaggle_train.ipynb",
    C.TRAINING_ROOT / "scripts" / "fetch_kaggle_outputs.py",
    C.TRAINING_ROOT / "tests" / "test_fetch_kaggle_outputs.py",
])
def test_owned_files_carry_no_forbidden_strings(path):
    text = path.read_text(encoding="utf-8").lower()
    banned = ["to" + "do", "fix" + "me", "place" + "holder", "clau" + "de", "anthro" + "pic", "gem" + "ini", "copi" + "lot", "open" + "ai", "chat" + "gpt",
              "co-authored" + "-by", "generated" + " with", "30 ms " + "end-to-end", "real-time on " + "any hardware"]
    hits = [b for b in banned if b in text]
    assert not hits, f"{path.name} contains {hits}"
    assert not re.search(r"[^\x00-\x7f]", path.read_text(encoding="utf-8")), f"{path.name} has non-ASCII text (emoji check)"


def test_a_bare_release_weights_name_is_downloaded_on_a_fresh_machine(tmp_path, monkeypatch):
    # Kaggle clones the repository fresh: `--model yolo11n.pt` names a release asset, not a local file.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(T, "WEIGHTS_DIR", tmp_path / "weights")
    fetched = []

    def fake_download(name, local):
        fetched.append(name)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"weights")
        return str(local)

    monkeypatch.setattr(T, "download_release_weights", fake_download)
    args = SimpleNamespace(model="yolo11n.pt")
    path, how = T.resolve_init_weights(args, SimpleNamespace(stage="day", model="yolo11s.pt"), None, dry_run=False)
    assert fetched == ["yolo11n.pt"] and Path(path) == tmp_path / "weights" / "yolo11n.pt" and "downloaded" in how
    # Second call finds it in training/weights without downloading again.
    path2, how2 = T.resolve_init_weights(args, SimpleNamespace(stage="day", model="yolo11s.pt"), None, dry_run=False)
    assert fetched == ["yolo11n.pt"] and path2 == path and "training/weights" in how2


@pytest.mark.parametrize("model", ["runs/x/best.pt", "missing_custom.pt", "yolo11n.onnx"])
def test_a_missing_non_release_model_path_is_still_a_usage_error(tmp_path, monkeypatch, model):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(T.UsageError, match="does not exist"):
        T.resolve_init_weights(SimpleNamespace(model=model), SimpleNamespace(stage="day", model="yolo11s.pt"), None, dry_run=False)
