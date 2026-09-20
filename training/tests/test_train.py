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


@pytest.mark.parametrize("stage,epochs,close,freeze", [("day", 30, 10, 10), ("ir", 8, 3, None)])
def test_shipped_configs_load_and_resolve(stage, epochs, close, freeze):
    cfg = T.load_stage_config(T.DEFAULT_CONFIGS[stage], stage)
    args = T.resolve_train_args(cfg, T.load_augment_policy())
    assert args["epochs"] == epochs
    assert args["close_mosaic"] == close
    assert args.get("freeze") == freeze
    assert args["seed"] == 42 and args["optimizer"] == "SGD" and args["deterministic"] is False
    assert args["augmentations"] == []           # the Albumentations decision: built-in defaults are disabled
    assert args["imgsz"] == 640 and args["batch"] == 32 and args["nbs"] == 64


def test_stage_configs_carry_the_documented_differences():
    day = T.load_stage_config(T.DEFAULT_CONFIGS["day"], "day")
    ir = T.load_stage_config(T.DEFAULT_CONFIGS["ir"], "ir")
    assert (day.train["lr0"], day.train["lrf"], day.train["warmup_epochs"], day.train["patience"]) == (0.005, 0.01, 3, 8)
    assert (ir.train["lr0"], ir.train["lrf"], ir.train["warmup_epochs"], ir.train["patience"]) == (0.001, 0.1, 1, 5)
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
    text = T.DEFAULT_CONFIGS["day"].read_text(encoding="utf-8").replace("  epochs: 30", "  epochs: 30\n  mosaic: 1.0")
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
    assert day.close_mosaic_epochs is None and ir.close_mosaic_epochs == 3
    policy = T.load_augment_policy()
    assert policy.mosaic_close_epochs == 10
    assert T.resolve_train_args(day, policy)["close_mosaic"] == 10
    assert T.resolve_train_args(ir, policy)["close_mosaic"] == 3


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
    resolved = T.ultralytics_args(T.resolve_train_args(T.load_stage_config(T.DEFAULT_CONFIGS["day"], "day"), policy))
    T.assert_forbidden_augmentation(resolved, policy)
    assert resolved.close_mosaic == 10 and resolved.augmentations == []


# --------------------------------------------------------------------------------------------
# the Albumentations decision, with the package present and absent
# --------------------------------------------------------------------------------------------


def test_albumentations_override_disables_the_builtin_transforms():
    pytest.importorskip("ultralytics")
    pytest.importorskip("albumentations")
    from ultralytics.data.augment import Albumentations

    default = Albumentations(p=1.0, transforms=None)
    assert default.transform is not None and len(default.transform.transforms) == 7  # what Ultralytics would apply on Kaggle
    ours = Albumentations(p=1.0, transforms=[])                                      # what train.py passes
    assert ours.transform is not None and list(ours.transform.transforms) == []
    labels = {"img": __import__("numpy").zeros((32, 32, 3), dtype="uint8")}
    assert ours(labels) is labels or (ours(labels)["img"] == labels["img"]).all()


def test_effective_albumentations_reports_the_transform_list():
    pytest.importorskip("ultralytics")
    pytest.importorskip("albumentations")
    from ultralytics.data.augment import Albumentations

    dataset = SimpleNamespace(transforms=SimpleNamespace(transforms=[Albumentations(p=1.0, transforms=[])]))
    assert T.effective_albumentations(dataset) == []
    dataset = SimpleNamespace(transforms=SimpleNamespace(transforms=[Albumentations(p=1.0, transforms=None)]))
    assert len(T.effective_albumentations(dataset)) == 7
    assert T.effective_albumentations(SimpleNamespace(transforms=SimpleNamespace(transforms=[]))) is None


def test_override_does_not_crash_when_albumentations_is_absent(monkeypatch):
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import Albumentations

    monkeypatch.setitem(sys.modules, "albumentations", None)   # makes `import albumentations` raise ImportError
    absent = Albumentations(p=1.0, transforms=[])
    assert absent.transform is None
    labels = {"img": object()}
    assert absent(labels) is labels
    dataset = SimpleNamespace(transforms=SimpleNamespace(transforms=[absent]))
    assert T.effective_albumentations(dataset) is None


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
        "epoch,session,elapsed_s,epoch_s,box_loss,cls_loss,dfl_loss,precision,recall,map50,map50_95,person_ap50,lr,fitness"
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
    assert any("class 4 (cart)" in w and "300" in w for w in report.warnings)      # the DATASET_SPEC 1.5 gate note


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
# reporting rules for this file itself
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    C.TRAINING_ROOT / "scripts" / "train.py",
    C.TRAINING_ROOT / "tests" / "test_train.py",
    C.TRAINING_ROOT / "configs" / "yolo11s_day.yaml",
    C.TRAINING_ROOT / "configs" / "yolo11s_ir.yaml",
    C.TRAINING_ROOT / "configs" / "data.yaml",
    C.TRAINING_ROOT / "requirements.txt",
])
def test_owned_files_carry_no_forbidden_strings(path):
    text = path.read_text(encoding="utf-8").lower()
    banned = ["to" + "do", "fix" + "me", "place" + "holder", "clau" + "de", "anthro" + "pic", "gem" + "ini", "copi" + "lot", "open" + "ai", "chat" + "gpt",
              "co-authored" + "-by", "generated" + " with", "30 ms " + "end-to-end", "real-time on " + "any hardware"]
    hits = [b for b in banned if b in text]
    assert not hits, f"{path.name} contains {hits}"
    assert not re.search(r"[^\x00-\x7f]", path.read_text(encoding="utf-8")), f"{path.name} has non-ASCII text (emoji check)"
