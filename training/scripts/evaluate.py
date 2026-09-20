"""Full detector evaluation: per-class AP, precision and recall, day and IR reported separately.

WHY this script exists. The Phase 2 targets (person mAP@50 >= 0.85 day, >= 0.75 IR) are stated per
modality, and MEASUREMENTS.md section 3 showed detection collapsing with object height. A single
mean over everything would hide both facts, so every number here belongs to exactly one slice and
the small-object table is part of the default output rather than an option:

  * SLICES  "day" is the visible camera, "ir" is LWIR replicated to 3 channels. Both keys are
            always present in the JSON, even when empty. Visible frames are refined into lighting
            sub-slices (day/daylight, day/night, day/unresolved); those live INSIDE day. No slice
            and no printed figure covers both modalities (_metrics.BlendError enforces it).
  * BUCKETS Person AP50, precision and recall by object pixel height, using the seven levels of
            MEASUREMENTS.md section 3 (77/54/38/27/19/14/9 px). --size-basis chooses whether the
            height is measured in detector-input pixels (default, the quantity section 3 varied)
            or in stored pixels (what the Phase 1 hard-set rule uses).
  * COMPOSITION Each slice prints how many images of which source and lighting it holds, because
            "person AP50 0.80 on day" means something different over 500 IDD frames than over 500
            KAIST frames.

The network is run once (NMS off, confidence floor 0.001) and the raw boxes are cached, so
sweep_conf.py can reuse them and this script can be re-run in seconds. NMS is applied afterwards
at --nms-iou by _predict.image_data_from_raw.

The test split is sealed until Phase 11 (docs/DATASET_SPEC.md section 2.5). --split test refuses
unless --unseal-test is passed, and that check runs before any path is resolved or any image read.

Exit codes: 0 done, 2 refused (bad flag, sealed split, missing weights or data, an image that is
neither _visible nor _lwir). Nothing measured on the synthetic fixture is a result; the banner and
the JSON notes say so whenever the fixture is detected.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _common as C  # noqa: E402
import _metrics as M  # noqa: E402
import _predict as P  # noqa: E402

SCHEMA = "truewatch.eval.v1"
SPLITS = ("train", "val", "test")
EXIT_REFUSED = 2

CLASSES_CSV_COLUMNS = ("slice", "class", "n_gt", "ap50", "ap50_95", "precision", "recall", "operating_conf")
SIZE_CSV_COLUMNS = ("slice", "class", "bucket_px", "range", "n_gt", "ap50", "precision", "recall", "conf")

SLICE_TITLES = {
    "day": "DAY (visible camera)",
    "ir": "IR (LWIR replicated to 3 channels)",
}


# --------------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------------


def fail(message: str, code: int = EXIT_REFUSED) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def as_refusal(fn, *args, **kwargs):
    """Run a shared helper that signals problems with SystemExit(message) and re-raise it as exit 2."""
    try:
        return fn(*args, **kwargs)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            fail(exc.code)
        raise


def check_split(split: str, unseal_test: bool) -> None:
    """Refuse an unknown split, and the sealed test split without an explicit unseal flag."""
    if split not in SPLITS:
        fail(f"--split must be one of {', '.join(SPLITS)} (got {split!r}). Phase 2 measures on val.")
    if split == "test" and not unseal_test:
        fail(
            "the test split is sealed until Phase 11 (docs/DATASET_SPEC.md section 2.5): it is the "
            "one held-out measurement, and evaluating on it earlier turns it into a tuning set. "
            "Use --split val. Pass --unseal-test only for the single final Phase 11 measurement."
        )


def check_suffixes(images: Sequence[Path]) -> None:
    """Every image must say whether it is visible or LWIR; a guessed slice would corrupt a metric."""
    bad = [p.name for p in images if C.modality_of_stem(p.stem) is None]
    if bad:
        fail(
            f"{len(bad)} image(s) carry neither a '_visible' nor a '_lwir' suffix, e.g. {bad[:3]}. "
            "Refusing to guess a slice: every image must be day or IR. Rebuild the dataset with "
            "datasets/scripts/09_build_yolo_ds.py, which names files {source}_{raw}_{modality}."
        )


# --------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Everything that changes a number. Recorded in the JSON so a result can be reproduced."""

    weights: Path
    imgsz: int
    batch: int
    device: str | None
    conf_floor: float
    nms_iou: float
    max_det: int
    report_conf: float
    n_boot: int
    size_basis: str
    size_classes: tuple[str, ...]

    @classmethod
    def from_args(cls, a: argparse.Namespace) -> "Settings":
        return cls(
            weights=Path(a.weights),
            imgsz=a.imgsz,
            batch=a.batch,
            device=a.device,
            conf_floor=a.conf_floor,
            nms_iou=a.nms_iou,
            max_det=a.max_det,
            report_conf=a.report_conf,
            n_boot=a.bootstrap,
            size_basis=a.size_basis,
            size_classes=tuple(a.size_classes),
        )

    def validate(self) -> None:
        if not self.weights.is_file():
            fail(f"weights not found: {self.weights}. Pass --weights path/to/best.pt (or best.onnx).")
        if self.imgsz < 32:
            fail(f"--imgsz must be at least 32 (got {self.imgsz})")
        if self.batch < 1 or self.max_det < 1:
            fail("--batch and --max-det must be at least 1")
        if not 0.0 < self.conf_floor < 1.0:
            fail(f"--conf-floor must be in (0, 1) (got {self.conf_floor}); AP needs the whole curve, use 0.001")
        if not 0.0 < self.nms_iou <= 1.0:
            fail(f"--nms-iou must be in (0, 1] (got {self.nms_iou})")
        if not 0.0 < self.report_conf < 1.0:
            fail(f"--report-conf must be in (0, 1) (got {self.report_conf})")
        if self.report_conf < self.conf_floor:
            fail("--report-conf is below --conf-floor, so no detection could reach it")
        if self.n_boot < 0:
            fail("--bootstrap must be 0 (off) or a positive resample count")

    def as_json(self) -> dict:
        return {
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device or "auto",
            "conf_floor": self.conf_floor,
            "nms_iou": self.nms_iou,
            "max_det": self.max_det,
            "report_conf": self.report_conf,
            "bootstrap": self.n_boot,
            "size_basis": self.size_basis,
            "size_classes": list(self.size_classes),
        }


def add_engine_args(ap: argparse.ArgumentParser) -> None:
    """Flags shared with eval_hardset.py: they define what is being measured and how."""
    ap.add_argument("--weights", required=True, help="detector weights, best.pt or an exported best.onnx")
    ap.add_argument("--data-root", default=None,
                    help="dataset root holding images/, labels/, data.yaml; when omitted: $TRUEWATCH_DATA_ROOT, "
                         "a Kaggle input mount, then datasets/processed/yolo")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size; the Jetson INT8 latency target (a design target) is defined at 640")
    ap.add_argument("--batch", type=int, default=16, help="inference batch size")
    ap.add_argument("--device", default=None, help="Ultralytics device string (cpu, 0, mps); default lets it choose")
    ap.add_argument("--conf-floor", type=float, default=P.DEFAULT_CONF_FLOOR,
                    help="lowest confidence kept; AP needs the whole precision-recall curve")
    ap.add_argument("--nms-iou", type=float, default=0.7, help="class-aware NMS IoU applied to the raw boxes")
    ap.add_argument("--max-det", type=int, default=300, help="detections kept per image after NMS")
    ap.add_argument("--report-conf", type=float, default=0.35,
                    help="fixed confidence for the report-conf precision/recall columns (0.35 matches MEASUREMENTS.md)")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="cluster-bootstrap resamples for the person AP50 interval (0 turns it off)")
    ap.add_argument("--size-basis", choices=("input", "stored"), default="input",
                    help="measure object height in detector-input pixels or in stored-file pixels")
    ap.add_argument("--size-classes", nargs="+", default=["person"],
                    help="classes that get the object-height bucket table")
    ap.add_argument("--index", default=None,
                    help="processed/index/split.jsonl for lighting and bootstrap clusters; when omitted it is "
                         "looked for next to the dataset root")
    ap.add_argument("--tag", default=None, help="name for the output files; when omitted: weights stem, split and short sha")
    ap.add_argument("--out-dir", default=str(C.RESULTS_DIR), help="where JSON, CSV and the prediction cache go")
    cache = ap.add_mutually_exclusive_group()
    cache.add_argument("--preds-cache", default=None,
                       help="read raw predictions from, and write them to, this .npz (reused when it matches)")
    cache.add_argument("--reuse-preds", action="store_true",
                       help="reuse <out-dir>/cache/preds_<tag>.npz when weights sha256, imgsz and floor match")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_engine_args(ap)
    ap.add_argument("--split", default="val", help="train, val or test; test is sealed until Phase 11")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N images of the split (a subset)")
    ap.add_argument("--unseal-test", action="store_true", help="allow --split test (Phase 11 final measurement only)")
    return ap


# --------------------------------------------------------------------------------------------
# Data location
# --------------------------------------------------------------------------------------------


def find_index(root: Path, explicit: str | None) -> Path | None:
    """The Phase 1 split index: --index, else next to the dataset root, else beside its parent.

    datasets/processed/yolo keeps no index of its own; 06_split.py writes processed/index/split.jsonl,
    which is the parent's index/ folder. The synthetic fixture keeps it inside the root.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            fail(f"--index {path} does not exist. Point it at processed/index/split.jsonl or omit the flag.")
        return path
    for candidate in (root / "index" / "split.jsonl", root.parent / "index" / "split.jsonl"):
        if candidate.is_file():
            return candidate
    return None


def is_synthetic(root: Path) -> bool:
    """True for the fixture that _synth.py writes; its data.yaml says so in its first comment."""
    try:
        return "synthetic fixture" in (root / "data.yaml").read_text(encoding="utf-8")
    except OSError:
        return False


def default_tag(weights: Path, split: str, sha256: str) -> str:
    return f"{weights.stem}_{split}_{sha256[:8]}"


def display_path(path: Path | str) -> str:
    """Repo-relative for files inside the repository, as given otherwise.

    Result files are committed to a public repository; an absolute path would carry the name of
    whoever ran the evaluation.
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(C.REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return str(p)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------------------------
# Inference and scoring
# --------------------------------------------------------------------------------------------


def cache_mismatch(raw: P.RawPreds, sha256: str, s: Settings, images: Sequence[Path]) -> str | None:
    """Why a cached prediction file cannot stand in for a fresh run, or None if it can."""
    meta = raw.meta
    if meta.get("weights_sha256") != sha256:
        return "weights sha256 differs"
    if meta.get("imgsz") != s.imgsz:
        return f"imgsz differs (cache {meta.get('imgsz')}, requested {s.imgsz})"
    if float(meta.get("conf_floor", -1.0)) != float(s.conf_floor):
        return f"conf floor differs (cache {meta.get('conf_floor')}, requested {s.conf_floor})"
    if meta.get("raw_max_det") != P.RAW_MAX_DET:
        return "raw box cap differs"
    if raw.paths != [str(p) for p in images]:
        return "the image list differs"
    return None


def get_raw(images: Sequence[Path], s: Settings, sha256: str, cache_path: Path, reuse: bool) -> tuple[P.RawPreds, str]:
    """Raw pre-NMS predictions: from a matching cache when asked to reuse, else by running the network."""
    if reuse and cache_path.is_file():
        try:
            raw = P.load_raw(cache_path)
        except Exception as exc:  # a truncated or foreign npz must not abort an evaluation
            print(f"cache {cache_path.name} unreadable ({exc}); running inference", flush=True)
        else:
            why = cache_mismatch(raw, sha256, s, images)
            if why is None:
                return raw, f"reused cached raw predictions {cache_path.name}"
            print(f"cache {cache_path.name} not reused: {why}; running inference", flush=True)
    raw = P.predict_raw(s.weights, images, imgsz=s.imgsz, batch=s.batch, device=s.device, conf_floor=s.conf_floor)
    P.save_raw(cache_path, raw)
    return raw, "ran inference"


def by_source_block(images: Sequence[M.ImageData], class_names: Sequence[str]) -> dict:
    """Person AP50 per source inside one slice, so a reader can see which corpus drives the number."""
    person = list(class_names).index("person") if "person" in class_names else None
    out: dict[str, dict] = {}
    for source in sorted({im.meta.source for im in images}):
        subset = [im for im in images if im.meta.source == source]
        n_gt, ap50 = 0, None
        if person is not None:
            matches = M.collect_matches(subset, person)
            n_gt, ap50 = matches.n_gt, M.average_precision(matches, 0)
        out[source] = {"n_images": len(subset), "person_n_gt": n_gt, "person_ap50": ap50}
    return out


def score_slice(images: Sequence[M.ImageData], class_names: Sequence[str], s: Settings) -> dict:
    result = M.evaluate_slice(images, class_names, report_conf=s.report_conf, n_boot=s.n_boot)
    result["size_buckets"] = {
        name: M.evaluate_size_buckets(images, class_names, name, C.SIZE_BUCKETS_PX, s.report_conf)
        for name in s.size_classes
    }
    result["by_source"] = by_source_block(images, class_names)
    return result


def slice_order(keys: Sequence[str]) -> list[str]:
    """day first, then ir, then the day sub-slices (daylight, night, unresolved)."""
    head = [k for k in ("day", "ir") if k in keys]
    subs = [k for k in ("day/daylight", "day/night", "day/unresolved") if k in keys]
    rest = sorted(k for k in keys if k not in head and k not in subs)
    return head + subs + rest


def score_all(image_data: Sequence[M.ImageData], class_names: Sequence[str], s: Settings) -> dict[str, dict]:
    """One evaluation per slice. 'day' and 'ir' always exist; the lighting refinements sit inside day."""
    groups = P.group_by_slice(image_data)
    for required in ("day", "ir", "day/daylight", "day/night"):
        groups.setdefault(required, [])
    slices = {key: score_slice(groups[key], class_names, s) for key in slice_order(list(groups))}
    return slices


def model_class_names(weights: Path) -> list[str] | None:
    """The class names stored in the weights, or None if they cannot be read.

    A model whose class order differs from data.yaml would be scored against the wrong ground truth
    without any error, since predictions carry only a class index. Loading the names is cheap.
    """
    try:
        from ultralytics import YOLO  # AGPL-3.0: training/ only

        names = YOLO(str(weights), task="detect").names
        return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    except Exception:
        return None


@dataclass
class Scored:
    slices: dict[str, dict]
    composition: dict[str, int]
    n_images: int
    unlabelled: int
    weights_sha256: str
    ultralytics: str | None
    inference_note: str
    cache_path: Path
    model_names: list[str] | None = None


def run_pipeline(
    images: Sequence[Path],
    s: Settings,
    resolver: C.MetaResolver,
    class_names: Sequence[str],
    cache_path: Path,
    reuse: bool,
) -> Scored:
    """Predict (or reuse), apply NMS, load ground truth, and score every slice for `images`."""
    check_suffixes(images)
    sha256 = C.sha256_file(s.weights)
    raw, inference_note = get_raw(images, s, sha256, cache_path, reuse)
    image_data, seen = as_refusal(
        P.image_data_from_raw, raw, resolver, s.nms_iou, s.max_det, 0.0, s.imgsz, s.size_basis
    )
    return Scored(
        slices=score_all(image_data, class_names, s),
        composition=seen["composition"],
        n_images=len(image_data),
        unlabelled=seen["unlabelled_images"],
        weights_sha256=sha256,
        ultralytics=raw.meta.get("ultralytics"),
        inference_note=inference_note,
        cache_path=cache_path,
        model_names=model_class_names(s.weights),
    )


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def base_notes(s: Settings, split: str, scored: Scored, index_used: bool, synthetic: bool,
               class_names: Sequence[str] | None = None) -> list[str]:
    notes = [
        f"measured on the '{split}' split; day and IR are separate slices and no figure covers both",
        "visible lighting: IDD is daylight and LLVIP is night by DATASET_SPEC 3.4 (an assumption, not "
        "measured per frame); KAIST lighting is the set's declared lighting in datasets/config/splits.yaml",
        f"object height for the size buckets is measured in {s.size_basis} pixels at imgsz {s.imgsz}",
        "AP uses the 101-point COCO interpolation, so a perfect detector reads 0.995, as in `yolo val`",
        scored.inference_note,
    ]
    if index_used:
        notes.append("bootstrap resamples whole sequences using the split index's sequence_key")
    else:
        notes.append(
            "no split index was found: KAIST lighting falls back to a 'setNN' token in the filename "
            "(else day/unresolved) and every frame is its own bootstrap cluster, which understates the interval"
        )
    if split == "train":
        notes.append("train split: the detector fitted these images, so these numbers are optimistic and are not a result")
    if split == "test":
        notes.append("test split UNSEALED with --unseal-test; this is the Phase 11 final measurement")
    if scored.unlabelled:
        notes.append(f"{scored.unlabelled} image(s) had no label file and were scored as background")
    if synthetic:
        notes.append("the dataset root is the synthetic fixture: these numbers prove the tooling runs and are NOT results")
    if class_names is not None and scored.model_names is not None and scored.model_names != list(class_names):
        notes.append(
            f"CLASS NAMES DIFFER: the weights carry {len(scored.model_names)} classes "
            f"{scored.model_names[:6]}{'...' if len(scored.model_names) > 6 else ''} while the dataset has "
            f"{list(class_names)}; predictions are matched to ground truth by class index, so per-class "
            "numbers are only meaningful if the two orders agree"
        )
    return notes


def make_report(
    *,
    schema: str,
    tag: str,
    s: Settings,
    split: str,
    scored: Scored,
    class_names: Sequence[str],
    index_used: bool,
    notes: Sequence[str],
) -> dict:
    """The fields eval_<tag>.json and hardset_<tag>.json have in common."""
    return {
        "schema": schema,
        "tag": tag,
        "created": utc_now(),
        "weights": {"path": display_path(s.weights), "sha256": scored.weights_sha256},
        "split": split,
        "imgsz": s.imgsz,
        "size_basis": s.size_basis,
        "nms_iou": s.nms_iou,
        "conf_floor": s.conf_floor,
        "report_conf": s.report_conf,
        "max_det": s.max_det,
        "bootstrap": s.n_boot,
        "device": s.device or "auto",
        "n_images": scored.n_images,
        "host": C.host_info(),
        "ultralytics": scored.ultralytics,
        "class_names": list(class_names),
        "composition": scored.composition,
        "index_used": index_used,
        "slices": scored.slices,
        "notes": list(notes),
    }


def _cell(value, digits: int = 6) -> str:
    return C.fmt(value, digits)


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)
    os.replace(tmp, path)


def write_classes_csv(path: Path, report: dict) -> None:
    rows = []
    for key in slice_order(list(report["slices"])):
        s = report["slices"][key]
        for name in report["class_names"]:
            c = s["classes"][name]
            rows.append([key, name, c["n_gt"], _cell(c["ap50"]), _cell(c["ap50_95"]),
                         _cell(c["precision"]), _cell(c["recall"]), _cell(s["operating_conf"], 3)])
    _write_csv(path, CLASSES_CSV_COLUMNS, rows)


def write_size_csv(path: Path, report: dict) -> None:
    rows = []
    for key in slice_order(list(report["slices"])):
        for name, table in report["slices"][key]["size_buckets"].items():
            for r in table:
                rows.append([key, name, r["bucket_px"], C.bucket_range_label(r["bucket_px"]), r["n_gt"],
                             _cell(r["ap50"]), _cell(r["precision"]), _cell(r["recall"]), _cell(r["conf"], 3)])
    _write_csv(path, SIZE_CSV_COLUMNS, rows)


# --------------------------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------------------------


def composition_line(composition: dict[str, int], slice_key: str) -> str:
    """'idd 3 [daylight 3] | kaist 4 [night 4]' for one slice, from the report's composition counts."""
    by_source: dict[str, dict[str, int]] = {}
    for key, n in composition.items():
        modality_slice, source, lighting = key.split("/", 2)
        if slice_key == "ir" and modality_slice != "ir":
            continue
        if slice_key != "ir":
            if modality_slice != "day":
                continue
            if slice_key != "day" and lighting != slice_key.split("/", 1)[1]:
                continue
        by_source.setdefault(source, {})[lighting] = n
    parts = []
    for source in sorted(by_source):
        lightings = by_source[source]
        total = sum(lightings.values())
        if slice_key == "ir":
            parts.append(f"{source} {total}")
        else:
            detail = ", ".join(f"{k} {v}" for k, v in sorted(lightings.items()))
            parts.append(f"{source} {total} [{detail}]")
    return " | ".join(parts) if parts else "none"


def format_table(header: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "  ") -> list[str]:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    def line(cells):
        return indent + "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))
    return [line(header), indent + "  ".join("-" * w for w in widths)] + [line(r) for r in rows]


def render_slice(key: str, s: dict, report: dict) -> list[str]:
    title = SLICE_TITLES.get(key, f"{key} (a lighting refinement inside day)")
    lines = ["", f"=== {title}: {s['n_images']} images ==="]
    if s["n_images"] == 0:
        return lines + ["  no images in this slice; every metric is n/a"]
    lines.append(f"  composition (images per source [lighting]): {composition_line(report['composition'], key)}")
    rc = f"{report['report_conf']:.2f}"
    rows = []
    for name in report["class_names"]:
        c = s["classes"][name]
        fixed = s["at_report_conf"]["classes"][name]
        rows.append([name, c["n_gt"], C.fmt(c["ap50"]), C.fmt(c["ap50_95"]), C.fmt(c["precision"]), C.fmt(c["recall"]),
                     C.fmt(fixed["precision"]), C.fmt(fixed["recall"])])
    mean = s["mean_over_present_classes"]
    rows.append(["mean (classes with GT)", "", C.fmt(mean["ap50"]), C.fmt(mean["ap50_95"]),
                 C.fmt(mean["precision"]), C.fmt(mean["recall"]), "", ""])
    lines += format_table(["class", "n_gt", "AP50", "AP50-95", "P@op", "R@op", f"P@{rc}", f"R@{rc}"], rows)
    lines.append(f"  op = F1-optimal confidence {s['operating_conf']:.3f} for this slice; n/a means no ground truth, never 0")
    for name, boot in s["bootstrap"].items():
        ci = boot["ap50_ci95"]
        text = "not computed" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(f"  {name} AP50 95% CI {text} ({boot['n_clusters']} clusters over {boot['n_images']} images)")
    src = s["by_source"]
    if src:
        cells = [f"{k}: {v['n_images']} img, {v['person_n_gt']} person GT, AP50 {C.fmt(v['person_ap50'])}" for k, v in sorted(src.items())]
        lines.append("  by source: " + "; ".join(cells))
    return lines


def render_size_table(key: str, class_name: str, rows: Sequence[dict], report: dict) -> list[str]:
    title = SLICE_TITLES.get(key, key).split(" (")[0]
    rc = f"{report['report_conf']:.2f}"
    body = [[f"{r['bucket_px']} px", C.bucket_range_label(r["bucket_px"]), r["n_gt"], C.fmt(r["ap50"]),
             C.fmt(r["precision"]), C.fmt(r["recall"])] for r in rows]
    header = ["height", "range", "n_gt", "AP50", f"P@{rc}", f"R@{rc}"]
    lines = ["", f"--- {class_name} by object height, {title} ({report['size_basis']} px at imgsz {report['imgsz']}) ---"]
    return lines + format_table(header, body)


def render_report(report: dict) -> str:
    """The human-readable tables: day, then IR, then the day lighting sub-slices, then size buckets."""
    host = report["host"]
    lines = [
        f"TRUEWATCH detector evaluation  ({report['schema']}, tag {report['tag']})",
        f"  weights   {Path(report['weights']['path']).name}  sha256 {report['weights']['sha256'][:12]}",
        f"  split     {report['split']}   images {report['n_images']}   imgsz {report['imgsz']}   "
        f"NMS IoU {report['nms_iou']}   conf floor {report['conf_floor']}",
        f"  host      {host.get('cpu')} | {host.get('platform')} | inference device {report['device']} "
        f"| ultralytics {report['ultralytics']}",
        f"  index     {'split index used' if report['index_used'] else 'no split index'}",
    ]
    if any("synthetic fixture" in n for n in report["notes"]):
        lines.append("  WARNING   synthetic fixture: these numbers prove the tooling runs and are NOT results")
    for note in report["notes"]:
        if note.startswith("CLASS NAMES DIFFER"):
            lines.append(f"  WARNING   {note}")
    if report["split"] == "train":
        lines.append("  WARNING   train split: the model fitted these images; numbers are optimistic")
    if report["split"] == "test":
        lines.append("  WARNING   TEST SPLIT UNSEALED (Phase 11 final measurement only)")
    for key in slice_order(list(report["slices"])):
        lines += render_slice(key, report["slices"][key], report)
    for key in ("day", "ir"):
        for name, rows in report["slices"][key]["size_buckets"].items():
            if report["slices"][key]["n_images"]:
                lines += render_size_table(key, name, rows, report)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    check_split(args.split, args.unseal_test)  # before any path is resolved: nothing is read on a refusal

    settings = Settings.from_args(args)
    settings.validate()
    root = as_refusal(C.resolve_data_root, args.data_root)
    class_names = C.load_class_names(root)
    unknown = [c for c in settings.size_classes if c not in class_names]
    if unknown:
        fail(f"--size-classes {unknown} are not classes of this dataset: {class_names}")

    images = as_refusal(C.list_split_images, root, args.split, args.limit)
    if not images:
        fail(f"images/{args.split} under {root} holds no images")

    index_path = find_index(root, args.index)
    resolver = C.MetaResolver(index_path)
    sha256 = C.sha256_file(settings.weights)
    tag = args.tag or default_tag(settings.weights, args.split, sha256)
    out_dir = Path(args.out_dir)
    cache_path = Path(args.preds_cache) if args.preds_cache else out_dir / "cache" / f"preds_{tag}.npz"
    reuse = bool(args.reuse_preds or args.preds_cache)

    scored = run_pipeline(images, settings, resolver, class_names, cache_path, reuse)
    notes = base_notes(settings, args.split, scored, resolver.has_index, is_synthetic(root), class_names)
    if args.limit:
        notes.append(f"--limit {args.limit}: only the first {args.limit} images of the split (sorted by name) were evaluated")
    report = make_report(schema=SCHEMA, tag=tag, s=settings, split=args.split, scored=scored,
                         class_names=class_names, index_used=resolver.has_index, notes=notes)

    json_path = out_dir / f"eval_{tag}.json"
    C.write_json(json_path, report)
    write_classes_csv(out_dir / f"eval_{tag}_classes.csv", report)
    write_size_csv(out_dir / f"eval_{tag}_size.csv", report)

    print(render_report(report))
    print(f"\nwrote {json_path}\n      {out_dir / f'eval_{tag}_classes.csv'}\n      {out_dir / f'eval_{tag}_size.csv'}"
          f"\n      {scored.cache_path} (raw predictions, reusable by sweep_conf.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
