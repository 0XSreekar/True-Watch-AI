"""DEVICE (or any host): person AP50 of the INT8 TensorRT engine against the FP32 ONNX, day and IR separately.

So the accuracy cost of INT8 is measured, not assumed. Both models see the SAME images, are decoded by the
SAME edge code (pipeline.appearance.AppearanceChannel at conf 0.001, NMS IoU 0.7, no tiling) and are scored
by the SAME repo code the Phase 2 numbers come from (training/scripts/_metrics.py: greedy one-to-one IoU
matching, 101-point interpolated AP). No Ultralytics is imported; copy training/scripts/_metrics.py and
_common.py next to the edge code (docs/JETSON.md).

The subset is a deterministic stratified pick from val that EXCLUDES every calibration image
(--calib-list), so INT8 is not scored on its own calibration data. Day and IR are never blended.

  python edge/tools/jetson/accuracy_int8.py --data-root <dataset> --onnx final.onnx \\
      --engine engines/final_int8_640_b1.engine --calib-list calib_list.txt --out-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EDGE = HERE.parent.parent
sys.path.insert(0, str(EDGE))
sys.path.insert(0, str(EDGE.parent / "training" / "scripts"))

from tools.jetson import common  # noqa: E402
from tools.jetson.build_calib_set import SLICE_OF, modality_of, pick, source_of  # noqa: E402

PERSON = 0


def read_labels_xyxy(path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO label file -> (classes (g,), boxes xyxy in pixels (g,4)). Missing file = background image."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                rows.append([float(p) for p in parts])
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1) if len(a) else np.zeros((0, 4))
    return a[:, 0].astype(int), boxes


def image_data(name: str, gt_cls, gt_box, dets):
    """_metrics.ImageData from ground truth arrays and a list of pipeline Detections."""
    from _common import ImageMeta
    from _metrics import ImageData

    stem = Path(name).stem
    mod = modality_of(stem)
    meta = ImageMeta(name=stem, source=source_of(stem), modality=mod, slice=SLICE_OF[mod], lighting="n/a", cluster=stem)
    pr_box = np.asarray([d.bbox for d in dets], dtype=np.float64).reshape(-1, 4)
    pr_cls = np.asarray([d.class_id for d in dets], dtype=int)
    pr_conf = np.asarray([d.score for d in dets], dtype=np.float64)
    h = lambda b: (b[:, 3] - b[:, 1]) if len(b) else np.zeros(0)  # noqa: E731
    return ImageData(meta=meta, gt_cls=gt_cls, gt_box=gt_box, gt_h=h(gt_box), pr_cls=pr_cls, pr_box=pr_box,
                     pr_conf=pr_conf, pr_h=h(pr_box))


def person_ap50(images) -> tuple[float | None, int]:
    from _metrics import average_precision, collect_matches

    m = collect_matches(images, PERSON)
    return average_precision(m, 0), int(m.n_gt)


def paired_delta_ci(a_images, b_images, n_boot: int = 200, seed: int = 42) -> list[float] | None:
    """95% interval of AP50(b) - AP50(a) by resampling images with the same weights for both models.
    Image-level (not sequence-level) resampling: frames of one clip are correlated, so it is optimistic."""
    from _metrics import average_precision, collect_matches

    ma, mb = collect_matches(a_images, PERSON), collect_matches(b_images, PERSON)
    rng = np.random.default_rng(seed)
    n, out = len(a_images), []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        x, y = average_precision(ma, 0, w), average_precision(mb, 0, w)
        if x is not None and y is not None:
            out.append(y - x)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if len(out) >= 20 else None


def score(fp32_by_slice: dict, int8_by_slice: dict, n_boot: int = 200) -> dict:
    """Person AP50 per slice for both models and the INT8 minus FP32 delta, from ImageData lists."""
    res = {}
    for sl in sorted(fp32_by_slice):
        a, b = fp32_by_slice[sl], int8_by_slice[sl]
        ap_a, n_gt = person_ap50(a)
        ap_b, _ = person_ap50(b)
        res[sl] = {"n_images": len(a), "n_person_gt": n_gt, "fp32_ap50": ap_a, "int8_ap50": ap_b,
                   "delta_int8_minus_fp32": None if ap_a is None or ap_b is None else ap_b - ap_a,
                   "delta_ci95": paired_delta_ci(a, b, n_boot) if ap_a is not None and ap_b is not None else None}
    return res


def render_markdown(rec: dict) -> str:
    def f(v):
        return "n/a" if v is None else f"{v:.4f}"
    lines = ["# INT8 accuracy check (person AP50, val subset)", ""]
    if not rec.get("measured_on_jetson"):
        lines += ["> **NOT RUN ON A JETSON.** Do not quote as a board result.", ""]
    lines += [f"Subset: {rec['n_images']} images, excludes the calibration set; NMS IoU 0.7, no tiling. "
              f"Model sha256 {rec['model_sha256']}.", "",
              "| slice | images | person GT | FP32 ONNX | INT8 engine | delta | delta 95% (image bootstrap) |",
              "|---|---|---|---|---|---|---|"]
    for sl, r in rec["slices"].items():
        ci = r["delta_ci95"]
        lines.append(f"| {sl} | {r['n_images']} | {r['n_person_gt']} | {f(r['fp32_ap50'])} | {f(r['int8_ap50'])} | "
                     f"{f(r['delta_int8_minus_fp32'])} | {'n/a' if not ci else f'{ci[0]:.4f} to {ci[1]:.4f}'} |")
    return "\n".join(lines) + "\n"


def run_model(session, class_names, names, folder: Path, labels: Path, cv2) -> dict:
    from pipeline.appearance import AppearanceChannel, AppearanceConfig

    ch = AppearanceChannel(session, 640, class_names, AppearanceConfig(conf=0.001, iou=0.7, max_det=300))
    by: dict[str, list] = {}
    for n in names:
        img = cv2.imread(str(folder / n))
        h, w = img.shape[:2]
        gt_cls, gt_box = read_labels_xyxy(labels / (Path(n).stem + ".txt"), w, h)
        by.setdefault(SLICE_OF[modality_of(Path(n).stem)], []).append(image_data(n, gt_cls, gt_box, ch.detect(img)))
    return by


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--engine", type=Path, required=True, help="the INT8 engine")
    ap.add_argument("--calib-list", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=600, help="images in the subset, half day half IR")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--manifest", type=Path, default=common.DEFAULT_MANIFEST)
    ap.add_argument("--fp32-provider", default="CPUExecutionProvider")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args(argv)

    import cv2
    import onnxruntime as ort
    from tools.jetson.benchmark import collect_fingerprint
    from tools.jetson.trt_session import TrtSession

    manifest = common.load_manifest(args.manifest)
    common.verify_onnx(args.onnx, manifest)
    folder, labels = args.data_root / "images" / "val", args.data_root / "labels" / "val"
    calib = frozenset(l.split("\t")[0] for l in args.calib_list.read_text().splitlines() if l.strip())
    names = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    subset = [r["name"] for r in pick(names, args.n, 0.5, args.seed, exclude=calib)]
    class_names = manifest.get("class_names") or []

    fp32 = ort.InferenceSession(str(args.onnx), providers=[args.fp32_provider])
    a = run_model(fp32, class_names, subset, folder, labels, cv2)
    b = run_model(TrtSession(args.engine, class_names), class_names, subset, folder, labels, cv2)
    fp = collect_fingerprint()
    rec = {"schema": "truewatch.jetson_int8_accuracy.v1", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "fingerprint": fp, "measured_on_jetson": common.is_jetson(fp), "model_sha256": manifest["sha256"],
           "engine": args.engine.name, "fp32_provider": args.fp32_provider, "n_images": len(subset), "seed": args.seed,
           "slices": score(a, b, args.n_boot)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (args.out_dir / f"jetson_int8_accuracy_{stamp}.json").write_text(common.scrub_paths(json.dumps(rec, indent=2)) + "\n")
    (args.out_dir / f"jetson_int8_accuracy_{stamp}.md").write_text(render_markdown(rec))
    print(render_markdown(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
