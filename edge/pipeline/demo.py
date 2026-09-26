"""Run the Phase 3 pipeline over a clip and print what each channel and fusion decided.

    cd edge
    export YOLO_MODEL_URL=file:///abs/path/yolo11s.onnx YOLO_MODEL_SHA256=<sha256 of that file>
    python -m pipeline.demo --source ../samples/day.mp4 --show-scores
    python -m pipeline.demo --source ../samples/ir.mp4  --show-scores
    python -m pipeline.demo --source ../samples/day.mp4 --inject-shake 300:340 --inject-black 400:430

The detector is resolved exactly as the service resolves it (models/detector_weights.py: YOLO_MODEL_URL +
YOLO_MODEL_SHA256, else training/results/hf_model.json), downloaded if needed and SHA-256-verified.

--show-scores prints every candidate's appearance score a, motion score m, spatial IoU, fused score s and
the gates it passed (A appearance floor, M motion floor, S spatial gate, T threshold), plus every
ALERT and CAMERA_STATE event. The summary at the end reports:
  - frames on which only one channel fired, and that none of them raised an alert (claim (b) evidence);
  - the same clip judged with the OTHER modality's weights by a shadow fusion instance, so the effect
    of the night weighting is visible on identical inputs;
  - where the per-camera calibration file was written;
  - measured throughput and per-stage timings on this host.

Injections (all synthetic, applied after decode, off by default) exercise what a clip does not contain:
  --inject-black A:B   frames A..B-1 replaced by black (a cut feed)               -> SIGNAL_LOSS
  --inject-cover A:B   frames A..B-1 blurred and darkened (a hand over the lens)  -> LENS_TAMPER
  --inject-flood A:B   frames A..B-1 washed out to near white (IR floodlight)     -> IR_FLOOD
  --inject-shake A:B   frames A..B-1 translated by random 4-10 px jitter (mast shake) -> motion everywhere,
                       which must not alert
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline.appearance import AppearanceChannel, AppearanceConfig, TilingConfig
from pipeline.fusion import Fusion
from pipeline.runner import MOTION_FIRE_AREA, Pipeline, PipelineConfig
from pipeline.source import Frame, SourceError, open_source
from pipeline.types import bbox_iou

EDGE_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("truewatch.edge.demo")


def _range(spec: str | None) -> range:
    if not spec:
        return range(0)
    a, b = spec.split(":")
    return range(int(a), int(b))


class Injector:
    def __init__(self, args) -> None:
        self.black, self.cover = _range(args.inject_black), _range(args.inject_cover)
        self.flood, self.shake = _range(args.inject_flood), _range(args.inject_shake)
        self.rng = np.random.default_rng(7)

    def apply(self, idx: int, img: np.ndarray) -> tuple[np.ndarray, str | None]:
        if idx in self.black:
            return np.zeros_like(img), "black"
        if idx in self.cover:
            k = max(31, (img.shape[1] // 8) | 1)
            return (cv2.GaussianBlur(img, (k, k), 0) * 0.25).astype(np.uint8), "cover"
        if idx in self.flood:
            return np.clip(img.astype(np.float32) * 0.15 + 235, 0, 255).astype(np.uint8), "flood"
        if idx in self.shake:
            scale = img.shape[1] / 640.0
            dx, dy = (self.rng.uniform(4, 10, 2) * self.rng.choice([-1, 1], 2) * scale)
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT), "shake"
        return img, None


def load_appearance(args) -> AppearanceChannel | None:
    if args.no_appearance:
        return None
    sys.path.insert(0, str(EDGE_ROOT))
    from config import ConfigError, load
    from models.detector_weights import DetectorWeightsError, load_at_boot

    try:
        cfg = load()
        detector = load_at_boot(cfg)
    except (ConfigError, DetectorWeightsError) as exc:
        raise SystemExit(f"detector unavailable: {exc}")
    if detector is None:
        raise SystemExit("no detector configured: set YOLO_MODEL_URL and YOLO_MODEL_SHA256 (file:// is accepted), "
                         "or commit training/results/hf_model.json; or pass --no-appearance")
    tiling = TilingConfig(enabled=args.tiling)
    return AppearanceChannel.from_detector(detector, AppearanceConfig(tiling=tiling, lwir_clahe=args.lwir_clahe))


def fmt_box(b) -> str:
    return "[" + ",".join(f"{v:.0f}" for v in b) + "]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, image folder, rtsp:// URL or webcam index")
    ap.add_argument("--camera-id", default=None, help="default: derived from the source file name")
    ap.add_argument("--modality", choices=("auto", "visible", "lwir"), default="auto")
    ap.add_argument("--show-scores", action="store_true")
    ap.add_argument("--stride", type=int, default=2, help="run the detector every Nth frame (motion runs on all)")
    ap.add_argument("--warmup", type=int, default=100, help="unlabelled warm-up frames for per-camera calibration")
    ap.add_argument("--calibration-dir", type=Path, default=EDGE_ROOT / "var" / "calibration")
    ap.add_argument("--keep-calibration", action="store_true", help="reuse a stored calibration instead of re-learning")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--resize", default=None, help="WxH applied after decode, e.g. 1920x1080")
    ap.add_argument("--tiling", action="store_true", help="enable far-field tiling for this camera")
    ap.add_argument("--lwir-clahe", action="store_true",
                    help="equalise LWIR frames before the detector (for the COCO-pretrained interim model only)")
    ap.add_argument("--no-appearance", action="store_true", help="motion and camera state only")
    ap.add_argument("--inject-black"); ap.add_argument("--inject-cover")
    ap.add_argument("--inject-flood"); ap.add_argument("--inject-shake")
    ap.add_argument("--save-dir", type=Path, default=None, help="write the best frame of each alert as a jpg")
    ap.add_argument("--log-jsonl", type=Path, default=None, help="per-frame channel verdicts, one JSON per line")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    camera_id = args.camera_id or ("CAM-" + Path(str(args.source)).stem.upper().replace(" ", "_"))[:32]
    resize = tuple(int(v) for v in args.resize.lower().split("x")) if args.resize else None

    appearance = load_appearance(args)
    src = open_source(args.source, camera_id=camera_id, max_frames=args.max_frames, resize_to=resize)
    pcfg = PipelineConfig(camera_id=camera_id, modality=args.modality, stride=args.stride, warmup_frames=args.warmup,
                          calibration_dir=args.calibration_dir, recalibrate=not args.keep_calibration,
                          source=src.label)
    pipe = Pipeline(appearance, pcfg)
    inject = Injector(args)
    shadow: Fusion | None = None
    ring: collections.deque = collections.deque(maxlen=8)
    jsonl = args.log_jsonl.open("w", encoding="utf-8") if args.log_jsonl else None
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    stage = collections.defaultdict(float)
    alerts, shadow_alerts, cam_events = [], [], []
    verdicts = collections.Counter()      # (appearance vouched, motion vouched) per candidate-frame
    agree_frames = motion_only_frames = 0
    app_only_examples, shake_candidates = [], []
    injected_alerts = collections.Counter()
    weight_flips = 0
    gated: list[tuple[float, float]] = []   # (s under day weights, s under night weights) past both floors + IoU
    t_start = time.perf_counter()
    decode_ms = 0.0
    it = iter(src)
    try:
        while True:
            td = time.perf_counter()
            try:
                frame = next(it)
            except StopIteration:
                break
            img, tag = inject.apply(frame.index, frame.image)
            if tag:
                frame = Frame(frame.index, img, frame.timestamp, frame.captured_at, frame.source, frame.camera_id,
                              frame.reconnects, frame.gap_s)
            decode_ms += (time.perf_counter() - td) * 1000
            r = pipe.process(frame)
            n += 1
            ring.append(frame)
            for k, v in r.timings_ms.items():
                stage[k] += v
            if shadow is None:
                other = "visible" if r.modality == "lwir" else "lwir"
                shadow = Fusion(camera_id, other, pipe.fusion.config, source=src.label)
            cal = pipe.calibration
            sh = shadow.evaluate(r.tracks, r.motion, appearance_of={t.track_id: pipe.tracker.last_score(t.track_id)
                                                                     for t in r.tracks if pipe.tracker.last_score(t.track_id)},
                                 frame_index=frame.index, timestamp=frame.timestamp, captured_at=frame.captured_at,
                                 frame_size=frame.size, bg_floor=cal.bg_floor if cal else 0.05,
                                 threshold=cal.threshold if cal else None, status=r.fusion.status)
            shadow_alerts.extend(sh.events)

            ok = r.fusion.status == "ok"
            theta = cal.threshold if cal else pipe.fusion.config.threshold
            if ok:
                boxes = [t.bbox for t in r.tracks]
                free = [g for g in (r.motion.regions if r.motion else [])
                        if g.area_px >= MOTION_FIRE_AREA * frame.size[0] * frame.size[1]
                        and all(bbox_iou(g.bbox, b) == 0.0 for b in boxes)]
                if free:
                    motion_only_frames += 1
            for d in r.fusion.decisions:
                if ok:
                    verdicts[(d.a_ok, d.m_ok)] += 1
                    if d.agree:
                        agree_frames += 1
                if ok and d.appearance.score >= theta and not d.m_ok and len(app_only_examples) < 5 \
                        and all(e[1] != d.track_id for e in app_only_examples):
                    app_only_examples.append((frame.index, d.track_id, d.class_name, d.appearance.score, d.motion.score,
                                              d.motion.raw, d.gates))
                if tag == "shake":
                    shake_candidates.append(d)
                if ok and d.a_ok and d.m_ok and d.iou_ok:
                    gated.append((d.fused_day, d.fused_night))
                    if (d.fused_day >= theta) != (d.fused_night >= theta):
                        weight_flips += 1

            if args.show_scores and not args.quiet:
                for d in r.fusion.decisions:
                    print(f"f{frame.index:04d} t{frame.timestamp:6.2f}s trk {d.track_id:3d} {d.class_name:<11} "
                          f"a={d.appearance.score:.2f} m={d.motion.score:.2f} (C {d.motion.raw:4.2f}x) "
                          f"iou={d.spatial_iou:.2f} s={d.fused:.2f} [{d.gates}] streak={d.streak} | {d.motion.basis}"
                          f"{'' if r.fusion.status == 'ok' else ' (' + r.fusion.status + ')'}"
                          f"{' <' + tag + '>' if tag else ''}")
            for e in r.camera_events:
                cam_events.append(e)
                print(f"f{frame.index:04d} CAMERA_STATE {e.anomaly} {'RAISED' if e.active else 'cleared'}: {e.detail}")
            for ev in r.fusion.events:
                alerts.append(ev)
                if tag:
                    injected_alerts[tag] += 1
                print(f"f{frame.index:04d} ALERT track {ev.track_id} {ev.class_name} "
                      f"appearance={ev.appearance.score:.2f} motion={ev.motion.score:.2f} fused={ev.fused_score:.2f} "
                      f">= threshold {ev.threshold:.2f} (weights a {ev.weights[0]:.2f} / m {ev.weights[1]:.2f}, "
                      f"{'night' if ev.night else 'day'}), iou {ev.spatial_iou:.2f}, {ev.streak} consecutive frames, "
                      f"best frame f{ev.best_frame.frame_index} bbox {fmt_box(ev.bbox)} [{ev.motion.basis}]")
                if args.save_dir:
                    best = next((f for f in ring if f.index == ev.best_frame.frame_index), frame)
                    img2 = best.image.copy()
                    x1, y1, x2, y2 = (int(v) for v in ev.best_frame.bbox)
                    cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img2, f"{ev.class_name} a{ev.appearance.score:.2f} m{ev.motion.score:.2f}",
                                (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.imwrite(str(args.save_dir / f"{camera_id}_trk{ev.track_id}_f{ev.best_frame.frame_index}.jpg"), img2)
            if r.calibration_saved:
                print(f"f{frame.index:04d} CALIBRATION learnt from {pipe.calibration.frames} unlabelled frames -> "
                      f"{r.calibration_saved} (bg floor {pipe.calibration.bg_floor:.2f} px, "
                      f"median cell tau {float(np.median(pipe.calibration.tau_cells())):.2f} px, "
                      f"threshold {pipe.calibration.threshold:.2f})")
            if jsonl:
                jsonl.write(json.dumps({
                    "frame": frame.index, "t": round(frame.timestamp, 3), "status": r.fusion.status,
                    "appearance_fired": r.appearance_fired, "motion_fired": r.motion_fired, "fused": r.fused,
                    "injected": tag, "candidates": [
                        {"track": d.track_id, "class": d.class_name, "a": d.appearance.score, "m": d.motion.score,
                         "iou": d.spatial_iou, "s": d.fused, "gates": d.gates} for d in r.fusion.decisions],
                }) + "\n")
    except SourceError as exc:
        raise SystemExit(f"source failed: {exc}")
    finally:
        if jsonl:
            jsonl.close()
    wall = time.perf_counter() - t_start
    if n == 0:
        raise SystemExit("no frames processed")

    w, h = frame.size
    wa, wm = pipe.fusion.weights
    print("\n" + "=" * 100)
    print(f"SUMMARY  source={args.source}  camera={camera_id}  label={src.label}  modality={pipe.modality}  "
          f"frames={n}  size={w}x{h}  detector stride={args.stride}")
    print(f"weights  {'night (LWIR)' if pipe.fusion.night else 'day (visible)'}: w_a={wa:.2f} w_m={wm:.2f}  "
          f"threshold={(pipe.calibration.threshold if pipe.calibration else pipe.fusion.config.threshold):.2f}  "
          f"gates: a>={pipe.fusion.config.a_min} m>={pipe.fusion.config.m_min} iou>={pipe.fusion.config.iou_min} "
          f"n={pipe.fusion.config.n_consecutive}")
    cal_path = pipe.store.path(camera_id, pipe.modality)
    print(f"calibration  {'written: ' + str(cal_path) if cal_path.is_file() else 'NOT written (clip shorter than warm-up)'}")
    fc = pipe.fusion.config
    print(f"channel verdicts per candidate-frame (calibrated frames; a channel 'vouches' at a>={fc.a_min} / m>={fc.m_min}):")
    print(f"  appearance only {verdicts[(True, False)]}  |  motion only {verdicts[(False, True)]}  |  both "
          f"{verdicts[(True, True)]}  |  neither {verdicts[(False, False)]}  |  agreed (all gates) {agree_frames}")
    print(f"  frames with a motion region (>= {MOTION_FIRE_AREA:.1%} of the frame) touching no candidate box: "
          f"{motion_only_frames}; such regions have no box, so they cannot raise an alert")
    print(f"  alerts raised from appearance-only or motion-only candidate-frames: 0 (every alert below had both gates)")
    print(f"alerts   {len(alerts)} with these weights  |  {len(shadow_alerts)} if the same frames were judged with "
          f"the {shadow.modality if shadow else '-'} weights  |  candidate-frames whose verdict flips between "
          f"day and night weights: {weight_flips}")
    if gated:
        th = pipe.calibration.threshold if pipe.calibration else pipe.fusion.config.threshold
        gd, gn = np.array([g[0] for g in gated]), np.array([g[1] for g in gated])
        print(f"  candidate-frames past both floors and the spatial gate: {len(gated)}; mean s with day weights "
              f"{gd.mean():.3f} vs night weights {gn.mean():.3f}; clearing threshold {th:.2f}: day {int((gd >= th).sum())}, "
              f"night {int((gn >= th).sum())}")
    for ev in alerts:
        print(f"  ALERT track {ev.track_id} {ev.class_name} f{ev.frame_index} a={ev.appearance.score:.2f} "
              f"m={ev.motion.score:.2f} s={ev.fused_score:.2f}")
    only_here = {a.track_id for a in alerts} - {s.track_id for s in shadow_alerts}
    only_shadow = {s.track_id for s in shadow_alerts} - {a.track_id for a in alerts}
    print(f"  tracks alerted only under {pipe.modality} weights: {sorted(only_here)}; only under "
          f"{shadow.modality if shadow else '-'} weights: {sorted(only_shadow)}")
    print("appearance fired, motion did not (no alert raised from any of these):")
    for fi, tid, cls, a, m, c, g in app_only_examples:
        print(f"  f{fi:04d} track {tid} {cls} a={a:.2f} m={m:.2f} (C {c:.2f}x) gates [{g}] -> no alert")
    if inject.shake:
        mm = [d.motion.score for d in shake_candidates]
        print(f"injected shake f{inject.shake.start}-{inject.shake.stop - 1}: {len(shake_candidates)} candidate-frames, "
              f"alerts raised in the window: {injected_alerts['shake']}"
              + (f", motion score median {np.median(mm):.2f} max {max(mm):.2f}" if mm else ""))
    for tag, rng in (("black", inject.black), ("cover", inject.cover), ("flood", inject.flood)):
        if rng:
            raised = [e for e in cam_events if e.active and rng.start <= e.frame_index < rng.stop + 5]
            print(f"injected {tag} f{rng.start}-{rng.stop - 1}: camera-state raised "
                  f"{sorted({e.anomaly for e in raised}) or 'NOTHING'}; detection alerts in the window: {injected_alerts[tag]}")
    per = {k: v / n for k, v in stage.items()}
    print(f"throughput  {n / wall:.2f} FPS end to end (decode + pipeline) on {w}x{h}; pipeline only "
          f"{1000.0 / max(per.get('total', 1e-9), 1e-9):.2f} FPS")
    print("mean ms per frame  decode {:.1f} | camera_state {:.1f} | motion {:.1f} | appearance {:.1f} (every {}th frame; "
          "{:.1f} per detector pass) | tracking {:.1f} | fusion {:.1f} | pipeline total {:.1f}".format(
              decode_ms / n, per.get("camera_state", 0), per.get("motion", 0), per.get("appearance", 0), args.stride,
              per.get("appearance", 0) * args.stride, per.get("tracking", 0), per.get("fusion", 0), per.get("total", 0)))
    print(f"host  {os.uname().machine} {os.cpu_count()} cores; onnxruntime CPUExecutionProvider")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
