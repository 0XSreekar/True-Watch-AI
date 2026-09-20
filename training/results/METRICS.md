# TRUEWATCH Phase 2: detector metrics

> **Every number below was measured on the split and host named beside it, or it is absent:** a target with no measurement is NOT ASSESSED, and nothing here is estimated, projected or rounded in the project's favour.  
> **This is not a test-set result, not a Jetson result and not an alert-level or end-to-end result;** the only figure quoted from the deck for latency is the slide-3 design target of about 30 ms for YOLO11-s INT8 inference at 640 px on a Jetson Orin Nano Super, hardware this project does not own.


## 1. Provenance and caveats

**No measurements yet.** No evaluation JSON (`--eval`) was provided, so this document contains no detector accuracy figure and every accuracy target is NOT ASSESSED. Nothing has been invented to fill the tables. To produce the numbers, run `evaluate.py` on a trained checkpoint (section 11 has the commands), then re-run this script with `--eval`.

Inputs read for this document (each figure below carries the host and split of its own input):

| kind | file | tag / label | created (UTC) | split | weights | measured on host |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| benchmark | benchmark_mac-apple-m5-cpu.json | mac-apple-m5-cpu | 2026-09-20T05:23:48+00:00 | n/a | yolo11s-random.onnx sha256 c8f390b82490 | Apple M5 (macOS-26.6.2-arm64-arm-64bit; 10 usable cores) |


## 2. Targets against measurements

Targets are the slide-5 and slide-3 figures quoted in `docs/PHASE_MINUS1_SCOPE.md` 5.2. They are targets, not results.

| target | threshold | measured | verdict | basis |
| :-- | :-- | :-- | :-- | :-- |
| Person mAP@50, day (visible camera) | >= 0.85 | n/a | NOT ASSESSED | no evaluation JSON was provided (--eval); run evaluate.py. |
| Person mAP@50, IR (LWIR replicated to 3 channels) | >= 0.75 | n/a | NOT ASSESSED | no evaluation JSON was provided (--eval); run evaluate.py. |
| False alerts per camera per day (detector-level proxy at best) | < 5 | n/a | NOT ASSESSED | no sweep JSON was provided (--sweep); run sweep_conf.py. |
| Detection range 150 m at 25 px per metre | 150 m | n/a | NOT ASSESSED | not a detector-only quantity. Owner: Phase 6 (tiled far-field inference, DORI grading and the ground-plane homography). |
| Nepali plate recognition | >= 85% | n/a | NOT ASSESSED | no plate model or plate evaluation exists yet. Owner: Phase 7 (ANPR on the Nepali plate set). |
| Missed fence crossings | < 2% | n/a | NOT ASSESSED | needs tracked events and rules, not a detector metric. Owner: Phases 4-5 (fusion, tracking and the rule engine; it is an event-level rate, not a detector metric). |
| Detector latency (design target of about 30 ms, INT8, 640 px, Jetson Orin Nano Super) | about 30 ms (design target) | n/a | NOT ASSESSED | no measurement on the target hardware exists; CPU rows are not comparable. Owner: no phase: it needs a Jetson Orin Nano Super, which this project does not own. |

No verdict can be given for a target with no measurement, so every row above is NOT ASSESSED.


## 3. Day (visible camera) per-class

No measurement yet: no evaluation JSON was provided. Run `evaluate.py` (section 11).


## 4. IR (LWIR replicated to 3 channels) per-class

No measurement yet: no evaluation JSON was provided. Run `evaluate.py` (section 11).


## 5. Small-object slices

Person metrics by object pixel height, using the seven levels of `docs/MEASUREMENTS.md` section 3 (77, 54, 38, 27, 19, 14, 9 px). Day and IR are separate tables.

No measurement yet: no evaluation JSON was provided. Run `evaluate.py` (section 11).


## 6. Visible lighting sub-slices

Refinements INSIDE the day slice, not extra slices and not additive with it. Lighting of IDD (daylight) and LLVIP (night) frames is taken from `DATASET_SPEC.md` 3.4, an assumption not measured per frame; KAIST lighting is the set's declared lighting in `datasets/config/splits.yaml`. `unresolved` means it could not be determined and is never counted as daylight.

No measurement yet: no evaluation JSON was provided. Run `evaluate.py` (section 11).


## 7. Hard set (regression gate)

No measurement yet: no hard-set JSON was provided. Run `eval_hardset.py` (section 11).


## 8. Operating points

Two different questions, two separate numbers, never merged: the F1-optimal point balances precision and recall; the false-alert-budget point is the highest-recall threshold that fits the false-alert target.

No measurement yet: no sweep JSON was provided. Run `sweep_conf.py` (section 11).


## 9. Latency by host

Reference point: the slide-3 design target of about 30 ms for YOLO11-s INT8 inference at 640 px on a Jetson Orin Nano Super, hardware this project does not own. It is a target. No row below was measured on that board, no row can be compared to that target, and an FP32 CPU figure is not evidence about INT8 on a Jetson in either direction.

| label | measured on host | model | config (imgsz) | inference ms p50 / p95 | pipeline ms p50 / p95 |
| :-- | :-- | :-- | :-- | --: | --: |
| mac-apple-m5-cpu | Apple M5 (macOS-26.6.2-arm64-arm-64bit; 10 usable cores) | yolo11s-random.onnx, fp32, opset 17, 37.9 MB | 640, batch 1, threads 0, CPUExecutionProvider, 200 runs | 41.9 / 49.7 | 61.5 / 68.7 |

`inference` is onnxruntime `session.run` only. `pipeline` adds letterbox, decode and NMS in numpy: a detector-stage latency, still not end-to-end (no capture, tracking or fusion). Both are single-frame, batch 1.

> FP32 on CPU through onnxruntime at batch 1; not INT8 and not the Jetson Orin Nano Super. The ~30 ms TARGET on slide 3 (YOLO11-s INT8 inference at 640 px on that board, hardware this project does not own) is a design budget, not a measurement, and is not comparable to any number in this file. pipeline_ms covers the detector stage only (letterbox, inference, decode, class-aware NMS in numpy) and is not end-to-end. Latency follows the graph and the host, not the weights, except that NMS cost grows with the number of candidate boxes; see pipeline_detail.


## 10. Top reasons for every target that is not MET

Computed from the inputs by a deterministic ranker (`make_metrics.py`, `rank_reasons`), not written by hand. For the mAP targets the ranking is by estimated AP50 points at stake; the estimates overlap (small people are part of the missed people), so they are not additive. A cause that alone stops a PARTIAL from becoming MET ranks first. For false alerts the order is blocking status, then severity. Fewer than three are shown when fewer exist.

### Person mAP@50, day (visible camera): NOT ASSESSED

1. No evaluation JSON was provided, so nothing was measured. Run evaluate.py (section 11).

### Person mAP@50, IR (LWIR replicated to 3 channels): NOT ASSESSED

1. No evaluation JSON was provided, so nothing was measured. Run evaluate.py (section 11).

### False alerts per camera per day (detector-level proxy at best): NOT ASSESSED

1. No sweep JSON was provided, so nothing was measured. Run sweep_conf.py (section 11).

### Detection range 150 m at 25 px per metre: NOT ASSESSED

1. Not measured: not a detector-only quantity. Owner: Phase 6 (tiled far-field inference, DORI grading and the ground-plane homography).

### Nepali plate recognition: NOT ASSESSED

1. Not measured: no plate model or plate evaluation exists yet. Owner: Phase 7 (ANPR on the Nepali plate set).

### Missed fence crossings: NOT ASSESSED

1. Not measured: needs tracked events and rules, not a detector metric. Owner: Phases 4-5 (fusion, tracking and the rule engine; it is an event-level rate, not a detector metric).

### Detector latency (design target of about 30 ms, INT8, 640 px, Jetson Orin Nano Super): NOT ASSESSED

1. Not measured: no measurement on the target hardware exists; CPU rows are not comparable. Owner: no phase: it needs a Jetson Orin Nano Super, which this project does not own.


## 11. Reproduction commands

Every command reads or writes files under `training/results/` or `training/weights/`; nothing is written into a tracked location, and no weights are committed.

```bash
# Run from the repository root. The dataset is found through $TRUEWATCH_DATA_ROOT, a Kaggle input mount or
# datasets/processed/yolo, in that order; pass --data-root to override.
WEIGHTS=training/runs/ir/weights/best.pt
TAG=phase2
RUN_DIR=training/runs/ir   # the --run-dir given to train.py; train_log.csv spans every session

# 0. Train, on a GPU (Kaggle). Two stages of one lineage; --auto-resume continues after a session ends.
training/.venv/bin/python training/scripts/train.py --stage day --run-dir training/runs/day --auto-resume
training/.venv/bin/python training/scripts/train.py --stage ir --run-dir training/runs/ir --auto-resume

# 1. Per-class AP, precision and recall, day and IR separately, plus the object-height buckets.
training/.venv/bin/python training/scripts/evaluate.py --weights "$WEIGHTS" --tag "$TAG" --split val --imgsz 640

# 2. Hard-set regression gate. The first run has no baseline; --write-baseline records one.
training/.venv/bin/python training/scripts/eval_hardset.py --weights "$WEIGHTS" --tag "$TAG" --imgsz 640 --baseline training/results/hardset_baseline.json

# 3. Confidence and NMS sweep, reusing the raw predictions step 1 cached (no second inference).
training/.venv/bin/python training/scripts/sweep_conf.py --preds "training/results/cache/preds_$TAG.npz" --tag "$TAG"

# 4. ONNX export with the torch versus onnxruntime parity check (exit non-zero above the tolerance).
training/.venv/bin/python training/scripts/export_onnx.py --weights "$WEIGHTS" --imgsz 640 --tag "$TAG"

# 5. CPU latency of the exported graph on THIS host; run it again on each host you want a row for.
training/.venv/bin/python training/scripts/benchmark_cpu.py --model "training/weights/$TAG.onnx" --imgsz 640 --label "mac-apple-m5-cpu" --out "training/results/benchmark_mac-apple-m5-cpu.json"

# 6. This report.
training/.venv/bin/python training/scripts/make_metrics.py --eval "training/results/eval_$TAG.json" --hardset "training/results/hardset_$TAG.json" --sweep "training/results/sweep_$TAG.json" --export "training/results/export_$TAG.json" --benchmark "training/results/benchmark_mac-apple-m5-cpu.json" --train-log "$RUN_DIR/train_log.csv" --out training/results/METRICS.md
```
