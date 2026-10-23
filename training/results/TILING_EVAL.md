# Tiled inference on the full official val (detector v2)

Question: does tiled inference close the day small-person gap (day person AP50 0.549 against 0.85, see `DAY_GAP_AUDIT.md`)?
Answer: it helps a lot and does not reach the target. Best measured: day person AP50 **0.675** (daylight 0.638) on the full val, 0.650 on the held-out half. **Day target 0.85: NOT MET with tiling.** IR does not regress. `eval_final.json` and the published model are untouched.

## 1. Method

- Model: v2 `best.pt` (SHA-256 `9c84ced9...f39921b`, the weights behind `hf_model.json`), torch via Ultralytics 8.4.155, one Kaggle GPU kernel `sreekar1206/truewatch-tiling-eval` (inference only, 1.14 h wall, 0.99 h of it window passes, 17,542 images, 17,542 full-frame + 300,546 window passes). Kernel record: `tiling/kernel_run.json`.
- Data: the whole official val split (17,542 images: day 11,060, IR 6,482), the same set as `eval_final.json`.
- Window layout and merge are `edge/pipeline/tiling.py`, the module the edge pipeline imports (`appearance.py` calls the same `merge_passes`); the evaluator loads it by path (`training/scripts/tiled_common.py`). Merge: per-scale score floor (full frame 0.001, windows `tile_conf`), drop a window box that touches an interior window edge when an untruncated same-class box covers at least 50% of it, then class-aware NMS across all passes at `merge_iou`, max 300. Matching, AP50, size buckets and slices are `_metrics.py` through `evaluate.py`'s own scoring.
- Layouts: `g2x2` (2x2 grid, 20% overlap, windows upscaled to 640), `g3x2` (3x2 grid), `n640` (640 px windows at original resolution over the whole frame, no overlap: 6 windows at 1080p), `strip` (the edge default band, top 55% of the frame, 640 px windows, 20% overlap).
- No leakage: val was split by hash of the correlated-frame cluster into a tuning half (10,117 images) and a held-out half (7,425), each holding every source and both modalities. `tile_conf` {0.02..0.3}, `merge_iou` {0.4..0.7} and truncation on/off were chosen on the tuning half only: maximise day person AP50 with day person precision at 0.35 within 0.05 of baseline and IR person AP50 within 0.01 of baseline. Grid (600 day evaluations): `tiling/tuning_grid.csv`; choices: `tiling/tuning_selected.json`.
- `g3x2`: every setting that held precision within 0.05 cost IR more than 0.01 AP50 on the tuning half, so it has no selected setting and no row below (its unconstrained best, 0.712 day AP50, bought it with precision 0.77).

## 2. Baseline reproduction (before anything else)

Full-frame pass, same NMS 0.7 / cap 300, against `eval_final.json` (full val, person AP50 / mAP50): day 0.5489 vs 0.5494, daylight 0.4864 vs 0.4866, IR 0.8434 vs 0.8448; day mAP50 0.6173 vs 0.6176, IR mAP50 0.7535 vs 0.7562. Largest difference 0.0026, inside the 0.005 bound (`tiling/baseline_check.json`). The residual comes from the evaluator batching images in 4s (square versus rectangular padding) while the kernel batches 16.

## 3. Results

Operating point = the evaluator's report confidence 0.35 for precision and recall. Day = visible camera slice (includes LLVIP night); daylight is the daylight-only sub-slice. CPU ms = onnxruntime on this Mac (M-series, default threads), median of 5 synthetic 1920x1080 frames through the edge `AppearanceChannel` (`tiling_cost_mac_cpu.json`); it is noisy by about 20% run to run, and 5 passes should cost about 190 ms at 37 ms per pass.

### Held-out half (report half, not used for tuning) (7425 images)

| variant | day person AP50 | daylight person AP50 | IR person AP50 | day mAP50 | IR mAP50 | day person precision @0.35 | day person recall @0.35 | passes/frame day / IR | CPU ms per 1080p frame |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| full frame 640 (v2) | 0.507 | 0.476 | 0.816 | 0.614 | 0.739 | 0.831 | 0.403 | 1.0 / 1.0 | 37 (1 passes) |
| 2x2 grid + full | 0.608 | 0.585 | 0.841 | 0.687 | 0.774 | 0.818 | 0.550 | 5.0 / 5.0 | 255 (5 passes) |
| 640 px native tiles + full | 0.650 | 0.629 | 0.838 | 0.726 | 0.769 | 0.815 | 0.560 | 5.0 / 3.8 | 445 (7 passes) |
| far-field strip + full (edge default layout) | 0.617 | 0.603 | 0.831 | 0.701 | 0.762 | 0.819 | 0.512 | 4.6 / 4.0 | 322 (5 passes) |

### Full official val (17542 images)

| variant | day person AP50 | daylight person AP50 | IR person AP50 | day mAP50 | IR mAP50 | day person precision @0.35 | day person recall @0.35 | passes/frame day / IR | CPU ms per 1080p frame |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| full frame 640 (v2) | 0.549 | 0.486 | 0.843 | 0.617 | 0.754 | 0.843 | 0.446 | 1.0 / 1.0 | 37 (1 passes) |
| 2x2 grid + full | 0.635 | 0.595 | 0.846 | 0.687 | 0.763 | 0.815 | 0.580 | 5.0 / 5.0 | 255 (5 passes) |
| 640 px native tiles + full | 0.675 | 0.638 | 0.854 | 0.729 | 0.773 | 0.818 | 0.592 | 5.0 / 4.2 | 445 (7 passes) |
| far-field strip + full (edge default layout) | 0.642 | 0.609 | 0.846 | 0.702 | 0.764 | 0.813 | 0.552 | 4.6 / 4.3 | 322 (5 passes) |

### Person AP50 by source, day slice (full val)

| variant | flir | idd | llvip | visdrone | IR aaupdt | IR birdsai | IR flir | IR hituav | IR llvip |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| full frame 640 (v2) | 0.666 | 0.522 | 0.833 | 0.371 | 0.805 | 0.087 | 0.758 | 0.860 | 0.911 |
| 2x2 grid + full | 0.760 | 0.605 | 0.827 | 0.534 | 0.784 | 0.060 | 0.817 | 0.855 | 0.902 |
| 640 px native tiles + full | 0.795 | 0.655 | 0.841 | 0.564 | 0.820 | 0.051 | 0.806 | 0.884 | 0.909 |
| far-field strip + full (edge default layout) | 0.781 | 0.610 | 0.804 | 0.567 | 0.806 | 0.070 | 0.804 | 0.863 | 0.900 |

### Day person recall at 0.35 by height in detector-input px (full val)

| variant | 77 px | 54 px | 38 px | 27 px | 19 px | 14 px | 9 px |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| full frame 640 (v2) | 0.811 | 0.771 | 0.712 | 0.578 | 0.409 | 0.237 | 0.051 |
| 2x2 grid + full | 0.833 | 0.808 | 0.785 | 0.721 | 0.644 | 0.500 | 0.205 |
| 640 px native tiles + full | 0.846 | 0.825 | 0.804 | 0.744 | 0.660 | 0.505 | 0.210 |
| far-field strip + full (edge default layout) | 0.811 | 0.775 | 0.741 | 0.653 | 0.549 | 0.446 | 0.238 |

Chosen merge parameters (tuning half): 
- 2x2 grid + full: {'tile_conf': 0.2, 'merge_iou': 0.4, 'trunc_ioa': 0.5, 'trunc_margin': 4.0}
- 640 px native tiles + full: {'tile_conf': 0.02, 'merge_iou': 0.5, 'trunc_ioa': 0.5, 'trunc_margin': 4.0}
- far-field strip + full (edge default layout): {'tile_conf': 0.02, 'merge_iou': 0.5, 'trunc_ioa': 0.5, 'trunc_margin': 4.0}

Reading the tables:

- Tiling lifts day person AP50 by +0.09 to +0.13 on the full val and recall at 0.35 from 0.446 to 0.55-0.59; recall at 9 px goes from 0.05 to about 0.21, and at 14 px from 0.24 to 0.50. The cost is precision (0.843 to 0.815-0.818), the unlabelled-small-people effect the audit predicted.
- IR person AP50 does not regress (full val 0.846-0.854 against 0.843; held-out half 0.831-0.841 against 0.816). The IR drop limit (0.02, floor 0.75) is met everywhere. BIRDSAI (IR) is 0.09 and stays low whatever the setting; it was already the weak IR source.
- Sources: IDD +0.08 to +0.13, FLIR +0.10 to +0.13, VisDrone +0.16 to +0.20; LLVIP (night, already 0.83) is flat.
- Native-resolution 640 px tiles (`n640`) beat a 2x2 grid with upscaled windows in every row, at 7 passes against 5 on 1080p.
- Why 0.85 is not reached: people under about 14 px at input are still mostly missed (recall 0.2 at 9 px, 25% of day GT) and the label audit shows many of those are unlabelled or not judgeable.

## 4. Verdict rows (verification-report style)

| Target | Value | Measured (full val) | Verdict | Note |
| :-- | :-- | :-- | :-- | :-- |
| Person AP50, day, single 640 pass (v2 as published) | >= 0.85 | 0.5494 (daylight 0.4866) | NOT MET | `METRICS.md`, unchanged |
| Person AP50, day, tiled (`DETECTOR_TILING=full`, 7 passes at 1080p) | >= 0.85 | 0.675 (daylight 0.638; held-out half 0.650) | NOT MET | 0.675 < 0.85; tiling stated; precision at 0.35 0.818 |
| Person AP50, day, tiled far-field strip (`DETECTOR_TILING=strip`, 5 passes) | >= 0.85 | 0.642 (daylight 0.609; held-out half 0.617) | NOT MET | 0.642 < 0.85 |
| Person AP50, IR, tiled | >= 0.75 | 0.854 (`full`), 0.846 (`strip`) | MET (no regression) | baseline 0.843; within the allowed 0.02 drop |

## 5. Edge wiring and the default

- `DETECTOR_TILING=off|strip|full` (edge `config.py`, read by `app.py`); presets live in `edge/pipeline/tiling.py` (`PRESETS`). Window score floor 0.10 and merge IoU 0.5 are the edge's candidate floor and NMS IoU; the evaluator optimum for the floor was 0.02, which the detector does not decode below 0.10 on the edge (the held-out effect is under 0.01 AP50 in the tuning grid).
- **Default stays `off`.** One pass is about 37 ms per 1080p frame on this Mac; `strip` is 5 passes and `full` is 7, a 5 to 12x cost, and even `full` leaves the day target unmet. Recommended for fixed cameras with a distant approach (people under about 30 px tall in the frame) where recall matters more than latency: `full`; on a tight CPU budget: `strip`.
- Side finding fixed: the edge ran all crops of a frame as one batch; on CPU onnxruntime a batch of 4 costs 365 ms against 36 ms for one crop (a batch of 2: 273 ms), so crops now run one per call (outputs identical).
- Not done: IR frames are tiled with the same layout (no regression measured); a per-camera choice by modality is possible but unneeded.

## 6. Reproduce

Kernel: `training/scripts/tiled_infer.py` (inputs: the v2 weights and the dataset shards). Scoring: `python training/scripts/tiled_eval.py baseline|tune|report --cache <tiled_cache> --out <dir>`, tables: `tiled_tables.py`, cost: `tiled_bench.py`. Result files are in `training/results/tiling/`.
