# v5 (day-person round 2) versus v2, on identical images

Verdict: **v5 is NOT adopted. v2 stays the detector.** The day person target (0.85) remains NOT MET.

Run: Kaggle kernel `sreekar1206/truewatch-train-v5` (version 2; version 1 failed in 4 minutes on a data-mount path bug, about 0.3 GPU-h),
one T4, 7.21 h of the session. Initialised from the v2 weights. Stage 1: 4 epochs at imgsz 960, batch 16 (`yolo11s_finetune_v5.yaml`,
4.04 h). Stage 2: 2 epochs at imgsz 640 with LWIR repeated twice (`yolo11s_ir_v5.yaml`, 1.34 h). Evaluation, export and the
comparison stay at 640, NMS IoU 0.7, evaluator batch 4 (no OOM). v2 numbers are the committed `eval_final.json` (same val split of
17,542 images, same `evaluate.py`).

Levers: (1) new data, 9,860 daytime Open Images photos with 164,124 person boxes (at least 6 people, at least 4 shorter than 15% of
the image; annotations CC BY 4.0, images CC BY 2.0, per-image licence checked; train only, see DATASET_SPEC 1.8), merged into the
v2 train set; (2) training at 960 px. The v4 image-repeat lever is off.

| Slice | Metric | v2 | v5 | Change |
|---|---|---|---|---|
| day | person AP50 | 0.5494 | 0.5461 | -0.0033 |
| day | person AP50, 95% CI (cluster bootstrap) | 0.495-0.610 | 0.491-0.610 | overlapping |
| day | mAP50 | 0.6176 | 0.6102 | -0.0074 |
| day | IDD person AP50 | 0.5223 | 0.5156 | -0.0068 |
| day | VisDrone person AP50 | 0.3735 | 0.3745 | +0.0009 |
| ir | person AP50 | 0.8448 | 0.8369 | -0.0079 |
| ir | mAP50 | 0.7562 | 0.7470 | -0.0092 |
| held-out Open Images (99 photos, side check) | person AP50 | 0.4442 | 0.5565 | +0.1123 |

Hard-set gate: passed (`hardset_v5_rejected.json`). Merged-build validation gates G1-G19 (v2 plus the added train images):
all PASS except G17 (plates, out of scope, SKIP) (`merged_validation_v5.json`).
IR person AP50 stays above 0.75 and within 0.02 of v2, so nothing broke, but there is no day gain to adopt.

Reading. The new data is learned: on its own held-out photos person AP50 rises from 0.444 to 0.557 (only 99 images, so indicative).
That gain does not transfer to the v2 val day slice, whose small and crowded people come from dashcam (IDD) and drone (VisDrone)
views that differ from Flickr photos of street, beach and market scenes. Training at 960 and returning to 640 did not help either:
the day change is -0.0033 inside a CI about 0.115 wide, and day mAP50 and both IR figures slipped slightly. Two rounds
(v4 reweighting, v5 new data and resolution) now show that more of the same kind of person data is not the lever for this slice.

What a further attempt would need (not launched, budget used about 7.5 of 10 GPU-h): data from the target viewpoint, that is
elevated or dashcam daytime crowds with a verified permissive licence (none was found among the sources checked), or a label
audit of the IDD and VisDrone val boxes (very small and heavily occluded people may cap the achievable AP50 on this split).
