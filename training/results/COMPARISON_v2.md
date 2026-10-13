# Run 1 versus the v2 lineage, on identical images

Both rows are `evaluate.py` on the validation split of dataset v2 (17,542 images: IDD, FLIR, LLVIP, VisDrone, HIT-UAV,
AAU-PD-T, BIRDSAI), 640 px, NMS IoU 0.7, on the same Kaggle T4. `previous` is run 1 (dataset v1, stage 1 + IR stage),
evaluated before the v2 fine-tune started from it; `final` is v2 (10 epochs on dataset v2 + 6-epoch IR stage). The
validation split also chose each checkpoint, so both are optimistic; the test split stays sealed.

| Slice | Metric | Run 1 | v2 |
|---|---|---|---|
| day | mAP50 | 0.499 | 0.618 |
| day | mAP50-95 | 0.315 | 0.386 |
| day | recall at the F1 point | 0.448 | 0.552 |
| day | precision at the F1 point | 0.775 | 0.793 |
| day | person AP50 | 0.464 | 0.549 |
| day/daylight | mAP50 | 0.476 | 0.602 |
| day/daylight | mAP50-95 | 0.305 | 0.380 |
| day/daylight | recall at the F1 point | 0.426 | 0.540 |
| day/daylight | precision at the F1 point | 0.768 | 0.782 |
| day/daylight | person AP50 | 0.383 | 0.487 |
| day/night | mAP50 | 0.720 | 0.703 |
| day/night | mAP50-95 | 0.433 | 0.422 |
| day/night | recall at the F1 point | 0.642 | 0.641 |
| day/night | precision at the F1 point | 0.799 | 0.789 |
| day/night | person AP50 | 0.808 | 0.813 |
| ir | mAP50 | 0.556 | 0.756 |
| ir | mAP50-95 | 0.364 | 0.466 |
| ir | recall at the F1 point | 0.517 | 0.691 |
| ir | precision at the F1 point | 0.767 | 0.795 |
| ir | person AP50 | 0.619 | 0.845 |

Person AP50 by source:

| Slice | Source | Run 1 | v2 |
|---|---|---|---|
| day | flir | 0.670 | 0.665 |
| day | idd | 0.515 | 0.522 |
| day | llvip | 0.828 | 0.835 |
| day | visdrone | 0.054 | 0.374 |
| ir | aaupdt | 0.223 | 0.806 |
| ir | birdsai | 0.001 | 0.089 |
| ir | flir | 0.760 | 0.759 |
| ir | hituav | 0.056 | 0.860 |
| ir | llvip | 0.912 | 0.911 |

v2 is kept: it is better on every slice except day/night, where it is within noise of run 1 (mAP50 0.720 vs 0.703).
The gains come from the sources run 1 never saw (HIT-UAV, AAU-PD-T, VisDrone). Verdicts against the slide-5 targets are in
METRICS.md; the exported model is pinned in hf_model.json.
