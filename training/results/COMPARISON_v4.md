# v4 (day-person round) versus v2, on identical images

Verdict: **v4 is NOT adopted. v2 stays the detector.** The day person target (0.85) remains NOT MET.

Run: Kaggle kernel `sreekar1206/truewatch-train-v4b`, one T4, 4.13 h of the session. Initialised from the v2 weights
(`runs_v2/ir/weights/best.pt`), stage 1 of 4 epochs (`yolo11s_finetune_v4.yaml`, LR 0.0015) then a 2-epoch IR stage
(`yolo11s_ir_v4.yaml`). Lever: image repeats only. Every visible-light image holding a person entered the epoch list twice
(`sampling.visible_person_repeat`, new in `train.py`). No new data: dataset v2 only (IDD, FLIR, LLVIP, VisDrone, HIT-UAV,
AAU-PD-T, BIRDSAI; licences as in the dataset record). The earlier v4 attempt streamed UVH-26 file by file from Hugging Face
and hit HTTP 429; this run does not touch it. v2 numbers are the committed `eval_final.json` (same val split of 17,542
images, same `evaluate.py`, 640 px, NMS IoU 0.7), so no comparison evaluation was run on the GPU.

| Slice | Metric | v2 | v4 | Change |
|---|---|---|---|---|
| day | person AP50 | 0.5494 | 0.5509 | +0.0015 |
| day | mAP50 | 0.6176 | 0.6147 | -0.0029 |
| day | VisDrone person AP50 | 0.3735 | 0.3829 | +0.0094 |
| day | IDD person AP50 | 0.5223 | 0.5211 | -0.0013 |
| ir | person AP50 | 0.8448 | 0.8362 | -0.0085 |
| ir | mAP50 | 0.7562 | 0.7436 | -0.0126 |

Hard-set gate: passed (`hardset_v4_rejected.json`).

Reading. The day person change (+0.0015) is far smaller than the slice-to-slice movement between the two runs and is
not distinguishable from noise; day mAP50 and both IR figures went down. IR person AP50 stays above 0.75 and within 0.02
of v2, so nothing broke, but there is no gain to adopt. More epochs of the same data at a low learning rate with
repeats is not the lever. The day person gap is concentrated in IDD (0.52) and VisDrone (0.37), small and crowded people.

What one more run would need (not launched): new daytime far-field person data with licences cleared and fetched as
archives (for example a capped VisDrone/other aerial or CCTV subset built into a Kaggle dataset first), and training at
a larger image size with a 640 export check, roughly 8-10 GPU-h.
