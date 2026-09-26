# TRUEWATCH Phase 2: detector metrics

> **Every number below was measured on the split and host named beside it, or it is absent:** a target with no measurement is NOT ASSESSED, and nothing here is estimated, projected or rounded in the project's favour.  
> **This is not a test-set result, not a Jetson result and not an alert-level or end-to-end result;** the only figure quoted from the deck for latency is the slide-3 design target of about 30 ms for YOLO11-s INT8 inference at 640 px on a Jetson Orin Nano Super, hardware this project does not own.


## 1. Provenance and caveats

**Split: val, not test.** The test split is sealed until Phase 11 (DATASET_SPEC 2.5). Val also picked the checkpoint: Ultralytics ranks epochs by validation fitness, keeps best.pt by it and stops early on it (DATASET_SPEC 2.4 lists val for epoch selection and early stopping). Every number here was therefore chosen on the same frames it is reported on, so it is optimistic. How optimistic is not measured; the test split will say, once.

Measured on val: day 11060 images, ir 6482 images, imgsz 640, NMS IoU 0.7. Day (the visible camera) and IR (LWIR replicated to 3 channels) are separate slices throughout; no figure in this document is computed over both.

Inputs read for this document (each figure below carries the host and split of its own input):

| kind | file | tag / label | created (UTC) | split | weights | measured on host |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| evaluation | eval_final.json | final | 2026-09-26T01:07:06Z | val | best.pt sha256 9c84ced99b63 | Intel(R) Xeon(R) CPU @ 2.00GHz (Linux-6.12.90+-x86_64-with-glibc2.35; 4 usable cores, cgroup limit 4 CPUs) |
| hard set | hardset_final.json | final | 2026-09-26T01:08:07Z | val | best.pt sha256 9c84ced99b63 | Intel(R) Xeon(R) CPU @ 2.00GHz (Linux-6.12.90+-x86_64-with-glibc2.35; 4 usable cores, cgroup limit 4 CPUs) |
| sweep | sweep_final.json | final | 2026-09-26T01:09:41+00:00 | val | best.pt sha256 9c84ced99b63 | Intel(R) Xeon(R) CPU @ 2.00GHz (Linux-6.12.90+-x86_64-with-glibc2.35; 4 usable cores, cgroup limit 4 CPUs) |
| export | export_final.json | final | 2026-09-26T01:35:47+00:00 | n/a | best.pt sha256 9c84ced99b63 | Apple M5 (macOS-26.6.2-arm64-arm-64bit; 10 usable cores) |
| benchmark | benchmark_mac-apple-m5-cpu.json | mac-apple-m5-cpu | 2026-09-20T05:23:48+00:00 | n/a | yolo11s-random.onnx sha256 c8f390b82490 | Apple M5 (macOS-26.6.2-arm64-arm-64bit; 10 usable cores) |

Notes the evaluation scripts recorded (the sweep's notes are in section 8):

- measured on the 'val' split; day and IR are separate slices and no figure covers both
- visible lighting: IDD is daylight and LLVIP is night by DATASET_SPEC 3.4 (an assumption, not measured per frame); KAIST lighting is the set's declared lighting in datasets/config/splits.yaml
- object height for the size buckets is measured in input pixels at imgsz 640
- AP uses the 101-point COCO interpolation, so a perfect detector reads 0.995, as in `yolo val`
- ran inference
- bootstrap resamples whole sequences using the split index's sequence_key
- hard set: the frozen regression images (DATASET_SPEC 2.6); the numbers are over those images only

ONNX export (`export_onnx.py`):

- file `final.onnx`, opset 17, imgsz 640, 37.9 MB, sha256 dd63db0a7b4e; dynamic axes {'images': {'0': 'batch'}, 'output0': {'0': 'batch'}}
- torch versus onnxruntime, raw max abs diff (output0 as emitted, box rows in pixels): 0.00195
- parity gate PASSED: max abs diff 6.2e-05 with box rows in grid units against tolerance 0.001, by batch {'1': 6.198883056640625e-05, '3': 6.103515625e-05}, input kind synthetic; the gate is not on raw pixels, see training/README.md 'ONNX parity'
- Hugging Face: https://huggingface.co/sreekar12/truewatch-yolo11s at revision `9fe209f0d9200d5cf77aecc65144634c1ef4c996`, pinned file https://huggingface.co/sreekar12/truewatch-yolo11s/resolve/9fe209f0d9200d5cf77aecc65144634c1ef4c996/final.onnx

Training log (`train_log.csv`):

- 6 rows, epochs 1 to 6 over 1 session(s)
- the epoch with the highest logged fitness is 4; that is the checkpoint the training run kept as best.pt, chosen on the val split
- median 3463 s per epoch, 4.96 h of epochs logged in total (measured on the training host, which the log does not name)
- the log's per-epoch validation figures (mAP, person AP50, fitness) are computed over the whole val set, day and IR together, because that is how Ultralytics validates. They chose the checkpoint and are deliberately not reproduced here: a figure over both modalities is not a result of this project. Sections 3 and 4 hold the per-modality numbers.


## 2. Targets against measurements

Targets are the slide-5 and slide-3 figures quoted in `docs/PHASE_MINUS1_SCOPE.md` 5.2. They are targets, not results.

| target | threshold | measured | verdict | basis |
| :-- | :-- | :-- | :-- | :-- |
| Person mAP@50, day (visible camera) | >= 0.85 | person AP50 0.5494, 95% CI [0.4948, 0.6104] (644 clusters, 55795 boxes); daylight-only 0.4866 (45391 boxes, lower 95% bound 0.4608) | NOT MET | 0.5494 < 0.85. |
| Person mAP@50, IR (LWIR replicated to 3 channels) | >= 0.75 | person AP50 0.8447, 95% CI [0.7818, 0.8761] (430 clusters, 19734 boxes) | MET | 0.8447 >= 0.75, lower 95% bound 0.7818 also >= target. |
| False alerts per camera per day (detector-level proxy at best) | < 5 | day: F1 point 0.656 FP/frame (90,656x budget); budget point unresolvable (11,060 frames); ir: F1 point 0.38 FP/frame (52,570x budget); budget point unresolvable (6,482 frames) | NOT MET | day: unresolvable, 11,060 frames cannot show the budget is met (at least 414,720 needed); ir: unresolvable, 6,482 frames cannot show the budget is met (at least 414,720 needed). |
| Detection range 150 m at 25 px per metre | 150 m | n/a | NOT ASSESSED | not a detector-only quantity. Owner: Phase 6 (tiled far-field inference, DORI grading and the ground-plane homography). |
| Nepali plate recognition | >= 85% | n/a | NOT ASSESSED | no plate model or plate evaluation exists yet. Owner: Phase 7 (ANPR on the Nepali plate set). |
| Missed fence crossings | < 2% | n/a | NOT ASSESSED | needs tracked events and rules, not a detector metric. Owner: Phases 4-5 (fusion, tracking and the rule engine; it is an event-level rate, not a detector metric). |
| Detector latency (design target of about 30 ms, INT8, 640 px, Jetson Orin Nano Super) | about 30 ms (design target) | n/a | NOT ASSESSED | no measurement on the target hardware exists; CPU rows are not comparable. Owner: no phase: it needs a Jetson Orin Nano Super, which this project does not own. |

How to read the verdicts:

- `MET`: the point estimate reaches the target and so does the lower 95% bound (cluster bootstrap over sequences).
- `PARTIAL`: reached but not established (lower bound below the target), or only the daylight-only reading reaches it and establishes it on its own (at least 30 person boxes and its own lower 95% bound at the target). A smaller or less certain daylight reading leaves the verdict NOT MET. IR has no narrower reading.
- Figures in this table are truncated to four decimals, never rounded: 0.84962 reads 0.8496.
- `NOT MET`: measured, and short of the target. For false alerts also: the budget point is unreachable, unresolvable or vacuous.
- `NOT ASSESSED`: no measurement. The false-alert target can be `PARTIAL` at best, because the sweep is a detector-level proxy.
- Person mAP@50 means AP50 of the person class alone. The mean over classes in sections 3 and 4 is not the target.
- Per-class AP50 on val, with val also having picked the checkpoint, is optimistic (section 1).


## 3. Day (visible camera) per-class

11060 images from the val split. Composition by source [lighting]: flir 1081 [daylight 694, night 372, unresolved 15] | idd 6136 [daylight 6136] | llvip 3296 [night 3296] | visdrone 547 [daylight 547].

| class | boxes | AP50 | AP50-95 | P at op | R at op | P at 0.35 | R at 0.35 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| person | 55795 | 0.549 | 0.277 | 0.790 | 0.474 | 0.843 | 0.447 |
| two_wheeler | 30540 | 0.560 | 0.325 | 0.792 | 0.489 | 0.840 | 0.464 |
| car | 38950 | 0.717 | 0.487 | 0.816 | 0.655 | 0.859 | 0.633 |
| truck | 9349 | 0.644 | 0.457 | 0.775 | 0.591 | 0.813 | 0.570 |
| cart | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| mean over classes with ground truth |  | 0.618 | 0.386 | 0.793 | 0.552 |  |  |

`op` is this slice's F1-optimal confidence 0.300 (F1 averaged over the classes present). n/a means no ground truth for that class in this slice; it is never 0. † fewer than 30 boxes: the figure is an anecdote, not an estimate.

Person AP50 0.549, 95% CI [0.495, 0.610] from a cluster bootstrap over 644 clusters (11060 images). Frames without a known sequence are their own cluster, which makes the interval optimistic.

Person AP50 by source, inside this slice only:

| source | images | person boxes | person AP50 |
| :-- | --: | --: | --: |
| flir | 1081 | 3216 | 0.665 |
| idd | 6136 | 29386 | 0.522 |
| llvip | 3296 | 9296 | 0.835 |
| visdrone | 547 | 13897 | 0.374 |


## 4. IR (LWIR replicated to 3 channels) per-class

6482 images from the val split. Composition by source [lighting]: aaupdt 386 | birdsai 1225 | flir 1144 | hituav 431 | llvip 3296.

| class | boxes | AP50 | AP50-95 | P at op | R at op | P at 0.35 | R at 0.35 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| person | 19734 | 0.845 | 0.438 | 0.847 | 0.784 | 0.861 | 0.773 |
| two_wheeler | 725 | 0.660 | 0.358 | 0.751 | 0.600 | 0.770 | 0.586 |
| car | 7730 | 0.851 | 0.597 | 0.851 | 0.770 | 0.869 | 0.760 |
| truck | 225 | 0.669 | 0.470 | 0.733 | 0.609 | 0.759 | 0.587 |
| cart | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| mean over classes with ground truth |  | 0.756 | 0.466 | 0.795 | 0.691 |  |  |

`op` is this slice's F1-optimal confidence 0.325 (F1 averaged over the classes present). n/a means no ground truth for that class in this slice; it is never 0. † fewer than 30 boxes: the figure is an anecdote, not an estimate.

Person AP50 0.845, 95% CI [0.782, 0.876] from a cluster bootstrap over 430 clusters (6482 images). Frames without a known sequence are their own cluster, which makes the interval optimistic.

Person AP50 by source, inside this slice only:

| source | images | person boxes | person AP50 |
| :-- | --: | --: | --: |
| aaupdt | 386 | 1016 | 0.806 |
| birdsai | 1225 | 513 | 0.089 |
| flir | 1144 | 4453 | 0.759 |
| hituav | 431 | 4456 | 0.860 |
| llvip | 3296 | 9296 | 0.911 |


## 5. Small-object slices

Person metrics by object pixel height, using the seven levels of `docs/MEASUREMENTS.md` section 3 (77, 54, 38, 27, 19, 14, 9 px). Day and IR are separate tables.

Heights are in detector-input pixels at imgsz 640 (letterboxed), the basis `MEASUREMENTS.md` section 3 varied.

### Day (visible camera)

| height | range | person boxes | AP50 | P at 0.35 | R at 0.35 | pre-fine-tuning recall vs native detections (different quantity) |
| :-- | :-- | --: | --: | --: | --: | --: |
| 77 px | >= 64.5 px | 13613 | 0.885 | 0.906 | 0.812 | 100% |
| 54 px | 45.3-64.5 px | 3326 | 0.843 | 0.841 | 0.771 | 100% |
| 38 px | 32.0-45.3 px | 4051 | 0.800 | 0.838 | 0.712 | 100% |
| 27 px | 22.6-32.0 px | 5473 | 0.704 | 0.826 | 0.578 | 102% |
| 19 px | 16.3-22.6 px | 6782 | 0.546 | 0.782 | 0.411 | 90% |
| 14 px | 11.2-16.3 px | 8462 | 0.356 | 0.738 | 0.238 | 80% |
| 9 px | < 11.2 px | 14088 | 0.103 | 0.709 | 0.051 | 49% |

The 19, 14 and 9 px buckets (heights under about 22.6 px) hold 53% of this slice's person boxes (29332 of 55795). † fewer than 30 boxes in the bucket.

The last column is copied from `docs/MEASUREMENTS.md` section 3: a COCO-pretrained model, before any fine-tuning, on 23 KAIST visible frames at confidence 0.35. It is recall against that model's own native-resolution detections (100% = it found again what it found at full size), not recall against ground truth, and it can exceed 100%. It shows where the pre-fine-tuning cliff was; it is not a baseline for the columns to its left.

### IR (LWIR replicated to 3 channels)

| height | range | person boxes | AP50 | P at 0.35 | R at 0.35 |
| :-- | :-- | --: | --: | --: | --: |
| 77 px | >= 64.5 px | 9310 | 0.928 | 0.913 | 0.884 |
| 54 px | 45.3-64.5 px | 984 | 0.862 | 0.845 | 0.799 |
| 38 px | 32.0-45.3 px | 939 | 0.831 | 0.797 | 0.768 |
| 27 px | 22.6-32.0 px | 1933 | 0.735 | 0.748 | 0.671 |
| 19 px | 16.3-22.6 px | 4032 | 0.830 | 0.858 | 0.744 |
| 14 px | 11.2-16.3 px | 1778 | 0.679 | 0.768 | 0.585 |
| 9 px | < 11.2 px | 758 | 0.353 | 0.646 | 0.253 |

The 19, 14 and 9 px buckets (heights under about 22.6 px) hold 33% of this slice's person boxes (6568 of 19734). † fewer than 30 boxes in the bucket.

There is no pre-fine-tuning column for IR: `MEASUREMENTS.md` section 3 was measured on visible frames only, so quoting it here would attribute a visible measurement to the IR camera.


## 6. Visible lighting sub-slices

Refinements INSIDE the day slice, not extra slices and not additive with it. Lighting of IDD (daylight) and LLVIP (night) frames is taken from `DATASET_SPEC.md` 3.4, an assumption not measured per frame; KAIST lighting is the set's declared lighting in `datasets/config/splits.yaml`. `unresolved` means it could not be determined and is never counted as daylight.

| lighting | images | composition by source | person boxes | person AP50 | P at op | R at op |
| :-- | :-- | :-- | --: | --: | --: | --: |
| daylight | 7377 | flir 694 [daylight 694] \| idd 6136 [daylight 6136] \| visdrone 547 [daylight 547] | 45391 | 0.487 | 0.764 | 0.416 |
| night | 3668 | flir 372 [night 372] \| llvip 3296 [night 3296] | 10398 | 0.813 | 0.816 | 0.756 |
| unresolved | 15 | flir 15 [unresolved 15] | 6 † | 0.187 | 1.000 | 0.167 |

`op` is each sub-slice's own F1-optimal confidence. n/a means no ground truth, never 0. † fewer than 30 boxes. A sub-slice dominated by one source (see composition) measures that source as much as it measures the lighting.


## 7. Hard set (regression gate)

Manifest entries 280, resolved to val images 280, images scored 280. Unresolved 0.

**Gate: PASSED** against `hardset_baseline.json` with a tolerance of 0.02 absolute. Passing means no gated number fell by more than that; it does not mean the numbers are good.

| check | slice | current | baseline | delta | boxes | status |
| :-- | :-- | --: | --: | --: | --: | --: |
| same_inference_settings | - | same | same | n/a | - | ok |
| same_image_set | - | 63b2abe3f7b05363 | 63b2abe3f7b05363 | n/a | - | ok |
| person_ap50 | day | 0.3747 | 0.2136 | 0.161 | - | ok |
| person_ap50 | ir | 0.8947 | 0.7030 | 0.192 | - | ok |
| person_recall_bucket_19px | day | 0.5545 | 0.1312 | 0.423 | 404 | ok |
| person_recall_bucket_14px | day | 0.2880 | 0.0411 | 0.247 | 802 | ok |
| person_recall_bucket_9px | day | 0.0731 | 0.0037 | 0.069 | 1081 | ok |
| person_recall_bucket_19px | ir | 0.8214 | 0.0000 | 0.821 | 28 | ok |
| person_recall_bucket_14px | ir | 0.8919 | 0.0000 | 0.892 | 37 | ok |
| person_recall_bucket_9px | ir | 0.4615 | 0.0000 | 0.462 | 26 | ok |

Person metrics over the hard-set images only. The hard set is a fixed subset chosen for difficulty (DATASET_SPEC 2.6), so these numbers are not comparable with the full val numbers above:

| slice | images | person boxes | person AP50 | P at op | R at op |
| :-- | --: | --: | --: | --: | --: |
| day | 80 | 2568 | 0.375 | 0.643 | 0.339 |
| ir | 200 | 921 | 0.895 | 0.892 | 0.875 |


## 8. Operating points

Two different questions, two separate numbers, never merged: the F1-optimal point balances precision and recall; the false-alert-budget point is the highest-recall threshold that fits the false-alert target.

Confidence at which the evaluation reports precision and recall (F1 averaged over classes, per slice):

| slice | operating conf |
| :-- | --: |
| day | 0.300 |
| ir | 0.325 |

Sweep `final` on the val split, class person, matching at IoU 0.50, measured on Intel(R) Xeon(R) CPU @ 2.00GHz (Linux-6.12.90+-x86_64-with-glibc2.35; 4 usable cores, cgroup limit 4 CPUs).

Budget arithmetic:

- 8 fps x 86,400 s = 691,200 frames per camera per day
- 5 allowed false alerts per camera per day / 1 (the fraction of false detections that become an alert; 1.0 is the worst case, no fusion or tracking credit) = 5 false detections per day
- budget = 7.234e-06 false positives per frame
- rule of three: zero false positives in n frames bounds the rate only to 3/n at 95%, so the budget cannot be resolved with fewer than 414,720 frames, and only if none of them produces a false positive

F1-optimal point:

| slice | frames | conf | NMS IoU | F1 | P / R | FP per frame | rate vs budget |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| day | 11060 | 0.250 | 0.500 | 0.607 | 0.791 / 0.492 | 0.656 | 90,656x |
| ir | 6482 | 0.300 | 0.500 | 0.825 | 0.863 / 0.790 | 0.38 | 52,570x |

False-alert-budget point (`met` needs the 95% UPPER bound within budget; the observed rate alone is not enough):

| slice | frames | status | conf | recall | false positives | FP per frame: observed / 95% upper | frames needed | frames available |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| day | 11060 | unresolvable | 0.900 | 0.005 | 0 | 0 / 0.000271 | 414,720 | 11,060 |
| ir | 6482 | unresolvable | 0.900 | 0.001 | 0 | 0 / 0.000463 | 414,720 | 6,482 |

- day: 0 false positive(s) in 11060 frames is within the budget, but the 95% upper bound 0.000271/frame is not, so these frames cannot show the budget is met. At least 414,720 frames are needed (and only if no false positive appears in them); the rule-of-three minimum is 414,720.
- ir: 0 false positive(s) in 6482 frames is within the budget, but the 95% upper bound 0.000463/frame is not, so these frames cannot show the budget is met. At least 414,720 frames are needed (and only if no false positive appears in them); the rule-of-three minimum is 414,720.

- The false-positive count is a detector-level proxy: every unmatched person detection at IoU 0.50 counts as a false alert (fp_to_alert as given). Fusion, tracking and rules (Phases 4-5) decide what becomes an alert; this is the rate before any of them, and nothing here shows how much they would remove.
- Frames are treated as independent trials. Frames of one sequence are correlated, so the effective sample is smaller than the frame count and the upper bound printed here is optimistic.
- Matching is done once at the confidence floor and then filtered by threshold, as Ultralytics and evaluate.py do, so these counts agree with the evaluation script's precision and recall.
- The val split keeps its natural composition (DATASET_SPEC 3.5): most frames contain people. Its false-positive rate is not the rate on a border camera's empty hours. Pass --empty-dir with --empty-modality for background frames.
- With --preds the inference host is not recorded in the cache; `host` is where this sweep ran.


## 9. Latency by host

Reference point: the slide-3 design target of about 30 ms for YOLO11-s INT8 inference at 640 px on a Jetson Orin Nano Super, hardware this project does not own. It is a target. No row below was measured on that board, no row can be compared to that target, and an FP32 CPU figure is not evidence about INT8 on a Jetson in either direction.

| label | measured on host | model | config (imgsz) | inference ms p50 / p95 | pipeline ms p50 / p95 |
| :-- | :-- | :-- | :-- | --: | --: |
| mac-apple-m5-cpu | Apple M5 (macOS-26.6.2-arm64-arm-64bit; 10 usable cores) | yolo11s-random.onnx, fp32, opset 17, 37.9 MB, weights random | 640, batch 1, threads 0, CPUExecutionProvider, 200 runs | 41.9 / 49.7 | suppressed: random weights |

`inference` is onnxruntime `session.run` only. `pipeline` adds letterbox, decode and NMS in numpy: a detector-stage latency, still not end-to-end (no capture, tracking or fusion). Both are single-frame, batch 1.

The pipeline figure is suppressed for mac-apple-m5-cpu: those rows timed a randomly initialised graph. Inference time depends on the graph and the host, not the weights, so that column stands. Pipeline time does not: a random head scores every anchor alike, NMS receives thousands of candidate boxes a trained detector never produces, and the figure is inflated. The JSON keeps the number; it is not reported here. Re-run `benchmark_cpu.py --onnx <trained export> --weights-kind trained --image <real frame>` for a pipeline row.

> FP32 on CPU through onnxruntime at batch 1; not INT8 and not the Jetson Orin Nano Super. The ~30 ms TARGET on slide 3 (YOLO11-s INT8 inference at 640 px on that board, hardware this project does not own) is a design budget, not a measurement, and is not comparable to any number in this file. pipeline_ms covers the detector stage only (letterbox, inference, decode, class-aware NMS in numpy) and is not end-to-end. Latency follows the graph and the host, not the weights, except that NMS cost grows with the number of candidate boxes; see pipeline_detail.


## 10. Top reasons for every target that is not MET

Computed from the inputs by a deterministic ranker (`make_metrics.py`, `rank_reasons`), not written by hand. For the mAP targets the ranking is by estimated AP50 points at stake; the estimates overlap (small people are part of the missed people), so they are not additive. A cause that alone stops a PARTIAL from becoming MET ranks first. For false alerts the order is blocking status, then severity. Fewer than three are shown when fewer exist.

### Person mAP@50, day (visible camera): NOT MET

1. Operating point: at the slice operating confidence 0.30 (F1 averaged over classes) recall is 0.474 and precision is 0.790, so the detector is recall-limited: it misses more people than it invents; the shortfall 1 - AP50 = 0.451 is split in proportion to (1 - recall) and (1 - precision). (about 32.2 AP50 points at stake)
2. Small people: 53% of person ground truth falls in the 19, 14 and 9 px buckets, under about 22.6 px (29332 of 55795 boxes); recall there is 0.188 against 0.743 for taller people at the report confidence, and AP50 0.278 against 0.829. (about 29.0 AP50 points at stake)
3. Source gap: visdrone is 25% of person ground truth in this slice and reads AP50 0.374 against 0.603 for the other sources. (about 5.7 AP50 points at stake)

### False alerts per camera per day (detector-level proxy at best): NOT MET

1. day: 11,060 frames cannot show the budget is met. 0 false positive(s) observed in them are compatible with a true rate of up to 0.000271/frame (37x the budget); at least 414,720 frames are needed.
2. ir: 6,482 frames cannot show the budget is met. 0 false positive(s) observed in them are compatible with a true rate of up to 0.000463/frame (64x the budget); at least 414,720 frames are needed.
3. day: the F1-optimal point (recall 0.492) runs at 0.656 FP/frame = 90,656x the budget; the budget point keeps recall 0.005. Meeting the budget costs recall 0.492 to 0.005.

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
WEIGHTS=/kaggle/working/runs_v2/ir/weights/best.pt
TAG=final
RUN_DIR=training/runs/ir   # the --run-dir given to train.py; train_log.csv spans every session

# 0. Train, on a GPU (Kaggle). Two stages of one lineage; --auto-resume continues after a session ends.
training/.venv/bin/python training/scripts/train.py --stage day --run-dir training/runs/day --auto-resume
training/.venv/bin/python training/scripts/train.py --stage ir --run-dir training/runs/ir --auto-resume

# 1. Per-class AP, precision and recall, day and IR separately, plus the object-height buckets.
training/.venv/bin/python training/scripts/evaluate.py --weights "$WEIGHTS" --tag "$TAG" --split val --imgsz 640 --nms-iou 0.7 --report-conf 0.35 --bootstrap 200 --size-basis input

# 2. Hard-set regression gate. The first run has no baseline; --write-baseline records one.
training/.venv/bin/python training/scripts/eval_hardset.py --weights "$WEIGHTS" --tag "$TAG" --imgsz 640 --baseline training/results/hardset_baseline.json

# 3. Confidence and NMS sweep, reusing the raw predictions step 1 cached (no second inference).
training/.venv/bin/python training/scripts/sweep_conf.py --preds "training/results/cache/preds_$TAG.npz" --tag "$TAG" --fps 8 --alerts-per-camera-day 5 --fp-to-alert 1

# 4. ONNX export with the torch versus onnxruntime parity check (exit non-zero above the tolerance).
training/.venv/bin/python training/scripts/export_onnx.py --weights "$WEIGHTS" --imgsz 640 --tag "$TAG"

# 5. CPU latency of the exported graph on THIS host; run it again on each host you want a row for.
training/.venv/bin/python training/scripts/benchmark_cpu.py --onnx "training/weights/$TAG.onnx" --weights-kind trained --image "$FRAME" --imgsz 640 --label "mac-apple-m5-cpu" --out "training/results/benchmark_mac-apple-m5-cpu.json"   # FRAME: a real camera frame

# 6. This report.
training/.venv/bin/python training/scripts/make_metrics.py --eval "training/results/eval_$TAG.json" --hardset "training/results/hardset_$TAG.json" --sweep "training/results/sweep_$TAG.json" --export "training/results/export_$TAG.json" --benchmark "training/results/benchmark_mac-apple-m5-cpu.json" --train-log "$RUN_DIR/train_log.csv" --out training/results/METRICS.md
```
