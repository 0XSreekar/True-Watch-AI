# Day person gap audit (detector v2, final.onnx)

Question: why is day person AP50 0.549 against a 0.85 target, and is the gap label noise or real misses?
No training was run. Official numbers (`eval_final.json`, `METRICS.md`) are untouched; everything below is a separate analysis.

## 0. What was measured, and what was not

- Model: `final.onnx`, SHA-256 `dd63db0a...f39921b`, identical to `hf_model.json`. Run with onnxruntime on CPU, letterbox 640, person score floor 0.001, NMS 0.7, operating conf 0.35.
- Data: the local v1/v2-style copy of the val split. **It is not the same val split as `eval_final.json`.** It matches for FLIR day (694 images) and LLVIP (3296), but has IDD 3401 images (eval_final: 6136) and **no VisDrone** (eval_final: 547 images, 13,897 person boxes). So IDD is a different, smaller subset, and VisDrone could not be audited at all. Reproduction check: local IDD AP50 0.550 (official 0.522), FLIR day 0.698 (0.665 on all FLIR incl. night), LLVIP 0.851 (0.835). Same ballpark, not identical; use local numbers for ratios, not for the official table.
- Review sample: **68 images by eye**, not 150 (IDD 40, FLIR day 20, LLVIP 8; random, fixed seed). Stratification by source only. Contact sheets (GT drawn green when matched, yellow when missed, v2 prediction magenta when unmatched) are kept out of git. Boxes under about 12 px at input are not judgeable by eye by anyone; they are counted as "too small to judge".
- Consequence: all counts below are small-sample estimates with wide intervals. VisDrone is a blocker (see section 5).

Finding the headline number is made of: the reported "day" slice is **not all daylight**. It contains LLVIP night frames (9,296 boxes, AP50 0.83) and FLIR night. Daylight-only person AP50 is **0.487** (METRICS.md section on lighting); night is 0.813. 

## 1. Label audit (by eye, IDD 40 / FLIR day 20 / LLVIP 8 images)

Counts are objects, at conf 0.35. "FN" = GT box with no matching prediction. "Pred without GT" = prediction with no matching GT.

| source | images | GT missed (FN) | of FN: real, fair person | of FN: too small to judge (<~12 px input) | of FN: label error / unfair (not a person, person inside a car, heavily hidden) | preds without GT | of those: real person with no label | of those: true false positive |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| IDD | 40 | 78 | 13 (17%, 95% CI 10-26) | 57 (73%, 62-82) | 8 (10%, 5-19) | 10 | 7 (70%, 40-89) | 3 |
| FLIR day | 20 | 22 | 6 | 15 | 1 | 4 | 4 | 0 |
| LLVIP (night visible) | 8 | 8 | 8 (dark or occluded people, genuine misses) | 0 | 0 | 1 | 0 | 1 |
| VisDrone | 0 | not auditable: val images not available locally |  |  |  |  |  |  |

Intervals are Wilson 95% on the counts; the pred-without-GT row has n = 10, so treat as indicative only. Observations that are not counts:

- IDD labels rider and bike together as one person box, include people sitting inside vehicles, and label very distant people down to a few pixels. The detector has learned the box convention (those are the green matches), but occluded drivers behind glass are scored as misses.
- Unlabelled real people exist in crowds (FLIR and IDD market scenes), so some "false positives" in the official precision are label misses.
- LLVIP "day" frames are night, dark, and the misses there are real people in shadow, not label problems.

### How much of the AP gap is label problems?

Measured directly on the local IDD / FLIR day / LLVIP predictions:

| source | AP50 as is | AP50 if every unmatched prediction with conf >= 0.5 is forgiven (upper bound of "label miss" effect) | AP50 counting only GT >= 11.2 px at input | >= 16.3 px | >= 22.6 px |
| :-- | --: | --: | --: | --: | --: |
| IDD (bootstrap by image 95% CI 0.54-0.56) | 0.550 | 0.563 | 0.702 | 0.774 | 0.831 |
| FLIR day (CI 0.67-0.72) | 0.698 | 0.706 | 0.737 | 0.805 | 0.865 |
| LLVIP | 0.851 | 0.867 | 0.852 | 0.853 | 0.853 |

Reading: forgiving every confident unmatched prediction lifts AP50 by only +0.01 to +0.02, and unfair GT is about 10% of IDD misses. Label problems therefore explain roughly 0.02-0.04 AP50 of an IDD gap of about 0.30 (under 15%), and under 0.02 for FLIR. The remaining 85%+ is genuine recall loss on small people. (The bootstrap resamples images, not sequences, so the CI is optimistic.)

## 2. Size analysis

Person height in detector-input pixels (640 letterbox), as in `MEASUREMENTS.md` section 3. Official day slice (`eval_final_size.csv`, includes LLVIP night):

| bucket | range | day GT boxes | share | recall at 0.35 | AP50 |
| :-- | :-- | --: | --: | --: | --: |
| 77 px | >= 64.5 | 13,613 | 24.4% | 0.812 | 0.885 |
| 54 px | 45.3-64.5 | 3,326 | 6.0% | 0.771 | 0.843 |
| 38 px | 32-45.3 | 4,051 | 7.3% | 0.712 | 0.800 |
| 27 px | 22.6-32 | 5,473 | 9.8% | 0.578 | 0.704 |
| 19 px | 16.3-22.6 | 6,782 | 12.2% | 0.411 | 0.546 |
| 14 px | 11.2-16.3 | 8,462 | 15.2% | 0.238 | 0.356 |
| 9 px | < 11.2 | 14,088 | 25.2% | 0.051 | 0.103 |

- Share of day person GT below 11.2 px: **25.2%**; below 16.3 px: **40.4%** (daylight only: 30.9% and 49.0%). The 14 px bucket is 11.2-16.3 px, so "below 14 px" lies inside it; the exact 14.0 px split is not in the CSV.
- Recall at 0.35 if small boxes are excluded (a **reporting choice, not a fix**; the people are still there):

| keep GT of at least | share of GT kept | day recall | daylight-only recall |
| :-- | --: | --: | --: |
| all | 100% | 0.451 | 0.389 |
| 11.2 px | 74.8% | 0.586 | 0.540 |
| 16.3 px | 59.6% | 0.675 | 0.646 |
| 22.6 px | 47.4% | 0.743 | 0.738 |

- Local IDD: 34% of boxes are under 14 px and 26.5% under 11.2 px; IDD recall by bucket 0.88 / 0.81 / 0.70 / 0.57 / 0.37 / 0.24 / 0.045. Those people are 3 to 4 times smaller after the 1920x1080 to 640 downscale than in the original frame (about 9 px in the input is about 28 px native, which a human can still see).
- Mechanism check, cheap and untrained: 2x2 overlapping tiles (60% of the frame each) plus the full frame, merged with NMS 0.6, on 300 IDD and 120 FLIR local images:

| | recall at 0.35 | precision at 0.35 | AP50 |
| :-- | --: | --: | --: |
| IDD full frame | 0.452 | 0.862 | 0.566 |
| IDD tiled | 0.613 | 0.646 | 0.580 |
| FLIR day full frame | 0.564 | 0.864 | 0.674 |
| FLIR day tiled | 0.730 | 0.797 | 0.792 |

Tiling lifts recall in every bucket (IDD 9 px: 0.05 to 0.20; 14 px: 0.26 to 0.54), gains +0.12 AP50 on FLIR and +0.01 on IDD. The IDD AP gain is limited by precision (duplicates across tile seams, and unlabelled small people counted as false positives). The merge here is a naive prototype with no tuning; it shows the direction, not a final number. Cost is about 5 forward passes per frame.

## 3. Why v4 and v5 gained nothing

The detector is recall-limited by input resolution, not by data volume or label quality: 25% of day boxes are under 11.2 px at input and recall there is 5%. Re-weighting (v4) and 10k more crowd photos (v5) both leave inference at 640 px for 1080p scenes. Open Images crowd photos are also closer-range than IDD/VisDrone far-field people. Training at 960 px was only evaluated at 640, where the extra resolution cannot help the far field.

## 4. Data search (licence read from the primary source; no downloads)

| candidate | URL | licence as read | size, viewpoint | verdict |
| :-- | :-- | :-- | :-- | :-- |
| Aerial Dataset for Car and Person Detection (Zenodo) | https://zenodo.org/records/21216709 | "Creative Commons Attribution 4.0 International" | 11 videos, about 13.8k frames, drone at 40 m and 60 m over a parking area, human and car boxes, YOLO and VOC | usable licence; one site, one day, not a street or crowd set; small diversity |
| Open Images V7 | https://storage.googleapis.com/openimages/web/factsfigures_v7.html | annotations CC BY 4.0, images CC BY 2.0 | already used in v5 | already in use, no gain |
| Cchead | not opened | search snippet only says CC BY 4.0; unverified | 2.37M head boxes, overhead and slope views | head boxes not person boxes, and licence unverified; do not use |
| WILDTRACK | https://www.epfl.ch/labs/cvlab/data/data-wildtrack/ | no licence stated on the page | 7 elevated GoPro views, 400 annotated frames | reject (no licence found) |
| CrowdHuman | https://www.crowdhuman.org/ | no licence stated, only "Copyright Megvii Inc. 2018" | ground-level crowds | reject |
| P-DESTRE | http://p-destre.di.ubi.pt/ | "freely available", no licence stated | UAV, box per pedestrian | reject until a licence is found |
| BDD100K | (search results only) | data terms limit commercial rights to members and affiliates | dashcam | reject |
| KAIST | search result | CC BY-NC-SA | | reject (NC) |
| Kaggle "People Detection" | search result | CC BY-ND 4.0 | | reject (ND) |
| VisDrone (already in training) | https://github.com/VisDrone/VisDrone-Dataset | the primary page states no licence; the repo's own `DATASET_SPEC.md` records CC BY-NC-SA 3.0, academic use | UAV, the 0.37 source | existing licence risk, see recommendation |

Result: no licence-clear daytime elevated or dashcam crowd dataset with person boxes was found beyond what v5 already used. The only verified CC BY item is a small single-site drone set.

## 5. Realistic day target and recommendation

Realistic with this data, the current 640 px model, no new data: day person AP50 stays near 0.55 (daylight 0.49). With tiled inference, FLIR-like far-field scenes gain about +0.1 AP50 and recall +0.15 to +0.17, so a day figure in the 0.60-0.65 range is plausible after tuning (estimate, not measured on the full val). A 0.85 AP50 is reachable only on subsets where people are at least about 20 to 25 px tall at input (IDD 0.83, FLIR 0.865, LLVIP 0.85 at the 22.6 px cut) and not on the all-size day slice; the all-size target would need people of 5 to 10 px to be found, which neither labels nor the optics support.

Recommendation, in order:

1. Report the 0.85 target against a size-qualified slice (person height at least 22.6 px at input, about 47% of day GT), alongside the all-size number, with the cut stated openly. This is a reporting choice, not a model improvement; the all-size figure must stay published.
2. Prototype tiled or higher-resolution inference for the far field (edge cost about 5 passes; can be limited to regions of interest). Tune seam duplicate handling, then measure on the full official val. This addresses the dominant cause (small-person recall); retraining at 960 px must be evaluated at 960 px, not 640.
3. Split the day slice into daylight and night in the headline: the 0.549 mixes 0.487 daylight with 0.813 night.
4. Re-label an audited slice (about 300 IDD and 300 FLIR day images: ignore regions for people inside vehicles, crowd, boxes under 12 px; add missing crowd people) as a separate, clearly named file; expected effect on AP50 is only +0.02 to +0.04, so it is a hygiene item, not the fix.
5. Resolve the VisDrone licence question (CC BY-NC-SA 3.0 per the repo docs) before it is used in published weights; it also holds the lowest AP (0.37) and is un-audited here.
6. Do not train further on re-weighted or Open Images crowd data; two rounds show no gain.

Blockers and caveats: VisDrone val images and the exact 6136-image IDD val list are not in the local copy (needed to reproduce the official figures exactly); 68 of the intended 150 images were reviewed by eye; the pred-without-GT counts rest on 15 predictions.
