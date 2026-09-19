# TRUEWATCH — measurements we ran ourselves

Every number below came from running the real models on public research
datasets. Nothing is estimated. This file is the authority for any figure this
project presents as measured; if a judge asks "how do you know", this is the
answer. See `docs/ARCHITECTURE_V2.md` §7 for the figures that are targets on
hardware this project does not own, and `docs/PHASE_MINUS1_SCOPE.md` §5.2 for
the full measured-versus-target split.

## 1. Night and infrared

Source: KAIST Multispectral Pedestrian, set00/V007, paired visible + LWIR frames.
Model: YOLO11-s, pretrained on COCO, no fine-tuning.

| | Visible | Infrared (LWIR) |
|---|---|---|
| Person detections | 9 | 8 |
| Mean confidence | 0.59 | 0.54 |
| IR recovery | - | 89% of visible |

On false-colour thermal imagery (separate public set, 61 images, 43 persons):
AP@50 = 0.133. Precision 0.364, recall 0.093 at conf 0.50.

### Why the motion channel carries the night
Dense Farneback optical flow over 100 paired frames, measured inside detected
object boxes versus background:

| | Flow on moving objects | Background | Contrast |
|---|---|---|---|
| Visible | 5.17 px | 2.74 px | 1.9x |
| LWIR | 2.73 px | 0.81 px | **3.4x** |

Absolute flow is lower on IR, but a moving person stands out 1.8x more from the
background. That is the justification for weighting motion higher after dark.

NOTE: the earlier wording "optical flow is unaffected by IR" was wrong and has
been removed from the deck. Whole-frame flow on LWIR is only 32% of visible.
The correct claim is the contrast ratio above.

## 2. Suspicious activity without a dataset

Source: 100 consecutive KAIST visible frames. YOLO11-s + ByteTrack.
14 tracks formed, 10 lasting 8+ frames.

Baseline learnt from the camera itself: majority flow left to right, 6 with, 4 against.

Rules: virtual fence crossing at x=320, wrong direction against the learnt
baseline, loitering (dwell >= 3 s and net displacement < 90 px).

| id | frames | dwell | net px | path px | rule fired |
|---|---|---|---|---|---|
| 1 | 70 | 3.5 s | 244 | 279 | - |
| 2 | 27 | 1.4 s | 151 | 158 | wrong direction |
| 3 | 33 | 1.6 s | 190 | 202 | wrong direction |
| 4 | 96 | 4.8 s | 316 | 367 | fence crossed |
| 5 | 20 | 1.0 s | 99 | 109 | wrong direction |
| 6 | 98 | 5.0 s | 70 | 163 | fence crossed + loitering |
| 9 | 92 | 4.7 s | 162 | 213 | fence crossed |
| 12 | 11 | 1.2 s | 3 | 10 | - |
| 26 | 13 | 0.7 s | 22 | 27 | - |
| 28 | 10 | 0.5 s | 34 | 35 | - |

Result: 10 tracks evaluated, 6 alerts, 4 normal. No labelled training data.
Every alert names the rule that fired.
Track 6 is the interesting one: 70 px net movement but 163 px path travelled,
i.e. it moved back and forth in place. That is what loitering looks like.

## 3. Objects too small to detect

Source: 23 KAIST visible frames, 51 person detections at native resolution,
median person height 77 px. Progressive downscale, conf 0.35.

| Person height | Detections | Recall vs native |
|---|---|---|
| 77 px | 51 | 100% |
| 54 px | 51 | 100% |
| 38 px | 51 | 100% |
| 27 px | 52 | 102% |
| 19 px | 46 | 90% |
| 14 px | 41 | 80% |
| 9 px | 25 | 49% |

The cliff is between 19 px and 9 px. This is the measured basis for tiling the
far field at full resolution and for grading cameras by DORI.

## 4. Nepali plate reading

Synthetic plate (rendered by us) composited onto a public vehicle photo at a
check-post barrier, with blur, dimming and sensor noise added.

| Model | Read | Result |
|---|---|---|
| Latin OCR (what off-the-shelf ANPR uses) | "9 9238" | wrong |
| PP-OCRv5 Devanagari | बा १२ प १२३४ | correct, 0.989 / 0.990 |

On a real (copyrighted, unusable) photo of a Nepali plate, the pretrained
Devanagari model read about half the characters correctly and the Latin model
returned nothing usable. Fine-tuning on Nepali plates is therefore required to
reach the 85% target. Do not quote the real-plate figure without that context.

## Datasets, all verified available

- KAIST Multispectral Pedestrian — Hugging Face, `richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset`, 23,210 files, paired visible/lwir
- LLVIP — Hugging Face, `jsonhash/LLVIP`
- IDD Detection — idd.insaan.iiit.ac.in, 46,588 images (31,569 / 10,225 / 4,794)
