---
license: agpl-3.0
library_name: onnx
pipeline_tag: object-detection
tags:
- object-detection
- license-plate-detection
- yolo11
- onnx
- nepal
---

# TRUEWATCH plate locator — version 0.2.0 (prototype)

A single-class detector that finds vehicle licence plates in a photo or a vehicle crop, so that a separate
recogniser can read them. It is the plate-finding step of the number-plate reader in the TRUEWATCH border-post
monitoring prototype (source: https://github.com/0XSreekar/True-Watch-AI, `edge/anpr/`). It does not read the
characters; pair it with a recogniser such as `sreekar12/truewatch-anpr-devanagari`.

**Status: research prototype.** Validated on public Nepali plate photographs only. Read the limitations before any use.

## Interface

- File `plate_locator.onnx`, FP32, about 11 MB, fine-tuned from YOLO11n (Ultralytics).
- Input `images`: float32 (batch, 3, H, W), RGB scaled to 0-1. The reference code letterboxes onto a 640 x 640
  canvas with 114-grey padding.
- Output `output0`: float32 (batch, 5, anchors): rows 0-3 are cx, cy, w, h in input pixels, row 4 is the plate score
  after the sigmoid. No NMS in the graph. The class is stored in the file's metadata as `item` (= plate).
- Reference pre/post-processing (letterbox, threshold, NMS at IoU 0.5, mapping back to image pixels):
  `edge/anpr/plate_locator.py` in the source repository.

```python
import cv2
from anpr.plate_locator import PlateLocator      # from the source repository, edge/ on the path

locator = PlateLocator("plate_locator.onnx", conf=0.25)
for (x1, y1, x2, y2), score in locator.locate(cv2.imread("vehicle.jpg")):
    print(round(score, 2), int(x1), int(y1), int(x2), int(y2))
```

## Evaluation (all numbers measured; script `edge/anpr/scripts/eval_plate_locator.py`)

A detection counts when it overlaps a labelled plate at IoU >= 0.5. Operating threshold 0.25 (recall first: a missed
plate cannot be read later, a false box is checked by the recogniser's plate grammar).

| Test | Result |
|---|---|
| Held-out test split, same sources as training, unseen images (642 images, 683 plates) | recall 97.2% (95% CI 95.7-98.2), precision 92.6% (90.5-94.3), AP50 0.966 |
| New camera source never trained on (48 random street photos, detections reviewed by eye) | recall 65.9% (95% CI 51-78), 29 found / 15 missed, 1 false box |
| Plate-free day and IR scenes (300 images) | false box on 14 images (4.7%); 2 phantom plate readings after the recogniser (0.7%) |
| End to end with the published recogniser, 147 full photos / 57 strict | 42.9% / 54.4% plates read exactly (classical finder: 8.8% / 5.3%; hand-drawn boxes: 41.5% / 52.6%) |

Stress on the held-out test images (threshold 0.60): recall 87.7% darkened, 88.4% blurred, 93.4% at half resolution,
85.1% at quarter resolution, 82.9% IR-like, 91.5% heavy JPEG, 65.0% strong sensor noise; precision >= 97% in every case.

## Version 0.2.0 changes

Fine-tuned from 0.1.0 for low light, near-IR style grey night images, sensor noise and small far plates, using
photometric copies of the training images, small-plate copies, pseudo-labelled street photos and plate-free
negatives. Held-out test recall at 0.60: 95.9% (0.1.0: 95.6%). Stress recall at 0.60, 0.1.0 to 0.2.0: night
87.7 to 95.0%, noise 65.0 to 91.2%, quarter resolution 85.1 to 89.6%, IR-like 82.9 to 87.0%, blur 88.4 to 93.7%.
New-source photos (48, by eye): 52 found / 5 missed against 29 / 15, but other photos of that source were used for
training, so it is no longer a fully independent test. On plate-free scenes the raw false-box count at 0.25 rose
(14 to 35), mostly real plates on night cars the labels missed; boxes that are not plates fell from 11 to 4.
Details: `ANPR_METRICS.md` section 3.4. File sha256 `95dd8cf9d164bbffddbd57bc6131acbf29e51e8a1e0abc08e24f9d2be3b7cfdb`. The evaluation tables above and the limitations
below are the 0.1.0 measurements.

**Physics limit.** Thermal (LWIR) cameras cannot show plate characters. The target conditions are near-IR with an
illuminator, and low light.

## Limitations

- **The in-domain figure overstates real-world performance.** On photos from a camera source it never saw, recall
  fell from 97% to about 66%. Misses are mostly small or distant plates, but also some clear full-size plates.
- Only Nepali plates (Devanagari and embossed Latin) in daylight street and parking photographs were in training.
  Not evaluated on real thermal/IR footage, at night, in rain, or at border-camera ranges; synthetic stress results
  are a proxy, not a measurement of those conditions.
- Plate-free night and IR scenes can produce whole-frame boxes; the reference pipeline only runs the locator on
  confirmed vehicles and discards readings that fail the plate grammar.
- Small samples: the new-source and end-to-end sets are tens to low hundreds of images; treat differences of a few
  points as noise.

## Responsible use

Licence-plate numbers identify vehicles and, through registration records, people. This model is published for
research and evaluation of the TRUEWATCH prototype. Any operational use must have a lawful basis and authorisation,
follow the data-protection and retention rules that apply where it is used, keep a human in the loop for any action
taken on a reading, and must not be used to track people outside that authority. Do not rely on it as the only
source for an enforcement decision.

## Training data and credits

Plate boxes from:
- `ishworsubedii/vehicle-number-plate-datasetnepal` (Kaggle), Apache-2.0.
- `mukulboro/nepali-private-license-plates` (Hugging Face), CC BY 4.0 — Kathmandu University parking-lot
  photographs; images used for the recogniser's held-out real test were excluded.
- `college-fyp-ewmvi/license-plate-detection-system-nktef` v4 (Roboflow Universe), CC BY 4.0.

Near-duplicates were removed and splits were made by source group so the same vehicle never sits in two splits. The
new-source check uses `vehicleannotating/nepali-vehicle-annotation` (Roboflow Universe, CC BY 4.0), which was not
used in training. 0.1.0: free Kaggle T4, YOLO11n, 80 epochs, image size 640. 0.2.0: 30 more epochs on a Mac (MPS), with the street photos of the new-source set (except the 48 review photos) pseudo-labelled and added.

## Licence

AGPL-3.0, inherited from the Ultralytics YOLO11 weights it was fine-tuned from. The training data licences above
require attribution, given in this card.
