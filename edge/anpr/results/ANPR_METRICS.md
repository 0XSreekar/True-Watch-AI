# ANPR recognition metrics

Slide 5 target: **85% of Nepali plates read correctly end-to-end, under 25 m.**
`docs/MEASUREMENTS.md` section 4 is the honest baseline this phase must not
overstate: on a synthetic plate, PP-OCRv5 Devanagari read it correctly
(0.989/0.990 confidence); the Latin model returned "9 9238" — wrong. On a
REAL plate, the pretrained Devanagari model read about half the characters.
The three numbers below are reported **separately and are never blended**:
synthetic, real, and the Latin-model baseline on the same images.

## 1. Pretrained baseline — BEFORE fine-tune

Measured with `edge/anpr/evaluate.py` against the synthetic corpus produced by
`datasets/plates/gen_plates.py --count 20500 --seed 42`
(20,500 samples; 18,424 train / 2,076 val, split by serial so no serial
appears in both — `datasets/plates/plates.yaml` `split.by: serial`). The
recognition heads used are the PRETRAINED `devanagari_PP-OCRv5_mobile_rec` and
`en_PP-OCRv5_mobile_rec` — no fine-tuning applied yet.

Reproduce with (see "Running the measurement" at the end of this file for the
full sequence, including the confusion-table build step that must run first):

```bash
python datasets/plates/gen_plates.py --count 20500 --seed 42 --out <scratch>/plates
python edge/anpr/evaluate.py --labels <scratch>/plates/labels/rec_gt_val.txt \
    --images-root <scratch>/plates --out edge/anpr/results/eval_synthetic_pretrained.json
```

### 1.1 Synthetic test set (2,076 held-out val images)

Measured by running `edge/anpr/evaluate.py` (script routing set to `auto`,
the production path) over the full 2,076-image val split of the seed-42,
20,500-sample corpus.

| Metric | Devanagari head (pretrained, auto-routed) |
|---|---|
| Exact-match rate | **35.60%** (0.3560) |
| Mean CER | 0.4626 |
| Sample count | 2,076 |

Distance-bucketed breakdown (bucket is a documented PROXY — rendered plate
width in px, per `datasets/plates/degrade.py`'s 60-260px scale range and its
comment "a plate at a barrier is 260 px wide; a plate at 40 m is 60 px". No
physical camera-to-plate distance was measured; this is not a claim of
measured range):

| Bucket (proxy) | Count | Exact-match | Mean CER |
|---|---|---|---|
| near (>= 200px, proxy) | 630 | 44.60% | 0.3707 |
| mid, under-25m proxy (120-199px) | 817 | 36.96% | 0.4563 |
| far, beyond-25m proxy (<120px) | 629 | 24.80% | 0.5628 |

Accuracy degrades with range, as expected, and even the near bucket is well
short of the 85% target — this is the pretrained baseline the plan predicts
would need fine-tuning, and it does.

### 1.2 Latin-model baseline, same synthetic images

The Latin head (`en_PP-OCRv5_mobile_rec`) run on the SAME 2,076 val images
(`--script latin`, bypassing script routing), scored against the SAME
Devanagari ground truth (the plate's true text is Devanagari; this number
answers "what would an off-the-shelf Latin-only ANPR stack have read", which
is the point `compare_latin.py` demonstrates on one image and this table
demonstrates at scale):

| Metric | Latin head |
|---|---|
| Exact-match rate | **0.00%** (0/2,076) |
| Mean CER | 1.0061 (worse than the trivial empty-string baseline of 1.0 — the Latin head does not just fail to read the plate, it actively emits wrong characters) |

The gap between 1.1 (35.60%, pretrained, no fine-tuning) and 1.2 (0.00%) is
exactly the claim MEASUREMENTS.md section 4 and this phase's innovation claim
(a) rest on: a Devanagari-aware head — even before fine-tuning — reads some
Nepali plates; a Latin-only head reads none.

### 1.3 Real plate photos

**INSUFFICIENT SAMPLE.** `DATASET_SPEC.md` section 6.5 requires >= 30 licensed
real-plate crops, each with source URL, licence and author recorded in
`datasets/plates/labels/real_eval_sources.csv`, before a real-plate number can
be reported. Zero such crops exist at the time of this measurement (no
Nepal-border check-post photography with a clear redistribution licence was
located, and the team has not yet photographed any). The only real-plate
observation available is the one already recorded in `docs/MEASUREMENTS.md`
section 4 (pretrained model read about half the characters on one real photo)
— that is a single-image anecdote, not a measured rate, and is not
extrapolated into a percentage here.

### 1.4 Verdict against the 85% target — pretrained

**NOT MET.** Synthetic exact-match is 35.60% against an 85% target (1.1), and
**INSUFFICIENT SAMPLE** for any real-plate claim (1.3). This is the expected,
documented reason fine-tuning is required — see section 2. It is also, on its
own, evidence for innovation claim (a): even without fine-tuning, the
Devanagari-aware pretrained head already reads more than a third of plates
exactly, where the Latin-only baseline (1.2) reads zero.

## 2. After fine-tuning

Fine-tuned locally on CPU (Apple M5, PaddlePaddle 3.3.1 CPU build, PaddleOCR v3.3.0
`tools/train.py`, 8 threads), because the Kaggle GPU quota was in use by detector
training. Config: `edge/anpr/configs/devanagari_ppocrv5_mobile_rec_lines_cpu.yml`
(pretrained `devanagari_PP-OCRv5_mobile_rec` weights, 1 epoch, Adam, cosine LR from
3e-4, no warmup, batch 32; 997 iterations in 3 h 25 min at ~2.7 samples/s).

**Training data is per-line crops, not whole plates.** A first run fed whole
two-line plates into the single-line 48x320 input; the untouched pretrained model
scores exact-match 0.0 on that input, so that run was stopped at step 250 and the
data rebuilt with `edge/anpr/scripts/build_line_labels.py`, which cuts each plate
with the same `rectify.prepare_lines()` the pipeline runs at inference: 31,922 line
crops from the 15,961 of 18,424 training plates that split into two lines (the rest
skipped, never mislabelled). Per-line validation inside training: exact 0.9194,
normalised edit similarity 0.9697.

End to end through the full pipeline (`evaluate.py`, detection-free crops ->
rectify -> line split -> recognise -> postprocess), on the SAME 2,076-image
synthetic validation split as section 1, with the fine-tuned model loaded through
`ANPR_DEVANAGARI_MODEL_DIR`:

| Metric | Pretrained (section 1.1) | **Fine-tuned, round 1** |
|---|---|---|
| Exact-match rate | 35.60% | **77.26%** |
| Mean CER | 0.4626 | **0.1106** |

| Bucket (proxy) | Count | Exact-match, pretrained | **Exact-match, fine-tuned** | Mean CER, fine-tuned |
|---|---|---|---|---|
| near (>= 200px, proxy) | 630 | 44.60% | **82.38%** | 0.0861 |
| mid, under-25m proxy (120-199px) | 817 | 36.96% | **78.82%** | 0.1058 |
| far, beyond-25m proxy (<120px) | 629 | 24.80% | **70.11%** | 0.1415 |

Latin-model baseline on the same images is unchanged (section 1.2: 0.00%).

### 2.1 Line-split fix and round 2

About 12% of plates did not split cleanly into two lines, and a single-line
recogniser cannot read an unsplit plate. `rectify.py` now measures ink inside the
printed border (commit c00538a), which cuts unsplit plates from 11.8% to 2.6%.
Round 2 then continued from round 1's `best_accuracy` on line crops re-cut with the
fixed splitter: 35,880 crops from 17,940 of 18,424 training plates, learning rate
1.5e-4 (half of round 1), otherwise the same schedule
(`edge/anpr/configs/devanagari_ppocrv5_mobile_rec_lines2_cpu.yml`; 1,121 iterations
in 4 h 03 min on the M5 CPU). Per-line validation inside training: exact 0.9313,
normalised edit similarity 0.9756 (round 1: 0.9194, 0.9697).

All four rows below run the current pipeline (fixed splitter) on the same 2,076
validation plates, so the pretrained row is re-measured, not copied from section 1.1:

| Model | Exact-match | Mean CER | near | mid (under 25 m) | far |
|---|---|---|---|---|---|
| Pretrained | 41.38% | 0.3783 | 51.27% | 43.82% | 28.30% |
| Round 1 | 86.37% | 0.0562 | 92.06% | 89.72% | 76.31% |
| Round 2, `best_accuracy` (step 997) | 87.76% | 0.0490 | 93.33% | 90.45% | 78.70% |
| **Round 2, final (step 1,121) — published** | **87.91%** | **0.0481** | 93.17% | 90.33% | 79.49% |

The two round-2 checkpoints differ by three plates, which is within noise; the final
one is published because it is ahead overall and on the far bucket. Choosing between
them on this split makes the published figure slightly optimistic.

Summaries: `eval_synthetic_pretrained.json` (section 1.1), `eval_synthetic_pretrained_splitfix.json`,
`eval_synthetic_finetuned_round1.json`, `eval_synthetic_finetuned.json` (round 2, published) and
`eval_synthetic_latin.json` in this directory.

Published: [sreekar12/truewatch-anpr-devanagari](https://huggingface.co/sreekar12/truewatch-anpr-devanagari),
pinned in `hf_ocr_model.json` (commit sha and per-file sha256).

**Verdict against the 85% target — fine-tuned, round 2:**
- Synthetic, under 25 m (near + mid proxies, 1,447 plates): 91.6% exact —
  **MET on synthetic plates** (target 85%; round 1 with the fixed splitter 90.7%,
  round 1 before it 80.4%).
- Real plates: **INSUFFICIENT SAMPLE** (section 1.3). No synthetic figure is
  extrapolated to real plates, so the slide-5 target is not claimed as met for
  real traffic.

### 2.2 Real photographs, real layouts and rounds 3 to G3

**Real test sets.** Crops of distinct vehicles from the Hugging Face dataset `mukulboro/nepali-private-license-plates`
(CC BY 4.0; Kathmandu University parking-lot photographs taken with the owners' permission), each transcribed twice,
independently. *Strict* (57 plates) keeps plates both readings agree on exactly with both confident;
*extended* (147 plates) keeps every exact agreement. One crop per vehicle, never used for training, and every
training plate whose text matched a test plate was removed. The sets are small: treat differences of a few points as noise.

**What the photographs showed.** Real two-line plates print zone, lot and class on the top line over a larger serial;
most private plates are red with white text; one-line plates, two- and three-line province plates and embossed Latin
plates are common; many crops are tilted and 50-100 px tall. The pipeline now keeps one-line plates at their own
aspect, splits province plates in three when a two-line reading fails, levels tilted crops (`rectify.deskew`), accepts
province and embossed registrations, and cuts a reading down to the most plausible registration inside it when junk
surrounds it (`recognise.read_plate`). All rows below use that same final pipeline.

**Training rounds.** Round 3 added 44 open-licence fonts (fonts that draw Devanagari digits with Latin shapes are
dropped automatically); round 4 the real layouts, one-line plates and the first real line crops; rounds G1-G3 ran on a
Kaggle T4 (PaddlePaddle 3.3.1, same PaddleOCR v3.3.0) with province plates, more verified real crops from
`ishworsubedii/vehicle-number-plate-datasetnepal` (Apache-2.0) and from a training subset of the CC BY 4.0 set, and
repeat views of verified plates matched to their text.

Exact-match / mean CER:

| Model | Real, strict | Real, extended | Synthetic, original | Synthetic, 44 fonts | Synthetic, real layout | Synthetic, one-line | Synthetic, province |
|---|---|---|---|---|---|---|---|
| Round 2 (published until 2026-09-26) | 24.6% / 0.569 | 10.2% / 0.684 | 88.1% / 0.046 | 56.4% / 0.135 | 29.7% / 0.254 | 11.8% / 0.465 | 0.0% / 0.806 |
| Round G1 | 36.8% / 0.350 | 20.4% / 0.446 | 87.1% / 0.051 | 80.3% / 0.079 | 78.3% / 0.075 | 57.7% / 0.168 | 36.4% / 0.281 |
| Round G2 | 36.8% / 0.353 | 23.1% / 0.425 | 87.4% / 0.051 | 80.6% / 0.078 | 79.0% / 0.073 | 58.9% / 0.160 | 39.5% / 0.252 |
| Round G3 (published 2026-09-26, superseded) | 38.6% / 0.347 | 25.2% / 0.413 | 87.8% / 0.051 | 81.4% / 0.076 | 79.8% / 0.069 | 60.7% / 0.151 | 40.9% / 0.244 |

Round G3 was published as revision `a428bfb`; section 2.3 records the rounds that replaced it.

**Verdict at round G3 (superseded by section 2.3):** met on synthetic plates under the 25 m proxy (section 2.1); **NOT MET on real
photographs** (38.6% strict, 25.2% extended). The limits are the amount of real training data
(a few hundred verified lines from under a hundred vehicles), crop resolution and tilt, and province headers that are
too small to read at range. Labelled plates from the deployment cameras are the next step that matters.

### 2.3 Roboflow photographs and rounds G4 to G6

**New real training data.** Three Roboflow Universe datasets, each CC BY 4.0: `sushant-kvyez/license-plate-s8svk`
(v11), `prajwal-ghimire-tngts/nepali-characters-detection` (v8) and `college-fyp-ewmvi/license-plate-detection-system-nktef`
(v4). Their per-character boxes were turned into line transcriptions, kept only when they parse as a valid Nepali
registration, and de-duplicated by image hash: 3,239 training lines from 1,853 plates. Plates whose embossed-series
characters were ambiguous were excluded. Round G4 added the first of these sets; G5 all three; G6 continued from G5
with inline-header province plates (a small province header above the registration, rendered synthetically) and crops
of those headers, read on their own by `recognise.read_plate`.

**Roboflow held-out set, and a leak found in it.** The three sets' own test splits were held out (208 photographs).
An audit on 2026-09-29 found that 94 of those 208 photographs show a vehicle whose registration text also occurs in the
training lines: these sets photograph the same vehicles many times. Only the 114 plates whose text never occurs in
training are reported below; the all-208 figures (G4 60.1%, G5 76.0%, G6 74.5%) are contaminated and are not used.
The two strict/extended real sets were re-checked against the G6 training bundle the same way: none of their 57 / 147
plate texts occurs in it.

Exact-match / mean CER, same final pipeline as section 2.2:

| Model | Real, strict (57) | Real, extended (147) | Roboflow held-out (114) | Synthetic, original (2,076) | Synthetic, one-line (993) | Synthetic, province (795) |
|---|---|---|---|---|---|---|
| Round G3 | 38.6% / 0.347 | 25.2% / 0.413 | – | 87.8% / 0.051 | 60.7% / 0.151 | 40.9% / 0.244 |
| Round G4 (published 2026-09-26, rev `5e35fd9`) | 42.1% / 0.304 | 27.9% / 0.351 | 59.6% / 0.125 | 87.9% / 0.051 | 59.1% / 0.157 | 39.6% / 0.257 |
| Round G5 | 49.1% / 0.253 | 34.0% / 0.302 | **78.1% / 0.069** | 87.7% / 0.052 | 60.9% / 0.154 | 39.5% / 0.251 |
| **Round G6 — published** | **52.6% / 0.262** | **38.8% / 0.304** | 72.8% / 0.105 | 88.0% / 0.052 | 61.0% / 0.152 | 40.4% / 0.223 |

G5 and G6 split the real sets: G6 is better on both fully independent sets (strict and extended), G5 on the Roboflow
held-out plates. G6 was published because the strict and extended sets share no vehicles or sources with any training
data. Pooled over extended + Roboflow held-out the two are level (140 vs 139 plates read exactly). The G6 strict figure
was re-run on 2026-09-29 with the committed code and reproduced exactly (52.63%, CER 0.2618).

Published: [sreekar12/truewatch-anpr-devanagari](https://huggingface.co/sreekar12/truewatch-anpr-devanagari) revision
`064b6a0` (`hf_ocr_model.json`, byte-for-byte verified after upload). The edge service downloads that pinned revision
at first use and checks every file's SHA-256 (`edge/anpr/ocr_weights.py`).

Summaries: `eval_real_hf_strict_finetuned.json`, `eval_real_hf_extended_finetuned.json`,
`eval_synthetic_finetuned.json` (all G6) and `eval_real_roboflow_clean.json` (G4-G6).

**Verdict against the 85% target (slide 5, real plates end to end under 25 m): NOT MET.** Best real-photograph
results are 52.6% (57 plates, strict), 38.8% (147, extended) and 72.8% (114, Roboflow held-out). The 88.0% synthetic
figure is a synthetic result and is not evidence about real plates. None of the real sets is range-labelled, so no
figure here is an "under 25 m" measurement either. The limits remain the amount of varied real training data,
crop resolution and tilt, and plate location: in the live pipeline plates are still found by classical image
processing (`detect_plate.py`), which found 3 of 13 plates on an earlier real-image check; a trained plate locator was
prepared but never trained.

### 2.4 Embossed (Latin) plate head: fine-tune measured and not adopted

A Latin recogniser was fine-tuned from `en_PP-OCRv5_mobile_rec` on Kaggle for embossed plates ("B AC 5763"): synthetic
embossed plates, glyph crops from Kaggle `theciceerguy/embossed-number-plate-detection-nepal-dataset` (MIT), rows cut
from Zenodo record 22861166 (CC BY 4.0, split by vehicle) and 23 verified real lines. On its own test lists it scored
99.9% (956 line crops) and 100% (24 hand-cut, eye-checked real line crops). Those are line-level numbers on crops
prepared the way the training crops were.

The check that matters is end to end. On the 28 embossed plates among the 147 extended real test plates, whole plate
crops through the full pipeline (G6 Devanagari head in both rows, `eval_real_embossed_latin.json`):

| Latin head | Exact-match | Mean CER | Plates |
|---|---|---|---|
| Pretrained `en_PP-OCRv5_mobile_rec` (in use) | 57.1% | 0.403 | 28 |
| Fine-tuned | 21.4% | 0.648 | 28 |

The fine-tune reads the pipeline's crops worse than the pretrained head and, because its weaker readings fail the
plate grammar more often, sends more plates to the Devanagari head. It is **not integrated and not published**; the
pipeline keeps the pretrained Latin head. Its weights are archived for a later round that trains on crops cut by
`rectify.py` itself. Sample size is small (28 plates).

## 3. Plate locator (learned) vs the classical finder

Added after the classical finder (`detect_plate.find_plates_in_crop`) located only 3 of 13 plates on real
images. A single-class YOLO11n plate detector (training script `training/plate_locator/`, Kaggle T4,
80 epochs, imgsz 640, about 1.2 GPU hours) is exported to ONNX and run with onnxruntime
(`anpr/plate_locator.py`); the classical finder remains the fallback. All numbers are REAL photographs
(no synthetic data in either table). Weights are not committed; the pinned manifest is
`results/plate_locator_model.json` (sha256 `4f3a0dc2...499ee`, still marked unpublished).

### 3.1 Locating the plate, held-out test split of the plate-box dataset (never trained on)

642 distinct full images, 683 plate boxes (Nepali road photos 549, parked-vehicle photos 88, other 5).
Classical finder was given the whole image as the "vehicle crop", top-1 box; the locator uses conf >= 0.25.
IoU >= 0.5 counts as found.

| finder | recall | precision | mAP50 |
|---|---|---|---|
| classical (top-1) | 0.138 | 0.148 | 0.043 |
| ONNX locator | 0.972 | 0.926 | 0.968 |

The training run's own test-split figure was mAP50 0.984, P 0.962, R 0.955. It was computed over 1,284
images because the Kaggle mount listed each file twice; this repository's figure above uses the
642 distinct images. Scoring is by source mix of one dataset; it does not show how the locator behaves on
camera types absent from it (night/IR, far range).

### 3.2 End to end on full photographs: locate, crop, read (plate exact match)

147 full photographs of Nepali plates with a hand-labelled plate text (57 of them the strict subset), none
of which is in the locator's training, validation or test split. Reader: the fine-tuned Devanagari/Latin
recogniser of section 2, unchanged. "Oracle" crops the labelled box, giving the ceiling set by the
recogniser alone.

| locating method | extended (n=147) | strict (n=57) |
|---|---|---|
| classical finder | 13 (8.8%) | 3 (5.3%) |
| ONNX locator | 63 (42.9%) | 31 (54.4%) |
| ground-truth box (ceiling) | 61 (41.5%) | 30 (52.6%) |

The locator removes the locating loss: end to end it now matches the recogniser's own ceiling. The
overall plate-read rate is therefore bound by the recogniser (section 2.3). The 85% real-data target
remains NOT MET (best real figure above: 54.4% strict, 42.9% extended).

## Running the measurement

```bash
# 1. Generate the corpus (not committed; datasets/plates/.gitignore covers it).
python datasets/plates/gen_plates.py --count 20500 --seed 42 --out <scratch>/plates

# 2. Build the confusion table from the TRAIN split (postprocess.py reads it).
python edge/anpr/evaluate.py \
    --labels <scratch>/plates/labels/rec_gt_train.txt \
    --images-root <scratch>/plates \
    --limit 3000 \
    --write-confusions edge/anpr/data/confusions.json

# 3. Evaluate the VAL split with automatic script routing (section 1.1;
#    never the train split the confusion table above was built from).
python edge/anpr/evaluate.py \
    --labels <scratch>/plates/labels/rec_gt_val.txt \
    --images-root <scratch>/plates \
    --out edge/anpr/results/eval_synthetic_pretrained.json

# 4. Evaluate the SAME val split forcing the Latin head (section 1.2).
python edge/anpr/evaluate.py \
    --labels <scratch>/plates/labels/rec_gt_val.txt \
    --images-root <scratch>/plates \
    --script latin \
    --out edge/anpr/results/eval_synthetic_latin_baseline.json
```

`evaluate.py` logs `summary` (count, exact_match_rate, mean_cer) and one
`bucket` line per distance bucket to stdout, and (with `--out`) writes the
same numbers plus every per-sample prediction to JSON. `--script latin` (or
`--script devanagari`) bypasses script routing and forces one head, which is
how the Latin-baseline table in 1.2 was produced — the same comparison
`compare_latin.py` demonstrates on one image, run here over the full val set.

The numbers in sections 1.1 and 1.2 above were produced by exactly these
commands against a 20,500-sample corpus built with `--seed 42`; the
per-sample JSON outputs are not committed (they are large and regenerable),
but are reproducible byte-for-byte from the seed.
