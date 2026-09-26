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
| **Round G3 — published** | 38.6% / 0.347 | 25.2% / 0.413 | 87.8% / 0.051 | 81.4% / 0.076 | 79.8% / 0.069 | 60.7% / 0.151 | 40.9% / 0.244 |

Published: [sreekar12/truewatch-anpr-devanagari](https://huggingface.co/sreekar12/truewatch-anpr-devanagari) revision
`a428bfb` (`hf_ocr_model.json`).

**Verdict against the 85% target:** met on synthetic plates under the 25 m proxy (section 2.1); **NOT MET on real
photographs** (38.6% strict, 25.2% extended). The limits are the amount of real training data
(a few hundred verified lines from under a hundred vehicles), crop resolution and tilt, and province headers that are
too small to read at range. Labelled plates from the deployment cameras are the next step that matters.

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
