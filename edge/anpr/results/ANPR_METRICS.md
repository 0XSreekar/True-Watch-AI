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

**Not yet run.** `edge/anpr/notebooks/kaggle_ocr_finetune.ipynb` fine-tunes
`devanagari_PP-OCRv5_mobile_rec` on the full 20,000+ synthetic train split on
Kaggle's free GPU tier. This section is filled in by pasting the notebook's
`finetuned_eval.json` output (produced by the SAME `evaluate.py` used for
section 1, so the two are directly comparable) once the orchestrator has run
it — this agent explicitly did not run it or upload anything, per the task's
instructions.

| Metric | Devanagari head (fine-tuned) |
|---|---|
| Exact-match rate | NOT YET MEASURED |
| Mean CER | NOT YET MEASURED |

| Bucket (proxy) | Count | Exact-match | Mean CER |
|---|---|---|---|
| near (>= 200px, proxy) | — | — | — |
| mid, under-25m proxy (120-199px) | — | — | — |
| far, beyond-25m proxy (<120px) | — | — | — |

**Verdict against the 85% target — fine-tuned: NOT YET MEASURED.**

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
