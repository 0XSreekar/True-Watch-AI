# TRUEWATCH — training

Fine-tuning, evaluation and export for the one model in the stack that is fine-tuned: the YOLO11-s
appearance channel (slide 3: *"fine-tuning planned on IDD, LLVIP, KAIST"*). These scripts run on free
Kaggle GPU hours, not in CI and not in `edge/`. `edge/` consumes an exported ONNX graph through
onnxruntime and nothing else.

> **Status (Phase 2).** The tooling is complete: the GPU-free test suite passes, and `scripts/smoke_test.py`
> runs every script end to end on a synthetic dataset, including a real SIGKILL-and-resume and a
> hand-truncated checkpoint. **No training run has been done, so there is no measured detector
> accuracy yet.** `results/METRICS.md` says so and lists every slide-5 target as NOT ASSESSED; it
> will not contain an accuracy figure until `make_metrics.py` is run on a real evaluation. The only
> measured figures in `results/` are CPU latency numbers (`benchmark_mac-apple-m5-cpu.json`), taken
> with a randomly initialised YOLO11-s of the right shape and labelled as such. The Hugging Face
> Spaces CPU benchmark, the Kaggle run and the Hub upload are still to do.

## Licence notice: Ultralytics is AGPL-3.0

The `ultralytics` package and YOLO11 are licensed under **AGPL-3.0**. This repository is public and is
itself AGPL-3.0 (`LICENSE` at the root; decided in Phase 1, `docs/ARCHITECTURE_V2.md` §11 item 3). The
AGPL's network clause extends source-availability obligations to software offered over a network, which
is how this system is deployed, so compliance here means keeping the repository public.

Ultralytics stays isolated in `training/` anyway, for three reasons that do not depend on the licence
argument:

1. **Runtime.** `edge/` has to run on a free two-vCPU Hugging Face Space and, later, on a Jetson. It
   loads an ONNX file with onnxruntime. Importing torch and Ultralytics there would cost gigabytes and
   seconds of cold start for no benefit.
2. **Auditability.** The boundary is one command, not an argument:
   `grep -rn "ultralytics" edge/ backend/ frontend/` prints nothing. Inside `training/`, only
   `train.py`, `export_onnx.py` and `_predict.py` import it, and `_predict.py` does so lazily.
3. **Optionality.** If the licence decision is reopened (it says to be, should the Ultralytics terms
   change), the serving path has no Ultralytics code to unpick and the detector can be swapped behind
   the ONNX contract.

Check the current Ultralytics licence text at source before any release beyond this repository. The
model card that `export_onnx.py --push-to-hub` writes states the AGPL-3.0 obligation and says the
training datasets are public research datasets with their own terms that are not redistributed.

## Layout

| Path | Purpose |
|---|---|
| `configs/yolo11s_day.yaml` | Stage 1: the bulk of training, COCO weights to the mixed corpus. Every hyperparameter carries its reason. |
| `configs/yolo11s_ir.yaml` | Stage 2: IR emphasis, starting from stage 1's best weights. |
| `configs/data.yaml` | Points at the Phase 1 output (`datasets/processed/yolo`). No `test:` key, by design. |
| `scripts/train.py` | Training. `--resume`, saves every epoch, logs to CSV, stops itself before the session cap. |
| `scripts/evaluate.py` | mAP@50, mAP@50-95, P, R per class, **day and IR separately**, small-object buckets, CIs. |
| `scripts/eval_hardset.py` | The Phase 1 hard set as a pass/fail regression gate. |
| `scripts/sweep_conf.py` | Confidence and NMS-IoU sweep: F1-optimal point and false-alert-budget point, two separate numbers. |
| `scripts/export_onnx.py` | `best.pt` to ONNX opset 17, dynamic batch, with the torch-vs-onnxruntime parity check. |
| `scripts/benchmark_cpu.py` | p50/p95 CPU latency at 640 px. Needs only numpy and onnxruntime. |
| `scripts/make_metrics.py` | Renders `results/METRICS.md`: measured against slide-5 targets, one verdict each. |
| `scripts/smoke_test.py` | Runs all of the above on a synthetic dataset, including a real kill-and-resume. |
| `scripts/_common.py` `_metrics.py` `_predict.py` `_synth.py` | Shared code. `_metrics.py` is pure numpy and is where the reporting rules are enforced. |
| `notebooks/kaggle_train.ipynb` | The notebook actually run on Kaggle. |
| `tests/` | GPU-free pytest suite. |
| `results/` | Metrics JSON, CSV and MD are committed. Weights never are. |

## Training plan

**One mixed day-and-IR model, trained in two stages of one lineage.** Not a day model plus an IR model.

1. **One graph on the edge.** `docs/ARCHITECTURE_V2.md` §1.1 runs one ONNX graph. Two models would mean two resident in a free CPU Space, and a router between them.
2. **No router.** A day/night classifier's errors would become detector errors at dusk, the hours a border post cares about most.
3. **Slide 4 makes it free.** *"IR replicated to 3 channels"* puts LWIR into the same input tensor as RGB, so no architecture change is needed, and `docs/MEASUREMENTS.md` §1 measured 89% IR recovery from a visible-only COCO model: the representation already transfers.
4. **The smaller domain is the harder target.** IR is the smaller share of the corpus and carries 0.75, not 0.85. Stage 2 exists to give IR more weight without splitting the data.
5. **The budget fits one lineage.** Kaggle gives 30 GPU-hours a week for the whole project; one lineage at ~16.5 GPU-hours (estimate below) fits in two sessions; two lineages do not.

| Decision | Choice | Reason |
|---|---|---|
| Stage 1 | 20 epochs, full corpus, COCO `yolo11s.pt` | The bulk of adaptation, sized to the session budget below. |
| Stage 2 | 6 epochs from stage 1 best weights, LWIR repeated ×2, LR 5× lower | Re-weights toward IR; same corpus, same head. |
| Image size | 640 | The ~30 ms figure is defined at 640 px, and the far field is handled by 640×640 SAHI tiles, so a bigger input is not needed. |
| Batch | 32 (nominal batch 64, so two accumulated steps) | Fits a 16 GB T4 under AMP. |
| Optimiser | SGD, momentum 0.937, weight decay 5e-4 | What Ultralytics would choose above 10,000 iterations; explicit so a resume cannot change it. |
| LR | Stage 1 0.005, cosine to ×0.01; stage 2 0.001 | Half the from-scratch COCO rate: fine-tuning should not overwrite pretrained features. |
| Warmup | 3 epochs (stage 2: 1); bias warmup LR 0.05 = 10× lr0 (stage 2: equal to lr0, no boost) | Momentum and bias warmup for the new 5-class head; Ultralytics' default 0.1 against stage 2's lr0 of 0.001 would have started the trained biases at 100×. |
| Freeze | Layers 0-9 frozen for epochs 0-2, then unfrozen (stage 1) | A freshly initialised head produces large early gradients that would wreck COCO features. |
| EMA | On (Ultralytics always keeps one); `best.pt` and `last.pt` hold EMA weights | Not switchable; stated so nobody looks for the switch. |
| Early stopping | Patience 6 (stage 2: 3) on Ultralytics fitness, restored from `train_log.csv` on every resume | Class-mean fitness keeps vehicles from dying while people improve; per-epoch AP50 of every class is logged for a human to watch. Ultralytics restarts patience on each resume; `train.py` puts it back. |
| Augmentation | Every key of `datasets/config/augment.yaml` applied or reported, never duplicated (see below) | Single source of truth between `datasets/` and `training/`. |
| Class imbalance | Images with `truck` or `cart` appear twice per epoch, no image more than twice, no loss reweighting | `docs/DATASET_SPEC.md` §3.1: oversampling a rare class in a detector mostly multiplies its backgrounds. |
| Sessions | `--max-hours` is a deadline: training stops at the epoch boundary where one more epoch would cross it | The notebook passes the time left in its 12 h session, keeping 45 min for evaluation, export and packaging. |
| Cart gate | A dataset that withdraws class 4 (four names in its `data.yaml`, or five with no cart label) trains the same 5-output head; class 4 gets no positives | `DATASET_SPEC.md` §1.5 keeps id 4 reserved, so the wire names and the ONNX output shape do not change. Preflight fails if a withdrawn class still has label rows. |

None of these values has been tuned: they were decided in advance from the reasons above, and the
first Kaggle epochs are the first evidence for or against them. The time budget is an **estimate**,
derived at the top of `configs/yolo11s_day.yaml`: ~80k images at 640 on one T4, paced by Kaggle's
four vCPUs at an assumed ~45 img/s, gives ~35 min per stage-1 epoch and ~48 min per stage-2 epoch,
so ~11.7 h + ~4.8 h = ~16.5 GPU-hours, two sessions of ~10 h of training each. `train_log.csv`
records seconds per epoch; check it after epoch 1. If an epoch takes much over 50 min, start a fresh
run with a smaller `--epochs` (it is ignored on resume) rather than let the stages spill into a
third or fourth session.

### Augmentation: every policy key is applied or reported

`train.py` routes each key of `datasets/config/augment.yaml` and stops (exit 3) on a key it has no
route for, so the policy cannot silently drift from what training does. The start-up log and
`run_state.json` (`augment_plan`) list the route of every key.

| Block | Keys | How training applies them |
|---|---|---|
| `always_on` | `mosaic`, `scale`, `translate`, `fliplr`, `hsv_*`, `mosaic_close_epochs` | Ultralytics arguments (`close_mosaic` clamped to the stage). |
| `always_on` | `downscale_upscale`, `jpeg`, `motion_blur` | Albumentations `Downscale` (INTER_AREA down, INTER_LINEAR up), `ImageCompression`, `MotionBlur`, on every training sample after mosaic. They replace Ultralytics' built-in Albumentations defaults, which are never used. |
| `infrared_only` | `clahe`, `gaussian_noise`, `brightness_contrast`, `thermal_washout` | A dataset hook applies them per SOURCE image, before mosaic, to LWIR images only (the modality is the `_lwir` token 09_build_yolo_ds.py puts in every filename). It works on one grey channel and replicates it, so B = G = R holds. |
| `forbidden` | all | Asserted against the resolved Ultralytics arguments; `erase_max_box_fraction` is reported as not applicable (the detection pipeline has no random erase). |
| `infrared_conversion` | all | The offline 3-channel conversion done by `datasets/`, reported, not a training augmentation. |

Albumentations (>= 2.0) is therefore a requirement. Ultralytics swallows an Albumentations failure
and trains on without it, so `train.py` checks the package before any GPU time and reads the
transform list back from the live dataset after the dataloader is built; either failure is exit 3,
naming the unapplied keys. Multi-GPU (DDP) is refused: Ultralytics' DDP children run a generated
script that carries neither the callbacks nor the dataset hook.

## Running it

### On the Mac, before spending any GPU time

```bash
cd training
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py --device cpu       # a few minutes; every step must print PASS
python -m pytest tests -q
```

### On Kaggle

1. Build the corpus with `datasets/notebooks/kaggle_build.ipynb`. Its `/kaggle/working/truewatch_ds/`
   output (YOLO tree, `data.yaml` with relative paths, `manifests/` including `hard_set.txt`,
   `reports/`) is attached to the training notebook as an input; the notebook finds it by searching
   `/kaggle/input/**/truewatch_ds/data.yaml`, whatever the mount path. If a split index
   (`index/split.jsonl`) is inside it, evaluation uses it for KAIST day/night and video clusters.
2. Open `notebooks/kaggle_train.ipynb` (it clones branch `fix/phase1-2-complete`). Accelerator GPU
   T4 ×2 or P100, Internet on. One GPU is used.
3. Run all. The first cell starts one session clock; each stage gets only the time left minus 45 min
   for evaluation, export and packaging, and stage 2 starts only if one estimated epoch still fits.
   `train.py` exits non-zero only on a real failure, and the notebook raises on it.
4. For the next session, add the previous session's output as an input. The notebook restores the run
   directories and the hard-set baseline, and `--auto-resume` continues from the last good checkpoint.
5. Once the final stage (stage 2) is complete, the notebook evaluates, runs the hard-set gate, exports,
   renders `METRICS.md` and publishes a private results dataset; before that it only packages the
   checkpoints. Copy only the small metrics files into `training/results/` and commit those.
6. The hard-set gate compares against `results/hardset_baseline.json`. When none exists the notebook
   records it once from the COCO-pretrained `yolo11s.pt` (`hardset_baseline_source.json` says
   `baseline: pretrained`), never from the model being judged. A failed gate blocks the upload.
7. The Hugging Face upload reads `HF_TOKEN` from Kaggle Secrets and is skipped, with a message, when it
   is absent; the upload can then be done from the Mac with `export_onnx.py --push-to-hub`.

## Checkpoint and resume

Ultralytics writes `last.pt` at the end of **every epoch**, and the optimiser, EMA, scaler, epoch and
best fitness are all inside it. Two things in Ultralytics 8.4.155 needed handling:

- `save_model` writes `last.pt` with a plain `write_bytes`, so a session killed mid-write leaves a
  truncated file. After every completed save, `train.py` copies it atomically (temp file plus
  `os.replace`) to `last_good.pt`. On resume it loads the candidates, takes the newest one that
  actually loads, and repairs `last.pt` from `last_good.pt` if needed.
- The checkpoint remembers absolute paths from the session that wrote it. On resume `train.py` passes
  `data=` and `save_dir=` explicitly, which Ultralytics' `check_resume` honours.

`--resume` requires a checkpoint and exits 2 without one; `--auto-resume` resumes if one exists.
`train_log.csv` is append-only, so it spans sessions; it carries `ap50_<class>` for every class
(blank, never 0, for a class with no validation ground truth). Ultralytics creates a fresh
early-stopping counter on every start, so on a resume `train.py` restores its best fitness and best
epoch from the log; patience therefore counts across sessions. A resume that has less time left
than one logged epoch trains nothing and exits 0 with `run_state.json` saying why. `scripts/smoke_test.py` kills a run with
SIGKILL after the second epoch, resumes it, asserts the log has no gap and no duplicate, then does it
again with `last.pt` truncated by hand.

## Evaluation rules

- **Day and IR are reported separately and never blended.** "Day" is the visible camera, "IR" is LWIR
  replicated to three channels, the same convention as gate G8 in `docs/DATASET_SPEC.md`. `_metrics.py`
  raises `BlendError` if asked for a metric over both, so no report can contain one by accident.
  Visible frames also get lighting sub-slices (daylight, night, unresolved) *inside* day: LLVIP's
  visible frames are night scenes and KAIST has night sets, so the visible slice is not all daylight.
- **Per class**, not just the mean. A class with no ground truth in a slice is `n/a`, never 0.
- **Size buckets** are the seven person heights of `docs/MEASUREMENTS.md` §3: 77, 54, 38, 27, 19, 14 and
  9 px. A test parses that document and fails if the constants drift. An object belongs to the level it
  is nearest on a log scale (edges at the geometric means: 64.5, 45.3, 32.0, 22.6, 16.3, 11.2 px). Heights
  are measured at the detector's input scale by default, the quantity §3 varied. Bucketed metrics use
  the COCO area-range protocol: ground truth outside the bucket is ignored rather than counted as a
  miss, and a large-object false positive is not charged to the 9 px bucket.
- **AP** is the 101-point interpolation used by COCO and Ultralytics. It caps a perfect detector at
  0.995, exactly as `yolo val` does; a test cross-checks the engine against Ultralytics' `ap_per_class`.
- **Uncertainty.** Person AP@50 comes with a 95% cluster bootstrap over sequences. Frames from one
  video are correlated; resampling frames would flatter the result.
- **Val, not test.** Phase 2 measures on the validation split. The test split is sealed until Phase 11
  and every script refuses to read it without `--unseal-test`. Validation also chose the checkpoint,
  so these numbers are optimistic, and `METRICS.md` says so at the top.

### Verdicts

`results/METRICS.md` compares measured figures with the slide-5 targets. It does not round in the
project's favour:

- **MET**: the point estimate reaches the target *and* so does the lower 95% bound.
- **PARTIAL**: the point estimate reaches it but the lower bound does not; or the full slice misses
  while the narrower daylight-only reading of "day" reaches it *and establishes it on its own*: at
  least 30 person boxes and its own lower 95% bound at the target. A smaller or less certain daylight
  reading leaves the verdict NOT MET.
- Figures in verdict text and the targets table are truncated to four decimals, never rounded, so
  0.84962 reads 0.8496 beside a 0.85 target.
- **NOT MET**: the point estimate is below the target.
- **NOT ASSESSED**: outside what a detector evaluation can show (detection range, plate recognition,
  missed fence crossings, the Jetson latency budget) or no measurement exists. It is never MET by default.

The false-alert target (under 5 per camera per day) is only ever PARTIAL at best here: a detector-level
false-positive rate is a proxy, and alert-level behaviour is decided by fusion in Phases 4 and 5.
`sweep_conf.py` states the arithmetic (frames per day at 8 fps, the per-frame budget, the rule-of-three
bound) and says plainly when the validation split is too small to resolve the budget at all.

## Export and hosting

```bash
python scripts/export_onnx.py --weights runs/ir/weights/best.pt --tag truewatch-yolo11s
```

Exports opset 17 with a dynamic batch axis, then runs torch and onnxruntime on the same input at batch
1 and 3 and prints both differences, labelled, on one line:

```
PARITY raw_max_abs_diff=1.785e-03 gated_max_abs_diff=8.917e-05 gated_box_units=grid tolerance=1e-03 -> PASS
```

### ONNX parity: what is printed, what is gated, and the deviation from the brief

The brief asks for "torch vs onnxruntime max-abs-diff < 1e-3, printed". Both halves are kept apart
on purpose:

- **raw** is the plain max |torch - onnxruntime| over `output0` exactly as the graph emits it: box rows
  in input pixels (0-640), score rows 0-1. It is always printed and recorded (`raw_max_abs_diff`).
- **gated** is what the 1e-3 tolerance is applied to. By default (`--box-units grid`) each anchor's four
  box rows are divided by that anchor's stride (8, 16 or 32), which is the unit the head predicts in
  before the Detect layer multiplies by the stride. Scores are compared raw.

**Deviation, stated plainly: the gate is not raw pixels, because raw < 1e-3 is not achievable against
FP32 torch.** Measured on the COCO-pretrained YOLO11-s at 640 px (ultralytics 8.4.155, torch 2.14.0,
onnxruntime 1.30.0, Apple M5 CPU, fixed-seed input):

| variant | raw max abs diff (px) | gated, grid units |
| :-- | --: | --: |
| onnxslim on, onnxruntime ORT_ENABLE_ALL (the shipped graph) | 1.785e-03 | 8.9e-05 |
| onnxslim on, ORT_DISABLE_ALL | 1.480e-03 | |
| onnxslim off, ORT_ENABLE_ALL | 1.526e-03 | |
| onnxslim off, ORT_DISABLE_ALL | 1.480e-03 | |
| one intra-op thread instead of the default pool | unchanged | |

Score rows differ by about 1.4e-07 in every variant. Against a float64 run of the same weights, torch's
own FP32 output is 1.39e-03 px off while the shipped ONNX graph is 6.0e-04 px off (6.2e-04 without onnxslim): the residual is torch's float32
rounding of box coordinates near 640 (one float32 step there is 6.1e-05) multiplied by the stride, not a
graph fault, and no FP32 export can guarantee raw < 1e-3 against FP32 torch. Divided by the stride, the
same difference is ten times under the tolerance. That gate is tighter than comparing boxes as fractions
of the image (`--box-units normalized`, pixels / 640, which is 20 to 80 times looser and was this
script's earlier default). `--box-units pixels` applies the tolerance to raw pixels as written and fails
on a correct export; when any gate fails, a float64 run of the same weights says whether the graph or
float32 noise is to blame. The script exits 1 when the gated difference is 1e-3 or more.

### Hugging Face Hub, and how the edge finds the file

Weights are hosted on the Hugging Face Hub (free, public) and are **never committed**: `.gitignore`
blocks `*.pt`, `*.pth`, `*.onnx`, `*.engine` and `*.safetensors` repository-wide, and `edge/models/`
carries its own copy of the rule.

```bash
export HF_TOKEN=...   # a write token; never a flag, never written to a file
python scripts/export_onnx.py --weights runs/ir/weights/best.pt --tag truewatch-yolo11s \
    --push-to-hub <user>/truewatch-detector
```

- The repository is created **public**. `--private` is refused: the edge downloads the model at boot
  without a token, so a private repository would stop every edge from starting. An existing private
  repository is uploaded to with a warning.
- The model card states the AGPL-3.0 terms, links the corresponding source (AGPL-3.0 section 13) given
  by `--source-url` (default `https://github.com/0XSreekar/True-Watch-AI`) and links `METRICS.md` there.
  It carries no accuracy figure.
- The upload writes `results/hf_model.json` and `results/export_<tag>.json`. `hf_model.json` holds the
  repository, the **commit sha** of the upload (`revision`), a `resolve_url` pinned to that commit
  (`https://huggingface.co/<repo>/resolve/<sha>/<file>`), the file's SHA-256 and size, the opset, input
  size and class names. Commit it: it is the only model artefact in git, and a later push to the same
  repository cannot change the bytes it points at.
- `edge/models/detector_weights.py` reads that file at boot (or `YOLO_MODEL_URL` + `YOLO_MODEL_SHA256`,
  which a container built from `edge/` alone must set), downloads over plain HTTPS into
  `edge/models/cache/`, verifies the SHA-256 before an atomic rename, and opens the graph with
  onnxruntime. Nothing in `edge/` imports Ultralytics.

## Performance figures

> The ~30 ms detector figure is the YOLO11-s INT8 inference budget at 640 px on a Jetson Orin Nano
> Super 8 GB, as stated on slide 3 of the submission deck. It is a **design target on hardware this
> project does not own**, and it is **not** end-to-end latency. The ~3 s explanation figure is
> likewise a target, and the explanation is asynchronous, so the operator sees the provisional alert
> before it arrives. Every number this repository presents as measured was produced on the host named
> beside it, on public datasets, and is recorded in `docs/MEASUREMENTS.md` or `training/results/`. No
> measurement in this repository was taken on a Jetson.

`benchmark_cpu.py` reports FP32 latency on a CPU, at batch 1 and 640 px, as p50 and p95 over many
runs, for onnxruntime alone (`inference_ms`) and for a detector-stage pipeline (`pipeline_ms`:
letterbox, run, decode, NMS). Those numbers are **not comparable to the Jetson INT8 target** and must
never be quoted beside it as if they were.

```bash
python scripts/benchmark_cpu.py --onnx weights/truewatch-yolo11s.onnx --weights-kind trained \
    --image <a real 720p frame> --label mac-apple-m5-cpu
```

The committed Mac file (`results/benchmark_mac-apple-m5-cpu.json`) was taken with a **randomly
initialised** YOLO11-s of the right shape. Its `inference_ms` stands, because inference time depends
on the graph and the host, not the weights. Its `pipeline_ms` does not: a random head scores every
anchor alike, so NMS receives all 8400 candidates and keeps 300 detections per frame, which a trained
detector never produces. Every record now carries `weights_kind` (trained / random / unknown, with
"random" inferred from the provenance line of older files) and `pipeline_representative` (true only for
trained weights on a real `--image`), and `make_metrics.py` suppresses the pipeline figure of a
random-weight record and says why. Re-run the command above on the trained export for a pipeline row.
There is no Hugging Face Spaces measurement yet: run the script in a Space (it needs only numpy and
onnxruntime), copy the JSON block it prints between `--- BEGIN BENCHMARK JSON ---` and
`--- END BENCHMARK JSON ---` from the Space's log into `results/benchmark_<label>.json`, and commit it.

## Before the first real run: check the dataset for filename collisions

`datasets/scripts/09_build_yolo_ds.py` names every final file `{source}_{raw stem}_{modality}`. If two
source frames share a raw stem (KAIST frames are numbered per video, IDD frames per drive), they map
to the same final filename. The copy step keeps the first image but the label file is overwritten by
the second, which silently pairs an image with the wrong boxes; the parity gate cannot see it because
the names still match. This has not been verified against real data, which is not on the machine this
phase was built on. `train.py`'s preflight looks for duplicate image rows in the dataset's
`manifest.tsv` and fails loudly if it finds any. Run `python scripts/train.py --stage day --dry-run`
on the real corpus before the first long run.

## Known limits

- Everything in `results/` about detection quality is empty until a real run. The tooling was verified
  on coloured rectangles, which proves it executes and that its numbers agree with Ultralytics' own
  validator, not that the detector is any good.
- Person AP@50 on public datasets is not a border-camera result. No source is pole-mounted at 4-6 m
  looking down a 150 m approach (`docs/DATASET_SPEC.md` §7, known biases).
- Lighting for IDD (daylight) and LLVIP visible (night) is taken from `docs/DATASET_SPEC.md` §3.4, not
  measured per frame. Frames whose lighting cannot be resolved are labelled unresolved and are never
  counted as daylight.
- If the cart class ends up with fewer than 300 hand-verified instances it is withdrawn from the
  shipped schema (`docs/DATASET_SPEC.md` §1.5); id 4 stays reserved and its AP is `n/a`. The head
  still has a class-4 output that was trained with no positives, so its scores should stay low, but
  consumers must drop class 4 while it is withdrawn (`run_state.json` `withdrawn_classes` says so).
- Only one of the two T4s in a "T4 ×2" session is used. DDP would roughly halve epoch time but runs
  outside `train.py`'s callbacks and dataset hook, so it is refused rather than half-supported.
