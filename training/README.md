# TRUEWATCH — training

Fine-tuning, evaluation and export for the one model in the stack that is fine-tuned: the YOLO11-s
appearance channel (slide 3: *"fine-tuning planned on IDD, LLVIP, KAIST"*). These scripts run on free
Kaggle GPU hours, not in CI and not in `edge/`. `edge/` consumes an exported ONNX graph through
onnxruntime and nothing else.

> **Status (Phase 2).** The tooling is complete: 253 GPU-free tests pass, and `scripts/smoke_test.py`
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
5. **The budget fits one lineage.** Kaggle gives 30 GPU-hours a week for the whole project, and the project plan sets aside about 8 for this phase. One lineage fits; two do not.

| Decision | Choice | Reason |
|---|---|---|
| Stage 1 | 30 epochs, full corpus, COCO `yolo11s.pt` | The bulk of adaptation. |
| Stage 2 | 8 epochs from stage 1 best weights, LWIR repeated ×2, LR 5× lower | Re-weights toward IR; same corpus, same head. |
| Image size | 640 | The ~30 ms figure is defined at 640 px, and the far field is handled by 640×640 SAHI tiles, so a bigger input is not needed. |
| Batch | 32 (nominal batch 64, so two accumulated steps) | Fits a 16 GB T4 under AMP. |
| Optimiser | SGD, momentum 0.937, weight decay 5e-4 | What Ultralytics would choose above 10,000 iterations; explicit so a resume cannot change it. |
| LR | Stage 1 0.005, cosine to ×0.01; stage 2 0.001 | Half the from-scratch COCO rate: fine-tuning should not overwrite pretrained features. |
| Warmup | 3 epochs (stage 2: 1) | Momentum and bias warmup for the new 5-class head. |
| Freeze | Layers 0-9 frozen for epochs 0-2, then unfrozen (stage 1) | A freshly initialised head produces large early gradients that would wreck COCO features. |
| EMA | On (Ultralytics always keeps one); `best.pt` and `last.pt` hold EMA weights | Not switchable; stated so nobody looks for the switch. |
| Early stopping | Patience 8 (stage 2: 5) on Ultralytics fitness | Class-mean fitness keeps vehicles from dying while people improve; per-epoch person AP50 is logged for a human to watch. |
| Augmentation | Read from `datasets/config/augment.yaml`, never duplicated; the `forbidden:` block is asserted at start | Single source of truth between `datasets/` and `training/`. |
| Class imbalance | Images with `truck` or `cart` appear twice per epoch, no image more than twice, no loss reweighting | `docs/DATASET_SPEC.md` §3.1: oversampling a rare class in a detector mostly multiplies its backgrounds. |
| Sessions | `--max-hours` (default 9.0) stops after the epoch that crosses it | Kaggle kills at 12 h and evaluation and export need the rest. |

None of these values has been tuned: they were decided in advance from the reasons above, and the
first Kaggle epochs are the first evidence for or against them. The 8 GPU-hour total is an
**estimate**. It depends on dataloader throughput on Kaggle's four vCPUs, which has not been
measured. `train_log.csv` records seconds per epoch; replace the estimate with the measurement after
the first epoch.

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

1. Build the corpus on the Mac (`datasets/README.md`), then copy `datasets/processed/index/split.jsonl`
   to `<upload dir>/index/split.jsonl` before `kaggle datasets create`. It lets evaluation tell KAIST
   day frames from KAIST night frames and group frames by video for the bootstrap.
2. Open `notebooks/kaggle_train.ipynb`. Accelerator GPU T4 ×2 or P100, Internet on. Add the corpus dataset.
3. Run all. Each session trains until `--max-hours`, writes `run_state.json`, and stops.
4. For the next session, add the previous session's output as an input. The notebook restores the run
   directory and `--auto-resume` continues.
5. The notebook ends by evaluating, exporting, rendering `METRICS.md` and publishing a private results
   dataset. Copy only the small metrics files into `training/results/` and commit those.

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
`train_log.csv` is append-only, so it spans sessions. `scripts/smoke_test.py` kills a run with
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
  while the narrower daylight-only reading of "day" reaches it.
- **NOT MET**: the point estimate is below the target.
- **NOT ASSESSED**: outside what a detector evaluation can show (detection range, plate recognition,
  missed fence crossings, the Jetson latency budget) or no measurement exists. It is never MET by default.

The false-alert target (under 5 per camera per day) is only ever PARTIAL at best here: a detector-level
false-positive rate is a proxy, and alert-level behaviour is decided by fusion in Phases 4 and 5.
`sweep_conf.py` states the arithmetic (frames per day at 8 fps, the per-frame budget, the rule-of-three
bound) and says plainly when the validation split is too small to resolve the budget at all.

## Export and hosting

```bash
python scripts/export_onnx.py --weights runs/ir/weights/best.pt
```

Exports opset 17 with a dynamic batch axis, then runs torch and onnxruntime on the same input at batch
1 and 3 and prints `PARITY max_abs_diff=... tolerance=1e-03 -> PASS`. It exits non-zero at 1e-3 or above.

Box rows are compared as **fractions of the input size** by default (`--box-units normalized`). The raw
graph emits boxes in input pixels (0-640), where one float32 step is 6e-5 and accumulated rounding
reaches 1.5e-3 even on a tiny model, while the graph is provably as close to a float64 run of the same
weights as torch itself is. A pixel-unit tolerance of 1e-3 would therefore fail correct exports. The
pixel-unit difference is always printed beside the normalized one, and `--box-units pixels` applies the
tolerance to raw pixels. Scores (0-1) are compared unscaled either way. Measured on the smoke model:
9.4e-06 normalized, 1.5e-03 in pixels.

Weights are hosted on the Hugging Face Hub (free, public) and are **never committed**: `.gitignore`
blocks `*.pt`, `*.pth`, `*.onnx`, `*.engine` and `*.safetensors` repository-wide, and `edge/models/`
carries its own copy of the rule. Upload with `--push-to-hub <user>/<repo>` and `HF_TOKEN` set in the
environment; nothing is uploaded otherwise. After a real upload, record the model URL in
`results/hf_model.json` and commit that file, not the weights. `edge/` resolves the model by URL at
boot through `YOLO_MODEL_ID`; that loader is not part of this phase.

## Performance figures

> The ~30 ms detector figure is the YOLO11-s INT8 inference budget at 640 px on a Jetson Orin Nano
> Super 8 GB, as stated on slide 3 of the submission deck. It is a **design target on hardware this
> project does not own**, and it is **not** end-to-end latency. The ~3 s explanation figure is
> likewise a target, and the explanation is asynchronous, so the operator sees the provisional alert
> before it arrives. Every number this repository presents as measured was produced on the host named
> beside it, on public datasets, and is recorded in `docs/MEASUREMENTS.md` or `training/results/`. No
> measurement in this repository was taken on a Jetson.

`benchmark_cpu.py` reports FP32 latency on a CPU, at batch 1 and 640 px, as p50 and p95 over many
runs, for onnxruntime alone and for a detector-stage pipeline (letterbox, run, decode, NMS). Those
numbers are **not comparable to the Jetson INT8 target** and must never be quoted beside it as if
they were. The Mac figures in `results/` were taken with a randomly initialised YOLO11-s of the right
shape, because latency depends on the graph and not the weights; each file says so. There is no
Hugging Face Spaces measurement yet: run the script in a Space (it needs only numpy and onnxruntime),
copy the JSON block it prints between `--- BEGIN BENCHMARK JSON ---` and `--- END BENCHMARK JSON ---`
from the Space's log into `results/benchmark_<label>.json`, and commit it.

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
  shipped schema (`docs/DATASET_SPEC.md` §1.5); id 4 stays reserved and its AP is `n/a`.
