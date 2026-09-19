# TRUEWATCH — training

Fine-tuning and model export. These scripts run on free Kaggle GPU hours, not
in CI and not on the development machine. Nothing in this folder is imported by
`edge/` at runtime: `edge/` consumes exported ONNX weights, resolved by id.

> **Phase 0 scaffolding.** This folder currently contains this file only.
> The fine-tuning and export scripts arrive in Phase 2.

## Planned contents

| File | Phase | Purpose |
|---|---|---|
| `finetune_yolo11s.py` | 2 | Fine-tune YOLO11-s on IDD Detection, LLVIP and KAIST |
| `export_onnx.py` | 2 | Export the fine-tuned weights to ONNX for `edge/` |
| `notebooks/kaggle_finetune.ipynb` | 2 | The notebook actually run on Kaggle |

Slide 3 of the submission deck names the datasets: *"appearance channel,
fine-tuning planned on IDD, LLVIP, KAIST"*. `docs/PHASE_MINUS1_SCOPE.md` §5.2
records the post-fine-tuning figures on slide 5 as **targets**, not results.
Nothing produced here may be reported as measured without naming the host it
was measured on.

## Licence note — Ultralytics YOLO11 is AGPL-3.0

`finetune_yolo11s.py` and `export_onnx.py` import
[Ultralytics](https://github.com/ultralytics/ultralytics), which is licensed
under **AGPL-3.0**. The AGPL's network clause extends source-availability
obligations to software made available over a network, not only to software
distributed as a binary. This repository is public.

**This is unresolved and blocks Phase 2.** `docs/PHASE_MINUS1_SCOPE.md` risk 1
and open question 4 set out the three options:

1. License the whole repository AGPL-3.0 and add the notice.
2. Keep every Ultralytics import isolated in a separately licensed repository
   or submodule, leaving the console repository untouched.
3. Swap the detector for a permissively licensed one — which invalidates every
   YOLO11-s number in `MEASUREMENTS.md`, and is therefore the last resort.

No file in this folder may import Ultralytics until that decision is recorded
and a `LICENSE` file exists at the repository root. Verify the current licence
text at source rather than relying on this note.

## Weights

Fine-tuned weights are not committed — they exceed normal Git limits and
`edge/models/.gitignore` excludes them. The intended home is a Hugging Face
model repository referenced by id (`docs/ARCHITECTURE_V2.md` §11, open
question 4).
