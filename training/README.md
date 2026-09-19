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

**RESOLVED in Phase 1: the whole repository is AGPL-3.0.** `LICENSE` at the
repository root carries the verbatim text from gnu.org. This was option 1 of the
three in `docs/PHASE_MINUS1_SCOPE.md` risk 1, chosen because the deployment is
network-served from a public repository, which makes AGPL compliance a matter of
keeping the repository public rather than an added obligation. Isolating
Ultralytics in a second repository would have cost a second artefact to manage
mid-build; swapping the detector would have invalidated every YOLO11-s number in
`docs/MEASUREMENTS.md`.

Verify the current Ultralytics licence text at source rather than relying on this
note; if it has changed, the decision is reopened.

## Weights

Fine-tuned weights are not committed — they exceed normal Git limits and
`edge/models/.gitignore` excludes them. The intended home is a Hugging Face
model repository referenced by id (`docs/ARCHITECTURE_V2.md` §11, open
question 4).
