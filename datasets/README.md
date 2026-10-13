# Datasets

**Scripts and manifests only. No dataset image, video or archive is ever committed.**

This directory builds one unified detection corpus for the single model in the stack that
is fine-tuned: YOLO11-s. The design and the reasoning behind every number below are in
[`docs/DATASET_SPEC.md`](../docs/DATASET_SPEC.md); the licences are in
[`docs/DATASET_CARD.md`](../docs/DATASET_CARD.md). This file is the operating manual.

Nothing here imports Ultralytics. That package is AGPL-3.0 and lives only in `training/`;
this directory emits Ultralytics *format*, which is just directories and text files.

## Sources

| source | modality | Kaggle mirror (third party) | role |
|---|---|---|---|
| IDD Detection | visible | `vinayak21574/idd-detection` | vehicles, people, Indian roads |
| Teledyne FLIR ADAS v2 | visible + thermal | `samdazel/teledyne-flir-adas-thermal-dataset-v2` | day/night visible and LWIR, vehicles and people |
| LLVIP | visible + infrared pairs | `afradhossain/llvip-dataset` | night pedestrians, registered pairs |
| KAIST | visible + LWIR | disabled | replaced by FLIR: no reachable copy of its annotations |

The mirrors are re-uploads of the original releases; the originals' licences govern
(dataset card). KAIST's converter (`02_convert_kaist.py`) is kept and is collision-safe, but
`sources.yaml` has it `enabled: false`.

## Layout

```
config/     schema, sources, splits + composition, augmentation - the parameters, in one place
scripts/    00..11, run in order, plus _lib.py
notebooks/  kaggle_build.ipynb - the whole build on a Kaggle CPU notebook
plates/     synthetic Nepali plate corpus for the ANPR fine-tune
manifests/  committed: file lists, hashes, split membership. Never the data.
reports/    generated: stats, histograms, the manual spot check. Git-ignored.
```

`scripts/_lib.py` holds what every script shares: the structured log format, the argparse
shape, the counter discipline that makes a skipped file a named, printed number rather than a
silent loss, source-root discovery, the naming rule, the infrared conversion, the tile plan,
the cart gate and the process-pool helpers.

## How the pipeline treats the raw sources

* **Read-only.** A source root is used where it lies: a download under `--raw`, or a mount
  such as `/kaggle/input/...`. Nothing is copied, nothing is extracted, nothing is written
  under a source root. Roots are found by a marker (`sources.yaml` `marker`).
* **Collision-proof names.** Every final file is `{source}_{stem of record image}_{modality}`,
  the rule `training/scripts` re-derive. IDD and KAIST reuse frame names across drives and
  videos (IDD: 2,694 stems occur in more than one drive), so their records point at a
  **symlink** under `processed/stage/` named `<subset>__<drive>__<frame>` (KAIST:
  `set_video_frame`). 09 asserts every final name unique across all sources before it writes.
* **Infrared** is converted (CLAHE, then B = G = R) at the moment it is written into the
  dataset. No 3-channel intermediate is stored.

## Run locally

```bash
python3.11 -m venv .venv-datasets && source .venv-datasets/bin/activate
pip install -r datasets/requirements.txt       # plus `pip install kaggle` for --mode download

# 0. Resolve the sources. Either download the mirrors (needs ~/.kaggle credentials, ~35 GB) ...
python datasets/scripts/00_fetch.py --dry-run
python datasets/scripts/00_fetch.py
# ... or point at copies that already exist, used in place:
python datasets/scripts/00_fetch.py --root idd=/data/idd --root flir=/data/flir --root llvip=/data/llvip

# 1-3. Convert each source to the unified schema.
python datasets/scripts/01_convert_idd.py
python datasets/scripts/02_convert_flir.py
python datasets/scripts/03_convert_llvip.py

# --- optional manual step, about 45 minutes ---
# 01 writes processed/cart_review/vehicle_fallback.csv. Mark `verdict` as cart or not_cart.
# Class 4 ships only at >= 300 verified carts (spec 1.5); otherwise 09 withdraws it and says so.

# 4-9. Build.
python datasets/scripts/04_ir_to_3ch.py        # LWIR decodability, B = G = R check, ir_std
python datasets/scripts/05_dedupe.py           # perceptual hash, within and across sources
python datasets/scripts/06_split.py --seed 42  # sequence-level splits, balance, hard set
python datasets/scripts/07_negatives.py        # background pool at 10% of train
python datasets/scripts/08_tile_farfield.py    # 640x640 tiles of the upper 40%, native res
python datasets/scripts/09_build_yolo_ds.py    # final tree + data.yaml -> processed/yolo

# 10-11. The numbers, then the gate.
python datasets/scripts/11_stats.py            # stats, charts, 100-image spot check
#    -> open reports/spotcheck/, look at the images, write PASS/FAIL to VERDICT.txt
python datasets/scripts/10_validate.py --skip-plates ; echo "exit=$?"
```

Every script takes `--dry-run`, `--seed` (default 42), `--limit`, `--force`, `--raw`,
`--processed` and `--workers` (default: `$TRUEWATCH_WORKERS`, else min(4, cores)). Every
script is idempotent: a re-run reads its state file instead of repeating work, and 09 removes
files a changed plan no longer contains. Converters take `--subset-ok` for a deliberately
partial source root (a fixture): listed files that are absent are counted instead of fatal.

Three environment variables relocate the outputs without code changes:
`TRUEWATCH_PROCESSED`, `TRUEWATCH_MANIFEST_DIR`, `TRUEWATCH_REPORT_DIR`. The Kaggle notebook
points the last two into its output directory so manifests and reports travel with the build.

`10_validate.py` exits non-zero if any gate fails. `--skip-plates` marks G17 (plate corpus)
out of scope for a detection-only build: it prints SKIP and is never counted as a PASS.
Phase 2 does not start until the gate exits 0, which requires a human verdict for G19.

## Run on Kaggle (the intended path)

1. New notebook, **accelerator none** (CPU, 4 cores), **internet on**.
2. Add the three inputs: `vinayak21574/idd-detection`,
   `samdazel/teledyne-flir-adas-thermal-dataset-v2`, `afradhossain/llvip-dataset`.
3. Import `datasets/notebooks/kaggle_build.ipynb`, check `BRANCH` in the first cell, run all.
4. The build lands in `/kaggle/working/truewatch_ds/`:
   `data.yaml` (relative `path: .`), `images/{train,val,test}`, `labels/...`, `manifest.tsv`,
   `index/split.jsonl`, `manifests/` (hard set, splits, negatives, cart gate, review sheet),
   `reports/` (stats, charts, validation table, per-step logs) and `reports/spotcheck/`.
5. Save a version; attach its output to the training notebook as an input. `training/`
   finds `data.yaml` under `/kaggle/input` itself.

The notebook auto-discovers each source by globbing for a marker file, because Kaggle mounts
inputs at either `/kaggle/input/<slug>/` or `/kaggle/input/datasets/<owner>/<slug>/`.

## Expected full-scale numbers

From the real source sizes (IDD 31,569 + 10,225 labelled, FLIR 10,318 / 1,085 / 3,749 RGB and
10,742 / 1,144 / 3,749 thermal, LLVIP 12,025 + 3,463 pairs) and the parameters in
`config/splits.yaml`; the Kaggle run prints the measured values.

| quantity | expected |
|---|---|
| untiled images, train / val / test | ~60k / ~12k / ~13.5k (0.70 / 0.14 / 0.16) |
| far-field object tiles | 6,000 (35% LWIR) |
| negatives in train | ~6.6k, ~10% of train |
| visible share, train / val / test | ~0.66 / 0.65 / 0.65 |
| IDD frames dropped by modality balancing | ~20k of 41.8k (listed in `manifests/balance_dropped.txt`) |
| output size | ~13-16 GB (JPEG q92, long side 1280 for untiled frames) |
| wall time, 4-core Kaggle CPU | ~1-1.5 h (05 and 09 dominate) |

## Plates

```bash
python datasets/plates/gen_plates.py --check-font        # resolve Noto Sans Devanagari
python datasets/plates/gen_plates.py --dry-run
python datasets/plates/gen_plates.py --count 20000 --seed 42 --clean
```

The font binary is not committed; `plates/fonts/README.md` has the fetch command. Labels
land in `plates/labels/` in PaddleOCR recognition format and are git-ignored along with
the images. The plate corpus is built separately from the detection corpus; validate it with
`10_validate.py` without `--skip-plates`.

## Runtime and disk, per step

Kaggle CPU session, full scale. Local Apple Silicon is roughly 2x faster per core.

| step | runtime | disk written |
|---|---|---|
| `00_fetch` (`--root` / kaggle_input) | seconds | none |
| `01_convert_idd` | 2-4 min | ~170 MB labels + symlinks |
| `02_convert_flir` | 1-2 min | ~120 MB labels |
| `03_convert_llvip` | under 1 min | ~120 MB labels |
| `04_ir_to_3ch` | 3-5 min | index only |
| `05_dedupe` | 10-25 min | index + hash cache |
| `06_split` | under 1 min | ~40 MB |
| `07_negatives` | under 1 min | ~40 MB |
| `08_tile_farfield` | 3-6 min | ~0.8 GB tiles |
| `09_build_yolo_ds` | 20-40 min | ~13-16 GB, the only large output |
| `11_stats`, `10_validate` | 3-6 min | ~20 MB |

## Kaggle upload of a locally built corpus

If the build is made locally instead, the finished `processed/yolo` is self-contained and can
be uploaded as a PRIVATE Kaggle dataset (a personal working copy, never a publication):

```bash
kaggle datasets init -p datasets/processed/yolo   # set id + title, keep it private
kaggle datasets create -p datasets/processed/yolo --dir-mode zip
```

Gate G18 (nothing staged for commit) only carries information inside a git checkout.
