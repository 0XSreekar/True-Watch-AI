# Datasets

**Scripts and manifests only. No dataset image, video or archive is ever committed.**

This directory builds one unified detection corpus for the single model in the stack that
is fine-tuned: YOLO11-s. The design and the reasoning behind every number below are in
[`docs/DATASET_SPEC.md`](../docs/DATASET_SPEC.md); the licences are in
[`docs/DATASET_CARD.md`](../docs/DATASET_CARD.md). This file is the operating manual.

Nothing here imports Ultralytics. That package is AGPL-3.0 and lives only in `training/`;
this directory emits Ultralytics *format*, which is just directories and text files.

## Layout

```
config/     schema, sources, splits, augmentation - the parameters, in one place
scripts/    00..11, run in order
plates/     synthetic Nepali plate corpus for the ANPR fine-tune
manifests/  committed: file lists, hashes, split membership. Never the data.
reports/    generated: stats, histograms, the manual spot check. Git-ignored.
```

## Setup

```bash
python3.11 -m venv .venv-datasets
source .venv-datasets/bin/activate
pip install -r datasets/requirements.txt
```

CPU only. Nothing in this directory needs CUDA, and every script runs unchanged on an
Apple Silicon Mac and on a Kaggle notebook.

## Run the whole pipeline

Every script takes `--dry-run`, `--seed` (default 42), `--limit`, `--force`, `--raw` and
`--processed`. Every script is idempotent: re-running one costs a state-file read, not
the work again.

```bash
# 0. Fetch. KAIST and LLVIP come from Hugging Face. IDD is licence-gated: this prints the
#    registration URL and waits for you to place the archive yourself.
python datasets/scripts/00_fetch.py --dry-run
python datasets/scripts/00_fetch.py

# 1-3. Convert each source to the unified schema.
python datasets/scripts/01_convert_idd.py
python datasets/scripts/02_convert_kaist.py
python datasets/scripts/03_convert_llvip.py

# --- manual step, about 45 minutes, the only labelling in this phase ---
# 01_convert_idd.py writes processed/cart_review/vehicle_fallback.csv. Mark each row
# cart or not_cart. Class 4 ships only at >= 300 verified instances (spec section 1.5).

# 4-9. Build.
python datasets/scripts/04_ir_to_3ch.py        # LWIR -> 3 channels, B = G = R
python datasets/scripts/05_dedupe.py           # perceptual hash, within and across sources
python datasets/scripts/06_split.py --seed 42  # sequence-level splits, asserts separation
python datasets/scripts/07_negatives.py        # background pool at 10% of train
python datasets/scripts/08_tile_farfield.py    # 640x640 tiles, 0.2 overlap, upper 40%
python datasets/scripts/09_build_yolo_ds.py    # final tree + data.yaml

# 10-11. The gate, then the numbers.
python datasets/scripts/11_stats.py            # writes reports/ and the 100-image spot check
#    -> open datasets/reports/spotcheck/, look at the images, write VERDICT.txt
python datasets/scripts/10_validate.py ; echo "exit=$?"
```

`10_validate.py` exits non-zero if any of its nineteen gates fails. Phase 2 does not start
until it exits 0.

## Plates

```bash
python datasets/plates/gen_plates.py --check-font        # resolve Noto Sans Devanagari
python datasets/plates/gen_plates.py --dry-run
python datasets/plates/gen_plates.py --count 20000 --seed 42 --clean
```

The font binary is not committed; `plates/fonts/README.md` has the fetch command. Labels
land in `plates/labels/` in PaddleOCR recognition format and are git-ignored along with
the images.

## Runtime and disk

Measured on an Apple Silicon Mac, CPU only. Fetch times are network-bound and will differ.

| step | runtime | disk after |
|---|---|---|
| `00_fetch` KAIST | 25-60 min | 15-20 GB |
| `00_fetch` LLVIP | 10-20 min | +4-6 GB |
| `00_fetch` IDD | manual download, 20-40 min | +18-22 GB |
| `01_convert_idd` | 8-15 min | +200 MB (labels) |
| `02_convert_kaist` | 20-40 min | +1-3 GB (masked frames) |
| `03_convert_llvip` | 5-10 min | +100 MB |
| `04_ir_to_3ch` | 25-45 min | +8-12 GB |
| `05_dedupe` | 20-40 min | +50 MB |
| `06_split` | under 2 min | +20 MB |
| `07_negatives` | 3-8 min (30+ with `--verify-hotspots`) | +5 MB |
| `08_tile_farfield` | 15-30 min | +6-10 GB |
| `09_build_yolo_ds` | 30-60 min | +12-18 GB |
| `10_validate` | 5-15 min | +10 MB |
| `11_stats` | 5-10 min | +50 MB |
| `gen_plates --count 20000` | 12-20 min | +1.5 GB |

Peak disk, if nothing is deleted between steps: about 75 GB. The only directory training
needs is `processed/yolo`, at 12-18 GB.

Memory: every script streams. Nothing loads a corpus into RAM, and the pipeline is sized
for a 16 GB machine.

## Kaggle

Kaggle's free tier gives 30 GPU-hours a week, and a private dataset may be up to 20 GB.
The source datasets cannot be re-hosted (see the dataset card), so what moves is the
**built** corpus, privately, as a personal working copy.

```bash
# locally, after 10_validate.py exits 0
pip install kaggle
mkdir -p upload && cp -r datasets/processed/yolo/* upload/
kaggle datasets init -p upload
# edit upload/dataset-metadata.json: set an id and title, keep it PRIVATE
kaggle datasets create -p upload --dir-mode zip

# in the notebook: attach the dataset, then
# data.yaml is at /kaggle/input/<slug>/data.yaml - no copy, no download, no GPU time on I/O
```

The plate corpus goes to its own small dataset; it is the only part that is ours to publish.

Two differences on Kaggle, both handled by flags rather than edits:

```bash
python datasets/scripts/10_validate.py \
  --processed /kaggle/working/processed \
  --dataset /kaggle/input/<slug> \
  --plates /kaggle/input/<plate-slug>
```

Gate G18 (nothing staged for commit) is meaningless outside a git checkout: with no
repository to inspect it sees nothing staged and passes trivially, and if `git` is absent
altogether it reports `-1` and fails. Either way it only carries information locally, so
run the gate on the Mac before uploading rather than in the notebook afterwards.
