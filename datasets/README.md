# Datasets

**Scripts only. No dataset file is ever committed to this repository.**

Each script fetches one public dataset into a local directory that `.gitignore`
excludes, prints what it fetched, and exits. Nothing here trains anything —
training lives in `training/` and runs on Kaggle.

The three datasets below are the ones `docs/MEASUREMENTS.md` records as verified
available, and the three slide 3 names for fine-tuning: *"appearance channel,
fine-tuning planned on IDD, LLVIP, KAIST"*.

| Script | Dataset | Source | Used for |
|---|---|---|---|
| `fetch_kaist.py` | KAIST Multispectral Pedestrian | Hugging Face `richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset`, 23,210 files | Capability 1 and 7. Every measured number in `docs/MEASUREMENTS.md` sections 1, 2 and 3 came from this set. |
| `fetch_llvip.py` | LLVIP visible/infrared pairs | Hugging Face `jsonhash/LLVIP` | Capability 7, IR fine-tuning |
| `fetch_idd.py` | IDD Detection, 46,588 images | `idd.insaan.iiit.ac.in` | Capability 2, vehicle classes |

## Usage

```bash
cd edge && source .venv/bin/activate && cd ..
python datasets/fetch_kaist.py --out ./var/datasets/kaist --limit 200
python datasets/fetch_llvip.py --out ./var/datasets/llvip
python datasets/fetch_idd.py   --out ./var/datasets/idd
```

`--limit` exists because the full KAIST set is large and every measured result
in `docs/MEASUREMENTS.md` used a few hundred frames. Fetch the subset first.

## Licences and registration

- **KAIST** and **LLVIP** are downloaded from Hugging Face. Set `HF_TOKEN` in
  `edge/.env` if you hit a rate limit; neither dataset is gated at the time of
  writing, but check the dataset card before redistributing anything.
- **IDD requires manual registration.** IIIT Hyderabad gates the download behind
  an account and a licence agreement. `fetch_idd.py` therefore does not scrape —
  it prints the registration URL, verifies a manually downloaded archive, and
  extracts it. This is a deliberate refusal to bypass a licence gate.

Each dataset carries its own licence and citation requirements. Using a dataset
here does not relicense it, and this repository's AGPL-3.0 licence does not
extend to any dataset fetched by these scripts.
