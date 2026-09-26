#!/usr/bin/env python3
"""Publish the fine-tuned Devanagari plate recogniser to the Hugging Face Hub.

Uploads an EXPORTED inference model directory (the `tools/export_model.py` output —
`inference.json`/`inference.pdiparams`/`inference.yml`, the same layout PaddleOCR's
`TextRecognition(model_dir=...)` reads, so `edge/anpr/recognise.py`'s
`ANPR_DEVANAGARI_MODEL_DIR` can point straight at a local download of this repo) to a
PUBLIC model repo on the Hub, writes a model card, and pins a manifest at
`edge/anpr/results/hf_ocr_model.json` recording the exact commit SHA, file list and
per-file sha256 so a later download can be verified byte-for-byte.

This script does NOT decide whether to upload — the orchestrator only runs it after
confirming the fine-tuned model's exact-match rate on `edge/anpr/results/*.json` beats
the pretrained baseline recorded in `edge/anpr/results/ANPR_METRICS.md`. Uploading a
model that regressed against the baseline would misrepresent the fine-tune's result.

Usage:
    python edge/anpr/scripts/push_ocr_to_hub.py \\
        --model-dir /path/to/exported/inference \\
        --repo-id sreekar12/truewatch-anpr-devanagari \\
        --metrics-json edge/anpr/results/eval_synthetic_finetuned.json \\
        --baseline-json edge/anpr/results/eval_synthetic_pretrained.json \\
        --manifest-out edge/anpr/results/hf_ocr_model.json

Requires `huggingface_hub` and a token at `~/.cache/huggingface/token` (or `HF_TOKEN`)
with write access to the target namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SOURCE_REPO_URL = "https://github.com/0XSreekar/True-Watch-AI"
BASE_MODEL_NAME = "devanagari_PP-OCRv5_mobile_rec"
BASE_MODEL_LICENSE = "Apache-2.0"


def log(level: str, message: str, **fields) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"{stamp} {level:<5} push_ocr_to_hub  | {message}"
    if extra:
        line = f"{line} | {extra}"
    print(line, file=sys.stderr if level in ("ERROR", "FATAL") else sys.stdout, flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(f"metrics file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", data)


def build_model_card(
    *,
    repo_id: str,
    finetuned: dict,
    baseline: dict,
    latin_baseline: dict | None,
    sample_count: int,
    real: dict | None = None,
    real_baseline: dict | None = None,
) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x * 100:.2f}%"

    def num(x, digits=4):
        return "n/a" if x is None else f"{x:.{digits}f}"

    lines = [
        "---",
        "license: apache-2.0",
        "base_model: PaddlePaddle/devanagari_PP-OCRv5_mobile_rec",
        "language:",
        "- ne",
        "- hi",
        "tags:",
        "- ocr",
        "- paddleocr",
        "- license-plate-recognition",
        "- devanagari",
        "pipeline_tag: image-to-text",
        "---",
        "",
        "# TrueWatch ANPR — fine-tuned Devanagari plate recogniser",
        "",
        "A text-recognition model fine-tuned from the pretrained "
        f"`{BASE_MODEL_NAME}` ({BASE_MODEL_LICENSE}) recognition head, on a "
        "synthetically rendered corpus of Nepali (Devanagari-script) vehicle "
        "plates. It reads ONE rectified text line and returns its string: "
        "Nepali plates are two-line, so the pipeline (`edge/anpr/rectify.py`) "
        "splits each plate into its lines first and joins the results. It "
        "does not detect or locate plates in a full scene.",
        "",
        "## Training data",
        "",
        "- 20,500 synthetically rendered plate images "
        "(`datasets/plates/gen_plates.py`, seed 42) with the generator's own "
        "train/validation split; training used per-line crops cut by the "
        "same line-splitting code the pipeline runs at inference. Later rounds "
        "add plates rendered in 44 open-licence Devanagari fonts (fonts that draw "
        "Devanagari digits with Latin shapes are rejected automatically) and in "
        "the layouts photographed plates actually use: red two-line plates with "
        "zone, lot and class on the top line, one-line plates, and two- and "
        "three-line province plates. Plates that did not split were skipped, "
        "never mislabelled.",
        "- Real photographs: line crops of hand-transcribed plates (each read "
        "independently where possible) and of repeat views of those plates matched "
        "to their verified text, from the Kaggle dataset "
        "`ishworsubedii/vehicle-number-plate-datasetnepal` (Apache-2.0) and a "
        "training subset of the Hugging Face dataset "
        "`mukulboro/nepali-private-license-plates` (CC BY 4.0, Kathmandu "
        "University parking-lot photographs). Real test plates were never "
        "trained on; training plates whose text matched a test plate were removed.",
        "- Synthetic plates follow a documented specification. "
        "The corpus's composition (zone codes, vehicle-class letters, colour "
        "series) follows a documented specification with several fields "
        "explicitly marked unverified against a primary source — see this "
        f"model's source repository, [{SOURCE_REPO_URL}]({SOURCE_REPO_URL}), "
        "`datasets/plates/plates.yaml` and `DATASET_SPEC.md`.",
        "",
        "## Metrics",
        "",
        "Measured with the source repository's `edge/anpr/evaluate.py` against "
        "the SAME 2,076-image synthetic validation split for both rows, so the "
        "comparison is apples-to-apples. Exact-match and CER are never blended "
        "across synthetic and real data.",
        "",
        "| | Exact-match | Mean CER | Sample count |",
        "|---|---|---|---|",
        f"| Pretrained baseline (`{BASE_MODEL_NAME}`, no fine-tune) | "
        f"{pct(baseline.get('exact_match_rate'))} | {num(baseline.get('mean_cer'))} | "
        f"{baseline.get('count', 'n/a')} |",
        f"| **This model (fine-tuned)** | "
        f"{pct(finetuned.get('exact_match_rate'))} | {num(finetuned.get('mean_cer'))} | "
        f"{finetuned.get('count', 'n/a')} |",
    ]
    if latin_baseline is not None:
        lines.append(
            f"| Latin-only OCR baseline, same images | "
            f"{pct(latin_baseline.get('exact_match_rate'))} | "
            f"{num(latin_baseline.get('mean_cer'))} | {latin_baseline.get('count', 'n/a')} |"
        )
    if real is not None:
        lines += [
            "",
            "### Real photographs (held out)",
            "",
            "Distinct vehicles from `mukulboro/nepali-private-license-plates` (CC BY 4.0), each "
            "transcribed independently twice; only plates both readings agreed on exactly are "
            "used, one crop per vehicle. The set is small, so treat these as indicative.",
            "",
            "| | Exact-match | Mean CER | Plates |",
            "|---|---|---|---|",
        ]
        if real_baseline is not None:
            lines.append(
                f"| Previous published model | {pct(real_baseline.get('exact_match_rate'))} | "
                f"{num(real_baseline.get('mean_cer'))} | {real_baseline.get('count', 'n/a')} |"
            )
        lines.append(
            f"| **This model** | {pct(real.get('exact_match_rate'))} | {num(real.get('mean_cer'))} | "
            f"{real.get('count', 'n/a')} |"
        )
        lines += [
            "",
            "Real-world accuracy is far below the synthetic figures: this is not yet a reliable "
            "reader of real plates, and the synthetic numbers above must not be read as one.",
        ]
    if real is None:
        lines += [
            "",
            "**Real-world plates: INSUFFICIENT SAMPLE.** No licensed real-plate "
            "photograph set exists yet for this project (source repo "
            "`DATASET_SPEC.md` section 6.5 requires >= 30 licensed crops with "
            "recorded source/licence/author before a real-plate rate can be "
            "reported). The numbers above are synthetic-only and must not be read "
            "as a real-world accuracy claim.",
        ]
    lines += [
        "",
        "## Intended use",
        "",
        "Recognition-only stage of an automatic number-plate reading pipeline "
        "for Devanagari-script plates, run after a separate plate-detection "
        "and rectification step (this model does not locate plates in an "
        "image). Built for the TrueWatch border-post monitoring project.",
        "",
        "## License",
        "",
        f"Apache-2.0, inherited from the base model `{BASE_MODEL_NAME}` "
        f"(PaddleOCR, {BASE_MODEL_LICENSE}). This repository contains only "
        "the fine-tuned weights and inference config, no code beyond the "
        "standard PaddleOCR inference format.",
        "",
        "## Source",
        "",
        f"Training and evaluation code: [{SOURCE_REPO_URL}]({SOURCE_REPO_URL}), "
        "`edge/anpr/`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="exported inference model directory")
    ap.add_argument("--repo-id", default="sreekar12/truewatch-anpr-devanagari")
    ap.add_argument("--metrics-json", required=True, help="finetuned eval JSON (edge/anpr/evaluate.py --out)")
    ap.add_argument("--baseline-json", required=True, help="pretrained baseline eval JSON, same val split")
    ap.add_argument("--latin-baseline-json", default=None, help="optional latin-head baseline eval JSON")
    ap.add_argument("--real-metrics-json", default=None, help="optional eval JSON on held-out real plate photographs")
    ap.add_argument("--real-baseline-json", default=None, help="optional eval JSON of the previous model on the same real plates")
    ap.add_argument("--manifest-out", required=True, help="where to write the pinned manifest JSON")
    ap.add_argument("--dry-run", action="store_true", help="build the card/manifest locally, do not call the Hub")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"model dir not found: {model_dir}")
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"no files found under {model_dir}")

    finetuned = load_summary(Path(args.metrics_json))
    baseline = load_summary(Path(args.baseline_json))
    latin_baseline = load_summary(Path(args.latin_baseline_json)) if args.latin_baseline_json else None
    real = load_summary(Path(args.real_metrics_json)) if args.real_metrics_json else None
    real_baseline = load_summary(Path(args.real_baseline_json)) if args.real_baseline_json else None

    ft_acc = finetuned.get("exact_match_rate")
    base_acc = baseline.get("exact_match_rate")
    if ft_acc is None or base_acc is None:
        raise SystemExit("exact_match_rate missing from one of the metrics files; refusing to decide")
    if ft_acc <= base_acc:
        log(
            "ERROR",
            "fine-tuned model does NOT beat the pretrained baseline on exact-match; not uploading",
            finetuned_exact_match=ft_acc,
            baseline_exact_match=base_acc,
        )
        return 3
    log("INFO", "fine-tuned model beats baseline", finetuned_exact_match=ft_acc, baseline_exact_match=base_acc)

    card_text = build_model_card(
        repo_id=args.repo_id,
        finetuned=finetuned,
        baseline=baseline,
        latin_baseline=latin_baseline,
        sample_count=finetuned.get("count", 0),
        real=real,
        real_baseline=real_baseline,
    )

    if args.dry_run:
        log("INFO", "dry run: not calling the Hugging Face Hub")
        print(card_text)
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    log("INFO", "authenticated", user=who.get("name"))

    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)

    card_path = model_dir.parent / "README.md"
    card_path.write_text(card_text, encoding="utf-8")

    upload_paths = list(files) + [card_path]
    for path in files:
        rel = path.relative_to(model_dir)
        log("INFO", "uploading", path=str(rel))
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=str(rel),
            repo_id=args.repo_id,
            repo_type="model",
        )
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    refs = api.list_repo_refs(args.repo_id, repo_type="model")
    commit_sha = refs.branches[0].target_commit if refs.branches else None

    manifest = {
        "schema": "truewatch.hf_ocr_model.v1",
        "repo_id": args.repo_id,
        "revision": commit_sha,
        "base_model": BASE_MODEL_NAME,
        "base_model_license": BASE_MODEL_LICENSE,
        "files": [str(p.relative_to(model_dir)) for p in files],
        "resolve_urls": [
            f"https://huggingface.co/{args.repo_id}/resolve/{commit_sha}/{p.relative_to(model_dir)}"
            for p in files
        ],
        "sha256": {str(p.relative_to(model_dir)): sha256_of(p) for p in files},
        "metrics": {
            "finetuned": finetuned,
            "pretrained_baseline": baseline,
        },
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log("INFO", "manifest written", path=str(manifest_path), revision=commit_sha)
    log("INFO", "done", repo_url=f"https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
