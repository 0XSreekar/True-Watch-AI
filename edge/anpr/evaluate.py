#!/usr/bin/env python3
"""End-to-end ANPR accuracy: exact-match, CER, and a distance-bucketed breakdown.

Reads a PaddleOCR-style recognition label file (`path\\ttext`, UTF-8, one
sample per line — the format `datasets/plates/gen_plates.py` writes to
`datasets/plates/labels/rec_gt_{train,val}.txt`). Each labelled image is
already a cropped plate strip (DATASET_SPEC.md section 6.5: "cropped plate
strips, so the detection context is not needed"), so this script exercises
`rectify.py` -> `recognise.py` -> `postprocess.py` directly, without
`detect_plate.py`.

Distance bucketing is a documented ASSUMPTION, not a measurement: the corpus
scales each plate's rendered width to 60-260 px (`datasets/plates/degrade.py`,
comment: "a plate at a barrier is 260 px wide; a plate at 40 m is 60 px"). This
script treats that same width as a linear proxy for range and reports three
buckets. No physical camera-to-plate distance was measured; MEASUREMENTS.md
records no calibration for it, so "under 25 m" in ANPR_METRICS.md is reported
against this proxy explicitly, never presented as a measured range.

Also builds the DATA-DRIVEN confusion table `postprocess.py` reads: run this
script once against the TRAIN split with --write-confusions to produce it,
then evaluate the VAL (or a held-out real) split separately so corrections are
never fit and measured on the same samples.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # so `python edge/anpr/evaluate.py` finds sibling packages

import cv2  # noqa: E402

import postprocess  # noqa: E402
import recognise  # noqa: E402
import rectify  # noqa: E402

# Width buckets in px, matching datasets/plates/degrade.py's scale_to_target range (60-260).
DISTANCE_BUCKETS = (
    ("near (proxy: plate >= 200px wide)", 200, 10_000),
    ("mid, under 25 m proxy (120-199px)", 120, 200),
    ("far, beyond 25 m proxy (<120px)", 0, 120),
)


def log(level: str, message: str, **fields) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"{stamp} {level:<5} anpr.evaluate     {message}"
    if extra:
        line = f"{line} | {extra}"
    print(line, file=sys.stderr if level in ("ERROR", "FATAL") else sys.stdout, flush=True)


def load_labels(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        image_path, _, text = raw.partition("\t")
        text = text.split("\t")[0]  # an optional third column holds per-line texts (build_line_labels.py)
        if not text:
            continue
        rows.append((image_path, unicodedata.normalize("NFC", text)))
    return rows


def char_error_rate(hypothesis: str, reference: str) -> float:
    """Levenshtein distance / len(reference). 0.0 for an exact match, 1.0+ for a garbled read."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    prev = list(range(len(hypothesis) + 1))
    for i, rchar in enumerate(reference, start=1):
        cur = [i] + [0] * len(hypothesis)
        for j, hchar in enumerate(hypothesis, start=1):
            cost = 0 if rchar == hchar else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1] / len(reference)


def bucket_for_width(width: int) -> str:
    for name, lo, hi in DISTANCE_BUCKETS:
        if lo <= width < hi:
            return name
    return DISTANCE_BUCKETS[-1][0]


@dataclass
class SampleResult:
    path: str
    reference: str
    hypothesis: str
    script: str
    exact_match: bool
    cer: float
    width_px: int
    bucket: str


def evaluate(rows: list[tuple[str, str]], images_root: Path, limit: int | None, script: str | None) -> list[SampleResult]:
    results: list[SampleResult] = []
    for index, (rel_path, reference) in enumerate(rows):
        if limit is not None and index >= limit:
            break
        image_path = images_root / rel_path
        if not image_path.is_file() and rel_path.startswith("out/"):
            # gen_plates.py always labels with the "out/images/..." prefix from its
            # OWN default layout, even when --out pointed somewhere else (e.g. a
            # scratch directory). Fall back to images_root/images/<name> so a label
            # file survives being paired with a differently-rooted image directory.
            image_path = images_root / rel_path[len("out/"):]
        image = cv2.imread(str(image_path))
        if image is None:
            log("WARN", "unreadable image, skipped", path=str(image_path))
            continue
        width = image.shape[1]
        lines, _used_quad = rectify.prepare_lines(image)

        if script in ("devanagari", "latin"):
            # Force one head, bypassing script routing — used to measure the
            # Latin-model baseline (ANPR_METRICS.md section 1.2): what an
            # off-the-shelf Latin-only ANPR stack would have read on these
            # SAME images, scored against the SAME Devanagari ground truth.
            chosen_lines = []
            for image_line in lines:
                devanagari_reading, latin_reading = recognise.recognise_line(image_line)
                chosen_lines.append(devanagari_reading if script == "devanagari" else latin_reading)
            hypothesis = unicodedata.normalize(
                "NFC", "".join(line.text.replace(" ", "") for line in chosen_lines)
            )
            used_script = script
        else:
            result = recognise.read_plate(image)
            hypothesis = result.text
            used_script = result.script

        results.append(
            SampleResult(
                path=rel_path,
                reference=reference,
                hypothesis=hypothesis,
                script=used_script,
                exact_match=hypothesis == reference,
                cer=char_error_rate(hypothesis, reference),
                width_px=width,
                bucket=bucket_for_width(width),
            )
        )
        if (index + 1) % 200 == 0:
            log("INFO", "progress", done=index + 1, total=len(rows))
    return results


def summarise(results: list[SampleResult]) -> dict:
    if not results:
        return {"count": 0, "exact_match_rate": None, "mean_cer": None, "buckets": {}}
    exact = sum(1 for r in results if r.exact_match)
    mean_cer = sum(r.cer for r in results) / len(results)
    buckets: dict[str, dict] = {}
    for name, _lo, _hi in DISTANCE_BUCKETS:
        subset = [r for r in results if r.bucket == name]
        if not subset:
            buckets[name] = {"count": 0, "exact_match_rate": None, "mean_cer": None}
            continue
        buckets[name] = {
            "count": len(subset),
            "exact_match_rate": round(sum(1 for r in subset if r.exact_match) / len(subset), 4),
            "mean_cer": round(sum(r.cer for r in subset) / len(subset), 4),
        }
    return {
        "count": len(results),
        "exact_match_rate": round(exact / len(results), 4),
        "mean_cer": round(mean_cer, 4),
        "buckets": buckets,
    }


def build_confusion_table(results: list[SampleResult]) -> dict[str, list[str]]:
    """Character-level substitutions observed on WRONG predictions, aligned by position.

    This is a measurement, not a guess: only pairs actually produced by the
    recognition model against ground truth on THIS run's samples are recorded.
    Deletions/insertions (length mismatches) are skipped for a position-aligned
    pass; they do not produce a same-position substitution pair.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for result in results:
        if result.exact_match:
            continue
        ref, hyp = result.reference, result.hypothesis
        for r_char, h_char in zip(ref, hyp):
            if r_char != h_char:
                counts[(r_char, h_char)] += 1
    table: dict[str, list[str]] = {}
    for (ref_char, wrong_char), count in counts.most_common():
        if count < 2:  # a single occurrence could be noise, not a real confusion pattern
            continue
        table.setdefault(wrong_char, [])
        if ref_char not in table[wrong_char]:
            table[wrong_char].append(ref_char)
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="rec_gt_{train,val}.txt from gen_plates.py")
    ap.add_argument("--images-root", required=True, help="directory the label paths are relative to")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="write per-run JSON summary here")
    ap.add_argument("--write-confusions", default=None, help="write the measured confusion table here")
    ap.add_argument(
        "--script",
        choices=["auto", "devanagari", "latin"],
        default="auto",
        help="'auto' uses recognise.py's script routing (the production path); "
        "'devanagari'/'latin' force one head, bypassing routing, to measure "
        "the Latin-model baseline (ANPR_METRICS.md section 1.2)",
    )
    args = ap.parse_args()

    rows = load_labels(Path(args.labels))
    if not rows:
        log("ERROR", "no labelled rows found", labels=args.labels)
        return 2
    log("INFO", "loaded labels", count=len(rows), labels=args.labels)

    script = None if args.script == "auto" else args.script
    results = evaluate(rows, Path(args.images_root), args.limit, script=script)
    summary = summarise(results)
    log("INFO", "summary", **{k: v for k, v in summary.items() if k != "buckets"})
    for name, stats in summary["buckets"].items():
        log("INFO", "bucket", name=name, **stats)

    if args.write_confusions:
        table = build_confusion_table(results)
        out_path = Path(args.write_confusions)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "schema": "truewatch.anpr_confusions.v1",
                    "source_labels": str(args.labels),
                    "sample_count": len(results),
                    "confusions": table,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        log("INFO", "confusion table written", path=str(out_path), pairs=sum(len(v) for v in table.values()))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"summary": summary, "samples": [asdict(r) for r in results]},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        log("INFO", "results written", path=str(out_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
