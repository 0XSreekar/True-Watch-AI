"""Hard-set regression gate: run the frozen Phase 1 hard set through the detector and compare.

WHY this exists. DATASET_SPEC 2.6 freezes 280 val frames (tiny objects, washed-out thermal,
crowds, occlusion) that every later phase touching the detector must not get worse on. An
overall mAP can rise while exactly these frames regress, so this is a separate pass/fail check.

Mapping. `datasets/manifests/hard_set.txt` lists the INTERMEDIATE `record["image"]` paths that
06_split.py chose, but the dataset the detector sees has final files named
`{source}_{raw stem}_{modality}` (09_build_yolo_ds.py). So each entry is mapped to final files:

  * with a split index (--index, or processed/index/split.jsonl found beside the dataset root):
    the record whose `image` equals the entry gives the final stem directly;
  * otherwise, or when the entry is not in the index, the entry's raw stem is matched against the
    final stems ending `_visible` or `_lwir`. When one raw stem matches BOTH a visible and an LWIR
    file (an LLVIP pair) the entry maps to BOTH images and both are counted;
  * an entry that matches nothing is unresolved, and an ambiguous match (two files of the same
    modality) is unresolved as well: this tool does not guess.

A gate that silently shrinks is worse than no gate, so unresolved entries end the run with exit 2
unless --allow-partial is given, and that choice is recorded in the JSON and printed. The hard set
is val only: an entry that resolves into images/train, or that the index places in test, is a
corrupt manifest and always ends the run with exit 2. This script never lists images/test.

The gate. Person AP50 per slice (day and IR separately, never blended) and person recall at
--report-conf for the size buckets of 19 px and below may each fall by at most --tolerance
(absolute) against `hardset_baseline.json`. Two identity checks come first, because numbers over a
different image set or with different inference settings are not comparable: the resolved image
list and the settings (imgsz, NMS IoU, floor, max-det, report conf, size basis) must equal the
baseline's.

Exit codes: 0 gate passed, or no baseline yet (`passed` is null, never "passed by default");
1 gate failed; 2 refused (entries unresolved, corrupt manifest, existing baseline without
--force-baseline, bad flag, missing data).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _common as C  # noqa: E402
import evaluate as E  # noqa: E402

SCHEMA = "truewatch.hardset.v1"
BASELINE_SCHEMA = "truewatch.hardset-baseline.v1"
EXIT_GATE_FAILED = 1
GATE_BUCKETS = tuple(b for b in C.SIZE_BUCKETS_PX if b <= 19)  # the measured cliff, MEASUREMENTS.md section 3
GATE_SLICES = (C.SLICE_DAY, C.SLICE_IR)
UNRESOLVED_LISTED = 20

_FINAL_STEM_RE = re.compile(r"^([^_]+)_(.+)_(visible|lwir)$")


# --------------------------------------------------------------------------------------------
# Manifest and index
# --------------------------------------------------------------------------------------------


def read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        E.fail(f"hard-set manifest not found: {path}. Run datasets/scripts/06_split.py or pass --hard-set.")
    entries = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    if not entries:
        E.fail(f"hard-set manifest {path} is empty; a gate over zero images would always pass")
    return entries


def load_index_records(path: Path | None) -> list[dict]:
    """Records of processed/index/split.jsonl that carry the fields the final-name rule needs."""
    if path is None:
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                rec = json.loads(line)
                if rec.get("source") and rec.get("modality") and rec.get("image"):
                    records.append(rec)
    return records


def final_stem_of(record: dict) -> str:
    """The rule 09_build_yolo_ds.py uses to name a placed file."""
    return f"{record['source']}_{Path(record['image']).stem}_{record['modality']}"


def stems_in_split_names_only(root: Path, split: str) -> set[str]:
    """File stems in images/<split>, from the directory listing alone; no file is opened.

    Used for train, to detect a hard-set entry that leaked into it. Test is never listed: the index's
    own `split` field is the only test signal, so the sealed folder stays untouched.
    """
    folder = root / "images" / split
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.iterdir() if p.suffix.lower() in C.IMAGE_SUFFIXES}


# --------------------------------------------------------------------------------------------
# Mapping a manifest entry to final images
# --------------------------------------------------------------------------------------------


@dataclass
class Mapping:
    entries: int
    images: list[Path]                 # unique, sorted, all inside images/val
    resolved_entries: int
    unresolved: list[str] = field(default_factory=list)   # includes the ambiguous ones
    ambiguous: list[str] = field(default_factory=list)
    leaked: list[dict] = field(default_factory=list)      # {"entry", "stem", "split"}
    pair_entries: int = 0              # entries that map to both a visible and an LWIR image
    index_used: bool = False


def _modality(stem: str) -> str | None:
    return C.modality_of_stem(stem)


def _is_single_or_pair(stems: Sequence[str]) -> bool:
    """One file, or exactly one visible and one LWIR file (an LLVIP pair). Anything else is a guess."""
    if len(stems) == 1:
        return True
    return len(stems) == 2 and {_modality(s) for s in stems} == {"visible", "lwir"}


def _index_candidates(recs: Sequence[dict]) -> list[tuple[str, str | None]]:
    """(final stem, split) per distinct final file. Where records disagree on the split, val wins:
    the hard set is val only, and the val file is the one this run would evaluate."""
    best: dict[str, str | None] = {}
    for rec in recs:
        stem, hint = final_stem_of(rec), rec.get("split")
        if stem not in best or hint == "val":
            best[stem] = hint
    return list(best.items())


def _placement(stem: str, split_hint: str | None, val_files: dict[str, Path], train_stems: set[str]) -> str:
    """'val', 'train', 'test', 'clash' (the same name exists in val and train, so a bare name cannot
    say which one the manifest meant) or 'missing'. The index's split beats the name."""
    if split_hint == "val":
        return "val" if stem in val_files else "missing"
    if split_hint in ("train", "test"):
        return split_hint
    in_val, in_train = stem in val_files, stem in train_stems
    if in_val and in_train:
        return "clash"
    return "val" if in_val else "train" if in_train else "missing"


def map_entries(
    entries: Sequence[str],
    records: Sequence[dict],
    val_files: dict[str, Path],
    train_stems: set[str],
) -> Mapping:
    """Map every manifest entry to final val images. See the module docstring for the rules."""
    by_image: dict[str, list[dict]] = {}
    by_raw_index: dict[str, list[dict]] = {}
    for rec in records:
        by_image.setdefault(rec["image"], []).append(rec)
        by_raw_index.setdefault(Path(rec["image"]).stem, []).append(rec)

    by_raw_final: dict[str, list[str]] = {}
    for stem in sorted(set(val_files) | train_stems):
        match = _FINAL_STEM_RE.match(stem)
        if match:
            by_raw_final.setdefault(match.group(2), []).append(stem)

    result = Mapping(entries=len(entries), images=[], resolved_entries=0, index_used=bool(records))
    chosen: dict[str, Path] = {}
    for entry in entries:
        raw = Path(entry).stem
        recs = by_image.get(entry) or by_raw_index.get(raw)
        if recs:
            candidates = _index_candidates(recs)
        else:
            candidates = [(s, None) for s in by_raw_final.get(raw, [])]

        if not candidates:
            result.unresolved.append(entry)
            continue
        if not _is_single_or_pair([s for s, _ in candidates]):
            result.unresolved.append(entry)
            result.ambiguous.append(entry)
            continue

        found: list[str] = []
        leaked_here = name_clash = False
        for stem, split_hint in candidates:
            where = _placement(stem, split_hint, val_files, train_stems)
            if where == "val":
                found.append(stem)
            elif where in ("train", "test"):
                leaked_here = True
                result.leaked.append({"entry": entry, "stem": stem, "split": where})
            elif where == "clash":
                name_clash = True
        if leaked_here:
            continue
        if name_clash:
            result.unresolved.append(entry)
            result.ambiguous.append(entry)
            continue
        if not found:
            result.unresolved.append(entry)
            continue
        result.resolved_entries += 1
        if len(found) == 2 and {_modality(s) for s in found} == {"visible", "lwir"}:
            result.pair_entries += 1
        for stem in found:
            chosen[stem] = val_files[stem]

    result.images = [chosen[k] for k in sorted(chosen)]
    return result


def assert_all_in_val(images: Sequence[Path], root: Path) -> None:
    """The hard set is drawn from val only: refuse if any resolved image sits elsewhere."""
    val_dir = (root / "images" / "val").resolve()
    outside = [p.name for p in images if p.resolve().parent != val_dir]
    if outside:
        E.fail(f"{len(outside)} resolved image(s) are not under images/val, e.g. {outside[:3]}. "
               "The hard set must be drawn from val only (DATASET_SPEC 2.6).")


def image_set_sha256(images: Sequence[Path]) -> str:
    """Identity of the evaluated image list, independent of where the dataset is mounted."""
    return hashlib.sha256("\n".join(sorted(p.stem for p in images)).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------------------------


def gate_settings(s: E.Settings) -> dict:
    """The inference settings that change a number; a baseline is only comparable at equal settings."""
    return {"imgsz": s.imgsz, "nms_iou": s.nms_iou, "conf_floor": s.conf_floor, "max_det": s.max_det,
            "report_conf": s.report_conf, "size_basis": s.size_basis}


def gate_numbers(slices: dict[str, dict]) -> dict:
    """The tolerance-relevant numbers: person AP50 and small-bucket person recall, per slice."""
    ap50: dict[str, float | None] = {}
    recall: dict[str, dict[str, float | None]] = {}
    n_gt: dict[str, dict[str, int]] = {}
    for name in GATE_SLICES:
        s = slices[name]
        ap50[name] = s["classes"]["person"]["ap50"]
        rows = {r["bucket_px"]: r for r in s["size_buckets"]["person"]}
        recall[name] = {str(b): rows[b]["recall"] for b in GATE_BUCKETS}
        n_gt[name] = {str(b): rows[b]["n_gt"] for b in GATE_BUCKETS}
    return {"person_ap50": ap50, "person_recall_at_report_conf": recall, "person_n_gt_by_bucket": n_gt}


def _check(name: str, slice_name: str | None, current, baseline, tolerance: float, n_gt: int | None = None) -> dict:
    """One gate row. A metric that existed in the baseline and is gone now fails; nothing is defaulted."""
    if baseline is None:
        ok, delta = True, None      # nothing to regress against (no ground truth in the baseline either)
    elif current is None:
        ok, delta = False, None     # measurable before, not now: the comparison cannot be made
    else:
        delta = current - baseline
        ok = delta >= -tolerance - 1e-9
    row = {"name": name, "slice": slice_name, "current": current, "baseline": baseline, "delta": delta, "ok": ok}
    if n_gt is not None:
        row["n_gt"] = n_gt
    return row


def compare_to_baseline(current: dict, baseline: dict, settings: dict, image_set: str, tolerance: float) -> list[dict]:
    checks = [
        {"name": "same_inference_settings", "slice": None, "current": settings, "baseline": baseline.get("settings"),
         "delta": None, "ok": settings == baseline.get("settings")},
        {"name": "same_image_set", "slice": None, "current": image_set[:16],
         "baseline": str(baseline.get("hard_set", {}).get("image_set_sha256", ""))[:16], "delta": None,
         "ok": image_set == baseline.get("hard_set", {}).get("image_set_sha256")},
    ]
    for name in GATE_SLICES:
        checks.append(_check("person_ap50", name, current["person_ap50"][name],
                             baseline.get("person_ap50", {}).get(name), tolerance))
    for name in GATE_SLICES:
        base_rows = baseline.get("person_recall_at_report_conf", {}).get(name, {})
        for bucket in GATE_BUCKETS:
            key = str(bucket)
            checks.append(_check(f"person_recall_bucket_{bucket}px", name,
                                 current["person_recall_at_report_conf"][name][key], base_rows.get(key),
                                 tolerance, current["person_n_gt_by_bucket"][name][key]))
    return checks


def load_baseline(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        E.fail(f"baseline {path} is unreadable ({exc}). Fix or remove it, or re-record with --write-baseline --force-baseline.")
    if not isinstance(data, dict) or data.get("schema") != BASELINE_SCHEMA:
        E.fail(f"baseline {path} is not a {BASELINE_SCHEMA} file. Re-record with --write-baseline --force-baseline.")
    return data


def build_gate(baseline_path: Path, baseline: dict | None, current: dict, settings: dict, image_set: str,
               tolerance: float, writing_baseline: bool) -> dict:
    gate = {"baseline": None, "tolerance": tolerance, "passed": None, "checks": []}
    if writing_baseline:
        gate["note"] = "this run records the baseline, so nothing was compared"
        return gate
    if baseline is None:
        gate["note"] = (f"no baseline at {baseline_path}; the gate did not run. Record one with --write-baseline "
                        "after a run you accept as the reference")
        return gate
    gate["baseline"] = E.display_path(baseline_path)
    gate["checks"] = compare_to_baseline(current, baseline, settings, image_set, tolerance)
    gate["passed"] = all(c["ok"] for c in gate["checks"])
    return gate


def baseline_document(report: dict, mapping: Mapping, numbers: dict, settings: dict, manifest_sha: str,
                      image_set: str, partial: bool) -> dict:
    return {
        "schema": BASELINE_SCHEMA,
        "created": report["created"],
        "date": report["created"][:10],
        "weights": {"name": Path(report["weights"]["path"]).name, "sha256": report["weights"]["sha256"]},
        "split": report["split"],
        "settings": settings,
        "hard_set": {"manifest_sha256": manifest_sha, "entries": mapping.entries, "resolved": mapping.resolved_entries,
                     "images": len(mapping.images), "image_set_sha256": image_set, "partial": partial},
        "host": report["host"],
        "device": report["device"],
        "ultralytics": report["ultralytics"],
        "n_images": {name: report["slices"][name]["n_images"] for name in GATE_SLICES},
        **numbers,
        "notes": [n for n in report["notes"] if "synthetic fixture" in n or "partial" in n.lower()],
    }


# --------------------------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------------------------


def render_mapping(mapping: Mapping, manifest: Path, allow_partial: bool) -> list[str]:
    lines = [
        f"hard set {manifest.name}: {mapping.entries} entries -> {mapping.resolved_entries} resolved to "
        f"{len(mapping.images)} images in images/val ({mapping.pair_entries} LLVIP pair(s) mapped to both files), "
        f"{len(mapping.unresolved)} unresolved ({len(mapping.ambiguous)} ambiguous); "
        f"{'split index used' if mapping.index_used else 'no split index: matched by raw stem'}",
    ]
    if mapping.unresolved and allow_partial:
        lines += ["", "!" * 78,
                  f"!! PARTIAL HARD SET: {len(mapping.unresolved)} of {mapping.entries} entries could not be resolved.",
                  "!! --allow-partial was given, so this gate covers fewer images than the frozen set.",
                  "!! A pass here is NOT a pass on the hard set. The JSON records partial=true.",
                  "!" * 78, ""]
    return lines


def settings_diff(current: dict, baseline: dict) -> tuple[str, str]:
    """Which inference settings differ, as 'imgsz 640' / 'imgsz 320'; 'same' when none do."""
    keys = sorted(k for k in set(current) | set(baseline) if current.get(k) != baseline.get(k))
    if not keys:
        return "same", "same"
    return ", ".join(f"{k} {current.get(k)}" for k in keys), ", ".join(f"{k} {baseline.get(k)}" for k in keys)


def render_gate(gate: dict, partial: bool = False) -> list[str]:
    lines = ["", f"=== HARD-SET GATE (tolerance {gate['tolerance']:.3f} absolute) ==="]
    tail = ["  !! PARTIAL hard set (--allow-partial): this result does not cover the frozen set"] if partial else []
    if gate["passed"] is None:
        return lines + [f"  NOT EVALUATED: {gate.get('note', 'no baseline')}"] + tail
    rows = []
    for c in gate["checks"]:
        if isinstance(c["current"], dict):
            cur, base = settings_diff(c["current"], c["baseline"] or {})
        elif isinstance(c["current"], str):
            cur, base = c["current"], str(c["baseline"])
        else:
            cur, base = C.fmt(c["current"]), C.fmt(c["baseline"])
        rows.append([c["name"], c["slice"] or "-", cur, base, C.fmt(c["delta"], 3, "-"), c.get("n_gt", ""),
                     "ok" if c["ok"] else "FAIL"])
    lines += E.format_table(["check", "slice", "current", "baseline", "delta", "n_gt", "status"], rows)
    lines.append(f"  gate baseline {gate['baseline']}")
    lines.append("  HARD-SET GATE: " + ("PASSED" if gate["passed"] else "FAILED (exit code 1)"))
    return lines + tail


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    E.add_engine_args(ap)
    ap.add_argument("--hard-set", default=str(C.HARD_SET_MANIFEST), help="frozen manifest of Phase 1 image paths")
    ap.add_argument("--baseline", default=None,
                    help="baseline JSON to compare with or write; when omitted: <out-dir>/hardset_baseline.json")
    ap.add_argument("--write-baseline", action="store_true",
                    help="record this run as the baseline instead of comparing; nothing is compared")
    ap.add_argument("--force-baseline", action="store_true", help="allow --write-baseline to replace an existing baseline")
    ap.add_argument("--tolerance", type=float, default=0.02, help="largest allowed absolute drop per gated number")
    ap.add_argument("--allow-partial", action="store_true",
                    help="evaluate the entries that resolve even if some do not (recorded in the JSON, printed loudly)")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = E.Settings.from_args(args)
    settings.validate()
    if not 0.0 <= args.tolerance < 1.0:
        E.fail(f"--tolerance must be in [0, 1) (got {args.tolerance})")
    if args.force_baseline and not args.write_baseline:
        E.fail("--force-baseline only makes sense together with --write-baseline")
    out_dir = Path(args.out_dir)
    baseline_path = Path(args.baseline) if args.baseline else out_dir / "hardset_baseline.json"
    if args.write_baseline and baseline_path.exists() and not args.force_baseline:
        E.fail(f"a baseline already exists at {baseline_path}; it is the reference every later run is judged "
               "against. Pass --force-baseline to replace it deliberately.")

    root = E.as_refusal(C.resolve_data_root, args.data_root)
    class_names = C.load_class_names(root)
    if "person" not in class_names:
        E.fail(f"the gate is defined on class 'person', which this dataset does not have: {class_names}")
    size_classes = tuple(dict.fromkeys(("person",) + settings.size_classes))
    settings = replace(settings, size_classes=size_classes)
    unknown = [c for c in size_classes if c not in class_names]
    if unknown:
        E.fail(f"--size-classes {unknown} are not classes of this dataset: {class_names}")

    manifest = Path(args.hard_set)
    entries = read_manifest(manifest)
    index_path = E.find_index(root, args.index)
    records = load_index_records(index_path)
    val_files = {p.stem: p for p in E.as_refusal(C.list_split_images, root, "val")}
    mapping = map_entries(entries, records, val_files, stems_in_split_names_only(root, "train"))
    print("\n".join(render_mapping(mapping, manifest, args.allow_partial)), flush=True)

    if mapping.leaked:
        shown = [f"{x['entry']} -> {x['stem']} ({x['split']})" for x in mapping.leaked[:5]]
        E.fail(f"{len(mapping.leaked)} hard-set image(s) resolve outside val: {shown}. The hard set is drawn "
               "from val only (DATASET_SPEC 2.6) and this manifest is corrupt; re-run datasets/scripts/06_split.py.")
    if mapping.unresolved and not args.allow_partial:
        E.fail(f"{len(mapping.unresolved)} of {mapping.entries} hard-set entries did not resolve to a val image, "
               f"e.g. {mapping.unresolved[:3]}. A gate that silently shrinks is worse than none: pass --index "
               "processed/index/split.jsonl or the right --data-root, or --allow-partial to accept a partial gate.")
    if not mapping.images:
        E.fail("no hard-set entry resolved to an image; there is nothing to evaluate")
    assert_all_in_val(mapping.images, root)

    resolver = C.MetaResolver(index_path)
    sha256 = C.sha256_file(settings.weights)
    tag = args.tag or f"{settings.weights.stem}_{sha256[:8]}"
    cache_path = Path(args.preds_cache) if args.preds_cache else out_dir / "cache" / f"preds_hardset_{tag}.npz"
    scored = E.run_pipeline(mapping.images, settings, resolver, class_names, cache_path,
                            bool(args.reuse_preds or args.preds_cache))

    partial = bool(mapping.unresolved)
    notes = E.base_notes(settings, "val", scored, resolver.has_index, E.is_synthetic(root), class_names)
    notes.append("hard set: the frozen regression images (DATASET_SPEC 2.6); the numbers are over those images only")
    if partial:
        notes.append(f"PARTIAL hard set: {len(mapping.unresolved)} of {mapping.entries} entries unresolved, accepted with --allow-partial")
    report = E.make_report(schema=SCHEMA, tag=tag, s=settings, split="val", scored=scored,
                           class_names=class_names, index_used=resolver.has_index, notes=notes)

    manifest_sha = C.sha256_file(manifest)
    image_set = image_set_sha256(mapping.images)
    report["hard_set"] = {
        "manifest": E.display_path(manifest),
        "manifest_sha256": manifest_sha,
        "entries": mapping.entries,
        "resolved": mapping.resolved_entries,
        "images": len(mapping.images),
        "pair_entries": mapping.pair_entries,
        "unresolved": mapping.unresolved[:UNRESOLVED_LISTED],
        "unresolved_count": len(mapping.unresolved),
        "ambiguous_count": len(mapping.ambiguous),
        "index_used": mapping.index_used,
        "allow_partial": bool(args.allow_partial),
        "partial": partial,
        "image_set_sha256": image_set,
    }
    numbers = gate_numbers(scored.slices)
    baseline = None if args.write_baseline else load_baseline(baseline_path)
    report["gate"] = build_gate(baseline_path, baseline, numbers, gate_settings(settings), image_set,
                                args.tolerance, args.write_baseline)

    json_path = out_dir / f"hardset_{tag}.json"
    C.write_json(json_path, report)
    print(E.render_report(report))
    print("\n".join(render_gate(report["gate"], partial)))
    print(f"\nwrote {json_path}\n      {cache_path} (raw predictions)")

    if args.write_baseline:
        C.write_json(baseline_path, baseline_document(report, mapping, numbers, gate_settings(settings),
                                                      manifest_sha, image_set, partial))
        print(f"      {baseline_path} (baseline recorded{'; PARTIAL hard set' if partial else ''})")
    return EXIT_GATE_FAILED if report["gate"]["passed"] is False else 0


if __name__ == "__main__":
    sys.exit(main())
