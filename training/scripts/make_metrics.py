#!/usr/bin/env python3
"""Render results/METRICS.md from the JSON that the Phase 2 scripts write. Nothing is typed in by hand.

WHY this exists. The submission's honesty rests on one rule: a figure is either measured, on a named
host and split, or it is absent. A hand-edited results table drifts from the runs behind it and
tempts whoever edits it to round in the project's favour. This script removes both problems: every
number in the document is read from a JSON that a script wrote, every verdict is computed by code
that is unit-tested against each branch, and the "why is this target not met" list is a
deterministic ranker over the same data, not prose.

Verdict rules (docs/PHASE_MINUS1_SCOPE.md 5.2 targets; _metrics.map_verdict decides the two mAP ones):

  MET           the point estimate reaches the target AND so does the lower 95% bound.
  PARTIAL       reached but not established (the lower bound is below the target), or only the
                narrower daylight reading reaches it (day only; IR has no narrower reading), or, for
                the false-alert target, the best a detector-level proxy can say.
  NOT MET       measured and short of the target; for false alerts, also when the budget point is
                unreachable, unresolvable or vacuous (recall 0).
  NOT ASSESSED  no measurement exists. Targets that belong to later phases (range, plates, fence
                crossings) and the Jetson latency target are always here, with the owner named.

MET is never printed for anything without a measurement, and with no inputs the document says
"no measurements yet" and lists every target as NOT ASSESSED. Any subset of inputs works.

Exit codes: 0 written; 2 an input file is missing, unreadable or of the wrong schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
import _metrics as M  # noqa: E402

SCHEMAS = {
    "eval": "truewatch.eval.v1",
    "hardset": "truewatch.hardset.v1",
    "sweep": "truewatch.sweep.v1",
    "export": "truewatch.export.v1",
    "benchmark": "truewatch.benchmark.v1",
}

TARGET_DAY_AP50 = 0.85
TARGET_IR_AP50 = 0.75
FALSE_ALERTS_PER_CAMERA_DAY = 5
SMALL_PX = 19            # buckets at or below this height are the "under ~20 px" objects
SMALL_EDGE_PX = C.bucket_upper_edges()[C.SIZE_BUCKETS_PX.index(SMALL_PX) - 1]   # where the 19 px bucket ends (22.6)
LOW_N = 30               # fewer ground-truth boxes than this and a metric is an anecdote
TOP_REASONS = 3
DATASET_SPEC_PARTS = "docs/DATASET_SPEC.md"

# The one sentence any mention of the Jetson figure goes through, so the word "target" can never be
# separated from it (ARCHITECTURE_V2 7.3).
JETSON_TARGET = (
    "the slide-3 design target of about 30 ms for YOLO11-s INT8 inference at 640 px on a Jetson Orin "
    "Nano Super, hardware this project does not own"
)

PENDING_PHASE = {
    "range": "Phase 6 (tiled far-field inference, DORI grading and the ground-plane homography)",
    "plate": "Phase 7 (ANPR on the Nepali plate set)",
    "fence": "Phases 4-5 (fusion, tracking and the rule engine; it is an event-level rate, not a detector metric)",
    "jetson": "no phase: it needs a Jetson Orin Nano Super, which this project does not own",
}


class InputError(Exception):
    """An input file cannot be used; the message says what to do."""


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


@dataclass
class Inputs:
    eval: dict | None = None
    hardset: dict | None = None
    sweep: dict | None = None
    benchmarks: list[dict] = field(default_factory=list)
    export: dict | None = None
    train_log: dict | None = None
    files: dict[str, list[str]] = field(default_factory=dict)

    @property
    def any_measurement(self) -> bool:
        return bool(self.eval or self.hardset or self.sweep or self.benchmarks or self.export or self.train_log)


def load_document(path: str | os.PathLike, kind: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise InputError(f"--{kind} {p} does not exist. Run the script that writes it first (see section 11 of the report).")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"{p} is not readable JSON ({exc}).")
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMAS[kind]:
        got = doc.get("schema") if isinstance(doc, dict) else type(doc).__name__
        raise InputError(f"{p} has schema {got!r}, expected {SCHEMAS[kind]!r}. Pass the file written by the matching script.")
    if kind in ("eval", "hardset"):
        slices = doc.get("slices")
        if not isinstance(slices, dict) or "day" not in slices or "ir" not in slices:
            raise InputError(f"{p} has no 'day' and 'ir' slices, so it cannot be reported per modality.")
    if kind == "sweep" and not isinstance(doc.get("slices"), dict):
        raise InputError(f"{p} has no 'slices'.")
    return doc


def summarise_train_log(path: str | os.PathLike) -> dict:
    """Facts about a train_log.csv that carry no accuracy figure.

    train.py's per-epoch validation is over the whole val set, day and IR together (Ultralytics
    validates that way), so its mAP, person AP50 and fitness columns are numbers over both
    modalities. They chose the checkpoint, and this report never reproduces them: a figure over both
    modalities is not a result of this project. Only the epoch range, the session count, the epoch
    that scored highest on fitness (the one best.pt holds) and the timing are read.
    Columns are found by name, so a renamed column degrades to n/a.
    """
    p = Path(path)
    if not p.is_file():
        raise InputError(f"--train-log {p} does not exist.")
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        columns = list(reader.fieldnames or [])

    def find(pattern: str) -> str | None:
        return next((name for name in columns if re.fullmatch(pattern, name.lower())), None)

    def number(row: dict, column: str | None) -> float | None:
        try:
            return float(row[column]) if column else None
        except (TypeError, ValueError, KeyError):
            return None

    epoch_col, fitness_col = find(r"epoch"), find(r"fitness")
    seconds_col = find(r"epoch_s|epoch_sec(ond)?s|epoch_time")
    session_col = find(r"session")
    epochs = [e for e in (number(r, epoch_col) for r in rows) if e is not None]
    fit = [(number(r, fitness_col), number(r, epoch_col)) for r in rows]
    fit = [(f, e) for f, e in fit if f is not None and e is not None]
    seconds = sorted(x for x in (number(r, seconds_col) for r in rows) if x is not None)
    whole = [int(e) for e in epochs]
    return {
        "file": p.name,
        "rows": len(rows),
        "columns": columns,
        "epoch_min": min(whole) if whole else None,
        "epoch_max": max(whole) if whole else None,
        "epochs_duplicated": sorted({e for e in whole if whole.count(e) > 1}),
        "sessions": len({r.get(session_col) for r in rows if r.get(session_col)}) if session_col else None,
        "best_fitness_epoch": int(max(fit)[1]) if fit else None,
        "epoch_seconds_median": seconds[len(seconds) // 2] if seconds else None,
        "hours_logged": sum(seconds) / 3600.0 if seconds else None,
    }


def load_inputs(
    eval_path=None, hardset_path=None, sweep_path=None, benchmark_paths=(), train_log=None, export_path=None
) -> Inputs:
    inputs = Inputs()
    for kind, path in (("eval", eval_path), ("hardset", hardset_path), ("sweep", sweep_path), ("export", export_path)):
        if path:
            setattr(inputs, kind, load_document(path, kind))
            inputs.files[kind] = [Path(path).name]
    for path in benchmark_paths or ():
        inputs.benchmarks.append(load_document(path, "benchmark"))
        inputs.files.setdefault("benchmark", []).append(Path(path).name)
    if train_log:
        inputs.train_log = summarise_train_log(train_log)
        inputs.files["train_log"] = [Path(train_log).name]
    return inputs


# --------------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------------


def f3(value) -> str:
    return C.fmt(value, 3)


def pct(value, digits: int = 0) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def sci(value) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def md_table(header: Sequence[str], rows: Sequence[Sequence], numeric_from: int = 1) -> str:
    align = [":--" if i < numeric_from else "--:" for i in range(len(header))]
    out = ["| " + " | ".join(cell(h) for h in header) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def short_sha(sha: str | None) -> str:
    return sha[:12] if sha else "n/a"


def host_summary(host: dict | None) -> str:
    if not host:
        return "n/a"
    quota = f", cgroup limit {host['cgroup_cpu_limit']:g} CPUs" if host.get("cgroup_cpu_limit") else ""
    return f"{host.get('cpu') or 'unknown CPU'} ({host.get('platform')}; {host.get('cpu_count_usable')} usable cores{quota})"


# --------------------------------------------------------------------------------------------
# Person statistics and verdicts
# --------------------------------------------------------------------------------------------


def person_stats(doc: dict | None, slice_key: str) -> dict | None:
    """Person-class numbers of one slice of an eval or hard-set document, or None if absent."""
    if not doc:
        return None
    s = doc.get("slices", {}).get(slice_key)
    if not s:
        return None
    row = s.get("classes", {}).get("person") or {}
    boot = s.get("bootstrap", {}).get("person") or {}
    ci = boot.get("ap50_ci95")
    return {
        "n_images": s.get("n_images", 0),
        "n_gt": row.get("n_gt", 0),
        "ap50": row.get("ap50"),
        "ap50_95": row.get("ap50_95"),
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "op_conf": s.get("operating_conf"),
        "ci": tuple(ci) if ci else None,
        "n_clusters": boot.get("n_clusters"),
        "ci_images": boot.get("n_images"),
    }


@dataclass
class Verdict:
    key: str
    name: str
    threshold: str
    measured: str
    verdict: str
    reason: str
    owner: str | None = None


def map_target_verdict(
    eval_doc: dict | None, slice_key: str, target: float, narrower_key: str | None = None
) -> tuple[str, str, str]:
    """(verdict, reason, measured text) for a person AP50 target. Delegates to _metrics.map_verdict."""
    if eval_doc is None:
        return M.NOT_ASSESSED, "no evaluation JSON was provided (--eval); run evaluate.py", "n/a"
    st = person_stats(eval_doc, slice_key)
    if st is None or st["n_images"] == 0:
        return M.NOT_ASSESSED, f"the evaluation holds no {slice_key} images", "n/a"
    narrower = person_stats(eval_doc, narrower_key) if narrower_key else None
    narrower_ap = narrower["ap50"] if narrower and narrower["n_gt"] else None
    ci_low = st["ci"][0] if st["ci"] else None
    verdict, reason = M.map_verdict(st["ap50"], ci_low, target, narrower_ap)
    if st["ap50"] is None:
        return verdict, reason, "n/a"
    measured = f"person AP50 {st['ap50']:.3f}"
    if st["ci"]:
        measured += f", 95% CI [{st['ci'][0]:.3f}, {st['ci'][1]:.3f}] ({st['n_clusters']} clusters, {st['n_gt']} boxes)"
    else:
        measured += f" ({st['n_gt']} boxes, no interval)"
    if narrower_ap is not None:
        measured += f"; daylight-only {narrower_ap:.3f}"
    return verdict, reason, measured


def _budget_is_vacuous(point: dict, n_gt: int) -> bool:
    return bool(point.get("vacuous")) or (
        n_gt > 0 and point.get("recall") == 0 and point.get("status") in ("met", "unresolvable")
    )


def false_alert_slice_verdict(name: str, s: dict) -> tuple[str, str]:
    """One modality's verdict for the false-alert target, and the reason."""
    if not s or not s.get("n_frames"):
        return M.NOT_ASSESSED, f"{name}: no frames were swept"
    point = s["false_alert_budget"]
    status = point["status"]
    if status == "met":
        if _budget_is_vacuous(point, s.get("n_gt", 0)):
            return M.NOT_MET, f"{name}: the budget is met only at recall 0, by detecting nothing"
        return M.PARTIAL, (f"{name}: false-positive upper bound {sci(point['fp_per_frame_upper95'])}/frame is within "
                           "budget, at detector level only (fusion and tracking, Phases 4-5, decide alerts)")
    if status == "unreachable":
        return M.NOT_MET, (f"{name}: unreachable, even at conf {point['conf']:.2f} there are {point['fp_count']} false "
                           f"positives in {point['n_frames']} frames")
    return M.NOT_MET, (f"{name}: unresolvable, {point['n_frames']:,} frames cannot show the budget is met "
                       f"(at least {point['frames_needed_to_resolve']:,} needed)")


def false_alert_verdict(sweep_doc: dict | None) -> tuple[str, str, str]:
    """(verdict, reason, measured text) for 'under 5 false alerts per camera per day'.

    PARTIAL at best: the count is a detector-level proxy. NOT MET when any modality's budget point is
    unreachable, unresolvable or vacuous. Never MET.
    """
    if sweep_doc is None:
        return M.NOT_ASSESSED, "no sweep JSON was provided (--sweep); run sweep_conf.py", "n/a"
    parts = [false_alert_slice_verdict(n, sweep_doc["slices"].get(n)) for n in C.MODALITY_SLICES]
    verdicts = [v for v, _ in parts]
    if M.NOT_MET in verdicts:
        overall = M.NOT_MET
    elif M.PARTIAL in verdicts:
        overall = M.PARTIAL
    else:
        overall = M.NOT_ASSESSED
    budget = sweep_doc["assumptions"]["budget_fp_per_frame"]
    measured = []
    for n in C.MODALITY_SLICES:
        s = sweep_doc["slices"].get(n)
        if not s or not s.get("n_frames"):
            measured.append(f"{n}: n/a")
            continue
        p = s["false_alert_budget"]
        rate = s["f1_optimal"].get("fp_per_frame")
        f1_part = "F1 point n/a" if s["f1_optimal"].get("f1") is None else (
            f"F1 point {sci(rate)} FP/frame" + (f" ({rate / budget:,.0f}x budget)" if rate else ""))
        measured.append(f"{n}: {f1_part}; budget point {p['status']} ({s['n_frames']:,} frames)")
    return overall, "; ".join(r for _, r in parts), "; ".join(measured)


def not_assessed(key: str, name: str, threshold: str, why: str) -> Verdict:
    return Verdict(key, name, threshold, "n/a", M.NOT_ASSESSED, why, PENDING_PHASE.get(key))


def compute_verdicts(inputs: Inputs) -> list[Verdict]:
    """The seven Phase 2 targets in slide order, each with a verdict from the rules above."""
    day_v, day_r, day_m = map_target_verdict(inputs.eval, "day", TARGET_DAY_AP50, "day/daylight")
    ir_v, ir_r, ir_m = map_target_verdict(inputs.eval, "ir", TARGET_IR_AP50)
    fa_v, fa_r, fa_m = false_alert_verdict(inputs.sweep)
    return [
        Verdict("day_map", "Person mAP@50, day (visible camera)", f">= {TARGET_DAY_AP50:.2f}", day_m, day_v, day_r),
        Verdict("ir_map", "Person mAP@50, IR (LWIR replicated to 3 channels)", f">= {TARGET_IR_AP50:.2f}", ir_m, ir_v, ir_r),
        Verdict("false_alerts", "False alerts per camera per day (detector-level proxy at best)",
                f"< {FALSE_ALERTS_PER_CAMERA_DAY}", fa_m, fa_v, fa_r),
        not_assessed("range", "Detection range 150 m at 25 px per metre", "150 m", "not a detector-only quantity"),
        not_assessed("plate", "Nepali plate recognition", ">= 85%", "no plate model or plate evaluation exists yet"),
        not_assessed("fence", "Missed fence crossings", "< 2%", "needs tracked events and rules, not a detector metric"),
        not_assessed("jetson", "Detector latency (design target of about 30 ms, INT8, 640 px, Jetson Orin Nano Super)",
                     "about 30 ms (design target)", "no measurement on the target hardware exists; CPU rows are not comparable"),
    ]


# --------------------------------------------------------------------------------------------
# The reasons ranker
# --------------------------------------------------------------------------------------------


@dataclass
class Reason:
    key: str
    text: str
    impact: float | None = None      # approximate AP50 points at stake; causes overlap, so not additive
    blocking: bool = False           # the single thing that stops a PARTIAL from being MET
    priority: int = 0                # only used by the false-alert ranker

    @property
    def note(self) -> str:
        """The parenthetical shown after the reason: what the impact number means for this reason."""
        if self.blocking:
            return "the one cause that alone stops this target from being MET"
        if self.impact is None:
            return ""
        return f"about {100 * self.impact:.1f} AP50 points at stake"


def rank_reasons(reasons: Sequence[Reason]) -> list[Reason]:
    """Blocking first, then priority, then larger impact, then name: the same input gives the same order."""
    return sorted(
        reasons,
        key=lambda r: (not r.blocking, r.priority, -(r.impact if r.impact is not None else -1.0), r.key),
    )


def _weighted(rows: Sequence[dict], key: str) -> float | None:
    usable = [r for r in rows if r.get("n_gt", 0) > 0 and r.get(key) is not None]
    den = sum(r["n_gt"] for r in usable)
    return sum(r["n_gt"] * r[key] for r in usable) / den if den else None


def reason_small_objects(s: dict) -> Reason | None:
    rows = (s.get("size_buckets") or {}).get("person")
    if not rows:
        return None
    total = sum(r["n_gt"] for r in rows)
    small = [r for r in rows if r["bucket_px"] <= SMALL_PX]
    large = [r for r in rows if r["bucket_px"] > SMALL_PX]
    n_small = sum(r["n_gt"] for r in small)
    if total == 0 or n_small == 0:
        return None
    share = n_small / total
    ap_small, ap_large = _weighted(small, "ap50"), _weighted(large, "ap50")
    rec_small, rec_large = _weighted(small, "recall"), _weighted(large, "recall")
    impact = share * max(0.0, ap_large - ap_small) if ap_small is not None and ap_large is not None else None
    text = (f"Small people: {pct(share)} of person ground truth falls in the {SMALL_PX}, 14 and 9 px buckets, under about "
            f"{SMALL_EDGE_PX:.1f} px ({n_small} of {total} boxes); recall there is {f3(rec_small)} against {f3(rec_large)} for "
            f"taller people at the report confidence, and AP50 {f3(ap_small)} against {f3(ap_large)}")
    return Reason("small_objects", text + ".", impact)


def reason_lighting(eval_doc: dict) -> Reason | None:
    day, night = person_stats(eval_doc, "day/daylight"), person_stats(eval_doc, "day/night")
    total = (person_stats(eval_doc, "day") or {}).get("n_gt", 0)
    if not day or not night or not day["n_gt"] or not night["n_gt"] or not total:
        return None
    if day["ap50"] is None or night["ap50"] is None:
        return None
    share = night["n_gt"] / total
    impact = share * max(0.0, day["ap50"] - night["ap50"])
    text = (f"Lighting: night visible frames hold {pct(share)} of day person ground truth ({night['n_gt']} of {total} boxes); "
            f"AP50 is {f3(night['ap50'])} at night against {f3(day['ap50'])} in daylight")
    return Reason("lighting_gap", text + ".", impact)


def reason_limiter(st: dict) -> Reason | None:
    p, r, ap = st["precision"], st["recall"], st["ap50"]
    if p is None or r is None or ap is None or st["op_conf"] is None:
        return None
    miss, false = 1.0 - r, 1.0 - p
    if miss + false <= 0:
        return None
    impact = max(0.0, 1.0 - ap) * max(miss, false) / (miss + false)
    if abs(miss - false) < 0.01:
        key, which = "limiter_balanced", "balanced: it misses about as many people as it invents"
    elif miss > false:
        key, which = "limiter_recall", "recall-limited: it misses more people than it invents"
    else:
        key, which = "limiter_precision", "precision-limited: it invents more detections than it misses"
    text = (f"Operating point: at the slice operating confidence {st['op_conf']:.2f} (F1 averaged over classes) recall is {f3(r)} "
            f"and precision is {f3(p)}, so the detector is {which}; the shortfall 1 - AP50 = {f3(1 - ap)} is split in "
            "proportion to (1 - recall) and (1 - precision)")
    return Reason(key, text + ".", impact)


def reason_source_gap(s: dict) -> Reason | None:
    sources = {k: v for k, v in (s.get("by_source") or {}).items()
               if v.get("person_n_gt", 0) > 0 and v.get("person_ap50") is not None}
    if len(sources) < 2:
        return None
    total = sum(v["person_n_gt"] for v in sources.values())
    best = None
    for name, v in sorted(sources.items()):
        others = [o for k, o in sources.items() if k != name]
        ref = sum(o["person_n_gt"] * o["person_ap50"] for o in others) / sum(o["person_n_gt"] for o in others)
        impact = (v["person_n_gt"] / total) * max(0.0, ref - v["person_ap50"])
        if best is None or impact > best[0]:
            best = (impact, name, v, ref)
    impact, name, v, ref = best
    text = (f"Source gap: {name} is {pct(v['person_n_gt'] / total)} of person ground truth in this slice and reads AP50 "
            f"{f3(v['person_ap50'])} against {f3(ref)} for the other sources")
    return Reason("source_gap", text + ".", impact)


def reason_interval(st: dict, target: float) -> Reason | None:
    ap, ci = st["ap50"], st["ci"]
    if ap is None:
        return None
    if ci is None:
        if ap >= target:
            return Reason("no_interval", f"No confidence interval was computed, so a point estimate of {f3(ap)} cannot be MET.",
                          None, blocking=True)
        return None
    lo, hi = ci
    clusters = f"{st['n_clusters']} clusters over {st['ci_images']} images"
    if ap >= target and lo < target:
        return Reason("interval", f"Sampling: AP50 {f3(ap)} reaches {target:.2f} but the lower 95% bound {f3(lo)} does not "
                                  f"({clusters}). More independent sequences (a narrower interval) or a higher AP50 would close "
                                  f"the gap of {target - lo:.3f}.", ap - lo, blocking=True)
    if ap < target <= hi:
        return Reason("interval", f"Sampling: the shortfall is inside the 95% interval [{f3(lo)}, {f3(hi)}] ({clusters}), so "
                                  "these frames cannot separate a miss from noise; frames without a known group count as "
                                  "their own cluster, which makes the interval optimistic.", (hi - lo) / 2)
    return None


def map_reasons(eval_doc: dict | None, slice_key: str, target: float) -> list[Reason]:
    """All candidate explanations for a mAP target that is not MET, ranked; the caller shows the top few."""
    if eval_doc is None:
        return [Reason("no_eval", "No evaluation JSON was provided, so nothing was measured. Run evaluate.py (section 11).")]
    st = person_stats(eval_doc, slice_key)
    if st is None or st["n_images"] == 0 or st["ap50"] is None:
        return [Reason("no_ground_truth", f"The evaluation holds no person ground truth in the {slice_key} slice, so there is "
                                          "nothing to explain. A slice with no ground truth cannot carry a verdict.")]
    s = eval_doc["slices"][slice_key]
    found = [reason_small_objects(s), reason_limiter(st), reason_source_gap(s), reason_interval(st, target)]
    if slice_key == "day":
        found.append(reason_lighting(eval_doc))
    found = [r for r in found if r]
    if not found:
        return [Reason("no_candidates", "None of the candidate explanations could be computed from this evaluation (it lacks "
                                        "size buckets, sub-slices or per-source rows). Re-run evaluate.py without --size-classes edits.")]
    return rank_reasons(found)


def false_alert_reasons(sweep_doc: dict | None) -> list[Reason]:
    if sweep_doc is None:
        return [Reason("no_sweep", "No sweep JSON was provided, so nothing was measured. Run sweep_conf.py (section 11).")]
    budget = sweep_doc["assumptions"]["budget_fp_per_frame"]
    out: list[Reason] = []
    for name in C.MODALITY_SLICES:
        s = sweep_doc["slices"].get(name)
        if not s or not s.get("n_frames"):
            out.append(Reason(f"{name}_no_frames", f"{name}: no frames were swept, so this modality has no false-alert measurement.",
                              blocking=True, priority=0))
            continue
        p, f1 = s["false_alert_budget"], s["f1_optimal"]
        n = s["n_frames"]
        if p["status"] == "unreachable":
            over = p["fp_per_frame"] / budget
            out.append(Reason(f"{name}_unreachable", f"{name}: even at conf {p['conf']:.2f} the detector makes {p['fp_count']} false "
                              f"positive(s) in {n:,} frames = {sci(p['fp_per_frame'])}/frame, {over:,.0f}x the budget of "
                              f"{sci(budget)}/frame; more frames cannot fix this.", blocking=True, priority=0))
        elif p["status"] == "unresolvable":
            upper = p["fp_per_frame_upper95"]
            out.append(Reason(f"{name}_unresolvable", f"{name}: {n:,} frames cannot show the budget is met. {p['fp_count']} false "
                              f"positive(s) observed in them are compatible with a true rate of up to {sci(upper)}/frame "
                              f"({upper / budget:,.0f}x the budget); at least {p['frames_needed_to_resolve']:,} frames are needed.", blocking=True, priority=0))
        if _budget_is_vacuous(p, s.get("n_gt", 0)):
            out.append(Reason(f"{name}_vacuous", f"{name}: the point that stays within budget has recall 0 (it detects nobody), so "
                              "the zero false-positive count says nothing about a working detector.", blocking=True, priority=1))
        elif p.get("conf") is not None and p["status"] != "unreachable" and f1.get("f1") is not None:
            over = f1["fp_per_frame"] / budget if f1["fp_per_frame"] else 0.0
            out.append(Reason(f"{name}_recall_cost", f"{name}: the F1-optimal point (recall {f3(f1['recall'])}) runs at "
                              f"{sci(f1['fp_per_frame'])} FP/frame = {over:,.0f}x the budget; the budget point keeps recall "
                              f"{f3(p['recall'])}. Meeting the budget costs recall {f3(f1['recall'])} to {f3(p['recall'])}.",
                              priority=2))
        elif f1.get("f1") is not None and f1["fp_per_frame"]:
            out.append(Reason(f"{name}_f1_over_budget", f"{name}: the F1-optimal point runs at {sci(f1['fp_per_frame'])} FP/frame = "
                              f"{f1['fp_per_frame'] / budget:,.0f}x the budget.", priority=2))
    if not sweep_doc.get("empty_frames"):
        out.append(Reason("composition", "Val frames keep their natural composition, most contain people; a border camera's day is "
                          "mostly empty road. No background frames (--empty-dir) were scored, so this rate is not the rate on "
                          "empty hours.", priority=3))
    out.append(Reason("proxy", "The count is a detector-level proxy: every unmatched person detection is treated as an alert. "
                      "Fusion, tracking and rules (Phases 4-5) are not built yet, so alert-level false alerts have not been "
                      "measured.", priority=4))
    return rank_reasons(out)


def reasons_for(verdict: Verdict, inputs: Inputs) -> list[Reason]:
    """Candidate reasons for one target that is not MET, ranked."""
    if verdict.key == "day_map":
        return map_reasons(inputs.eval, "day", TARGET_DAY_AP50)
    if verdict.key == "ir_map":
        return map_reasons(inputs.eval, "ir", TARGET_IR_AP50)
    if verdict.key == "false_alerts":
        return false_alert_reasons(inputs.sweep)
    return [Reason("not_measured", f"Not measured: {verdict.reason}. Owner: {verdict.owner}.")]


# --------------------------------------------------------------------------------------------
# Provenance checks
# --------------------------------------------------------------------------------------------


def is_synthetic(inputs: Inputs) -> bool:
    """True when any input says it was measured on the synthetic fixture (evaluate.py and sweep_conf.py write the note)."""
    for doc in (inputs.eval, inputs.hardset, inputs.sweep):
        if doc and any("synthetic fixture" in str(n).lower() for n in doc.get("notes", [])):
            return True
    for b in inputs.benchmarks:
        text = f"{b.get('label', '')} {b.get('model', {}).get('weights_provenance', '')}".lower()
        if "synthetic" in text or "smoke" in text:
            return True
    return False


def consistency_warnings(inputs: Inputs) -> list[str]:
    """Inputs that cannot describe the same detector or the same split, said plainly."""
    warnings = []
    shas = {}
    for label, doc, path in (("eval", inputs.eval, ("weights", "sha256")), ("hard set", inputs.hardset, ("weights", "sha256")),
                             ("sweep", inputs.sweep, ("weights", "sha256")), ("export", inputs.export, ("source_weights", "sha256"))):
        if doc and isinstance(doc.get(path[0]), dict) and doc[path[0]].get(path[1]):
            shas[label] = doc[path[0]][path[1]]
    if len(set(shas.values())) > 1:
        detail = ", ".join(f"{k} {short_sha(v)}" for k, v in shas.items())
        warnings.append(f"The inputs were produced from DIFFERENT weights ({detail}). Numbers below describe more than one "
                        "detector and must not be read as one result.")
    splits = {label: doc.get("split") for label, doc in (("eval", inputs.eval), ("sweep", inputs.sweep)) if doc and doc.get("split")}
    if len(set(splits.values())) > 1:
        warnings.append("The evaluation and the sweep were measured on different splits (" +
                        ", ".join(f"{k}: {v}" for k, v in splits.items()) + ").")
    sizes = {}
    for label, doc in (("eval", inputs.eval), ("hard set", inputs.hardset), ("sweep", inputs.sweep)):
        if doc and doc.get("imgsz"):
            sizes[label] = doc["imgsz"]
    if inputs.export and inputs.export.get("onnx", {}).get("imgsz"):
        sizes["export"] = inputs.export["onnx"]["imgsz"]
    if len(set(sizes.values())) > 1:
        warnings.append("Inference sizes differ between inputs (" + ", ".join(f"{k}: {v}" for k, v in sizes.items()) + ").")
    return warnings


# --------------------------------------------------------------------------------------------
# Section renderers
# --------------------------------------------------------------------------------------------

HONESTY_LINE_1 = ("**Every number below was measured on the split and host named beside it, or it is absent:** a target with no "
                  "measurement is NOT ASSESSED, and nothing here is estimated, projected or rounded in the project's favour.")
HONESTY_LINE_2 = ("**This is not a test-set result, not a Jetson result and not an alert-level or end-to-end result;** "
                  f"the only figure quoted from the deck for latency is {JETSON_TARGET}.")

SYNTHETIC_BANNER = ("> **SYNTHETIC FIXTURE.** At least one input was measured on the coloured-rectangle fixture that _synth.py writes. "
                    "Its numbers prove the tooling runs. They say nothing about the detector and must not be quoted as results.")


def split_caveat(split: str | None) -> str:
    if split == "test":
        return ("**Split: test, UNSEALED.** This is the single final measurement (Phase 11). It must not be used to choose "
                "anything; any further tuning against it makes it a second validation set.")
    if split == "train":
        return ("**Split: train.** The detector fitted these images. These numbers are optimistic by construction and are a "
                "wiring check, not a result.")
    return ("**Split: val, not test.** The test split is sealed until Phase 11 (DATASET_SPEC 2.5). Val also picked the checkpoint: "
            "Ultralytics ranks epochs by validation fitness, keeps best.pt by it and stops early on it (DATASET_SPEC 2.4 lists val "
            "for epoch selection and early stopping). Every number here was therefore chosen on the same frames it is reported on, "
            "so it is optimistic. How optimistic is not measured; the test split will say, once.")


def section_header(n: int, title: str) -> str:
    return f"\n## {n}. {title}\n"


def render_provenance(inputs: Inputs) -> str:
    out = [section_header(1, "Provenance and caveats")]
    if not inputs.eval:
        out.append("**No measurements yet.** No evaluation JSON (`--eval`) was provided, so this document contains no detector "
                   "accuracy figure and every accuracy target is NOT ASSESSED. Nothing has been invented to fill the tables. "
                   "To produce the numbers, run `evaluate.py` on a trained checkpoint (section 11 has the commands), then "
                   "re-run this script with `--eval`.\n")
        if not inputs.any_measurement:
            out.append("No other input was provided either: no hard-set run, sweep, benchmark, export or training log.\n")
    else:
        out.append(split_caveat(inputs.eval.get("split")) + "\n")
        ev = inputs.eval
        per_slice = ", ".join(f"{k} {ev['slices'][k].get('n_images', 0)} images" for k in ("day", "ir"))
        out.append(f"Measured on {ev.get('split')}: {per_slice}, imgsz {ev.get('imgsz')}, NMS IoU {ev.get('nms_iou')}. "
                   "Day (the visible camera) and IR (LWIR replicated to 3 channels) are separate slices throughout; no figure "
                   "in this document is computed over both.\n")
    for w in consistency_warnings(inputs):
        out.append(f"> **WARNING.** {w}\n")

    rows = []
    for kind, label in (("eval", "evaluation"), ("hardset", "hard set"), ("sweep", "sweep")):
        doc = getattr(inputs, kind)
        if doc:
            rows.append([label, inputs.files[kind][0], doc.get("tag"), doc.get("created"), doc.get("split"),
                         f"{Path(doc['weights']['path']).name} sha256 {short_sha(doc['weights'].get('sha256'))}"
                         if doc.get("weights") else "n/a", host_summary(doc.get("host"))])
    if inputs.export:
        ex = inputs.export
        rows.append(["export", inputs.files["export"][0], ex.get("tag"), ex.get("created"), "n/a",
                     f"{Path(ex['source_weights']['path']).name} sha256 {short_sha(ex['source_weights'].get('sha256'))}",
                     host_summary(ex.get("host"))])
    for b, name in zip(inputs.benchmarks, inputs.files.get("benchmark", [])):
        rows.append(["benchmark", name, b.get("label"), b.get("created"), "n/a",
                     f"{b['model'].get('file')} sha256 {short_sha(b['model'].get('sha256'))}", host_summary(b.get("host"))])
    if rows:
        out.append("Inputs read for this document (each figure below carries the host and split of its own input):\n")
        out.append(md_table(["kind", "file", "tag / label", "created (UTC)", "split", "weights", "measured on host"], rows, numeric_from=9))
        out.append("")

    notes = []
    for doc in (inputs.eval, inputs.hardset):
        for n in (doc or {}).get("notes", []):
            if n not in notes:
                notes.append(n)
    if notes:
        out.append("Notes the evaluation scripts recorded (the sweep's notes are in section 8):\n")
        out += [f"- {n}" for n in notes]
        out.append("")
    if inputs.export:
        out.append(render_export(inputs.export))
    if inputs.train_log:
        out.append(render_train_log(inputs.train_log))
    return "\n".join(out)


def render_export(ex: dict) -> str:
    onnx, par = ex.get("onnx", {}), ex.get("parity", {})
    verdict = "PASSED" if par.get("passed") else "FAILED"
    lines = [
        "ONNX export (`export_onnx.py`):\n",
        f"- file `{onnx.get('file')}`, opset {onnx.get('opset')}, imgsz {onnx.get('imgsz')}, "
        f"{(onnx.get('size_bytes') or 0) / 1e6:.1f} MB, sha256 {short_sha(onnx.get('sha256'))}; dynamic axes {onnx.get('dynamic_axes')}",
        f"- torch versus onnxruntime parity {verdict}: max abs diff {sci(par.get('max_abs_diff'))} against tolerance "
        f"{sci(par.get('tolerance'))}, by batch {par.get('by_batch')}, input kind {par.get('input_kind')}",
    ]
    hf = ex.get("hf")
    lines.append(f"- Hugging Face: {hf.get('url')}" if hf else "- Hugging Face: not pushed")
    return "\n".join(lines) + "\n"


def render_train_log(t: dict) -> str:
    lines = ["Training log (`train_log.csv`):\n"]
    if t["epoch_min"] is None:
        lines.append(f"- {t['rows']} rows; no `epoch` column was found (columns: {', '.join(t['columns'])})")
    else:
        sessions = f" over {t['sessions']} session(s)" if t["sessions"] else ""
        lines.append(f"- {t['rows']} rows, epochs {t['epoch_min']} to {t['epoch_max']}{sessions}")
    if t["epochs_duplicated"]:
        lines.append(f"- epochs logged more than once: {t['epochs_duplicated']} (a resumed session overlapped)")
    if t["best_fitness_epoch"] is not None:
        lines.append(f"- the epoch with the highest logged fitness is {t['best_fitness_epoch']}; that is the checkpoint the "
                     "training run kept as best.pt, chosen on the val split")
    else:
        lines.append("- no fitness column was found, so the log cannot say which epoch was kept")
    if t["epoch_seconds_median"] is not None:
        lines.append(f"- median {t['epoch_seconds_median']:.0f} s per epoch, {t['hours_logged']:.2f} h of epochs logged in total "
                     "(measured on the training host, which the log does not name)")
    lines.append("- the log's per-epoch validation figures (mAP, person AP50, fitness) are computed over the whole val set, day and "
                 "IR together, because that is how Ultralytics validates. They chose the checkpoint and are deliberately not "
                 "reproduced here: a figure over both modalities is not a result of this project. Sections 3 and 4 hold the "
                 "per-modality numbers.")
    return "\n".join(lines) + "\n"


def render_verdict_cell(v: Verdict, synthetic: bool) -> str:
    if synthetic and v.verdict != M.NOT_ASSESSED:
        return f"{v.verdict} (synthetic fixture: not a result)"
    return v.verdict


def render_targets(inputs: Inputs, verdicts: list[Verdict], synthetic: bool) -> str:
    out = [section_header(2, "Targets against measurements"),
           "Targets are the slide-5 and slide-3 figures quoted in `docs/PHASE_MINUS1_SCOPE.md` 5.2. They are targets, not results.\n"]
    rows = [[v.name, v.threshold, v.measured, render_verdict_cell(v, synthetic), v.reason + "." + (f" Owner: {v.owner}." if v.owner else "")]
            for v in verdicts]
    out.append(md_table(["target", "threshold", "measured", "verdict", "basis"], rows, numeric_from=9))
    out.append("")
    if inputs.eval:
        out.append("How to read the verdicts:\n")
        out.append(f"- `{M.MET}`: the point estimate reaches the target and so does the lower 95% bound (cluster bootstrap over sequences).")
        out.append(f"- `{M.PARTIAL}`: reached but not established (lower bound below the target), or only the daylight-only "
                   "reading reaches it. IR has no narrower reading.")
        out.append(f"- `{M.NOT_MET}`: measured, and short of the target. For false alerts also: the budget point is unreachable, "
                   "unresolvable or vacuous.")
        out.append(f"- `{M.NOT_ASSESSED}`: no measurement. The false-alert target can be `{M.PARTIAL}` at best, because the "
                   "sweep is a detector-level proxy.")
        out.append("- Person mAP@50 means AP50 of the person class alone. The mean over classes in sections 3 and 4 is not the target.")
        out.append("- Per-class AP50 on val, with val also having picked the checkpoint, is optimistic (section 1).\n")
    else:
        out.append("No verdict can be given for a target with no measurement, so every row above is NOT ASSESSED.\n")
    return "\n".join(out)


def composition_by_source(composition: dict[str, int], slice_key: str) -> str:
    """'idd 3 [daylight 3] | kaist 4 [night 4]' for one slice from the eval's composition counts."""
    by_source: dict[str, dict[str, int]] = {}
    for key, n in composition.items():
        parts = key.split("/", 2)
        if len(parts) != 3:
            continue
        modality, source, lighting = parts
        if slice_key == "ir" and modality != "ir":
            continue
        if slice_key != "ir" and modality != "day":
            continue
        if slice_key not in ("day", "ir") and lighting != slice_key.split("/", 1)[1]:
            continue
        by_source.setdefault(source, {})[lighting] = n
    cells = []
    for source in sorted(by_source):
        total = sum(by_source[source].values())
        if slice_key == "ir":
            cells.append(f"{source} {total}")
        else:
            detail = ", ".join(f"{k} {v}" for k, v in sorted(by_source[source].items()))
            cells.append(f"{source} {total} [{detail}]")
    return " | ".join(cells) if cells else "none"


def render_slice_classes(inputs: Inputs, slice_key: str, number: int, title: str) -> str:
    out = [section_header(number, f"{title} per-class")]
    ev = inputs.eval
    if ev is None:
        out.append("No measurement yet: no evaluation JSON was provided. Run `evaluate.py` (section 11).\n")
        return "\n".join(out)
    s = ev["slices"][slice_key]
    if not s.get("n_images"):
        out.append(f"The evaluation holds no {slice_key} images, so every metric is n/a. It is not zero.\n")
        return "\n".join(out)
    names = ev.get("class_names") or list(s["classes"])
    rc = f"{ev.get('report_conf', 0.35):.2f}"
    out.append(f"{s['n_images']} images from the {ev.get('split')} split. Composition by source [lighting]: "
               f"{composition_by_source(ev.get('composition', {}), slice_key)}.\n")
    rows = []
    for name in names:
        c = s["classes"][name]
        fixed = s["at_report_conf"]["classes"][name]
        flag = " †" if 0 < c["n_gt"] < LOW_N else ""
        rows.append([name, f"{c['n_gt']}{flag}", f3(c["ap50"]), f3(c["ap50_95"]), f3(c["precision"]), f3(c["recall"]),
                     f3(fixed["precision"]), f3(fixed["recall"])])
    mean = s["mean_over_present_classes"]
    rows.append(["mean over classes with ground truth", "", f3(mean["ap50"]), f3(mean["ap50_95"]), f3(mean["precision"]),
                 f3(mean["recall"]), "", ""])
    out.append(md_table(["class", "boxes", "AP50", "AP50-95", "P at op", "R at op", f"P at {rc}", f"R at {rc}"], rows))
    out.append("")
    out.append(f"`op` is this slice's F1-optimal confidence {s['operating_conf']:.3f} (F1 averaged over the classes present). "
               "n/a means no ground truth for that class in this slice; it is never 0. † fewer than "
               f"{LOW_N} boxes: the figure is an anecdote, not an estimate.\n")
    st = person_stats(ev, slice_key)
    if st and st["ci"]:
        out.append(f"Person AP50 {f3(st['ap50'])}, 95% CI [{st['ci'][0]:.3f}, {st['ci'][1]:.3f}] from a cluster bootstrap over "
                   f"{st['n_clusters']} clusters ({st['ci_images']} images). Frames without a known sequence are their own "
                   "cluster, which makes the interval optimistic.\n")
    elif st and st["ap50"] is not None:
        out.append("No confidence interval was computed for person AP50 in this slice.\n")
    src = s.get("by_source") or {}
    if src:
        out.append("Person AP50 by source, inside this slice only:\n")
        out.append(md_table(["source", "images", "person boxes", "person AP50"],
                            [[k, v["n_images"], f"{v['person_n_gt']}{' †' if 0 < v['person_n_gt'] < LOW_N else ''}",
                              f3(v["person_ap50"])] for k, v in sorted(src.items())]))
        out.append("")
    return "\n".join(out)


def render_small_objects(inputs: Inputs) -> str:
    out = [section_header(5, "Small-object slices"),
           "Person metrics by object pixel height, using the seven levels of `docs/MEASUREMENTS.md` section 3 "
           "(77, 54, 38, 27, 19, 14, 9 px). Day and IR are separate tables.\n"]
    ev = inputs.eval
    if ev is None:
        out.append("No measurement yet: no evaluation JSON was provided. Run `evaluate.py` (section 11).\n")
        return "\n".join(out)
    basis = ev.get("size_basis", "input")
    if basis == "input":
        out.append(f"Heights are in detector-input pixels at imgsz {ev.get('imgsz')} (letterboxed), the basis "
                   "`MEASUREMENTS.md` section 3 varied.\n")
    else:
        out.append(f"Heights are in stored-file pixels. `MEASUREMENTS.md` section 3 varied detector-input pixels, so its "
                   f"column below is not aligned with these buckets at imgsz {ev.get('imgsz')}.\n")
    rc = f"{ev.get('report_conf', 0.35):.2f}"
    for key, label in (("day", "Day (visible camera)"), ("ir", "IR (LWIR replicated to 3 channels)")):
        s = ev["slices"][key]
        out.append(f"### {label}\n")
        rows = (s.get("size_buckets") or {}).get("person")
        if not s.get("n_images") or not rows:
            out.append("No images or no size-bucket table for this slice; every metric is n/a.\n")
            continue
        body = []
        for r in rows:
            row = [f"{r['bucket_px']} px", C.bucket_range_label(r["bucket_px"]), f"{r['n_gt']}{' †' if 0 < r['n_gt'] < LOW_N else ''}",
                   f3(r["ap50"]), f3(r["precision"]), f3(r["recall"])]
            if key == "day":
                ref = C.MEASURED_PRETRAINED_RECALL_VS_NATIVE.get(r["bucket_px"])
                row.append(pct(ref))
            body.append(row)
        header = ["height", "range", "person boxes", "AP50", f"P at {rc}", f"R at {rc}"]
        if key == "day":
            header.append("pre-fine-tuning recall vs native detections (different quantity)")
        out.append(md_table(header, body, numeric_from=2))
        out.append("")
        total = sum(r["n_gt"] for r in rows)
        small = sum(r["n_gt"] for r in rows if r["bucket_px"] <= SMALL_PX)
        out.append(f"The {SMALL_PX}, 14 and 9 px buckets (heights under about {SMALL_EDGE_PX:.1f} px) hold "
                   f"{pct(small / total) if total else 'n/a'} of this slice's person boxes ({small} of {total}). "
                   f"† fewer than {LOW_N} boxes in the bucket.\n")
        if key == "day":
            out.append("The last column is copied from `docs/MEASUREMENTS.md` section 3: a COCO-pretrained model, before any "
                       "fine-tuning, on 23 KAIST visible frames at confidence 0.35. It is recall against that model's own "
                       "native-resolution detections (100% = it found again what it found at full size), not recall against "
                       "ground truth, and it can exceed 100%. It shows where the pre-fine-tuning cliff was; it is not a "
                       "baseline for the columns to its left.\n")
        else:
            out.append("There is no pre-fine-tuning column for IR: `MEASUREMENTS.md` section 3 was measured on visible frames "
                       "only, so quoting it here would attribute a visible measurement to the IR camera.\n")
    return "\n".join(out)


def render_lighting(inputs: Inputs) -> str:
    out = [section_header(6, "Visible lighting sub-slices"),
           "Refinements INSIDE the day slice, not extra slices and not additive with it. Lighting of IDD (daylight) and LLVIP "
           "(night) frames is taken from `DATASET_SPEC.md` 3.4, an assumption not measured per frame; KAIST lighting is the "
           "set's declared lighting in `datasets/config/splits.yaml`. `unresolved` means it could not be determined and is "
           "never counted as daylight.\n"]
    ev = inputs.eval
    if ev is None:
        out.append("No measurement yet: no evaluation JSON was provided. Run `evaluate.py` (section 11).\n")
        return "\n".join(out)
    rows = []
    for key in ("day/daylight", "day/night", "day/unresolved"):
        s = ev["slices"].get(key)
        if s is None:
            continue
        st = person_stats(ev, key)
        rows.append([key.split("/")[1], s.get("n_images", 0), composition_by_source(ev.get("composition", {}), key),
                     f"{st['n_gt']}{' †' if 0 < st['n_gt'] < LOW_N else ''}", f3(st["ap50"]), f3(st["precision"]), f3(st["recall"])])
    if not rows:
        out.append("The evaluation carries no lighting sub-slices.\n")
        return "\n".join(out)
    out.append(md_table(["lighting", "images", "composition by source", "person boxes", "person AP50", "P at op", "R at op"], rows, numeric_from=3))
    out.append("")
    out.append("`op` is each sub-slice's own F1-optimal confidence. n/a means no ground truth, never 0. † fewer than "
               f"{LOW_N} boxes. A sub-slice dominated by one source (see composition) measures that source as much as it "
               "measures the lighting.\n")
    return "\n".join(out)


def fmt_gate_pair(check: dict) -> tuple[str, str]:
    """(current, baseline) cell texts for one gate row. Settings rows carry dicts: show 'same' or what differs."""
    cur, base = check.get("current"), check.get("baseline")
    if isinstance(cur, dict) or isinstance(base, dict):
        cur, base = cur or {}, base or {}
        if cur == base:
            return "same", "same"
        keys = sorted(k for k in set(cur) | set(base) if cur.get(k) != base.get(k))
        return ", ".join(f"{k} {cur.get(k)}" for k in keys), ", ".join(f"{k} {base.get(k)}" for k in keys)

    def one(value) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)

    return one(cur), one(base)


def render_hardset(inputs: Inputs) -> str:
    out = [section_header(7, "Hard set (regression gate)")]
    hs = inputs.hardset
    if hs is None:
        out.append("No measurement yet: no hard-set JSON was provided. Run `eval_hardset.py` (section 11).\n")
        return "\n".join(out)
    h, gate = hs.get("hard_set", {}), hs.get("gate", {})
    entries, resolved = h.get("entries"), h.get("resolved")
    out.append(f"Manifest entries {entries}, resolved to val images {resolved}, images scored {h.get('images')}. "
               f"Unresolved {h.get('unresolved_count', len(h.get('unresolved', [])))}"
               + (f", e.g. {h.get('unresolved')[:3]}" if h.get("unresolved") else "") + ".\n")
    if h.get("partial") or (entries is not None and resolved is not None and resolved < entries):
        out.append("> **WARNING: PARTIAL HARD SET.** Some manifest entries did not resolve, so the gate below covers fewer images "
                   "than the frozen set. A gate that silently shrinks is worse than none.\n")
    passed = gate.get("passed")
    if passed is None:
        out.append("**Gate: NO BASELINE.** This is a first run: nothing was compared, so the gate has neither passed nor failed. "
                   "Write a baseline with `eval_hardset.py --write-baseline`, and compare later runs against it.\n")
    elif passed:
        out.append(f"**Gate: PASSED** against `{Path(gate['baseline']).name if gate.get('baseline') else 'baseline'}` with a "
                   f"tolerance of {gate.get('tolerance')} absolute. Passing means no gated number fell by more than that; it "
                   "does not mean the numbers are good.\n")
    else:
        out.append(f"**Gate: FAILED** against `{Path(gate['baseline']).name if gate.get('baseline') else 'baseline'}` with a "
                   f"tolerance of {gate.get('tolerance')} absolute. `eval_hardset.py` exits 1 on this.\n")
    checks = gate.get("checks") or []
    if checks:
        rows = [[c.get("name"), c.get("slice") or "-", *fmt_gate_pair(c), f3(c.get("delta")) if c.get("delta") is not None else "n/a",
                 c.get("n_gt", "-"), "ok" if c.get("ok") else "FAIL"] for c in checks]
        out.append(md_table(["check", "slice", "current", "baseline", "delta", "boxes", "status"], rows, numeric_from=2))
        out.append("")
    rows = []
    for key in ("day", "ir"):
        st = person_stats(hs, key)
        if st is None or not st["n_images"]:
            rows.append([key, 0, "n/a", "n/a", "n/a", "n/a"])
            continue
        rows.append([key, st["n_images"], f"{st['n_gt']}{' †' if 0 < st['n_gt'] < LOW_N else ''}", f3(st["ap50"]), f3(st["precision"]), f3(st["recall"])])
    out.append("Person metrics over the hard-set images only. The hard set is a fixed subset chosen for difficulty "
               "(DATASET_SPEC 2.6), so these numbers are not comparable with the full val numbers above:\n")
    out.append(md_table(["slice", "images", "person boxes", "person AP50", "P at op", "R at op"], rows))
    out.append("")
    return "\n".join(out)


def render_operating_points(inputs: Inputs) -> str:
    out = [section_header(8, "Operating points"),
           "Two different questions, two separate numbers, never merged: the F1-optimal point balances precision and recall; the "
           "false-alert-budget point is the highest-recall threshold that fits the false-alert target.\n"]
    ev, sw = inputs.eval, inputs.sweep
    if ev:
        rows = [[k, f"{ev['slices'][k]['operating_conf']:.3f}"] for k in ("day", "ir") if ev["slices"][k].get("n_images")]
        if rows:
            out.append("Confidence at which the evaluation reports precision and recall (F1 averaged over classes, per slice):\n")
            out.append(md_table(["slice", "operating conf"], rows))
            out.append("")
    if sw is None:
        out.append("No measurement yet: no sweep JSON was provided. Run `sweep_conf.py` (section 11).\n")
        return "\n".join(out)
    a = sw["assumptions"]
    out.append(f"Sweep `{sw.get('tag')}` on the {sw.get('split')} split, class {a.get('class')}, matching at IoU 0.50, measured on "
               f"{host_summary(sw.get('host'))}.\n")
    out.append("Budget arithmetic:\n")
    out.append(f"- {a['fps']:g} fps x 86,400 s = {a['frames_per_day']:,.0f} frames per camera per day")
    out.append(f"- {a['alerts_per_camera_per_day']:g} allowed false alerts per camera per day / {a['fp_to_alert']:g} (the fraction of "
               f"false detections that become an alert; 1.0 is the worst case, no fusion or tracking credit) = "
               f"{a['alerts_per_camera_per_day'] / a['fp_to_alert']:g} false detections per day")
    out.append(f"- budget = {a['budget_fp_per_frame']:.3e} false positives per frame")
    min_frames = a.get("rule_of_three_min_frames") or math.ceil(3.0 / a["budget_fp_per_frame"])
    out.append(f"- rule of three: zero false positives in n frames bounds the rate only to 3/n at 95%, so the budget cannot be "
               f"resolved with fewer than {min_frames:,} frames, and only if none of them produces a false positive\n")
    rows_f1, rows_budget = [], []
    for name in C.MODALITY_SLICES:
        s = sw["slices"].get(name)
        if not s or not s.get("n_frames"):
            rows_f1.append([name, 0, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            rows_budget.append([name, 0, "no frames", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        f1, b = s["f1_optimal"], s["false_alert_budget"]
        over = f"{f1['fp_per_frame'] / a['budget_fp_per_frame']:,.0f}x" if f1.get("fp_per_frame") else ("0" if f1.get("f1") is not None else "n/a")
        rows_f1.append([name, s["n_frames"], f3(f1.get("conf")), f3(f1.get("nms_iou")), f3(f1.get("f1")), f"{f3(f1.get('precision'))} / {f3(f1.get('recall'))}",
                        sci(f1.get("fp_per_frame")), over])
        status = b["status"] + (" (vacuous, recall 0)" if _budget_is_vacuous(b, s.get("n_gt", 0)) else "")
        needed = "none" if b["status"] == "met" else ("cannot help" if b["status"] == "unreachable" else f"{b['frames_needed_to_resolve']:,}")
        rows_budget.append([name, s["n_frames"], status, f3(b.get("conf")), f3(b.get("recall")),
                            f"{b['fp_count']}" if b.get("fp_count") is not None else "n/a",
                            f"{sci(b.get('fp_per_frame'))} / {sci(b.get('fp_per_frame_upper95'))}", needed,
                            f"{b['n_frames']:,}"])
    out.append("F1-optimal point:\n")
    out.append(md_table(["slice", "frames", "conf", "NMS IoU", "F1", "P / R", "FP per frame", "rate vs budget"], rows_f1))
    out.append("")
    out.append("False-alert-budget point (`met` needs the 95% UPPER bound within budget; the observed rate alone is not enough):\n")
    out.append(md_table(["slice", "frames", "status", "conf", "recall", "false positives", "FP per frame: observed / 95% upper",
                         "frames needed", "frames available"], rows_budget))
    out.append("")
    for name in C.MODALITY_SLICES:
        s = sw["slices"].get(name)
        if s and s.get("false_alert_budget", {}).get("note"):
            out.append(f"- {name}: {s['false_alert_budget']['note']}")
    out.append("")
    if sw.get("empty_frames"):
        e = sw["empty_frames"]
        out.append(f"Background frames: {e['n_frames']} all-negative frames were scored and added to the {e['modality']} slice only.\n")
    for n in sw.get("notes", []):
        out.append(f"- {n}")
    out.append("")
    return "\n".join(out)


def render_latency(inputs: Inputs) -> str:
    out = [section_header(9, "Latency by host"),
           f"Reference point: {JETSON_TARGET}. It is a target. No row below was measured on that board, no row can be "
           "compared to that target, and an FP32 CPU figure is not evidence about INT8 on a Jetson in either direction.\n"]
    if not inputs.benchmarks:
        out.append("No measurement yet: no benchmark JSON was provided. Run `benchmark_cpu.py` on the exported ONNX (section 11).\n")
        return "\n".join(out)
    rows = []
    for b in inputs.benchmarks:
        m, c, ms, pm = b["model"], b["config"], b["inference_ms"], b["pipeline_ms"]
        flag = " (fixture model)" if ("synthetic" in f"{b.get('label')} {m.get('weights_provenance')}".lower() or "smoke" in f"{b.get('label')}".lower()) else ""
        rows.append([f"{b.get('label')}{flag}", host_summary(b.get("host")),
                     f"{m.get('file')}, {m.get('precision')}, opset {m.get('opset')}, {(m.get('size_bytes') or 0) / 1e6:.1f} MB",
                     f"{c.get('imgsz')}, batch {c.get('batch')}, threads {c.get('threads')}, {c.get('provider')}, {c.get('runs')} runs",
                     f"{ms['p50']:.1f} / {ms['p95']:.1f}", f"{pm['p50']:.1f} / {pm['p95']:.1f}"])
    out.append(md_table(["label", "measured on host", "model", "config (imgsz)", "inference ms p50 / p95", "pipeline ms p50 / p95"], rows, numeric_from=4))
    out.append("")
    out.append("`inference` is onnxruntime `session.run` only. `pipeline` adds letterbox, decode and NMS in numpy: a detector-stage "
               "latency, still not end-to-end (no capture, tracking or fusion). Both are single-frame, batch 1.\n")
    caveats = []
    for b in inputs.benchmarks:
        if b.get("caveat") and b["caveat"] not in caveats:
            caveats.append(b["caveat"])
    for cav in caveats:
        out.append(f"> {cav}\n")
    return "\n".join(out)


def render_reasons(inputs: Inputs, verdicts: list[Verdict]) -> str:
    out = [section_header(10, "Top reasons for every target that is not MET"),
           "Computed from the inputs by a deterministic ranker (`make_metrics.py`, `rank_reasons`), not written by hand. For the "
           "mAP targets the ranking is by estimated AP50 points at stake; the estimates overlap (small people are part of the "
           "missed people), so they are not additive. A cause that alone stops a PARTIAL from becoming MET ranks first. For "
           "false alerts the order is blocking status, then severity. Fewer than three are shown when fewer exist.\n"]
    shown = False
    for v in verdicts:
        if v.verdict == M.MET:
            continue
        shown = True
        out.append(f"### {v.name}: {v.verdict}\n")
        reasons = reasons_for(v, inputs)[:TOP_REASONS]
        for i, r in enumerate(reasons, 1):
            note = f" ({r.note})" if r.note and v.key in ("day_map", "ir_map") else ""
            out.append(f"{i}. {r.text}{note}")
        out.append("")
    if not shown:
        out.append("Every target is MET.\n")
    return "\n".join(out)


def rel(path: str | None) -> str:
    return str(path) if path else ""


def reproduction_commands(inputs: Inputs) -> str:
    ev, sw = inputs.eval, inputs.sweep
    weights = (ev or {}).get("weights", {}).get("path") or (inputs.hardset or {}).get("weights", {}).get("path") or "training/runs/ir/weights/best.pt"
    tag = (ev or {}).get("tag") or "phase2"
    imgsz = (ev or {}).get("imgsz") or 640
    py = "training/.venv/bin/python"
    eval_flags = f'--weights "$WEIGHTS" --tag "$TAG" --split val --imgsz {imgsz}'
    if ev:
        eval_flags += (f" --nms-iou {ev.get('nms_iou', 0.7)} --report-conf {ev.get('report_conf', 0.35)}"
                       f" --bootstrap {ev.get('bootstrap', 200)} --size-basis {ev.get('size_basis', 'input')}")
    sweep_flags = '--preds "training/results/cache/preds_$TAG.npz" --tag "$TAG"'
    label = (inputs.benchmarks[0].get("label") if inputs.benchmarks else None) or "local-cpu"
    if sw:
        a = sw["assumptions"]
        sweep_flags += f" --fps {a['fps']:g} --alerts-per-camera-day {a['alerts_per_camera_per_day']:g} --fp-to-alert {a['fp_to_alert']:g}"
    lines = [
        "# Run from the repository root. The dataset is found through $TRUEWATCH_DATA_ROOT, a Kaggle input mount or",
        "# datasets/processed/yolo, in that order; pass --data-root to override.",
        f"WEIGHTS={weights}",
        f"TAG={tag}",
        "RUN_DIR=training/runs/ir   # the --run-dir given to train.py; train_log.csv spans every session",
        "",
        "# 0. Train, on a GPU (Kaggle). Two stages of one lineage; --auto-resume continues after a session ends.",
        f"{py} training/scripts/train.py --stage day --run-dir training/runs/day --auto-resume",
        f"{py} training/scripts/train.py --stage ir --run-dir training/runs/ir --auto-resume",
        "",
        "# 1. Per-class AP, precision and recall, day and IR separately, plus the object-height buckets.",
        f"{py} training/scripts/evaluate.py {eval_flags}",
        "",
        "# 2. Hard-set regression gate. The first run has no baseline; --write-baseline records one.",
        f'{py} training/scripts/eval_hardset.py --weights "$WEIGHTS" --tag "$TAG" --imgsz {imgsz} --baseline training/results/hardset_baseline.json',
        "",
        "# 3. Confidence and NMS sweep, reusing the raw predictions step 1 cached (no second inference).",
        f"{py} training/scripts/sweep_conf.py {sweep_flags}",
        "",
        "# 4. ONNX export with the torch versus onnxruntime parity check (exit non-zero above the tolerance).",
        f'{py} training/scripts/export_onnx.py --weights "$WEIGHTS" --imgsz {imgsz} --tag "$TAG"',
        "",
        "# 5. CPU latency of the exported graph on THIS host; run it again on each host you want a row for.",
        f'{py} training/scripts/benchmark_cpu.py --model "training/weights/$TAG.onnx" --imgsz {imgsz} --label "{label}" --out "training/results/benchmark_{label}.json"',
        "",
        "# 6. This report.",
        f'{py} training/scripts/make_metrics.py --eval "training/results/eval_$TAG.json" --hardset "training/results/hardset_$TAG.json" '
        f'--sweep "training/results/sweep_$TAG.json" --export "training/results/export_$TAG.json" '
        f'--benchmark "training/results/benchmark_{label}.json" --train-log "$RUN_DIR/train_log.csv" --out training/results/METRICS.md',
    ]
    return "\n".join(lines)


def render_reproduction(inputs: Inputs) -> str:
    out = [section_header(11, "Reproduction commands"),
           "Every command reads or writes files under `training/results/` or `training/weights/`; nothing is written into a tracked "
           "location, and no weights are committed.\n",
           "```bash", reproduction_commands(inputs), "```", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------------------------


def render_metrics(inputs: Inputs) -> str:
    synthetic = is_synthetic(inputs)
    verdicts = compute_verdicts(inputs)
    head = ["# TRUEWATCH Phase 2: detector metrics", "", f"> {HONESTY_LINE_1}  ", f"> {HONESTY_LINE_2}", ""]
    if synthetic:
        head += [SYNTHETIC_BANNER, ""]
    parts = [
        "\n".join(head),
        render_provenance(inputs),
        render_targets(inputs, verdicts, synthetic),
        render_slice_classes(inputs, "day", 3, "Day (visible camera)"),
        render_slice_classes(inputs, "ir", 4, "IR (LWIR replicated to 3 channels)"),
        render_small_objects(inputs),
        render_lighting(inputs),
        render_hardset(inputs),
        render_operating_points(inputs),
        render_latency(inputs),
        render_reasons(inputs, verdicts),
        render_reproduction(inputs),
    ]
    return "\n".join(parts).rstrip() + "\n"


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--eval", help="eval_<tag>.json from evaluate.py; without it every accuracy target is NOT ASSESSED")
    ap.add_argument("--hardset", help="hardset_<tag>.json from eval_hardset.py")
    ap.add_argument("--sweep", help="sweep_<tag>.json from sweep_conf.py")
    ap.add_argument("--benchmark", action="append", default=[], help="benchmark_<label>.json from benchmark_cpu.py; repeat once per host")
    ap.add_argument("--train-log", help="train_log.csv written by train.py")
    ap.add_argument("--export", help="export_<tag>.json from export_onnx.py")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "METRICS.md"), help="where the report is written")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inputs = load_inputs(args.eval, args.hardset, args.sweep, args.benchmark, args.train_log, args.export)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = render_metrics(inputs)
    write_text_atomic(Path(args.out), text)
    verdicts = compute_verdicts(inputs)
    print(f"wrote {args.out}")
    for v in verdicts:
        print(f"  {v.verdict:<13} {v.name}")
    if not inputs.eval:
        print("no evaluation JSON was provided: this is the 'no measurements yet' document")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
