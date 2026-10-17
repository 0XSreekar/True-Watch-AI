#!/usr/bin/env python3
"""Fence-crossing miss rate against ground truth (slide 5 target: under 2% of crossings missed).

Input: a recording, the media times (seconds into the recording) at which someone actually crossed the
fence, and the camera's rule config holding that fence (rules/config.py format, e.g.
rules/examples/RXL-01.json). The recording runs through the service's own path (ingest at TARGET_FPS ->
CameraService -> rules engine); every 'fence crossed' firing is taken with the crossing time the rule
reports (rules/fence.py crossed_at_s). Ground-truth crossings are matched one-to-one to firings within
+/- `--tolerance` seconds (closest first). Unmatched ground truth = missed; unmatched firings = fence
alerts with no real crossing (reported, not part of the miss rate).

    miss rate = missed / n, with a Wilson 95% interval, compared with the 2% target.

The target is only demonstrated when the interval's UPPER bound is below 2%; with no miss at all that
needs n >= 189 crossings (Wilson upper bound at 0/189 is 1.99%), so one recording of a few crossings
cannot show it however well it goes.

Ground truth file: JSON, either a list of seconds [12.4, 31.0, ...] or a list of objects
[{"t": 12.4, "direction": "in"}, ...]. Record each crossing as a separate pass (the rule fires once
per track per fence).

    cd edge
    python -m tools.measure_fence_misses --recording /data/fence_walks.mp4 --truth /data/fence_walks.json \\
        --rules rules/examples/RXL-01.json --target-fps 8 --json out/fence.json
    python -m tools.measure_fence_misses --self-check 50     # generated footage, 50 scripted crossings
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from tools.harness import quiet_reason_logs, RunResult, real_appearance, render_crossings, run_recording, wilson_interval

TARGET_MISS_RATE = 0.02


def load_truth(path: Path) -> list[float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("crossings", [])
    return sorted(float(x["t"] if isinstance(x, dict) else x) for x in raw)


def match(truth: list[float], detected: list[float], tolerance: float) -> tuple[list[tuple[float, float]], list[float], list[float]]:
    """Greedy one-to-one matching by smallest time difference. -> (pairs, missed truth, unmatched detections)."""
    cands = sorted((abs(d - t), i, j) for i, t in enumerate(truth) for j, d in enumerate(detected)
                   if abs(d - t) <= tolerance)
    used_t, used_d, pairs = set(), set(), []
    for _, i, j in cands:
        if i in used_t or j in used_d:
            continue
        used_t.add(i); used_d.add(j)
        pairs.append((truth[i], detected[j]))
    missed = [t for i, t in enumerate(truth) if i not in used_t]
    extra = [d for j, d in enumerate(detected) if j not in used_d]
    return sorted(pairs), missed, extra


def summarise(run: RunResult, truth: list[float], tolerance: float, target_fps: float) -> dict:
    # a crossing during the calibration warm-up could not have been alerted on; it is reported, not scored
    warm = run.warmup_s if run.warmup_s is not None else float("inf")
    scored = [t for t in truth if t >= warm]
    during_warmup = [t for t in truth if t < warm]
    detected = [c["t"] for c in run.fence_crossings]
    pairs, missed, extra = match(scored, detected, tolerance)
    n, k = len(scored), len(missed)
    lo, hi = wilson_interval(k, n)
    rate = k / n if n else float("nan")
    if n == 0:
        verdict = "no scored crossings"
    elif hi < TARGET_MISS_RATE:
        verdict = "meets the <2% target (95% upper bound below 2%)"
    elif lo >= TARGET_MISS_RATE:
        verdict = "fails the <2% target (95% lower bound at or above 2%)"
    else:
        verdict = f"not demonstrated: the 95% interval straddles 2%; more crossings needed (n={n})"
    errs = [d - t for t, d in pairs]
    return {
        "metric": "fence-crossing miss rate",
        "recording": run.recording,
        "n": n,
        "detected": n - k,
        "missed": k,
        "miss_rate": round(rate, 4) if n else None,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "target": TARGET_MISS_RATE,
        "verdict": verdict,
        "tolerance_s": tolerance,
        "target_fps": target_fps,
        "excluded_during_warmup": during_warmup,
        "warmup_s": run.warmup_s,
        "unmatched_fence_alerts": extra,
        "timing_error_s": {"mean": round(sum(errs) / len(errs), 3), "max_abs": round(max(abs(e) for e in errs), 3)}
        if errs else None,
        "missed_at": missed,
        "frames": run.frames,
        "media_s": run.media_s,
    }


def table(s: dict) -> str:
    lo, hi = s["wilson95"]
    rate = "n/a" if s["miss_rate"] is None else f"{100 * s['miss_rate']:.2f}%"
    return "\n".join([
        f"recording   {Path(s['recording']).name}  ({s['media_s']:.1f} s, {s['frames']} frames at {s['target_fps']} fps)",
        f"crossings   n={s['n']}  detected={s['detected']}  missed={s['missed']}  (tolerance +/-{s['tolerance_s']} s;"
        f" {len(s['excluded_during_warmup'])} during warm-up not scored)",
        f"miss rate   {rate}  Wilson 95% [{100 * lo:.2f}%, {100 * hi:.2f}%]  target < {100 * s['target']:.0f}%",
        f"verdict     {s['verdict']}",
        f"extra       {len(s['unmatched_fence_alerts'])} fence alert(s) with no ground-truth crossing",
    ])


def self_check(n: int, target_fps: float, workdir: Path, blind: int = 2) -> dict:
    """`n` scripted crossings, of which `blind` the scripted detector never reports (must count as missed)."""
    blind_idx = set(range(3, 3 + 2 * blind, 2)) if blind else set()
    script = render_crossings(workdir / "crossings.mp4", n, blind=blind_idx)
    run = run_recording(script.path, script.appearance(), camera_id="SYN-FENCE", target_fps=target_fps,
                        rules=_rules(script.fence))
    report = summarise(run, [c["t"] for c in script.crossings], 1.0, target_fps)
    report["expected"] = {"n": n, "missed": len(blind_idx)}
    return report


def _rules(raw: dict):
    from rules.config import CameraRuleConfig

    return CameraRuleConfig.from_dict(raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", type=Path)
    ap.add_argument("--truth", type=Path, help="JSON list of ground-truth crossing times (media seconds)")
    ap.add_argument("--rules", type=Path, help="the camera's rule config JSON holding the fence")
    ap.add_argument("--tolerance", type=float, default=1.0, help="seconds either side of a true crossing")
    ap.add_argument("--target-fps", type=float, default=None, help="default: TARGET_FPS of the edge config")
    ap.add_argument("--stride", type=int, default=None, help="detector stride; default: DETECTOR_STRIDE")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--self-check", type=int, metavar="N", default=None,
                    help="generate N scripted crossings (2 of them invisible to the detector) and measure them")
    args = ap.parse_args(argv)
    quiet_reason_logs()

    from config import load

    cfg = load()
    fps = args.target_fps or cfg.target_fps
    if args.self_check:
        with tempfile.TemporaryDirectory() as d:
            report = self_check(args.self_check, fps, Path(d))
        ok = report["n"] == report["expected"]["n"] and report["missed"] == report["expected"]["missed"] \
            and not report["unmatched_fence_alerts"]
        print(table(report))
        print("expected   ", report["expected"])
    else:
        if not (args.recording and args.truth and args.rules):
            ap.error("--recording, --truth and --rules are required (or --self-check N)")
        from rules.config import load_file

        rules = load_file(args.rules)
        run = run_recording(args.recording, real_appearance(), camera_id=rules.camera_id, target_fps=fps,
                            stride=args.stride or cfg.detector_stride, rules=rules)
        report = summarise(run, load_truth(args.truth), args.tolerance, fps)
        print(table(report))
        ok = True
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
