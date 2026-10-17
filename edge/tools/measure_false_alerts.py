#!/usr/bin/env python3
"""False alerts per camera per 24 h, measured on event-free recordings (slide 5 target).

Give it one or more recordings in which NOTHING should raise an alert (no person, vehicle or cart that
an operator would want to see: an empty border stretch at different hours, wind, rain, dusk, night). It
runs each through the service's own path (ingest at TARGET_FPS -> CameraService: calibration, fusion,
rules), counts every truewatch.event.v1 the service would emit, and reports

    alerts per camera per 24 h = alerts / alertable camera-hours x 24

with the extrapolation stated: hours of footage, how many cameras they came from, and how many of those
hours could alert at all (the calibration warm-up of each recording cannot). A 95% interval is given
with it (exact Poisson on the alert count), because 0 alerts in 2 h is not "0 per day".

Every alert counted here is, by the footage's premise, false. The tool cannot check that premise: whoever
records the footage must confirm it is event-free (or cut the event out).

    cd edge
    python -m tools.measure_false_alerts CAM-A=/data/cam_a_night.mp4 CAM-A=/data/cam_a_day.mp4 \\
        CAM-B=/data/cam_b.mp4 --target-fps 8 --rules-dir var/rules --json out/false_alerts.json
    python -m tools.measure_false_alerts --self-check        # proves the harness on generated footage

A recording given without CAMERA= is its own camera. The detector is the one the service loads
(YOLO_MODEL_URL/SHA256 or the committed manifest).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from tools.harness import quiet_reason_logs, RunResult, poisson_interval, real_appearance, render_quiet, run_recording


def summarise(runs: list[RunResult], target_fps: float) -> dict:
    cameras = sorted({r.camera_id for r in runs})
    footage_s = sum(r.media_s for r in runs)
    alertable_s = sum(r.alertable_s for r in runs)
    alerts = sum(len(r.events) for r in runs)
    by_kind: dict[str, int] = {}
    for r in runs:
        for e in r.events:
            by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
    cam_hours = alertable_s / 3600.0
    lo, hi = poisson_interval(alerts)
    per_day = (lambda k: k / cam_hours * 24.0) if cam_hours > 0 else (lambda k: float("nan"))
    return {
        "metric": "false alerts per camera per 24 h",
        "alerts": alerts,
        "alerts_by_kind": by_kind,
        "cameras": len(cameras),
        "camera_ids": cameras,
        "recordings": len(runs),
        "footage_hours": round(footage_s / 3600.0, 4),
        "alertable_camera_hours": round(cam_hours, 4),
        "alerts_per_camera_per_24h": round(per_day(alerts), 3),
        "ci95_per_camera_per_24h": [round(per_day(lo), 3), round(per_day(hi), 3)],
        "target_fps": target_fps,
        "extrapolation": (f"{alerts} alert(s) in {cam_hours:.3f} alertable camera-hours "
                          f"({footage_s / 3600.0:.3f} h of footage from {len(cameras)} camera(s), calibration "
                          f"warm-up excluded), scaled linearly to 24 h per camera; assumes the footage is "
                          f"event-free and representative of the camera's day"),
        "per_recording": [{"recording": r.recording, "camera_id": r.camera_id, "frames": r.frames,
                           "media_s": r.media_s, "warmup_s": r.warmup_s, "alertable_s": round(r.alertable_s, 3),
                           "alerts": len(r.events), "fused_events": r.fused_events, "events": r.events}
                          for r in runs],
    }


def table(s: dict) -> str:
    lines = [f"{'recording':<40} {'camera':<12} {'media s':>8} {'alertable s':>11} {'alerts':>6}"]
    for r in s["per_recording"]:
        lines.append(f"{Path(r['recording']).name[:40]:<40} {r['camera_id'][:12]:<12} {r['media_s']:>8.1f} "
                     f"{r['alertable_s']:>11.1f} {r['alerts']:>6d}")
    lo, hi = s["ci95_per_camera_per_24h"]
    lines += ["", f"alerts {s['alerts']} {s['alerts_by_kind']}",
              f"rate   {s['alerts_per_camera_per_24h']} per camera per 24 h (95% CI {lo} to {hi})",
              f"basis  {s['extrapolation']}"]
    return "\n".join(lines)


def parse_recordings(specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for spec in specs:
        cam, sep, path = spec.partition("=")
        if not sep or "/" in cam or "\\" in cam:
            cam, path = "CAM-" + Path(spec).stem.upper()[:24], spec
        out.append((cam, Path(path)))
    return out


def self_check(target_fps: float, workdir: Path, seconds: float = 60.0) -> dict:
    """Generated footage: two event-free recordings (expect 0 alerts) and one with 3 scripted walks
    (expect 3 detection alerts): the harness counts what is there and nothing else."""
    quiet = [render_quiet(workdir / f"quiet_{i}.mp4", seconds, seed=11 + i) for i in range(2)]
    control = render_quiet(workdir / "control.mp4", seconds, intrusions=3, seed=31)
    quiet_runs = [run_recording(s.path, s.appearance(), camera_id=f"SYN-{i}", target_fps=target_fps)
                  for i, s in enumerate(quiet)]
    control_run = run_recording(control.path, control.appearance(), camera_id="SYN-CONTROL", target_fps=target_fps)
    return {"event_free": summarise(quiet_runs, target_fps), "positive_control": summarise([control_run], target_fps),
            "expected": {"event_free_alerts": 0, "positive_control_alerts": len(control.crossings)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="*", help="[CAMERA_ID=]path of an event-free recording")
    ap.add_argument("--target-fps", type=float, default=None, help="default: TARGET_FPS of the edge config")
    ap.add_argument("--stride", type=int, default=None, help="detector stride; default: DETECTOR_STRIDE")
    ap.add_argument("--rules-dir", type=Path, default=None,
                    help="per-camera rule files <camera_id>.json, so rule alerts are counted too")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop each recording after this media time")
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    ap.add_argument("--self-check", action="store_true", help="run on generated footage instead")
    args = ap.parse_args(argv)
    quiet_reason_logs()

    from config import load

    cfg = load()
    fps = args.target_fps or cfg.target_fps
    stride = args.stride or cfg.detector_stride
    if args.self_check:
        with tempfile.TemporaryDirectory() as d:
            report = self_check(fps, Path(d))
        for name in ("event_free", "positive_control"):
            print(f"== {name}\n{table(report[name])}\n")
        print("expected", report["expected"])
        ok = (report["event_free"]["alerts"] == 0
              and report["positive_control"]["alerts"] == report["expected"]["positive_control_alerts"])
    else:
        if not args.recordings:
            ap.error("give at least one recording, or --self-check")
        from rules.config import ConfigStore

        store = ConfigStore(directory=args.rules_dir) if args.rules_dir else None
        appearance = real_appearance()
        runs = []
        for cam, path in parse_recordings(args.recordings):
            rules = None
            if store is not None:
                try:
                    rules = store.get(cam)
                except Exception as exc:  # noqa: BLE001 - a camera without a rule file runs without rules
                    print(f"{cam}: no rules ({exc})", file=sys.stderr)
            runs.append(run_recording(path, appearance, camera_id=cam, target_fps=fps, stride=stride, rules=rules,
                                      max_seconds=args.max_seconds))
        report = summarise(runs, fps)
        print(table(report))
        ok = True
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
