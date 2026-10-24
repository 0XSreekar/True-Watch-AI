#!/usr/bin/env python3
"""Slide-5 field measurements on MEVA footage: fence misses and false alerts against KPF ground truth.

    cd edge
    python -m tools.meva_measure gt      --root DATA            # ground-truth crossings per camera (no pipeline)
    python -m tools.meva_measure run     --root DATA            # every manifest clip through the real service path
    python -m tools.meva_measure report  --root DATA --json out  # match, score, write the results JSON

The fence of each camera is one polyline in the pixels of the full frame (tools/meva_fences.json), chosen from
the annotation density and a first frame BEFORE any pipeline result was looked at. Ground-truth crossings use
the fence rule's own segment-intersection test (rules.fence.evaluate) on each annotated person or vehicle
track's ground point (bottom-centre of its box), sampled at the annotation's native 30 fps.

Matching (fence): a firing matches a ground-truth crossing when their times differ by at most --tolerance seconds
(default 2.0) and the firing's crossing point lies within --radius pixels (default 250) of the true one; matching
is one-to-one, closest in time first. Unmatched truth = missed; unmatched firings = extra.

False alerts: an alert (any truewatch.event.v1 the service emitted) is SUPPORTED when its box overlaps
(IoU >= 0.1) an annotated person or vehicle box within +/-1 s. Alerts with no such box are UNSUPPORTED: false if
the scene was really empty, but also what an unannotated passer-by produces (MEVA is activity-centric), which is
why --sheets lets the unsupported ones be checked by eye. The event-free measurement counts only time more than
--margin seconds away from any annotated person or vehicle box.

The pixel fence, the pixel thresholds and the absence of any homography are the service's fallback mode; MEVA
cameras are not calibrated for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from tools.harness import RunResult, poisson_interval, quiet_reason_logs, real_appearance, run_recording, wilson_interval
from tools.measure_fence_misses import TARGET_MISS_RATE
from tools.meva_data import FPS, ground_point, load_clip

FENCES = Path(__file__).with_name("meva_fences.json")
CLASSES = ("person", "vehicle")


@dataclass
class _Track:
    track_id: int
    history: list


def fence_lines(cam: str):
    from rules.fence import FenceLine

    spec = json.loads(FENCES.read_text())["cameras"][cam]
    return [FenceLine(name=f"meva-{cam}", points=[tuple(p) for p in spec["points"]], direction="both")]


def gt_crossings(clip: dict, cam: str) -> list[dict]:
    """One entry per annotated person/vehicle track whose ground point crosses the camera's fence."""
    from rules.fence import evaluate_all

    fences = fence_lines(cam)
    out = []
    for tid, tr in clip["tracks"].items():
        if tr["cls"] not in CLASSES or len(tr["boxes"]) < 2:
            continue
        hist = [(f / FPS, *ground_point(b)) for f, b in tr["boxes"]]
        for c in evaluate_all(_Track(tid, hist), fences):
            out.append({"t": round(c.crossed_at_s, 3), "track": tid, "cls": tr["cls"], "point": c.point_px,
                        "direction": c.direction})
    return sorted(out, key=lambda c: c["t"])


def manifest(root: Path) -> list[dict]:
    return json.loads((root / "manifest.json").read_text())["clips"]


RESULTS: Path | None = None       # --results: keep one run's per-clip outputs apart from another's


def results_dir(root: Path) -> Path:
    d = RESULTS or root / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_gt(root: Path) -> None:
    for c in manifest(root):
        clip = load_clip(root, c["stem"])
        g = gt_crossings(clip, c["camera"])
        print(c["camera"], c["stem"][:19], "tracks", len(clip["tracks"]), "GT crossings",
              sum(x["cls"] == "person" for x in g), "person", sum(x["cls"] == "vehicle" for x in g), "vehicle")


def cmd_run(root: Path, target_fps: float | None, stride: int | None) -> None:
    import cv2
    from config import load
    from rules.config import CameraRuleConfig

    cfg = load()
    fps = target_fps or cfg.target_fps
    stride = stride or cfg.detector_stride
    app = real_appearance()
    for c in manifest(root):
        out = results_dir(root) / f"{c['stem']}.json"
        if out.exists():
            continue
        path = root / "video" / f"{c['stem']}.r13.avi"
        cap = cv2.VideoCapture(str(path))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        cam = c["camera"]
        spec = json.loads(FENCES.read_text())["cameras"][cam]
        rules = CameraRuleConfig.from_dict({"camera_id": cam, "fences": [
            {"name": f"meva-{cam}", "points": spec["points"], "direction": "both"}]})
        run = run_recording(path, app, camera_id=cam, target_fps=fps, stride=stride, rules=rules)
        out.write_text(json.dumps({"stem": c["stem"], "camera": cam, "size": [w, h], "target_fps": fps, "stride": stride,
                                   "frames": run.frames, "media_s": run.media_s, "warmup_s": run.warmup_s,
                                   "events": run.events, "fence_crossings": run.fence_crossings,
                                   "fused_events": run.fused_events, "rule_fires": run.rule_fires,
                                   "rule_suppressed": run.rule_suppressed}))
        print(c["stem"], run.frames, "frames", len(run.events), "alerts", len(run.fence_crossings), "fence firings", flush=True)


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _boxes_near(clip: dict, t: float, win: float) -> list[tuple[str, tuple]]:
    f0, f1 = (t - win) * FPS, (t + win) * FPS
    out = []
    for tr in clip["tracks"].values():
        if tr["cls"] not in CLASSES:
            continue
        out += [(tr["cls"], b) for f, b in tr["boxes"] if f0 <= f <= f1]
    return out


def empty_windows(clip: dict, media_s: float, margin: float) -> list[tuple[float, float]]:
    """Intervals of the clip with no annotated person/vehicle box within `margin` seconds."""
    busy = sorted({int(f // FPS) for tr in clip["tracks"].values() if tr["cls"] in CLASSES for f, _ in tr["boxes"]})
    n = int(media_s) + 1
    busy_set = set()
    for s in busy:
        busy_set.update(range(int(s - margin), int(s + margin) + 1))
    wins, start = [], None
    for s in range(n + 1):
        free = s < n and s not in busy_set
        if free and start is None:
            start = s
        if not free and start is not None:
            wins.append((float(start), float(min(s, media_s))))
            start = None
    return [w for w in wins if w[1] - w[0] >= 1.0]


def cmd_report(root: Path, tolerance: float, radius: float, margin: float, out_json: Path | None) -> dict:
    from tools.measure_false_alerts import summarise as fa_summarise

    per_clip, all_events, empty_runs = [], [], []
    fence = {"person": [0, 0], "vehicle": [0, 0]}          # [detected, missed]
    cams: dict[str, dict] = {}
    extra_total = 0
    for c in manifest(root):
        r = json.loads((results_dir(root) / f"{c['stem']}.json").read_text())
        clip = load_clip(root, c["stem"])
        cam, w, h = c["camera"], *r["size"]
        warm = r["warmup_s"] if r["warmup_s"] is not None else float("inf")
        gt = [g for g in gt_crossings(clip, cam)]
        scored = [g for g in gt if g["t"] >= warm]
        det = r["fence_crossings"]
        # match by time, then require the crossing point to be near the true one
        cands = sorted((abs(d["t"] - g["t"]), i, j) for i, g in enumerate(scored) for j, d in enumerate(det)
                       if abs(d["t"] - g["t"]) <= tolerance and d.get("point_px")
                       and ((d["point_px"][0] - g["point"][0]) ** 2 + (d["point_px"][1] - g["point"][1]) ** 2) ** .5 <= radius)
        used_g, used_d, hit = set(), set(), {}
        for _, i, j in cands:
            if i in used_g or j in used_d:
                continue
            used_g.add(i); used_d.add(j); hit[i] = det[j]
        cs = cams.setdefault(cam, {"person": [0, 0], "vehicle": [0, 0], "extra": 0, "clips": 0})
        cs["clips"] += 1
        for i, g in enumerate(scored):
            k = 0 if i in hit else 1
            fence[g["cls"]][k] += 1
            cs[g["cls"]][k] += 1
        extra = len(det) - len(used_d)
        extra_total += extra; cs["extra"] += extra
        # alerts against ground truth
        sup = 0
        alerts = []
        for e in r["events"]:
            bx = e.get("bbox_norm_xywh")
            near = _boxes_near(clip, e["t"], 1.0)
            ok = False
            if bx:
                box = (bx[0] * w, bx[1] * h, (bx[0] + bx[2]) * w, (bx[1] + bx[3]) * h)
                ok = any(_iou(box, b) >= 0.1 for _, b in near)
            sup += ok
            alerts.append({**e, "supported": ok})
        wins = empty_windows(clip, r["media_s"], margin)
        ev_empty = [e for e in alerts if any(a <= e["t"] < b for a, b in wins)]
        rr = RunResult(recording=c["stem"], camera_id=cam)
        rr.media_s = sum(b - a for a, b in wins)
        rr.warmup_s = 0.0
        rr.events = [{k: v for k, v in e.items() if k != "supported"} for e in ev_empty if e["t"] >= warm]
        rr.media_s = sum(max(0.0, b - max(a, warm)) for a, b in wins)
        empty_runs.append(rr)
        per_clip.append({"stem": c["stem"], "camera": cam, "media_s": r["media_s"], "warmup_s": r["warmup_s"],
                         "rule_fires": r.get("rule_fires"), "rule_suppressed": r.get("rule_suppressed"),
                         "gt_person": sum(g["cls"] == "person" for g in scored), "gt_vehicle": sum(g["cls"] == "vehicle" for g in scored),
                         "fence_firings": len(det), "extra_firings": extra, "alerts": len(alerts), "alerts_supported": sup,
                         "alerts_unsupported": len(alerts) - sup, "empty_window_s": round(rr.media_s, 1),
                         "alerts_in_empty": len(rr.events)})
        all_events += [{"stem": c["stem"], "camera": cam, **e} for e in alerts]

    def block(d: list[int]) -> dict:
        k, n = d[1], d[0] + d[1]
        lo, hi = wilson_interval(k, n)
        verdict = ("no crossings" if not n else "meets <2% (upper bound below 2%)" if hi < TARGET_MISS_RATE else
                   "fails <2% (lower bound at/above 2%)" if lo >= TARGET_MISS_RATE else "not demonstrated (interval straddles 2%)")
        return {"n": n, "detected": d[0], "missed": k, "miss_rate": round(k / n, 4) if n else None,
                "wilson95": [round(lo, 4), round(hi, 4)], "verdict": verdict}

    fence_out = {"tolerance_s": tolerance, "radius_px": radius,
                 "person": block(fence["person"]), "vehicle": block(fence["vehicle"]),
                 "all": block([fence["person"][0] + fence["vehicle"][0], fence["person"][1] + fence["vehicle"][1]]),
                 "extra_firings": extra_total,
                 "per_camera": {cam: {"person": block(v["person"]), "vehicle": block(v["vehicle"]), "extra_firings": v["extra"],
                                      "clips": v["clips"]} for cam, v in sorted(cams.items())}}
    footage_s = sum(p["media_s"] for p in per_clip)
    n_alert = len(all_events)
    n_unsup = sum(not e["supported"] for e in all_events)
    cam_hours = footage_s / 3600.0
    lo, hi = poisson_interval(n_unsup)
    day = lambda k: round(k / cam_hours * 24.0, 2)
    ef = fa_summarise(empty_runs, 8.0)
    policy = alert_policy_block(per_clip, all_events, cam_hours)
    out = {"dataset": "MEVA (Kitware Inc. and IARPA), CC BY 4.0", "clips": len(per_clip), "footage_hours": round(footage_s / 3600.0, 3),
           "cameras": len(cams), "fence": fence_out,
           "all_footage_alerts": {"alerts": n_alert, "supported_by_annotation": n_alert - n_unsup, "unsupported": n_unsup,
                                  "unsupported_per_camera_per_24h_pooled": day(n_unsup),
                                  "unsupported_ci95": [day(lo), day(hi)], "all_alerts_per_camera_per_24h_pooled": day(n_alert),
                                  "note": "pooled over clips: each clip counted as one camera-slot of its own length"},
           "alert_policy": policy,
           "annotated_empty_footage": {k: ef[k] for k in ("alerts", "alerts_by_kind", "cameras", "footage_hours", "alertable_camera_hours",
                                                           "alerts_per_camera_per_24h", "ci95_per_camera_per_24h")},
           "margin_s": margin, "per_clip": per_clip}
    (results_dir(root) / "all_events.json").write_text(json.dumps(all_events))
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(out, indent=1))
    return out


def alert_policy_block(per_clip: list[dict], events: list[dict], cam_hours: float) -> dict:
    """Operator alerts under ALERT_POLICY=rules (rule firings that cleared the throttle) vs the legacy 'all'
    (every emitted event, observations included), per camera per 24 h, plus the rule firings before the
    throttle (the load an operator would see without the 5-per-camera-per-day budget)."""
    day = lambda k: round(k / cam_hours * 24.0, 1) if cam_hours > 0 else None
    by_cam: dict[str, dict] = {}
    for p in per_clip:
        b = by_cam.setdefault(p["camera"], {"media_s": 0.0, "rules": 0, "all": 0, "firings": 0})
        b["media_s"] += p["media_s"]
        b["firings"] += sum((p.get("rule_fires") or {}).values())
    for e in events:
        b = by_cam[e["camera"]]
        b["all"] += 1
        b["rules"] += e.get("kind") != "observation"
    per_cam = {cam: {k: (round(b[k] / b["media_s"] * 86400.0, 1) if b["media_s"] else None) for k in ("rules", "all", "firings")}
               | {"alerts_rules": b["rules"], "events_all": b["all"], "rule_firings": b["firings"]}
               for cam, b in sorted(by_cam.items())}
    n_rules = sum(b["rules"] for b in by_cam.values())
    n_fire = sum(b["firings"] for b in by_cam.values())
    by_rule: dict[str, int] = {}
    for p in per_clip:
        for k, v in (p.get("rule_fires") or {}).items():
            by_rule[k] = by_rule.get(k, 0) + v
    return {"alerts_rules": n_rules, "events_all": len(events), "rule_firings_before_throttle": n_fire,
            "rule_firings_by_rule": by_rule,
            "per_camera_per_24h_pooled": {"rules": day(n_rules), "all": day(len(events)), "firings": day(n_fire)},
            "per_camera": per_cam,
            "note": ("each 5-minute clip is a fresh service, so the throttle (5 new alerts per camera per 24 h, 12 per "
                     "shift) restarts per clip; delivered 'rules' alerts are therefore capped per clip and the "
                     "pre-throttle firings are the honest load figure")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gt", "run", "report"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--radius", type=float, default=250.0)
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--target-fps", type=float, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=None, help="per-clip output directory (default ROOT/results)")
    a = ap.parse_args(argv)
    global RESULTS
    RESULTS = a.results
    quiet_reason_logs()
    if a.cmd == "gt":
        cmd_gt(a.root)
    elif a.cmd == "run":
        cmd_run(a.root, a.target_fps, a.stride)
    else:
        print(json.dumps(cmd_report(a.root, a.tolerance, a.radius, a.margin, a.json), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
