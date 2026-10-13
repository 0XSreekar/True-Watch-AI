#!/usr/bin/env python3
"""Re-run the docs/MEASUREMENTS.md section 2 rule-sanity table.

Section 2: YOLO11-s + ByteTrack over 100 consecutive KAIST visible frames; 10 tracks of 8+ frames; a
baseline direction learnt from the tracks themselves (6 with, 4 against); three rules: virtual fence at
x = 320, wrong direction against the learnt baseline, loitering (dwell >= 3 s and net < 90 px).

Two modes:

  --table   (default, no data needed) rebuilds ten sample tracks whose frames, dwell, net and path equal
            the committed table, runs them through the SAME signal functions the tracker uses
            (pipeline.tracking.dwell_seconds / net_displacement / path_length) and the three rules
            below, and checks that every row's metrics and "rule fired" come out exactly as published.
            This is a consistency check of the definitions, not a re-measurement: the table does not
            record positions, so each track's path is constructed to have the published net and path.
  --frames DIR
            the real re-run: the appearance channel (ONNX, via YOLO_MODEL_URL/SHA256 or the manifest)
            and pipeline.tracking.ByteTracker over DIR's images in order (e.g. a KAIST visible folder),
            printing the same table. The section 2 set was KAIST set00/V007; any other sequence gives
            its own numbers, and they are printed as observed, not compared.

The rule predicates here are the minimal ones section 2 describes. The rule engine proper (edge/rules/,
Phase 4) owns rules in production; this script exists so the deck's table stays reproducible.

    cd edge
    python tools/reproduce_rule_sanity.py
    python tools/reproduce_rule_sanity.py --frames ../var/datasets/kaist/kaist_train/set00/V000/visible --start 1150
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

EDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDGE_ROOT))

from pipeline.tracking import dwell_seconds, net_displacement, path_length  # noqa: E402

FPS = 20.0
FENCE_X = float(os.environ.get("FENCE_X", 320))
LOITER_DWELL_S = float(os.environ.get("LOITER_DWELL_SECONDS", 3))
LOITER_NET_PX = float(os.environ.get("LOITER_NET_PX", 90))
MIN_FRAMES = 8                   # section 2 evaluates tracks lasting 8+ frames
# Section 2 does not state a minimum displacement for "wrong direction"; its table implies one: track 5
# (net 99 px, against) fired, tracks 12/26/28 (net 3-34 px) did not although one of them is counted
# "against" in the 6/4 baseline. 50 px is the value this script uses; it separates the two groups.
WRONG_DIR_MIN_NET_PX = 50.0

# The committed table: id, frames, dwell s, net px, path px, rule fired, direction (ltr = with the baseline)
TABLE = [
    (1, 70, 3.5, 244, 279, "-", "ltr"),
    (2, 27, 1.4, 151, 158, "wrong direction", "rtl"),
    (3, 33, 1.6, 190, 202, "wrong direction", "rtl"),
    (4, 96, 4.8, 316, 367, "fence crossed", "ltr"),
    (5, 20, 1.0, 99, 109, "wrong direction", "rtl"),
    (6, 98, 5.0, 70, 163, "fence crossed + loitering", "ltr"),
    (9, 92, 4.7, 162, 213, "fence crossed", "ltr"),
    (12, 11, 1.2, 3, 10, "-", "rtl"),
    (26, 13, 0.7, 22, 27, "-", "ltr"),
    (28, 10, 0.5, 34, 35, "-", "ltr"),
]


@dataclass
class SampleTrack:
    track_id: int
    frames: list[int]
    centres: list[tuple[float, float]]


def direction(t: SampleTrack) -> str:
    return "ltr" if t.centres[-1][0] >= t.centres[0][0] else "rtl"


def crosses_fence(t: SampleTrack, x: float = FENCE_X) -> bool:
    return any((a[0] - x) * (b[0] - x) < 0 or (b[0] == x and a[0] != x) for a, b in zip(t.centres, t.centres[1:]))


def metrics(t: SampleTrack) -> dict:
    return {"frames": len(t.frames), "dwell_s": dwell_seconds(t.frames[0], t.frames[-1], 1 / FPS),
            "net_px": net_displacement(t.centres), "path_px": path_length(t.centres)}


def evaluate(tracks: list[SampleTrack]) -> tuple[list[tuple[SampleTrack, dict, list[str]]], dict]:
    tracks = [t for t in tracks if len(t.frames) >= MIN_FRAMES]
    dirs = [direction(t) for t in tracks]
    dominant = max(("ltr", "rtl"), key=dirs.count) if dirs else "ltr"
    baseline = {"dominant": dominant, "with": dirs.count(dominant), "against": len(dirs) - dirs.count(dominant)}
    rows = []
    for t in tracks:
        m = metrics(t)
        fired = []
        if crosses_fence(t):
            fired.append("fence crossed")
        if direction(t) != dominant and m["net_px"] >= WRONG_DIR_MIN_NET_PX:
            fired.append("wrong direction")
        if m["dwell_s"] >= LOITER_DWELL_S and m["net_px"] < LOITER_NET_PX:
            fired.append("loitering")
        rows.append((t, m, fired))
    return rows, baseline


def print_table(rows, baseline, title: str) -> None:
    print(title)
    print(f"baseline learnt from the tracks: majority {baseline['dominant']}, {baseline['with']} with, "
          f"{baseline['against']} against")
    print(f"rules: fence x={FENCE_X:g}; wrong direction (net >= {WRONG_DIR_MIN_NET_PX:g} px); "
          f"loitering dwell >= {LOITER_DWELL_S:g} s and net < {LOITER_NET_PX:g} px\n")
    print("| id | frames | dwell | net px | path px | rule fired |")
    print("|---|---|---|---|---|---|")
    for t, m, fired in rows:
        print(f"| {t.track_id} | {m['frames']} | {m['dwell_s']:.1f} s | {m['net_px']:.0f} | {m['path_px']:.0f} | "
              f"{' + '.join(fired) or '-'} |")
    alerts = sum(1 for _, _, f in rows if f)
    print(f"\n{len(rows)} tracks evaluated, {alerts} alerts, {len(rows) - alerts} normal.")


# ------------------------------------------------------------------------------------------ --table


def span_for(frames: int, dwell: float) -> int:
    """Smallest first-to-last frame distance (>= frames - 1) whose dwell rounds to the published value."""
    for span in range(frames - 1, frames + 200):
        if round(span / FPS, 1) == dwell:
            return span
    raise ValueError(f"no span gives {dwell} s for {frames} frames")


def build_sample(track_id, frames, dwell, net, path, fired, direction_) -> SampleTrack:
    span = span_for(frames, dwell)
    idx = sorted({round(i * span / (frames - 1)) for i in range(frames)}) if frames > 1 else [0]
    assert len(idx) == frames and idx[-1] == span
    sign = 1.0 if direction_ == "ltr" else -1.0
    overshoot = (path - net) / 2.0                          # go past the end point, then come back
    crosses = "fence" in fired
    x0 = (FENCE_X - net / 2.0) if crosses and sign > 0 else (FENCE_X + net / 2.0) if crosses else (
        FENCE_X - 40.0 - net - overshoot if sign > 0 else FENCE_X + 40.0 + net + overshoot)
    # polyline x0 -> turn -> x0 + sign*net; the turn is itself a sample, so the sampled path length is exact
    turn = x0 + sign * (net + overshoot)
    l0 = net + overshoot
    n = frames
    j = n - 1 if overshoot == 0 else min(n - 2, max(1, round((n - 1) * l0 / path)))
    centres = []
    for i in range(n):
        x = x0 + sign * l0 * i / j if i <= j else turn - sign * overshoot * (i - j) / (n - 1 - j)
        centres.append((x, 240.0))
    return SampleTrack(track_id, [1000 + i for i in idx], centres)


def run_table() -> int:
    samples = [build_sample(*row) for row in TABLE]
    rows, baseline = evaluate(samples)
    print_table(rows, baseline, "MEASUREMENTS.md section 2, re-evaluated on sample tracks rebuilt from the table")
    ok = baseline["with"] == 6 and baseline["against"] == 4
    for (t, m, fired), row in zip(rows, TABLE):
        _, frames, dwell, net, path, pub_fired, _ = row
        got = " + ".join(fired) or "-"
        same = (m["frames"] == frames and round(m["dwell_s"], 1) == dwell and round(m["net_px"]) == net
                and round(m["path_px"]) == path and got == pub_fired)
        ok &= same
        if not same:
            print(f"MISMATCH track {t.track_id}: got {m} {got!r}, table says {row}")
    alerts = sum(1 for _, _, f in rows if f)
    ok &= alerts == 6 and len(rows) == 10
    print("\nRESULT:", "every row, the 6/4 baseline and the 6 alerts / 4 normal match the published table" if ok
          else "MISMATCH against the published table")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------ --frames


def run_frames(folder: Path, start: int, count: int, conf: float) -> int:
    import cv2

    from config import load
    from models.detector_weights import load_at_boot
    from pipeline.appearance import AppearanceChannel, AppearanceConfig
    from pipeline.tracking import ByteTracker

    detector = load_at_boot(load())
    if detector is None:
        raise SystemExit("no detector configured: set YOLO_MODEL_URL and YOLO_MODEL_SHA256")
    ch = AppearanceChannel.from_detector(detector, AppearanceConfig(conf=0.1))
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".png"})[start:start + count]
    if not files:
        raise SystemExit(f"no frames in {folder} from index {start}")
    tracker = ByteTracker()
    for k, p in enumerate(files):
        img = cv2.imread(str(p))
        dets = [d for d in ch.detect(img, frame_index=k, timestamp=k / FPS) if d.class_name == "person" and d.score >= conf]
        tracker.update(dets, k, k / FPS)
    seen: dict[int, SampleTrack] = {}
    for t in list(tracker.finished) + tracker.all_tracks():     # tracks that ended early, then the live ones
        seen[t.track_id] = SampleTrack(t.track_id, [o[0] for o in t.observations], [(o[2], o[3]) for o in t.observations])
    rows, baseline = evaluate(sorted(seen.values(), key=lambda s: s.track_id))
    print_table(rows, baseline, f"{len(files)} consecutive frames from {folder} starting at index {start} "
                                f"(observed on this sequence; section 2 used KAIST set00/V007)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args(argv)
    if args.frames:
        return run_frames(args.frames, args.start, args.count, args.conf)
    return run_table()


if __name__ == "__main__":
    raise SystemExit(main())
