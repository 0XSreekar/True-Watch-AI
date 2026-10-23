"""Render tiling_report.json (tiled_eval.py report) and the CPU cost json as markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NAMES = {"baseline": "full frame 640 (v2)", "g2x2": "2x2 grid + full", "g3x2": "3x2 grid + full",
         "n640": "640 px native tiles + full", "strip": "far-field strip + full (edge default layout)"}


def f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def row(name, r, cost):
    day, ir, dl = r["day"], r["ir"], r["day/daylight"]
    p = day["at_report_conf"]["classes"]["person"]
    pp = r.get("passes_per_frame", {"day": 1.0, "ir": 1.0})
    return (f"| {name} | {f(day['classes']['person']['ap50'])} | {f(dl['classes']['person']['ap50'])} | "
            f"{f(ir['classes']['person']['ap50'])} | {f(day['mean_over_present_classes']['ap50'])} | "
            f"{f(ir['mean_over_present_classes']['ap50'])} | {f(p['precision'])} | {f(p['recall'])} | "
            f"{f(pp['day'], 1)} / {f(pp['ir'], 1)} | {cost} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--cost", required=True)
    a = ap.parse_args()
    rep = json.loads(Path(a.report).read_text())
    cost = json.loads(Path(a.cost).read_text())["layouts"]
    cost_of = {"baseline": cost["full_frame"], **{k: v for k, v in cost.items() if k != "full_frame"}}
    out = []
    for scope, title in (("held_out_half", "Held-out half (report half, not used for tuning)"), ("full_val", "Full official val")):
        s = rep[scope]
        out += [f"### {title} ({s['n_images']} images)", "",
                "| variant | day person AP50 | daylight person AP50 | IR person AP50 | day mAP50 | IR mAP50 | day person precision @0.35 | day person recall @0.35 | passes/frame day / IR | CPU ms per 1080p frame |",
                "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |"]
        for k in ("baseline", "g2x2", "g3x2", "n640", "strip"):
            if k in s:
                c = cost_of[k]
                out.append(row(NAMES[k], s[k], f"{c['ms_median']:.0f} ({c['passes_per_1080p_frame']} passes)"))
        out.append("")
    # per source and size buckets on full val
    s = rep["full_val"]
    out += ["### Person AP50 by source, day slice (full val)", "",
            "| variant | " + " | ".join(sorted(s["baseline"]["day"]["by_source"])) + " | " + " | ".join(f"IR {x}" for x in sorted(s["baseline"]["ir"]["by_source"])) + " |",
            "| :-- |" + " --: |" * (len(s["baseline"]["day"]["by_source"]) + len(s["baseline"]["ir"]["by_source"]))]
    for k in ("baseline", "g2x2", "g3x2", "n640", "strip"):
        if k in s:
            d, i = s[k]["day"]["by_source"], s[k]["ir"]["by_source"]
            out.append(f"| {NAMES[k]} | " + " | ".join(f(d[x]["person_ap50"]) for x in sorted(d)) + " | " + " | ".join(f(i[x]["person_ap50"]) for x in sorted(i)) + " |")
    out += ["", "### Day person recall at 0.35 by height in detector-input px (full val)", ""]
    levels = [b["bucket_px"] for b in s["baseline"]["day"]["size_buckets"]["person"]]
    out += ["| variant | " + " | ".join(f"{l} px" for l in levels) + " |", "| :-- |" + " --: |" * len(levels)]
    for k in ("baseline", "g2x2", "g3x2", "n640", "strip"):
        if k in s:
            out.append(f"| {NAMES[k]} | " + " | ".join(f(b["recall"]) for b in s[k]["day"]["size_buckets"]["person"]) + " |")
    out += ["", "Chosen merge parameters (tuning half): "]
    for k in ("g2x2", "g3x2", "n640", "strip"):
        if k in s:
            out.append(f"- {NAMES[k]}: {s[k]['params']}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
