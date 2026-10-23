#!/usr/bin/env python3
"""Contact sheets for eyeballing MEVA ground truth and pipeline output (images stay outside the repository).

    python -m tools.meva_sheets traj   --root DATA --camera G336 --out DIR    # GT ground-point paths on a first frame
    python -m tools.meva_sheets gt     --root DATA --stem CLIP --out DIR      # frames with GT boxes (annotation completeness)
    python -m tools.meva_sheets alerts --root DATA --stem CLIP --out DIR      # frames at each pipeline alert / fence firing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tools.meva_data import FPS, ground_point, load_clip
from tools.meva_measure import FENCES, manifest

COL = {"person": (0, 255, 0), "vehicle": (255, 160, 0), "other": (160, 160, 160)}


def _frame(path: Path, t: float):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def _fence(img, cam):
    pts = json.loads(FENCES.read_text())["cameras"].get(cam) if FENCES.exists() else None
    if pts:
        cv2.polylines(img, [np.array(pts["points"], np.int32)], False, (0, 0, 255), 4)


def _draw_gt(img, clip, t, win=0.04):
    for tid, tr in clip["tracks"].items():
        for f, b in tr["boxes"]:
            if abs(f / FPS - t) <= win:
                c = COL.get(tr["cls"], COL["other"])
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 3)
                cv2.putText(img, f"{tr['cls'][0]}{tid}", (int(b[0]), int(b[1]) - 4), 0, 1.0, c, 2)
                break


def _sheet(tiles, cols, out):
    while len(tiles) % cols:
        tiles.append(tiles[0] * 0)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def _small(img, label, w=640):
    h = int(img.shape[0] * w / img.shape[1])
    s = cv2.resize(img, (w, h))
    cv2.putText(s, label, (6, 22), 0, 0.65, (0, 255, 255), 2)
    return s


def traj(root, cam, out):
    sel = json.loads((root / "selection.json").read_text())
    clips = [{"stem": x} for x in sel if x.endswith("." + cam)]
    have = [c for c in clips if (root / "video" / f"{c['stem']}.r13.avi").exists()]
    base = _frame(root / "video" / f"{have[0]['stem']}.r13.avi", 2.0)
    for k, c in enumerate(clips):
        for tr in load_clip(root, c["stem"])["tracks"].values():
            if tr["cls"] in ("person", "vehicle") and len(tr["boxes"]) > 5:
                pts = np.array([ground_point(b) for _, b in tr["boxes"]], np.int32)
                cv2.polylines(base, [pts], False, [(0, 255, 0), (255, 160, 0)][tr["cls"] == "vehicle"], 2)
    _fence(base, cam)
    cv2.imwrite(str(out / f"traj_{cam}.jpg"), cv2.resize(base, (1280, int(base.shape[0] * 1280 / base.shape[1]))))


def gt(root, stem, out, n=6):
    clip, tiles = load_clip(root, stem), []
    path = root / "video" / f"{stem}.r13.avi"
    for k in range(n):
        t = 10 + k * (270 / n)
        f = _frame(path, t)
        if f is not None:
            _draw_gt(f, clip, t)
            tiles.append(_small(f, f"{stem[11:19]} {stem.split('.')[4]} t={t:.0f}s"))
    _sheet(tiles, 3, out / f"gt_{stem}.jpg")


def alerts(root, stem, out, max_n=12):
    r = json.loads((root / "results" / f"{stem}.json").read_text())
    clip, tiles = load_clip(root, stem), []
    path = root / "video" / f"{stem}.r13.avi"
    items = [("alert", e["t"], e) for e in r["events"]] + [("fence", c["t"], c) for c in r["fence_crossings"]]
    for kind, t, e in sorted(items, key=lambda x: x[1])[:max_n]:
        f = _frame(path, t)
        if f is None:
            continue
        h, w = f.shape[:2]
        _draw_gt(f, clip, t)
        _fence(f, r["camera"])
        bx = e.get("bbox_norm_xywh")
        if bx:
            cv2.rectangle(f, (int(bx[0] * w), int(bx[1] * h)), (int((bx[0] + bx[2]) * w), int((bx[1] + bx[3]) * h)), (0, 0, 255), 4)
        if e.get("point_px"):
            cv2.circle(f, (int(e["point_px"][0]), int(e["point_px"][1])), 14, (0, 0, 255), -1)
        tiles.append(_small(f, f"{kind} t={t:.1f}s"))
    if tiles:
        _sheet(tiles, 3, out / f"alerts_{stem}.jpg")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["traj", "gt", "alerts"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--camera")
    ap.add_argument("--stem")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    {"traj": lambda: traj(a.root, a.camera, a.out), "gt": lambda: gt(a.root, a.stem, a.out),
     "alerts": lambda: alerts(a.root, a.stem, a.out)}[a.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
