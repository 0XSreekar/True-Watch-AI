#!/usr/bin/env python3
"""MEVA data helpers for the field measurements (slide 5): clip index, download manifest, KPF -> ground truth.

Data: MEVA "Multiview Extended Video with Activities", Kitware Inc. and IARPA, licensed CC BY 4.0
(https://mevadata.org/resources/MEVA-data-license.txt). Video is read from the public S3 bucket
mevadata-public-01; annotations come from https://gitlab.kitware.com/meva/meva-data-repo
(annotation/DIVA-phase-2/MEVA/kitware-meva-training). Nothing here is committed: the video, the annotation
checkout and the manifest live under a data root outside the repository.

    python -m tools.meva_data inventory --root DATA   # annotated clips that exist in the bucket, per camera
    python -m tools.meva_data select    --root DATA   # write the chosen outdoor clip list (selection.json)
    python -m tools.meva_data download  --root DATA   # fetch the selection, write manifest.json (size + sha256)

A KPF clip has <clip>.geom.yml (one line per object per frame: box 'x1 y1 x2 y2' in pixels of the full
frame, 'id1' = track id, 'ts0' = frame index at the clip's 30 fps), <clip>.types.yml (track id -> class)
and <clip>.activities.yml. Annotations are activity-centric: people and vehicles that take part in no
annotated activity are often not boxed at all, which is why the false-alert measurement checks the
completeness of the boxes first (see docs/VERIFICATION_REPORT.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

BUCKET = "https://mevadata-public-01.s3.amazonaws.com/"
ANN_REL = "meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware-meva-training"
LICENCE = "CC BY 4.0 - MEVA (Kitware Inc. and IARPA), https://mevadata.org/resources/MEVA-data-license.txt"
FPS = 30.0

# Cameras whose view is outdoor (checked by eye on a first frame of every annotated camera).
OUTDOOR = ["G300", "G328", "G336", "G339", "G341", "G424", "G436", "G505", "G506", "G638", "G639"]

_GEOM = re.compile(r"'g0': '([\d.\-]+) ([\d.\-]+) ([\d.\-]+) ([\d.\-]+)'.*?'id1': (\d+).*?'ts0': (\d+)")
_TYPE = re.compile(r"'cset3': \{([^}]*)\}.*?'id1': (\d+)")


def ann_dir(root: Path) -> Path:
    return Path(root) / ANN_REL


def clip_stem_parts(stem: str) -> dict:
    date, start, end, site, cam = stem.split(".")[:5]
    return {"stem": stem, "date": date, "start": start, "end": end, "site": site, "camera": cam,
            "hour": int(start.split("-")[0])}


def read_geom(path: Path) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    """track id -> [(frame, (x1, y1, x2, y2))] sorted by frame."""
    tracks: dict[int, list] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        m = _GEOM.search(line)
        if m:
            x1, y1, x2, y2, tid, ts = m.groups()
            tracks[int(tid)].append((int(ts), (float(x1), float(y1), float(x2), float(y2))))
    for v in tracks.values():
        v.sort()
    return dict(tracks)


def read_types(path: Path) -> dict[int, str]:
    """track id -> the highest-scoring class of its type set (person / vehicle / other ...)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = _TYPE.search(line)
        if m:
            scores = {k.strip().strip("'"): float(v) for k, v in (kv.split(":") for kv in m.group(1).split(",") if ":" in kv)}
            if scores:
                out[int(m.group(2))] = max(scores, key=scores.get)
    return out


def load_clip(root: Path, stem: str) -> dict:
    """{'tracks': {tid: {'cls': str, 'boxes': [(frame, box)]}}} for one clip."""
    (g,) = (ann_dir(root) / stem.split(".")[0]).glob(f"*/{stem}.geom.yml")
    geom = read_geom(g)
    types = read_types(g.with_name(f"{stem}.types.yml"))
    return {"tracks": {tid: {"cls": types.get(tid, "other"), "boxes": b} for tid, b in geom.items()}}


def ground_point(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-centre of a box: the same contact point the tracker's history uses."""
    return ((box[0] + box[2]) / 2.0, box[3])


def list_bucket(cache: Path) -> list[tuple[str, int]]:
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    keys, tok = [], None
    while True:
        q = {"list-type": "2", "prefix": "drops-123-r13/"}
        if tok:
            q["continuation-token"] = tok
        body = urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(q)).read().decode()
        keys += [(m.group(1), int(m.group(2))) for m in re.finditer(r"<Contents><Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", body)]
        m = re.search(r"<NextContinuationToken>([^<]+)", body)
        if not m:
            break
        tok = m.group(1)
    cache.write_text(json.dumps(keys))
    return keys


def annotated_clips(root: Path) -> dict[str, dict]:
    """stem -> {key, size, camera, hour, n_person, n_vehicle} for annotated clips present in the bucket."""
    keys = list_bucket(Path(root) / "bucket_keys.json")
    avail = {Path(k).name[:-8]: (k, s) for k, s in keys if k.endswith(".avi")}
    out = {}
    for g in sorted(ann_dir(root).glob("*/*/*.geom.yml")):
        stem = g.name[: -len(".geom.yml")]
        if stem not in avail:
            continue
        types = read_types(g.with_name(stem + ".types.yml"))
        geom = read_geom(g)
        cls = [types.get(t, "other") for t in geom]
        out[stem] = {"key": avail[stem][0], "size": avail[stem][1], **clip_stem_parts(stem),
                     "n_person": cls.count("person"), "n_vehicle": cls.count("vehicle")}
    return out


def select(clips: dict[str, dict], per_camera: int = 3, cameras: list[str] = OUTDOOR) -> list[str]:
    """Per outdoor camera: `per_camera` clips with at least two person tracks, spread evenly over the
    camera's annotated (date, hour) slots so that the morning, midday and dusk hours are all represented."""
    chosen = []
    for cam in cameras:
        pool = sorted((c for c in clips.values() if c["camera"] == cam and c["n_person"] >= 2),
                      key=lambda c: (c["date"], c["start"]))
        if not pool:
            continue
        idx = sorted({round(i * (len(pool) - 1) / max(1, per_camera - 1)) for i in range(per_camera)})
        chosen += [pool[i]["stem"] for i in idx]
    return chosen


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def download(root: Path, stems: list[str], clips: dict[str, dict]) -> list[dict]:
    vdir = Path(root) / "video"
    vdir.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor

    def one(stem: str) -> dict:
        c = clips[stem]
        dst = vdir / f"{stem}.r13.avi"
        if not dst.exists() or dst.stat().st_size != c["size"]:
            tmp = dst.with_suffix(".part")
            with urllib.request.urlopen(BUCKET + urllib.parse.quote(c["key"])) as r, open(tmp, "wb") as f:
                while blk := r.read(1 << 20):
                    f.write(blk)
            tmp.rename(dst)
        row = {"stem": stem, "camera": c["camera"], "site": c["site"], "date": c["date"], "start": c["start"],
               "end": c["end"], "size_bytes": dst.stat().st_size, "sha256": sha256_file(dst), "source_key": c["key"],
               "n_person_tracks": c["n_person"], "n_vehicle_tracks": c["n_vehicle"]}
        print("fetched", stem, row["size_bytes"], flush=True)
        return row

    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(one, stems))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inventory", "select", "download"])
    ap.add_argument("--root", type=Path, required=True, help="data root holding meva-data-repo/ and video/")
    ap.add_argument("--per-camera", type=int, default=3)
    args = ap.parse_args(argv)
    clips = annotated_clips(args.root)
    sel_path = args.root / "selection.json"
    if args.cmd == "inventory":
        by_cam: dict[str, list] = defaultdict(list)
        for c in clips.values():
            by_cam[c["camera"]].append(c)
        for cam, cs in sorted(by_cam.items()):
            print(cam, len(cs), "clips; person tracks", sum(c["n_person"] for c in cs), "vehicle", sum(c["n_vehicle"] for c in cs))
    elif args.cmd == "select":
        stems = select(clips, args.per_camera)
        sel_path.write_text(json.dumps(stems, indent=1))
        print(len(stems), "clips,", round(sum(clips[s]["size"] for s in stems) / 1e9, 2), "GB")
    else:
        stems = json.loads(sel_path.read_text())
        rows = download(args.root, stems, clips)
        (args.root / "manifest.json").write_text(json.dumps({"licence": LICENCE, "clips": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
