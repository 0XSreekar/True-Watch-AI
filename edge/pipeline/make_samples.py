"""Build the demo clips samples/day.mp4 and samples/ir.mp4 from public datasets you already have locally.

No media is committed: the dataset licences are not read as permitting redistribution, so the clips are
rebuilt on each machine and samples/*.mp4 is git-ignored.

Default source: KAIST Multispectral Pedestrian (paired visible + LWIR, 20 fps, consecutive frames), the
set docs/MEASUREMENTS.md sections 1 and 2 were measured on. set00/V001 frames 1430-1864 are a stretch in
which the recording vehicle is stopped at a crossing and pedestrians walk through, i.e. the closest a
vehicle-mounted dataset gets to a fixed pole camera. day.mp4 is the visible stream, ir.mp4 the LWIR stream
of the SAME frames, so the day/night weighting difference is judged on identical scenes.

    cd edge
    python -m pipeline.make_samples                             # KAIST from ../var/datasets/kaist/kaist_train
    python -m pipeline.make_samples --also-1080p                # plus day_1080p.mp4 / ir_1080p.mp4 for FPS
    python -m pipeline.make_samples --from llvip                # LLVIP instead (see the caveat below)

LLVIP caveat. LLVIP (paired visible + infrared, night) ships its frames sampled from video with seconds
between consecutive files, so a clip built from it is a slideshow: the detector and the camera-state
monitor can be exercised on it, but dense optical flow between frames seconds apart is meaningless and the
motion channel must not be judged on it. `--from llvip` therefore writes llvip_visible.mp4 and
llvip_ir.mp4, never day.mp4 / ir.mp4.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

EDGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EDGE_ROOT.parent
DEFAULT_KAIST = Path(os.environ.get("KAIST_ROOT", REPO_ROOT / "var/datasets/kaist/kaist_train"))
DEFAULT_LLVIP = Path(os.environ.get("LLVIP_ZIP", REPO_ROOT / "var/datasets/llvip/LLVIP.zip"))
DEFAULT_OUT = REPO_ROOT / "samples"


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        raise SystemExit(f"OpenCV cannot write {path}")
    return w


def write_clip(frames, path: Path, fps: float, size: tuple[int, int] | None = None) -> int:
    writer, n = None, 0
    for img in frames:
        if size is not None and (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        if writer is None:
            writer = _writer(path, fps, (img.shape[1], img.shape[0]))
        writer.write(img)
        n += 1
    if writer is not None:
        writer.release()
    return n


def kaist_frames(root: Path, seq: str, stream: str, start: int, end: int):
    folder = root / seq / stream
    if not folder.is_dir():
        raise SystemExit(f"KAIST folder not found: {folder} (set --kaist-root or KAIST_ROOT)")
    for i in range(start, end):
        p = folder / f"I{i:05d}.jpg"
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            break
        yield img


def llvip_frames(zip_path: Path, stream: str, prefix: str, limit: int):
    if not zip_path.is_file():
        raise SystemExit(f"LLVIP zip not found: {zip_path} (set --llvip-zip or LLVIP_ZIP)")
    with zipfile.ZipFile(zip_path) as z:
        pat = re.compile(rf"LLVIP/{stream}/test/({prefix}\d+)\.jpg$")
        names = sorted(n for n in z.namelist() if pat.search(n))[:limit]
        if not names:
            raise SystemExit(f"no LLVIP {stream} test frames with prefix {prefix}")
        for n in names:
            buf = np.frombuffer(z.read(n), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                yield img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="origin", choices=("kaist", "llvip"), default="kaist")
    ap.add_argument("--kaist-root", type=Path, default=DEFAULT_KAIST)
    ap.add_argument("--kaist-seq", default="set00/V001")
    ap.add_argument("--start", type=int, default=1430)
    ap.add_argument("--end", type=int, default=1865)
    ap.add_argument("--llvip-zip", type=Path, default=DEFAULT_LLVIP)
    ap.add_argument("--llvip-prefix", default="19", help="LLVIP test sequence prefix (file names 19xxxx)")
    ap.add_argument("--llvip-limit", type=int, default=200)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--also-1080p", action="store_true", help="also write *_1080p.mp4 upscaled to 1920x1080")
    args = ap.parse_args(argv)

    out: Path = args.out
    if args.origin == "kaist":
        fps = 20.0
        jobs = [("day", "visible"), ("ir", "lwir")]
        for name, stream in jobs:
            n = write_clip(kaist_frames(args.kaist_root, args.kaist_seq, stream, args.start, args.end), out / f"{name}.mp4", fps)
            print(f"{out / f'{name}.mp4'}: {n} frames at {fps:g} fps from KAIST {args.kaist_seq}/{stream} "
                  f"I{args.start:05d}-I{args.start + n - 1:05d}")
            if args.also_1080p:
                n2 = write_clip(kaist_frames(args.kaist_root, args.kaist_seq, stream, args.start, args.end),
                                out / f"{name}_1080p.mp4", fps, (1920, 1080))
                print(f"{out / f'{name}_1080p.mp4'}: {n2} frames upscaled to 1920x1080 (for FPS measurement only)")
    else:
        fps = 2.0
        print("LLVIP frames are seconds apart: these clips are for appearance and camera-state checks only,",
              "not for the motion channel.", file=sys.stderr)
        for name, stream in (("llvip_visible", "visible"), ("llvip_ir", "infrared")):
            n = write_clip(llvip_frames(args.llvip_zip, stream, args.llvip_prefix, args.llvip_limit),
                           out / f"{name}.mp4", fps, (1280, 1024))
            print(f"{out / f'{name}.mp4'}: {n} frames at {fps:g} fps from LLVIP {stream}/test {args.llvip_prefix}xxxx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
