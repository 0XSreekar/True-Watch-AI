"""DEVICE: build the FP16 and INT8 TensorRT engines from the pinned ONNX detector with trtexec.

640x640, batch 1, static shape. The ONNX is checked against training/results/hf_model.json first. INT8 needs
a calibration cache; if --calib-cache does not exist and --calib-images is given, make_calib_cache.py
writes it. Every trtexec log is kept next to the engines. The first build takes minutes on an Orin Nano;
that is build time, not inference time, and is recorded in build_info.json.

  python edge/tools/jetson/build_engines.py --onnx final.onnx --out-dir engines \\
      --calib-images calib/ --calib-list calib_list.txt
  python edge/tools/jetson/build_engines.py --onnx final.onnx --out-dir engines --dry-run   # print commands
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from tools.jetson.common import DEFAULT_MANIFEST, build_cmd, load_manifest, scrub_paths, shell_join, verify_onnx  # noqa: E402


def plan(onnx: Path, out_dir: Path, manifest: dict, calib: Path, trtexec: str, precisions: list[str],
         workspace_mb: int | None) -> list[tuple[str, list[str], Path]]:
    imgsz = int(manifest["imgsz"])
    stem = Path(manifest["file"]).stem
    steps = []
    for p in precisions:
        engine = out_dir / f"{stem}_{p}_{imgsz}_b1.engine"
        steps.append((p, build_cmd(onnx, engine, p, imgsz, calib=calib if p == "int8" else None, trtexec=trtexec,
                                   workspace_mb=workspace_mb, timing_cache=out_dir / "timing.cache"), engine))
    return steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    ap.add_argument("--precisions", nargs="+", default=["fp16", "int8"], choices=["fp16", "int8"])
    ap.add_argument("--calib-cache", type=Path, default=None, help="default <out-dir>/calib.cache")
    ap.add_argument("--calib-images", type=Path, default=None)
    ap.add_argument("--calib-list", type=Path, default=None)
    ap.add_argument("--workspace-mb", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    cache = args.calib_cache or args.out_dir / "calib.cache"
    steps = plan(args.onnx, args.out_dir, manifest, cache, args.trtexec, args.precisions, args.workspace_mb)
    if args.dry_run:
        for p, cmd, _ in steps:
            print(f"# {p}\n{shell_join(cmd)}")
        return 0

    digest = verify_onnx(args.onnx, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    info = {"onnx_sha256": digest, "revision": manifest["revision"], "builds": {}}
    if "int8" in args.precisions and not cache.exists():
        if not args.calib_images:
            raise SystemExit(f"{cache} missing: pass --calib-cache or --calib-images (see docs/JETSON.md)")
        cmd = [sys.executable, str(HERE / "make_calib_cache.py"), "--onnx", str(args.onnx), "--images",
               str(args.calib_images), "--out", str(cache)]
        if args.calib_list:
            cmd += ["--list", str(args.calib_list)]
        t0 = time.time()
        subprocess.run(cmd, check=True)
        info["calibration_s"] = round(time.time() - t0, 1)
    for p, cmd, engine in steps:
        log = args.out_dir / f"build_{p}.log"
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        info["builds"][p] = {"engine": engine.name, "build_s": round(time.time() - t0, 1), "returncode": rc,
                             "command": scrub_paths(shell_join(cmd))}
        (args.out_dir / "build_info.json").write_text(json.dumps(info, indent=2) + "\n")
        if rc != 0:
            raise SystemExit(f"trtexec failed for {p}; see {log}")
        print(f"built {engine.name} in {info['builds'][p]['build_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
