"""DEVICE: measure the detector and the pipeline on the Jetson and write ONE json + ONE markdown table.

Stages (each can be skipped; a skipped stage is absent from the record, never filled with a default):
  trt       trtexec GPU-compute p50/p95/p99 per engine in --engines, warm-up excluded
  e2e       per-frame latency (decode -> detector -> motion -> fusion -> rules) on a 1080p clip, for
            DETECTOR_TILING off, strip and full, through the real CameraService
  sustained --sustained-s seconds (default 600) of the pipeline looped over the clip: fps per window,
            temperature samples
  vlm       edge/vlm/latency.py run on this device

The record carries a hardware fingerprint (module, JetPack/L4T, TensorRT, CUDA, nvpmodel mode, whether
jetson_clocks has pinned the clocks). `measured_on_jetson` is false unless the fingerprint says the host
is a Jetson; the markdown then opens with a banner. Nothing here estimates a number.

  python edge/tools/jetson/benchmark.py --engines engines --clip clip_1080p.mp4 --out-dir results
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGE = HERE.parent.parent
sys.path.insert(0, str(EDGE))

from tools.jetson import common  # noqa: E402


# ---------------------------------------------------------------------------------- fingerprint


def _run(cmd: list[str], timeout: float = 20) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(errors="replace").replace("\x00", "").strip()
    except OSError:
        return None


def read_temps(root: str = "/sys/class/thermal") -> dict[str, float]:
    """Thermal zones in degrees C, by zone type."""
    out: dict[str, float] = {}
    for zone in sorted(Path(root).glob("thermal_zone*")):
        t, v = _read(str(zone / "type")), _read(str(zone / "temp"))
        if t and v and re.fullmatch(r"-?\d+", v):
            out[f"{zone.name}:{t}"] = int(v) / 1000.0
    return out


def collect_fingerprint() -> dict:
    nvp = _run(["nvpmodel", "-q"])
    jc = _run(["jetson_clocks", "--show"])
    jc_parsed = common.parse_jetson_clocks_show(jc) if jc else {}
    try:
        import tensorrt
        trt_version = tensorrt.__version__
    except Exception:
        trt_version = None
    fp = {
        "device_model": _read("/proc/device-tree/model"),
        "tegra_release": next(iter((_read("/etc/nv_tegra_release") or "").splitlines()), None),
        "l4t": None,
        "jetpack": _run(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"]),
        "tensorrt": trt_version,
        "cuda": (re.search(r"release ([\d.]+)", _run(["nvcc", "--version"]) or "") or [None, None])[1],
        "nvpmodel": common.parse_nvpmodel(nvp) if nvp else None,
        "jetson_clocks": jc_parsed or None,
        "jetson_clocks_pinned": common.clocks_pinned(jc_parsed) if jc_parsed else None,
        "kernel": platform.release(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "temps_c_start": read_temps(),
    }
    m = re.search(r"R(\d+).*REVISION: ([\d.]+)", fp["tegra_release"] or "")
    fp["l4t"] = f"{m.group(1)}.{m.group(2)}" if m else None
    return fp


# --------------------------------------------------------------------------------------- stages


def stage_trt(engines: list[Path], trtexec: str, out_dir: Path, duration_s: int, warmup_ms: int) -> dict:
    res: dict = {}
    for eng in engines:
        times = out_dir / f"{eng.stem}_times.json"
        cmd = common.bench_cmd(eng, trtexec=trtexec, warmup_ms=warmup_ms, duration_s=duration_s, times_json=times)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = proc.stdout + proc.stderr
        (out_dir / f"{eng.stem}_trtexec.log").write_text(common.scrub_paths(log))
        if proc.returncode != 0:
            res[eng.stem] = {"error": f"trtexec returned {proc.returncode}"}
            continue
        parsed = common.parse_trtexec_log(log)
        samples = common.parse_export_times(times.read_text()) if times.exists() else []
        row = common.latency_stats(samples) if samples else {}
        gpu = parsed["sections"].get("GPU Compute Time", {})
        row.update({"trtexec_log_p50_ms": gpu.get("median"), "trtexec_log_p95_ms": gpu.get("p95"),
                    "trtexec_log_p99_ms": gpu.get("p99"), "throughput_qps": parsed["throughput_qps"],
                    "trtexec_tensorrt": parsed["trt_version"], "command": common.scrub_paths(common.shell_join(cmd))})
        if not samples and gpu:      # no --exportTimes file: fall back to the log's own percentiles
            row.update({"p50_ms": gpu.get("median"), "p95_ms": gpu.get("p95"), "p99_ms": gpu.get("p99")})
        res[eng.stem] = row
        times.unlink(missing_ok=True)
    return res


def window_fps(stamps: list[float], window_s: float) -> list[float]:
    """Frames per second in consecutive windows of `window_s`; `stamps` are frame-completion times in s.
    The final partial window is dropped."""
    if not stamps:
        return []
    t0, out, i = stamps[0], [], 1      # the first stamp opens window 1; it is not a frame completed inside it
    k = 1
    while t0 + k * window_s <= stamps[-1]:
        end = t0 + k * window_s
        n = 0
        while i < len(stamps) and stamps[i] <= end:
            n += 1
            i += 1
        out.append(n / window_s)
        k += 1
    return out


def make_service(engine: Path, class_names: list[str], tiling: str, cal_dir: Path):
    from pipeline.appearance import AppearanceChannel, AppearanceConfig
    from pipeline.events import EventSink
    from pipeline.service import CameraService
    from pipeline.tiling import preset
    from tools.jetson.trt_session import TrtSession

    session = TrtSession(engine, class_names)
    appearance = AppearanceChannel(session, 640, class_names, AppearanceConfig(tiling=preset(tiling)))
    detector_ms: list[float] = []
    raw_detect = appearance.detect

    def timed(*a, **k):
        out = raw_detect(*a, **k)
        detector_ms.append(appearance.last_ms)
        return out

    appearance.detect = timed
    svc = CameraService(appearance, camera_id="JETSON", post_id="BENCH", sink=EventSink(maxlen=10), stride=2,
                        calibration_dir=cal_dir, source="file")
    return svc, detector_ms


def stage_e2e(engine: Path, class_names: list[str], clip: Path, tiling: str, target_fps: float, max_frames: int) -> tuple[dict, dict]:
    from pipeline.ingest import frames

    with tempfile.TemporaryDirectory() as tmp:
        svc, det_ms = make_service(engine, class_names, tiling, Path(tmp))
        per_frame, stamps, excluded, size = [], [], 0, None
        try:
            it = frames(clip, is_rtsp=False, camera_id="JETSON", target_fps=target_fps, max_frames=max_frames)
            while True:
                t0 = time.perf_counter()
                try:
                    frame = next(it)
                except StopIteration:
                    break
                size = frame.shape
                svc.process(frame)
                t1 = time.perf_counter()
                if svc.pipeline.calibration is None or excluded < 20:   # unlabelled calibration warm-up: not the steady path
                    excluded += 1
                    del det_ms[:]
                    continue
                per_frame.append((t1 - t0) * 1000)
                stamps.append(t1)
        finally:
            svc.close()
    if not per_frame:
        return {"error": "no frames measured (clip shorter than the calibration warm-up?)"}, {}
    row = common.latency_stats(per_frame)
    row["detector_p50_ms"] = common.percentile(det_ms, 50) if det_ms else None
    row["detector_runs"] = len(det_ms)
    row["fps"] = len(per_frame) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else None
    row["excluded_warmup_frames"] = excluded
    row["target_fps"] = target_fps
    return row, {"width": size[0], "height": size[1]}


def stage_sustained(engine: Path, class_names: list[str], clip: Path, tiling: str, seconds: float,
                    target_fps: float, window_s: float = 60.0) -> dict:
    from pipeline.ingest import frames

    stamps, temps = [], []
    with tempfile.TemporaryDirectory() as tmp:
        svc, _ = make_service(engine, class_names, tiling, Path(tmp))
        start = time.perf_counter()
        next_temp = start
        try:
            for frame in frames(clip, is_rtsp=False, camera_id="JETSON", target_fps=target_fps, loop=True):
                svc.process(frame)
                now = time.perf_counter()
                stamps.append(now)
                if now >= next_temp:
                    t = read_temps()
                    if t:
                        temps.append(max(t.values()))
                    next_temp = now + 30
                if now - start >= seconds:
                    break
        finally:
            svc.close()
    wins = window_fps(stamps, window_s)
    return {"seconds": round(stamps[-1] - start, 1) if stamps else 0, "tiling": tiling, "frames": len(stamps),
            "fps_mean": len(stamps) / (stamps[-1] - start) if stamps else None,
            "fps_windows": wins, "window_s": window_s,
            "fps_first_window": wins[0] if wins else None, "fps_last_window": wins[-1] if wins else None,
            "max_temp_c": max(temps) if temps else None, "temp_samples_c": temps}


def stage_vlm(out_dir: Path, clips: list[Path], repeats: int) -> dict:
    cmd = [sys.executable, str(EDGE / "vlm" / "latency.py"), "--repeats", str(repeats), "--label", "jetson",
           "--out-dir", str(out_dir)]
    if clips:
        cmd += ["--clips", *map(str, clips)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    path = out_dir / "moondream2_jetson.json"
    if proc.returncode != 0 or not path.exists():
        return {"error": f"vlm latency run failed ({proc.returncode})", "stderr_tail": common.scrub_paths(proc.stderr[-800:])}
    d = json.loads(path.read_text())
    return {k: d.get(k) for k in ("p50_ms", "p95_ms", "n_samples", "n_timed", "n_failed", "device", "hardware",
                                  "quantized", "frame_source")}


# ------------------------------------------------------------------------------------------ main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", type=Path, required=True, help="directory with the .engine files")
    ap.add_argument("--engine-for-pipeline", default="int8", choices=["int8", "fp16"])
    ap.add_argument("--clip", type=Path, default=None, help="1080p video for e2e and sustained")
    ap.add_argument("--vlm-clips", type=Path, nargs="*", default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=common.DEFAULT_MANIFEST)
    ap.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    ap.add_argument("--stages", nargs="+", default=["trt", "e2e", "sustained", "vlm"], choices=["trt", "e2e", "sustained", "vlm"])
    ap.add_argument("--trt-duration-s", type=int, default=30)
    ap.add_argument("--trt-warmup-ms", type=int, default=2000)
    ap.add_argument("--target-fps", type=float, default=8.0, help="ingest rate (the edge default)")
    ap.add_argument("--e2e-frames", type=int, default=600)
    ap.add_argument("--sustained-s", type=float, default=600.0)
    ap.add_argument("--vlm-repeats", type=int, default=24)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    manifest = common.load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fp = collect_fingerprint()
    rec: dict = {"schema": common.SCHEMA, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "label": args.label, "fingerprint": fp, "measured_on_jetson": common.is_jetson(fp),
                 "model": {"repo_id": manifest.get("repo_id"), "revision": manifest["revision"], "sha256": manifest["sha256"]},
                 "warnings": []}
    if not rec["measured_on_jetson"]:
        rec["warnings"].append("host is not a Jetson: nothing in this record is a board measurement")
    if fp.get("jetson_clocks_pinned") is False:
        rec["warnings"].append("jetson_clocks has not pinned the clocks; report this run as default-clock")
    if (fp.get("nvpmodel") or {}).get("name") not in (None, "MAXN_SUPER"):
        rec["warnings"].append(f"power mode is {(fp['nvpmodel'] or {}).get('name')}, not MAXN_SUPER")

    engines = sorted(args.engines.glob("*.engine"))
    class_names = manifest.get("class_names") or []
    if "trt" in args.stages:
        rec["detector_trt"] = stage_trt(engines, args.trtexec, args.out_dir, args.trt_duration_s, args.trt_warmup_ms)
    pipe_engine = next((e for e in engines if f"_{args.engine_for_pipeline}_" in e.name), None)
    if {"e2e", "sustained"} & set(args.stages):
        if not (args.clip and pipe_engine):
            raise SystemExit("e2e/sustained need --clip and an engine matching --engine-for-pipeline")
        rec["pipeline_engine"] = pipe_engine.name
    if "e2e" in args.stages:
        rec["pipeline_e2e"] = {}
        for tiling in ("off", "strip", "full"):
            row, clip_info = stage_e2e(pipe_engine, class_names, args.clip, tiling, args.target_fps, args.e2e_frames)
            rec["pipeline_e2e"][tiling] = row
            rec["clip"] = {"name": args.clip.name, **clip_info}
            if clip_info and clip_info.get("height") != 1080:
                rec["warnings"].append(f"clip is {clip_info['width']}x{clip_info['height']}, not 1080p")
    if "sustained" in args.stages:
        rec["sustained"] = stage_sustained(pipe_engine, class_names, args.clip, "off", args.sustained_s, args.target_fps)
    if "vlm" in args.stages:
        rec["vlm"] = stage_vlm(args.out_dir, args.vlm_clips or [], args.vlm_repeats)
    fp["temps_c_end"] = read_temps()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = args.out_dir / f"jetson_bench_{stamp}"
    base.with_suffix(".json").write_text(common.scrub_paths(json.dumps(rec, indent=2)) + "\n")
    base.with_suffix(".md").write_text(common.scrub_paths(common.render_markdown(rec)))
    print(f"wrote {base.name}.json and .md  (measured_on_jetson={rec['measured_on_jetson']})")
    for w in rec["warnings"]:
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
