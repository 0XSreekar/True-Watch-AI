"""Pure helpers shared by the Jetson kit: manifest check, trtexec command lines and log parser,
percentiles, hardware fingerprint parsers, result rendering. Standard library only, so every function
here is unit-tested on a development machine.

Nothing in this module measures anything. A result record is only marked `measured_on_jetson` when the
fingerprint it was collected with says the host is a Jetson (see `is_jetson`); anything else is a dry run
and renders with a banner saying so.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "training" / "results" / "hf_model.json"
WORDING_UNMEASURED = "design target, not measured on hardware"
SCHEMA = "truewatch.jetson_bench.v1"


class KitError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------ manifest


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sha = str(data.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise KitError(f"{path}: sha256 must be 64 lowercase hex characters")
    for key in ("file", "imgsz", "resolve_url", "revision"):
        if key not in data:
            raise KitError(f"{path}: missing {key!r}")
    return data


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_onnx(onnx: Path, manifest: dict) -> str:
    """Raise unless `onnx` is byte-identical to the pinned release. Returns the digest."""
    got = sha256_file(onnx)
    if got != manifest["sha256"]:
        raise KitError(f"{onnx.name} sha256 {got} != pinned {manifest['sha256']} (training/results/hf_model.json)")
    return got


# ------------------------------------------------------------------------------------------ trtexec


def build_cmd(onnx: Path, engine: Path, precision: str, imgsz: int = 640, *, calib: Path | None = None,
              trtexec: str = "trtexec", workspace_mb: int | None = None, timing_cache: Path | None = None,
              input_name: str = "images") -> list[str]:
    """trtexec command that builds a static-shape, batch-1 engine. Precision is fp16 or int8."""
    if precision not in ("fp16", "int8"):
        raise KitError(f"precision must be fp16 or int8, got {precision!r}")
    cmd = [trtexec, f"--onnx={onnx}", f"--saveEngine={engine}", f"--shapes={input_name}:1x3x{imgsz}x{imgsz}",
           "--fp16"]
    if precision == "int8":
        if calib is None:
            raise KitError("int8 needs a calibration cache (--calib)")
        # --fp16 stays on beside --int8 so a layer with no int8 kernel falls back to fp16, not fp32.
        cmd += ["--int8", f"--calib={calib}"]
    if workspace_mb:
        cmd.append(f"--memPoolSize=workspace:{int(workspace_mb)}M")
    if timing_cache:
        cmd.append(f"--timingCacheFile={timing_cache}")
    cmd.append("--skipInference")      # build and save only; latency is a separate, longer run
    return cmd


def bench_cmd(engine: Path, *, trtexec: str = "trtexec", warmup_ms: int = 2000, duration_s: int = 30,
              iterations: int = 1000, times_json: Path | None = None) -> list[str]:
    """trtexec latency run on a prebuilt engine. Warm-up is trtexec's own --warmUp (excluded from stats)."""
    cmd = [trtexec, f"--loadEngine={engine}", f"--warmUp={int(warmup_ms)}", f"--duration={int(duration_s)}",
           f"--iterations={int(iterations)}", "--useSpinWait", "--separateProfileRun"]
    if times_json:
        cmd.append(f"--exportTimes={times_json}")
    return cmd


def shell_join(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


_STAT = re.compile(r"(min|max|mean|median|percentile\((\d+)%\))\s*=\s*([0-9.]+)\s*ms")
_SECTION = re.compile(r"^(?:\[[^\]]*\]\s*)*(Latency|Enqueue Time|H2D Latency|GPU Compute Time|D2H Latency):\s*(.*)$")


def parse_trtexec_log(text: str) -> dict:
    """Pull the performance summary out of a trtexec log. Missing sections are absent, never 0.

    Returns {"sections": {name: {min,max,mean,median,p90,p95,p99}}, "throughput_qps", "trt_version",
    "passed"}. Times in ms.
    """
    out: dict = {"sections": {}, "throughput_qps": None, "trt_version": None, "passed": False}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("&&&& PASSED"):
            out["passed"] = True
        m = re.search(r"TensorRT\.trtexec\s+\[TensorRT v(\d+)\]", line)
        if m:
            v = m.group(1)           # v100300 = 10.3.0 ; v8602 = 8.6.2
            out["trt_version"] = f"{int(v[:-4] or 0)}.{int(v[-4:-2])}.{int(v[-2:])}" if len(v) > 4 else v
        m = re.search(r"Throughput:\s*([0-9.]+)\s*qps", line)
        if m:
            out["throughput_qps"] = float(m.group(1))
        m = _SECTION.match(line)
        if not m:
            continue
        stats: dict = {}
        for s in _STAT.finditer(m.group(2)):
            key = s.group(1) if s.group(2) is None else f"p{s.group(2)}"
            stats[key] = float(s.group(3))
        if stats:
            out["sections"][m.group(1)] = stats
    return out


def percentile(values: Iterable[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default), p in 0..100."""
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("percentile of an empty sample")
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def latency_stats(samples_ms: Sequence[float]) -> dict:
    return {"n": len(samples_ms), "mean_ms": sum(samples_ms) / len(samples_ms),
            "p50_ms": percentile(samples_ms, 50), "p95_ms": percentile(samples_ms, 95),
            "p99_ms": percentile(samples_ms, 99), "max_ms": max(samples_ms)}


def parse_export_times(text: str) -> list[float]:
    """GPU compute times (ms) from trtexec --exportTimes JSON: a list of {"computeMs": x, ...}."""
    return [float(r["computeMs"]) for r in json.loads(text) if "computeMs" in r]


# ------------------------------------------------------------------------------------ fingerprint


def parse_nvpmodel(text: str) -> dict:
    """`nvpmodel -q`: 'NV Power Mode: MAXN_SUPER' then the numeric mode id on the next line."""
    mode = re.search(r"NV Power Mode:\s*(\S+)", text)
    ident = re.search(r"^\s*(\d+)\s*$", text, re.M)
    return {"name": mode.group(1) if mode else None, "id": int(ident.group(1)) if ident else None}


def parse_jetson_clocks_show(text: str) -> dict:
    """`jetson_clocks --show`: GPU and EMC clocks, and the governor and clocks of cpu0."""
    out: dict = {"gpu": None, "emc": None, "cpu0": None}
    for line in text.splitlines():
        s = line.strip()
        freqs = {k: int(v) for k, v in re.findall(r"(MinFreq|MaxFreq|CurrentFreq)=(\d+)", s)}
        if re.match(r"^GPU\b", s):
            out["gpu"] = freqs
        elif re.match(r"^EMC\b", s):
            out["emc"] = freqs
        elif s.startswith("cpu0:"):
            gov = re.search(r"Governor=(\S+)", s)
            out["cpu0"] = {"governor": gov.group(1) if gov else None, **freqs}
    return out


def clocks_pinned(parsed: dict) -> bool | None:
    """True when the GPU's current clock sits at its maximum (what `jetson_clocks` does). None if unknown."""
    gpu = parsed.get("gpu") or {}
    if "CurrentFreq" in gpu and "MaxFreq" in gpu:
        return gpu["CurrentFreq"] >= gpu["MaxFreq"]
    return None


def is_jetson(fp: dict) -> bool:
    text = ((fp.get("device_model") or "") + " " + (fp.get("tegra_release") or "")).lower()
    return "jetson" in text or "tegra" in text


# --------------------------------------------------------------------------------------- rendering


def _f(v, d=2):
    return "n/a" if v is None else f"{v:.{d}f}"


def render_markdown(rec: dict) -> str:
    fp = rec.get("fingerprint", {})
    lines = ["# Jetson benchmark", ""]
    if not rec.get("measured_on_jetson"):
        lines += ["> **NOT A JETSON MEASUREMENT.** The fingerprint does not identify this host as a Jetson. "
                  "Do not quote any number below as a board result.", ""]
    lines += ["## Hardware fingerprint", "", "| field | value |", "|---|---|"]
    for k in ("device_model", "jetpack", "l4t", "tensorrt", "cuda", "nvpmodel", "jetson_clocks_pinned", "kernel", "python"):
        lines.append(f"| {k} | {fp.get(k) if fp.get(k) is not None else 'not recorded'} |")
    lines += [f"| model sha256 | {rec.get('model', {}).get('sha256', 'not recorded')} |", ""]
    det = rec.get("detector_trt") or {}
    if det:
        lines += ["## Detector, TensorRT GPU compute time (warm-up excluded)", "",
                  "| engine | n | p50 ms | p95 ms | p99 ms | qps |", "|---|---|---|---|---|---|"]
        for name, r in det.items():
            lines.append(f"| {name} | {r.get('n', 'n/a')} | {_f(r.get('p50_ms'))} | {_f(r.get('p95_ms'))} | "
                         f"{_f(r.get('p99_ms'))} | {_f(r.get('throughput_qps'), 1)} |")
        lines.append("")
    e2e = rec.get("pipeline_e2e") or {}
    if e2e:
        clip = rec.get("clip", {})
        lines += ["## End-to-end per frame (decode, detector, motion, fusion, rules)", "",
                  f"Clip {clip.get('name', 'n/a')}, {clip.get('width', '?')}x{clip.get('height', '?')}.", "",
                  "| tiling | frames | frame p50 ms | p95 ms | p99 ms | detector p50 ms | fps |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in e2e.items():
            lines.append(f"| {name} | {r.get('n', 'n/a')} | {_f(r.get('p50_ms'))} | {_f(r.get('p95_ms'))} | "
                         f"{_f(r.get('p99_ms'))} | {_f(r.get('detector_p50_ms'))} | {_f(r.get('fps'), 1)} |")
        lines.append("")
    sus = rec.get("sustained")
    if sus:
        lines += ["## Sustained run", "",
                  f"{sus.get('seconds')} s, tiling {sus.get('tiling')}, {sus.get('frames')} frames, mean "
                  f"{_f(sus.get('fps_mean'))} fps; first window {_f(sus.get('fps_first_window'))} fps, last window "
                  f"{_f(sus.get('fps_last_window'))} fps; max temperature {_f(sus.get('max_temp_c'), 1)} C.", ""]
    vlm = rec.get("vlm")
    if vlm:
        lines += ["## VLM (edge/vlm/latency.py on the device)", "",
                  f"p50 {_f((vlm.get('p50_ms') or 0) / 1000)} s, p95 {_f((vlm.get('p95_ms') or 0) / 1000)} s, "
                  f"n={vlm.get('n_samples')}, device {vlm.get('device')}", ""]
    if not (det or e2e or sus or vlm):
        lines.append(f"No stage ran. Status of every latency claim: {WORDING_UNMEASURED}.")
    return "\n".join(lines).rstrip() + "\n"


def scrub_paths(text: str) -> str:
    """Remove home and temp directory prefixes from anything written to a result file."""
    return re.sub(r"(/private)?/(tmp|var/folders)/[^\s\"']*|/Users/[^/\s\"']+|/home/[^/\s\"']+", "<path>", text)
