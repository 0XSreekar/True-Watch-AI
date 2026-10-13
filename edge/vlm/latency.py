"""Measure and record local Moondream 2 latency, honestly, on the machine
that runs this.

This is a script, not a test: it loads the REAL model (no stub runtime),
warms it up once (excluded from the measured sample — a cold load is a
one-time service-boot cost, not a per-alert one), then times a realistic
number of `judge.explain` calls end to end (prompt build + `describe()` +
guardrail check) against frames that look like what the pipeline actually
hands the judge. It writes a machine-readable JSON file under
`edge/vlm/results/` and prints a short summary to stdout.

Frame source
------------
Real frames from video clips (`--clips file.mp4 dir/ ...`: frames are taken evenly across the clips), else
real LLIVIP thermal/RGB pairs when the dataset zip named by `--llvip-zip` is present, else synthetic frames
built the same way `edge/tests/` builds its own sample clips (an OpenCV-drawn scene), labelled as synthetic
in the output so a reader never mistakes a synthetic-frame number for one measured on real footage. Either
way every frame actually passes through the model, nothing here is a mocked or estimated number.

Output is one JSON file per run (default `edge/vlm/results/`), named for the device and label, holding the
hardware, device, model revision, library versions, n, p50/p95 and every sample, so the number quoted in
edge/README.md can be traced to the run that produced it.

Reading these numbers
----------------------
The PDF's "~3 s" is explicitly a Jetson-class edge-device target, not a
promise about a development laptop; `edge/README.md` says so next to
these figures, and so does the JSON each run writes. p50/p95 measured here
are this machine's numbers, offered as evidence the pipeline design (a
model that can take seconds and never blocks the ~30 ms alert) is sound —
not as a substitute for measuring on the target hardware before shipping.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Run as a script, sys.path[0] is this directory, where queue.py would shadow the standard library's
# `queue` for every module imported below. Drop it and put the edge root first instead.
_here = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _here]
sys.path.insert(0, str(_here.parent))

from vlm.judge import DEFAULT_MAX_TOKENS, JudgeInput, explain  # noqa: E402
from vlm.prompt import CameraContext, EventContext  # noqa: E402

log = logging.getLogger("truewatch.vlm.latency")

HERE = Path(__file__).resolve().parent
DEFAULT_LLVIP_ZIP = Path(__file__).resolve().parents[2] / "var" / "datasets" / "llvip" / "LLVIP.zip"


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def _synthetic_frame(seed: int) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    frame = np.full((384, 384, 3), int(rng.integers(20, 60)), dtype=np.uint8)
    color = tuple(int(x) for x in rng.integers(120, 230, size=3))
    cv2.rectangle(frame, (90, 190), (270, 260), color, -1)
    cv2.rectangle(frame, (170, 140), (270, 190), color, -1)
    cv2.circle(frame, (120, 260), 22, (25, 25, 25), -1)
    cv2.circle(frame, (240, 260), 22, (25, 25, 25), -1)
    cv2.putText(frame, "PILLAR 42", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
    return frame[:, :, ::-1]  # BGR, matching the pipeline's own frame convention


def _frames_from_llvip(zip_path: Path, count: int) -> list[np.ndarray]:
    import cv2

    frames: list[np.ndarray] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            n for n in zf.namelist()
            if n.lower().endswith((".jpg", ".jpeg", ".png")) and "/infrared/" in n.lower()
        ]
        if not names:
            names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        names.sort()
        step = max(1, len(names) // max(count, 1))
        for name in names[::step][:count]:
            data = zf.read(name)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
    return frames


def _frames_from_clips(paths: list[Path], count: int) -> tuple[list[np.ndarray], list[str]]:
    """`count` frames spread evenly over the given video files (BGR), and the clip names they came from."""
    import cv2

    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.mp4")) if p.is_dir() else [p])
    frames: list[np.ndarray] = []
    names: list[str] = []
    per_clip = max(1, -(-count // max(1, len(files))))
    for f in files:
        cap = cv2.VideoCapture(str(f))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue
        # skip the first and last 10%: clips open on empty road and end on the tail of the pass
        for pos in np.linspace(total * 0.1, total * 0.9, per_clip, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
                names.append(f.name)
        cap.release()
    return frames[:count], names[:count]


def load_frames(count: int, llvip_zip: Path | None, clips: list[Path] | None = None) -> tuple[list[np.ndarray], str]:
    """Return `(frames, source_label)`. `source_label` is "clips: <names>", "llvip" or "synthetic"."""
    if clips:
        frames, names = _frames_from_clips(clips, count)
        if frames:
            log.info("using %d real frames from %d clip(s)", len(frames), len(set(names)))
            return frames, "clips: " + ", ".join(sorted(set(names)))
        log.warning("no frames could be read from %s; falling back", [c.name for c in clips])
    if llvip_zip is not None and llvip_zip.exists():
        try:
            frames = _frames_from_llvip(llvip_zip, count)
            if frames:
                log.info("using %d real LLVIP frames from %s", len(frames), llvip_zip)
                return frames, "llvip"
        except Exception:  # noqa: BLE001
            log.exception("could not read LLVIP zip at %s; falling back to synthetic frames", llvip_zip)
    log.info("LLVIP zip not available at %s; using %d synthetic frames", llvip_zip, count)
    return [_synthetic_frame(i) for i in range(count)], "synthetic"


# --------------------------------------------------------------------------
# Events the judge is asked to explain
# --------------------------------------------------------------------------


def _sample_events() -> list[JudgeInput]:
    """A handful of realistic fused events, built the same way
    `edge/tests/test_vlm_guardrails.py` does, so the sentences produced here
    are directly comparable to that suite's fixtures."""
    from rules.engine import (
        RULE_FENCE_CROSSED,
        RULE_GROUP_FORMED,
        RULE_LOITERING,
        RULE_OUT_OF_HOURS,
        RULE_WRONG_DIRECTION,
        FiredRule,
    )

    def fired(name: str, evidence: dict, reason: str) -> FiredRule:
        return FiredRule(rule_name=name, track_id=6, evidence=evidence, reason=reason)

    from vlm.prompt import POST_TIMEZONE

    def ts(h: int, m: int) -> float:
        return datetime(2026, 8, 14, h, m, 7, tzinfo=POST_TIMEZONE).timestamp()

    raxaul = CameraContext.from_camera_row(
        {"id": "RXL-01", "name": "BOP Raxaul — Pillar 42", "sector": "Raxaul", "ir": False},
        sanctioned=[("06:00", "20:00")],
        image_right_bearing_deg=90.0,
    )
    panitanki = CameraContext.from_camera_row(
        {"id": "PNT-07", "name": "Panitanki — River Bank", "sector": "Panitanki", "ir": True},
        sanctioned=[("05:00", "19:00")],
    )
    galgalia = CameraContext.from_camera_row(
        {"id": "GLG-05", "name": "Galgalia — Culvert Approach", "sector": "Galgalia", "ir": True}
    )

    events = [
        JudgeInput(
            EventContext(raxaul, ts(2, 41), "truck"),
            (
                fired(RULE_OUT_OF_HOURS, {"local_time": "02:41:07"}, "outside sanctioned hours"),
                fired(
                    RULE_WRONG_DIRECTION,
                    {"heading_deg": 268.0, "angle_from_baseline_deg": 176.0},
                    "against the learnt flow",
                ),
            ),
        ),
        JudgeInput(
            EventContext(panitanki, ts(1, 12), "person"),
            (
                fired(
                    RULE_FENCE_CROSSED,
                    {"fence_name": "Channel", "direction": "in"},
                    "crossed fence 'Channel' inbound",
                ),
                fired(
                    RULE_GROUP_FORMED,
                    {"track_ids": [3, 4, 5, 6, 7, 8], "size": 6, "duration_s": 48.2},
                    "group of 6 held together",
                ),
            ),
        ),
        JudgeInput(
            EventContext(galgalia, ts(23, 7), "person"),
            (fired(RULE_LOITERING, {"dwell_s": 42.3, "net_displacement_m": 1.2, "unit": "m"}, "loitered 42.3s"),),
        ),
    ]
    return events


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass
class Sample:
    event_index: int
    frame_index: int
    latency_ms: int
    source: str  # "vlm" | "template"
    violations: tuple[str, ...]
    text: str
    raw: str | None = None


@dataclass
class LatencyReport:
    label: str
    measured_at_utc: str
    hardware: str
    machine: str
    device: str | None
    library_versions: dict[str, str]
    model_id: str
    model_revision: str
    frame_source: str
    frame_shape: str
    max_tokens: int
    timeout_s: float
    quantized: bool
    load_time_s: float | None
    warm_up_ok: bool
    n_samples: int
    n_timed: int                # samples whose describe() returned (the percentiles are over these)
    n_failed: int               # timeouts, load failures, runtime errors: reported, not timed
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    accepted: int
    rejected: int
    rejection_rate: float
    samples: list[Sample]
    sample_sentences: list[dict[str, Any]]


_FAILURE_PREFIXES = ("timeout", "unavailable", "runtime_error", "no_runtime", "prompt_error")


def _hardware_label() -> str:
    import platform
    import subprocess

    if sys.platform == "darwin":
        try:
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                   timeout=5).stdout.strip()
            mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                     timeout=5).stdout.strip())
            if brand:
                return f"{brand}, {mem // 2**30} GB unified memory"
        except Exception:  # noqa: BLE001
            pass
    return f"{platform.processor() or platform.machine()}"


def _platform_label() -> str:
    import platform

    return f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("torch", "transformers", "numpy", "opencv-python", "Pillow"):
        try:
            from importlib.metadata import version

            out[name] = version(name)
        except Exception:  # noqa: BLE001
            try:
                from importlib.metadata import version

                out[name] = version(name + "-headless")
            except Exception:  # noqa: BLE001
                continue
    return out


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), p)) if values else 0.0


def run(n_repeats: int, llvip_zip: Path | None, timeout_s: float, *, clips: list[Path] | None = None,
        device: str | None = None, label: str = "", max_tokens: int = DEFAULT_MAX_TOKENS,
        attempt_quantization: bool = True, frame_max_side: int = 768) -> LatencyReport:
    from pipeline.explain import downscale
    from vlm.runtime import MODEL_ID, PINNED_REVISION, MoondreamRuntime

    events = _sample_events()
    frames, frame_source = load_frames(max(n_repeats, len(events)), llvip_zip, clips)
    if not frames:
        raise RuntimeError("no frames available (no clip, LLVIP zip or synthetic frame generation produced any)")
    # the same downscale the live service applies before a frame waits for the model (pipeline/explain.py)
    frames = [downscale(f, frame_max_side) for f in frames]

    runtime = MoondreamRuntime(device=device, attempt_quantization=attempt_quantization)
    log.info("warming up moondream2 (excluded from the measured sample)...")
    warm_ok = runtime.warm_up(timeout_s=max(60.0, timeout_s))

    samples: list[Sample] = []
    accepted_sentences: list[dict[str, Any]] = []
    for i in range(n_repeats):
        event = events[i % len(events)]
        frame = frames[i % len(frames)]
        event_with_frames = JudgeInput(event.context, event.fired, frames=(frame,))
        result = explain(event_with_frames, runtime, timeout_s=timeout_s, max_tokens=max_tokens)
        samples.append(
            Sample(
                event_index=i % len(events),
                frame_index=i % len(frames),
                latency_ms=result.latency_ms or 0,
                source=result.source,
                violations=result.violations,
                text=result.text,
                raw=result.raw,
            )
        )
        if result.source == "vlm" and len(accepted_sentences) < 6:
            accepted_sentences.append(
                {
                    "text": result.text,
                    "raw_model_output": result.raw,
                    "latency_ms": result.latency_ms,
                    "camera": event.context.camera.name,
                    "rules_fired": [f.rule_name for f in event.fired],
                }
            )
        log.info("sample %d/%d: source=%s latency=%sms text=%r",
                 i + 1, n_repeats, result.source, result.latency_ms, result.text)

    failed = [s for s in samples if any(v.startswith(_FAILURE_PREFIXES) for v in s.violations)]
    timed = [float(s.latency_ms) for s in samples if s not in failed and s.latency_ms]
    accepted = sum(1 for s in samples if s.source == "vlm")

    return LatencyReport(
        label=label or (runtime.device or "unknown"),
        measured_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        hardware=_hardware_label(),
        machine=_platform_label(),
        device=runtime.device,
        library_versions=_library_versions(),
        model_id=MODEL_ID,
        model_revision=PINNED_REVISION,
        frame_source=frame_source,
        frame_shape="x".join(str(v) for v in frames[0].shape[:2][::-1]),
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        quantized=runtime.quantized,
        load_time_s=round(runtime._load_time_s, 2) if runtime._load_time_s else None,
        warm_up_ok=warm_ok,
        n_samples=len(samples),
        n_timed=len(timed),
        n_failed=len(failed),
        p50_ms=round(_percentile(timed, 50), 1),
        p95_ms=round(_percentile(timed, 95), 1),
        min_ms=min(timed) if timed else 0.0,
        max_ms=max(timed) if timed else 0.0,
        mean_ms=round(float(statistics.fmean(timed)), 1) if timed else 0.0,
        accepted=accepted,
        rejected=len(samples) - accepted,
        rejection_rate=round((len(samples) - accepted) / len(samples), 3) if samples else 0.0,
        samples=samples,
        sample_sentences=accepted_sentences,
    )


# --------------------------------------------------------------------------
# Report writers
# --------------------------------------------------------------------------


def write_json(report: LatencyReport, path: Path) -> None:
    payload = asdict(report)
    payload["samples"] = [asdict(s) for s in report.samples]
    payload["note"] = (
        "MEASURED on the hardware named above, with this repository's code and the pinned model revision. "
        "The PDF's ~3 s is a Jetson-class target, not a measurement of this machine."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=24, help="number of explain() calls to measure")
    parser.add_argument("--clips", type=Path, nargs="*", default=None,
                        help="video files or directories of .mp4 to take real frames from")
    parser.add_argument("--llvip-zip", type=Path, default=DEFAULT_LLVIP_ZIP)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default=None,
                        help="default: VLM_DEVICE, else auto (MPS where available)")
    parser.add_argument("--label", default="", help="name for the run, used in the output file name")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-quantize", action="store_true", help="skip the int8 attempt on CPU")
    # A slow device must be measured, not cut short into a column of timeouts, so this is generous.
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)

    report = run(args.repeats, args.llvip_zip, args.timeout_s, clips=args.clips, device=args.device,
                 label=args.label, max_tokens=args.max_tokens, attempt_quantization=not args.no_quantize)
    out = args.out_dir / f"moondream2_{report.label}.json"
    write_json(report, out)

    print(
        f"\n{report.hardware} device={report.device} n={report.n_samples} (timed {report.n_timed}, "
        f"failed {report.n_failed}) p50={report.p50_ms:.0f}ms p95={report.p95_ms:.0f}ms "
        f"accepted={report.accepted}/{report.n_samples} quantized={report.quantized} "
        f"frames={report.frame_source}\nwrote {out.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
