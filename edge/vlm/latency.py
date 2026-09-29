"""Measure and record local Moondream 2 latency, honestly, on the machine
that runs this.

This is a script, not a test: it loads the REAL model (no stub runtime),
warms it up once (excluded from the measured sample — a cold load is a
one-time service-boot cost, not a per-alert one), then times a realistic
number of `judge.explain` calls end to end (prompt build + `describe()` +
guardrail check) against frames that look like what the pipeline actually
hands the judge. It writes both a human-readable `LATENCY.md` and a
machine-readable `latency.json` next to this file, and prints a short
summary to stdout.

Frame source
------------
Real LLIVIP thermal/RGB pairs when the dataset zip named by
`--llvip-zip` (default: the path this repo's build plan names,
`var/datasets/llvip/LLVIP.zip`) is present and readable; otherwise
synthetic frames built the same way `edge/tests/` builds its own sample
clips (an OpenCV-drawn scene), clearly labelled as synthetic in the output
so a reader never mistakes a synthetic-frame number for one measured on
real footage. Either way every frame actually passes through the model —
nothing here is a mocked or estimated number.

Reading these numbers
----------------------
The PDF's "~3 s" is explicitly a Jetson-class edge-device target, not a
promise about a development laptop; `edge/README.md` says so next to
these figures, and so does the generated LATENCY.md. p50/p95 measured here
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
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm.judge import JudgeInput, explain, fallback_reason  # noqa: E402
from vlm.prompt import CameraContext, EventContext  # noqa: E402
from vlm import guardrails  # noqa: E402
from vlm.prompt import build_facts  # noqa: E402

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


def load_frames(count: int, llvip_zip: Path | None) -> tuple[list[np.ndarray], str]:
    """Return `(frames, source_label)`. `source_label` is "llvip" or "synthetic"."""
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
    latency_ms: int
    source: str  # "vlm" | "template"
    violations: tuple[str, ...]
    text: str


@dataclass
class LatencyReport:
    measured_at_utc: str
    machine: str
    frame_source: str
    quantized: bool
    load_time_s: float | None
    warm_up_ok: bool
    n_samples: int
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


def _platform_label() -> str:
    import platform

    return f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"


def run(n_repeats: int, llvip_zip: Path | None, timeout_s: float) -> LatencyReport:
    from vlm.runtime import MoondreamRuntime

    events = _sample_events()
    frames, frame_source = load_frames(max(n_repeats, len(events)), llvip_zip)
    if not frames:
        raise RuntimeError("no frames available (neither LLVIP nor synthetic frame generation produced any)")

    runtime = MoondreamRuntime()
    log.info("warming up moondream2 (excluded from the measured sample)...")
    warm_ok = runtime.warm_up(timeout_s=max(60.0, timeout_s))

    samples: list[Sample] = []
    accepted_sentences: list[dict[str, Any]] = []
    for i in range(n_repeats):
        event = events[i % len(events)]
        event_with_frames = JudgeInput(event.context, event.fired, frames=(frames[i % len(frames)],))
        result = explain(event_with_frames, runtime, timeout_s=timeout_s)
        samples.append(
            Sample(
                event_index=i % len(events),
                latency_ms=result.latency_ms or 0,
                source=result.source,
                violations=result.violations,
                text=result.text,
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
        log.info(
            "sample %d/%d: source=%s latency=%sms text=%r",
            i + 1, n_repeats, result.source, result.latency_ms, result.text,
        )

    latencies = [s.latency_ms for s in samples if s.latency_ms]
    accepted = sum(1 for s in samples if s.source == "vlm")
    rejected = sum(1 for s in samples if s.source == "template" and s.violations)
    sorted_lat = sorted(latencies) or [0]

    def pct(p: float) -> float:
        if not sorted_lat:
            return 0.0
        idx = min(len(sorted_lat) - 1, int(round(p / 100 * (len(sorted_lat) - 1))))
        return float(sorted_lat[idx])

    return LatencyReport(
        measured_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        machine=_platform_label(),
        frame_source=frame_source,
        quantized=runtime.quantized,
        load_time_s=runtime._load_time_s,
        warm_up_ok=warm_ok,
        n_samples=len(samples),
        p50_ms=pct(50),
        p95_ms=pct(95),
        min_ms=float(min(sorted_lat)),
        max_ms=float(max(sorted_lat)),
        mean_ms=float(statistics.fmean(latencies)) if latencies else 0.0,
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
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown(report: LatencyReport, path: Path) -> None:
    lines = [
        "# Moondream 2 latency — measured, not estimated",
        "",
        f"Measured {report.measured_at_utc} on {report.machine}.",
        "",
        "> The PDF's ~3 s VLM figure is a **Jetson-class edge-device target**, not a claim about "
        "this development machine — see edge/README.md. The numbers below are what local Moondream 2 "
        "actually takes on the hardware this was measured on, CPU-only, no shortcuts.",
        "",
        f"- Frame source: **{report.frame_source}**"
        + (" (real LLVIP thermal/RGB frames)" if report.frame_source == "llvip" else " (synthetic — LLVIP zip not available on this box)"),
        f"- Quantised (dynamic int8 linear layers): **{report.quantized}**"
        + ("" if report.quantized else " — running the checkpoint's default bf16 weights on CPU instead; see vlm/runtime.py"),
        f"- Model load time (one-time, service boot): **{report.load_time_s:.1f}s**" if report.load_time_s else "- Model load time: not recorded",
        f"- Warm-up succeeded: **{report.warm_up_ok}**",
        f"- Samples measured: **{report.n_samples}**",
        "",
        "## Latency (ms, `describe()` wall time, per `judge.Explanation.latency_ms`)",
        "",
        "| Metric | ms |",
        "|---|---|",
        f"| p50 | {report.p50_ms:.0f} |",
        f"| p95 | {report.p95_ms:.0f} |",
        f"| min | {report.min_ms:.0f} |",
        f"| max | {report.max_ms:.0f} |",
        f"| mean | {report.mean_ms:.0f} |",
        "",
        "## Guardrail acceptance",
        "",
        f"- Accepted (VLM sentence used): **{report.accepted}** / {report.n_samples}",
        f"- Rejected (fell back to the rule string): **{report.rejected}** / {report.n_samples} "
        f"(rejection rate **{report.rejection_rate:.0%}**)",
        "",
        "A small CPU model rejecting some fraction of its own outputs is expected and reported here "
        "honestly, not hidden — the rule-string fallback exists exactly for this, and the alert's "
        "`reason` is never empty either way (see `vlm/judge.py`).",
        "",
        "## Sample sentences that passed guardrails",
        "",
    ]
    if report.sample_sentences:
        for i, s in enumerate(report.sample_sentences[:3], start=1):
            lines.append(f"{i}. **{s['text']}**")
            lines.append(f"   (camera: {s['camera']}, rules fired: {', '.join(s['rules_fired'])}, latency: {s['latency_ms']} ms)")
    else:
        lines.append(
            "None of the sampled runs produced a guardrail-passing VLM sentence on this machine "
            "in this run — every alert nonetheless carried the rule-string fallback reason (never empty). "
            "See the raw samples in `latency.json` for the rejected outputs and why each was rejected."
        )
    lines.append("")
    lines.append("## Raw samples")
    lines.append("")
    lines.append("| # | event | source | latency (ms) | violations | text |")
    lines.append("|---|---|---|---|---|---|")
    for s in report.samples:
        violations = ", ".join(s.violations) or "-"
        text = s.text.replace("|", "\\|")
        lines.append(f"| {s.event_index} | event {s.event_index} | {s.source} | {s.latency_ms} | {violations} | {text} |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=8, help="number of describe() calls to measure")
    parser.add_argument("--llvip-zip", type=Path, default=DEFAULT_LLVIP_ZIP)
    # CPU bf16 decode for a ~1.9B model, without a hardware bf16 unit, is genuinely
    # this slow (measured ~120s/call on this development machine, see LATENCY.md) —
    # the whole point of this script is to report that honestly rather than cut the
    # timeout short and record every sample as a timeout fallback.
    parser.add_argument("--timeout-s", type=float, default=200.0)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)

    report = run(args.repeats, args.llvip_zip, args.timeout_s)
    write_json(report, args.out_dir / "latency.json")
    write_markdown(report, args.out_dir / "LATENCY.md")

    print(
        f"\np50={report.p50_ms:.0f}ms p95={report.p95_ms:.0f}ms "
        f"accepted={report.accepted}/{report.n_samples} "
        f"rejection_rate={report.rejection_rate:.0%} "
        f"quantized={report.quantized} frame_source={report.frame_source}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
