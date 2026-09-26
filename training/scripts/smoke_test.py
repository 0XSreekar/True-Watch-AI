#!/usr/bin/env python3
"""End-to-end preflight: run the real scripts on a tiny synthetic dataset before spending GPU hours.

Every step is the same command line a real run uses (train, kill and resume, evaluate, hard set,
sweep, export with the parity check, CPU benchmark, metrics report). A path error, a version
mismatch or a broken resume shows up here in a few minutes instead of at hour eight.

Two of the steps prove the checkpoint story rather than assume it:

  * KILL AND RESUME. Training is started in a child process and killed with SIGKILL once the
    second epoch is in train_log.csv AND weights/last_good.pt carries that epoch (the epoch index
    stored inside the checkpoint is polled, not a fixed sleep); a second invocation with --resume must continue with no gap
    and no duplicate epoch in train_log.csv.
  * TORN CHECKPOINT. Ultralytics writes last.pt with a plain write, so a session killed mid-write
    leaves a truncated file. A second run is killed the same way, last.pt is then truncated by
    hand, and --resume must fall back to last_good.pt and still finish.

Nothing measured here means anything: the images are coloured rectangles. Output goes under
--workdir and never into training/results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
import _synth  # noqa: E402

SCRIPTS = C.SCRIPTS_DIR


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str, float]] = []

    def add(self, name: str, ok: bool, detail: str, seconds: float) -> None:
        self.rows.append((name, ok, detail, seconds))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({seconds:.1f}s)  {detail}", flush=True)

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _, _ in self.rows)

    def table(self) -> str:
        width = max(len(n) for n, *_ in self.rows)
        lines = [f"{'step'.ljust(width)}  result  seconds  detail"]
        for name, ok, detail, seconds in self.rows:
            lines.append(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL':<6}  {seconds:7.1f}  {detail}")
        return "\n".join(lines)


def run(args: list[str], log: Path, env: dict | None = None) -> tuple[int, str]:
    """Run a script, tee its output to `log`, return (exit code, output)."""
    cmd = [sys.executable, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    out = proc.stdout + proc.stderr
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ {' '.join(cmd)}\n{out}", encoding="utf-8")
    return proc.returncode, out


def train_args(a, data: Path, run_dir: Path, epochs: int, stage: str = "day", extra: list[str] | None = None) -> list[str]:
    base = [
        str(SCRIPTS / "train.py"), "--stage", stage, "--data-root", str(data), "--run-dir", str(run_dir),
        "--imgsz", str(a.imgsz), "--epochs", str(epochs), "--batch", "8", "--workers", "0", "--device", a.device,
    ]
    if stage == "day":
        base += ["--model", a.model, "--unfreeze-epoch", "1"]
    return base + (extra or [])


def epochs_in_log(run_dir: Path) -> list[int]:
    path = run_dir / "train_log.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [int(float(row["epoch"])) for row in csv.DictReader(fh) if row.get("epoch")]


def checkpoint_epoch(path: Path) -> int | None:
    """The 0-based epoch index stored in an Ultralytics checkpoint, or None if it is missing or unreadable.

    last_good.pt is replaced atomically (os.replace), so a read sees either the old or the new file,
    never half of one. Loading needs torch and Ultralytics, which this preflight already requires.
    """
    if not path.exists():
        return None
    try:
        import torch

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
        return int(epoch) if epoch is not None else None
    except Exception:
        return None


def wait_for_last_good(run_dir: Path, epochs_wanted: int, deadline: float, proc: subprocess.Popen) -> bool:
    """Poll until weights/last_good.pt holds epoch `epochs_wanted` (1-based) or later.

    The epoch marker inside the checkpoint is what is waited on, not a fixed sleep and not the order in
    which train.py happens to write the log row and the copy: on a slow disk a fixed pause can kill the
    run before last_good.pt of the second epoch exists, and the resume would then be tested against an
    older checkpoint than the log claims. The file is only re-read when its mtime changes.
    """
    last_good = run_dir / "weights" / "last_good.pt"
    seen_mtime = None
    while time.time() < deadline and proc.poll() is None:
        try:
            mtime = last_good.stat().st_mtime_ns
        except FileNotFoundError:
            mtime = None
        if mtime is not None and mtime != seen_mtime:
            seen_mtime = mtime
            epoch = checkpoint_epoch(last_good)
            if epoch is not None and epoch + 1 >= epochs_wanted:
                return True
        time.sleep(0.2)
    return False


def kill_after_epochs(cmd: list[str], run_dir: Path, epochs_wanted: int, log: Path, timeout: float) -> tuple[bool, list[int]]:
    """Start training, SIGKILL it once `epochs_wanted` epochs are logged AND last_good.pt holds them.

    Returns (killed, epochs). `killed` is False when training ended, or the timeout passed, before
    both conditions held, so the caller's check fails instead of testing a resume from the wrong point.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen([sys.executable, *cmd], stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline and proc.poll() is None:
            if len(epochs_in_log(run_dir)) >= epochs_wanted:
                ready = wait_for_last_good(run_dir, epochs_wanted, deadline, proc)
                if proc.poll() is not None:
                    break  # training finished on its own: nothing was killed
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return ready, epochs_in_log(run_dir)
            time.sleep(0.5)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        return False, epochs_in_log(run_dir)


def contiguous(epochs: list[int], expected_last: int) -> bool:
    return sorted(epochs) == list(range(1, expected_last + 1)) and len(set(epochs)) == len(epochs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--workdir", default=None, help="scratch directory (default: a temp dir, removed on success)")
    ap.add_argument("--device", default="cpu", help="cpu, mps or a CUDA index such as 0")
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=4, help="epochs in the kill-and-resume run (needs >= 3)")
    ap.add_argument("--model", default=None, help="COCO weights to start from (default: training/weights/yolo11n.pt, else yolo11n.pt)")
    ap.add_argument("--quick", action="store_true", help="skip the torn-checkpoint run and the stage 2 run")
    ap.add_argument("--keep", action="store_true", help="keep the workdir even when everything passes")
    a = ap.parse_args()
    if a.epochs < 3:
        raise SystemExit("--epochs must be at least 3 so there is something to resume")
    local = C.TRAINING_ROOT / "weights" / "yolo11n.pt"
    a.model = a.model or (str(local) if local.exists() else "yolo11n.pt")

    tmp_made = a.workdir is None
    work = Path(a.workdir) if a.workdir else Path(subprocess.check_output(["mktemp", "-d"], text=True).strip())
    work.mkdir(parents=True, exist_ok=True)
    logs = work / "logs"
    res = work / "results"
    data = work / "data"
    rep = Report()
    print(f"workdir: {work}\ndevice: {a.device}  imgsz: {a.imgsz}  init weights: {a.model}\n", flush=True)

    # ---- 1. synthetic dataset
    t = time.time()
    summary = _synth.make_dataset(data, n_train=64, n_val=32, seed=0, size=(320, 256))
    rep.add("synthetic dataset", all(v > 0 for v in summary.values()) and len(summary) == 4, json.dumps(summary), time.time() - t)

    # ---- 2. training run A: killed after 2 epochs, then resumed
    run_a = work / "runs" / "a" / "day"
    t = time.time()
    killed, seen = kill_after_epochs(train_args(a, data, run_a, a.epochs), run_a, 2, logs / "train_a_killed.log", timeout=1800)
    rep.add("train: killed mid-run (SIGKILL)", killed and len(seen) >= 2 and (run_a / "weights" / "last.pt").exists(),
            f"epochs logged before the kill: {seen}", time.time() - t)

    t = time.time()
    code, out = run(train_args(a, data, run_a, a.epochs, extra=["--resume"]), logs / "train_a_resumed.log")
    done = epochs_in_log(run_a)
    state = C.read_json(run_a / "run_state.json", {})
    rep.add("train: --resume continues, no gap, no duplicate",
            code == 0 and contiguous(done, a.epochs) and bool(state.get("complete")) and "Resuming training" in out,
            f"exit {code}, epochs in train_log.csv: {done}, complete={state.get('complete')}", time.time() - t)
    best = run_a / "weights" / "best.pt"

    # ---- 3. torn checkpoint
    if not a.quick:
        run_b = work / "runs" / "b" / "day"
        t = time.time()
        killed, seen = kill_after_epochs(train_args(a, data, run_b, a.epochs), run_b, 2, logs / "train_b_killed.log", timeout=1800)
        last = run_b / "weights" / "last.pt"
        torn = False
        if killed and last.exists():
            raw = last.read_bytes()
            last.write_bytes(raw[: len(raw) // 2])  # what a kill mid-write leaves behind
            torn = True
        code, out = run(train_args(a, data, run_b, a.epochs, extra=["--resume"]), logs / "train_b_resumed.log")
        done = epochs_in_log(run_b)
        rep.add("train: torn last.pt falls back to last_good.pt",
                torn and code == 0 and contiguous(done, a.epochs) and (run_b / "weights" / "last_good.pt").exists(),
                f"exit {code}, epochs: {done}", time.time() - t)

    # ---- 4. stage 2 starts from stage 1's best weights
    weights = best
    if not a.quick and best.exists():
        t = time.time()
        run_ir = work / "runs" / "a" / "ir"
        code, out = run(train_args(a, data, run_ir, 2, stage="ir"), logs / "train_ir.log")
        ir_state = C.read_json(run_ir / "run_state.json", {})
        rep.add("train: stage 2 (ir) from stage 1 best.pt", code == 0 and bool(ir_state.get("complete")) and (run_ir / "weights" / "best.pt").exists(),
                f"exit {code}, complete={ir_state.get('complete')}", time.time() - t)
        if (run_ir / "weights" / "best.pt").exists():
            weights = run_ir / "weights" / "best.pt"

    if not weights.exists():
        rep.add("weights available for the remaining steps", False, "no best.pt was produced", 0.0)
        print("\n" + rep.table())
        return 1

    common = ["--data-root", str(data), "--imgsz", str(a.imgsz), "--device", a.device, "--out-dir", str(res)]

    # ---- 5. evaluate: day and ir separately
    t = time.time()
    code, out = run([str(SCRIPTS / "evaluate.py"), "--weights", str(weights), "--tag", "smoke", "--batch", "8", "--bootstrap", "20", *common], logs / "evaluate.log")
    ev = C.read_json(res / "eval_smoke.json", {})
    slices = ev.get("slices", {})
    allowed = {"day", "ir", "day/daylight", "day/night", "day/unresolved"}
    rep.add("evaluate: day and ir reported separately, nothing blended",
            code == 0 and {"day", "ir"} <= set(slices) and set(slices) <= allowed
            and slices["day"].get("n_images", 0) > 0 and slices["ir"].get("n_images", 0) > 0,
            f"exit {code}, slices: {sorted(slices)}", time.time() - t)
    sizes = slices.get("day", {}).get("size_buckets", {}).get("person", [])
    rep.add("evaluate: size buckets are exactly 77/54/38/27/19/14/9",
            [r["bucket_px"] for r in sizes] == [77, 54, 38, 27, 19, 14, 9], str([r.get("bucket_px") for r in sizes]), 0.0)

    # ---- 6. hard set: baseline, then the gate against itself
    t = time.time()
    hs_args = [str(SCRIPTS / "eval_hardset.py"), "--weights", str(weights), "--tag", "smoke", "--batch", "8",
               "--hard-set", str(data / "hard_set.txt"), "--index", str(data / "index" / "split.jsonl"),
               "--baseline", str(res / "hardset_baseline.json"), *common]
    code1, _ = run(hs_args + ["--write-baseline"], logs / "hardset_baseline.log")
    code2, _ = run(hs_args, logs / "hardset_gate.log")
    hs = C.read_json(res / "hardset_smoke.json", {})
    rep.add("hard set: baseline written, gate passes against itself",
            code1 == 0 and code2 == 0 and hs.get("gate", {}).get("passed") is True,
            f"exits {code1}/{code2}, passed={hs.get('gate', {}).get('passed')}", time.time() - t)

    # ---- 7. sweep
    t = time.time()
    code, out = run([str(SCRIPTS / "sweep_conf.py"), "--preds", str(res / "cache" / "preds_smoke.npz"), "--tag", "smoke",
                     "--data-root", str(data), "--out-dir", str(res)], logs / "sweep.log")
    sw = C.read_json(res / "sweep_smoke.json", {})
    both = all(k in sw.get("slices", {}) for k in ("day", "ir"))
    two_points = both and all("f1_optimal" in s and "false_alert_budget" in s for s in sw["slices"].values())
    rep.add("sweep: two separate operating points per modality", code == 0 and two_points, f"exit {code}", time.time() - t)

    # ---- 8. export and parity
    t = time.time()
    wdir = work / "onnx"
    code, out = run([str(SCRIPTS / "export_onnx.py"), "--weights", str(weights), "--imgsz", str(a.imgsz), "--tag", "smoke",
                     "--out", str(wdir), "--results-dir", str(res)], logs / "export.log")
    parity = re.search(r"PARITY raw_max_abs_diff=(\S+) gated_max_abs_diff=(\S+) gated_box_units=\S+ tolerance=\S+ -> (PASS|FAIL)", out)
    onnx_file = next(iter(sorted(wdir.glob("*.onnx"))), None)
    rep.add("export: torch vs onnxruntime parity printed and passing",
            code == 0 and bool(parity) and parity.group(3) == "PASS" and onnx_file is not None,
            parity.group(0) if parity else f"exit {code}, no PARITY line", time.time() - t)

    # ---- 9. benchmark (a handful of runs; the point is that it runs)
    if onnx_file is not None:
        t = time.time()
        code, out = run([str(SCRIPTS / "benchmark_cpu.py"), "--model", str(onnx_file), "--imgsz", str(a.imgsz), "--runs", "20",
                         "--warmup", "3", "--label", "smoke", "--weights-kind", "trained", "--out", str(res / "benchmark_smoke.json")],
                        logs / "benchmark.log")
        bench = C.read_json(res / "benchmark_smoke.json", {})
        ms = bench.get("inference_ms", {})
        rep.add("benchmark: p50 and p95 reported", code == 0 and "p50" in ms and "p95" in ms,
                f"p50={C.fmt(ms.get('p50'), 1)} ms, p95={C.fmt(ms.get('p95'), 1)} ms (smoke model, imgsz {a.imgsz})", time.time() - t)

    # ---- 10. metrics report
    t = time.time()
    md = work / "METRICS.smoke.md"
    cmd = [str(SCRIPTS / "make_metrics.py"), "--eval", str(res / "eval_smoke.json"), "--out", str(md)]
    for flag, name in (("--hardset", "hardset_smoke.json"), ("--sweep", "sweep_smoke.json"), ("--export", "export_smoke.json")):
        if (res / name).exists():
            cmd += [flag, str(res / name)]
    if (res / "benchmark_smoke.json").exists():
        cmd += ["--benchmark", str(res / "benchmark_smoke.json")]
    if (run_a / "train_log.csv").exists():
        cmd += ["--train-log", str(run_a / "train_log.csv")]
    code, out = run(cmd, logs / "metrics.log")
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    rep.add("metrics: report renders with a verdict per target", code == 0 and "NOT ASSESSED" in text and ("MET" in text),
            f"exit {code}, {len(text)} characters", time.time() - t)

    print("\n" + rep.table())
    print("\nThese numbers come from coloured rectangles and mean nothing. The point of this run is that every step executes.")
    if rep.ok and tmp_made and not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"logs and outputs kept in {work}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
