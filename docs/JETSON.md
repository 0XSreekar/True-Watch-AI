# Jetson measurement kit

Status: **ready to run, never run.** The project does not own a Jetson. Until a board has produced a result file
with this kit, the wording for every latency claim is:

> design target, not measured on hardware

That covers slide 3's "about 30 ms detector inference (YOLO11-s INT8 budget at 640 px, not end-to-end)" and
"VLM about 3 s". The Mac numbers in `training/results/METRICS.md` section 9 and `edge/vlm/results/` are not
evidence about the board in either direction.

## What the kit measures

| Question | Tool | Output |
|---|---|---|
| Detector GPU latency, FP16 and INT8, 640x640, batch 1 | `benchmark.py` stage `trt` (trtexec) | p50 / p95 / p99, warm-up excluded |
| Whole-frame latency, 1080p, tiling off / strip / full | `benchmark.py` stage `e2e` | p50 / p95 / p99 per frame, detector p50, fps |
| Does it hold up for 10 minutes | `benchmark.py` stage `sustained` | fps per 60 s window, temperatures |
| VLM reason time | `benchmark.py` stage `vlm` (runs `edge/vlm/latency.py`) | p50 / p95 |
| What INT8 costs in accuracy | `accuracy_int8.py` | day and IR person AP50, INT8 vs FP32 ONNX, paired delta |

All device tools live in `edge/tools/jetson/` and import no Ultralytics. The record carries a hardware
fingerprint (module, JetPack and L4T, TensorRT, CUDA, `nvpmodel` mode, whether `jetson_clocks` pinned the clocks)
and `measured_on_jetson`, which is false on any host that is not a Jetson; the markdown then opens with a banner.

## 0. Hardware and software

- Jetson Orin Nano Super Developer Kit (8 GB). Adequate power supply and active cooling.
- JetPack 6.2 or newer (L4T 36.4+), which carries TensorRT 10.x, CUDA 12.6 and the Super power mode. Flash with
  NVIDIA SDK Manager, or write the JetPack 6.2 SD card image. Record the version that results; the benchmark does.
- NVMe or a fast SD card. Engine builds and the 1080p clip need a few GB.

## 1. Host (Mac or Linux): build the calibration set

Needs the Phase 1 dataset root (`images/val`, `labels/val`). Output goes OUTSIDE the repository (the script refuses
a path inside any git repo).

```bash
python edge/tools/jetson/build_calib_set.py --data-root <dataset root> --out-dir ~/truewatch-jetson
```

This writes `calib_list.txt`, `calib_manifest.json` and `calib_500.tar.gz`: 500 val images, 250 visible ("day") and
250 LWIR ("ir"), round-robin over source datasets, deterministic for a given seed.

Licences: the manifest lists the licence string of every source used, copied from `datasets/config/sources.yaml`.
Most are `open_question` (the original publisher's terms govern and redistribution is assumed forbidden), so the
tarball is for calibrating on your own board. Do not publish it, attach it to a release or commit it.

Also copy to the board: the repository (or `edge/`, `training/results/hf_model.json`,
`training/scripts/_metrics.py`, `training/scripts/_common.py`), the published `final.onnx` (download from the
`resolve_url` in `hf_model.json`; the kit refuses any file whose sha256 differs from the manifest), the calibration
tarball plus list, a 1080p test clip, and a few val images plus labels for the accuracy check (the accuracy script
reads `images/val` and `labels/val` under `--data-root`, so copy the whole val folders or a subset that contains
the picked files).

## 2. Jetson: install

```bash
sudo apt install -y python3-pip python3-opencv python3-numpy
pip3 install pycuda onnxruntime pyyaml        # TensorRT comes with JetPack: python3 -c "import tensorrt"
pip3 install -r edge/requirements.txt         # whatever the pipeline needs; skip packages JetPack provides
tar xzf calib_500.tar.gz -C calib/            # mkdir calib first
```

`onnxruntime` (CPU) is only for the FP32 reference in the accuracy check. If the pip wheel is missing for your
Python, use the Jetson-AI-Lab wheel index.

## 3. Power mode and clocks (do this before any measurement)

```bash
sudo nvpmodel -m 2          # Orin Nano Super: MAXN_SUPER. Confirm with: sudo nvpmodel -q
sudo jetson_clocks          # pin clocks at their maximum. Confirm with: sudo jetson_clocks --show
```

Run the headline set twice if you want both numbers: once pinned at MAXN_SUPER and once at the default
(unpinned) mode the unit ships in. The benchmark records the mode and whether clocks are pinned, and warns when
they are not. Quote the mode next to every number.

## 4. Build engines

```bash
python3 edge/tools/jetson/build_engines.py --onnx final.onnx --out-dir engines \
    --calib-images calib/ --calib-list calib_list.txt
```

Writes `engines/final_fp16_640_b1.engine`, `engines/final_int8_640_b1.engine`, `calib.cache`, per-engine trtexec
logs and `build_info.json` (build times). `--dry-run` prints the trtexec commands without running them. INT8 is
built with `--fp16` also enabled so a layer with no INT8 kernel falls back to FP16.

Engines are specific to the GPU and TensorRT version that built them: rebuild after any JetPack update.

## 5. Benchmark

```bash
python3 edge/tools/jetson/benchmark.py --engines engines --clip clip_1080p.mp4 --out-dir results
```

Defaults: trtexec 30 s per engine after a 2 s warm-up; 600 frames per tiling mode at the edge default of 8 fps
ingest; sustained run of 600 s with tiling off; 24 VLM explanations. `--stages trt` alone is a 2-minute smoke
test. `--engine-for-pipeline fp16` runs the pipeline stages on the FP16 engine. VLM needs the Moondream 2 weights
in the local Hugging Face cache (it never downloads in `auto` mode); the VLM stage records an error entry rather
than a number if it cannot run, and the Mac MPS and CPU numbers in `edge/vlm/results/` stay what they are.

Results: `results/jetson_bench_<UTC stamp>.json` and `.md`.

What the stages mean, so nobody over-reads them:

- `trt` is GPU compute time of the engine for one 640x640 input. This is the number comparable to the "about 30 ms"
  budget. It excludes host-side letterbox, decode of the output and NMS, and it is not end-to-end.
- `e2e` is per processed frame: decode, then `CameraService.process` (detector on every second frame, motion on
  every frame, fusion, rules). It is therefore bimodal; read p50 and p95 together with `detector_p50_ms`. The
  unlabelled calibration warm-up frames are excluded and counted. The clip should be 1080p; a warning is recorded
  if it is not. `full` tiling runs 7 passes and `strip` 5 per detector frame at 1080p.
- `sustained` shows whether throughput decays; compare the first and last 60 s window and the temperatures.

## 6. INT8 accuracy

```bash
python3 edge/tools/jetson/accuracy_int8.py --data-root <dataset> --onnx final.onnx \
    --engine engines/final_int8_640_b1.engine --calib-list calib_list.txt --out-dir results
```

Scores the same 600 val images (300 day, 300 IR, excluding every calibration image) through the FP32 ONNX and the
INT8 engine with the same decoder, and reports person AP50 per slice, the INT8 minus FP32 delta and a paired
image-level bootstrap interval (optimistic: frames of one clip are correlated). The matching and AP code is
`training/scripts/_metrics.py`, the code behind `eval_final.json`. If IR loses more than day, say so; do not
average the slices.

## 7. Recording the results

1. Commit the `.json` and `.md` from `results/` as `edge/tools/jetson/results/` (small text files; check them for
   paths and nothing else sensitive; the tools already scrub home directories). Never commit engines, images or the
   calibration tarball.
2. `training/results/METRICS.md`: this file is generated by `training/scripts/make_metrics.py`. Add the measured row
   there (the "Detector latency" row in the target table and the latency table in section 9) with the board's
   fingerprint and the `trt` p50 / p95. Change the verdict only if a measurement exists; "about 30 ms" is met only
   if the INT8 `trt` p50 is at or below it at the recorded power mode, and say that it excludes pre/post-processing.
3. `docs/VERIFICATION_REPORT.md`: update claim 3 rows (detector latency, VLM about 3 s) from NOT MEASURED to the
   measured values with the mode and JetPack version, and drop the "Jetson not measured" wording.
4. Slide 3: replace the budget wording only with the measured figure and its caveats (INT8 GPU compute, 640 px,
   power mode). Keep "not end-to-end" unless the `e2e` rows support a statement.

## Pitfalls

- **Wrong power mode.** The kit's default after flashing may not be MAXN_SUPER. A number at 15 W is a 15 W number.
- **Thermal throttling.** Short runs flatter the board. The sustained stage exists to expose decay; check the
  temperatures and the first/last window fps. Passive cooling in a closed case will throttle.
- **First-run engine build is slow** (minutes per engine, longer with INT8 calibration). It is not inference time
  and is recorded separately. Run once, reuse engines.
- **Engine mismatch.** "could not deserialize" means the engine was built by another TensorRT version or GPU.
  Rebuild on the device.
- **Cold start.** The first inferences after load are slower; the kit excludes trtexec warm-up and the pipeline's
  calibration frames. Do not add a first-call timing to a claim.
- **Clocks drift back.** `jetson_clocks` does not survive a reboot; re-pin before each session.
- **INT8 calibration data.** Calibrating on images unlike the deployment (day only, say) costs IR accuracy. The
  set is half IR on purpose, and the accuracy check reports both slices.
- **DLA is not used.** Everything runs on the GPU; do not quote a DLA figure from this kit.
- **VLM memory.** Moondream 2 and the detector share 8 GB; the VLM stage runs after the detector stages in a
  separate process, so its timing is not contended by the pipeline. A live service would be.

## Tests

`edge/tests/test_jetson_kit.py` covers, on a development machine: calibration list determinism, command-line
construction, the trtexec log parser, percentile math, the result renderer and its not-a-Jetson banner, the
`nvpmodel` and `jetson_clocks` parsers, and the INT8-versus-FP32 scoring code. Its sample trtexec log and
`nvpmodel` / `jetson_clocks` texts are synthetic test fixtures written from the documented formats with artificial
values. Nothing that needs TensorRT, CUDA or the board (`trt_session.py`, `make_calib_cache.py`, the engine
builds, the benchmark stages) has been executed anywhere; expect to fix small environment problems on first run.
