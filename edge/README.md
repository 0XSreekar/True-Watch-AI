# TRUEWATCH — edge inference service

Everything that touches pixels lives here: decode, the appearance channel, the
motion channel, fusion, tracking, the rule engine, ANPR, face detection, the
plain-language explanation, and the SHA-256 hash taken at capture.

One post in the field runs one copy of this service against its own cameras.
It reaches the rest of the system through exactly one call —
`POST /api/ingest/event` on the Express backend, authenticated with a shared
secret. It never talks to the browser.

> **Phase 3.** The service shell, the ingest path, the appearance channel, the
> motion channel, ByteTrack, dual-channel fusion with per-camera calibration and
> camera-state anomalies are in (`pipeline/`). The rule engine, ANPR, face, the
> explanation model and evidence hashing arrive in later phases.

## Where this fits

See [`docs/ARCHITECTURE_V2.md`](../docs/ARCHITECTURE_V2.md):

- §1.2 maps each of the seven methodology steps on slide 3 of the submission
  deck to the module in this folder that implements it.
- §4 is the event contract this service posts to the backend, and the exact
  mapping into the console's frozen alert shape.
- §5 lists every file this folder will hold by Phase 12, and the phase that
  creates each one.
- §7 is the latency honesty budget. Read it before quoting any number.

## Local development

Python 3.11 or newer. The container pins 3.11; 3.12+ works locally.

```bash
cd edge
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # TRUEWATCH_INGEST_KEY may stay empty in development
```

Generate a clip to run against, then start the service:

```bash
python tools/make_sample_clip.py --out ./var/samples/sample.mp4
uvicorn app:app --reload --port 8000
```

```bash
curl localhost:8000/healthz          # what this instance is pointed at
curl localhost:8000/pipeline/probe   # open the source, read one frame, report it
curl -X POST localhost:8000/pipeline/start
curl localhost:8000/pipeline/status  # frames_seen climbs at TARGET_FPS
curl -X POST localhost:8000/pipeline/stop
```

Tests:

```bash
python -m pytest            # generates its own clips; no dataset needed
```

The root `npm run dev` starts the console and the API only. This service is
started separately, and the console works without it — `apiClient.withFallback`
keeps the operator console watchable when nothing upstream is running.

## Phase 3 pipeline: two channels, one decision

```
source.py -> camera_state.py -> motion.py (every frame) -+-> tracking.py (ByteTrack) -> fusion.py -> FusedEvent
                                appearance.py (every 2nd) -+          calibration.py (per camera) --^
```

| Module | What it does |
|---|---|
| `pipeline/types.py` | The frozen contract later phases build on: `Detection`, `Track`, `ChannelScore`, `FusedEvent`, `CameraStateEvent`. |
| `pipeline/source.py` | RTSP / file / image folder / webcam, monotonic timestamps, RTSP reconnect with backoff. The docstring has a mediamtx + ffmpeg loopback recipe for demonstrating an "existing IP camera" without one. |
| `pipeline/appearance.py` | YOLO11-s ONNX through onnxruntime (no Ultralytics), letterbox, class-aware NMS, class map by **name** (COCO and the fine-tuned head both map to person / two_wheeler / car / truck / cart), optional far-field tiling per camera. |
| `pipeline/motion.py` | Dense Farneback flow with pinned parameters at 640 px working width, per-box motion score from object-vs-local-background contrast, motion mask and regions, global-shift (shake) removal. |
| `pipeline/tracking.py` | ByteTrack implemented in-repo (scipy's Hungarian solver only), class-gated, Kalman prediction across skipped detector frames, dwell / net / path signals of `MEASUREMENTS.md` section 2. |
| `pipeline/fusion.py` | The agreement rule: channel floors, spatial gate (IoU of box and flow peak), weighted score against the per-camera threshold, temporal gate. Day/night weights derived from `MEASUREMENTS.md` section 1. |
| `pipeline/calibration.py` | Per-camera, per-modality motion thresholds learnt from an unlabelled warm-up, stored as JSON, resettable. |
| `pipeline/camera_state.py` | SIGNAL_LOSS, LENS_TAMPER and IR_FLOOD raised as CAMERA_STATE anomalies, never treated as darkness; fusion stands down while one is active. |
| `pipeline/runner.py` | One camera's per-frame loop; the detector runs concurrently with the flow of the same frame. |
| `pipeline/demo.py` | `python -m pipeline.demo`; prints channel scores, alerts, camera-state events, a day/night weighting comparison and timings. |
| `tools/reproduce_flow_contrast.py` | Re-runs the section 1 flow-contrast measurement on paired visible/LWIR frames. |
| `tools/reproduce_rule_sanity.py` | Re-runs the section 2 rule-sanity table (from the table, or live on a frame folder). |

### Running the demo

Media is never committed. Build the clips from a local KAIST copy (set00/V001 frames 1430-1864, paired
visible and LWIR, 20 fps):

```bash
cd edge
python -m pipeline.make_samples --kaist-root ../var/datasets/kaist/kaist_train --also-1080p
export YOLO_MODEL_URL=file:///abs/path/to/detector.onnx YOLO_MODEL_SHA256=<its sha256>
python -m pipeline.demo --source ../samples/day.mp4 --show-scores
python -m pipeline.demo --source ../samples/ir.mp4  --show-scores
python -m pipeline.demo --source ../samples/day.mp4 --inject-shake 250:300 --inject-black 340:370 \
       --inject-cover 375:400 --inject-flood 405:430          # synthetic shake and camera-state events
```

LLVIP is not used for the demo clips: its files are sampled seconds apart, so dense flow between
consecutive files is meaningless (`make_samples.py --from llvip` builds slideshow clips for appearance and
camera-state checks only). KAIST is recorded from a vehicle, so the demo clips contain ego-motion a pole
camera would not have; the per-camera calibration and the local-ring contrast absorb part of it.

### Measured throughput (not estimated)

Host: Apple M5, 10 cores, macOS, CPU only (onnxruntime CPUExecutionProvider, OpenCV 5.0), FP32 detector
at 640 px. Other jobs were running on the same machine during some runs (load average 2 to 6), which is
why repeated runs differ; the numbers below are the ones recorded, not the best case. The 1080p clips are
KAIST 640x512 frames upscaled to 1920x1080 by `make_samples.py --also-1080p`, so the decode and resize
cost of a 1080p stream is real but the scene detail is not 1080p detail.

| Clip | Detector stride | End-to-end FPS (decode + pipeline) | Mean ms/frame: motion / detector pass |
|---|---|---|---|
| day_1080p.mp4, 1920x1080 | 2 (default) | **16.5** (12.5 under heavier load) | 36 / 80 |
| ir_1080p.mp4, 1920x1080 | 2 (default) | **17.9** (12.8 under heavier load) | 37 / 70 |
| day_1080p.mp4, 1920x1080 | 1 (detector every frame) | **9.0** (8.2 under heavier load) | 51 / 107 |
| day.mp4, 640x512, with injections | 2 | 27.7 | 29 / 36 |

The slide 3 figure of 8 to 10 FPS at 1080p is a design target for a Jetson; on this Apple Silicon CPU the
pipeline measured 16.5 to 17.9 FPS at 1080p with the detector on every second frame and 9.0 FPS with it on
every frame. These are CPU numbers on this host only, not Jetson numbers, and not an end-to-end alert
latency.

Why the detector runs on every second frame while motion runs on every frame: Farneback measures
displacement between the two frames it is given, and the section 1 contrast was measured on consecutive
frames, so skipping frames for motion would change the quantity; the detector is the expensive stage and
its boxes move a few pixels per frame, which the tracker's Kalman prediction bridges. `pipeline/runner.py`
documents this, and the detector also runs in a worker thread concurrently with the flow of the same
frame.

### What the demo showed (interim COCO-pretrained YOLO11-s, not the fine-tuned model)

- **Appearance only, no alert:** on `day.mp4`, 197 candidate-frames had the detector vouching and motion
  not (e.g. track 7, a = 0.67, m = 0.00), and none raised an alert.
- **Motion only, no alert:** 246 frames had motion regions touching no candidate box; none can alert. A
  synthetic 4-10 px camera shake over frames 250-299 produced 140 candidate-frames and 0 alerts (median
  motion score 0.12).
- **Both agree, alert:** e.g. `ALERT track 7 person appearance=0.76 motion=0.64 fused=0.70 >= threshold
  0.55 (weights a 0.53 / m 0.47, day), iou 0.24, 3 consecutive frames`; on `ir.mp4`: `ALERT track 4
  person appearance=0.67 motion=1.00 fused=0.90 (weights a 0.30 / m 0.70, night)`.
- **Night weighting:** on the same candidate-frames, `ir.mp4` scores a mean fused 0.773 with the LWIR
  weights against 0.705 with the day weights; on `day.mp4` the day weights give 2 alerts and the night
  weights 1. On `ir.mp4` the alert count does not change (1 either way) because the COCO-pretrained
  detector finds few people on LWIR (14 person boxes against 109 on visible over the section 1
  reproduction below), so few night candidates exist to be rescued.
- **Calibration:** written to `edge/var/calibration/<camera>.<modality>.json` after 100 warm-up frames.
- **Camera state:** a black frame raised SIGNAL_LOSS, a blurred and darkened frame LENS_TAMPER, a
  washed-out frame IR_FLOOD; 0 detection alerts inside any of those windows.
- **Section 1 re-run** (`tools/reproduce_flow_contrast.py`, KAIST set00/V000 frames 1150-1250, not the
  V007 set of the record): visible 5.25 / 2.19 px = 2.40x, LWIR 2.21 / 0.45 px = 4.87x, whole-frame LWIR
  flow 24% of visible. Same direction as the record (1.9x to 3.4x, 32%), different sequence.
- **Section 2 re-run** (`tools/reproduce_rule_sanity.py`): every row of the published table, the 6/4
  baseline and the 6 alerts / 4 normal come out of the tracker's signal definitions exactly. Dwell is
  t_last - t_first: track 3's 33 frames printed as 1.6 s rules out counting both ends.

Known limit for Phase 8: `ARCHITECTURE_V2.md` section 4.3 maps each console chip as `score >= threshold`,
but fusion deliberately lets a strong channel carry a weak one past the per-channel floors, so an agreed
event can have one channel score below the fused threshold. The mapping should compare each channel
against its floor (`a_min`, `m_min`), or the event should carry those floors.

## The two sources take one decode path

A live ONVIF/RTSP pull and a replayed file differ only in the string handed to
OpenCV, so the substitution recorded in `docs/PHASE_MINUS1_SCOPE.md` section 6 is
a configuration change rather than a second code path. They differ in how the
frame rate is reduced, because they have different clocks:

- A **file** is decimated against its own declared rate — keep every Nth frame.
  Dropping by wall-clock would empty a short clip on a fast machine.
- A **live pull** drops any frame arriving inside the target interval, because on
  a live source a stale frame is worse than a missing one.

Every frame and every event carries `source` (`rtsp` or `file`), so nothing
downstream can present a replay as a live camera.

## Rules for this folder

- **No model weights in git.** The detector is downloaded at boot by
  `models/detector_weights.py` from the commit-pinned URL in
  `training/results/hf_model.json` (or `YOLO_MODEL_URL` + `YOLO_MODEL_SHA256`),
  over plain HTTPS with no token, and its SHA-256 is checked before the file is
  used. `models/.gitignore` keeps the downloaded copy out of git.
- **No Ultralytics here.** The serving path runs the exported ONNX graph with
  onnxruntime only; Ultralytics (AGPL-3.0) stays inside `training/`.
- **No secrets in git.** Every value comes from the environment. `.env.example`
  carries names and safe defaults only.
- **No dataset files in git.** `datasets/` holds fetch scripts, never data.
- **Honesty flags are not optional.** Every event carries `source`
  (`rtsp` or `file`) and `explanation.source` (`vlm` or `template`), so a
  replayed file is never presented as a live camera and template text is never
  presented as model output.
