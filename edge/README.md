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
| `tools/measure_false_alerts.py` | Alerts per camera per 24 h over event-free recordings, through the service path (see below). |
| `tools/measure_fence_misses.py` | Fence-crossing miss rate against ground-truth crossing times, Wilson 95% interval vs the 2% target. |

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
- **Calibration:** written to `edge/var/calibration/<camera>.<modality>.json` after 5 s of warm-up on the
  clip's own clock (at least 25 frames; `--warmup-seconds`, or `--warmup N` for a fixed frame count).
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

- A **file** is decimated on its own timeline: a frame is kept when its media time
  (decoded frames / declared fps) reaches the next 1/TARGET_FPS slot, so 20 fps to
  8 fps keeps 8 frames a second (2-3 source frames apart). Each frame carries that
  media time, and the pipeline uses it as its clock, paced or not. Dropping by
  wall-clock would empty a short clip on a fast machine.
- A **live pull** drops any frame arriving inside the target interval, because on
  a live source a stale frame is worse than a missing one.

Every frame and every event carries `source` (`rtsp` or `file`), so nothing
downstream can present a replay as a live camera.

## Frame rate

Fusion behaves the same at any ingest rate (docs/VERIFICATION_REPORT.md section 4.1 has the measured
table). Three things make that true:

- **Flow is px per reference interval (1/20 s), not px per frame.** `pipeline/motion.py` scales each
  pair's flow by `REFERENCE_INTERVAL_S / dt`, dt from the frame timestamps. At 20 fps the factor is 1, so
  the section 1 numbers and the Farneback parameters are untouched; at 8 fps a walker's 7.5 px/frame
  reads as 3 px per 1/20 s, the same as at 20 fps. A pair more than 1 s apart is a cut, not flow.
- **Calibration warm-up is 5 s of the camera's clock** (`CALIBRATION_WARMUP_SECONDS`, floor
  `CALIBRATION_WARMUP_MIN_FRAMES` = 25), not 100 frames (which was 12.5 s at 8 fps). Stored calibrations
  are schema v2; a v1 file (px per frame) is ignored and the camera re-learns.
- **The detector stride has a ceiling in seconds** (`DETECTOR_MAX_INTERVAL_S` = 0.125): below about
  16 fps the detector runs on every frame, because boxes 0.25 s apart move further than a small box is
  wide and the tracker cannot associate them.

`TARGET_FPS` defaults to 8: on the samples it gives the same alerts as the native 20 fps for 56-61% of
the compute per second of footage (the detector still runs 8 times a second, against 10 at 20 fps stride 2). The temporal gate stays 3 frames and the appearance
freshness window stays `stride` frames: both are counted in frames the detector actually ran on.

## Measuring the slide-5 field targets

Neither target can be measured without footage; the tools are ready and proven on generated input.

    # false alerts per camera per 24 h: event-free recordings, optionally CAMERA_ID=path
    python -m tools.measure_false_alerts CAM-A=/data/a_night.mp4 CAM-A=/data/a_day.mp4 \
        --rules-dir var/rules --json out/false_alerts.json
    # fence miss rate: recording + JSON list of true crossing times (media seconds) + the camera's fence
    python -m tools.measure_fence_misses --recording /data/walks.mp4 --truth /data/walks.json \
        --rules rules/examples/RXL-01.json --json out/fence.json
    # self-checks on generated footage with a scripted detector
    python -m tools.measure_false_alerts --self-check
    python -m tools.measure_fence_misses --self-check 50

Both run each recording through ingest and `CameraService` exactly as the service does (TARGET_FPS,
DETECTOR_STRIDE, calibration, fusion, rules) with the configured detector. The false-alert report states
its extrapolation (alerts / alertable camera-hours x 24, warm-up excluded) with an exact Poisson 95%
interval; whoever records the footage must confirm it is event-free. The fence report matches true
crossings to `fence crossed` firings within +/-1 s (one-to-one); the 2% target is only shown when the
Wilson upper bound is below 2%, which needs at least 189 crossings with no miss. Record each crossing as
a separate pass: the rule fires once per track per fence.

## ANPR (Nepali Devanagari + Bhutan Latin) and face detection

`edge/anpr/` reads licence plates on two grammars, auto-routed by script:

- **Nepal — Devanagari, two-line.** `detect_plate.py` locates a plate-shaped
  region inside a vehicle detection using classical CV (no second detector
  model); `rectify.py` perspective-corrects it, upscales small/far plates, and
  splits it into its two lines from the crop's own ink profile; `recognise.py`
  reads each line with PP-OCRv5's `devanagari_PP-OCRv5_mobile_rec` head;
  `postprocess.py` validates the result against the zone/lot/class/serial
  grammar and applies a measured (not guessed) confusion correction table.
- **Bhutan — Latin, single line, `BP-N-ANNNN`.** Same pipeline, PP-OCRv5's
  `en_PP-OCRv5_mobile_rec` head, validated against the Bhutan grammar.

Both heads run on every plate; `recognise.py` reports which script's output it
used (`RecognitionResult.script`), never guessing the script before either
head has read the pixels. `edge/anpr/compare_latin.py` reproduces
`docs/MEASUREMENTS.md` section 4 — the same plate image through both heads,
side by side — and is the artefact for innovation claim (a): off-the-shelf
Latin-only ANPR stacks have no head that was ever trained to read Devanagari
digits, so a Nepali plate is not partially read, it is not read at all.

```bash
python edge/anpr/compare_latin.py                    # renders its own sample plate
python edge/anpr/evaluate.py --labels datasets/plates/labels/rec_gt_val.txt \
    --images-root <the --out directory gen_plates.py used> \
    --write-confusions edge/anpr/data/confusions.json
```

**Fine-tuning.** The pretrained Devanagari head is a strong start but not
sufficient on real plates (`docs/MEASUREMENTS.md` section 4: about half the
characters on a real photo). `edge/anpr/notebooks/kaggle_ocr_finetune.ipynb`
fine-tunes it on the >= 20,000-sample synthetic corpus
(`datasets/plates/gen_plates.py`); `edge/anpr/scripts/package_plates_for_kaggle.py`
builds the upload folder for `kaggle datasets create`. Point
`ANPR_DEVANAGARI_MODEL_DIR` (and `ANPR_LATIN_MODEL_DIR`, if ever needed) at the
exported checkpoint to use it instead of the pretrained weights — nothing else
in `recognise.py` changes. Results, both before and after fine-tuning, are
recorded in `edge/anpr/results/ANPR_METRICS.md`.

**Why PaddleOCR's native CPU inference instead of an ONNX export.** The plan
prefers ONNX when practical; here it is not. `paddleocr.TextRecognition`
already runs the small (~10 MB) SVTR/CTC recognition graph through Paddle
Inference's CPU backend, which is already latency-comparable to onnxruntime
for this model size. Exporting a *fine-tuned* checkpoint to ONNX adds a
`paddle2onnx` conversion step for every future fine-tune, and PaddleOCR's own
pre/post-processing (the CTC decode against `charset.txt`, the exact resize
policy `RecResizeImg` uses) would have to be reimplemented by hand around a
bare ONNX session — a second thing to keep in sync with every retrain, for a
model that is already small and CPU-cheap. `edge/models/detector_weights.py`'s
ONNX-only path is the right call for the appearance detector, which is a much
larger network with a stable, un-retrained export; it is the wrong trade here.

**Face — `edge/face/`.** `detect.py` wraps `cv2.FaceDetectorYN` (YuNet,
OpenCV Zoo, Apache-2.0) — pretrained, no fine-tuning, matching the PDF. Its
weights are fetched at first use by `yunet_weights.py`, pinned by URL and
SHA-256 the same way `models/detector_weights.py` fetches the appearance
detector. `privacy.py` implements slide 4's privacy commitment: every stored
face is blurred by default (`blur_faces_in_frame`); the unblurred crop is
never written to disk in the clear — it is encrypted at rest with a Fernet key
(`FACE_UNSEAL_KEY`, environment only, never committed) that only the
Supervisor role can use; `unseal()` refuses every other role and writes an
audit row on EVERY attempt, granted or refused; `purge_expired()` deletes any
sealed crop older than 30 days and audits the purge.

**Licences.** PP-OCRv5 (PaddleOCR) is Apache-2.0. YuNet (OpenCV Zoo) is
Apache-2.0. Both permit commercial and derivative use, including a fine-tuned
checkpoint.

## Plain-language explanation (edge/vlm)

Every confirmed alert carries a `reason` written in plain words — slide 2's
claim: "every alert carries a written reason, so the operator acts instead
of reviewing footage." How that reason gets there:

- **t=0ms** — a fused event becomes an alert immediately, with a
  deterministic sentence rendered from the Phase 4 rule evidence
  (`vlm/prompt.py::rule_sentence`). No model is involved at this point, so
  this never delays the ~30 ms alert path.
- **t~3s (Jetson target)** — a background worker (`vlm/queue.py`) asks the
  local vision-language model for a sentence, runs it through guardrails
  (`vlm/guardrails.py`), and — only if it passes — patches the alert's
  `reason` in place. The queue is bounded and drop-oldest under load, and
  never blocks the caller; see `vlm/queue.py`'s module docstring for the
  exact mechanism.
- **The reason is never empty.** A rejected, timed-out, crashed, or
  unavailable model falls back to the rule-string sentence
  (`vlm/judge.py::fallback_reason`), which itself has three further fallback
  levels below it (see that function's docstring). `tests/test_vlm_async.py`
  and `tests/test_live_explain.py` (the real service path) kill, hang, crash
  and feed the model bad output and assert the reason is never blank.

### Live wiring (pipeline/service.py, pipeline/explain.py)

The order inside `CameraService.process` is the contract, and
`tests/test_live_explain.py` asserts it on the real service:

1. the frame is processed and fusion / the rules produce events;
2. each event gets its reason: the rule sentence goes into
   `event["explanation"]` with `source: "template"` (never empty, never the
   model's);
3. the events are emitted (`EventSink.emit`: ring, log file, delivery outbox);
4. only then are each event's frame (a downscaled copy, `VLM_FRAME_MAX_SIDE`),
   rule evidence and camera context queued (`EventExplainer.submit`: a queue
   append that never blocks and never raises);
5. a worker thread runs the model; if the sentence passes the guardrails,
   `EventSink.emit_patch` replaces the reason for the same `event_id` (ring
   copy, log file and a reason patch to the backend, see below). Rejection,
   a timeout, an exception, a missing model or a dropped job all leave the
   rule sentence in place and send nothing.

**Drop policy.** At most `VLM_QUEUE_MAX` jobs wait (default 8, each one
downscaled frame). When the queue is full the OLDEST waiting job is dropped
and counted (`explain.dropped` in `GET /pipeline/status`); its event keeps its
rule sentence, and the newest event, the one an operator is most likely
looking at, is the one that gets explained. `VLM_COOLDOWN_S` (default 0) makes
the model rest between jobs to cap how much compute it takes from the frame
loop.

**Enabling.** `VLM_ENABLED=auto` (default) loads Moondream 2 only if its
pinned weights are already in the local Hugging Face cache, so a boot never
starts a surprise 3.8 GB download and an air-gapped box never reaches out; it
is always off under pytest. `VLM_ENABLED=1` loads it (downloading if needed),
`VLM_ENABLED=0` never does. The model loads in a background thread; until it
is ready, or if it never loads, every reason stays the rule sentence.
`VLM_DEVICE=auto|cpu|mps` picks the device (auto uses Apple's MPS when torch
reports it, otherwise CPU). Set `HF_HUB_OFFLINE=1` in a deployment: without
it the loader sends HEAD requests to the Hugging Face hub even when the
weights are cached.

### MEASURED Moondream 2 latency on this hardware

Measured with `python vlm/latency.py --clips <sample clips> --device <d>`
(each run writes one JSON under `vlm/results/`, holding every sample). Real
frames from the three sample clips (day, day 1080p, thermal), downscaled to
768 px like the live service does, prompt and guardrails exactly as in
production, `max_tokens` 80, model revision
`5d6c926f44e26b07957b0dd315bbedcb4c17a5fe`, torch 2.14, transformers 5.17.
These are numbers for THIS machine only, not for the deployment target.

| Device | Hardware | n | p50 | p95 | Result file |
|---|---|---|---|---|---|
| **MPS** (Apple GPU) | Apple M5, 16 GB unified memory | 24 | **2.93 s** | **4.11 s** | `vlm/results/moondream2_mps.json` |
| CPU (bf16) | Apple M5, 16 GB unified memory | 6 (about 4.5 min each, too slow for 20) | 272.8 s | 309.9 s | `vlm/results/moondream2_cpu.json` |

- The checkpoint hard-codes bfloat16 in its image preprocessing, so it stays
  bf16 on either device. On this CPU that is very slow; the MPS path is the
  practical one on an Apple-silicon Mac, which is why `VLM_DEVICE=auto`
  prefers it.
- Dynamic int8 quantisation is NOT usable on this torch build (`NoQEngine`
  from `quantized::linear_prepack`); the runtime logs it and runs bf16.
- Guardrails accepted 7 of the 24 MPS sentences (29 %); the other 17 fell back
  to the rule sentence (invented time, invented object, more than the facts
  allow). That is the fallback doing its job with a 1.9B model, and the reason
  was never empty. See the JSON for each rejection.
- **The PDF's ~3 s is a Jetson-class target, not a measurement.** The MPS
  p50 above happens to land near it on this Mac's GPU; nothing here says a
  Jetson will. Re-run `vlm/latency.py` on the target before quoting a figure
  for it.

### End to end on a sample clip (measured, shared Mac)

The service path (detector ONNX, motion, tracking, fusion, rules with a
virtual fence, VLM on MPS, delivery to a local stub backend) over the sample
clips, unpaced, other work running on the same machine so treat FPS as
approximate:

| Clip | VLM | FPS | Events (fusion + rule) | Reason patches |
|---|---|---|---|---|
| day 640x512, 435 frames | off | 32.0 | 9 (4 + 5) | 0 |
| day 640x512 | MPS | 29.9 | 9 (4 + 5) | 0 (all 9 rejected by the guardrails, rule sentence kept) |
| thermal 640x512 | off | 17.7 | 12 (7 + 5) | 0 |
| thermal 640x512 | MPS | 7.7 to 9.7 | 12 (7 + 5) | 2 accepted (5.4 s and 31.2 s after their event, the second queued behind others) |

A later re-run on the day clip after the wire change (VLM on MPS, 28.6 FPS, 7 events): 2 model sentences accepted, patched 5.1 s and 14.3 s after their event; 16 messages delivered (7 created, 7 rule-sentence updates, 2 model updates).

Events per second of video: 0.41 (day), 0.55 (thermal). In every row all
events reached the backend created (delivered 9/9 and 12/12) with the rule
sentence at t=0; the model never delayed emission. Finding: while the model
is busy on MPS it takes shared compute from the frame loop (thermal clip:
17.7 to under 10 FPS), so on a shared-memory machine set `VLM_COOLDOWN_S` or
accept the drop; nothing blocks, throughput just falls.

**Model: [Moondream 2](https://huggingface.co/vikhyatk/moondream2)**
(`vikhyatk/moondream2`), Apache-2.0, ~1.9B parameters, pinned to commit
`5d6c926f44e26b07957b0dd315bbedcb4c17a5fe` — not floating `main`, so an
upstream push cannot silently change what code `trust_remote_code=True` runs
here. (This commit was chosen over the model card's own tagged `2025-06-21`
release because that tag's `encode_image` raises `KeyError: 'variant'` on
the exact `settings={"max_tokens": ...}` call this runtime needs to make to
bound generation length; see `vlm/runtime.py`'s module docstring for the
full explanation.) **Local by default, CPU-only.** The model's own weights load in
`bfloat16` (its shipped reduced-precision default); this service additionally
attempts dynamic int8 quantisation of the linear layers at load time
(`torch.ao.quantization.quantize_dynamic`) and reports on this machine
whether that actually helped (it was not usable on the measured machine, see
the measured section above). The device is CPU or Apple MPS (`VLM_DEVICE`).

**The PDF's ~3 s figure is a Jetson-class edge-device target, not a claim
about a development laptop.** `vlm/results/*.json` record what local
Moondream 2 actually takes on the machine each was measured on; re-run
`python vlm/latency.py` on the target hardware before a demo that needs a
real figure.

**Pre-download for an air-gapped box.** On a machine that still has network,
before it goes dark, pull both repositories Moondream 2 needs into the
standard Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`):

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('vikhyatk/moondream2', revision='5d6c926f44e26b07957b0dd315bbedcb4c17a5fe'); \
  snapshot_download('moondream/starmie-v1')"  # the tokenizer repo moondream2 also loads at init
```

Copy `~/.cache/huggingface` to the air-gapped box (or point `HF_HOME` at a
copy of it), then set `HF_HUB_OFFLINE=1` in the deployment environment so a
cache miss fails loudly (`VlmUnavailable`, which degrades to the rule-string
reason) instead of silently trying to reach the network.

**Development-only hosted fallback.** `vlm/runtime.py::HostedVlmRuntime`
exists only so a developer on a machine too slow for local CPU inference can
keep working. It is:
- **off by default** — nothing loads it unless `TRUEWATCH_VLM_HOSTED=1` is
  set (`vlm/runtime.py::build_default_runtime`);
- **loudly labelled** — enabling it logs a `logging.warning` naming the
  endpoint and stating in plain words that this breaks the air-gapped
  deployment claim, every time it is constructed;
- **never the shipped default**, and any screenshot or recording taken while
  it is enabled must say so. It requires its own endpoint URL and key
  (`TRUEWATCH_VLM_HOSTED_URL`, `TRUEWATCH_VLM_HOSTED_KEY`) — no default
  endpoint is baked in, so opting in without also configuring a real
  endpoint simply raises `VlmUnavailable`.

## Alert policy: only a fired rule alerts (ALERT_POLICY)

Dual-channel agreement (appearance and motion agree on a track) is necessary but not sufficient for an
operator alert. With `ALERT_POLICY=rules` (the default) an event is delivered to the backend
(`POST /api/ingest/event`), given a reason request and sealed as evidence only when at least one rule has
fired on its track: fence crossed, loitering, wrong direction, out of hours or grouping (a rule event, or a
fusion or plate event on a track whose `rules_fired` is not empty). One alert per (track, rule), under the
same throttle as before. A fusion-only event ("someone is here") is a local observation:

- it stays in the in-memory ring behind `GET /pipeline/events` and in `EVENTS_LOG_PATH`, marked
  `"kind": "observation"` (`total` counts alerts, `observations` counts these);
- it is still indexed for footage search (`SEARCH_ENABLED`);
- it is never delivered, never given a model reason request and never sealed. If a rule later fires on the
  same track, that rule event is the alert and is sealed at that moment; the clip ring (`EVIDENCE_CLIP_SECONDS`)
  already holds the preceding seconds.

`ALERT_POLICY=all` is the legacy behaviour (every fusion event alerts). It is kept for comparison only and
is not recommended: on MEVA it gave about 5,000 alerts per camera per day (docs/VERIFICATION_REPORT.md
section 5). The backend's alert shape is unchanged; observations never reach it.

### Configuring rules for a camera

Under the rules policy a camera with no rule file can never alert, and the service logs a loud
`NO RULE CONFIGURATION` warning at start. Put one JSON file per camera, named `<CAMERA_ID>.json`, in
`RULES_CONFIG_DIR` (default `./var/rules`); the file is hot-reloaded on change. Start from
`rules/examples/RXL-01.json`:

- `fences`: polylines in the pixels of the full frame (`direction` in / out / both);
- `sanctioned_hours`: the post's sanctioned crossing windows (local time); outside them any agreed track
  fires `out of hours`. Omit the key to switch the rule off;
- `loitering`, `direction`, `grouping`: thresholds (metres when a `homography` is given, pixels otherwise).
  Loitering, wrong direction and grouping are on for every configured camera with their defaults.

Out of hours is NOT on by default for border cameras. Sanctioned windows differ per post and per crossing,
and a guessed window either silences real crossings (too wide) or turns every night-time passer-by into
an alert, which is the "everything alerts" behaviour this policy removes (too narrow). The operator sets
the post's real windows in its rule file; a camera with neither fences nor sanctioned hours gets a start-up
warning that only loitering, wrong direction and grouping can alert there.

## Delivery to the backend (pipeline/events.py)

`EventSink` hands every created event and every reason patch to
`BackendDelivery`, a background sender. Configuration: `BACKEND_BASE_URL`
(alias `BACKEND_URL`), `TRUEWATCH_INGEST_KEY`, `EDGE_DELIVERY_ENABLED`
(default true; without a key delivery is off and the service says so),
`EDGE_DELIVERY_TIMEOUT_S` (3), `EDGE_DELIVERY_MAX_ATTEMPTS` (5),
`EDGE_DELIVERY_OUTBOX_MAX` (1000).

Wire format (ARCHITECTURE_V2 section 4.1, unchanged):

```http
POST {BACKEND_BASE_URL}/api/ingest/event
Content-Type: application/json
X-Truewatch-Ingest-Key: <TRUEWATCH_INGEST_KEY>
X-Truewatch-Timestamp: <RFC3339 UTC, request creation time>
X-Truewatch-Nonce: <uuid4, fresh on every attempt>
```

**Created event** = the section 4.2 `truewatch.event.v1` body, `stage:
"provisional"`, with `explanation: null` on the wire (the backend's validator
rejects anything else while provisional). Locally, in the ring, the log and
`GET /pipeline/events`, the event carries the rule sentence,
`{"text": "...", "source": "template", "model": null, "latency_ms": null}`,
so the reason is never empty.

**Reason updates** use the backend's `truewatch.update.v1`
(`POST /api/ingest/update`, same headers, strict schema). Up to two are sent
per event, in order behind the created event: at once the rule sentence
(`source: "template"`), and later, only if the model's sentence passed the
guardrails, the model's:

```json
{"schema": "truewatch.update.v1", "event_id": "<same uuid as the created event>",
 "explanation": {"text": "...", "source": "vlm", "model": "moondream2", "latency_ms": 2840}}
```

Nothing but `schema`, `event_id` and `explanation`. Idempotent, and always
after its created event (one ordered sender); an update whose created event
was lost is skipped.

Behaviour: the pipeline thread only appends to the outbox, which never blocks.
A full outbox drops the oldest message (counted). Each message gets up to
`MAX_ATTEMPTS` tries with exponential backoff (0.5 s doubling to 8 s) on a
transport error or 408/425/429/5xx, each with a new nonce and the same
`event_id` (the idempotency key). 2xx and 409 count as delivered (409 = the
backend already has it); other 4xx are dropped at once and logged. Counters are in
`GET /pipeline/status` under `delivery`. `tests/test_delivery.py` runs a real
local HTTP server, including a dead port.

## Face detection in the live loop (edge/face/stage.py)

`FACE_ENABLED=true` runs YuNet on the head region of PERSON tracks whose box
is at least `FACE_MIN_PERSON_HEIGHT_FRAC` (0.25) of the frame height, once
per track with a bounded retry schedule, and stores a blurred crop (faces
blurred) under `FACE_STORE_DIR`. The unblurred face is stored only sealed
with `FACE_UNSEAL_KEY`; with no key only blurred crops are kept, never a clear
one. Unsealing stays Supervisor-only with an audit row per attempt, and
sealed crops are purged after 30 days (hourly job, audited). Nothing is added
to the event contract. Measured cost: YuNet takes 1.4 ms on a 216x200 head
crop and 2.7 ms on 400x300 (Apple M5, CPU), called a few times per track, so
it does not hurt FPS; it is off by default because it stores personal data,
not for speed. The sample clips contain no close-range faces (0 found), so
detection quality was covered by unit tests with a stand-in detector.

## Semantic footage search (edge/search)

`POST /api/search {query}` embeds the query with SigLIP's text tower and
ranks archived clips by cosine similarity against SigLIP image embeddings of
their keyframes (`search/query.py`). The response shape is copied exactly
from `backend/src/services/search.service.js` — the FootageSearch component
the console already renders — so nothing in `frontend/` changes.

**Model: [SigLIP](https://huggingface.co/google/siglip-base-patch16-224)**
(`google/siglip-base-patch16-224`), a dual encoder released under Apache-2.0.
~400 MB. Pre-download it the same way, before the box goes dark:

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('google/siglip-base-patch16-224')"
```

**Match score is an absolute, calibrated probability, not a per-batch
rescale.** `search/query.py::_score_percents` maps each result's cosine
similarity through SigLIP's own learned sigmoid calibration —
`sigmoid(cosine * logit_scale + logit_bias)`, the exact function the
checkpoint was trained to produce a match probability with (see that
function's docstring, and `SiglipModel.forward`'s own docstring example,
which does the same thing). An earlier version of this module stretched
each batch of results into a fixed `[55, 99]` display band by min-max
rescaling that batch's own scores — which meant the *top* result of any
batch, including a batch of five completely irrelevant clips, always read as
roughly "99% match". That was a ranking indicator wearing a confidence
number's clothes. The fixed version still preserves ranking exactly (sigmoid
is monotonic), but an unrelated query now legitimately scores low —
`tests/test_search.py::test_unrelated_query_does_not_score_as_a_strong_match`
is the regression test for this.

**Keeping the index current (`search/incremental.py`).** With
`SEARCH_ENABLED=true` the service embeds the frame of every fusion-agreed
event (one clip per track, up to 3 keyframes) on a bounded background worker,
never blocking the pipeline (a full queue drops the oldest), into the same
in-memory index `/api/search` reads, and saves it to `SEARCH_INDEX_PATH`.
Sealed clips written under `SEARCH_CLIPS_DIR` are picked up every minute
through the backfill code. Measured on the sample clip: 9 events, 9 keyframes
indexed, 0 errors, queries answered in the unchanged FootageSearch shape.

Index: a plain numpy brute-force cosine scan (`search/index.py`), justified
for a single post's archive (bounded well under 100k keyframes) over
FAISS-CPU — see that module's docstring for the measured latency at 100k
vectors and the reasoning.

## Evidence chain (edge/evidence)

Every clip is SHA-256 hashed **while it is written**, entered as a block in an
append-only hash chain, folded into one Merkle root per IST day, signed on the
board with Ed25519 and mirrored to sector HQ. One command re-derives all of it
from the bytes on disk. Nothing in `evidence/` opens a network connection; the
HQ mirror is a second directory (removable media or a separate volume in the
field).

```bash
python -m evidence.seed_demo                                   # a real 3-day demo chain in ./var/evidence
python -m evidence.verify --from 2026-09-01 --to 2026-09-30    # PASS/FAIL table, exit 0 / 1 (2 = unreadable)
python -m evidence.tamper_demo                                 # flip one byte in one sealed clip, re-verify
python -m evidence.tamper_demo --block N --offset K --restore  # flip it back
python -m evidence.certificate --block 6 --out cert.md         # Section 63 (BSA 2023) certificate
```

`EVIDENCE_DIR` (default `./var/evidence`) and `EVIDENCE_HQ_DIR` (default
`<EVIDENCE_DIR>-hq`) select the store and the mirror; `TRUEWATCH_DEVICE_ID`
names the post the first time a store is created.

**API for the pipeline** (the pipeline wiring itself is not part of this module):

```python
from evidence import EvidenceStore, chain_view, verify_view

store = EvidenceStore(cfg.evidence_dir)          # recovers and logs a gap after a crash
w = store.open_clip("RXL-01", started_at)        # hash at capture
w.write_frame(jpeg_bytes)                        # each byte hashed in the same call that writes it
block = w.seal(ended_at)                         # fsync, move into clips/<day>/, append the block
block = store.seal_clip(path, "RXL-01", started_at, ended_at)   # a file another writer closed
event["evidence"] = block.evidence_dict()        # the truewatch.event.v1 `evidence` object
store.heartbeat()                                # now and then, so a later gap starts accurately
store.close()                                    # clean shutdown

chain_view(root, limit=5)   # FROZEN GET  /api/evidence/chain  -> [{ n, hash, camera, time }] (+ full_hash)
verify_view(root)           # FROZEN POST /api/evidence/verify -> { intact, verified, total, blocks: [{ n, ok }] }
```

`hash` is the frozen `<10hex>…<6hex>` display form of the block hash; `time`
is `DD MON · HH:MM` in IST. `full_hash` is the only added (optional) field.

**Design decisions** (each module's docstring has the full specification):

| Decision | Choice | Why |
|---|---|---|
| Clip digest | `SHA-256(bytes ‖ "TWCLIP1" ‖ u64be(frames) ‖ u64be(length))` | Frame count and length are bound into the digest; a truncated or padded file cannot match. |
| Block hash | SHA-256 over `truewatch.block.v1` LF, then `name=value` LF lines in the fixed order `n, clip_id, camera_id, started_at, ended_at, sealed_at, clip_sha256, prev_block_hash`; ASCII; every value grammar-checked | Not JSON: no key-order, whitespace or escaping ambiguity. Genesis prev is 64 zeros. |
| Storage | JSON Lines, `O_APPEND` + fsync per block | Append-only by construction; a power cut can only tear the last line; readable with a text editor. |
| Merkle | RFC 6962-style `0x00` leaf / `0x01` node prefixes; odd node **promoted unchanged** | Duplicate-last makes `[a,b,c]` and `[a,b,c,c]` share a root (CVE-2012-2459). |
| Day | IST calendar day of `sealed_at` | Fixed +05:30 offset, no DST; `sealed_at` never decreases, so days are contiguous. |
| Signing | Ed25519 (`cryptography`), key generated on the box, `keys/signing_key.pem` mode 0600 in a 0700 dir | Verify trusts the key registered at HQ, never the box's own copy. |
| HQ mirror | Write-once public key and daily roots | A rewritten root on the box cannot overwrite HQ's copy. |
| Gaps | Every restart appends exactly one hash-chained gap record; the signed daily manifest carries the gap count and head | A gap is disclosed, not a failure; deleting one breaks a signature. |

**Power loss.** On every start the store reads its run marker, quarantines a
torn final ledger line and any half-written clips (never deletes them), and
appends one gap `{start: last time the store was alive, end: now, kind, reason}`.
The chain then resumes from the last complete block.

**Section 63 certificate.** The legal wording is in
`evidence/templates/section63_certificate.md` and carries a visible note that it
is a template which must be reviewed by a qualified legal practitioner before
any real use. The code fills in only the technical facts: record, device,
process, hashes recomputed at generation time, Merkle inclusion proof, signature
and HQ-mirror status, gaps, and blank signatory blocks.

**Residual risk, stated plainly.** Blocks sealed today are protected by the
chain and the clip hashes but not yet by a signature; someone with root on the
box could truncate today's tail before the day is signed. A physically separate
HQ mirror is what makes signed days tamper-evident against the box's own
operator.

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
