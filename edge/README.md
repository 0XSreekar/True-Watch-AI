# TRUEWATCH — edge inference service

Everything that touches pixels lives here: decode, the appearance channel, the
motion channel, fusion, tracking, the rule engine, ANPR, face detection, the
plain-language explanation, and the SHA-256 hash taken at capture.

One post in the field runs one copy of this service against its own cameras.
It reaches the rest of the system through exactly one call —
`POST /api/ingest/event` on the Express backend, authenticated with a shared
secret. It never talks to the browser.

> **Phase 1.** The service shell and the ingest path are in. The appearance and
> motion channels, fusion, tracking, the rule engine, ANPR, face, the
> explanation model and evidence hashing arrive in Phases 2 through 9.

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
