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
