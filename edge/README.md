# TRUEWATCH — edge inference service

Everything that touches pixels lives here: decode, the appearance channel, the
motion channel, fusion, tracking, the rule engine, ANPR, face detection, the
plain-language explanation, and the SHA-256 hash taken at capture.

One post in the field runs one copy of this service against its own cameras.
It reaches the rest of the system through exactly one call —
`POST /api/ingest/event` on the Express backend, authenticated with a shared
secret. It never talks to the browser.

> **Phase 0 scaffolding.** This folder currently contains package markers,
> dependency and container definitions, and this file. There is no pipeline
> code yet. `app.py` arrives in Phase 1.

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

```bash
cd edge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in TRUEWATCH_INGEST_KEY
```

The root `npm run dev` starts the console and the API only. This service is
started separately, and the console works without it — `apiClient.withFallback`
keeps the operator console watchable when nothing upstream is running.

## Rules for this folder

- **No model weights in git.** Weights are referenced by id and resolved at
  runtime; `models/.gitignore` enforces it.
- **No secrets in git.** Every value comes from the environment. `.env.example`
  carries names and safe defaults only.
- **No dataset files in git.** `datasets/` holds fetch scripts, never data.
- **Honesty flags are not optional.** Every event carries `source`
  (`rtsp` or `file`) and `explanation.source` (`vlm` or `template`), so a
  replayed file is never presented as a live camera and template text is never
  presented as model output.
