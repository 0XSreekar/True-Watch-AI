# TRUEWATCH — Architecture v2

Problem Statement **SIH26187** · AI-Based Intelligent Video Analytics Platform for Border
Surveillance using existing CCTV Infrastructure · Theme Blockchain & Cybersecurity ·
PS Category Software · Team ID 130977 · Team Name Future Bytes ·
Organization Ministry of Home Affairs; Sashastra Seema Bal (SSB), Police II Division. *(Slide 1)*

**Authority.** `SIH26187_TRUEWATCH_final-1.pdf` is the single source of truth for every technical
decision in this document; the slide is quoted wherever a decision rests on it. `MEASUREMENTS.md`
is the single source of truth for any number described as measured.
`docs/PHASE_MINUS1_SCOPE.md` fixes what is in and out of scope. Where the PDF is silent or
ambiguous, the point is recorded under **Open Questions** (§11) rather than guessed.

**Status of this document.** Blueprint only. Phase 0 writes documentation and empty scaffolding.
No pipeline logic, no database code, no endpoint behaviour is implemented here.

This file replaces `docs/ARCHITECTURE.md`, which described the two-process prototype.

---

## 1. Runtime topology

### 1.1 Three processes

| Process | Language / runtime | Why it exists | Where it runs (dev) | Where it runs (demo) |
|---|---|---|---|---|
| `edge/` | Python 3.11, FastAPI + Uvicorn | Everything that touches pixels. Decode, the appearance channel, the motion channel, fusion, tracking, the rule engine, ANPR, face, the explanation model, and the hash at capture. Slide 3 puts all of this on the board at the post: *"Existing IP camera — no new hardware on the pole."* The models named on slide 3 (YOLO11-s, ByteTrack, YuNet, PP-OCRv5 Devanagari, Moondream 2, SigLIP) are Python-only in practice. | Development machine (Apple Silicon), `http://localhost:8000` | Hugging Face Spaces, free CPU tier |
| `backend/` | Node 20, Express 5 ESM | The existing API becomes a backend-for-frontend. It receives events from `edge/`, persists them, serves the **frozen contract** to the console unchanged, and streams live updates. It is the only process the browser talks to. | `http://localhost:4000` | Render, free tier |
| `frontend/` | React 19, Vite 7 | The operator console. **Unchanged by this plan.** | `http://localhost:5173` | Vercel, free tier |

One post in the field runs one `edge/` process against its own cameras. Slide 4 frames scale as
replication, not coordination — *"Designed to scale to SSB's 734 border outposts"* — so nothing in
this topology requires posts to talk to each other.

### 1.2 Data flow — slide 3's seven methodology steps

The diagram below is the *"Methodology and process for implementation"* column of slide 3, drawn
once, with the owning process and module named at each step.

```text
                         ╔═══════════════════════════════════════════════╗
                         ║  edge/   Python · FastAPI · one per BOP        ║
                         ╚═══════════════════════════════════════════════╝

  ┌─ STEP 1 ─────────────────────────────────────────────────────────────────────────┐
  │ Existing IP camera — no new hardware on the pole                                 │
  │ ONVIF Profile S / RTSP pull, or a file replayed through the same decode path     │
  │ edge/pipeline/ingest.py                                                          │
  └───────────────────────────────────┬──────────────────────────────────────────────┘
                                      │  decoded frame + monotonic timestamp
                    ┌─────────────────┴──────────────────┐
                    │                                    │
  ┌─ STEP 2a ───────▼──────────────┐   ┌─ STEP 2b ───────▼───────────────────────────┐
  │ APPEARANCE channel             │   │ MOTION channel                              │
  │ YOLO11-s (ONNX Runtime)        │   │ Dense Farneback optical flow                │
  │ person / truck / car /         │   │ object-to-background flow contrast          │
  │ two-wheeler                    │   │ measured 1.9x visible, 3.4x LWIR            │
  │ edge/pipeline/appearance.py    │   │ edge/pipeline/motion.py                     │
  └────────────────┬───────────────┘   └─────────────────┬───────────────────────────┘
                   │  score_appearance                   │  score_motion
                   └──────────────────┬──────────────────┘
                                      │
  ┌─ STEP 3 ─────────────────────────▼───────────────────────────────────────────────┐
  │ Both channels must agree — fusion against a per-camera threshold                 │
  │ night weighting derived from the measured flow contrast, not hardcoded           │
  │ edge/pipeline/fusion.py                                                          │
  └───────────────────────────────────┬──────────────────────────────────────────────┘
                                      │  agreed detections
  ┌─ STEP 4 ─────────────────────────▼───────────────────────────────────────────────┐
  │ Detector inference, ~30 ms TARGET (YOLO11-s INT8 budget at 640 px, NOT           │
  │ end-to-end) · fusion follows · ByteTrack assigns a persistent track id ·         │
  │ named rules fire on tracks against a per-camera baseline                         │
  │ edge/pipeline/track.py · edge/rules/engine.py · edge/rules/baseline.py           │
  │ ANPR and face run on crops off this path:                                        │
  │ edge/pipeline/anpr.py (PP-OCRv5 Devanagari) · edge/pipeline/face.py (YuNet)      │
  └───────────────────────────────────┬──────────────────────────────────────────────┘
                                      │  provisional alert  ──────────┐
                                      │                               │ emitted immediately,
  ┌─ STEP 5 ─────────────────────────▼──────────────────────────┐    │ stage = "prov"
  │ VLM explains the alert — about 3 s TARGET, ASYNCHRONOUS     │    │
  │ Moondream 2 writes the plain-language reason                │    │
  │ edge/pipeline/explain.py                                    │    │
  └───────────────────────────────────┬─────────────────────────┘    │
                                      │  reason text, stage = "conf" │
  ┌─ STEP 6 ─────────────────────────▼───────────────────────────────▼───────────────┐
  │ Clip sealed, chained and rooted — SHA-256 at capture, daily root                 │
  │ edge/evidence/hasher.py · edge/evidence/chain.py · edge/evidence/merkle.py       │
  │ (signing at sector HQ is described, not run — PHASE_MINUS1_SCOPE §6)             │
  └───────────────────────────────────┬──────────────────────────────────────────────┘
                                      │
                       POST /api/ingest/event  (shared-secret header)
                                      │
                         ╔════════════▼══════════════════════════════════╗
                         ║  backend/   Node · Express 5 · BFF            ║
                         ╚═══════════════════════════════════════════════╝
                                      │
  ┌─ STEP 7 ─────────────────────────▼───────────────────────────────────────────────┐
  │ Event log goes to the existing control room                                      │
  │ ingest.controller → ingest.service → mapEventToAlert() → alerts repository       │
  │ backend/src/realtime/hub.js fans the alert out over GET /api/stream (SSE)        │
  └───────────────────────────────────┬──────────────────────────────────────────────┘
                                      │  frozen alert shape, unchanged
                         ╔════════════▼══════════════════════════════════╗
                         ║  frontend/   React 19 · UNCHANGED             ║
                         ║  useConsole → AlertQueue, LiveWall,           ║
                         ║  EvidenceChainPanel, SectorMap,               ║
                         ║  FootageSearch, AnalyticsPanel                ║
                         ╚═══════════════════════════════════════════════╝
```

### 1.3 Step-to-module map

| # | Slide 3 step (quoted) | Process | Module | Phase |
|---|---|---|---|---|
| 1 | *"Existing IP camera — no new hardware on the pole"* | `edge/` | `edge/pipeline/ingest.py` | 1 |
| 2 | *"Appearance + Motion — YOLO11-s and optical flow"* | `edge/` | `edge/pipeline/appearance.py`, `edge/pipeline/motion.py` | 2, 3 |
| 3 | *"Both channels must agree — fusion against a per-camera threshold"* | `edge/` | `edge/pipeline/fusion.py` | 4 |
| 4 | *"Detector inference, ~30 ms target — YOLO11-s INT8 inference budget; fusion follows"* | `edge/` | `edge/pipeline/track.py`, `edge/rules/engine.py`, `edge/rules/baseline.py` | 5 |
| 5 | *"VLM explains the alert — about 3 s, asynchronous"* | `edge/` | `edge/pipeline/explain.py` | 7 |
| 6 | *"Clip sealed, chained and rooted — SHA-256 at capture, daily root signed at sector HQ"* | `edge/` | `edge/evidence/{hasher,chain,merkle}.py` | 9 |
| 7 | *"Event log goes to the existing control room"* | `backend/` → `frontend/` | `backend/src/{controllers/ingest.controller.js,services/ingest.service.js,realtime/hub.js}` | 8, 10 |

Every one of the seven boxes on slide 3 is claimed by exactly one named module. Steps 2 and 4 each
own more than one module because slide 3 collapses two channels into one box (step 2) and the
detector, tracker and rule engine into one box (step 4).

---

## 2. Why a separate Python service

1. Every model slide 3 names — YOLO11-s, ByteTrack, YuNet, PP-OCRv5 Devanagari, Moondream 2,
   SigLIP — ships as Python. Reimplementing any of them in Node would invalidate the measured
   results in `MEASUREMENTS.md`, which were produced by the Python originals.
2. The console must stay watchable while the pipeline is being rewritten hourly during a six-day
   build; a crash in the decoder must not take down the operator console with it.
3. Slide 3 puts the pipeline on the board at the post and the event log in the control room —
   two processes in the deck's own topology, not one.
4. **The honest cost:** two deployment surfaces instead of one, a second dependency tree, and
   cold starts on the free Spaces CPU tier that can exceed thirty seconds on first request.
5. **Mitigation:** the demo runs entirely from `localhost` (`PHASE_MINUS1_SCOPE` risk 5); the
   hosted Space is pre-warmed and offered as a link afterwards; and `backend/` keeps its existing
   mock services as the fallback path, so the console degrades to sample data rather than to an
   error when `edge/` is unreachable.

