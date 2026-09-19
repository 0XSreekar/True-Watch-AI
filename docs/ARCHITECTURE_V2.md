# TRUEWATCH — Architecture v2

Problem Statement **SIH26187** · AI-Based Intelligent Video Analytics Platform for Border
Surveillance using existing CCTV Infrastructure · Theme Blockchain & Cybersecurity ·
PS Category Software · Team ID 130977 · Team Name Future Bytes ·
Organization Ministry of Home Affairs; Sashastra Seema Bal (SSB), Police II Division. *(Slide 1)*

**Authority.** `SIH26187_TRUEWATCH_final-1.pdf` is the single source of truth for every technical
decision in this document; the slide is quoted wherever a decision rests on it. `docs/MEASUREMENTS.md`
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

---

## 3. Frozen contract

### 3.1 Every existing endpoint

Verified against `backend/src/routes/`, `backend/src/controllers/` and `frontend/src/services/`
in this commit. **No path, method, request shape or response shape in this table changes, ever.**
`apiClient` unwraps the outer `{ data: … }`, so the "Response (unwrapped)" column is what the
component actually receives.

| Method | Path | Request | Response (unwrapped) | UI consumer | Real producer after Phase 8 |
|---|---|---|---|---|---|
| GET | `/api/health` | — | `{ status, service, time }` — **not** wrapped in `data` | none (ops/uptime check) | unchanged; extended with edge reachability in Phase 10 |
| POST | `/api/auth/login` | `{ officialId, password }` | `{ ok, token, expiresIn, user:{id,name,role,unit,initials} }` | `useAuthForm` → `LoginForm` (`/login`) | `auth.service` against the `users` table |
| POST | `/api/auth/register` | `{ name, officialId, unit, email, password, role }` | `{ ok, pendingApproval, token, expiresIn, user }` | `useAuthForm` → `SignupForm` (`/signup`) | `auth.service` against the `users` table |
| GET | `/api/auth/units` | — | `string[]` | `SignupPage` | `users`/unit registry |
| GET | `/api/auth/roles` | — | `[{ k, dd }]` | `SignupPage` | static role table |
| GET | `/api/cameras` | `?sector=` optional | `[{ id, name, sector, ir, road }]` | `useConsole` → `LiveWall` | `cameras` table via `camerasRepository` |
| GET | `/api/cameras/posts` | — | `[{ id, name, x, y, road, alert }]` | `useConsole` → `SectorMap` | `posts` table via `postsRepository` |
| GET | `/api/cameras/sectors` | — | `string[]` | **defined in `cameras.service.js` as `fetchSectors`, not called by any component** | derived from `cameras` table |
| GET | `/api/cameras/:id` | — | `{ id, name, sector, ir, road }` or 404 | **no frontend service exists** | `cameras` table |
| GET | `/api/alerts` | — | `[alert]` (plus a sibling `budget` key that `apiClient` discards) | **defined as `fetchAlerts`, not called by any component**; `useConsole` builds its list from `simulate` | `alerts` table via `alertsRepository` |
| GET | `/api/alerts/budget` | — | `{ used, total, near }` | **no frontend service exists**; `useConsole` computes the budget locally | derived from `alerts` per shift |
| POST | `/api/alerts/simulate` | — | one `alert` (201) | `useConsole.spawn` → `AlertQueue` | replaced as the *demo* path; live alerts arrive via SSE, this endpoint stays for offline demos |
| POST | `/api/alerts/:id/act` | — | the `alert` with `stage: 'sealed'`, or 404 | `useConsole.actOnAlert` → `AlertQueue` **ACT · SEAL EVIDENCE** | `alerts` + `evidence_blocks`, seals the clip |
| POST | `/api/alerts/:id/dismiss` | — | the `alert` with `stage: 'gone'`, or 404 | `useConsole.dismissAlert` → `AlertQueue` **DISMISS** | `alerts` + per-camera baseline update (slide 4: *"the operator dismisses one and that camera's baseline updates"*) |
| GET | `/api/search/placeholders` | — | `string[]` | `useConsole` → `ConsoleTopBar`, `FootageSearch` | static, or learnt from query history |
| POST | `/api/search` | `{ query }` | `{ query, tookMs, windowScanned, results:[{id,camera,time,chips,score}] }` | `useConsole.runSearch` → `FootageSearch` | SigLIP embedding index (slide 3) |
| GET | `/api/evidence/chain` | — | `[{ n, hash, camera, time }]` | `useConsole` → `EvidenceChainPanel` | `evidence_blocks` table |
| POST | `/api/evidence/verify` | — | `{ intact, verified, total, blocks:[{n, ok}] }` | `useConsole.verifyChain` → `EvidenceChainPanel` **VERIFY CHAIN** | real chain walk over `evidence_blocks` |
| GET | `/api/analytics` | — | `{ camBars, dayNight, agreement, traffic }` | `useConsole` → `AnalyticsPanel` | aggregate over `alerts` |
| GET | `/api/analytics/traffic` | — | `{ sample: number }` | `useConsole` traffic ticker → `AnalyticsPanel` | rolling window over `tracks` |

> **Three endpoints exist that the Phase 0 brief's frozen list omitted:** `GET /api/alerts/budget`,
> `GET /api/cameras/sectors`, `GET /api/cameras/:id`. They are live in `backend/src/routes/` today,
> so they are treated as frozen on the same terms as the rest. See Open Question 1.

### 3.2 Frozen payload shapes

```jsonc
// alert — produced by alerts.service.js, rendered by AlertQueue
{
  "id": "A175814…",
  "camera": { "id": "RXL-01", "name": "BOP Raxaul — Pillar 42",
              "sector": "Raxaul", "ir": false, "road": true },
  "stage": "prov",            // 'prov' | 'sealed' | 'gone'  (the client also sets 'conf')
  "time": "02:41:07",
  "reason": "Loaded vehicle moving north at 02:41, …",   // string, never null
  "confidence": 93,
  "plate": "बा १२ च ४५६७",     // or null
  "hash": "a3f91c04…e21b7d",
  "seed": 42.17,
  "channels": { "appearance": true, "motion": true }
}

// evidence block — rendered by EvidenceChainPanel
{ "n": 1041, "hash": "…", "camera": "RXL-01", "time": "14 AUG · 01:12" }

// verify response
{ "intact": true, "verified": 5, "total": 5, "blocks": [{ "n": 1041, "ok": true }] }
```

Two client-only fields, added by `useConsole` and never sent by the API: `typed` (reason
typewriter position) and `conf` (confidence bar fill). `edge/` and `backend/` must not emit them.

### 3.3 Additive-only extensions

**New optional alert fields.** All are absent from every response until the phase named. The
existing `AlertQueue` reads only `stage`, `time`, `camera.{name,id,ir}`, `seed`, `plate`,
`reason`, `conf`, `typed`, `hash` — it never enumerates keys — so an absent field is invisible
to it. Consumers added later must treat the documented default as the meaning of absence.

| Field | Type | Default when absent | Why the UI is unaffected | Phase |
|---|---|---|---|---|
| `track_id` | `integer` | `null` — "no persistent track" | not read by any current component | 5 |
| `bbox` | `[x, y, w, h]` normalised 0–1 | `null` — "no box to draw"; `LiveWall` keeps its synthetic `feed` canvas | `LiveWall` draws from `SceneCanvas`, not from alert data | 6 |
| `channel_scores` | `{ appearance: number, motion: number }` 0–1 | `null` — chips fall back to the boolean `channels`, which is frozen and still sent | `AlertQueue` renders two static chips at `stage === 'prov'` regardless | 4 |
| `rule_fired` | `string` (`"fence crossed"`, `"wrong direction"`, `"loitering"`, or `"+"`-joined) | `null` — "no named rule; confidence only" | not read today; the rule name already reaches the operator inside `reason` | 5 |
| `dori_band` | `"detect" \| "observe" \| "recognise" \| "identify"` | `null` — `LiveWall` keeps the fixed `DORI_BANDS` strip from `landingContent.js` | the strip is static per tile | 6 |
| `evidence_id` | `string` | `null` — "not sealed yet"; the frozen `hash` string remains the display value | `AlertQueue` prints `alert.hash`, not `evidence_id` | 9 |
| `clip_url` | `string` (URL) | `null` — "no clip retrievable" | no component links a clip today | 9 |
| `thumb_url` | `string` (URL) | `null` — `SceneCanvas` continues to render the synthetic feed from `seed` | the thumbnail slot is a canvas, not an `<img>` | 6 |

**New endpoints.** Purely additive; none replaces or shadows an existing path.

| Method | Path | Purpose | Auth | Phase |
|---|---|---|---|---|
| GET | `/api/stream` | Server-sent events. One `event: alert` per new or updated alert, payload = the frozen alert shape plus whichever optional fields exist. Heartbeat comment every 15 s. `useConsole` is the intended consumer *later*; the polling path stays as the fallback so the console works with the stream absent. | session token | 10 |
| POST | `/api/ingest/event` | The only way `edge/` reaches `backend/`. Body = §4's event schema. Returns `{ accepted: true, alert_id, duplicate: false }`. | `X-Truewatch-Ingest-Key` shared secret + `X-Truewatch-Timestamp` + `X-Truewatch-Nonce` | 8 |
| GET | `/api/evidence/:id` | One evidence block with its predecessor link, clip digest and sealing time. Does **not** change `GET /api/evidence/chain`. | session token | 9 |
| GET | `/api/evidence/root/:date` | The Merkle root for one day (`YYYY-MM-DD`), with leaf count and the block range it covers. | session token | 9 |

---

## 4. Internal event contract — `edge/` → `backend/`

### 4.1 Transport and authentication

```http
POST /api/ingest/event HTTP/1.1
Content-Type: application/json
X-Truewatch-Ingest-Key: <hex, from TRUEWATCH_INGEST_KEY; never hardcoded, never in git>
X-Truewatch-Timestamp: <RFC3339 UTC, request creation time>
X-Truewatch-Nonce: <uuid4, unique per request>
```

- The key lives in the environment of both processes (`TRUEWATCH_INGEST_KEY`). It is compared
  with a constant-time comparison, never with `===`.
- A request whose `X-Truewatch-Timestamp` is more than `INGEST_MAX_SKEW_SECONDS` (default 120)
  away from server time is rejected `401`.
- `X-Truewatch-Nonce` is stored for twice the skew window. A repeat is rejected `409` and the
  response carries `{ accepted: false, duplicate: true }`. This is the replay control in §9.
- Transport is plain HTTP on `localhost` in development and HTTPS in any hosted deployment. The
  shared secret is not a substitute for TLS and is not described as one.

### 4.2 Event schema

```jsonc
{
  "schema": "truewatch.event.v1",        // required, exact string; anything else → 400
  "event_id":  "uuid4",                  // required, idempotency key
  "post_id":   "RXL",                    // required
  "camera_id": "RXL-01",                 // required, must exist in the cameras table
  "captured_at": "2026-08-14T02:41:07.312Z",   // required, RFC3339 UTC, clock of the edge box

  "stage": "provisional",                // required: 'provisional' | 'confirmed' | 'sealed'

  "detection": {                         // required
    "track_id": 6,                       // integer, or null before a track forms
    "class": "person",                   // 'person'|'truck'|'car'|'two_wheeler'|'cart'
    "bbox": [0.41, 0.52, 0.06, 0.18],    // normalised [x, y, w, h], origin top-left
    "channel_scores": { "appearance": 0.87, "motion": 0.63 },   // each 0..1
    "agreed": true,                      // fusion verdict — false events are NOT sent
    "threshold": 0.55,                   // the per-camera threshold this was judged against
    "ir": true,                          // whether the source frame was LWIR
    "dori_band": "observe"               // 'detect'|'observe'|'recognise'|'identify'|null
  },

  "rule": {                              // required; null only when no named rule fired
    "fired": ["fence crossed", "loitering"],
    "baseline": { "dominant_direction": "ltr", "with": 6, "against": 4 },
    "metrics": { "dwell_s": 5.0, "net_px": 70, "path_px": 163 }
  },

  "explanation": {                       // null while stage === 'provisional'
    "text": "Loaded vehicle moving north at 02:41, outside sanctioned hours, …",
    "source": "vlm",                     // 'vlm' | 'template'  — see risk 4
    "model": "moondream2",
    "latency_ms": 2840
  },

  "anpr": {                              // null when no plate was read
    "text": "बा १२ प १२३४",
    "script": "devanagari",              // 'devanagari' | 'latin'
    "line_confidences": [0.989, 0.990]
  },

  "evidence": {                          // null while stage !== 'sealed'
    "evidence_id": "ev_01J…",
    "block_n": 1041,
    "clip_sha256": "64 hex chars",
    "prev_sha256": "64 hex chars, or 64 zeros for the genesis block",
    "clip_url": "http://…/clips/ev_01J….mp4",
    "thumb_url": "http://…/clips/ev_01J….jpg",
    "sealed_at": "2026-08-14T02:41:11.004Z"
  },

  "confidence": 93,                      // required, integer 0..100, fused
  "source": "rtsp"                       // 'rtsp' | 'file'  — honesty flag, see PHASE_MINUS1_SCOPE §6
}
```

`explanation.source` exists so the console can never present template output as model output.
`source: "file"` exists so a replayed file is never presented as a live camera. Both are required
by `PHASE_MINUS1_SCOPE` §6 and risk 4.

### 4.3 The mapping function

`backend/src/services/ingest.service.js` holds exactly one exported mapper. Nothing else in the
backend may construct an alert from an event.

```js
// backend/src/services/ingest.service.js   — Phase 8
// Converts a truewatch.event.v1 into the FROZEN alert shape.
// Every frozen key is produced unconditionally. Every additive key is omitted when unknown,
// never emitted as undefined.
export const mapEventToAlert = (event, camera) => {
  const frozen = {
    id:         `A${event.event_id.replace(/-/g, '').slice(0, 16)}`,
    camera,                                   // the full row: { id, name, sector, ir, road }
    stage:      STAGE[event.stage],           // provisional→'prov', confirmed→'conf', sealed→'sealed'
    time:       hhmmss(event.captured_at),    // 'HH:MM:SS' — matches utils/random.js stamp()
    reason:     event.explanation?.text ?? reasonFallback(event),   // never null: AlertQueue slices it
    confidence: event.confidence,
    plate:      event.anpr?.text ?? null,
    hash:       shortHash(event.evidence?.clip_sha256),  // 'aaaaaaaa…bbbbbb', or a pending marker
    seed:       seedFrom(event.event_id),     // stable 0..100 float; SceneCanvas needs a number
    channels: {
      appearance: (event.detection.channel_scores.appearance ?? 0) >= event.detection.threshold,
      motion:     (event.detection.channel_scores.motion     ?? 0) >= event.detection.threshold,
    },
  };

  const additive = {};
  if (event.detection.track_id      != null) additive.track_id       = event.detection.track_id;
  if (event.detection.bbox          != null) additive.bbox           = event.detection.bbox;
  if (event.detection.channel_scores!= null) additive.channel_scores = event.detection.channel_scores;
  if (event.detection.dori_band     != null) additive.dori_band      = event.detection.dori_band;
  if (event.rule?.fired?.length)            additive.rule_fired      = event.rule.fired.join(' + ');
  if (event.evidence?.evidence_id   != null) additive.evidence_id    = event.evidence.evidence_id;
  if (event.evidence?.clip_url      != null) additive.clip_url       = event.evidence.clip_url;
  if (event.evidence?.thumb_url     != null) additive.thumb_url      = event.evidence.thumb_url;

  return { ...frozen, ...additive };
};
```

Three invariants this mapper must hold, and which Phase 8 tests assert:

1. `reason` is always a non-empty string. `AlertQueue` calls `alert.reason.slice(0, alert.typed)`;
   `null` would throw and blank the queue.
2. `seed` is always a finite number. `SceneCanvas` passes it into the `feed` scene.
3. `camera` is always the full camera row, never a bare id. `AlertQueue` reads
   `alert.camera.name`, `alert.camera.id` and `alert.camera.ir`.

---

## 5. Target repo layout at the end of Phase 12

Phase in brackets is the phase that **creates** the file. `[0]` means this Phase 0 commit.

```text
truewatch/
├── package.json                                  [exists] workspaces: frontend, backend
├── README.md                                     [exists] latency wording updated in [11]
├── CONTRIBUTING.md                               [exists]
├── LICENSE                                       [1] see Open Question 3
├── .env.example                                  — per workspace, see below
│
├── docs/
│   ├── ARCHITECTURE_V2.md                        [0] this file
│   ├── PHASE_MINUS1_SCOPE.md                     [-1]
│   ├── MEASUREMENTS.md                           [0] the measured record; see §7
│   ├── THREAT_MODEL.md                           [9] expands §9 below
│   └── images/truewatch-automation-flow.png      [exists]
│
├── edge/                                         Python 3.11 · FastAPI
│   ├── __init__.py                               [0]
│   ├── app.py                                    [1] FastAPI app, /healthz, /pipeline/start
│   ├── config.py                                 [1] env only, no literals
│   ├── requirements.txt                          [0]
│   ├── Dockerfile                                [0] HF Spaces target
│   ├── README.md                                 [0]
│   ├── .env.example                              [0]
│   ├── pipeline/
│   │   ├── __init__.py                           [0]
│   │   ├── ingest.py                             [1] ONVIF/RTSP pull + file replay, one decode path
│   │   ├── appearance.py                         [2] YOLO11-s ONNX, appearance channel
│   │   ├── motion.py                             [3] Farneback flow, object-vs-background contrast
│   │   ├── fusion.py                             [4] per-camera threshold, night weighting
│   │   ├── track.py                              [5] ByteTrack, persistent track ids
│   │   ├── tiling.py                             [6] SAHI tiled inference on the far-field strip
│   │   ├── anpr.py                               [7] PP-OCRv5 Devanagari + Latin comparison
│   │   ├── face.py                               [7] YuNet, blurred on egress
│   │   ├── explain.py                            [7] Moondream 2, async, template fallback
│   │   └── emit.py                               [8] POSTs truewatch.event.v1 to the backend
│   ├── rules/
│   │   ├── __init__.py                           [0]
│   │   ├── engine.py                             [5] fence / wrong direction / loitering
│   │   ├── baseline.py                           [5] per-camera baseline, updated on dismiss
│   │   └── homography.py                         [6] pixel thresholds → metres (slide 4)
│   ├── evidence/
│   │   ├── __init__.py                           [0]
│   │   ├── hasher.py                             [9] SHA-256 at capture
│   │   ├── chain.py                              [9] append-only link, verify walk
│   │   └── merkle.py                             [9] daily root
│   ├── models/
│   │   ├── __init__.py                           [0]
│   │   ├── registry.py                           [2] resolves weights by id; no weights in git
│   │   └── .gitignore                            [0] ignores *.onnx, *.pt, *.engine
│   └── tests/
│       ├── __init__.py                           [0]
│       ├── test_fusion.py                        [4]
│       ├── test_rules.py                         [5] reproduces MEASUREMENTS §2 byte-for-byte
│       ├── test_chain.py                         [9] tamper test: one byte → non-zero exit
│       └── test_event_contract.py                [8] schema round-trip against §4.2
│
├── training/                                     runs on Kaggle, not in CI
│   ├── README.md                                 [0] AGPL-3.0 note for Ultralytics
│   ├── finetune_yolo11s.py                       [2] IDD / LLVIP / KAIST
│   ├── export_onnx.py                            [2]
│   └── notebooks/kaggle_finetune.ipynb           [2]
│
├── datasets/                                     scripts only — no data, ever
│   ├── README.md                                 [1]
│   ├── fetch_kaist.py                            [1]
│   ├── fetch_llvip.py                            [1]
│   ├── fetch_idd.py                              [1]
│   └── make_synthetic_plates.py                  [7] the ≥20-plate synthetic set
│
├── backend/
│   ├── .env.example                              [exists] extended [0]
│   └── src/
│       ├── app.js  server.js  config/index.js    [exists] config extended [8]
│       ├── routes/       …ingest.routes.js       [8]   stream.routes.js [10]
│       ├── controllers/  …ingest.controller.js   [8]   stream.controller.js [10]
│       ├── services/     …ingest.service.js      [8]   mapEventToAlert lives here
│       ├── middleware/    ingestAuth.js          [8]   shared secret, skew, nonce
│       ├── db/
│       │   ├── .gitkeep                          [0]
│       │   ├── index.js                          [8] node:sqlite connection, WAL on
│       │   ├── schema.sql                        [8] §6 below
│       │   └── migrate.js                        [8]
│       ├── repositories/
│       │   ├── .gitkeep                          [0]
│       │   ├── alerts.repository.js              [8]
│       │   ├── cameras.repository.js             [8]
│       │   ├── evidence.repository.js            [9]
│       │   └── audit.repository.js               [9]
│       ├── realtime/
│       │   ├── .gitkeep                          [0]
│       │   └── hub.js                            [10] SSE fan-out
│       └── data/mockData.js                      [exists] stays as the outage fallback
│
└── frontend/                                     UNCHANGED through Phase 12
```

Nothing under `frontend/` appears in this tree with a phase number. That is deliberate.

---

## 6. Data model

SQLite, one file per post, `node:sqlite` (built into Node 20, no dependency added).
Write-ahead logging on; `foreign_keys` on.

```sql
-- backend/src/db/schema.sql   [Phase 8]

CREATE TABLE cameras (
  id        TEXT PRIMARY KEY,             -- 'RXL-01'
  name      TEXT NOT NULL,                -- 'BOP Raxaul — Pillar 42'
  sector    TEXT NOT NULL,
  ir        INTEGER NOT NULL DEFAULT 0,   -- 0/1, serialised to boolean at the edge of the API
  road      INTEGER NOT NULL DEFAULT 1,
  post_id   TEXT REFERENCES posts(id),
  rtsp_url  TEXT,                         -- never returned by /api/cameras
  dori_band TEXT                          -- 'detect'|'observe'|'recognise'|'identify'
);

CREATE TABLE posts (
  id     TEXT PRIMARY KEY,                -- 'RXL'
  name   TEXT NOT NULL,
  x      REAL NOT NULL,                   -- map percentage, as SectorMap expects
  y      REAL NOT NULL,
  road   INTEGER NOT NULL DEFAULT 1,
  alert  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE alerts (
  id             TEXT PRIMARY KEY,        -- the frozen 'A…' id
  event_id       TEXT NOT NULL UNIQUE,    -- ingest idempotency
  camera_id      TEXT NOT NULL REFERENCES cameras(id),
  stage          TEXT NOT NULL CHECK (stage IN ('prov','conf','sealed','gone')),
  captured_at    TEXT NOT NULL,           -- RFC3339 UTC, the sortable truth
  time_label     TEXT NOT NULL,           -- 'HH:MM:SS', the frozen display string
  reason         TEXT NOT NULL,           -- never null — AlertQueue slices it
  reason_source  TEXT NOT NULL DEFAULT 'template' CHECK (reason_source IN ('vlm','template')),
  confidence     INTEGER NOT NULL CHECK (confidence BETWEEN 0 AND 100),
  plate          TEXT,                    -- nullable, Devanagari or Latin
  plate_script   TEXT,                    -- 'devanagari'|'latin'|NULL
  hash           TEXT NOT NULL,           -- the frozen short display hash
  seed           REAL NOT NULL,           -- SceneCanvas needs a finite number
  ch_appearance  INTEGER NOT NULL,        -- frozen channels.appearance
  ch_motion      INTEGER NOT NULL,        -- frozen channels.motion
  -- additive, all nullable
  track_id       INTEGER REFERENCES tracks(id),
  bbox           TEXT,                    -- JSON '[x,y,w,h]'
  score_appearance REAL,
  score_motion     REAL,
  rule_fired     TEXT,
  dori_band      TEXT,
  evidence_id    TEXT REFERENCES evidence_blocks(evidence_id),
  clip_url       TEXT,
  thumb_url      TEXT,
  ingest_source  TEXT NOT NULL DEFAULT 'file' CHECK (ingest_source IN ('rtsp','file'))
);
CREATE INDEX idx_alerts_captured   ON alerts(captured_at DESC);
CREATE INDEX idx_alerts_camera     ON alerts(camera_id, captured_at DESC);
CREATE INDEX idx_alerts_stage      ON alerts(stage, captured_at DESC);

CREATE TABLE tracks (
  id           INTEGER PRIMARY KEY,
  camera_id    TEXT NOT NULL REFERENCES cameras(id),
  class        TEXT NOT NULL,
  first_frame  INTEGER NOT NULL,
  last_frame   INTEGER NOT NULL,
  dwell_s      REAL NOT NULL,
  net_px       REAL NOT NULL,
  path_px      REAL NOT NULL,
  direction    TEXT,                      -- 'ltr'|'rtl'
  created_at   TEXT NOT NULL
);
CREATE INDEX idx_tracks_camera ON tracks(camera_id, created_at DESC);

CREATE TABLE evidence_blocks (
  n            INTEGER PRIMARY KEY,       -- the frozen block number
  evidence_id  TEXT NOT NULL UNIQUE,
  camera_id    TEXT NOT NULL REFERENCES cameras(id),
  time_label   TEXT NOT NULL,             -- '14 AUG · 01:12', the frozen display string
  display_hash TEXT NOT NULL,             -- the frozen short 'aaaaaaaa…bbbbbb' form
  clip_sha256  TEXT NOT NULL,             -- full 64 hex
  prev_sha256  TEXT NOT NULL,             -- 64 zeros at genesis
  block_sha256 TEXT NOT NULL UNIQUE,      -- SHA-256(prev || clip || camera || sealed_at)
  clip_path    TEXT NOT NULL,
  sealed_at    TEXT NOT NULL,
  alert_id     TEXT REFERENCES alerts(id)
);
CREATE INDEX idx_evidence_sealed ON evidence_blocks(sealed_at DESC);
CREATE INDEX idx_evidence_camera ON evidence_blocks(camera_id, n);

CREATE TABLE daily_roots (
  day         TEXT PRIMARY KEY,           -- 'YYYY-MM-DD'
  merkle_root TEXT NOT NULL,
  leaf_count  INTEGER NOT NULL,
  first_n     INTEGER NOT NULL,
  last_n      INTEGER NOT NULL,
  computed_at TEXT NOT NULL,
  signature   TEXT,                       -- NULL: HQ signing is described, not run (scope §6)
  signed_at   TEXT
);

CREATE TABLE users (
  id            TEXT PRIMARY KEY,         -- official id
  name          TEXT NOT NULL,
  unit          TEXT NOT NULL,
  email         TEXT,
  role          TEXT NOT NULL CHECK (role IN ('Operator','Supervisor','HQ Analyst')),
  initials      TEXT NOT NULL,
  password_hash TEXT NOT NULL,            -- never a plaintext column
  approved      INTEGER NOT NULL DEFAULT 0,
  created_at    TEXT NOT NULL
);

CREATE TABLE audit_log (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  at         TEXT NOT NULL,
  actor_id   TEXT REFERENCES users(id),
  action     TEXT NOT NULL,               -- 'alert.act'|'alert.dismiss'|'face.unmask'|'evidence.verify'
  subject    TEXT NOT NULL,               -- alert id, evidence id, …
  detail     TEXT,                        -- JSON
  prev_sha256 TEXT NOT NULL,              -- the log is itself chained (threat 6)
  sha256      TEXT NOT NULL UNIQUE
);
CREATE INDEX idx_audit_at    ON audit_log(at DESC);
CREATE INDEX idx_audit_actor ON audit_log(actor_id, at DESC);
```

**Coverage of the frozen shapes.** `alert.id → alerts.id`; `alert.camera.{id,name,sector,ir,road}
→ cameras`; `stage → alerts.stage`; `time → alerts.time_label`; `reason → alerts.reason`;
`confidence → alerts.confidence`; `plate → alerts.plate`; `hash → alerts.hash`;
`seed → alerts.seed`; `channels.{appearance,motion} → alerts.{ch_appearance,ch_motion}`.
Evidence block `{n, hash, camera, time} → evidence_blocks.{n, display_hash, camera_id,
time_label}`. Verify `{intact, verified, total, blocks[{n, ok}]}` is computed by walking
`evidence_blocks` ordered by `n` and comparing each `prev_sha256` with its predecessor's
`block_sha256`; nothing is stored for it.

**Why SQLite, not Postgres, for a six-day free build.** One: the deployment unit is one board at
one post with one writer, which is precisely SQLite's shape — Postgres would add a server to
operate for no concurrency that exists. Two: `node:sqlite` is in Node 20's standard library, so
the dependency count stays at zero and free-tier hosting needs no database add-on. Three: the
database is a file, so the tamper demonstration (alter one byte, verify fails) and the reset
between demo runs are both a file copy.

---

## 7. Latency and honesty budget

### 7.1 What the deck actually claims

Slide 3, box *"Design estimates, not yet measured on hardware"*, verbatim:
*"4 x 1080p at 8 to 10 FPS with the VLM on alerts only; 2 streams if face and OCR run
continuously. The 30 ms figure is the YOLO11-s INT8 inference budget at 640 px, not end-to-end
latency."* Hardware, same slide: *"Jetson Orin Nano Super 8 GB, one board per post."*

So both headline numbers are **hardware targets on a Jetson Orin Nano Super 8 GB that this
project does not own**:

| Figure | What it is | What it is not |
|---|---|---|
| **~30 ms** | the YOLO11-s **INT8 inference budget at 640 px** on Jetson Orin Nano Super | not end-to-end latency, not measured, not achieved here |
| **~3 s** | the asynchronous explanation budget for Moondream 2 on the same board | not measured, and asynchronous by design — the operator sees the provisional alert first |

`PHASE_MINUS1_SCOPE` §5.2 already files both under TARGET, and §6 files Jetson benchmarking and
TensorRT INT8 quantisation as out of scope. Nothing below changes that.

### 7.2 What to expect on the hardware that exists

Ranges, not promises. Each is to be replaced by a measured number, labelled with its host, as
Phases 2–7 land. No number here may be attributed to a Jetson.

| Stage | (a) Apple Silicon, CPU / MPS | (b) HF Spaces free CPU (2 vCPU) |
|---|---|---|
| YOLO11-s ONNX, 640 px, FP32, one frame | tens of ms to low hundreds of ms | several hundred ms; often >1 s under contention |
| Farneback dense flow, 640 px pair | low tens of ms | high tens to low hundreds of ms |
| Fusion + ByteTrack + rules | sub-millisecond to low ms | low ms |
| **Provisional alert, end-to-end** | well above the 30 ms Jetson detector budget | further above again |
| Moondream 2 explanation | seconds, plausibly tens of seconds | likely unusable; risk 4's template fallback is the expected path |
| Cold start after idle | n/a | tens of seconds on the free tier |

### 7.3 Required README wording

The README must carry this, or wording that keeps every one of its distinctions:

> **Performance figures.** The ~30 ms detector figure is the YOLO11-s INT8 inference budget at
> 640 px on a Jetson Orin Nano Super 8 GB, as stated on slide 3 of the submission deck. It is a
> **design target on hardware this project does not own**, and it is **not** end-to-end latency.
> The ~3 s explanation figure is likewise a target, and the explanation is asynchronous — the
> operator sees the provisional alert before it arrives. Every number this repository presents as
> measured was produced on the host named beside it, on public datasets, and is recorded in
> `MEASUREMENTS.md`. No measurement in this repository was taken on a Jetson.

Three phrases that must never appear: "30 ms end-to-end", "real-time on any hardware", and any
Jetson number without the word *target* or *estimate* attached.

---

## 8. Configuration and secrets

No secret value appears in this repository. Every `.env.example` below carries names and safe
defaults only. `.gitignore` already excludes `**/.env` and `**/.env.*` while keeping
`**/.env.example`.

### 8.1 `backend/.env.example`

```ini
NODE_ENV=development
PORT=4000
CORS_ORIGIN=http://localhost:5173

# --- Persistence (Phase 8) ---
DATABASE_PATH=./var/truewatch.db

# --- Ingest from the edge service (Phase 8) ---
# Shared secret. Generate per deployment: openssl rand -hex 32
TRUEWATCH_INGEST_KEY=
INGEST_MAX_SKEW_SECONDS=120
INGEST_NONCE_TTL_SECONDS=240

# --- Live stream (Phase 10) ---
STREAM_HEARTBEAT_SECONDS=15

# --- Session tokens (Phase 8) ---
JWT_SECRET=
JWT_EXPIRES_IN=12h

# --- Evidence (Phase 9) ---
EVIDENCE_STORE_PATH=./var/evidence

# --- Edge service, for health reporting only ---
EDGE_BASE_URL=http://localhost:8000
```

### 8.2 `edge/.env.example`

```ini
EDGE_HOST=0.0.0.0
EDGE_PORT=8000
EDGE_LOG_LEVEL=info

# --- Where events go (Phase 8) ---
BACKEND_BASE_URL=http://localhost:4000
# Must match TRUEWATCH_INGEST_KEY in backend/.env exactly.
TRUEWATCH_INGEST_KEY=

# --- Ingest source (Phase 1) ---
# 'rtsp' or 'file'. Reported on every event as source, so a replayed file is never
# presented as a live camera.
INGEST_MODE=file
RTSP_URL=
REPLAY_FILE_PATH=./var/samples/kaist_set00_v007.mp4
TARGET_FPS=8

# --- Models (Phase 2). Ids only; weights are never committed. ---
YOLO_MODEL_ID=
YOLO_IMG_SIZE=640
ONNX_PROVIDER=CPUExecutionProvider
OCR_MODEL_ID=
FACE_MODEL_ID=
VLM_MODEL_ID=
EMBED_MODEL_ID=
HF_TOKEN=

# --- Fusion and rules (Phases 4, 5) ---
FUSION_THRESHOLD_DEFAULT=0.55
NIGHT_MOTION_WEIGHT=0.65
LOITER_DWELL_SECONDS=3
LOITER_NET_PX=90
FENCE_X=320

# --- Evidence (Phase 9) ---
EVIDENCE_DIR=./var/evidence
CLIP_SECONDS=6
```

### 8.3 `frontend/.env.example` — unchanged

```ini
VITE_API_BASE_URL=/api
VITE_API_PROXY_TARGET=http://localhost:4000
VITE_PORT=5173
```

No new frontend variable is introduced. The SSE endpoint is reached through the same
`VITE_API_BASE_URL`, so the console needs no configuration change to consume it later.

### 8.4 Rules

- `TRUEWATCH_INGEST_KEY` and `JWT_SECRET` ship **empty** in both examples. A process that finds
  them empty in `NODE_ENV=production` must refuse to start rather than fall back to a default.
  The current `config/index.js` default `'dev-only-insecure-secret'` is acceptable in development
  and must be made fatal in production in Phase 8.
- `HF_TOKEN` is read from the environment only. Model weights are referenced by id and never
  committed — see Open Question 4.

---

## 9. Threat model

Theme: Blockchain & Cybersecurity. Six threats, each with the control that answers it and the
phase that builds the control. `docs/THREAT_MODEL.md` expands this in Phase 9.

| # | Threat | Attack in one line | Control | Phase |
|---|---|---|---|---|
| 1 | **Evidence tampering** — a stored clip or its record is altered after sealing | Someone with file access edits a clip, or rewrites a row, to change what the record says | SHA-256 at capture, not afterwards (slide 3). Each block stores `prev_sha256` and a `block_sha256` over `prev ‖ clip ‖ camera ‖ sealed_at`, so the chain is append-only; a day folds into one Merkle root. `verify` walks the chain, exits non-zero and names the first broken block. The tamper test — alter one byte, watch it fail — is part of the demo. | 9 |
| 2 | **Clip substitution** — a genuine block is kept but a different clip is swapped behind it | Replace the file at `clip_path` with different footage while leaving the row untouched | The digest is over the clip bytes, not the path. Verification re-hashes the file and compares with `clip_sha256`; a substituted clip fails even though the chain links are intact, and the failure names the block. `clip_path` is never accepted from a request body. | 9 |
| 3 | **Replay of ingest events** — a captured event POST is resent to manufacture or duplicate alerts | Someone who observes one `/api/ingest/event` request resends it, or replays a night's worth of events | Three layers: `event_id` is `UNIQUE` on `alerts`, so a duplicate is a no-op returning `{ duplicate: true }`; `X-Truewatch-Timestamp` outside `INGEST_MAX_SKEW_SECONDS` is rejected; `X-Truewatch-Nonce` is remembered for `INGEST_NONCE_TTL_SECONDS` and a repeat is rejected `409`. | 8 |
| 4 | **Operator repudiation** — an operator denies having sealed, dismissed or unmasked | "I never dismissed that alert" / "I never unmasked that face", with no way to settle it | `audit_log` records actor, action, subject and time for every `alert.act`, `alert.dismiss`, `face.unmask` and `evidence.verify`, and the log is itself hash-chained (`prev_sha256`, `sha256`), so an entry cannot be removed without breaking the log. Face unmasking is supervisor-gated (slide 4: *"unsealing is access-controlled"*). | 9 |
| 5 | **Forged ingest** — anything other than the post's own edge service posts events | An attacker on the network, or a malicious container, POSTs fabricated alerts | `X-Truewatch-Ingest-Key` compared in constant time, plus threat 3's timestamp and nonce. `camera_id` must resolve to a known camera. HTTPS in any hosted deployment; the shared secret is explicitly not a substitute for transport security. | 8 |
| 6 | **Privacy leak of face and plate data** — biometric data leaves the post or outlives its retention | Face crops or plate images exported to the console, a log, or a bug report | Slide 4: *"face data stays on the post, 30-day retention, unsealing is access-controlled."* Face crops are blurred on egress; the unblurred crop never leaves `edge/`. Retention is enforced by a scheduled purge, and each unmask is an `audit_log` entry under threat 4. `rtsp_url` is never returned by `/api/cameras`. | 7, 9 |

---

## 10. Build order

Phases 1–12 below are the sequence this document assumes. See Open Question 5 — these names are
proposed here, not handed down.

```text
        ┌──────────────────────────────────────────────────────────────┐
        │  P0  architecture + scaffolding          ← this commit        │
        └───────────────────────────┬──────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
     ┌─────────────┐        ┌──────────────┐        ┌──────────────────┐
     │ P1 ingest   │        │ P2 detector  │        │ P8 backend       │
     │ + datasets  │        │ + training   │        │ persistence      │
     │ + licence   │        │ (KAGGLE)     │        │ + /api/ingest    │
     └──────┬──────┘        └──────┬───────┘        └────────┬─────────┘
            │                      │                          │
            │              ┌───────┴────────┐                 │
            ▼              ▼                ▼                 ▼
     ┌─────────────┐  ┌──────────┐   ┌─────────────┐   ┌──────────────┐
     │ P3 motion   │  │ P6 tiling│   │ P7 anpr /   │   │ P10 SSE      │
     │   channel   │  │  + DORI  │   │ face / vlm  │   │   stream     │
     └──────┬──────┘  └────┬─────┘   └──────┬──────┘   └──────┬───────┘
            │              │                │                 │
            ▼              │                │                 │
     ┌─────────────┐       │                │                 │
     │ P4 fusion   │       │                │                 │
     └──────┬──────┘       │                │                 │
            ▼              │                │                 │
     ┌─────────────┐       │                │                 │
     │ P5 tracking │       │                │                 │
     │  + rules    │       │                │                 │
     └──────┬──────┘       │                │                 │
            └──────────────┴────────┬───────┴─────────────────┘
                                    ▼
                           ┌──────────────────┐
                           │ P9 evidence      │
                           │ chain + audit    │
                           └────────┬─────────┘
                                    ▼
                           ┌──────────────────┐
                           │ P11 deploy +     │
                           │ README honesty   │
                           └────────┬─────────┘
                                    ▼
                           ┌──────────────────┐
                           │ P12 demo rehearse│
                           └──────────────────┘
```

| Phase | Name | Blocked by | Blocks |
|---|---|---|---|
| 1 | Ingest path, dataset scripts, licence decision | 0 | 2, 3 |
| 2 | Appearance channel + Kaggle fine-tune + ONNX export | 1 | 4, 6 |
| 3 | Motion channel | 1 | 4 |
| 4 | Fusion, per-camera threshold, night weighting | 2, 3 | 5 |
| 5 | ByteTrack + rule engine + baseline | 4 | 9 |
| 6 | Tiled far-field inference + DORI grading + homography | 2 | 9 |
| 7 | ANPR, face, explanation | 2 | 9 |
| 8 | SQLite, repositories, `/api/ingest/event` | 0 | 9, 10 |
| 9 | Evidence chain, Merkle root, audit log, tamper test | 5, 6, 7, 8 | 11 |
| 10 | SSE stream, console fed live | 8 | 11 |
| 11 | Deployment + README honesty pass | 9, 10 | 12 |
| 12 | Demo rehearsal against `PHASE_MINUS1_SCOPE` §7 | 11 | — |

**What runs in parallel while a Kaggle job trains.** Phase 2's training job occupies the GPU and
nothing else. While it runs, all of the following are unblocked and touch no shared file:

- **Phase 8** (backend persistence and `/api/ingest/event`) — depends only on Phase 0.
- **Phase 3** (motion channel) — depends on Phase 1 only; the measured Farneback work in
  `MEASUREMENTS.md` §1 needs no fine-tuned weights.
- **Phase 10** (SSE) once Phase 8 lands.
- **Phase 9's** chain and audit code, which is pure hashing and testable with fixture clips.

Phase 2 is the only phase that consumes the 30 GPU-hours-per-week Kaggle budget. Risk 3 in
`PHASE_MINUS1_SCOPE` applies: checkpoint every epoch, and keep the pretrained-only results as the
shipping baseline so a lost run costs nothing already promised.

---

## 11. Open questions

Four of the eight below are resolved. They are kept on the record rather than deleted, so each
decision and its reason stay visible. Four remain open, and two of those — the licence and the
weights location — block Phase 2.

1. **RESOLVED — the three endpoints missing from the frozen list are frozen too.**
   `GET /api/alerts/budget`, `GET /api/cameras/sectors` and `GET /api/cameras/:id` are live in
   `backend/src/routes/` but were absent from the Phase 0 brief's frozen inventory. They are
   frozen on the same terms as the rest, and §3.1 is the inventory of record.
   **Still open:** `fetchAlerts`, `fetchSectors` and `fetchTrafficSample`'s siblings exist in
   `frontend/src/services/` with no component consuming them, and `GET /api/alerts` is therefore
   dead in the UI today — `useConsole` builds its queue from `POST /api/alerts/simulate` only.
   Confirm whether Phase 10 should switch the console to `GET /api/alerts` + SSE, which would be
   a `useConsole` change and so is **outside** the "frontend unchanged" rule as written.
2. **RESOLVED — the measured record is in the repository** at `docs/MEASUREMENTS.md`, committed
   in Phase 0. Every honesty claim in §7 and in `PHASE_MINUS1_SCOPE` §5.1 now points at a file a
   judge can open. Its header gained a cross-reference to §7 and to the measured-versus-target
   split; **no number in it was altered.**
3. **Licence.** `PHASE_MINUS1_SCOPE` risk 1 and open question 4 are still open: Ultralytics
   YOLO11 is AGPL-3.0 and this repository is public. The decision is needed **before the first
   commit that imports Ultralytics**, which is Phase 2. No `LICENSE` file exists today.
4. **Where fine-tuned weights live.** `PHASE_MINUS1_SCOPE` open question 5. §5 assumes a Hugging
   Face model repo referenced by id, with `edge/models/.gitignore` excluding `*.onnx`, `*.pt` and
   `*.engine`. Confirm.
5. **RESOLVED — the Phase 1–12 sequence in §10 is adopted as the plan of record.** The brief
   named Phase 8 and Phase 12 but supplied no list. Every "Phase N" annotation in §3.3, §5 and §9
   keys off §10. Re-scoping a phase later means updating §5's per-file column in the same commit.
6. **RESOLVED — Phase -1 is merged.** `chore/phase-minus1-scope-lock` was merged into `main`
   ahead of this branch, so `docs/PHASE_MINUS1_SCOPE.md` is on `main` and every reference to it
   from this document resolves.
7. **`PHASE_MINUS1_SCOPE` open questions 1, 2, 3 and 6 are still unanswered** — the "cart" class,
   the two capabilities with no console surface, which border the demo claims, and the
   8-cameras-versus-6-cameras figure. §7 of that document assumes option (i) for the console
   surfaces. Nothing in this architecture forecloses any of those answers, but open question 2
   there becomes urgent at Phase 6, where a fence-drawing overlay would be a `frontend/` change.
8. **The PDF is silent on the wire format between the board and the control room.** Slide 3 ends
   at *"Event log goes to the existing control room"* without naming a protocol. §4's
   `truewatch.event.v1` over HTTPS with a shared secret is this document's choice, not the deck's.
