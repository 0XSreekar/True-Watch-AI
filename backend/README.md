# TRUE WATCH — backend

Express API for the operator console. It persists to SQLite, accepts detections from the edge
service, streams them live, and serves the frozen contract the console was built against.

```bash
npm install                    # repo root (npm workspaces)
node ../scripts/setup-env.mjs  # copies backend/.env.example to backend/.env
npm run dev --workspace truewatch-backend
npm test --workspace truewatch-backend
```

A fresh clone runs with no edge service and no secrets: the first boot creates
`backend/var/truewatch.db` (git-ignored), applies the migrations and seeds it from
`src/data/mockData.js` (cameras, posts, a five-block hash chain, demo analytics history and one demo
account per role). `POST /api/alerts/simulate` is the demo path for producing alerts without the edge.

## Layout

```
src/config      the only reader of process.env; startup validation
src/db          better-sqlite3, ordered migrations, first-boot seed
src/repositories  one per table; the only code that writes SQL
src/services    business rules; depend on repositories and adapters, never on Express
src/controllers HTTP shape only; never touch the database
src/routes      wiring and per-route guards
src/adapters    edge HTTP client, evidence verifier
src/realtime    in-process bus and the SSE hub
src/middleware  auth guards, request id, access log, error handler
```

## The frozen contract

Unchanged paths, methods, request and response shapes. `tests/contract.test.js` asserts the exact key
set of every one and fails loudly if any drifts.

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | `{ status, service, time }` (not wrapped) |
| POST | `/api/auth/login`, `/api/auth/register` | same bodies and responses as before |
| GET | `/api/auth/units`, `/api/auth/roles` | |
| GET | `/api/cameras`, `/api/cameras/posts`, `/api/cameras/sectors`, `/api/cameras/:id` | |
| GET | `/api/alerts`, `/api/alerts/budget` | `GET /api/alerts` also carries a sibling `budget` |
| POST | `/api/alerts/simulate` | demo path (fixture data) |
| POST | `/api/alerts/:id/act`, `/api/alerts/:id/dismiss` | act seals; dismiss removes and notifies the edge |
| GET | `/api/search/placeholders` | |
| POST | `/api/search` | proxied to the edge when `EDGE_BASE_URL` is set, else fixtures |
| GET | `/api/evidence/chain` | latest 20 blocks, ascending |
| POST | `/api/evidence/verify` | edge verifier if it answers, else a local walk |
| GET | `/api/analytics`, `/api/analytics/traffic` | computed from alert and track rows |

Envelope: success `{ data }`, failure `{ error }`. Alerts carry the frozen keys always and the additive
keys (`track_id`, `bbox`, `channel_scores`, `rule_fired`, `dori_band`, `evidence_id`, `clip_url`,
`thumb_url`, and the honesty flags `reason_source`, `source` on edge alerts) only when known.

## New endpoints

### `POST /api/ingest/event` — edge to backend

Header `X-Ingest-Key: <TRUEWATCH_INGEST_KEY>` (constant-time compare). `X-Truewatch-Ingest-Key` is
accepted as an alias. Optional `X-Truewatch-Timestamp` (RFC 3339, within `INGEST_MAX_SKEW_SECONDS`) and
`X-Truewatch-Nonce` (remembered `INGEST_NONCE_TTL_SECONDS`; a repeat is 409); set
`INGEST_REQUIRE_REPLAY_HEADERS=true` to make both mandatory.

Body: one `truewatch.event.v1` (docs/ARCHITECTURE_V2.md 4.2), validated strictly; unknown fields at any
level are 400. `camera_id` must exist and belong to `post_id`.

| Status | Body |
|---|---|
| 201 | `{ accepted: true, alert_id, duplicate: false }` (plus `merged: true` when it enriched an existing alert) |
| 200 | `{ accepted: true, alert_id, duplicate: true }` — same `event_id` again; nothing written |
| 400 | `{ error, details: ["$.path: reason"] }` — details never echo values |
| 401 | missing or wrong key, stale timestamp |
| 409 | `{ accepted: false, duplicate: true }` — nonce reused |
| 429 | rate limited (`INGEST_RATE_LIMIT_PER_MINUTE`) |

Idempotency is on `event_id` (ledger table, so it holds for merged events too). The edge emits several
events for one track (fusion, a rule, a plate); a new event on a track that already has a live alert
today enriches that alert (plate, rule, reason) instead of raising a second one.

### `POST /api/ingest/update` — the reason patch

Same key and replay headers. Body `truewatch.update.v1`:

```jsonc
{
  "schema": "truewatch.update.v1",
  "event_id": "<the event_id already ingested>",
  "stage": "confirmed",                       // optional: 'confirmed' | 'sealed'; only moves forward
  "explanation": { "text": "…", "source": "vlm", "model": "moondream2", "latency_ms": 2840 },
  "anpr": { "text": "BR 06 GA 4821", "script": "latin" },
  "evidence": { "evidence_id": "…", "block_n": 1046, "clip_sha256": "64 hex", "prev_sha256": "64 hex",
                "clip_url": "http://…", "thumb_url": "http://…", "sealed_at": "RFC3339Z" }
}
```

At least one of `stage`, `explanation`, `anpr`, `evidence`; `stage: "sealed"` requires `evidence`, which
is stored as the next chain block. Response `200 { accepted, alert_id, updated }`; replaying an identical
update is `updated: false` and publishes nothing. A template reason never overwrites a model reason.
404 unknown `event_id`, 409 if the alert was dismissed.

### `GET /api/stream` — server-sent events

```
event: alert.created   data: <full alert>
event: alert.updated   data: { id, patch: { reason, reason_source, plate, hash, … }, alert }
event: alert.stage     data: { id, stage, previous, alert }
```

`id:` is a per-process counter. A `: heartbeat` comment every `STREAM_HEARTBEAT_SECONDS` (15), a
`retry: 3000` hint, and a final `event: shutdown` when the server stops. Not buffered: a client that
stops reading is dropped. Polling `GET /api/alerts` is unchanged.

### Auth (additive)

- `POST /api/auth/refresh` — reads the `tw_refresh` httpOnly cookie, rotates it, answers the same body as
  login. A reused refresh token revokes the user's whole session family. 401 `{ error }` otherwise.
- `POST /api/auth/logout` — revokes the refresh token, clears the cookie, `{ data: { ok: true } }`.
- `GET /api/auth/me` — `{ data: { user } }`, needs a token.
- `POST /api/auth/approve/:id` — Supervisor; lets a self-registered account use its requested role.

Access tokens are HS256 JWTs (`JWT_EXPIRES_IN`, default 15m). The refresh token is only ever a cookie
(`HttpOnly`, `SameSite=Strict`, `Path=/api/auth`, `Secure` in production) and is stored as a hash.
Registration creates a pending account; until approved it can use the console but no privileged gate.
`AUTH_REQUIRED=false` (default) leaves console endpoints open so today's console, which sends no
token, keeps working; set it to `true` once Phase 9 sends `Authorization: Bearer`. Health and the auth
routes are always public.

### Supervisor-only (always gated, regardless of `AUTH_REQUIRED`)

| Endpoint | Body | Audit action |
|---|---|---|
| `POST /api/faces/:alertId/unseal` | `{ reason }` (5+ chars) | `face.unmask` |
| `PUT /api/cameras/:id/fence` | `{ polygon: [[x, y], …] }` normalised 0..1, 3–64 points | `camera.fence.edit` |
| `PUT /api/cameras/:id/hours` | `{ start: "HH:MM", end: "HH:MM" }` | `camera.hours.edit` |
| `GET /api/audit?limit=` | — | (read) returns entries plus `chain: { intact, entries, firstBadId }` |

Also audited: `alert.act`, `alert.dismiss`, `evidence.verify`, `auth.login`, `auth.logout`,
`user.approve`. The log is hash-chained (`prev_sha256`, `sha256`). With no token the row is written with a
null actor and `detail.unauthenticated: true`. 401 without a token, 403 for the wrong role.

Also additive: `GET /api/cameras/:id/settings`, `GET /api/evidence/:id`, `GET /api/evidence/root/:date`.

## Edge integration

All of it lives in `src/adapters/` and degrades to local behaviour with a short timeout.

| Feature | Edge call | Fallback |
|---|---|---|
| search | `POST {EDGE_BASE_URL}/api/search {query}` | fixture results (also when the index is empty) |
| verify | `POST {EDGE_BASE_URL}/evidence/verify` returning `{ intact, verified, total, blocks }` | local walk: block hashes, links, and clip re-hash when the file is under `EVIDENCE_STORE_PATH` |
| dismiss | `POST {EDGE_BASE_URL}/pipeline/feedback { action, alert_id, event_id, camera_id, track_id, origin }` | none needed; not awaited |

The verify and feedback routes are the contract the edge is expected to serve; `evidenceVerifier.js`
holds a `strategies` list so a `python -m evidence.verify` strategy can be added beside them.

## Data notes

`docs/ARCHITECTURE_V2.md` section 6 is implemented with these changes: `tracks.id` is a surrogate key
(`edge_track_id` holds the edge's number, which collides across cameras); `alerts.origin` is
`edge | simulate | seed`, and `seed` rows (demo history behind analytics) never appear in the queue;
`ingest_events` is the idempotency ledger; plus `refresh_tokens`, `ingest_nonces`, `camera_settings`,
`seed_state`. Analytics: `camBars` alerts per camera, `dayNight` by the operator-facing time label,
`agreement` the fused confidence of the latest 12 alerts, `traffic` tracks per minute over the last 12
minutes; each falls back to the fixture only when there is nothing to aggregate. The time label is server
local time; set `TZ` (e.g. `Asia/Kolkata`) for the post.

## Configuration

See `.env.example`; every variable is documented there. In development, unset secrets use insecure,
clearly named defaults and log a warning. `NODE_ENV=production` exits with a message listing what is
missing unless `JWT_SECRET` and `TRUEWATCH_INGEST_KEY` (32+ characters each) are set.
