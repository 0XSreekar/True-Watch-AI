/**
 * Ordered, append-only migrations. Each runs once, inside a transaction, and is recorded in
 * schema_migrations, so booting against an existing file is always safe.
 *
 * The first migration is docs/ARCHITECTURE_V2.md section 6 with two deliberate changes:
 *  - tracks.id is a surrogate key. The edge numbers tracks per camera per run, so its ids collide
 *    across cameras; the edge id lives in tracks.edge_track_id and is what the API reports.
 *  - alerts.origin records where a row came from ('edge' | 'simulate' | 'seed'), so the demo
 *    history that feeds analytics never shows up in the live queue.
 */
export const MIGRATIONS = [
  {
    id: 1,
    name: 'core_schema',
    sql: `
      CREATE TABLE posts (
        id     TEXT PRIMARY KEY,
        name   TEXT NOT NULL,
        x      REAL NOT NULL,
        y      REAL NOT NULL,
        road   INTEGER NOT NULL DEFAULT 1,
        alert  INTEGER NOT NULL DEFAULT 0
      );

      CREATE TABLE cameras (
        id        TEXT PRIMARY KEY,
        name      TEXT NOT NULL,
        sector    TEXT NOT NULL,
        ir        INTEGER NOT NULL DEFAULT 0,
        road      INTEGER NOT NULL DEFAULT 1,
        post_id   TEXT REFERENCES posts(id),
        rtsp_url  TEXT,
        dori_band TEXT
      );

      CREATE TABLE tracks (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        edge_track_id  INTEGER NOT NULL,
        camera_id      TEXT NOT NULL REFERENCES cameras(id),
        class          TEXT NOT NULL,
        first_frame    INTEGER NOT NULL DEFAULT 0,
        last_frame     INTEGER NOT NULL DEFAULT 0,
        dwell_s        REAL NOT NULL DEFAULT 0,
        net_px         REAL NOT NULL DEFAULT 0,
        path_px        REAL NOT NULL DEFAULT 0,
        direction      TEXT,
        day            TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        UNIQUE (camera_id, edge_track_id, day)
      );
      CREATE INDEX idx_tracks_camera ON tracks(camera_id, created_at DESC);
      CREATE INDEX idx_tracks_created ON tracks(created_at DESC);

      CREATE TABLE evidence_blocks (
        n            INTEGER PRIMARY KEY,
        evidence_id  TEXT NOT NULL UNIQUE,
        camera_id    TEXT NOT NULL REFERENCES cameras(id),
        time_label   TEXT NOT NULL,
        display_hash TEXT NOT NULL,
        clip_sha256  TEXT NOT NULL,
        prev_sha256  TEXT NOT NULL,
        block_sha256 TEXT NOT NULL UNIQUE,
        clip_path    TEXT NOT NULL,
        sealed_at    TEXT NOT NULL,
        alert_id     TEXT
      );
      CREATE INDEX idx_evidence_sealed ON evidence_blocks(sealed_at DESC);
      CREATE INDEX idx_evidence_camera ON evidence_blocks(camera_id, n);

      CREATE TABLE alerts (
        id             TEXT PRIMARY KEY,
        event_id       TEXT NOT NULL UNIQUE,
        camera_id      TEXT NOT NULL REFERENCES cameras(id),
        stage          TEXT NOT NULL CHECK (stage IN ('prov','conf','sealed','gone')),
        captured_at    TEXT NOT NULL,
        time_label     TEXT NOT NULL,
        reason         TEXT NOT NULL,
        reason_source  TEXT NOT NULL DEFAULT 'template' CHECK (reason_source IN ('vlm','template')),
        confidence     INTEGER NOT NULL CHECK (confidence BETWEEN 0 AND 100),
        plate          TEXT,
        plate_script   TEXT,
        hash           TEXT NOT NULL,
        seed           REAL NOT NULL,
        ch_appearance  INTEGER NOT NULL,
        ch_motion      INTEGER NOT NULL,
        track_id       INTEGER REFERENCES tracks(id),
        bbox           TEXT,
        score_appearance REAL,
        score_motion     REAL,
        rule_fired     TEXT,
        dori_band      TEXT,
        evidence_id    TEXT REFERENCES evidence_blocks(evidence_id),
        clip_url       TEXT,
        thumb_url      TEXT,
        ingest_source  TEXT NOT NULL DEFAULT 'file' CHECK (ingest_source IN ('rtsp','file')),
        origin         TEXT NOT NULL DEFAULT 'edge' CHECK (origin IN ('edge','simulate','seed')),
        created_at     TEXT NOT NULL
      );
      CREATE INDEX idx_alerts_captured ON alerts(captured_at DESC);
      CREATE INDEX idx_alerts_camera   ON alerts(camera_id, captured_at DESC);
      CREATE INDEX idx_alerts_stage    ON alerts(stage, captured_at DESC);
      CREATE INDEX idx_alerts_track    ON alerts(track_id);

      CREATE TABLE daily_roots (
        day         TEXT PRIMARY KEY,
        merkle_root TEXT NOT NULL,
        leaf_count  INTEGER NOT NULL,
        first_n     INTEGER NOT NULL,
        last_n      INTEGER NOT NULL,
        computed_at TEXT NOT NULL,
        signature   TEXT,
        signed_at   TEXT
      );

      CREATE TABLE users (
        id            TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        unit          TEXT NOT NULL,
        email         TEXT,
        role          TEXT NOT NULL CHECK (role IN ('Operator','Supervisor','HQ Analyst')),
        initials      TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        approved      INTEGER NOT NULL DEFAULT 0,
        created_at    TEXT NOT NULL
      );

      CREATE TABLE audit_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        at          TEXT NOT NULL,
        actor_id    TEXT REFERENCES users(id),
        action      TEXT NOT NULL,
        subject     TEXT NOT NULL,
        detail      TEXT,
        prev_sha256 TEXT NOT NULL,
        sha256      TEXT NOT NULL UNIQUE
      );
      CREATE INDEX idx_audit_at    ON audit_log(at DESC);
      CREATE INDEX idx_audit_actor ON audit_log(actor_id, at DESC);
    `,
  },
  {
    id: 2,
    name: 'auth_ingest_and_settings',
    sql: `
      CREATE TABLE refresh_tokens (
        token_hash TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL REFERENCES users(id),
        expires_at TEXT NOT NULL,
        revoked_at TEXT,
        created_at TEXT NOT NULL
      );
      CREATE INDEX idx_refresh_user ON refresh_tokens(user_id);

      CREATE TABLE ingest_nonces (
        nonce   TEXT PRIMARY KEY,
        seen_at INTEGER NOT NULL
      );
      CREATE INDEX idx_nonces_seen ON ingest_nonces(seen_at);

      CREATE TABLE camera_settings (
        camera_id  TEXT PRIMARY KEY REFERENCES cameras(id),
        fence      TEXT,
        hours      TEXT,
        updated_by TEXT,
        updated_at TEXT NOT NULL
      );

      CREATE TABLE seed_state (
        name      TEXT PRIMARY KEY,
        seeded_at TEXT NOT NULL
      );
    `,
  },
  {
    id: 3,
    name: 'ingest_event_ledger',
    // Every event_id the edge has ever sent, including ones merged into an existing alert, so a
    // replay is a no-op whether the event created an alert or only enriched one.
    sql: `
      CREATE TABLE ingest_events (
        event_id    TEXT PRIMARY KEY,
        alert_id    TEXT NOT NULL REFERENCES alerts(id),
        received_at TEXT NOT NULL
      );
      CREATE INDEX idx_ingest_events_alert ON ingest_events(alert_id);
    `,
  },
];
