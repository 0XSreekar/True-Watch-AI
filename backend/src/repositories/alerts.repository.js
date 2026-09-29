const SELECT = `
  SELECT a.*, c.name AS cam_name, c.sector AS cam_sector, c.ir AS cam_ir, c.road AS cam_road,
         t.edge_track_id AS edge_track_id
    FROM alerts a
    JOIN cameras c ON c.id = a.camera_id
    LEFT JOIN tracks t ON t.id = a.track_id`;

/** Columns a caller may change after insert. Everything else is immutable evidence of the event. */
const PATCHABLE = new Set([
  'stage',
  'reason',
  'reason_source',
  'plate',
  'plate_script',
  'hash',
  'evidence_id',
  'clip_url',
  'thumb_url',
  'dori_band',
  'rule_fired',
]);

/**
 * Turns a joined row into the FROZEN alert shape. Every frozen key is always present; each
 * additive key appears only when the database actually knows it (docs/ARCHITECTURE_V2.md 3.3).
 * `reason_source` and `source` are additive honesty flags and are only emitted for edge alerts,
 * so a simulated alert is byte-for-byte the frozen shape.
 */
export const toAlert = (row) => {
  const alert = {
    id: row.id,
    camera: {
      id: row.camera_id,
      name: row.cam_name,
      sector: row.cam_sector,
      ir: Boolean(row.cam_ir),
      road: Boolean(row.cam_road),
    },
    stage: row.stage,
    time: row.time_label,
    reason: row.reason,
    confidence: row.confidence,
    plate: row.plate ?? null,
    hash: row.hash,
    seed: row.seed,
    channels: { appearance: Boolean(row.ch_appearance), motion: Boolean(row.ch_motion) },
  };

  if (row.edge_track_id != null) alert.track_id = row.edge_track_id;
  if (row.bbox != null) alert.bbox = JSON.parse(row.bbox);
  if (row.score_appearance != null && row.score_motion != null) {
    alert.channel_scores = { appearance: row.score_appearance, motion: row.score_motion };
  }
  if (row.rule_fired != null) alert.rule_fired = row.rule_fired;
  if (row.dori_band != null) alert.dori_band = row.dori_band;
  if (row.evidence_id != null) alert.evidence_id = row.evidence_id;
  if (row.clip_url != null) alert.clip_url = row.clip_url;
  if (row.thumb_url != null) alert.thumb_url = row.thumb_url;
  if (row.origin === 'edge') {
    alert.reason_source = row.reason_source;
    alert.source = row.ingest_source;
  }
  return alert;
};

export const createAlertsRepository = (db) => {
  const byId = db.prepare(`${SELECT} WHERE a.id = ?`);
  const byEventId = db.prepare(`${SELECT} WHERE a.event_id = ?`);
  const liveByTrack = db.prepare(
    `${SELECT} WHERE a.camera_id = ? AND t.edge_track_id = ? AND t.day = ?
        AND a.stage != 'gone' AND a.origin = 'edge'
      ORDER BY a.rowid ASC LIMIT 1`,
  );
  const live = db.prepare(
    `${SELECT} WHERE a.stage != 'gone' AND a.origin != 'seed'
      ORDER BY a.captured_at DESC, a.rowid DESC LIMIT ?`,
  );
  const insert = db.prepare(
    `INSERT INTO alerts (
       id, event_id, camera_id, stage, captured_at, time_label, reason, reason_source, confidence,
       plate, plate_script, hash, seed, ch_appearance, ch_motion, track_id, bbox,
       score_appearance, score_motion, rule_fired, dori_band, evidence_id, clip_url, thumb_url,
       ingest_source, origin, created_at)
     VALUES (
       @id, @event_id, @camera_id, @stage, @captured_at, @time_label, @reason, @reason_source, @confidence,
       @plate, @plate_script, @hash, @seed, @ch_appearance, @ch_motion, @track_id, @bbox,
       @score_appearance, @score_motion, @rule_fired, @dori_band, @evidence_id, @clip_url, @thumb_url,
       @ingest_source, @origin, @created_at)`,
  );
  const usedSince = db.prepare(
    "SELECT COUNT(*) FROM alerts WHERE origin != 'seed' AND created_at >= ?",
  ).pluck();
  const perCamera = db.prepare(
    `SELECT c.id AS id, COUNT(a.id) AS v
       FROM cameras c LEFT JOIN alerts a ON a.camera_id = c.id
      GROUP BY c.id ORDER BY c.rowid`,
  );
  const dayNight = db.prepare(
    `SELECT SUM(CASE WHEN CAST(substr(time_label, 1, 2) AS INTEGER) BETWEEN 6 AND 17 THEN 1 ELSE 0 END) AS day,
            SUM(CASE WHEN CAST(substr(time_label, 1, 2) AS INTEGER) BETWEEN 6 AND 17 THEN 0 ELSE 1 END) AS night
       FROM alerts`,
  );
  const latestConfidence = db.prepare(
    'SELECT confidence FROM alerts ORDER BY captured_at DESC, rowid DESC LIMIT ?',
  ).pluck();
  const total = db.prepare('SELECT COUNT(*) FROM alerts').pluck();

  return {
    findById: (id) => {
      const row = byId.get(id);
      return row ? toAlert(row) : null;
    },
    findByEventId: (eventId) => {
      const row = byEventId.get(eventId);
      return row ? toAlert(row) : null;
    },
    /** Internal row (with event_id, origin, captured_at) for services that need more than the API shape. */
    findRowById: (id) => byId.get(id) || null,
    findRowByEventId: (eventId) => byEventId.get(eventId) || null,
    /** The not-dismissed edge alert already raised for this camera track today, if any. */
    findRowLiveByTrack: (cameraId, edgeTrackId, day) => liveByTrack.get(cameraId, edgeTrackId, day) || null,
    listLive: (limit = 50) => live.all(limit).map(toAlert),

    insert: (row) => {
      insert.run({
        reason_source: 'template',
        plate: null,
        plate_script: null,
        track_id: null,
        bbox: null,
        score_appearance: null,
        score_motion: null,
        rule_fired: null,
        dori_band: null,
        evidence_id: null,
        clip_url: null,
        thumb_url: null,
        ingest_source: 'file',
        origin: 'edge',
        created_at: new Date().toISOString(),
        ...row,
      });
      return toAlert(byId.get(row.id));
    },

    /** Applies a whitelisted column patch and returns the refreshed alert. */
    patch: (id, patch) => {
      const columns = Object.keys(patch).filter((k) => PATCHABLE.has(k));
      if (columns.length) {
        const assignments = columns.map((c) => `${c} = @${c}`).join(', ');
        db.prepare(`UPDATE alerts SET ${assignments} WHERE id = @id`).run({
          id,
          ...Object.fromEntries(columns.map((c) => [c, patch[c]])),
        });
      }
      const row = byId.get(id);
      return row ? toAlert(row) : null;
    },

    countSince: (iso) => usedSince.get(iso),
    total: () => total.get(),
    countsByCamera: () => perCamera.all(),
    dayNightCounts: () => {
      const r = dayNight.get();
      return { day: r.day || 0, night: r.night || 0 };
    },
    /** Confidence of the latest `n` alerts, oldest first. */
    latestConfidences: (n) => latestConfidence.all(n).reverse(),
  };
};
