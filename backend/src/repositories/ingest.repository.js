/** Replay-protection nonces for POST /api/ingest/*. Stored for INGEST_NONCE_TTL_SECONDS. */
export const createNonceRepository = (db) => {
  const insert = db.prepare('INSERT OR IGNORE INTO ingest_nonces (nonce, seen_at) VALUES (?, ?)');
  const prune = db.prepare('DELETE FROM ingest_nonces WHERE seen_at < ?');

  return {
    /** True when the nonce was new; false when it has been seen before (a replay). */
    remember: (nonce, nowMs = Date.now()) => insert.run(nonce, nowMs).changes === 1,
    prune: (olderThanMs) => prune.run(olderThanMs).changes,
  };
};

/** Every event_id ever accepted, mapped to the alert it created or enriched. */
export const createEventLedgerRepository = (db) => {
  const get = db.prepare('SELECT alert_id FROM ingest_events WHERE event_id = ?').pluck();
  const insert = db.prepare(
    'INSERT OR IGNORE INTO ingest_events (event_id, alert_id, received_at) VALUES (?, ?, ?)',
  );

  return {
    alertIdFor: (eventId) => get.get(eventId) || null,
    record: (eventId, alertId) => insert.run(eventId, alertId, new Date().toISOString()),
  };
};
