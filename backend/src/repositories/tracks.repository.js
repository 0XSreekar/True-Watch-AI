const dayOf = (iso) => iso.slice(0, 10);

/** Tracks the edge reports alongside events; feeds the traffic ticker. */
export const createTracksRepository = (db) => {
  const find = db.prepare(
    'SELECT id FROM tracks WHERE camera_id = ? AND edge_track_id = ? AND day = ?',
  );
  const insert = db.prepare(
    `INSERT INTO tracks (edge_track_id, camera_id, class, dwell_s, net_px, path_px, direction, day, created_at)
     VALUES (@edge_track_id, @camera_id, @class, @dwell_s, @net_px, @path_px, @direction, @day, @created_at)`,
  );
  const refresh = db.prepare(
    'UPDATE tracks SET dwell_s = @dwell_s, net_px = @net_px, path_px = @path_px WHERE id = @id',
  );
  const since = db.prepare('SELECT COUNT(*) FROM tracks WHERE created_at >= ?').pluck();
  const between = db.prepare(
    'SELECT COUNT(*) FROM tracks WHERE created_at >= ? AND created_at < ?',
  ).pluck();

  return {
    /** Returns the surrogate id of the (camera, edge track, day) row, creating it once. */
    upsert: ({ edgeTrackId, cameraId, className, capturedAt, metrics = {}, direction = null }) => {
      const day = dayOf(capturedAt);
      const values = {
        dwell_s: Number(metrics.dwell_s) || 0,
        net_px: Number(metrics.net_px) || 0,
        path_px: Number(metrics.path_px) || 0,
      };
      const existing = find.get(cameraId, edgeTrackId, day);
      if (existing) {
        refresh.run({ id: existing.id, ...values });
        return existing.id;
      }
      return Number(
        insert.run({
          edge_track_id: edgeTrackId,
          camera_id: cameraId,
          class: className,
          direction,
          day,
          created_at: capturedAt,
          ...values,
        }).lastInsertRowid,
      );
    },
    countSince: (iso) => since.get(iso),
    /** Tracks created in each of `buckets` consecutive windows ending now, oldest first. */
    perWindow: ({ buckets = 12, windowMs = 60000, now = Date.now() } = {}) => {
      const out = [];
      for (let i = buckets - 1; i >= 0; i -= 1) {
        const end = now - i * windowMs;
        out.push(between.get(new Date(end - windowMs).toISOString(), new Date(end).toISOString()));
      }
      return out;
    },
  };
};
