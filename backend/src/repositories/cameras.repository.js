/**
 * Cameras and posts. Rows store booleans as 0/1; they leave here as the frozen API shapes.
 * rtsp_url and dori_band are internal columns and are never part of a public camera.
 */
const toCamera = (row) => ({
  id: row.id,
  name: row.name,
  sector: row.sector,
  ir: Boolean(row.ir),
  road: Boolean(row.road),
});

export const createCamerasRepository = (db) => {
  const all = db.prepare('SELECT * FROM cameras ORDER BY rowid');
  const bySector = db.prepare('SELECT * FROM cameras WHERE sector = ? ORDER BY rowid');
  const byId = db.prepare('SELECT * FROM cameras WHERE id = ?');
  const sectors = db.prepare('SELECT sector FROM cameras GROUP BY sector ORDER BY MIN(rowid)').pluck();
  const count = db.prepare('SELECT COUNT(*) FROM cameras').pluck();
  const insert = db.prepare(
    `INSERT INTO cameras (id, name, sector, ir, road, post_id, rtsp_url, dori_band)
     VALUES (@id, @name, @sector, @ir, @road, @post_id, @rtsp_url, @dori_band)`,
  );

  return {
    list: ({ sector } = {}) => (sector ? bySector.all(sector) : all.all()).map(toCamera),
    get: (id) => {
      const row = byId.get(id);
      return row ? toCamera(row) : null;
    },
    /** Internal view: includes the owning post, used to validate ingest. */
    getWithPost: (id) => byId.get(id) || null,
    sectors: () => sectors.all(),
    count: () => count.get(),
    create: (camera) =>
      insert.run({
        rtsp_url: null,
        dori_band: null,
        post_id: null,
        ...camera,
        ir: camera.ir ? 1 : 0,
        road: camera.road ? 1 : 0,
      }),
  };
};

export const createPostsRepository = (db) => {
  // A post lights up when the seeded flag is set or any of its cameras has a live alert.
  const list = db.prepare(
    `SELECT p.id, p.name, p.x, p.y, p.road,
            MAX(p.alert,
                EXISTS (SELECT 1 FROM alerts a JOIN cameras c ON c.id = a.camera_id
                        WHERE c.post_id = p.id AND a.stage IN ('prov','conf') AND a.origin != 'seed')) AS alert
       FROM posts p ORDER BY p.rowid`,
  );
  const insert = db.prepare(
    'INSERT INTO posts (id, name, x, y, road, alert) VALUES (@id, @name, @x, @y, @road, @alert)',
  );
  const count = db.prepare('SELECT COUNT(*) FROM posts').pluck();

  return {
    list: () =>
      list.all().map((r) => ({
        id: r.id,
        name: r.name,
        x: r.x,
        y: r.y,
        road: Boolean(r.road),
        alert: Boolean(r.alert),
      })),
    count: () => count.get(),
    create: (post) => insert.run({ ...post, road: post.road ? 1 : 0, alert: post.alert ? 1 : 0 }),
  };
};

/** Supervisor-tunable per-camera fence polygon and sanctioned hours. */
export const createCameraSettingsRepository = (db) => {
  const get = db.prepare('SELECT * FROM camera_settings WHERE camera_id = ?');
  const upsert = db.prepare(
    `INSERT INTO camera_settings (camera_id, fence, hours, updated_by, updated_at)
     VALUES (@camera_id, @fence, @hours, @updated_by, @updated_at)
     ON CONFLICT(camera_id) DO UPDATE SET
       fence = COALESCE(excluded.fence, camera_settings.fence),
       hours = COALESCE(excluded.hours, camera_settings.hours),
       updated_by = excluded.updated_by,
       updated_at = excluded.updated_at`,
  );

  const toSettings = (cameraId, row) => ({
    camera_id: cameraId,
    fence: row?.fence ? JSON.parse(row.fence) : null,
    hours: row?.hours ? JSON.parse(row.hours) : null,
    updated_by: row?.updated_by ?? null,
    updated_at: row?.updated_at ?? null,
  });

  return {
    get: (cameraId) => toSettings(cameraId, get.get(cameraId)),
    update: (cameraId, { fence, hours }, actorId) => {
      upsert.run({
        camera_id: cameraId,
        fence: fence ? JSON.stringify(fence) : null,
        hours: hours ? JSON.stringify(hours) : null,
        updated_by: actorId ?? null,
        updated_at: new Date().toISOString(),
      });
      return toSettings(cameraId, get.get(cameraId));
    },
  };
};
