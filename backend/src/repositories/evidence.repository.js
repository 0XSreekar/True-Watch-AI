import { sha256 } from '../utils/crypto.js';

/** The hash a block commits to: SHA-256(prev || clip || camera || sealed_at). */
export const blockDigest = ({ prev_sha256: prev, clip_sha256: clip, camera_id: camera, sealed_at: sealedAt }) =>
  sha256(prev, clip, camera, sealedAt);

export const createEvidenceRepository = (db) => {
  const ascending = db.prepare('SELECT * FROM evidence_blocks ORDER BY n ASC');
  const tail = db.prepare('SELECT * FROM evidence_blocks ORDER BY n DESC LIMIT ?');
  const byId = db.prepare('SELECT * FROM evidence_blocks WHERE evidence_id = ?');
  const byN = db.prepare('SELECT * FROM evidence_blocks WHERE n = ?');
  const tip = db.prepare('SELECT * FROM evidence_blocks ORDER BY n DESC LIMIT 1');
  const count = db.prepare('SELECT COUNT(*) FROM evidence_blocks').pluck();
  const forDay = db.prepare(
    'SELECT * FROM evidence_blocks WHERE substr(sealed_at, 1, 10) = ? ORDER BY n ASC',
  );
  const insert = db.prepare(
    `INSERT INTO evidence_blocks
       (n, evidence_id, camera_id, time_label, display_hash, clip_sha256, prev_sha256, block_sha256,
        clip_path, sealed_at, alert_id)
     VALUES (@n, @evidence_id, @camera_id, @time_label, @display_hash, @clip_sha256, @prev_sha256,
             @block_sha256, @clip_path, @sealed_at, @alert_id)`,
  );

  /** Frozen block shape: { n, hash, camera, time }. */
  const toBlock = (row) => ({
    n: row.n,
    hash: row.display_hash,
    camera: row.camera_id,
    time: row.time_label,
  });

  return {
    /** The latest `limit` blocks in ascending order, in the frozen shape. */
    chain: (limit = 20) => tail.all(limit).reverse().map(toBlock),
    allRows: () => ascending.all(),
    count: () => count.get(),
    tip: () => tip.get() || null,
    findByEvidenceId: (id) => byId.get(id) || null,
    findByN: (n) => byN.get(n) || null,
    forDay: (day) => forDay.all(day),
    insert: (row) => {
      insert.run({ alert_id: null, ...row });
      return byN.get(row.n);
    },
    toBlock,
  };
};

export const createDailyRootsRepository = (db) => {
  const get = db.prepare('SELECT * FROM daily_roots WHERE day = ?');
  const list = db.prepare('SELECT * FROM daily_roots ORDER BY day DESC LIMIT ?');
  const upsert = db.prepare(
    `INSERT INTO daily_roots (day, merkle_root, leaf_count, first_n, last_n, computed_at)
     VALUES (@day, @merkle_root, @leaf_count, @first_n, @last_n, @computed_at)
     ON CONFLICT(day) DO UPDATE SET
       merkle_root = excluded.merkle_root, leaf_count = excluded.leaf_count,
       first_n = excluded.first_n, last_n = excluded.last_n, computed_at = excluded.computed_at`,
  );

  return {
    get: (day) => get.get(day) || null,
    list: (limit = 30) => list.all(limit),
    upsert: (row) => {
      upsert.run(row);
      return get.get(row.day);
    },
  };
};
