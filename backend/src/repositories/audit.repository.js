import { ZERO_HASH, sha256 } from '../utils/crypto.js';

const entryDigest = ({ prev, at, actorId, action, subject, detail }) =>
  sha256(prev, '|', at, '|', actorId ?? '', '|', action, '|', subject, '|', detail ?? '');

/**
 * Append-only, hash-chained audit trail (threat 4 in docs/ARCHITECTURE_V2.md): each row commits
 * to the previous row, so removing or editing an entry breaks every hash after it.
 */
export const createAuditRepository = (db) => {
  const last = db.prepare('SELECT sha256 FROM audit_log ORDER BY id DESC LIMIT 1').pluck();
  const insert = db.prepare(
    `INSERT INTO audit_log (at, actor_id, action, subject, detail, prev_sha256, sha256)
     VALUES (@at, @actor_id, @action, @subject, @detail, @prev_sha256, @sha256)`,
  );
  const recent = db.prepare('SELECT * FROM audit_log ORDER BY id DESC LIMIT ?');
  const ascending = db.prepare('SELECT * FROM audit_log ORDER BY id ASC');

  const append = db.transaction(({ actorId = null, action, subject, detail = null }) => {
    const at = new Date().toISOString();
    const detailJson = detail == null ? null : JSON.stringify(detail);
    const prev = last.get() || ZERO_HASH;
    const digest = entryDigest({ prev, at, actorId, action, subject: String(subject), detail: detailJson });
    const info = insert.run({
      at,
      actor_id: actorId,
      action,
      subject: String(subject),
      detail: detailJson,
      prev_sha256: prev,
      sha256: digest,
    });
    return { id: Number(info.lastInsertRowid), at, action, subject: String(subject), sha256: digest };
  });

  const present = (row) => ({
    id: row.id,
    at: row.at,
    actor_id: row.actor_id,
    action: row.action,
    subject: row.subject,
    detail: row.detail ? JSON.parse(row.detail) : null,
    sha256: row.sha256,
  });

  return {
    append,
    recent: (limit = 100) => recent.all(limit).map(present),
    /** Recomputes every link. Returns { intact, entries, firstBadId }. */
    verify: () => {
      let prev = ZERO_HASH;
      let entries = 0;
      for (const row of ascending.iterate()) {
        const expected = entryDigest({
          prev,
          at: row.at,
          actorId: row.actor_id,
          action: row.action,
          subject: row.subject,
          detail: row.detail,
        });
        if (row.prev_sha256 !== prev || row.sha256 !== expected) {
          return { intact: false, entries, firstBadId: row.id };
        }
        prev = row.sha256;
        entries += 1;
      }
      return { intact: true, entries, firstBadId: null };
    },
  };
};
