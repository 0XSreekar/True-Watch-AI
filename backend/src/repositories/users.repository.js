/** Public view of a user row: the frozen `user` object of the session response. */
export const toSessionUser = (row) => ({
  id: row.id,
  name: row.name,
  role: row.role,
  unit: row.unit,
  initials: row.initials,
});

export const createUsersRepository = (db) => {
  const byId = db.prepare('SELECT * FROM users WHERE id = ?');
  const insert = db.prepare(
    `INSERT INTO users (id, name, unit, email, role, initials, password_hash, approved, created_at)
     VALUES (@id, @name, @unit, @email, @role, @initials, @password_hash, @approved, @created_at)`,
  );
  const count = db.prepare('SELECT COUNT(*) FROM users').pluck();

  return {
    findById: (id) => byId.get(id) || null,
    count: () => count.get(),
    create: (user) => {
      insert.run({
        email: null,
        approved: 0,
        created_at: new Date().toISOString(),
        ...user,
        approved: user.approved ? 1 : 0,
      });
      return byId.get(user.id);
    },
  };
};

/** Server-side refresh tokens. Only a SHA-256 of the opaque token is stored. */
export const createRefreshTokensRepository = (db) => {
  const insert = db.prepare(
    'INSERT INTO refresh_tokens (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)',
  );
  const get = db.prepare('SELECT * FROM refresh_tokens WHERE token_hash = ?');
  const revoke = db.prepare(
    'UPDATE refresh_tokens SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL',
  );
  const revokeAllForUser = db.prepare(
    'UPDATE refresh_tokens SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL',
  );
  const prune = db.prepare('DELETE FROM refresh_tokens WHERE expires_at < ?');

  return {
    create: (tokenHash, userId, expiresAt) =>
      insert.run(tokenHash, userId, expiresAt, new Date().toISOString()),
    find: (tokenHash) => get.get(tokenHash) || null,
    /** Returns true only for the call that actually revoked it, so rotation is single-use. */
    revoke: (tokenHash) => revoke.run(new Date().toISOString(), tokenHash).changes === 1,
    revokeAllForUser: (userId) => revokeAllForUser.run(new Date().toISOString(), userId).changes,
    pruneExpired: () => prune.run(new Date().toISOString()).changes,
  };
};
