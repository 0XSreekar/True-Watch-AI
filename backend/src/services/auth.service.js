import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

import { ROLES, UNITS } from '../data/mockData.js';
import { randomToken, sha256 } from '../utils/crypto.js';

const ISSUER = 'truewatch-api';
const ROLE_NAMES = ROLES.map((r) => r.k);
const REGISTER_FIELDS = ['name', 'officialId', 'unit', 'email', 'password', 'role'];
const OFFICIAL_ID = /^[A-Za-z0-9/_.-]{3,32}$/;
const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

const initialsOf = (name) =>
  name
    .replace(/[^A-Za-z\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .slice(-2)
    .map((w) => w[0].toUpperCase())
    .join('') || 'OP';

/**
 * Authentication. The request and response shapes of login/register are the frozen ones:
 *   login    -> { ok, token, expiresIn, user:{ id, name, role, unit, initials } }
 *   register -> { ok, pendingApproval, token, expiresIn, user }
 * `token` is now a real short-lived JWT; the refresh token travels only as an httpOnly cookie
 * (see auth.controller.js) and is stored server-side as a hash, so logout can revoke it.
 */
export const createAuthService = ({ repos, config, audit }) => {
  // Compared against when the user is unknown, so "no such user" costs the same as "wrong password".
  const decoy = bcrypt.hashSync('decoy-password-for-timing', config.bcryptRounds);

  const toUser = (row) => ({
    id: row.id,
    name: row.name,
    role: row.role,
    unit: row.unit,
    initials: row.initials,
  });

  const issueAccessToken = (row) =>
    jwt.sign({ role: row.role, appr: Boolean(row.approved) }, config.jwtSecret, {
      algorithm: 'HS256',
      subject: row.id,
      issuer: ISSUER,
      expiresIn: config.jwtExpiresInSeconds,
      jwtid: randomToken(9),
    });

  const issueRefreshToken = (row) => {
    const token = randomToken(32);
    const expiresAt = new Date(Date.now() + config.refreshTtlSeconds * 1000).toISOString();
    repos.refreshTokens.create(sha256(token), row.id, expiresAt);
    return { token, maxAgeMs: config.refreshTtlSeconds * 1000 };
  };

  const session = (row) => ({
    token: issueAccessToken(row),
    expiresIn: config.jwtExpiresInSeconds,
    user: toUser(row),
  });

  return {
    /** Returns { ok:false, error } or { ok:true, ...session, refresh } (refresh is stripped by the controller). */
    async login({ officialId, password } = {}) {
      if (!officialId || typeof officialId !== 'string') return { ok: false, error: 'Official ID required' };
      if (!password || typeof password !== 'string' || password.length < 4) {
        return { ok: false, error: 'Password too short' };
      }
      if (officialId.length > 64 || password.length > 128) return { ok: false, error: 'Invalid credentials' };

      const row = repos.users.findById(officialId);
      const matches = await bcrypt.compare(password, row ? row.password_hash : decoy);
      if (!row || !matches) return { ok: false, error: 'Invalid credentials' };

      audit.record(toUser(row), 'auth.login', row.id, null);
      return { ok: true, ...session(row), refresh: issueRefreshToken(row) };
    },

    async register(payload = {}) {
      const missing = REGISTER_FIELDS.filter((k) => !payload[k]);
      if (missing.length) return { ok: false, error: `Missing: ${missing.join(', ')}` };
      if (REGISTER_FIELDS.some((k) => typeof payload[k] !== 'string')) {
        return { ok: false, error: 'Invalid field types' };
      }
      const { name, officialId, unit, email, password, role } = payload;
      if (name.trim().length < 3 || name.length > 100) return { ok: false, error: 'Invalid name' };
      if (!OFFICIAL_ID.test(officialId)) return { ok: false, error: 'Invalid official ID' };
      if (!UNITS.includes(unit)) return { ok: false, error: 'Unknown unit' };
      if (!EMAIL.test(email) || email.length > 254) return { ok: false, error: 'Invalid email' };
      if (password.length < 8 || password.length > 128) return { ok: false, error: 'Password must be 8 to 128 characters' };
      if (!ROLE_NAMES.includes(role)) return { ok: false, error: 'Unknown role' };
      if (repos.users.findById(officialId)) return { ok: false, error: 'Official ID already registered' };

      const row = repos.users.create({
        id: officialId,
        name: name.trim(),
        unit,
        email,
        role,
        initials: initialsOf(name),
        password_hash: await bcrypt.hash(password, config.bcryptRounds),
        approved: 0,
      });
      return { ok: true, pendingApproval: true, ...session(row), refresh: issueRefreshToken(row) };
    },

    /**
     * Rotates a refresh token. A token that was already used is treated as stolen: every session
     * of that user is revoked. Returns null for any invalid token.
     */
    async refresh(token) {
      if (!token || typeof token !== 'string') return null;
      const hash = sha256(token);
      const stored = repos.refreshTokens.find(hash);
      if (!stored) return null;
      if (stored.revoked_at) {
        repos.refreshTokens.revokeAllForUser(stored.user_id);
        return null;
      }
      if (Date.parse(stored.expires_at) <= Date.now()) return null;
      if (!repos.refreshTokens.revoke(hash)) return null;

      const row = repos.users.findById(stored.user_id);
      if (!row) return null;
      return { ok: true, ...session(row), refresh: issueRefreshToken(row) };
    },

    logout(token, actor) {
      if (token && typeof token === 'string') {
        const hash = sha256(token);
        const stored = repos.refreshTokens.find(hash);
        if (stored && repos.refreshTokens.revoke(hash)) {
          audit.record(actor ?? { id: stored.user_id }, 'auth.logout', stored.user_id, null);
        }
      }
      return { ok: true };
    },

    /** Verifies a bearer token and resolves the live user row. Null when invalid or the user is gone. */
    authenticate(token) {
      try {
        const claims = jwt.verify(token, config.jwtSecret, { algorithms: ['HS256'], issuer: ISSUER });
        const row = repos.users.findById(claims.sub);
        if (!row) return null;
        return { ...toUser(row), approved: Boolean(row.approved) };
      } catch {
        return null;
      }
    },

    /** Supervisor action: lets a self-registered account use the role it asked for. */
    approve(userId, actor) {
      const row = repos.users.findById(userId);
      if (!row) return null;
      repos.users.approve(userId);
      audit.record(actor, 'user.approve', userId, { role: row.role });
      return { id: userId, approved: true, role: row.role };
    },

    listUnits: () => UNITS,
    listRoles: () => ROLES,
  };
};

export default createAuthService;
