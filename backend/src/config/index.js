import path from 'node:path';
import { fileURLToPath } from 'node:url';

import dotenv from 'dotenv';

dotenv.config();

const BACKEND_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

const int = (value, fallback) => {
  const n = Number.parseInt(value ?? '', 10);
  return Number.isFinite(n) ? n : fallback;
};

const bool = (value, fallback) => {
  if (value === undefined || value === null || value === '') return fallback;
  return ['1', 'true', 'yes', 'on'].includes(String(value).trim().toLowerCase());
};

const list = (value, fallback) =>
  (value || fallback)
    .split(',')
    .map((o) => o.trim())
    .filter(Boolean);

/** Turns '15m' / '12h' / '900' into seconds. Falls back when the value is unparseable. */
export const durationSeconds = (value, fallback) => {
  const match = /^(\d+)\s*([smhd]?)$/i.exec(String(value ?? '').trim());
  if (!match) return fallback;
  const unit = { '': 1, s: 1, m: 60, h: 3600, d: 86400 }[match[2].toLowerCase()];
  return Number(match[1]) * unit;
};

const resolvePath = (value) => {
  if (value === ':memory:') return value;
  return path.isAbsolute(value) ? value : path.resolve(BACKEND_ROOT, value);
};

/**
 * Development-only defaults. They are intentionally weak and are refused outright when
 * NODE_ENV=production (see validateConfig). Nothing here is ever generated silently.
 */
export const DEV_DEFAULTS = Object.freeze({
  jwtSecret: 'dev-only-insecure-jwt-secret-do-not-deploy',
  ingestKey: 'dev-only-insecure-ingest-key-do-not-deploy',
  seedPassword: 'dev-only-demo-password',
});

/**
 * Builds the backend configuration from an environment map. Pure: tests pass their own map.
 * This is the only module that reads process.env.
 */
export const loadConfig = (env = process.env) => {
  const nodeEnv = env.NODE_ENV || 'development';
  const production = nodeEnv === 'production';
  const devDefaults = !production;

  return {
    env: nodeEnv,
    production,
    port: int(env.PORT, 4000),
    corsOrigins: list(env.CORS_ORIGIN, 'http://localhost:5173'),
    trustProxy: bool(env.TRUST_PROXY, false),
    bodyLimit: env.BODY_LIMIT || '100kb',

    databasePath: resolvePath(env.DATABASE_PATH || './var/truewatch.db'),
    evidenceStorePath: resolvePath(env.EVIDENCE_STORE_PATH || './var/evidence'),

    ingestKey: env.TRUEWATCH_INGEST_KEY || (devDefaults ? DEV_DEFAULTS.ingestKey : ''),
    ingestMaxSkewSeconds: int(env.INGEST_MAX_SKEW_SECONDS, 120),
    ingestNonceTtlSeconds: int(env.INGEST_NONCE_TTL_SECONDS, 240),
    ingestRateLimitPerMinute: int(env.INGEST_RATE_LIMIT_PER_MINUTE, 300),

    streamHeartbeatSeconds: int(env.STREAM_HEARTBEAT_SECONDS, 15),
    streamMaxClients: int(env.STREAM_MAX_CLIENTS, 100),

    jwtSecret: env.JWT_SECRET || (devDefaults ? DEV_DEFAULTS.jwtSecret : ''),
    jwtExpiresIn: env.JWT_EXPIRES_IN || '15m',
    jwtExpiresInSeconds: durationSeconds(env.JWT_EXPIRES_IN || '15m', 900),
    refreshTtlSeconds: durationSeconds(env.REFRESH_TTL || '7d', 604800),
    bcryptRounds: int(env.BCRYPT_ROUNDS, 10),
    authRequired: bool(env.AUTH_REQUIRED, false),
    seedDemoUsers: bool(env.SEED_DEMO_USERS, devDefaults),
    seedUserPassword: env.SEED_USER_PASSWORD || (devDefaults ? DEV_DEFAULTS.seedPassword : ''),

    apiRateLimitPerMinute: int(env.API_RATE_LIMIT_PER_MINUTE, 600),
    authRateLimitPer15Min: int(env.AUTH_RATE_LIMIT_PER_15_MIN, 30),

    edgeBaseUrl: (env.EDGE_BASE_URL || '').replace(/\/+$/, '') || null,
    edgeTimeoutMs: int(env.EDGE_TIMEOUT_MS, 2500),
    edgeSearchTimeoutMs: int(env.EDGE_SEARCH_TIMEOUT_MS, 8000),

    shiftHours: int(env.SHIFT_HOURS, 8),
    logRequests: nodeEnv !== 'test',
  };
};

export class ConfigError extends Error {
  constructor(problems) {
    super(`Invalid configuration:\n  - ${problems.join('\n  - ')}`);
    this.name = 'ConfigError';
    this.problems = problems;
  }
}

/**
 * Startup validation. Production refuses to boot without real secrets; development accepts the
 * documented defaults above so a fresh clone runs, and says so in the log.
 * Returns the list of development warnings.
 */
export const validateConfig = (config) => {
  const problems = [];
  if (config.production) {
    if (!config.jwtSecret) problems.push('JWT_SECRET is required in production (openssl rand -hex 32)');
    else if (config.jwtSecret.length < 32) problems.push('JWT_SECRET must be at least 32 characters');
    if (!config.ingestKey) {
      problems.push('TRUEWATCH_INGEST_KEY is required in production (openssl rand -hex 32)');
    } else if (config.ingestKey.length < 32) {
      problems.push('TRUEWATCH_INGEST_KEY must be at least 32 characters');
    }
    if (config.seedDemoUsers && !config.seedUserPassword) {
      problems.push('SEED_USER_PASSWORD is required when SEED_DEMO_USERS is enabled in production');
    }
    if (config.corsOrigins.includes('*')) problems.push('CORS_ORIGIN must be an explicit allow-list, not "*"');
  }
  if (!config.databasePath) problems.push('DATABASE_PATH must not be empty');
  if (problems.length) throw new ConfigError(problems);

  const warnings = [];
  if (!config.production) {
    if (config.jwtSecret === DEV_DEFAULTS.jwtSecret) warnings.push('JWT_SECRET unset: using the insecure development default');
    if (config.ingestKey === DEV_DEFAULTS.ingestKey) warnings.push('TRUEWATCH_INGEST_KEY unset: using the insecure development default');
  }
  return warnings;
};

export const config = loadConfig();

export default config;
