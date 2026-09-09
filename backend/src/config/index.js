import dotenv from 'dotenv';

dotenv.config();

const int = (value, fallback) => {
  const n = Number.parseInt(value ?? '', 10);
  return Number.isFinite(n) ? n : fallback;
};

/**
 * Single source of truth for backend configuration.
 * Add new sections here (db, auth, vlm…) rather than reading process.env elsewhere.
 */
export const config = {
  env: process.env.NODE_ENV || 'development',
  port: int(process.env.PORT, 4000),
  corsOrigins: (process.env.CORS_ORIGIN || 'http://localhost:5173')
    .split(',')
    .map((o) => o.trim())
    .filter(Boolean),

  // Reserved for the real implementation — read but unused by the demo.
  databaseUrl: process.env.DATABASE_URL || null,
  jwtSecret: process.env.JWT_SECRET || 'dev-only-insecure-secret',
  jwtExpiresIn: process.env.JWT_EXPIRES_IN || '12h',
  vlmEndpoint: process.env.VLM_ENDPOINT || null,
};

export default config;
