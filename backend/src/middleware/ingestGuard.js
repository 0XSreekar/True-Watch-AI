import { safeEqual } from '../utils/crypto.js';

/**
 * Shared-secret gate for every /api/ingest route (threat 5 in docs/ARCHITECTURE_V2.md).
 * The key comes from configuration, is compared in constant time and is never logged.
 */
export const requireIngestKey = (config) => (req, res, next) => {
  const presented = req.get('x-ingest-key') ?? req.get('x-truewatch-ingest-key');
  if (!config.ingestKey || !presented || !safeEqual(presented, config.ingestKey)) {
    return res.status(401).json({ accepted: false, error: 'Invalid or missing ingest key' });
  }
  return next();
};

/** Timestamp-skew and nonce replay checks (threat 3). Runs after the key check. */
export const replayGuard = (ingest) => (req, res, next) => {
  const verdict = ingest.checkFreshness({
    timestamp: req.get('x-truewatch-timestamp'),
    nonce: req.get('x-truewatch-nonce'),
  });
  if (verdict.ok) return next();
  return res.status(verdict.status).json({
    accepted: false,
    error: verdict.error,
    ...(verdict.duplicate ? { duplicate: true } : {}),
  });
};
