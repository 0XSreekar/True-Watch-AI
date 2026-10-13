import crypto from 'node:crypto';

export const ZERO_HASH = '0'.repeat(64);

export const sha256 = (...parts) => {
  const h = crypto.createHash('sha256');
  for (const part of parts) h.update(String(part));
  return h.digest('hex');
};

/**
 * Constant-time string comparison. Both sides are hashed first so the comparison runs over
 * equal-length buffers and the length of the secret is not leaked either.
 */
export const safeEqual = (a, b) => {
  const left = crypto.createHash('sha256').update(String(a ?? '')).digest();
  const right = crypto.createHash('sha256').update(String(b ?? '')).digest();
  return crypto.timingSafeEqual(left, right);
};

export const randomToken = (bytes = 32) => crypto.randomBytes(bytes).toString('base64url');

/** '64 hex chars' -> 'aaaaaaaa…bbbbbb', the short form the console prints. */
export const shortHash = (hex, head = 8, tail = 6) =>
  typeof hex === 'string' && hex.length >= head + tail ? `${hex.slice(0, head)}…${hex.slice(-tail)}` : null;

/** Deterministic float in [0, 100) from any string; SceneCanvas only needs a stable number. */
export const seedFrom = (value) => {
  const digest = crypto.createHash('sha256').update(String(value)).digest();
  return Number(((digest.readUInt32BE(0) / 0xffffffff) * 100).toFixed(2));
};
