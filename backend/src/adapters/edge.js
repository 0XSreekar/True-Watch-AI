/**
 * The only place the backend talks HTTP to the edge service. Every method is bounded by a
 * timeout, validates the shape it gets back, and answers `null` (or `false`) instead of throwing
 * when the edge is absent, slow or wrong, so callers can fall back without a try/catch each.
 *
 * Paths are fixed constants and the base URL comes from EDGE_BASE_URL only; nothing a client
 * sends is ever used to build a URL.
 */
const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

const isVerifyResult = (v) =>
  isObject(v) &&
  typeof v.intact === 'boolean' &&
  Number.isInteger(v.verified) &&
  Number.isInteger(v.total) &&
  Array.isArray(v.blocks) &&
  v.blocks.every((b) => isObject(b) && Number.isInteger(b.n) && typeof b.ok === 'boolean');

const isCamera = (c) =>
  isObject(c) && typeof c.id === 'string' && typeof c.name === 'string' && typeof c.sector === 'string';

const isSearchResult = (v) =>
  isObject(v) &&
  typeof v.query === 'string' &&
  typeof v.tookMs === 'number' &&
  typeof v.windowScanned === 'string' &&
  Array.isArray(v.results) &&
  v.results.every(
    (r) =>
      isObject(r) &&
      isCamera(r.camera) &&
      typeof r.time === 'string' &&
      Array.isArray(r.chips) &&
      typeof r.score === 'number',
  );

export const createEdgeClient = ({ baseUrl, timeoutMs = 2500, searchTimeoutMs = 8000, log = () => {} }) => {
  const enabled = Boolean(baseUrl);

  const call = async (method, path, body, timeout) => {
    if (!enabled) return null;
    try {
      const response = await fetch(`${baseUrl}${path}`, {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(timeout),
      });
      if (!response.ok) {
        log('warn', `edge ${method} ${path} answered ${response.status}`);
        return null;
      }
      return await response.json();
    } catch (error) {
      log('warn', `edge ${method} ${path} unavailable: ${error.name}`);
      return null;
    }
  };

  return {
    enabled,

    /** Edge-side chain verification. Null when the edge is absent or answers off-contract. */
    async verifyEvidence() {
      const result = await call('POST', '/evidence/verify', {}, timeoutMs);
      return isVerifyResult(result) ? result : null;
    },

    /**
     * The edge's own hash chain (the authoritative one: clips are sealed there at capture), in the
     * frozen `{ n, hash, camera, time }` shape. Null when the edge is absent or answers off-contract.
     */
    async evidenceChain(limit) {
      const result = await call('GET', `/evidence/chain?limit=${Number(limit) || 20}`, null, timeoutMs);
      const ok =
        Array.isArray(result) &&
        result.every(
          (b) => b && Number.isInteger(b.n) && typeof b.hash === 'string' && typeof b.camera === 'string' && typeof b.time === 'string',
        );
      return ok ? result.map(({ n, hash, camera, time }) => ({ n, hash, camera, time })) : null;
    },

    /** Semantic footage search on the edge. Null when unavailable or off-contract. */
    async searchFootage(query) {
      const result = await call('POST', '/api/search', { query }, searchTimeoutMs);
      return isSearchResult(result) ? result : null;
    },

    /**
     * Operator dismissed an alert: tell the edge so that camera's baseline can learn from it.
     * Best effort by design; a dismissal must never fail because the edge is asleep.
     */
    async sendBaselineFeedback(feedback) {
      const result = await call('POST', '/pipeline/feedback', feedback, timeoutMs);
      return result !== null;
    },
  };
};

export default createEdgeClient;
