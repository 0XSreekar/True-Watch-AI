import assert from 'node:assert/strict';

/**
 * Exact-shape assertions for the frozen contract.
 *
 *   'string' | 'number' | 'boolean' | 'null'   a primitive type
 *   'string|null'                              a union of primitives
 *   { $array: shape, min? }                    an array whose every element matches shape
 *   { a: shape, b: shape }                     an object with EXACTLY those keys (no more, no fewer)
 *   { $exact: [..] }                           a fixed-length tuple
 *
 * A failure names the JSON path, so a broken UI contract is obvious from the test output alone.
 */
const typeOf = (v) => {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  return typeof v;
};

export const expectShape = (value, shape, path = '$') => {
  if (typeof shape === 'string') {
    const allowed = shape.split('|');
    assert.ok(allowed.includes(typeOf(value)), `${path}: expected ${shape}, got ${typeOf(value)}`);
    return;
  }
  if (shape.$array !== undefined) {
    assert.ok(Array.isArray(value), `${path}: expected an array, got ${typeOf(value)}`);
    assert.ok(value.length >= (shape.min ?? 0), `${path}: expected at least ${shape.min} elements`);
    value.forEach((item, i) => expectShape(item, shape.$array, `${path}[${i}]`));
    return;
  }
  if (shape.$exact !== undefined) {
    assert.ok(Array.isArray(value) && value.length === shape.$exact.length, `${path}: expected ${shape.$exact.length} elements`);
    shape.$exact.forEach((s, i) => expectShape(value[i], s, `${path}[${i}]`));
    return;
  }
  assert.equal(typeOf(value), 'object', `${path}: expected an object, got ${typeOf(value)}`);
  assert.deepEqual(Object.keys(value).sort(), Object.keys(shape).sort(), `${path}: key set changed`);
  for (const [key, sub] of Object.entries(shape)) expectShape(value[key], sub, `${path}.${key}`);
};

/** The frozen shapes, written out once. Any change here is a deliberate contract change. */
export const SHAPES = {
  camera: { id: 'string', name: 'string', sector: 'string', ir: 'boolean', road: 'boolean' },
  post: { id: 'string', name: 'string', x: 'number', y: 'number', road: 'boolean', alert: 'boolean' },
  channels: { appearance: 'boolean', motion: 'boolean' },
  budget: { used: 'number', total: 'number', near: 'boolean' },
  block: { n: 'number', hash: 'string', camera: 'string', time: 'string' },
  verify: { intact: 'boolean', verified: 'number', total: 'number', blocks: { $array: { n: 'number', ok: 'boolean' } } },
  user: { id: 'string', name: 'string', role: 'string', unit: 'string', initials: 'string' },
  role: { k: 'string', dd: 'string' },
  analytics: {
    camBars: { $array: { id: 'string', v: 'number' }, min: 1 },
    dayNight: { $array: { k: 'string', v: 'number', c: 'string' }, min: 2 },
    agreement: { $array: 'number', min: 1 },
    traffic: { $array: 'number', min: 1 },
  },
  search: {
    query: 'string',
    tookMs: 'number',
    windowScanned: 'string',
    results: {
      $array: {
        id: 'number',
        camera: { id: 'string', name: 'string', sector: 'string', ir: 'boolean', road: 'boolean' },
        time: 'string',
        chips: { $array: 'string' },
        score: 'number',
      },
    },
  },
};

SHAPES.alert = {
  id: 'string',
  camera: SHAPES.camera,
  stage: 'string',
  time: 'string',
  reason: 'string',
  confidence: 'number',
  plate: 'string|null',
  hash: 'string',
  seed: 'number',
  channels: SHAPES.channels,
};

/** Every key an alert may carry beyond the frozen ones (docs/ARCHITECTURE_V2.md 3.3, plus honesty flags). */
export const ADDITIVE_ALERT_KEYS = [
  'track_id',
  'bbox',
  'channel_scores',
  'rule_fired',
  'dori_band',
  'evidence_id',
  'clip_url',
  'thumb_url',
  'reason_source',
  'source',
];

/** An alert must contain every frozen key with the frozen types, and may add only documented keys. */
export const expectAlert = (alert) => {
  const frozen = Object.keys(SHAPES.alert);
  const extra = Object.keys(alert).filter((k) => !frozen.includes(k));
  const undocumented = extra.filter((k) => !ADDITIVE_ALERT_KEYS.includes(k));
  assert.deepEqual(undocumented, [], `alert carries undocumented keys: ${undocumented.join(', ')}`);
  const projected = Object.fromEntries(frozen.map((k) => [k, alert[k]]));
  expectShape(projected, SHAPES.alert, '$alert');
  assert.ok(['prov', 'conf', 'sealed', 'gone'].includes(alert.stage), `unknown stage ${alert.stage}`);
  assert.ok(alert.reason.length > 0, 'reason must never be empty');
  assert.ok(Number.isFinite(alert.seed), 'seed must be finite');
};
