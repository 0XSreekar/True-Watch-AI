import { isRfc3339Utc } from '../utils/time.js';
import { arr, bool, custom, int, lit, num, obj, str, validate } from '../utils/validate.js';

/**
 * docs/ARCHITECTURE_V2.md section 4.2 as an executable, strict schema.
 * Unknown fields are rejected at every level (400), not ignored.
 */

export const EVENT_SCHEMA_ID = 'truewatch.event.v1';
export const UPDATE_SCHEMA_ID = 'truewatch.update.v1';

const UUID = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;
const SHA256_HEX = /^[0-9a-f]{64}$/;
const ID = /^[A-Za-z0-9_.-]{1,64}$/;
const HTTP_URL = /^https?:\/\/[^\s]{1,2040}$/;

const unit = num({ min: 0, max: 1 });
const rfc3339 = custom(isRfc3339Utc, 'must be an RFC 3339 UTC timestamp ending in Z');

const explanation = obj({
  text: str({ min: 1, max: 1000 }),
  source: str({ oneOf: ['vlm', 'template'] }),
  model: str({ max: 128 }).nullable().optional(),
  latency_ms: int({ min: 0, max: 600000 }).nullable().optional(),
});

const anpr = obj({
  text: str({ min: 1, max: 64 }),
  script: str({ oneOf: ['devanagari', 'latin'] }),
  line_confidences: arr(unit, { max: 8 }).optional(),
});

const evidence = obj({
  evidence_id: str({ pattern: ID }),
  block_n: int({ min: 0 }),
  clip_sha256: str({ pattern: SHA256_HEX }),
  prev_sha256: str({ pattern: SHA256_HEX }),
  clip_url: str({ pattern: HTTP_URL }).nullable().optional(),
  thumb_url: str({ pattern: HTTP_URL }).nullable().optional(),
  sealed_at: rfc3339,
});

const eventSchema = obj({
  schema: lit(EVENT_SCHEMA_ID),
  event_id: str({ pattern: UUID }),
  post_id: str({ pattern: ID }),
  camera_id: str({ pattern: ID }),
  captured_at: rfc3339,
  stage: str({ oneOf: ['provisional', 'confirmed', 'sealed'] }),
  detection: obj({
    track_id: int({ min: 0 }).nullable(),
    class: str({ oneOf: ['person', 'truck', 'car', 'two_wheeler', 'cart'] }),
    bbox: arr(unit, { min: 4, max: 4 }).nullable(),
    channel_scores: obj({ appearance: unit, motion: unit }),
    agreed: lit(true),
    threshold: unit,
    ir: bool(),
    dori_band: str({ oneOf: ['detect', 'observe', 'recognise', 'identify'] }).nullable(),
  }),
  rule: obj({
    fired: arr(str({ min: 1, max: 64 }), { max: 8 }),
    baseline: obj({
      dominant_direction: str({ max: 16 }).nullable(),
      with: int({ min: 0 }),
      against: int({ min: 0 }),
    }).nullable(),
    metrics: obj({
      dwell_s: num({ min: 0 }).optional(),
      net_px: num({ min: 0 }).optional(),
      path_px: num({ min: 0 }).optional(),
    }),
  }).nullable(),
  explanation: explanation.nullable(),
  anpr: anpr.nullable(),
  evidence: evidence.nullable(),
  confidence: int({ min: 0, max: 100 }),
  source: str({ oneOf: ['rtsp', 'file'] }),
});

const updateSchema = obj({
  schema: lit(UPDATE_SCHEMA_ID),
  event_id: str({ pattern: UUID }),
  stage: str({ oneOf: ['confirmed', 'sealed'] }).optional(),
  explanation: explanation.optional(),
  anpr: anpr.optional(),
  evidence: evidence.optional(),
});

/** Cross-field rules the per-field schema cannot express. Returns extra error strings. */
const eventConsistency = (event) => {
  const errors = [];
  if (event.stage === 'provisional' && event.explanation) {
    errors.push('$.explanation: must be null while stage is provisional');
  }
  if (event.stage !== 'provisional' && !event.explanation) {
    errors.push('$.explanation: required once stage is confirmed or sealed');
  }
  if (event.stage !== 'sealed' && event.evidence) errors.push('$.evidence: must be null until stage is sealed');
  if (event.stage === 'sealed' && !event.evidence) errors.push('$.evidence: required when stage is sealed');
  return errors;
};

const updateConsistency = (update) => {
  const errors = [];
  if (!update.stage && !update.explanation && !update.anpr && !update.evidence) {
    errors.push('$: must carry at least one of stage, explanation, anpr, evidence');
  }
  if (update.stage === 'sealed' && !update.evidence) {
    errors.push('$.evidence: required when stage is sealed');
  }
  return errors;
};

export const validateEvent = (body) => {
  const result = validate(eventSchema, body);
  if (!result.ok) return result;
  const extra = eventConsistency(body);
  return extra.length ? { ok: false, errors: extra } : result;
};

export const validateUpdate = (body) => {
  const result = validate(updateSchema, body);
  if (!result.ok) return result;
  const extra = updateConsistency(body);
  return extra.length ? { ok: false, errors: extra } : result;
};
