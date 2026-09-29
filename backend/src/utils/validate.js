/**
 * A small strict validator. Objects reject unknown keys, which is what the ingest contract needs:
 * a producer that sends a field this build does not know about is a producer out of step with the
 * contract, and silently dropping the field would hide that.
 *
 * Nodes:  str({ min, max, pattern, oneOf }), int({ min, max }), num({ min, max }), bool(),
 *         lit(value), arr(item, { min, max }), obj(fields), and `.nullable()` / `.optional()`.
 * Errors carry paths and rule names only, never the offending values, so a rejected payload
 * cannot be reflected back to a caller or written to a log.
 */
const make = (check) => {
  const node = { check, isOptional: false, isNullable: false };
  node.optional = () => ({ ...node, isOptional: true });
  node.nullable = () => ({ ...node, isNullable: true });
  return node;
};

export const str = ({ min = 0, max = 4096, pattern, oneOf } = {}) =>
  make((v, path, errors) => {
    if (typeof v !== 'string') return errors.push(`${path}: must be a string`);
    if (v.length < min || v.length > max) return errors.push(`${path}: length must be ${min}..${max}`);
    if (oneOf && !oneOf.includes(v)) return errors.push(`${path}: must be one of ${oneOf.join('|')}`);
    if (pattern && !pattern.test(v)) return errors.push(`${path}: has an invalid format`);
    return undefined;
  });

export const int = ({ min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER } = {}) =>
  make((v, path, errors) => {
    if (!Number.isInteger(v)) return errors.push(`${path}: must be an integer`);
    if (v < min || v > max) return errors.push(`${path}: must be ${min}..${max}`);
    return undefined;
  });

export const num = ({ min = -Infinity, max = Infinity } = {}) =>
  make((v, path, errors) => {
    if (typeof v !== 'number' || !Number.isFinite(v)) return errors.push(`${path}: must be a finite number`);
    if (v < min || v > max) return errors.push(`${path}: must be ${min}..${max}`);
    return undefined;
  });

export const bool = () =>
  make((v, path, errors) => {
    if (typeof v !== 'boolean') errors.push(`${path}: must be a boolean`);
  });

export const lit = (value) =>
  make((v, path, errors) => {
    if (v !== value) errors.push(`${path}: must equal ${JSON.stringify(value)}`);
  });

export const custom = (predicate, message) =>
  make((v, path, errors) => {
    if (!predicate(v)) errors.push(`${path}: ${message}`);
  });

export const arr = (item, { min = 0, max = 1000 } = {}) =>
  make((v, path, errors) => {
    if (!Array.isArray(v)) return errors.push(`${path}: must be an array`);
    if (v.length < min || v.length > max) return errors.push(`${path}: length must be ${min}..${max}`);
    v.forEach((element, i) => run(item, element, `${path}[${i}]`, errors));
    return undefined;
  });

export const obj = (fields) =>
  make((v, path, errors) => {
    if (v === null || typeof v !== 'object' || Array.isArray(v)) return errors.push(`${path}: must be an object`);
    for (const key of Object.keys(v)) {
      if (!(key in fields)) errors.push(`${path}.${key}: unknown field`);
    }
    for (const [key, node] of Object.entries(fields)) {
      const present = Object.prototype.hasOwnProperty.call(v, key) && v[key] !== undefined;
      if (!present) {
        if (!node.isOptional) errors.push(`${path}.${key}: required`);
      } else {
        run(node, v[key], `${path}.${key}`, errors);
      }
    }
    return undefined;
  });

const run = (node, value, path, errors) => {
  if (value === null) {
    if (!node.isNullable) errors.push(`${path}: must not be null`);
    return;
  }
  node.check(value, path, errors);
};

/** Returns { ok: true } or { ok: false, errors: string[] } (capped so a bad payload stays cheap). */
export const validate = (schema, value) => {
  const errors = [];
  run(schema, value, '$', errors);
  return errors.length ? { ok: false, errors: errors.slice(0, 12) } : { ok: true };
};
