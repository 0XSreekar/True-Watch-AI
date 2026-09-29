const pad = (n) => (n < 10 ? `0${n}` : `${n}`);

const MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];

/** 'HH:MM:SS' in the server's local time, matching what the operator console has always shown. */
export const hhmmss = (iso) => {
  const d = new Date(iso);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
};

/** '14 AUG · 01:12', the frozen evidence-block display label. */
export const evidenceLabel = (iso) => {
  const d = new Date(iso);
  return `${pad(d.getDate())} ${MONTHS[d.getMonth()]} · ${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

const RFC3339_UTC = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?Z$/;

/** Strict RFC 3339 UTC ('Z' suffix) that also parses to a real instant. */
export const isRfc3339Utc = (value) =>
  typeof value === 'string' && RFC3339_UTC.test(value) && !Number.isNaN(Date.parse(value));

export const dayOf = (iso) => new Date(iso).toISOString().slice(0, 10);
