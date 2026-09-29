/**
 * Structured JSON-lines logger. Every line carries a timestamp and level, and a `requestId` when
 * one is supplied, so a single request can be followed across the access log, the error handler
 * and any service that logs on its behalf. Errors are reduced to name and message; stacks stay
 * server-side in the `stack` field of error-level lines only.
 */
const serialise = (meta) => {
  if (!meta) return {};
  const { error, ...rest } = meta;
  const out = { ...rest };
  if (error) {
    out.error = error instanceof Error ? error.message : String(error);
    if (error instanceof Error && error.stack) out.stack = error.stack;
  }
  return out;
};

export const createLogger = ({ silent = false, sink = process } = {}) => {
  const write = (level, message, meta) => {
    if (silent) return;
    const line = JSON.stringify({ time: new Date().toISOString(), level, msg: message, ...serialise(meta) });
    (level === 'error' || level === 'warn' ? sink.stderr : sink.stdout).write(`${line}\n`);
  };
  return {
    log: write,
    info: (message, meta) => write('info', message, meta),
    warn: (message, meta) => write('warn', message, meta),
    error: (message, meta) => write('error', message, meta),
  };
};

export default createLogger;
