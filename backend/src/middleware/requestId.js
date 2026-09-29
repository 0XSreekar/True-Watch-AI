import { randomUUID } from 'node:crypto';

const SAFE_ID = /^[A-Za-z0-9._-]{8,64}$/;

/**
 * Gives every request an id: the caller's X-Request-Id when it is a short, safe token (so a proxy
 * can correlate), otherwise a fresh UUID. The id is echoed in the response header.
 */
export const requestId = (req, res, next) => {
  const incoming = req.get('x-request-id');
  req.id = incoming && SAFE_ID.test(incoming) ? incoming : randomUUID();
  res.set('X-Request-Id', req.id);
  next();
};

export default requestId;
