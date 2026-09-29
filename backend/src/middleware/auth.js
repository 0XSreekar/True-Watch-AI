import { forbidden, unauthorized } from '../utils/httpError.js';

const bearerToken = (req) => {
  const header = req.get('authorization');
  if (!header) return null;
  const [scheme, token] = header.split(' ');
  return scheme?.toLowerCase() === 'bearer' && token ? token : null;
};

/**
 * Builds the three gates the routes use.
 *
 *  - guards.console    every console endpoint. With AUTH_REQUIRED=false (the default until the
 *                      frontend sends a token) it only *identifies* the caller when a valid token is
 *                      present, so audit rows carry an actor; with AUTH_REQUIRED=true it also
 *                      rejects anonymous callers with 401.
 *  - guards.requireAuth  always 401 without a valid token.
 *  - guards.role(...names)  always on, regardless of AUTH_REQUIRED: 401 without a token, 403 when
 *                      the account is unapproved or its role is not listed.
 *
 * The user is re-read from the database on every request, so a role change or a deleted account
 * takes effect immediately rather than when the token expires.
 */
export const createGuards = ({ config, auth }) => {
  const identify = (req) => {
    const token = bearerToken(req);
    req.user = token ? auth.authenticate(token) ?? undefined : undefined;
  };

  const requireAuth = (req, _res, next) => {
    identify(req);
    return req.user ? next() : next(unauthorized());
  };

  const consoleGuard = (req, _res, next) => {
    identify(req);
    if (config.authRequired && !req.user) return next(unauthorized());
    return next();
  };

  const role =
    (...allowed) =>
    (req, _res, next) => {
      identify(req);
      if (!req.user) return next(unauthorized());
      if (!req.user.approved) return next(forbidden('Account is pending approval'));
      if (!allowed.includes(req.user.role)) return next(forbidden(`Requires role: ${allowed.join(' or ')}`));
      return next();
    };

  /** Identify only; never rejects. For endpoints that work anonymously but audit a known caller. */
  const optional = (req, _res, next) => {
    identify(req);
    next();
  };

  return { console: consoleGuard, requireAuth, optional, role };
};

export default createGuards;
