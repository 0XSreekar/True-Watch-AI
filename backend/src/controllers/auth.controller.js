import { forbidden, notFound } from '../utils/httpError.js';

export const REFRESH_COOKIE = 'tw_refresh';

/**
 * Login/register keep their frozen envelopes: success `{ data: { ok, token, expiresIn, user } }`
 * (register adds pendingApproval), failure `{ error }` with 401 / 422. The refresh token is set
 * as an httpOnly cookie and is never present in a response body.
 */
export const createAuthController = ({ auth, config }) => {
  const cookieOptions = (maxAgeMs) => ({
    httpOnly: true,
    secure: config.production,
    sameSite: 'strict',
    path: '/api/auth',
    maxAge: maxAgeMs,
  });

  const respondWithSession = (res, result, status = 200) => {
    const { refresh, ...body } = result;
    res.cookie(REFRESH_COOKIE, refresh.token, cookieOptions(refresh.maxAgeMs));
    return res.status(status).json({ data: body });
  };

  /** Defence in depth for cookie-authenticated calls: a browser Origin must be on the allow-list. */
  const assertTrustedOrigin = (req) => {
    const origin = req.get('origin');
    if (origin && !config.corsOrigins.includes(origin)) throw forbidden('Origin not allowed');
  };

  return {
    login: async (req, res) => {
      const result = await auth.login(req.body || {});
      if (!result.ok) return res.status(401).json({ error: result.error });
      return respondWithSession(res, result);
    },

    register: async (req, res) => {
      const result = await auth.register(req.body || {});
      if (!result.ok) return res.status(422).json({ error: result.error });
      return respondWithSession(res, result, 201);
    },

    refresh: async (req, res) => {
      assertTrustedOrigin(req);
      const result = await auth.refresh(req.cookies?.[REFRESH_COOKIE]);
      if (!result) {
        res.clearCookie(REFRESH_COOKIE, cookieOptions());
        return res.status(401).json({ error: 'Session expired' });
      }
      return respondWithSession(res, result);
    },

    logout: (req, res) => {
      assertTrustedOrigin(req);
      const result = auth.logout(req.cookies?.[REFRESH_COOKIE], req.user ?? null);
      res.clearCookie(REFRESH_COOKIE, cookieOptions());
      return res.json({ data: result });
    },

    me: (req, res) => res.json({ data: { user: req.user } }),

    approve: (req, res) => {
      const result = auth.approve(req.params.id, req.user);
      if (!result) throw notFound('User not found');
      return res.json({ data: result });
    },

    units: (_req, res) => res.json({ data: auth.listUnits() }),
    roles: (_req, res) => res.json({ data: auth.listRoles() }),
  };
};

export default createAuthController;
