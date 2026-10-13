import { Router } from 'express';
import rateLimit from 'express-rate-limit';

import { createAuthController } from '../controllers/auth.controller.js';

/**
 * Public: login, register, units, roles, refresh, logout.
 * Signed-in: GET /me. Supervisor only: POST /approve/:id.
 *
 * Credential endpoints are throttled per client IP. Successful logins do not count against the
 * limit, so the throttle bites on guessing, not on a busy shift change.
 */
export const createAuthRoutes = ({ services, guards, config }) => {
  const router = Router();
  const controller = createAuthController({ auth: services.auth, config });

  const credentialLimiter = (options = {}) =>
    rateLimit({
      windowMs: 15 * 60_000,
      limit: config.authRateLimitPer15Min,
      standardHeaders: 'draft-7',
      legacyHeaders: false,
      message: { error: 'Too many attempts, try again later' },
      ...options,
    });

  router.post('/login', credentialLimiter({ skipSuccessfulRequests: true }), controller.login);
  router.post('/register', credentialLimiter(), controller.register);
  router.get('/units', controller.units);
  router.get('/roles', controller.roles);

  router.post('/refresh', credentialLimiter(), controller.refresh);
  router.post('/logout', guards.optional, controller.logout);
  router.get('/me', guards.requireAuth, controller.me);
  router.post('/approve/:id', guards.role('Supervisor'), controller.approve);

  return router;
};

export default createAuthRoutes;
