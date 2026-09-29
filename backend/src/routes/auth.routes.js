import { Router } from 'express';

import { createAuthController } from '../controllers/auth.controller.js';

/**
 * Public: login, register, units, roles, refresh, logout.
 * Signed-in: GET /me. Supervisor only: POST /approve/:id.
 */
export const createAuthRoutes = ({ services, guards, config }) => {
  const router = Router();
  const controller = createAuthController({ auth: services.auth, config });

  router.post('/login', controller.login);
  router.post('/register', controller.register);
  router.get('/units', controller.units);
  router.get('/roles', controller.roles);

  router.post('/refresh', controller.refresh);
  router.post('/logout', guards.optional, controller.logout);
  router.get('/me', guards.requireAuth, controller.me);
  router.post('/approve/:id', guards.role('Supervisor'), controller.approve);

  return router;
};

export default createAuthRoutes;
