import { Router } from 'express';

import { createCamerasController } from '../controllers/cameras.controller.js';

/**
 * Frozen reads: GET /, /sectors, /posts, /:id.
 * Additive: GET /:id/settings (console), PUT /:id/fence and PUT /:id/hours (Supervisor only, audited).
 */
export const createCamerasRoutes = ({ services, guards }) => {
  const router = Router();
  const controller = createCamerasController(services.cameras);

  router.use(guards.console);
  router.get('/', controller.index);
  router.get('/sectors', controller.sectors);
  router.get('/posts', controller.posts);
  router.get('/:id', controller.show);
  router.get('/:id/settings', controller.settings);
  router.put('/:id/fence', guards.role('Supervisor'), controller.updateFence);
  router.put('/:id/hours', guards.role('Supervisor'), controller.updateHours);

  return router;
};

export default createCamerasRoutes;
