import { Router } from 'express';

import { createCamerasController } from '../controllers/cameras.controller.js';

export const createCamerasRoutes = ({ services, guards }) => {
  const router = Router();
  const controller = createCamerasController(services.cameras);

  router.use(guards.console);
  router.get('/', controller.index);
  router.get('/sectors', controller.sectors);
  router.get('/posts', controller.posts);
  router.get('/:id', controller.show);

  return router;
};

export default createCamerasRoutes;
