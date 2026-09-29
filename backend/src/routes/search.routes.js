import { Router } from 'express';

import { createSearchController } from '../controllers/search.controller.js';

export const createSearchRoutes = ({ services, guards }) => {
  const router = Router();
  const controller = createSearchController(services.search);

  router.use(guards.console);
  router.get('/placeholders', controller.placeholders);
  router.post('/', controller.run);

  return router;
};

export default createSearchRoutes;
