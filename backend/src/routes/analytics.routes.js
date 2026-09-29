import { Router } from 'express';

import { createAnalyticsController } from '../controllers/analytics.controller.js';

export const createAnalyticsRoutes = ({ services, guards }) => {
  const router = Router();
  const controller = createAnalyticsController(services.analytics);

  router.use(guards.console);
  router.get('/', controller.index);
  router.get('/traffic', controller.traffic);

  return router;
};

export default createAnalyticsRoutes;
