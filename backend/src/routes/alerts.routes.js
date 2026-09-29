import { Router } from 'express';

import { createAlertsController } from '../controllers/alerts.controller.js';

export const createAlertsRoutes = ({ services, guards }) => {
  const router = Router();
  const controller = createAlertsController(services.alerts);

  router.use(guards.console);
  router.get('/', controller.index);
  router.get('/budget', controller.budget);
  router.post('/simulate', controller.simulate);
  router.post('/:id/act', controller.act);
  router.post('/:id/dismiss', controller.dismiss);

  return router;
};

export default createAlertsRoutes;
