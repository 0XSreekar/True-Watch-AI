import { Router } from 'express';

import { createEvidenceController } from '../controllers/evidence.controller.js';

export const createEvidenceRoutes = ({ services, guards }) => {
  const router = Router();
  const controller = createEvidenceController(services.evidence);

  router.use(guards.console);
  router.get('/chain', controller.chain);
  router.post('/verify', controller.verify);
  router.get('/root/:date', controller.root);
  router.get('/:id', controller.block);

  return router;
};

export default createEvidenceRoutes;
