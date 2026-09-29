import { Router } from 'express';

import { createAlertsRoutes } from './alerts.routes.js';
import { createAnalyticsRoutes } from './analytics.routes.js';
import { createAuditRoutes } from './audit.routes.js';
import { createAuthRoutes } from './auth.routes.js';
import { createCamerasRoutes } from './cameras.routes.js';
import { createEvidenceRoutes } from './evidence.routes.js';
import { createFacesRoutes } from './faces.routes.js';
import { createSearchRoutes } from './search.routes.js';

/** Every /api route except the ones that must be mounted before the JSON parser (ingest, stream). */
export const createRoutes = (container) => {
  const router = Router();

  router.get('/health', (_req, res) =>
    res.json({ status: 'ok', service: 'truewatch-api', time: new Date().toISOString() }),
  );

  router.use('/auth', createAuthRoutes(container));
  router.use('/cameras', createCamerasRoutes(container));
  router.use('/alerts', createAlertsRoutes(container));
  router.use('/search', createSearchRoutes(container));
  router.use('/evidence', createEvidenceRoutes(container));
  router.use('/analytics', createAnalyticsRoutes(container));
  router.use('/faces', createFacesRoutes(container));
  router.use('/audit', createAuditRoutes(container));

  return router;
};

export default createRoutes;
