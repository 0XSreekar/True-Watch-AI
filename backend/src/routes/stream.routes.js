import { Router } from 'express';

/**
 * GET /api/stream — server-sent events. Purely additive: GET /api/alerts polling is untouched,
 * so the console works with the stream absent.
 *
 * Events: alert.created (full alert), alert.updated ({ id, patch, alert } — the VLM reason patch
 * and other field changes), alert.stage ({ id, stage, previous, alert }).
 */
export const createStreamRoutes = ({ hub }) => {
  const router = Router();
  router.get('/', hub.handler);
  return router;
};

export default createStreamRoutes;
