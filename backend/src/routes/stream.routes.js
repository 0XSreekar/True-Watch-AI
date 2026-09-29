import { Router } from 'express';

/**
 * GET /api/stream — server-sent events. Purely additive: GET /api/alerts polling is untouched,
 * so the console works with the stream absent.
 *
 * Events: alert.created (full alert), alert.updated ({ id, patch, alert } — the VLM reason patch
 * and other field changes), alert.stage ({ id, stage, previous, alert }).
 *
 * Under AUTH_REQUIRED=true the caller must send an Authorization header, so a browser client uses
 * fetch streaming rather than EventSource (which cannot set headers). The token is deliberately
 * not accepted in the query string.
 */
export const createStreamRoutes = ({ hub, guards }) => {
  const router = Router();
  router.get('/', guards.console, hub.handler);
  return router;
};

export default createStreamRoutes;
