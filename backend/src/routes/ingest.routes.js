import express, { Router } from 'express';
import rateLimit from 'express-rate-limit';

import { createIngestController } from '../controllers/ingest.controller.js';
import { replayGuard, requireIngestKey } from '../middleware/ingestGuard.js';

/**
 * POST /api/ingest/event   one truewatch.event.v1 (create, or enrich the track's live alert)
 * POST /api/ingest/update  one truewatch.update.v1 (VLM reason patch, stage advance, sealed block)
 *
 * The order matters: throttle first (so key guessing is rate-limited), then authenticate, then
 * replay checks, and only then parse the body, so an unauthenticated caller costs nothing.
 */
export const createIngestRoutes = ({ config, services }) => {
  const router = Router();
  const controller = createIngestController(services.ingest);

  router.use(
    rateLimit({
      windowMs: 60_000,
      limit: config.ingestRateLimitPerMinute,
      standardHeaders: 'draft-7',
      legacyHeaders: false,
      message: { accepted: false, error: 'Too many ingest requests' },
    }),
  );
  router.use(requireIngestKey(config));
  router.use(replayGuard(services.ingest));
  router.use(express.json({ limit: config.bodyLimit }));

  router.post('/event', controller.event);
  router.post('/update', controller.update);

  return router;
};

export default createIngestRoutes;
