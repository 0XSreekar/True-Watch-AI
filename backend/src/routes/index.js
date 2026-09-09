import { Router } from 'express';
import alerts from './alerts.routes.js';
import analytics from './analytics.routes.js';
import auth from './auth.routes.js';
import cameras from './cameras.routes.js';
import evidence from './evidence.routes.js';
import search from './search.routes.js';

const router = Router();

router.get('/health', (_req, res) =>
  res.json({ status: 'ok', service: 'truewatch-api', time: new Date().toISOString() }),
);

router.use('/auth', auth);
router.use('/cameras', cameras);
router.use('/alerts', alerts);
router.use('/search', search);
router.use('/evidence', evidence);
router.use('/analytics', analytics);

// Future modules mount here, e.g.
// router.use('/fences', fences);
// router.use('/reports', reports);

export default router;
