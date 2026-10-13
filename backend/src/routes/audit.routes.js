import { Router } from 'express';

/** GET /api/audit — Supervisor only. Recent entries plus a recomputation of the whole hash chain. */
export const createAuditRoutes = ({ services, guards }) => {
  const router = Router();

  router.get('/', guards.role('Supervisor'), (req, res) => {
    const limit = Number.parseInt(req.query.limit, 10);
    res.json({
      data: {
        entries: services.audit.recent(Number.isInteger(limit) && limit > 0 ? Math.min(limit, 500) : 100),
        chain: services.audit.verify(),
      },
    });
  });

  return router;
};

export default createAuditRoutes;
