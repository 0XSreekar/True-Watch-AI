import { Router } from 'express';

/** POST /api/faces/:alertId/unseal — Supervisor only, audited. Body: { reason }. */
export const createFacesRoutes = ({ services, guards }) => {
  const router = Router();

  router.post('/:alertId/unseal', guards.role('Supervisor'), (req, res) =>
    res.json({ data: services.faces.unseal(req.params.alertId, req.body, req.user) }),
  );

  return router;
};

export default createFacesRoutes;
