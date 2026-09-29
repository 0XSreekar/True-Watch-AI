/** Envelope: success `{ data }`, failure `{ error }`. The list and simulate answers also carry `budget`. */
export const createAlertsController = (alerts) => ({
  index: (req, res) => {
    const limit = Number.parseInt(req.query.limit, 10);
    res.json({
      data: alerts.listAlerts({ limit: Number.isInteger(limit) && limit > 0 ? Math.min(limit, 200) : undefined }),
      budget: alerts.getBudget(),
    });
  },

  // Demo path: fixture-driven alert for offline demos. Live alerts arrive via POST /api/ingest/event.
  simulate: (_req, res) => res.status(201).json({ data: alerts.simulateAlert(), budget: alerts.getBudget() }),

  act: (req, res) => {
    const alert = alerts.actOnAlert(req.params.id, req.user ?? null);
    if (!alert) return res.status(404).json({ error: 'Alert not found' });
    return res.json({ data: alert });
  },

  dismiss: (req, res) => {
    const alert = alerts.dismissAlert(req.params.id, req.user ?? null);
    if (!alert) return res.status(404).json({ error: 'Alert not found' });
    return res.json({ data: alert });
  },

  budget: (_req, res) => res.json({ data: alerts.getBudget() }),
});

export default createAlertsController;
