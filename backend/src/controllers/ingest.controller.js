/**
 * Edge -> backend controllers. Responses follow docs/ARCHITECTURE_V2.md 3.3:
 * `{ accepted, alert_id, duplicate }`, flat rather than wrapped in `data`, because the only
 * consumer is the edge service, not the console.
 */
export const createIngestController = (ingest) => ({
  event: (req, res) => {
    const { duplicate, merged, alert } = ingest.ingestEvent(req.body);
    const body = { accepted: true, alert_id: alert.id, duplicate };
    if (merged) body.merged = true;
    return res.status(duplicate ? 200 : 201).json(body);
  },

  update: (req, res) => {
    const { alert, updated } = ingest.applyUpdate(req.body);
    return res.json({ accepted: true, alert_id: alert.id, updated });
  },
});

export default createIngestController;
