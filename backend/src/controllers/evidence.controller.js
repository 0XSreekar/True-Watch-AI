import { badRequest } from '../utils/httpError.js';

const DAY = /^\d{4}-\d{2}-\d{2}$/;

export const createEvidenceController = (evidence) => ({
  chain: async (_req, res) => res.json({ data: await evidence.getChain() }),

  verify: async (req, res) => res.json({ data: await evidence.verifyChain(req.user ?? null) }),

  block: (req, res) => res.json({ data: evidence.getBlock(req.params.id) }),

  root: (req, res) => {
    if (!DAY.test(req.params.date) || Number.isNaN(Date.parse(req.params.date))) {
      throw badRequest('Date must be YYYY-MM-DD');
    }
    return res.json({ data: evidence.getDailyRoot(req.params.date) });
  },
});

export default createEvidenceController;
