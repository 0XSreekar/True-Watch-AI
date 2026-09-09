import * as evidence from '../services/evidence.service.js';

export const chain = (_req, res) => res.json({ data: evidence.getChain() });

export const verify = (_req, res) => res.json({ data: evidence.verifyChain() });
