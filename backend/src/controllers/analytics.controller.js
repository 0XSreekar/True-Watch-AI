import * as analytics from '../services/analytics.service.js';

export const index = (_req, res) => res.json({ data: analytics.getAnalytics() });

export const traffic = (_req, res) => res.json({ data: { sample: analytics.nextTrafficSample() } });
