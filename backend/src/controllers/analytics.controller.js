export const createAnalyticsController = (analytics) => ({
  index: (_req, res) => res.json({ data: analytics.getAnalytics() }),
  traffic: (_req, res) => res.json({ data: { sample: analytics.nextTrafficSample() } }),
});

export default createAnalyticsController;
