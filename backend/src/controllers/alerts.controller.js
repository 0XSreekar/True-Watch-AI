import * as alerts from '../services/alerts.service.js';

export const index = (_req, res) =>
  res.json({ data: alerts.listAlerts(), budget: alerts.getBudget() });

export const simulate = (_req, res) =>
  res.status(201).json({ data: alerts.simulateAlert(), budget: alerts.getBudget() });

export const act = (req, res) => {
  const alert = alerts.actOnAlert(req.params.id);
  if (!alert) return res.status(404).json({ error: 'Alert not found' });
  return res.json({ data: alert });
};

export const dismiss = (req, res) => {
  const alert = alerts.dismissAlert(req.params.id);
  if (!alert) return res.status(404).json({ error: 'Alert not found' });
  return res.json({ data: alert });
};

export const budget = (_req, res) => res.json({ data: alerts.getBudget() });
