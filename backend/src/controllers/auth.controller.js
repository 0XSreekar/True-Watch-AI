import * as auth from '../services/auth.service.js';

export const login = (req, res) => {
  const result = auth.login(req.body || {});
  if (!result.ok) return res.status(401).json({ error: result.error });
  return res.json({ data: result });
};

export const register = (req, res) => {
  const result = auth.register(req.body || {});
  if (!result.ok) return res.status(422).json({ error: result.error });
  return res.status(201).json({ data: result });
};

export const units = (_req, res) => res.json({ data: auth.listUnits() });

export const roles = (_req, res) => res.json({ data: auth.listRoles() });
