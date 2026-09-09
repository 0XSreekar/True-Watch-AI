import * as search from '../services/search.service.js';

export const placeholders = (_req, res) => res.json({ data: search.listPlaceholders() });

export const run = (req, res) => {
  const query = (req.body?.query ?? req.query.q ?? '').toString();
  res.json({ data: search.search(query) });
};
