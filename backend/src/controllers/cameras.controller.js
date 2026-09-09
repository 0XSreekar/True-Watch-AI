import * as cameras from '../services/cameras.service.js';

export const index = (req, res) => {
  res.json({ data: cameras.listCameras({ sector: req.query.sector }) });
};

export const show = (req, res) => {
  const camera = cameras.getCamera(req.params.id);
  if (!camera) return res.status(404).json({ error: 'Camera not found' });
  return res.json({ data: camera });
};

export const posts = (_req, res) => res.json({ data: cameras.listPosts() });

export const sectors = (_req, res) => res.json({ data: cameras.listSectors() });
