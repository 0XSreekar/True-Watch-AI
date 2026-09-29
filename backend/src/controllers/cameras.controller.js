export const createCamerasController = (cameras) => ({
  index: (req, res) => {
    res.json({ data: cameras.listCameras({ sector: req.query.sector }) });
  },

  show: (req, res) => {
    const camera = cameras.getCamera(req.params.id);
    if (!camera) return res.status(404).json({ error: 'Camera not found' });
    return res.json({ data: camera });
  },

  posts: (_req, res) => res.json({ data: cameras.listPosts() }),

  sectors: (_req, res) => res.json({ data: cameras.listSectors() }),

  settings: (req, res) => res.json({ data: cameras.getSettings(req.params.id) }),

  updateFence: (req, res) => res.json({ data: cameras.updateFence(req.params.id, req.body, req.user ?? null) }),

  updateHours: (req, res) => res.json({ data: cameras.updateHours(req.params.id, req.body, req.user ?? null) }),
});

export default createCamerasController;
