export const createSearchController = (search) => ({
  placeholders: (_req, res) => res.json({ data: search.listPlaceholders() }),

  run: async (req, res) => {
    const query = (req.body?.query ?? req.query.q ?? '').toString();
    res.json({ data: await search.search(query) });
  },
});

export default createSearchController;
