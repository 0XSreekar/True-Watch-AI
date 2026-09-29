import { SEARCH_CHIPS, SEARCH_PLACEHOLDERS } from '../data/mockData.js';
import { pad } from '../utils/random.js';

const MAX_QUERY_LENGTH = 500;

/**
 * Plain-language footage search. With EDGE_BASE_URL set the query is proxied to the edge's
 * semantic index (`POST /api/search`) and its answer is returned as-is, because it is already the
 * shape FootageSearch renders. Without the edge, or when the edge has nothing indexed, the
 * fixture results below keep the demo watchable.
 */
export const createSearchService = ({ repos, edge }) => {
  const fixtureResults = (query, limit = 6) => {
    const cameras = repos.cameras.list();
    return {
      query,
      tookMs: 1400,
      windowScanned: '6 weeks',
      results: Array.from({ length: limit }, (_, i) => ({
        id: i,
        camera: cameras[i % cameras.length],
        time: `1${2 + i} AUG · ${pad(20 + (i % 4))}:${pad(11 + i * 6)}`,
        chips: SEARCH_CHIPS[i % SEARCH_CHIPS.length],
        score: 96 - i * 4,
      })),
    };
  };

  return {
    listPlaceholders: () => SEARCH_PLACEHOLDERS,

    async search(rawQuery) {
      const query = String(rawQuery ?? '').slice(0, MAX_QUERY_LENGTH);
      if (edge.enabled && query.trim()) {
        const live = await edge.searchFootage(query);
        if (live && !(live.results.length === 0 && live.windowScanned === 'no footage indexed')) return live;
      }
      return fixtureResults(query);
    },
  };
};

export default createSearchService;
