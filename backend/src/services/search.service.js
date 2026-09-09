import { CAMERAS, SEARCH_CHIPS, SEARCH_PLACEHOLDERS } from '../data/mockData.js';
import { pad } from '../utils/random.js';

/**
 * Plain-language footage search.
 * Real system: embed the query, run it against the clip index, rerank with the VLM.
 */
export const listPlaceholders = () => SEARCH_PLACEHOLDERS;

export const search = (query, { limit = 6 } = {}) => {
  const results = [];
  for (let i = 0; i < limit; i += 1) {
    const camera = CAMERAS[i % CAMERAS.length];
    results.push({
      id: i,
      camera,
      time: `1${2 + i} AUG · ${pad(20 + (i % 4))}:${pad(11 + i * 6)}`,
      chips: SEARCH_CHIPS[i % SEARCH_CHIPS.length],
      score: 96 - i * 4,
    });
  }
  return { query, tookMs: 1400, windowScanned: '6 weeks', results };
};
