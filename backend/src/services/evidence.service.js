import { CAMERAS } from '../data/mockData.js';
import { hex, pad } from '../utils/random.js';

/**
 * SHA-256 hash chain over captured clips.
 * Real system: digests computed on the edge box at capture and anchored periodically.
 */
const chain = Array.from({ length: 5 }, (_, i) => ({
  n: 1041 + i,
  hash: `${hex(10)}…${hex(6)}`,
  camera: CAMERAS[i % CAMERAS.length].id,
  time: `14 AUG · ${pad(1 + i)}:${pad(12 + i * 7)}`,
}));

export const getChain = () => chain;

/** Verifies every block links to its predecessor. Always intact in the demo. */
export const verifyChain = () => ({
  intact: true,
  verified: chain.length,
  total: chain.length,
  blocks: chain.map((b) => ({ n: b.n, ok: true })),
});
