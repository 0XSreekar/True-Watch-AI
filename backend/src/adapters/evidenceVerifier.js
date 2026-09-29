import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { blockDigest } from '../repositories/evidence.repository.js';

/**
 * Evidence verification behind one small interface: `verify() -> { intact, verified, total, blocks }`.
 *
 * Strategies, tried in order:
 *   1. the edge's own verifier over HTTP (when EDGE_BASE_URL is set and it answers in shape);
 *   2. a local walk over the stored hash links.
 * The seam is deliberate: a strategy that shells out to `python -m evidence.verify` slots in as
 * another entry in `strategies` without touching the service or the controller.
 *
 * The local walk checks, per block: the block hash recomputes from its own fields, its
 * `prev_sha256` equals its predecessor's `block_sha256`, and, when the clip file exists under
 * EVIDENCE_STORE_PATH, that the file still hashes to `clip_sha256` (threat 2, clip substitution).
 * `clip_path` comes from the database, never from a request, and must stay inside the store.
 */
const CHUNK = 1024 * 1024;

const hashFile = (file) => {
  const hash = crypto.createHash('sha256');
  const fd = fs.openSync(file, 'r');
  try {
    const buffer = Buffer.allocUnsafe(CHUNK);
    for (;;) {
      const read = fs.readSync(fd, buffer, 0, CHUNK, null);
      if (read === 0) break;
      hash.update(buffer.subarray(0, read));
    }
  } finally {
    fs.closeSync(fd);
  }
  return hash.digest('hex');
};

/** Returns the on-disk path when the clip exists inside the store, otherwise null (not checkable). */
const resolveClip = (storeRoot, clipPath) => {
  if (!storeRoot || storeRoot === ':memory:' || clipPath.includes('://')) return null;
  const root = path.resolve(storeRoot);
  const target = path.resolve(root, clipPath);
  if (target !== root && !target.startsWith(root + path.sep)) return null;
  try {
    return fs.statSync(target).isFile() ? target : null;
  } catch {
    return null;
  }
};

export const verifyStoredChain = (rows, { evidenceStorePath } = {}) => {
  let previous = null;
  const results = rows.map((row) => {
    let ok = blockDigest(row) === row.block_sha256;
    if (ok && previous) ok = row.prev_sha256 === previous.block_sha256;
    if (ok) {
      const file = resolveClip(evidenceStorePath, row.clip_path);
      if (file) ok = hashFile(file) === row.clip_sha256;
    }
    previous = row;
    return { n: row.n, ok };
  });
  const verified = results.filter((r) => r.ok).length;
  return { intact: verified === results.length, verified, total: results.length, blocks: results };
};

export const createEvidenceVerifier = ({ repos, edge, config, windowSize = 20 }) => {
  const local = async () => {
    const full = verifyStoredChain(repos.evidence.allRows(), config);
    // The response lists the same recent window as GET /api/evidence/chain; the counts span the chain.
    return { ...full, blocks: full.blocks.slice(-windowSize) };
  };

  const viaEdge = async () => {
    if (!edge.enabled) return null;
    const result = await edge.verifyEvidence();
    return result ? { ...result, blocks: result.blocks.slice(-windowSize) } : null;
  };

  const strategies = [viaEdge, local];

  return {
    async verify() {
      for (const strategy of strategies) {
        const result = await strategy();
        if (result) return result;
      }
      return local();
    },
  };
};

export default createEvidenceVerifier;
