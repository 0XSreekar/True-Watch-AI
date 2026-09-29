import { sha256 } from '../utils/crypto.js';
import { notFound } from '../utils/httpError.js';

const CHAIN_WINDOW = 20;

/** Merkle root over block hashes: pairs are hashed left||right, an odd node is paired with itself. */
export const merkleRoot = (leaves) => {
  let level = leaves;
  while (level.length > 1) {
    const next = [];
    for (let i = 0; i < level.length; i += 2) next.push(sha256(level[i], level[i + 1] ?? level[i]));
    level = next;
  }
  return level[0];
};

export const createEvidenceService = ({ repos, verifier, audit }) => ({
  /** Frozen shape: [{ n, hash, camera, time }], the latest window in ascending order. */
  getChain: () => repos.evidence.chain(CHAIN_WINDOW),

  /** Frozen shape: { intact, verified, total, blocks:[{ n, ok }] }. Every verification is audited. */
  async verifyChain(actor) {
    const result = await verifier.verify();
    audit.record(actor, 'evidence.verify', 'chain', {
      intact: result.intact,
      verified: result.verified,
      total: result.total,
    });
    return result;
  },

  /** One block with its predecessor link. The stored clip path is never exposed. */
  getBlock(evidenceId) {
    const row = repos.evidence.findByEvidenceId(evidenceId);
    if (!row) throw notFound('Evidence block not found');
    return {
      n: row.n,
      evidence_id: row.evidence_id,
      camera: row.camera_id,
      time: row.time_label,
      hash: row.display_hash,
      clip_sha256: row.clip_sha256,
      prev_sha256: row.prev_sha256,
      block_sha256: row.block_sha256,
      sealed_at: row.sealed_at,
      alert_id: row.alert_id,
    };
  },

  /** The Merkle root for one UTC day of sealed blocks, recomputed and stored on request. */
  getDailyRoot(day) {
    const blocks = repos.evidence.forDay(day);
    if (!blocks.length) {
      const stored = repos.dailyRoots.get(day);
      if (!stored) throw notFound('No sealed evidence for that day');
      return stored;
    }
    return repos.dailyRoots.upsert({
      day,
      merkle_root: merkleRoot(blocks.map((b) => b.block_sha256)),
      leaf_count: blocks.length,
      first_n: blocks[0].n,
      last_n: blocks[blocks.length - 1].n,
      computed_at: new Date().toISOString(),
    });
  },
});

export default createEvidenceService;
