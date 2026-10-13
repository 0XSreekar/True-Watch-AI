import { randomUUID } from 'node:crypto';

import { REASONS, PLATES, SHIFT_ALARM_BUDGET } from '../data/mockData.js';
import { blockDigest } from '../repositories/evidence.repository.js';
import { ZERO_HASH, sha256, shortHash } from '../utils/crypto.js';
import { evidenceLabel } from '../utils/time.js';
import { chance, hex, pick, stamp } from '../utils/random.js';

/**
 * Alert lifecycle: provisional (~30 ms, both channels agree) -> confirmed (~3 s, the VLM judge
 * writes the reason) -> sealed (hash-chained evidence). Live alerts arrive through
 * POST /api/ingest/event; this service reads and advances them.
 */
export const createAlertsService = ({ repos, bus, edge, audit, transaction, config }) => {
  const publishStage = (alert, previous) =>
    bus.publish('alert.stage', { id: alert.id, stage: alert.stage, previous, alert });

  /** Alerts raised in the current shift, capped at the budget. Shifts are aligned to the epoch. */
  const getBudget = () => {
    const span = config.shiftHours * 3600 * 1000;
    const shiftStart = new Date(Math.floor(Date.now() / span) * span).toISOString();
    const used = Math.min(SHIFT_ALARM_BUDGET, repos.alerts.countSince(shiftStart));
    return { used, total: SHIFT_ALARM_BUDGET, near: used >= SHIFT_ALARM_BUDGET - 3 };
  };

  /**
   * DEMO PATH. Produces one plausible alert from the mock fixtures so the console can be shown
   * with the edge service asleep. Rows are tagged origin 'simulate' and carry only the frozen keys.
   * The real path is POST /api/ingest/event.
   */
  const simulateAlert = () => {
    const camera = pick(repos.cameras.list());
    const now = new Date();
    const alert = repos.alerts.insert({
      id: `A${Date.now()}${Math.floor(Math.random() * 99)}`,
      event_id: `sim-${randomUUID()}`,
      camera_id: camera.id,
      stage: 'prov',
      captured_at: now.toISOString(),
      time_label: stamp(now),
      reason: pick(REASONS),
      confidence: 88 + Math.floor(Math.random() * 10),
      plate: chance(0.45) ? pick(PLATES) : null,
      hash: `${hex(8)}…${hex(6)}`,
      seed: Math.random() * 100,
      ch_appearance: 1,
      ch_motion: 1,
      origin: 'simulate',
    });
    bus.publish('alert.created', alert);
    return alert;
  };

  /**
   * Seals a demo alert into the chain by digesting its record. There is no clip behind a simulated
   * alert, so the block's clip_path is `record://<id>` and the digest is over the alert record;
   * real clips are sealed by the edge and arrive through the update endpoint.
   */
  const sealRecord = (row, alert) => {
    const tip = repos.evidence.tip();
    const sealedAt = new Date().toISOString();
    const clip = sha256('alert-record', JSON.stringify(alert));
    const block = {
      n: tip ? tip.n + 1 : 1,
      evidence_id: `ev_${row.id}`,
      camera_id: row.camera_id,
      time_label: evidenceLabel(sealedAt),
      display_hash: shortHash(clip, 10, 6),
      clip_sha256: clip,
      prev_sha256: tip ? tip.block_sha256 : ZERO_HASH,
      clip_path: `record://${row.id}`,
      sealed_at: sealedAt,
      alert_id: row.id,
    };
    block.block_sha256 = blockDigest(block);
    return repos.evidence.insert(block);
  };

  return {
    listAlerts: ({ limit } = {}) => repos.alerts.listLive(limit),
    getBudget,
    simulateAlert,

    /** Operator acted: the alert is sealed. Returns null when the alert does not exist or is gone. */
    actOnAlert(id, actor) {
      const row = repos.alerts.findRowById(id);
      if (!row || row.stage === 'gone') return null;

      const alert = transaction(() => {
        const patch = { stage: 'sealed' };
        if (!row.evidence_id && row.origin === 'simulate') {
          const block = sealRecord(row, repos.alerts.findById(id));
          patch.evidence_id = block.evidence_id;
          patch.hash = shortHash(block.clip_sha256);
        }
        const sealed = repos.alerts.patch(id, patch);
        audit.record(actor, 'alert.act', id, { camera: row.camera_id, previous_stage: row.stage });
        return sealed;
      });

      if (row.stage !== 'sealed') publishStage(alert, row.stage);
      return alert;
    },

    /** Operator dismissed: the alert leaves the queue and the edge is told (best effort). */
    dismissAlert(id, actor) {
      const row = repos.alerts.findRowById(id);
      if (!row || row.stage === 'gone') return null;

      const alert = transaction(() => {
        const gone = repos.alerts.patch(id, { stage: 'gone' });
        audit.record(actor, 'alert.dismiss', id, { camera: row.camera_id, previous_stage: row.stage });
        return gone;
      });

      publishStage(alert, row.stage);
      // Deliberately not awaited: the response must not wait on, or fail because of, the edge.
      void edge.sendBaselineFeedback({
        action: 'dismiss',
        alert_id: id,
        event_id: row.event_id,
        camera_id: row.camera_id,
        track_id: alert.track_id ?? null,
        origin: row.origin,
      });
      return alert;
    },
  };
};

export default createAlertsService;
