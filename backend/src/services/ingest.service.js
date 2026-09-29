import { blockDigest } from '../repositories/evidence.repository.js';
import { badRequest, conflict, notFound } from '../utils/httpError.js';
import { seedFrom, shortHash } from '../utils/crypto.js';
import { dayOf, evidenceLabel, hhmmss } from '../utils/time.js';
import { validateEvent, validateUpdate } from './ingest.schema.js';

/**
 * The edge -> backend boundary (docs/ARCHITECTURE_V2.md section 4).
 *
 * `mapEventToAlert` is the only function in the backend that turns an event into an alert.
 * Everything else here is bookkeeping around it: idempotency, enrichment of an alert the same
 * track already raised, later-stage updates (the VLM reason patch, sealing) and publication.
 */

const STAGE = { provisional: 'prov', confirmed: 'conf', sealed: 'sealed' };
const STAGE_RANK = { prov: 0, conf: 1, sealed: 2 };
const PENDING_HASH = 'pending…';

const CLASS_LABEL = {
  person: 'Person',
  truck: 'Truck',
  car: 'Car',
  two_wheeler: 'Two-wheeler',
  cart: 'Cart',
};

/** Non-empty by construction: AlertQueue calls reason.slice(), so null would blank the queue. */
export const templateReason = ({ className, cameraName, rules, confidence }) => {
  const subject = CLASS_LABEL[className] || 'Object';
  const rule = rules.length ? ` (${rules.join(' + ')})` : '';
  return `${subject} flagged on ${cameraName}${rule}; appearance and motion channels agree at ${confidence}% confidence.`;
};

/**
 * truewatch.event.v1 -> the FROZEN alert shape (+ additive keys when known).
 * Every frozen key is produced unconditionally; every additive key is omitted when unknown.
 */
export const mapEventToAlert = (event, camera) => {
  const scores = event.detection.channel_scores;
  const rules = event.rule?.fired ?? [];
  const frozen = {
    id: `A${event.event_id.replace(/-/g, '').slice(0, 16)}`,
    camera,
    stage: STAGE[event.stage],
    time: hhmmss(event.captured_at),
    reason:
      event.explanation?.text ??
      templateReason({
        className: event.detection.class,
        cameraName: camera.name,
        rules,
        confidence: event.confidence,
      }),
    confidence: event.confidence,
    plate: event.anpr?.text ?? null,
    hash: shortHash(event.evidence?.clip_sha256) ?? PENDING_HASH,
    seed: seedFrom(event.event_id),
    channels: {
      appearance: (scores.appearance ?? 0) >= event.detection.threshold,
      motion: (scores.motion ?? 0) >= event.detection.threshold,
    },
  };

  const additive = {};
  if (event.detection.track_id != null) additive.track_id = event.detection.track_id;
  if (event.detection.bbox != null) additive.bbox = event.detection.bbox;
  if (scores != null) additive.channel_scores = scores;
  if (event.detection.dori_band != null) additive.dori_band = event.detection.dori_band;
  if (rules.length) additive.rule_fired = rules.join(' + ');
  if (event.evidence?.evidence_id != null) additive.evidence_id = event.evidence.evidence_id;
  if (event.evidence?.clip_url != null) additive.clip_url = event.evidence.clip_url;
  if (event.evidence?.thumb_url != null) additive.thumb_url = event.evidence.thumb_url;

  return { ...frozen, ...additive };
};

const PATCH_KEYS = ['reason', 'reason_source', 'plate', 'rule_fired', 'hash', 'evidence_id', 'clip_url', 'thumb_url'];

export const createIngestService = ({ repos, bus, transaction, config }) => {
  const cameraFor = (event) => {
    const camera = repos.cameras.getWithPost(event.camera_id);
    if (!camera) throw badRequest('Unknown camera_id');
    if (camera.post_id && camera.post_id !== event.post_id) {
      throw badRequest('post_id does not match the camera_id');
    }
    return camera;
  };

  /** Stores the sealed block for an alert. Idempotent on evidence_id; a reused block number is refused. */
  const storeEvidence = (evidence, cameraId, alertId) => {
    const existing = repos.evidence.findByEvidenceId(evidence.evidence_id);
    if (existing) return existing;
    if (repos.evidence.findByN(evidence.block_n)) {
      throw conflict('Evidence block number is already in use');
    }
    const row = {
      n: evidence.block_n,
      evidence_id: evidence.evidence_id,
      camera_id: cameraId,
      time_label: evidenceLabel(evidence.sealed_at),
      display_hash: shortHash(evidence.clip_sha256, 10, 6),
      clip_sha256: evidence.clip_sha256,
      prev_sha256: evidence.prev_sha256,
      clip_path: evidence.clip_url || `${evidence.evidence_id}.mp4`,
      sealed_at: evidence.sealed_at,
      alert_id: alertId,
    };
    row.block_sha256 = blockDigest(row);
    return repos.evidence.insert(row);
  };

  /**
   * Applies later-stage information to an existing alert row and reports what changed.
   * Rules: a stage only moves forward; a template reason never overwrites a model reason.
   */
  const applyChanges = (row, changes) => {
    const before = repos.alerts.findById(row.id);
    const patch = {};

    if (changes.explanation) {
      const { text, source } = changes.explanation;
      const downgrade = row.reason_source === 'vlm' && source === 'template';
      if (!downgrade) {
        patch.reason = text;
        patch.reason_source = source;
      }
    }

    if (changes.anpr) {
      patch.plate = changes.anpr.text;
      patch.plate_script = changes.anpr.script;
    }

    if (changes.rules?.length) {
      const merged = [...new Set([...(row.rule_fired ? row.rule_fired.split(' + ') : []), ...changes.rules])];
      const joined = merged.join(' + ');
      if (joined !== row.rule_fired) {
        patch.rule_fired = joined;
        if (!changes.explanation && row.reason_source === 'template') {
          patch.reason = templateReason({
            className: changes.className,
            cameraName: before.camera.name,
            rules: merged,
            confidence: row.confidence,
          });
        }
      }
    }

    let evidenceId = row.evidence_id;
    if (changes.evidence) {
      const block = storeEvidence(changes.evidence, row.camera_id, row.id);
      evidenceId = block.evidence_id;
      patch.evidence_id = block.evidence_id;
      patch.hash = shortHash(block.clip_sha256);
      if (changes.evidence.clip_url) patch.clip_url = changes.evidence.clip_url;
      if (changes.evidence.thumb_url) patch.thumb_url = changes.evidence.thumb_url;
    }

    if (changes.stage) {
      const wanted = STAGE[changes.stage];
      const sealable = wanted !== 'sealed' || evidenceId != null;
      if (sealable && STAGE_RANK[wanted] > STAGE_RANK[row.stage]) patch.stage = wanted;
    }

    const after = repos.alerts.patch(row.id, patch);
    const changed = {};
    for (const key of PATCH_KEYS) {
      if (JSON.stringify(after[key]) !== JSON.stringify(before[key])) changed[key] = after[key];
    }
    if (after.plate !== before.plate) changed.plate = after.plate;

    return {
      alert: after,
      patch: changed,
      stageChanged: after.stage !== before.stage,
      previousStage: before.stage,
    };
  };

  const publishChanges = (result) => {
    if (Object.keys(result.patch).length) {
      bus.publish('alert.updated', { id: result.alert.id, patch: result.patch, alert: result.alert });
    }
    if (result.stageChanged) {
      bus.publish('alert.stage', {
        id: result.alert.id,
        stage: result.alert.stage,
        previous: result.previousStage,
        alert: result.alert,
      });
    }
  };

  return {
    /** Replay controls (threat 3): a stale timestamp is 401, a reused nonce is 409. */
    checkFreshness({ timestamp, nonce, now = Date.now() }) {
      if (config.ingestRequireReplayHeaders && (!timestamp || !nonce)) {
        return { ok: false, status: 401, error: 'Timestamp and nonce headers are required' };
      }
      if (timestamp) {
        const sent = Date.parse(timestamp);
        if (Number.isNaN(sent) || Math.abs(now - sent) > config.ingestMaxSkewSeconds * 1000) {
          return { ok: false, status: 401, error: 'Request timestamp outside the accepted window' };
        }
      }
      if (nonce) {
        if (!/^[A-Za-z0-9_-]{8,64}$/.test(nonce)) {
          return { ok: false, status: 401, error: 'Malformed nonce' };
        }
        repos.nonces.prune(now - config.ingestNonceTtlSeconds * 1000);
        if (!repos.nonces.remember(nonce, now)) {
          return { ok: false, status: 409, error: 'Nonce already used', duplicate: true };
        }
      }
      return { ok: true };
    },

    /**
     * Accepts one truewatch.event.v1. Returns { duplicate, merged, alert }.
     * - same event_id again            -> duplicate, nothing written
     * - new event on a live track      -> enriches that track's alert instead of raising a second
     * - otherwise                      -> a new alert
     */
    ingestEvent(body) {
      const check = validateEvent(body);
      if (!check.ok) throw badRequest('Invalid event', { details: check.errors });
      const event = body;

      const knownAlertId = repos.eventLedger.alertIdFor(event.event_id);
      if (knownAlertId) return { duplicate: true, merged: false, alert: repos.alerts.findById(knownAlertId) };

      const camera = cameraFor(event);
      const publicCamera = repos.cameras.get(camera.id);

      const outcome = transaction(() => {
        const trackId = event.detection.track_id;
        const live =
          trackId == null ? null : repos.alerts.findRowLiveByTrack(camera.id, trackId, dayOf(event.captured_at));

        if (live) {
          const result = applyChanges(live, {
            stage: event.stage === 'provisional' ? null : event.stage,
            explanation: event.explanation,
            anpr: event.anpr,
            evidence: event.evidence,
            rules: event.rule?.fired,
            className: event.detection.class,
          });
          repos.eventLedger.record(event.event_id, live.id);
          return { merged: true, result };
        }

        const mapped = mapEventToAlert(event, publicCamera);
        const surrogateTrack =
          trackId == null
            ? null
            : repos.tracks.upsert({
                edgeTrackId: trackId,
                cameraId: camera.id,
                className: event.detection.class,
                capturedAt: event.captured_at,
                metrics: event.rule?.metrics,
              });

        const alert = repos.alerts.insert({
          id: mapped.id,
          event_id: event.event_id,
          camera_id: camera.id,
          stage: mapped.stage,
          captured_at: event.captured_at,
          time_label: mapped.time,
          reason: mapped.reason,
          reason_source: event.explanation?.source ?? 'template',
          confidence: mapped.confidence,
          plate: mapped.plate,
          plate_script: event.anpr?.script ?? null,
          hash: mapped.hash,
          seed: mapped.seed,
          ch_appearance: mapped.channels.appearance ? 1 : 0,
          ch_motion: mapped.channels.motion ? 1 : 0,
          track_id: surrogateTrack,
          bbox: mapped.bbox ? JSON.stringify(mapped.bbox) : null,
          score_appearance: mapped.channel_scores?.appearance ?? null,
          score_motion: mapped.channel_scores?.motion ?? null,
          rule_fired: mapped.rule_fired ?? null,
          dori_band: mapped.dori_band ?? null,
          ingest_source: event.source,
          origin: 'edge',
        });

        let created = alert;
        if (event.evidence) {
          const block = storeEvidence(event.evidence, camera.id, alert.id);
          created = repos.alerts.patch(alert.id, {
            evidence_id: block.evidence_id,
            clip_url: event.evidence.clip_url ?? null,
            thumb_url: event.evidence.thumb_url ?? null,
          });
        }
        repos.eventLedger.record(event.event_id, alert.id);
        return { merged: false, alert: created };
      });

      if (outcome.merged) {
        publishChanges(outcome.result);
        return { duplicate: false, merged: true, alert: outcome.result.alert };
      }
      bus.publish('alert.created', outcome.alert);
      return { duplicate: false, merged: false, alert: outcome.alert };
    },

    /**
     * Accepts one truewatch.update.v1 for an alert already ingested: the VLM reason patch
     * (~3 s after the alert), a stage advance, a late plate read, or the sealed evidence block.
     * Idempotent: replaying the same update changes nothing and publishes nothing.
     */
    applyUpdate(body) {
      const check = validateUpdate(body);
      if (!check.ok) throw badRequest('Invalid update', { details: check.errors });

      const alertId = repos.eventLedger.alertIdFor(body.event_id);
      if (!alertId) throw notFound('Unknown event_id');
      const row = repos.alerts.findRowById(alertId);
      if (row.stage === 'gone') throw conflict('Alert is no longer active');

      const result = transaction(() =>
        applyChanges(row, {
          stage: body.stage,
          explanation: body.explanation,
          anpr: body.anpr,
          evidence: body.evidence,
        }),
      );
      publishChanges(result);
      return {
        alert: result.alert,
        updated: Object.keys(result.patch).length > 0 || result.stageChanged,
      };
    },
  };
};

export default createIngestService;
