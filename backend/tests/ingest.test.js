import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { after, before, describe, it } from 'node:test';

import { mapEventToAlert } from '../src/services/ingest.service.js';
import { INGEST_KEY, sampleEvent, sampleUpdate, startApp } from './helpers.js';
import { expectAlert } from './shape.js';

const sha = (c) => c.repeat(64);

describe('POST /api/ingest/event', () => {
  let t;
  before(async () => {
    t = await startApp();
  });
  after(async () => {
    await t.close();
  });

  const alertRows = () => t.container.db.prepare("SELECT COUNT(*) FROM alerts WHERE origin = 'edge'").pluck().get();

  it('accepts a valid event, answers 201 { accepted, alert_id, duplicate:false } and stores the alert', async () => {
    const event = sampleEvent({ detection: { ...sampleEvent().detection, track_id: 101 } });
    const res = await t.ingest(event);
    assert.equal(res.status, 201);
    assert.deepEqual(Object.keys(res.body).sort(), ['accepted', 'alert_id', 'duplicate']);
    assert.equal(res.body.accepted, true);
    assert.equal(res.body.duplicate, false);

    const listed = (await t.request('GET', '/api/alerts')).body.data.find((a) => a.id === res.body.alert_id);
    assert.ok(listed, 'ingested alert appears in GET /api/alerts');
    expectAlert(listed);
    assert.equal(listed.stage, 'prov');
    assert.equal(listed.confidence, 93);
    assert.equal(listed.camera.id, 'RXL-01');
    assert.equal(listed.track_id, 101);
    assert.equal(listed.rule_fired, 'fence crossed');
    assert.deepEqual(listed.bbox, [0.41, 0.52, 0.06, 0.18]);
    assert.equal(listed.source, 'file');
    assert.equal(listed.reason_source, 'template');
  });

  it('is idempotent on event_id: replaying the same event creates exactly one alert', async () => {
    const event = sampleEvent({ detection: { ...sampleEvent().detection, track_id: 202 } });
    const before = alertRows();
    const first = await t.ingest(event);
    const second = await t.ingest(event);
    const third = await t.ingest(event);
    assert.equal(first.status, 201);
    assert.equal(second.status, 200);
    assert.equal(second.body.duplicate, true);
    assert.equal(third.body.duplicate, true);
    assert.equal(second.body.alert_id, first.body.alert_id);
    assert.equal(alertRows(), before + 1);
  });

  it('rejects a missing key, a wrong key and a key of the wrong length with 401', async () => {
    const event = sampleEvent();
    const before = alertRows();
    assert.equal((await t.request('POST', '/api/ingest/event', { body: event })).status, 401);
    assert.equal((await t.ingest(event, { 'X-Ingest-Key': 'wrong' })).status, 401);
    assert.equal((await t.ingest(event, { 'X-Ingest-Key': `${INGEST_KEY}x` })).status, 401);
    assert.equal(alertRows(), before, 'nothing is stored for an unauthenticated request');
    const res = await t.request('POST', '/api/ingest/event', { body: event });
    assert.deepEqual(res.body, { accepted: false, error: 'Invalid or missing ingest key' });
  });

  it('answers 401 before it looks at a malformed body', async () => {
    const res = await t.request('POST', '/api/ingest/event', { body: '{not json', raw: true });
    assert.equal(res.status, 401);
  });

  it('rejects an unknown top-level field with 400 and does not store anything', async () => {
    const before = alertRows();
    const res = await t.ingest({ ...sampleEvent(), sneaky: true });
    assert.equal(res.status, 400);
    assert.equal(res.body.error, 'Invalid event');
    assert.ok(res.body.details.some((d) => d.includes('sneaky') && d.includes('unknown field')));
    assert.equal(alertRows(), before);
  });

  it('rejects unknown fields at every nesting level', async () => {
    const base = sampleEvent();
    const variants = [
      { ...base, detection: { ...base.detection, extra: 1 } },
      { ...base, detection: { ...base.detection, channel_scores: { ...base.detection.channel_scores, third: 0.1 } } },
      { ...base, rule: { ...base.rule, extra: [] } },
      { ...base, rule: { ...base.rule, baseline: { ...base.rule.baseline, extra: 1 } } },
    ];
    for (const body of variants) {
      const res = await t.ingest(body);
      assert.equal(res.status, 400, JSON.stringify(res.body));
    }
  });

  it('rejects invalid values with 400 and never echoes the offending value', async () => {
    const base = sampleEvent();
    const bad = [
      { ...base, schema: 'truewatch.event.v2' },
      { ...base, event_id: 'not-a-uuid' },
      { ...base, camera_id: 'has spaces' },
      { ...base, captured_at: '2026-08-14 02:41:07' },
      { ...base, stage: 'sealed' },
      { ...base, confidence: 101 },
      { ...base, confidence: 92.5 },
      { ...base, source: 'ftp' },
      { ...base, detection: { ...base.detection, agreed: false } },
      { ...base, detection: { ...base.detection, class: 'dragon' } },
      { ...base, detection: { ...base.detection, bbox: [0, 0, 1] } },
      { ...base, detection: { ...base.detection, channel_scores: { appearance: 1.5, motion: 0.2 } } },
      { ...base, explanation: { text: 'x', source: 'vlm' } },
      { ...base, evidence: { evidence_id: 'ev_1', block_n: 1, clip_sha256: sha('a'), prev_sha256: sha('0'), sealed_at: base.captured_at } },
    ];
    for (const body of bad) {
      const res = await t.ingest(body);
      assert.equal(res.status, 400, JSON.stringify(body).slice(0, 120));
      assert.doesNotMatch(res.text, /dragon|ftp|not-a-uuid|has spaces/);
    }
    const missing = sampleEvent();
    delete missing.detection;
    assert.equal((await t.ingest(missing)).status, 400);
  });

  it('rejects an unknown camera and a post that does not own the camera', async () => {
    assert.equal((await t.ingest(sampleEvent({ camera_id: 'ZZZ-99' }))).status, 400);
    assert.equal((await t.ingest(sampleEvent({ post_id: 'JGB' }))).status, 400);
  });

  it('rejects malformed JSON and an oversized body once authenticated', async () => {
    const malformed = await t.request('POST', '/api/ingest/event', {
      body: '{"schema":',
      raw: true,
      headers: { 'X-Ingest-Key': INGEST_KEY },
    });
    assert.equal(malformed.status, 400);
    assert.deepEqual(malformed.body, { error: 'Malformed JSON body' });

    const huge = await t.request('POST', '/api/ingest/event', {
      body: JSON.stringify({ pad: 'x'.repeat(200_000) }),
      raw: true,
      headers: { 'X-Ingest-Key': INGEST_KEY },
    });
    assert.equal(huge.status, 413);
  });

  it('accepts the key in the X-Truewatch-Ingest-Key header as well', async () => {
    const res = await t.request('POST', '/api/ingest/event', {
      body: sampleEvent({ detection: { ...sampleEvent().detection, track_id: null } }),
      headers: { 'X-Truewatch-Ingest-Key': INGEST_KEY },
    });
    assert.equal(res.status, 201);
  });

  it('a later event on the same live track enriches the alert instead of raising a second one', async () => {
    const track = { ...sampleEvent().detection, track_id: 303 };
    const first = await t.ingest(sampleEvent({ detection: track, rule: null }));
    const before = alertRows();
    const second = await t.ingest(
      sampleEvent({
        detection: track,
        rule: { fired: ['loitering'], baseline: null, metrics: { dwell_s: 31 } },
        anpr: { text: 'बा १२ च ४५६७', script: 'devanagari', line_confidences: [0.98, 0.97] },
      }),
    );
    assert.equal(second.status, 201);
    assert.equal(second.body.alert_id, first.body.alert_id);
    assert.equal(second.body.merged, true);
    assert.equal(alertRows(), before);

    const alert = t.container.repos.alerts.findById(first.body.alert_id);
    assert.equal(alert.plate, 'बा १२ च ४५६७');
    assert.equal(alert.rule_fired, 'loitering');
  });
});

describe('replay headers', () => {
  let t;
  before(async () => {
    t = await startApp();
  });
  after(async () => {
    await t.close();
  });

  it('rejects a stale timestamp with 401', async () => {
    const stale = new Date(Date.now() - 10 * 60_000).toISOString();
    const res = await t.ingest(sampleEvent(), { 'X-Truewatch-Timestamp': stale, 'X-Truewatch-Nonce': randomUUID() });
    assert.equal(res.status, 401);
  });

  it('rejects a reused nonce with 409 { accepted:false, duplicate:true }', async () => {
    const nonce = randomUUID();
    const headers = { 'X-Truewatch-Timestamp': new Date().toISOString(), 'X-Truewatch-Nonce': nonce };
    const first = await t.ingest(sampleEvent(), headers);
    assert.equal(first.status, 201);
    const replay = await t.ingest(sampleEvent(), headers);
    assert.equal(replay.status, 409);
    assert.equal(replay.body.accepted, false);
    assert.equal(replay.body.duplicate, true);
  });

  it('can require both headers on every request', async () => {
    const strict = await startApp({ INGEST_REQUIRE_REPLAY_HEADERS: 'true' });
    try {
      assert.equal((await strict.ingest(sampleEvent())).status, 401);
      const ok = await strict.ingest(sampleEvent(), {
        'X-Truewatch-Timestamp': new Date().toISOString(),
        'X-Truewatch-Nonce': randomUUID(),
      });
      assert.equal(ok.status, 201);
    } finally {
      await strict.close();
    }
  });
});

describe('ingest rate limit', () => {
  it('answers 429 once the per-minute budget is spent', async () => {
    const t = await startApp({ INGEST_RATE_LIMIT_PER_MINUTE: '3' });
    try {
      const statuses = [];
      for (let i = 0; i < 5; i += 1) statuses.push((await t.ingest(sampleEvent())).status);
      assert.deepEqual(statuses.slice(0, 3), [201, 201, 201]);
      assert.deepEqual(statuses.slice(3), [429, 429]);
    } finally {
      await t.close();
    }
  });
});

describe('POST /api/ingest/update (reason patch, stage advance, sealing)', () => {
  let t;
  before(async () => {
    t = await startApp();
  });
  after(async () => {
    await t.close();
  });

  const create = async (track) => {
    const event = sampleEvent({ detection: { ...sampleEvent().detection, track_id: track } });
    const res = await t.ingest(event);
    return { event, id: res.body.alert_id };
  };

  it('patches the reason of an existing alert and reports updated:true', async () => {
    const { event, id } = await create(401);
    const res = await t.update(sampleUpdate(event.event_id));
    assert.equal(res.status, 200);
    assert.deepEqual(res.body, { accepted: true, alert_id: id, updated: true });
    const alert = t.container.repos.alerts.findById(id);
    assert.equal(alert.reason, 'Six people crossed the dry channel in single file.');
    assert.equal(alert.reason_source, 'vlm');
    assert.equal(alert.stage, 'prov');
  });

  it('is idempotent: replaying the same update changes nothing', async () => {
    const { event } = await create(402);
    const body = sampleUpdate(event.event_id);
    assert.equal((await t.update(body)).body.updated, true);
    assert.equal((await t.update(body)).body.updated, false);
  });

  it('never replaces a model reason with a template one', async () => {
    const { event, id } = await create(403);
    await t.update(sampleUpdate(event.event_id));
    await t.update(sampleUpdate(event.event_id, { explanation: { text: 'Template text.', source: 'template' } }));
    assert.equal(t.container.repos.alerts.findById(id).reason_source, 'vlm');
  });

  it('advances the stage forward only', async () => {
    const { event, id } = await create(404);
    await t.update(sampleUpdate(event.event_id, { stage: 'confirmed' }));
    assert.equal(t.container.repos.alerts.findById(id).stage, 'conf');
    await t.update({ schema: 'truewatch.update.v1', event_id: event.event_id, anpr: { text: 'BR 06 GA 4821', script: 'latin' } });
    assert.equal(t.container.repos.alerts.findById(id).plate, 'BR 06 GA 4821');
  });

  it('seals: stores the evidence block, links it to the alert and extends the chain', async () => {
    const { event, id } = await create(405);
    const tip = t.container.repos.evidence.tip();
    const res = await t.update({
      schema: 'truewatch.update.v1',
      event_id: event.event_id,
      stage: 'sealed',
      evidence: {
        evidence_id: 'ev_test_405',
        block_n: tip.n + 1,
        clip_sha256: sha('a'),
        prev_sha256: tip.block_sha256,
        clip_url: 'http://edge.local/clips/ev_test_405.mp4',
        thumb_url: 'http://edge.local/clips/ev_test_405.jpg',
        sealed_at: new Date().toISOString(),
      },
    });
    assert.equal(res.status, 200);
    const alert = t.container.repos.alerts.findById(id);
    assert.equal(alert.stage, 'sealed');
    assert.equal(alert.evidence_id, 'ev_test_405');
    assert.equal(alert.hash, `${sha('a').slice(0, 8)}…${sha('a').slice(-6)}`);
    assert.equal(alert.clip_url, 'http://edge.local/clips/ev_test_405.mp4');

    const verify = await t.request('POST', '/api/evidence/verify');
    assert.equal(verify.body.data.intact, true);
    assert.equal(verify.body.data.total, tip.n - 1041 + 2);
  });

  it('refuses sealed without evidence, unknown events and non-active alerts', async () => {
    const { event, id } = await create(406);
    assert.equal((await t.update({ schema: 'truewatch.update.v1', event_id: event.event_id, stage: 'sealed' })).status, 400);
    assert.equal((await t.update(sampleUpdate(randomUUID()))).status, 404);
    assert.equal((await t.update({ schema: 'truewatch.update.v1', event_id: event.event_id })).status, 400);
    assert.equal((await t.update({ ...sampleUpdate(event.event_id), extra: 1 })).status, 400);
    await t.request('POST', `/api/alerts/${id}/dismiss`);
    assert.equal((await t.update(sampleUpdate(event.event_id, { explanation: { text: 'late', source: 'vlm' } }))).status, 409);
  });

  it('requires the ingest key', async () => {
    const res = await t.request('POST', '/api/ingest/update', { body: sampleUpdate(randomUUID()) });
    assert.equal(res.status, 401);
  });
});

describe('mapEventToAlert', () => {
  const camera = { id: 'RXL-01', name: 'BOP Raxaul — Pillar 42', sector: 'Raxaul', ir: false, road: true };

  it('produces every frozen key with valid values for a minimal event', () => {
    const minimal = sampleEvent({
      detection: { ...sampleEvent().detection, track_id: null, bbox: null, dori_band: null },
      rule: null,
    });
    const alert = mapEventToAlert(minimal, camera);
    expectAlert(alert);
    assert.equal(alert.stage, 'prov');
    assert.equal(alert.plate, null);
    assert.ok(alert.reason.length > 0);
    assert.ok(Number.isFinite(alert.seed) && alert.seed >= 0 && alert.seed < 100);
    assert.equal(alert.camera, camera);
    assert.equal('track_id' in alert, false, 'unknown additive keys are omitted, never undefined');
    assert.equal('bbox' in alert, false);
    assert.equal('rule_fired' in alert, false);
    assert.equal(alert.hash, 'pending…');
  });

  it('derives channel booleans from the per-camera threshold', () => {
    const event = sampleEvent({
      detection: { ...sampleEvent().detection, channel_scores: { appearance: 0.9, motion: 0.3 }, threshold: 0.55 },
    });
    assert.deepEqual(mapEventToAlert(event, camera).channels, { appearance: true, motion: false });
  });

  it('maps stages, plate, explanation and evidence', () => {
    const event = sampleEvent({
      stage: 'sealed',
      explanation: { text: 'Loaded vehicle moving north.', source: 'vlm', model: 'm', latency_ms: 10 },
      anpr: { text: 'BR 06 GA 4821', script: 'latin' },
      evidence: {
        evidence_id: 'ev_1',
        block_n: 5,
        clip_sha256: sha('b'),
        prev_sha256: sha('0'),
        clip_url: 'http://x/y.mp4',
        thumb_url: 'http://x/y.jpg',
        sealed_at: new Date().toISOString(),
      },
    });
    const alert = mapEventToAlert(event, camera);
    assert.equal(alert.stage, 'sealed');
    assert.equal(mapEventToAlert({ ...event, stage: 'confirmed', evidence: null }, camera).stage, 'conf');
    assert.equal(alert.reason, 'Loaded vehicle moving north.');
    assert.equal(alert.plate, 'BR 06 GA 4821');
    assert.equal(alert.evidence_id, 'ev_1');
    assert.equal(alert.clip_url, 'http://x/y.mp4');
    assert.equal(alert.thumb_url, 'http://x/y.jpg');
    assert.equal(alert.hash, `${sha('b').slice(0, 8)}…${sha('b').slice(-6)}`);
  });

  it('is deterministic for a given event id', () => {
    const event = sampleEvent();
    assert.deepEqual(mapEventToAlert(event, camera), mapEventToAlert(event, camera));
  });
});
