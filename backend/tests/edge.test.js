import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it } from 'node:test';

import { blockDigest } from '../src/repositories/evidence.repository.js';
import { ZERO_HASH, sha256 } from '../src/utils/crypto.js';
import { sampleEvent, startApp, startFakeEdge, tempDbPath, waitFor } from './helpers.js';
import { SHAPES, expectShape } from './shape.js';

const CLOSED_PORT_URL = 'http://127.0.0.1:1';

describe('edge-backed search', () => {
  const edgeResult = {
    query: 'white pickup',
    tookMs: 12.5,
    windowScanned: '2 weeks',
    results: [
      { id: 0, camera: { id: 'RXL-01', name: 'BOP Raxaul — Pillar 42', sector: 'Raxaul', ir: false, road: true }, time: '14 AUG · 02:41', chips: ['white', 'pickup'], score: 91 },
    ],
  };

  it('proxies to the edge and returns its answer unchanged', async () => {
    const edge = await startFakeEdge({ 'POST /api/search': (body) => ({ body: { ...edgeResult, query: body.query } }) });
    const t = await startApp({ EDGE_BASE_URL: edge.url });
    try {
      const res = await t.request('POST', '/api/search', { body: { query: 'white pickup' } });
      assert.equal(res.status, 200);
      expectShape(res.body, { data: SHAPES.search });
      assert.deepEqual(res.body.data, edgeResult);
      assert.deepEqual(edge.seen[0].body, { query: 'white pickup' });
    } finally {
      await t.close();
      await edge.close();
    }
  });

  it('falls back to the demo results when the edge is down, slow, off-contract or has nothing indexed', async () => {
    const cases = {
      down: { url: CLOSED_PORT_URL },
      offContract: { routes: { 'POST /api/search': () => ({ body: { nope: true } }) } },
      error: { routes: { 'POST /api/search': () => ({ status: 500, body: {} }) } },
      empty: { routes: { 'POST /api/search': (b) => ({ body: { query: b.query, tookMs: 1, windowScanned: 'no footage indexed', results: [] } }) } },
    };
    for (const [name, spec] of Object.entries(cases)) {
      const edge = spec.routes ? await startFakeEdge(spec.routes) : null;
      const t = await startApp({ EDGE_BASE_URL: edge?.url ?? spec.url });
      try {
        const res = await t.request('POST', '/api/search', { body: { query: 'anything' } });
        assert.equal(res.status, 200, name);
        expectShape(res.body, { data: SHAPES.search });
        assert.equal(res.body.data.results.length, 6, `${name}: demo results`);
      } finally {
        await t.close();
        await edge?.close();
      }
    }
  });
});

describe('dismiss feedback to the edge', () => {
  it('posts the dismissal to the edge without making the response wait for it', async () => {
    const edge = await startFakeEdge({ 'POST /pipeline/feedback': () => ({ body: { ok: true } }) });
    const t = await startApp({ EDGE_BASE_URL: edge.url });
    try {
      const id = (await t.request('POST', '/api/alerts/simulate')).body.data.id;
      const res = await t.request('POST', `/api/alerts/${id}/dismiss`);
      assert.equal(res.status, 200);
      assert.equal(res.body.data.stage, 'gone');
      await waitFor(() => edge.seen.length === 1, { message: 'feedback call' });
      assert.equal(edge.seen[0].body.action, 'dismiss');
      assert.equal(edge.seen[0].body.alert_id, id);
      assert.equal(edge.seen[0].body.camera_id, res.body.data.camera.id);
    } finally {
      await t.close();
      await edge.close();
    }
  });

  it('is non-fatal when the edge is absent', async () => {
    const t = await startApp({ EDGE_BASE_URL: CLOSED_PORT_URL, EDGE_TIMEOUT_MS: '200' });
    try {
      const id = (await t.request('POST', '/api/alerts/simulate')).body.data.id;
      const res = await t.request('POST', `/api/alerts/${id}/dismiss`);
      assert.equal(res.status, 200);
      assert.equal(res.body.data.stage, 'gone');
    } finally {
      await t.close();
    }
  });

  it('is non-fatal when the edge hangs, bounded by the timeout', async () => {
    const edge = await startFakeEdge({ 'POST /pipeline/feedback': () => ({ hang: true }) });
    const t = await startApp({ EDGE_BASE_URL: edge.url, EDGE_TIMEOUT_MS: '150' });
    try {
      const id = (await t.request('POST', '/api/alerts/simulate')).body.data.id;
      const started = Date.now();
      const res = await t.request('POST', `/api/alerts/${id}/dismiss`);
      assert.equal(res.status, 200);
      assert.ok(Date.now() - started < 1000, 'the response does not wait for the edge');
      await waitFor(() => edge.seen.length === 1, { message: 'feedback attempted' });
    } finally {
      await t.close();
      await edge.close();
    }
  });
});

describe('evidence verification', () => {
  it('uses the edge verifier when it answers in shape', async () => {
    const answer = { intact: false, verified: 4, total: 5, blocks: [{ n: 1041, ok: true }, { n: 1042, ok: false }] };
    const edge = await startFakeEdge({ 'POST /evidence/verify': () => ({ body: answer }) });
    const t = await startApp({ EDGE_BASE_URL: edge.url });
    try {
      const res = await t.request('POST', '/api/evidence/verify');
      assert.deepEqual(res.body.data, answer);
    } finally {
      await t.close();
      await edge.close();
    }
  });

  it('falls back to the local walk when the edge is down or answers off-contract', async () => {
    const edge = await startFakeEdge({ 'POST /evidence/verify': () => ({ body: { ok: 'maybe' } }) });
    for (const url of [edge.url, CLOSED_PORT_URL]) {
      const t = await startApp({ EDGE_BASE_URL: url });
      try {
        const res = await t.request('POST', '/api/evidence/verify');
        assert.equal(res.status, 200);
        expectShape(res.body, { data: SHAPES.verify });
        assert.equal(res.body.data.intact, true);
        assert.equal(res.body.data.total, 5);
      } finally {
        await t.close();
      }
    }
    await edge.close();
  });

  it('the local walk names a block whose stored hash was altered', async () => {
    const t = await startApp();
    try {
      t.container.db.prepare('UPDATE evidence_blocks SET clip_sha256 = ? WHERE n = 1043').run(sha256('someone edited this'));
      const res = await t.request('POST', '/api/evidence/verify');
      assert.equal(res.body.data.intact, false);
      assert.equal(res.body.data.total, 5);
      assert.equal(res.body.data.verified, 4);
      assert.deepEqual(res.body.data.blocks.filter((b) => !b.ok).map((b) => b.n), [1043]);
    } finally {
      await t.close();
    }
  });

  it('the local walk catches a rewritten link even when the block hash was recomputed', async () => {
    const t = await startApp();
    try {
      const row = t.container.db.prepare('SELECT * FROM evidence_blocks WHERE n = 1044').get();
      const forged = { ...row, prev_sha256: sha256('a different history') };
      t.container.db
        .prepare('UPDATE evidence_blocks SET prev_sha256 = ?, block_sha256 = ? WHERE n = 1044')
        .run(forged.prev_sha256, blockDigest(forged));
      const res = await t.request('POST', '/api/evidence/verify');
      assert.equal(res.body.data.intact, false);
      assert.ok(res.body.data.blocks.some((b) => !b.ok));
    } finally {
      await t.close();
    }
  });

  it('re-hashes stored clips, so a substituted clip fails although the chain links are intact', async () => {
    const store = tempDbPath();
    try {
      const t = await startApp({ EVIDENCE_STORE_PATH: store.dir });
      try {
        const clipPath = path.join(store.dir, 'ev_file_1.mp4');
        fs.writeFileSync(clipPath, 'genuine footage bytes');
        const tip = t.container.repos.evidence.tip();
        const row = {
          n: tip.n + 1,
          evidence_id: 'ev_file_1',
          camera_id: 'RXL-01',
          time_label: '29 SEP · 12:00',
          display_hash: 'abcdef0123…456789',
          clip_sha256: sha256('genuine footage bytes'),
          prev_sha256: tip.block_sha256,
          clip_path: 'ev_file_1.mp4',
          sealed_at: new Date().toISOString(),
        };
        row.block_sha256 = blockDigest(row);
        t.container.repos.evidence.insert(row);

        assert.equal((await t.request('POST', '/api/evidence/verify')).body.data.intact, true);
        fs.writeFileSync(clipPath, 'substituted footage');
        const tampered = await t.request('POST', '/api/evidence/verify');
        assert.equal(tampered.body.data.intact, false);
        assert.deepEqual(tampered.body.data.blocks.filter((b) => !b.ok).map((b) => b.n), [row.n]);
      } finally {
        await t.close();
      }
    } finally {
      store.cleanup();
    }
  });

  it('ignores a clip_path that escapes the store', async () => {
    const store = tempDbPath();
    try {
      const t = await startApp({ EVIDENCE_STORE_PATH: store.dir });
      try {
        const tip = t.container.repos.evidence.tip();
        const row = {
          n: tip.n + 1,
          evidence_id: 'ev_escape',
          camera_id: 'RXL-01',
          time_label: '29 SEP · 12:00',
          display_hash: 'abcdef0123…456789',
          clip_sha256: sha256('x'),
          prev_sha256: tip.block_sha256,
          clip_path: '../../../../etc/hosts',
          sealed_at: new Date().toISOString(),
        };
        row.block_sha256 = blockDigest(row);
        t.container.repos.evidence.insert(row);
        const res = await t.request('POST', '/api/evidence/verify');
        assert.equal(res.status, 200);
        assert.equal(res.body.data.intact, true, 'not checked against a file outside the store, and not a crash');
      } finally {
        await t.close();
      }
    } finally {
      store.cleanup();
    }
  });

  it('exposes one block and a daily Merkle root without leaking the stored clip path', async () => {
    const t = await startApp();
    try {
      const block = await t.request('GET', '/api/evidence/ev_seed_1042');
      assert.equal(block.status, 200);
      assert.equal(block.body.data.n, 1042);
      assert.equal(block.body.data.prev_sha256.length, 64);
      assert.doesNotMatch(block.text, /clip_path|seed\/ev_seed/);
      assert.equal((await t.request('GET', '/api/evidence/ev_missing')).status, 404);

      const root = await t.request('GET', '/api/evidence/root/2026-08-14');
      assert.equal(root.status, 200);
      assert.equal(root.body.data.leaf_count, 5);
      assert.match(root.body.data.merkle_root, /^[0-9a-f]{64}$/);
      assert.equal((await t.request('GET', '/api/evidence/root/2020-01-01')).status, 404);
      assert.equal((await t.request('GET', '/api/evidence/root/not-a-date')).status, 400);
    } finally {
      await t.close();
    }
  });

  it('the genesis link is 64 zeros', async () => {
    const t = await startApp();
    try {
      assert.equal(t.container.repos.evidence.allRows()[0].prev_sha256, ZERO_HASH);
    } finally {
      await t.close();
    }
  });
});

describe('simulate stays a working demo path', () => {
  it('works with no edge configured and feeds the same queue', async () => {
    const t = await startApp({ EDGE_BASE_URL: '' });
    try {
      const sim = await t.request('POST', '/api/alerts/simulate');
      assert.equal(sim.status, 201);
      await t.ingest(sampleEvent());
      const list = (await t.request('GET', '/api/alerts')).body.data;
      assert.ok(list.some((a) => a.id === sim.body.data.id));
      assert.equal(list.length, 2);
    } finally {
      await t.close();
    }
  });
});
