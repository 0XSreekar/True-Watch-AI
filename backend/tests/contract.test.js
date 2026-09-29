import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { ANALYTICS, CAMERAS, POSTS, ROLES, UNITS } from '../src/data/mockData.js';
import { SEED_PASSWORD, USERS, startApp } from './helpers.js';
import { SHAPES, expectAlert, expectShape } from './shape.js';

/**
 * The frozen frontend contract. One test per endpoint, asserting the exact response envelope and
 * key set. If any of these fails, the console would break: change the implementation, not the test.
 */
describe('frozen contract', () => {
  let t;
  before(async () => {
    t = await startApp();
  });
  after(async () => {
    await t.close();
  });

  const get = (url) => t.request('GET', url);
  const post = (url, body) => t.request('POST', url, { body });

  describe('GET /api/health', () => {
    it('answers { status, service, time } and is not wrapped in data', async () => {
      const res = await get('/api/health');
      assert.equal(res.status, 200);
      expectShape(res.body, { status: 'string', service: 'string', time: 'string' });
      assert.equal(res.body.status, 'ok');
    });
  });

  describe('auth', () => {
    it('POST /api/auth/login answers { data: { ok, token, expiresIn, user } }', async () => {
      const res = await post('/api/auth/login', { officialId: USERS.operator, password: SEED_PASSWORD, remember: true });
      assert.equal(res.status, 200);
      expectShape(res.body, { data: { ok: 'boolean', token: 'string', expiresIn: 'number', user: SHAPES.user } });
      assert.equal(res.body.data.ok, true);
    });

    it('POST /api/auth/login failures are { error } with status 401', async () => {
      for (const body of [{}, { officialId: USERS.operator }, { officialId: USERS.operator, password: 'no' }, { officialId: USERS.operator, password: 'wrong-password' }]) {
        const res = await post('/api/auth/login', body);
        assert.equal(res.status, 401);
        expectShape(res.body, { error: 'string' });
      }
    });

    it('POST /api/auth/login keeps the two original validation messages', async () => {
      assert.equal((await post('/api/auth/login', {})).body.error, 'Official ID required');
      assert.equal((await post('/api/auth/login', { officialId: 'X' })).body.error, 'Password too short');
    });

    it('POST /api/auth/register answers 201 { data: { ok, pendingApproval, token, expiresIn, user } }', async () => {
      const res = await post('/api/auth/register', {
        name: 'Ct. Test Person',
        officialId: 'SSB/777001',
        unit: UNITS[1],
        email: 'test.person@example.org',
        password: 'a-long-enough-password',
        role: 'Operator',
      });
      assert.equal(res.status, 201);
      expectShape(res.body, {
        data: { ok: 'boolean', pendingApproval: 'boolean', token: 'string', expiresIn: 'number', user: SHAPES.user },
      });
      assert.equal(res.body.data.pendingApproval, true);
    });

    it('POST /api/auth/register failures are { error } with status 422', async () => {
      const missing = await post('/api/auth/register', { name: 'Only Name' });
      assert.equal(missing.status, 422);
      expectShape(missing.body, { error: 'string' });
      assert.match(missing.body.error, /^Missing: /);
    });

    it('GET /api/auth/units answers string[]', async () => {
      const res = await get('/api/auth/units');
      assert.equal(res.status, 200);
      expectShape(res.body, { data: { $array: 'string', min: 1 } });
      assert.deepEqual(res.body.data, UNITS);
    });

    it('GET /api/auth/roles answers [{ k, dd }] with the three roles', async () => {
      const res = await get('/api/auth/roles');
      assert.equal(res.status, 200);
      expectShape(res.body, { data: { $array: SHAPES.role, min: 3 } });
      assert.deepEqual(res.body.data, ROLES);
    });
  });

  describe('cameras', () => {
    it('GET /api/cameras answers the camera rows exactly as the fixtures define them', async () => {
      const res = await get('/api/cameras');
      assert.equal(res.status, 200);
      expectShape(res.body, { data: { $array: SHAPES.camera, min: 1 } });
      assert.deepEqual(res.body.data, CAMERAS);
      assert.deepEqual(Object.keys(res.body.data[0]), Object.keys(CAMERAS[0]));
    });

    it('GET /api/cameras never exposes internal columns', async () => {
      const res = await get('/api/cameras');
      assert.doesNotMatch(res.text, /rtsp|post_id|dori/);
    });

    it('GET /api/cameras?sector= filters by sector', async () => {
      const res = await get('/api/cameras?sector=Raxaul');
      assert.deepEqual(res.body.data, CAMERAS.filter((c) => c.sector === 'Raxaul'));
      const all = await get('/api/cameras?sector=all');
      assert.equal(all.body.data.length, CAMERAS.length);
    });

    it('GET /api/cameras/posts answers post rows in the fixture shape', async () => {
      const res = await get('/api/cameras/posts');
      assert.equal(res.status, 200);
      expectShape(res.body, { data: { $array: SHAPES.post, min: 1 } });
      assert.deepEqual(Object.keys(res.body.data[0]), Object.keys(POSTS[0]));
      assert.deepEqual(res.body.data.map((p) => p.id), POSTS.map((p) => p.id));
    });

    it('GET /api/cameras/sectors answers string[]', async () => {
      const res = await get('/api/cameras/sectors');
      expectShape(res.body, { data: { $array: 'string', min: 1 } });
      assert.deepEqual(res.body.data, [...new Set(CAMERAS.map((c) => c.sector))]);
    });

    it('GET /api/cameras/:id answers one camera, or 404 { error }', async () => {
      const ok = await get('/api/cameras/RXL-01');
      assert.equal(ok.status, 200);
      expectShape(ok.body, { data: SHAPES.camera });
      const missing = await get('/api/cameras/NOPE-99');
      assert.equal(missing.status, 404);
      expectShape(missing.body, { error: 'string' });
    });
  });

  describe('alerts', () => {
    it('GET /api/alerts answers { data: [alert], budget } (budget is a sibling of data)', async () => {
      await post('/api/alerts/simulate');
      const res = await get('/api/alerts');
      assert.equal(res.status, 200);
      assert.deepEqual(Object.keys(res.body).sort(), ['budget', 'data']);
      expectShape(res.body.budget, SHAPES.budget);
      assert.ok(res.body.data.length >= 1);
      res.body.data.forEach(expectAlert);
    });

    it('GET /api/alerts/budget answers { used, total, near }', async () => {
      const res = await get('/api/alerts/budget');
      expectShape(res.body, { data: SHAPES.budget });
      assert.equal(res.body.data.total, 12);
    });

    it('POST /api/alerts/simulate answers 201 with exactly the frozen alert keys plus a sibling budget', async () => {
      const res = await post('/api/alerts/simulate');
      assert.equal(res.status, 201);
      assert.deepEqual(Object.keys(res.body).sort(), ['budget', 'data']);
      expectShape(res.body.data, SHAPES.alert);
      expectAlert(res.body.data);
      assert.equal(res.body.data.stage, 'prov');
      assert.deepEqual(res.body.data.channels, { appearance: true, motion: true });
      expectShape(res.body.budget, SHAPES.budget);
    });

    it('POST /api/alerts/:id/act answers the alert with stage "sealed", or 404 { error }', async () => {
      const { data } = (await post('/api/alerts/simulate')).body;
      const res = await post(`/api/alerts/${data.id}/act`);
      assert.equal(res.status, 200);
      assert.deepEqual(Object.keys(res.body), ['data']);
      expectAlert(res.body.data);
      assert.equal(res.body.data.stage, 'sealed');
      assert.equal(res.body.data.id, data.id);
      const missing = await post('/api/alerts/does-not-exist/act');
      assert.equal(missing.status, 404);
      expectShape(missing.body, { error: 'string' });
    });

    it('POST /api/alerts/:id/dismiss answers the alert with stage "gone", removes it, then 404s', async () => {
      const { data } = (await post('/api/alerts/simulate')).body;
      const res = await post(`/api/alerts/${data.id}/dismiss`);
      assert.equal(res.status, 200);
      expectAlert(res.body.data);
      assert.equal(res.body.data.stage, 'gone');
      const list = (await get('/api/alerts')).body.data;
      assert.ok(!list.some((a) => a.id === data.id), 'dismissed alert must leave the queue');
      const again = await post(`/api/alerts/${data.id}/dismiss`);
      assert.equal(again.status, 404);
      expectShape(again.body, { error: 'string' });
    });
  });

  describe('search', () => {
    it('GET /api/search/placeholders answers string[]', async () => {
      const res = await get('/api/search/placeholders');
      expectShape(res.body, { data: { $array: 'string', min: 1 } });
    });

    it('POST /api/search answers { query, tookMs, windowScanned, results:[{ id, camera, time, chips, score }] }', async () => {
      const res = await post('/api/search', { query: 'white pickup near pillar 42' });
      assert.equal(res.status, 200);
      expectShape(res.body, { data: SHAPES.search });
      assert.equal(res.body.data.query, 'white pickup near pillar 42');
      assert.ok(res.body.data.results.length > 0);
    });
  });

  describe('evidence', () => {
    it('GET /api/evidence/chain answers [{ n, hash, camera, time }]', async () => {
      const res = await get('/api/evidence/chain');
      assert.equal(res.status, 200);
      expectShape(res.body, { data: { $array: SHAPES.block, min: 5 } });
      const ns = res.body.data.map((b) => b.n);
      assert.deepEqual(ns, [...ns].sort((a, b) => a - b), 'blocks are ascending');
    });

    it('POST /api/evidence/verify answers { intact, verified, total, blocks:[{ n, ok }] }', async () => {
      const res = await post('/api/evidence/verify');
      assert.equal(res.status, 200);
      expectShape(res.body, { data: SHAPES.verify });
      assert.equal(res.body.data.intact, true);
      assert.equal(res.body.data.verified, res.body.data.total);
    });
  });

  describe('analytics', () => {
    it('GET /api/analytics answers { camBars, dayNight, agreement, traffic } in the fixture shapes', async () => {
      const res = await get('/api/analytics');
      assert.equal(res.status, 200);
      expectShape(res.body, { data: SHAPES.analytics });
      assert.deepEqual(res.body.data.camBars.map((c) => c.id), ANALYTICS.camBars.map((c) => c.id));
      assert.deepEqual(res.body.data.dayNight.map((d) => d.k), ANALYTICS.dayNight.map((d) => d.k));
      assert.deepEqual(res.body.data.dayNight.map((d) => d.c), ANALYTICS.dayNight.map((d) => d.c));
      assert.equal(res.body.data.dayNight.reduce((sum, d) => sum + d.v, 0), 100);
      assert.equal(res.body.data.agreement.length, ANALYTICS.agreement.length);
      assert.equal(res.body.data.traffic.length, ANALYTICS.traffic.length);
    });

    it('GET /api/analytics/traffic answers { sample: number }', async () => {
      const res = await get('/api/analytics/traffic');
      expectShape(res.body, { data: { sample: 'number' } });
    });
  });
});
