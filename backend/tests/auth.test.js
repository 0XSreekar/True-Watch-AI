import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { REFRESH_COOKIE } from '../src/controllers/auth.controller.js';
import { SEED_PASSWORD, USERS, startApp } from './helpers.js';

const auditActions = (t) =>
  t.container.repos.audit.recent(200).map((e) => ({ action: e.action, actor: e.actor_id, subject: e.subject }));

describe('authentication', () => {
  let t;
  before(async () => {
    t = await startApp();
  });
  after(async () => {
    await t.close();
  });

  it('stores only bcrypt hashes, never a plaintext password', () => {
    const rows = t.container.db.prepare('SELECT id, password_hash FROM users').all();
    assert.ok(rows.length >= 3);
    for (const row of rows) {
      assert.match(row.password_hash, /^\$2[aby]\$\d\d\$/);
      assert.ok(!row.password_hash.includes(SEED_PASSWORD));
    }
  });

  it('login issues a JWT access token and an httpOnly refresh cookie, and never puts the refresh token in the body', async () => {
    const res = await t.request('POST', '/api/auth/login', { body: { officialId: USERS.operator, password: SEED_PASSWORD } });
    assert.equal(res.status, 200);
    assert.match(res.body.data.token, /^[\w-]+\.[\w-]+\.[\w-]+$/);
    assert.equal(res.body.data.expiresIn, 900, 'short access-token lifetime');
    assert.equal(res.body.data.user.role, 'Operator');
    const cookie = res.cookies.find((c) => c.startsWith(`${REFRESH_COOKIE}=`));
    assert.ok(cookie, 'refresh cookie is set');
    assert.match(cookie, /HttpOnly/i);
    assert.match(cookie, /SameSite=Strict/i);
    assert.match(cookie, /Path=\/api\/auth/);
    const secret = cookie.split(';')[0].split('=')[1];
    assert.ok(!res.text.includes(secret), 'refresh token must not appear in the response body');
  });

  it('rejects a wrong password and an unknown user with the same message', async () => {
    const wrong = await t.request('POST', '/api/auth/login', { body: { officialId: USERS.operator, password: 'not-the-password' } });
    const unknown = await t.request('POST', '/api/auth/login', { body: { officialId: 'SSB/000000', password: 'not-the-password' } });
    assert.equal(wrong.status, 401);
    assert.equal(unknown.status, 401);
    assert.deepEqual(wrong.body, unknown.body);
  });

  it('refresh rotates the token: the new cookie works once and the old one is dead', async () => {
    const { cookie } = await t.login('operator');
    const first = await t.request('POST', '/api/auth/refresh', { headers: { Cookie: cookie } });
    assert.equal(first.status, 200);
    assert.match(first.body.data.token, /^[\w-]+\.[\w-]+\.[\w-]+$/);
    assert.equal(first.body.data.user.id, USERS.operator);
    const rotated = first.cookies[0].split(';')[0];
    assert.notEqual(rotated, cookie);

    const reuse = await t.request('POST', '/api/auth/refresh', { headers: { Cookie: cookie } });
    assert.equal(reuse.status, 401, 'a used refresh token is rejected');
    const afterTheft = await t.request('POST', '/api/auth/refresh', { headers: { Cookie: rotated } });
    assert.equal(afterTheft.status, 401, 'reuse revokes the whole session family');
  });

  it('refresh without a cookie, or from a foreign origin, is refused', async () => {
    assert.equal((await t.request('POST', '/api/auth/refresh')).status, 401);
    const { cookie } = await t.login('operator');
    const foreign = await t.request('POST', '/api/auth/refresh', { headers: { Cookie: cookie, Origin: 'https://evil.example' } });
    assert.equal(foreign.status, 403);
  });

  it('logout revokes the refresh token and clears the cookie', async () => {
    const { cookie, token } = await t.login('operator');
    const out = await t.request('POST', '/api/auth/logout', { headers: { Cookie: cookie }, token });
    assert.equal(out.status, 200);
    assert.deepEqual(out.body, { data: { ok: true } });
    assert.match(out.cookies[0], new RegExp(`${REFRESH_COOKIE}=;`));
    assert.equal((await t.request('POST', '/api/auth/refresh', { headers: { Cookie: cookie } })).status, 401);
  });

  it('expired and tampered access tokens are rejected', async () => {
    const short = await startApp({ JWT_EXPIRES_IN: '1s' });
    try {
      const { token } = await short.login('supervisor');
      assert.equal((await short.request('GET', '/api/auth/me', { token })).status, 200);
      await new Promise((r) => setTimeout(r, 2100));
      assert.equal((await short.request('GET', '/api/auth/me', { token })).status, 401);
    } finally {
      await short.close();
    }
    const { token } = await t.login('operator');
    const [h, p, s] = token.split('.');
    const forged = `${h}.${Buffer.from(JSON.stringify({ ...JSON.parse(Buffer.from(p, 'base64url')), role: 'Supervisor' })).toString('base64url')}.${s}`;
    assert.equal((await t.request('GET', '/api/auth/me', { token: forged })).status, 401);
    const none = `${Buffer.from('{"alg":"none","typ":"JWT"}').toString('base64url')}.${p}.`;
    assert.equal((await t.request('GET', '/api/auth/me', { token: none })).status, 401);
  });

  it('register creates a pending account whose requested role grants nothing until a supervisor approves it', async () => {
    const res = await t.request('POST', '/api/auth/register', {
      body: {
        name: 'Insp. New Supervisor',
        officialId: 'SSB/888001',
        unit: '20th Bn — Raxaul',
        email: 'new.sup@example.org',
        password: 'a-long-enough-password',
        role: 'Supervisor',
      },
    });
    assert.equal(res.status, 201);
    assert.equal(res.body.data.pendingApproval, true);
    const asNew = { token: res.body.data.token };
    const denied = await t.request('GET', '/api/audit', asNew);
    assert.equal(denied.status, 403);
    assert.match(denied.body.error, /pending approval/);

    const sup = await t.login('supervisor');
    assert.equal((await t.request('POST', '/api/auth/approve/SSB%2F888001', { token: sup.token })).status, 200);
    assert.equal((await t.request('GET', '/api/audit', asNew)).status, 200);

    const again = await t.request('POST', '/api/auth/register', {
      body: { name: 'Dup', officialId: 'SSB/888001', unit: '20th Bn — Raxaul', email: 'a@b.co', password: 'a-long-enough-password', role: 'Operator' },
    });
    assert.equal(again.status, 422);
  });

  it('register validates role, unit and password strength', async () => {
    const valid = { name: 'Ct. Valid Name', officialId: 'SSB/888002', unit: '20th Bn — Raxaul', email: 'v@b.co', password: 'a-long-enough-password', role: 'Operator' };
    for (const patch of [{ role: 'Commander' }, { unit: 'Nowhere Bn' }, { password: 'short' }, { email: 'nope' }, { officialId: '!!' }]) {
      const res = await t.request('POST', '/api/auth/register', { body: { ...valid, ...patch } });
      assert.equal(res.status, 422, JSON.stringify(patch));
    }
  });
});

describe('role gates and audit', () => {
  let t;
  let alertId;
  before(async () => {
    t = await startApp();
    alertId = (await t.request('POST', '/api/alerts/simulate')).body.data.id;
  });
  after(async () => {
    await t.close();
  });

  it('face unseal: anonymous 401, Operator 403, HQ Analyst 403, Supervisor 200 with an audit row', async () => {
    const url = `/api/faces/${alertId}/unseal`;
    const body = { reason: 'Identify suspect flagged at the culvert' };
    assert.equal((await t.request('POST', url, { body })).status, 401);
    const op = await t.login('operator');
    const denied = await t.request('POST', url, { body, token: op.token });
    assert.equal(denied.status, 403);
    assert.deepEqual(denied.body, { error: 'Requires role: Supervisor' });
    assert.equal((await t.request('POST', url, { body, token: (await t.login('analyst')).token })).status, 403);

    assert.ok(!auditActions(t).some((e) => e.action === 'face.unmask'), 'a denied attempt writes no unmask row');

    const sup = await t.login('supervisor');
    const ok = await t.request('POST', url, { body, token: sup.token });
    assert.equal(ok.status, 200);
    assert.equal(ok.body.data.unsealed, true);
    assert.equal(ok.body.data.alert_id, alertId);

    const row = t.container.repos.audit.recent(50).find((e) => e.action === 'face.unmask');
    assert.ok(row, 'audit row written');
    assert.equal(row.actor_id, USERS.supervisor);
    assert.equal(row.subject, alertId);
    assert.equal(row.detail.reason, body.reason);
  });

  it('face unseal needs a reason and an existing alert', async () => {
    const sup = await t.login('supervisor');
    assert.equal((await t.request('POST', `/api/faces/${alertId}/unseal`, { body: {}, token: sup.token })).status, 400);
    assert.equal((await t.request('POST', '/api/faces/A-nope/unseal', { body: { reason: 'valid reason' }, token: sup.token })).status, 404);
  });

  it('fence and hours edits are Supervisor-only, validated and audited', async () => {
    const polygon = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]];
    const op = await t.login('operator');
    const sup = await t.login('supervisor');
    assert.equal((await t.request('PUT', '/api/cameras/RXL-01/fence', { body: { polygon }, token: op.token })).status, 403);
    assert.equal((await t.request('PUT', '/api/cameras/RXL-01/hours', { body: { start: '06:00', end: '20:00' }, token: op.token })).status, 403);

    const fence = await t.request('PUT', '/api/cameras/RXL-01/fence', { body: { polygon }, token: sup.token });
    assert.equal(fence.status, 200);
    assert.deepEqual(fence.body.data.fence, polygon);
    const hours = await t.request('PUT', '/api/cameras/RXL-01/hours', { body: { start: '06:00', end: '20:30' }, token: sup.token });
    assert.deepEqual(hours.body.data.hours, { start: '06:00', end: '20:30' });
    assert.deepEqual(hours.body.data.fence, polygon, 'editing hours keeps the fence');

    const read = await t.request('GET', '/api/cameras/RXL-01/settings');
    assert.deepEqual(read.body.data.hours, { start: '06:00', end: '20:30' });

    assert.equal((await t.request('PUT', '/api/cameras/RXL-01/fence', { body: { polygon: [[0, 0]] }, token: sup.token })).status, 400);
    assert.equal((await t.request('PUT', '/api/cameras/RXL-01/hours', { body: { start: '25:00', end: '20:30' }, token: sup.token })).status, 400);
    assert.equal((await t.request('PUT', '/api/cameras/NOPE/fence', { body: { polygon }, token: sup.token })).status, 404);

    const actions = auditActions(t).map((e) => e.action);
    assert.ok(actions.includes('camera.fence.edit'));
    assert.ok(actions.includes('camera.hours.edit'));
  });

  it('act, dismiss and verify write audit rows naming the actor', async () => {
    const op = await t.login('operator');
    const a = (await t.request('POST', '/api/alerts/simulate')).body.data.id;
    const b = (await t.request('POST', '/api/alerts/simulate')).body.data.id;
    await t.request('POST', `/api/alerts/${a}/act`, { token: op.token });
    await t.request('POST', `/api/alerts/${b}/dismiss`, { token: op.token });
    await t.request('POST', '/api/evidence/verify', { token: op.token });
    const rows = auditActions(t);
    assert.ok(rows.some((r) => r.action === 'alert.act' && r.actor === USERS.operator && r.subject === a));
    assert.ok(rows.some((r) => r.action === 'alert.dismiss' && r.actor === USERS.operator && r.subject === b));
    assert.ok(rows.some((r) => r.action === 'evidence.verify' && r.actor === USERS.operator));
  });

  it('with no token the console still works and the audit row is marked unauthenticated', async () => {
    const id = (await t.request('POST', '/api/alerts/simulate')).body.data.id;
    assert.equal((await t.request('POST', `/api/alerts/${id}/act`)).status, 200);
    const row = t.container.repos.audit.recent(5).find((e) => e.action === 'alert.act' && e.subject === id);
    assert.equal(row.actor_id, null);
    assert.equal(row.detail.unauthenticated, true);
  });

  it('GET /api/audit is Supervisor-only and the log chain verifies; tampering is detected', async () => {
    assert.equal((await t.request('GET', '/api/audit')).status, 401);
    assert.equal((await t.request('GET', '/api/audit', { token: (await t.login('operator')).token })).status, 403);
    const sup = await t.login('supervisor');
    const res = await t.request('GET', '/api/audit', { token: sup.token });
    assert.equal(res.status, 200);
    assert.equal(res.body.data.chain.intact, true);
    assert.ok(res.body.data.entries.length > 3);

    t.container.db.prepare("UPDATE audit_log SET action = 'nothing.happened' WHERE id = 1").run();
    const tampered = await t.request('GET', '/api/audit', { token: sup.token });
    assert.equal(tampered.body.data.chain.intact, false);
    assert.equal(tampered.body.data.chain.firstBadId, 1);
  });
});

describe('AUTH_REQUIRED', () => {
  it('defaults to off: console endpoints answer without a token', async () => {
    const t = await startApp();
    try {
      assert.equal((await t.request('GET', '/api/alerts')).status, 200);
      assert.equal((await t.request('GET', '/api/cameras')).status, 200);
    } finally {
      await t.close();
    }
  });

  it('when on, console endpoints and the stream need a token; health and auth stay public', async () => {
    const t = await startApp({ AUTH_REQUIRED: 'true' });
    try {
      for (const [method, url] of [
        ['GET', '/api/alerts'],
        ['GET', '/api/alerts/budget'],
        ['POST', '/api/alerts/simulate'],
        ['GET', '/api/cameras'],
        ['GET', '/api/cameras/posts'],
        ['GET', '/api/analytics'],
        ['GET', '/api/analytics/traffic'],
        ['GET', '/api/evidence/chain'],
        ['POST', '/api/evidence/verify'],
        ['GET', '/api/search/placeholders'],
        ['POST', '/api/search'],
        ['GET', '/api/stream'],
      ]) {
        const res = await t.request(method, url, method === 'POST' ? { body: {} } : {});
        assert.equal(res.status, 401, `${method} ${url}`);
        assert.deepEqual(res.body, { error: 'Authentication required' });
      }
      assert.equal((await t.request('GET', '/api/health')).status, 200);
      assert.equal((await t.request('GET', '/api/auth/units')).status, 200);
      assert.equal((await t.request('GET', '/api/auth/roles')).status, 200);
      const { token } = await t.login('operator');
      assert.equal((await t.request('GET', '/api/alerts', { token })).status, 200);
      assert.equal((await t.request('GET', '/api/evidence/chain', { token })).status, 200);
    } finally {
      await t.close();
    }
  });

  it('ingest is not governed by user tokens: it uses its own key', async () => {
    const t = await startApp({ AUTH_REQUIRED: 'true' });
    try {
      const { sampleEvent } = await import('./helpers.js');
      assert.equal((await t.ingest(sampleEvent())).status, 201);
    } finally {
      await t.close();
    }
  });
});
