import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { MIGRATIONS } from '../src/db/migrations.js';
import { SEED_PASSWORD, USERS, sampleEvent, sampleUpdate, startApp, tempDbPath } from './helpers.js';

describe('persistence across a restart', () => {
  it('a second app instance on the same database file sees everything the first one wrote', async () => {
    const db = tempDbPath();
    try {
      const env = { DATABASE_PATH: db.file };

      const first = await startApp(env);
      const event = sampleEvent({ detection: { ...sampleEvent().detection, track_id: 9 } });
      const ingested = await first.ingest(event);
      await first.update(sampleUpdate(event.event_id));
      const simulated = (await first.request('POST', '/api/alerts/simulate')).body.data;
      await first.request('POST', `/api/alerts/${simulated.id}/act`);
      const registered = await first.request('POST', '/api/auth/register', {
        body: {
          name: 'Ct. Persisted Person',
          officialId: 'SSB/999001',
          unit: '20th Bn — Raxaul',
          email: 'persisted@example.org',
          password: 'a-long-enough-password',
          role: 'Operator',
        },
      });
      assert.equal(registered.status, 201);
      const alertsBefore = (await first.request('GET', '/api/alerts')).body.data;
      const chainBefore = (await first.request('GET', '/api/evidence/chain')).body.data;
      await first.close();

      const second = await startApp(env);
      try {
        const alertsAfter = (await second.request('GET', '/api/alerts')).body.data;
        assert.deepEqual(alertsAfter, alertsBefore, 'alerts survive a restart unchanged');
        const restored = alertsAfter.find((a) => a.id === ingested.body.alert_id);
        assert.equal(restored.reason, 'Six people crossed the dry channel in single file.', 'the patched reason survived');
        assert.equal(alertsAfter.find((a) => a.id === simulated.id).stage, 'sealed');

        assert.deepEqual((await second.request('GET', '/api/evidence/chain')).body.data, chainBefore);
        assert.equal((await second.request('POST', '/api/evidence/verify')).body.data.intact, true);

        const replay = await second.ingest(event);
        assert.equal(replay.status, 200);
        assert.equal(replay.body.duplicate, true, 'the idempotency ledger survives too');
        assert.equal(replay.body.alert_id, ingested.body.alert_id);

        const login = await second.request('POST', '/api/auth/login', {
          body: { officialId: 'SSB/999001', password: 'a-long-enough-password' },
        });
        assert.equal(login.status, 200);
        assert.equal((await second.request('POST', '/api/auth/login', { body: { officialId: USERS.supervisor, password: SEED_PASSWORD } })).status, 200);

        const audit = second.container.repos.audit.verify();
        assert.equal(audit.intact, true);
        assert.ok(audit.entries >= 2);
      } finally {
        await second.close();
      }
    } finally {
      db.cleanup();
    }
  });

  it('seeds once: a restart does not duplicate the catalogue, chain or demo users', async () => {
    const db = tempDbPath();
    try {
      const env = { DATABASE_PATH: db.file };
      const counts = (t) => ({
        cameras: t.container.repos.cameras.count(),
        posts: t.container.repos.posts.count(),
        blocks: t.container.repos.evidence.count(),
        users: t.container.repos.users.count(),
        alerts: t.container.repos.alerts.total(),
      });
      const first = await startApp(env);
      const initial = counts(first);
      await first.close();
      const second = await startApp(env);
      const third = counts(second);
      await second.close();
      assert.deepEqual(third, initial);
      assert.deepEqual(initial, { cameras: 6, posts: 7, blocks: 5, users: 3, alerts: 42 });
    } finally {
      db.cleanup();
    }
  });

  it('migrations are recorded once and are safe to run again', async () => {
    const db = tempDbPath();
    try {
      const env = { DATABASE_PATH: db.file };
      const a = await startApp(env);
      await a.close();
      const b = await startApp(env);
      const rows = b.container.db.prepare('SELECT id, name FROM schema_migrations ORDER BY id').all();
      assert.deepEqual(rows.map((r) => r.id), MIGRATIONS.map((m) => m.id));
      await b.close();
    } finally {
      db.cleanup();
    }
  });

  it('a fresh database demos with no edge service: seeded cameras, posts, chain and analytics', async () => {
    const t = await startApp({ EDGE_BASE_URL: '' });
    try {
      assert.equal((await t.request('GET', '/api/cameras')).body.data.length, 6);
      assert.equal((await t.request('GET', '/api/cameras/posts')).body.data.length, 7);
      assert.equal((await t.request('GET', '/api/evidence/chain')).body.data.length, 5);
      const analytics = (await t.request('GET', '/api/analytics')).body.data;
      assert.deepEqual(analytics.camBars.map((c) => c.v), [9, 6, 11, 4, 7, 5]);
      assert.deepEqual(analytics.dayNight.map((d) => d.v), [38, 62]);
      assert.deepEqual(analytics.agreement, [91, 93, 92, 95, 94, 96, 95, 97, 96, 97, 98, 97]);
      const live = (await t.request('GET', '/api/alerts')).body.data;
      assert.deepEqual(live, [], 'seeded history never appears in the live queue');
      const sim = await t.request('POST', '/api/alerts/simulate');
      assert.equal(sim.status, 201);
    } finally {
      await t.close();
    }
  });

  it('analytics move as real alerts arrive', async () => {
    const t = await startApp();
    try {
      const before = (await t.request('GET', '/api/analytics')).body.data;
      await t.ingest(sampleEvent({ confidence: 60 }));
      const after = (await t.request('GET', '/api/analytics')).body.data;
      const rxl = (d) => d.camBars.find((c) => c.id === 'RXL-01').v;
      assert.equal(rxl(after), rxl(before) + 1);
      assert.equal(after.agreement.at(-1), 60);
    } finally {
      await t.close();
    }
  });
});
