// Run with: node --test frontend/tests
// No test runner is installed in the frontend workspace; node:test ships with Node 20+, so this
// adds no dependency. It exercises the pure service modules only (no React, no Vite globals).
import assert from 'node:assert/strict';
import http from 'node:http';
import { describe, it } from 'node:test';

import {
  addAlert,
  confirmAlert,
  patchAlert,
  planSnapshot,
  removeAlert,
  stageAlert,
  toDisplayAlert,
} from '../src/services/alertReducer.js';
import { backoffDelay, createLiveAlerts } from '../src/services/liveAlerts.js';
import { createSseParser } from '../src/services/sse.js';

const frames = (text, chunk) => {
  const out = [];
  const comments = [];
  const p = createSseParser({ onFrame: (f) => out.push(f), onComment: (c) => comments.push(c) });
  for (let i = 0; i < text.length; i += chunk) p.feed(text.slice(i, i + chunk));
  return { out, comments };
};

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const until = async (fn, ms = 3000) => {
  const end = Date.now() + ms;
  while (!fn()) {
    if (Date.now() > end) throw new Error('timed out');
    await wait(10);
  }
};

const alert = (id, extra = {}) => ({
  id,
  camera: { id: 'RXL-01', name: 'Raxaul', ir: false },
  stage: 'prov',
  time: '10:00:00',
  reason: 'Two people near the fence.',
  confidence: 90,
  plate: null,
  hash: 'abc',
  seed: 1,
  ...extra,
});

describe('SSE parser', () => {
  const wire =
    'retry: 3000\n\n: connected\n\nid: 1\nevent: alert.created\ndata: {"id":"A1"}\n\n' +
    ': heartbeat\n\r\nid: 2\r\nevent: alert.updated\r\ndata: {"a":\r\ndata: 1}\r\n\r\n';

  for (const size of [1, 3, 7, 1000]) {
    it(`parses frames and comments across ${size}-char chunks`, () => {
      const { out, comments } = frames(wire, size);
      assert.deepEqual(comments, ['connected', 'heartbeat']);
      const real = out.filter((f) => f.event !== 'message');
      assert.equal(real.length, 2);
      assert.deepEqual(real[0], { id: '1', event: 'alert.created', data: '{"id":"A1"}', retry: undefined });
      assert.equal(real[1].event, 'alert.updated');
      assert.equal(real[1].data, '{"a":\n1}');
    });
  }
});

describe('alert reducer', () => {
  const stamp = () => '00:00:00';

  it('patches a reason in place: same position, other cards untouched', () => {
    const a = toDisplayAlert(alert('A'), stamp);
    const b = toDisplayAlert(alert('B'), stamp);
    const list = [b, a];
    const next = patchAlert(list, 'A', { reason: 'Six people crossed.', reason_source: 'vlm' });
    assert.deepEqual(next.map((x) => x.id), ['B', 'A']);
    assert.equal(next[0], b, 'untouched card keeps identity');
    assert.equal(next[1].reason, 'Six people crossed.');
    assert.equal(next[1].stage, 'prov', 'local reveal stage is not overwritten by a patch');
    assert.equal(patchAlert(list, 'A', { reason: a.reason }), list, 'no-op returns the same array');
    assert.equal(patchAlert(list, 'missing', { reason: 'x' }), list);
  });

  it('retypes a finished reason and continues a partial one', () => {
    const done = { ...toDisplayAlert(alert('A'), stamp), stage: 'conf', typed: 26, conf: 90 };
    assert.equal(patchAlert([done], 'A', { reason: 'New text here.' })[0].typed, 0);
    const partial = { ...done, typed: 10 };
    assert.equal(patchAlert([partial], 'A', { reason: 'New text here, longer.' })[0].typed, 10);
  });

  it('applies stage rules and ignores duplicates', () => {
    const a = toDisplayAlert(alert('A'), stamp);
    const conf = confirmAlert([a], 'A');
    assert.equal(conf[0].stage, 'conf');
    assert.equal(confirmAlert(conf, 'A'), conf);
    const sealed = stageAlert(conf, 'A', 'sealed');
    assert.equal(sealed[0].stage, 'sealed');
    assert.equal(stageAlert(sealed, 'A', 'prov'), sealed);
    assert.equal(stageAlert(sealed, 'A', 'sealed'), sealed);
    const gone = stageAlert(sealed, 'A', 'gone');
    assert.equal(gone[0].stage, 'gone');
    assert.equal(removeAlert(gone, 'A').length, 0);
  });

  it('never adds a known id twice and caps the queue', () => {
    let list = [];
    for (let i = 0; i < 9; i += 1) list = addAlert(list, toDisplayAlert(alert(`A${i}`), stamp));
    assert.equal(list.length, 6);
    assert.equal(list[0].id, 'A8');
    assert.equal(addAlert(list, list[0]), list);
  });

  it('plans a snapshot: new ones oldest-first, patches skip touched ids', () => {
    const snap = [alert('C'), alert('B', { reason: 'later' }), alert('A', { reason: 'later' })];
    const plan = planSnapshot(snap, { seen: new Set(['A', 'B']), skip: new Set(['B']) });
    assert.deepEqual(plan.fresh.map((x) => x.id), ['C']);
    assert.deepEqual(plan.patches.map((p) => p.id), ['A']);
  });
});

describe('backoff', () => {
  it('doubles, then holds at the cap', () => {
    const mid = () => 0.5; // jitter factor 1.0
    const opts = { base: 1000, factor: 2, max: 30_000 };
    assert.deepEqual(
      [0, 1, 2, 3, 4, 5, 6, 10].map((n) => backoffDelay(n, opts, mid)),
      [1000, 2000, 4000, 8000, 16000, 30000, 30000, 30000],
    );
  });
});

// A stand-in for GET /api/stream that can be killed and restarted on the same port.
const startStreamServer = async (port = 0) => {
  const clients = new Set();
  let connections = 0;
  let lastAuth = null;
  const server = http.createServer((req, res) => {
    connections += 1;
    lastAuth = req.headers.authorization ?? null;
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.write('retry: 3000\n\n: connected\n\n');
    clients.add(res);
    res.on('close', () => clients.delete(res));
  });
  await new Promise((r) => server.listen(port, '127.0.0.1', r));
  const send = (event, data) => clients.forEach((c) => c.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
  return {
    port: server.address().port,
    send,
    connections: () => connections,
    lastAuth: () => lastAuth,
    clients: () => clients.size,
    kill: () =>
      new Promise((r) => {
        clients.forEach((c) => c.destroy());
        server.closeAllConnections();
        server.close(r);
      }),
  };
};

describe('live alerts controller', () => {
  it('streams events, falls back to polling with backoff when killed, and reconnects', async () => {
    let srv = await startStreamServer();
    const { port } = srv;
    const got = [];
    const modes = [];
    let snapshots = 0;

    const live = createLiveAlerts({
      streamUrl: `http://127.0.0.1:${port}/api/stream`,
      getToken: () => 'tok-1',
      fetchSnapshot: async () => {
        snapshots += 1;
        return [alert('S1')];
      },
      backoff: { base: 30, factor: 2, max: 120 },
      pollMs: 40,
      onMode: (m) => modes.push(m),
      onCreated: (a) => got.push(['created', a.id]),
      onUpdated: (u) => got.push(['updated', u.id, u.patch.reason]),
      onStage: (s) => got.push(['stage', s.id, s.stage]),
    });
    live.start();
    live.start(); // idempotent: must not open a second stream

    await until(() => live.mode() === 'stream');
    await until(() => srv.clients() === 1);
    assert.equal(srv.connections(), 1);
    assert.equal(srv.lastAuth(), 'Bearer tok-1');

    srv.send('alert.created', alert('A1'));
    srv.send('alert.updated', { id: 'A1', patch: { reason: 'vlm text' }, alert: alert('A1') });
    srv.send('alert.stage', { id: 'A1', stage: 'sealed', previous: 'prov', alert: alert('A1') });
    await until(() => got.length === 3);
    assert.deepEqual(got, [
      ['created', 'A1'],
      ['updated', 'A1', 'vlm text'],
      ['stage', 'A1', 'sealed'],
    ]);

    // Kill the stream: the controller must go to polling and keep the queue refreshed.
    const before = snapshots;
    await srv.kill();
    await until(() => live.mode() === 'polling');
    await until(() => snapshots >= before + 2);

    // Bring the server back on the same port: it must reconnect on its own.
    srv = await startStreamServer(port);
    await until(() => live.mode() === 'stream', 4000);
    assert.deepEqual(modes.slice(0, 3), ['connecting', 'stream', 'polling']);

    live.stop();
    await until(() => srv.clients() === 0);
    const count = srv.connections();
    await wait(300);
    assert.equal(srv.connections(), count, 'no reconnect after stop()');
    await srv.kill();
  });

  it('polls (never silent) when the stream endpoint is refused from the start', async () => {
    let snapshots = 0;
    const live = createLiveAlerts({
      streamUrl: 'http://127.0.0.1:9/api/stream', // nothing listens on the discard port
      fetchSnapshot: async () => {
        snapshots += 1;
        return [];
      },
      backoff: { base: 20, factor: 2, max: 60 },
      pollMs: 30,
    });
    live.start();
    await until(() => live.mode() === 'polling');
    await until(() => snapshots >= 3);
    live.stop();
  });

  it('refreshes the token once on a 401 and reconnects with the new one', async () => {
    let auth = [];
    const server = http.createServer((req, res) => {
      auth.push(req.headers.authorization);
      if (req.headers.authorization !== 'Bearer fresh') {
        res.writeHead(401, { 'Content-Type': 'application/json' });
        res.end('{"error":"Unauthorized"}');
        return;
      }
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.write(': connected\n\n');
    });
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    let token = 'stale';
    const live = createLiveAlerts({
      streamUrl: `http://127.0.0.1:${server.address().port}/api/stream`,
      getToken: () => token,
      refreshToken: async () => {
        token = 'fresh';
        return token;
      },
    });
    live.start();
    await until(() => live.mode() === 'stream');
    assert.deepEqual(auth, ['Bearer stale', 'Bearer fresh']);
    live.stop();
    server.closeAllConnections();
    await new Promise((r) => server.close(r));
  });
});
