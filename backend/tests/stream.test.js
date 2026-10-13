import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { openStream, sampleEvent, sampleUpdate, startApp, waitFor } from './helpers.js';
import { expectAlert } from './shape.js';

describe('GET /api/stream', () => {
  let t;
  before(async () => {
    t = await startApp({ STREAM_HEARTBEAT_SECONDS: '1' });
  });
  after(async () => {
    await t.close();
  });

  it('answers text/event-stream with no-cache headers', async () => {
    const stream = await openStream(t.base);
    try {
      assert.equal(stream.response.status, 200);
      assert.match(stream.response.headers.get('content-type'), /^text\/event-stream/);
      assert.match(stream.response.headers.get('cache-control'), /no-cache/);
      await waitFor(() => stream.comments.includes(': connected'), { message: 'connect comment' });
    } finally {
      await stream.close();
    }
  });

  it('delivers alert.created and then the reason patch as alert.updated', async () => {
    const stream = await openStream(t.base);
    try {
      const event = sampleEvent();
      const created = await t.ingest(event);

      const createdFrame = await stream.waitForEvent('alert.created', (a) => a.id === created.body.alert_id);
      expectAlert(createdFrame.data);
      assert.equal(createdFrame.data.stage, 'prov');
      assert.match(createdFrame.id, /^\d+$/);

      await t.update(sampleUpdate(event.event_id));
      const patched = await stream.waitForEvent('alert.updated', (p) => p.id === created.body.alert_id);
      assert.deepEqual(Object.keys(patched.data).sort(), ['alert', 'id', 'patch']);
      assert.equal(patched.data.patch.reason, 'Six people crossed the dry channel in single file.');
      assert.equal(patched.data.patch.reason_source, 'vlm');
      assert.equal(patched.data.alert.reason, patched.data.patch.reason);
      expectAlert(patched.data.alert);

      const order = stream.frames.filter((f) => f.data?.id === created.body.alert_id).map((f) => f.event);
      assert.deepEqual(order, ['alert.created', 'alert.updated']);
    } finally {
      await stream.close();
    }
  });

  it('publishes alert.stage for sealing by an operator and for dismissal', async () => {
    const stream = await openStream(t.base);
    try {
      const a = (await t.request('POST', '/api/alerts/simulate')).body.data.id;
      const b = (await t.request('POST', '/api/alerts/simulate')).body.data.id;
      await stream.waitForEvent('alert.created', (x) => x.id === b);

      await t.request('POST', `/api/alerts/${a}/act`);
      const sealed = await stream.waitForEvent('alert.stage', (x) => x.id === a);
      assert.equal(sealed.data.stage, 'sealed');
      assert.equal(sealed.data.previous, 'prov');

      await t.request('POST', `/api/alerts/${b}/dismiss`);
      const gone = await stream.waitForEvent('alert.stage', (x) => x.id === b);
      assert.equal(gone.data.stage, 'gone');
    } finally {
      await stream.close();
    }
  });

  it('sends a heartbeat comment on the configured interval', async () => {
    const stream = await openStream(t.base);
    try {
      await waitFor(() => stream.comments.includes(': heartbeat'), { timeout: 3000, message: 'heartbeat' });
    } finally {
      await stream.close();
    }
  });

  it('a duplicate ingest or a no-op update publishes nothing', async () => {
    const stream = await openStream(t.base);
    try {
      const event = sampleEvent();
      await t.ingest(event);
      await t.update(sampleUpdate(event.event_id));
      await stream.waitForEvent('alert.updated', () => true);
      const count = stream.frames.length;
      await t.ingest(event);
      await t.update(sampleUpdate(event.event_id));
      await new Promise((r) => setTimeout(r, 150));
      assert.equal(stream.frames.length, count);
    } finally {
      await stream.close();
    }
  });

  it('releases the client on disconnect', async () => {
    const stream = await openStream(t.base);
    await waitFor(() => t.container.hub.size() >= 1, { message: 'client registered' });
    await stream.close();
    await waitFor(() => t.container.hub.size() === 0, { message: 'client released' });
  });

  it('ends every open stream cleanly on shutdown', async () => {
    const own = await startApp();
    const a = await openStream(own.base);
    const b = await openStream(own.base);
    await waitFor(() => own.container.hub.size() === 2, { message: 'two clients' });
    own.container.hub.closeAll();
    await a.waitForEnd();
    await b.waitForEnd();
    assert.ok(a.frames.some((f) => f.event === 'shutdown'));
    assert.equal(own.container.hub.size(), 0);
    await own.close();
  });

  it('caps concurrent clients', async () => {
    const own = await startApp({ STREAM_MAX_CLIENTS: '1' });
    const first = await openStream(own.base);
    try {
      await waitFor(() => own.container.hub.size() === 1);
      const second = await own.request('GET', '/api/stream');
      assert.equal(second.status, 503);
    } finally {
      await first.close();
      await own.close();
    }
  });

  it('leaves polling untouched: GET /api/alerts reflects the same alerts', async () => {
    const event = sampleEvent();
    const { body } = await t.ingest(event);
    const polled = (await t.request('GET', '/api/alerts')).body.data;
    assert.ok(polled.some((a) => a.id === body.alert_id));
  });
});
