import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';

import { createApp } from '../src/app.js';
import { loadConfig } from '../src/config/index.js';

export const SEED_PASSWORD = 'test-seed-password-1';
export const INGEST_KEY = 'test-ingest-key-0123456789abcdef0123456789abcdef';
export const JWT_SECRET = 'test-jwt-secret-0123456789abcdef0123456789abcdef';

export const USERS = {
  operator: 'SSB/104382',
  supervisor: 'SSB/200117',
  analyst: 'SSB/300245',
};

/** A fresh, isolated API on an ephemeral port. Each call has its own database and rate limiters. */
export const startApp = async (env = {}, { logger } = {}) => {
  const config = loadConfig({
    NODE_ENV: 'test',
    DATABASE_PATH: ':memory:',
    BCRYPT_ROUNDS: '4',
    TRUEWATCH_INGEST_KEY: INGEST_KEY,
    JWT_SECRET,
    SEED_USER_PASSWORD: SEED_PASSWORD,
    API_RATE_LIMIT_PER_MINUTE: '100000',
    INGEST_RATE_LIMIT_PER_MINUTE: '100000',
    AUTH_RATE_LIMIT_PER_15_MIN: '100000',
    EDGE_BASE_URL: '',
    ...env,
  });
  const app = createApp(logger ? { config, logger } : { config });
  const server = await new Promise((resolve) => {
    const s = app.listen(0, '127.0.0.1', () => resolve(s));
  });
  const base = `http://127.0.0.1:${server.address().port}`;

  const request = async (method, url, { body, headers = {}, token, raw } = {}) => {
    const response = await fetch(base + url, {
      method,
      headers: {
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...headers,
      },
      body: body === undefined ? undefined : raw ? body : JSON.stringify(body),
    });
    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch {
      json = null;
    }
    return { status: response.status, body: json, text, headers: response.headers, cookies: response.headers.getSetCookie() };
  };

  const login = async (who) => {
    const res = await request('POST', '/api/auth/login', {
      body: { officialId: USERS[who], password: SEED_PASSWORD },
    });
    assert.equal(res.status, 200, `login as ${who} failed`);
    return { token: res.body.data.token, cookie: res.cookies[0]?.split(';')[0], user: res.body.data.user };
  };

  const ingest = (body, headers = {}) =>
    request('POST', '/api/ingest/event', { body, headers: { 'X-Ingest-Key': INGEST_KEY, ...headers } });

  const update = (body, headers = {}) =>
    request('POST', '/api/ingest/update', { body, headers: { 'X-Ingest-Key': INGEST_KEY, ...headers } });

  return {
    app,
    config,
    server,
    base,
    container: app.locals.container,
    request,
    login,
    ingest,
    update,
    async close() {
      app.locals.close();
      server.closeAllConnections?.();
      await new Promise((resolve) => server.close(resolve));
    },
  };
};

/** A valid provisional truewatch.event.v1. */
export const sampleEvent = (overrides = {}) => ({
  schema: 'truewatch.event.v1',
  event_id: randomUUID(),
  post_id: 'RXL',
  camera_id: 'RXL-01',
  captured_at: new Date().toISOString(),
  stage: 'provisional',
  detection: {
    track_id: null,
    class: 'person',
    bbox: [0.41, 0.52, 0.06, 0.18],
    channel_scores: { appearance: 0.87, motion: 0.63 },
    agreed: true,
    threshold: 0.55,
    ir: true,
    dori_band: 'observe',
  },
  rule: {
    fired: ['fence crossed'],
    baseline: { dominant_direction: 'ltr', with: 6, against: 4 },
    metrics: { dwell_s: 5.0, net_px: 70, path_px: 163 },
  },
  explanation: null,
  anpr: null,
  evidence: null,
  confidence: 93,
  source: 'file',
  ...overrides,
});

export const sampleUpdate = (eventId, overrides = {}) => ({
  schema: 'truewatch.update.v1',
  event_id: eventId,
  explanation: { text: 'Six people crossed the dry channel in single file.', source: 'vlm', model: 'moondream2', latency_ms: 2840 },
  ...overrides,
});

export const waitFor = async (predicate, { timeout = 3000, interval = 20, message = 'condition' } = {}) => {
  const deadline = Date.now() + timeout;
  for (;;) {
    const value = await predicate();
    if (value) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${message}`);
    await new Promise((r) => setTimeout(r, interval));
  }
};

/** Opens GET /api/stream and parses frames into { id, event, data } as they arrive. */
export const openStream = async (base, { headers = {} } = {}) => {
  const controller = new AbortController();
  const response = await fetch(`${base}/api/stream`, { headers, signal: controller.signal });
  const frames = [];
  const comments = [];
  let ended = false;

  const pump = (async () => {
    const decoder = new TextDecoder();
    let buffer = '';
    try {
      for await (const chunk of response.body) {
        buffer += decoder.decode(chunk, { stream: true });
        let cut;
        while ((cut = buffer.indexOf('\n\n')) !== -1) {
          const raw = buffer.slice(0, cut);
          buffer = buffer.slice(cut + 2);
          if (raw.startsWith(':')) {
            comments.push(raw);
          } else {
            const frame = {};
            for (const line of raw.split('\n')) {
              const i = line.indexOf(':');
              const field = line.slice(0, i);
              const value = line.slice(i + 1).trimStart();
              if (field === 'data') frame.data = JSON.parse(value);
              else frame[field] = value;
            }
            frames.push(frame);
          }
        }
      }
    } catch {
      // aborted by close()
    }
    ended = true;
  })();

  return {
    response,
    frames,
    comments,
    isEnded: () => ended,
    waitForEvent: (name, predicate = () => true, timeout = 3000) =>
      waitFor(() => frames.find((f) => f.event === name && predicate(f.data)), { timeout, message: `stream event ${name}` }),
    waitForEnd: (timeout = 3000) => waitFor(() => ended, { timeout, message: 'stream to end' }),
    close: async () => {
      controller.abort();
      await pump;
    },
  };
};

/** A throwaway HTTP server standing in for the edge service. `routes` maps 'METHOD /path' to a handler. */
export const startFakeEdge = async (routes) => {
  const seen = [];
  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => {
      body += c;
    });
    req.on('end', () => {
      const parsed = body ? JSON.parse(body) : null;
      seen.push({ method: req.method, path: req.url, body: parsed });
      const handler = routes[`${req.method} ${req.url}`];
      if (!handler) {
        res.writeHead(404).end();
        return;
      }
      const result = handler(parsed);
      if (result.hang) return; // never answers: exercises the caller's timeout
      res.writeHead(result.status ?? 200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result.body));
    });
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return {
    url: `http://127.0.0.1:${server.address().port}`,
    seen,
    close: () => {
      server.closeAllConnections();
      return new Promise((resolve) => server.close(resolve));
    },
  };
};

export const tempDbPath = () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'truewatch-test-'));
  return { dir, file: path.join(dir, 'truewatch.db'), cleanup: () => fs.rmSync(dir, { recursive: true, force: true }) };
};
