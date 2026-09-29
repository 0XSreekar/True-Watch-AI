import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { ConfigError, DEV_DEFAULTS, loadConfig, validateConfig } from '../src/config/index.js';
import { createShutdown } from '../src/lifecycle.js';
import { createLogger } from '../src/utils/logger.js';
import { INGEST_KEY, SEED_PASSWORD, USERS, openStream, sampleEvent, startApp, waitFor } from './helpers.js';

const capture = () => {
  const lines = [];
  const sink = { stdout: { write: (l) => lines.push(l) }, stderr: { write: (l) => lines.push(l) } };
  return { lines, parsed: () => lines.map((l) => JSON.parse(l)), logger: createLogger({ sink }) };
};

describe('error handling', () => {
  for (const env of ['test', 'development', 'production']) {
    it(`never leaks a stack trace, message or path to the client (NODE_ENV=${env})`, async () => {
      const { logger, parsed } = capture();
      const secrets = env === 'production' ? { JWT_SECRET: 'j'.repeat(40), TRUEWATCH_INGEST_KEY: 'k'.repeat(40) } : {};
      const t = await startApp({ NODE_ENV: env, ...secrets }, { logger });
      try {
        t.container.services.cameras.listCameras = () => {
          throw new Error('SQLITE_CORRUPT at /var/lib/truewatch/secret.db');
        };
        const res = await t.request('GET', '/api/cameras');
        assert.equal(res.status, 500);
        assert.deepEqual(res.body, { error: 'Internal server error' });
        assert.doesNotMatch(res.text, /stack|SQLITE|secret\.db|\/var\/lib|at .*\.js/);
        const errorLine = parsed().find((l) => l.level === 'error');
        assert.ok(errorLine, 'the detail is logged server-side');
        assert.match(errorLine.error, /SQLITE_CORRUPT/);
        assert.equal(errorLine.requestId, res.headers.get('x-request-id'));
      } finally {
        await t.close();
      }
    });
  }

  it('answers unknown routes with { error } and malformed or oversized bodies with 400 / 413', async () => {
    const t = await startApp();
    try {
      const missing = await t.request('GET', '/api/nope');
      assert.equal(missing.status, 404);
      assert.deepEqual(Object.keys(missing.body), ['error']);

      const bad = await t.request('POST', '/api/search', { body: '{"query":', raw: true, headers: { 'Content-Type': 'application/json' } });
      assert.equal(bad.status, 400);
      assert.deepEqual(bad.body, { error: 'Malformed JSON body' });

      const big = await t.request('POST', '/api/search', { body: JSON.stringify({ query: 'x'.repeat(300_000) }), raw: true });
      assert.equal(big.status, 413);
      assert.deepEqual(big.body, { error: 'Request body too large' });
    } finally {
      await t.close();
    }
  });
});

describe('headers, CORS and request ids', () => {
  it('sets security headers and hides the framework', async () => {
    const t = await startApp();
    try {
      const res = await t.request('GET', '/api/health');
      assert.equal(res.headers.get('x-content-type-options'), 'nosniff');
      assert.ok(res.headers.get('content-security-policy'));
      assert.equal(res.headers.get('x-powered-by'), null);
      assert.equal(res.headers.get('cache-control'), 'no-store');
    } finally {
      await t.close();
    }
  });

  it('CORS only allows origins on the allow-list', async () => {
    const t = await startApp({ CORS_ORIGIN: 'http://localhost:5173, https://console.example.org' });
    try {
      const allowed = await t.request('GET', '/api/health', { headers: { Origin: 'https://console.example.org' } });
      assert.equal(allowed.headers.get('access-control-allow-origin'), 'https://console.example.org');
      assert.equal(allowed.headers.get('access-control-allow-credentials'), 'true');

      const blocked = await t.request('GET', '/api/health', { headers: { Origin: 'https://evil.example' } });
      assert.equal(blocked.headers.get('access-control-allow-origin'), null);

      const preflight = await t.request('OPTIONS', '/api/alerts/simulate', {
        headers: { Origin: 'http://localhost:5173', 'Access-Control-Request-Method': 'POST' },
      });
      assert.equal(preflight.headers.get('access-control-allow-origin'), 'http://localhost:5173');
    } finally {
      await t.close();
    }
  });

  it('gives every response a request id, keeping a safe incoming one and replacing an unsafe one', async () => {
    const t = await startApp();
    try {
      const generated = await t.request('GET', '/api/health');
      assert.match(generated.headers.get('x-request-id'), /^[0-9a-f-]{36}$/);
      const kept = await t.request('GET', '/api/health', { headers: { 'X-Request-Id': 'proxy-req-12345' } });
      assert.equal(kept.headers.get('x-request-id'), 'proxy-req-12345');
      const replaced = await t.request('GET', '/api/health', { headers: { 'X-Request-Id': 'bad id\twith spaces' } });
      assert.match(replaced.headers.get('x-request-id'), /^[0-9a-f-]{36}$/);
    } finally {
      await t.close();
    }
  });

  it('tags every access-log line with the request id and never logs query strings or secrets', async () => {
    const { logger, parsed } = capture();
    const t = await startApp({ NODE_ENV: 'development' }, { logger });
    try {
      await t.request('GET', '/api/cameras?sector=Raxaul');
      await t.request('POST', '/api/auth/login', { body: { officialId: USERS.operator, password: SEED_PASSWORD } });
      await t.ingest(sampleEvent());
      const lines = parsed().filter((l) => l.msg === 'request');
      assert.ok(lines.length >= 3);
      for (const line of lines) {
        assert.match(line.requestId, /^[0-9a-f-]{36}$/);
        assert.doesNotMatch(JSON.stringify(line), /sector=|password|Bearer/i);
      }
      assert.ok(!JSON.stringify(parsed()).includes(INGEST_KEY));
      assert.ok(!JSON.stringify(parsed()).includes(SEED_PASSWORD));
    } finally {
      await t.close();
    }
  });
});

describe('rate limiting', () => {
  it('limits the console API per client', async () => {
    const t = await startApp({ API_RATE_LIMIT_PER_MINUTE: '3' });
    try {
      const statuses = [];
      for (let i = 0; i < 5; i += 1) statuses.push((await t.request('GET', '/api/cameras')).status);
      assert.deepEqual(statuses, [200, 200, 200, 429, 429]);
      assert.deepEqual((await t.request('GET', '/api/cameras')).body, { error: 'Too many requests' });
    } finally {
      await t.close();
    }
  });

  it('throttles repeated failed logins but not successful ones', async () => {
    const t = await startApp({ AUTH_RATE_LIMIT_PER_15_MIN: '3' });
    try {
      const bad = { officialId: USERS.operator, password: 'wrong-password' };
      const statuses = [];
      for (let i = 0; i < 5; i += 1) statuses.push((await t.request('POST', '/api/auth/login', { body: bad })).status);
      assert.deepEqual(statuses, [401, 401, 401, 429, 429]);
    } finally {
      await t.close();
    }
    const busy = await startApp({ AUTH_RATE_LIMIT_PER_15_MIN: '3' });
    try {
      for (let i = 0; i < 6; i += 1) {
        const ok = await busy.request('POST', '/api/auth/login', { body: { officialId: USERS.operator, password: SEED_PASSWORD } });
        assert.equal(ok.status, 200);
      }
    } finally {
      await busy.close();
    }
  });
});

describe('startup configuration', () => {
  it('production refuses to start without a JWT secret and an ingest key, naming both', () => {
    const config = loadConfig({ NODE_ENV: 'production' });
    assert.equal(config.jwtSecret, '');
    assert.equal(config.ingestKey, '');
    assert.throws(
      () => validateConfig(config),
      (error) => {
        assert.ok(error instanceof ConfigError);
        assert.match(error.message, /JWT_SECRET is required in production/);
        assert.match(error.message, /TRUEWATCH_INGEST_KEY is required in production/);
        return true;
      },
    );
  });

  it('production rejects short secrets and a wildcard CORS origin, and accepts real ones', () => {
    assert.throws(() => validateConfig(loadConfig({ NODE_ENV: 'production', JWT_SECRET: 'short', TRUEWATCH_INGEST_KEY: 'short' })), /at least 32/);
    assert.throws(
      () => validateConfig(loadConfig({ NODE_ENV: 'production', JWT_SECRET: 'j'.repeat(32), TRUEWATCH_INGEST_KEY: 'k'.repeat(32), CORS_ORIGIN: '*' })),
      /allow-list/,
    );
    const ok = loadConfig({ NODE_ENV: 'production', JWT_SECRET: 'j'.repeat(32), TRUEWATCH_INGEST_KEY: 'k'.repeat(32) });
    assert.deepEqual(validateConfig(ok), []);
    assert.equal(ok.seedDemoUsers, false, 'no demo accounts in production unless asked for');
  });

  it('development accepts the documented defaults and warns about them', () => {
    const config = loadConfig({ NODE_ENV: 'development' });
    assert.equal(config.jwtSecret, DEV_DEFAULTS.jwtSecret);
    const warnings = validateConfig(config);
    assert.equal(warnings.length, 2);
    assert.match(warnings.join(' '), /insecure development default/);
  });

  it('parses durations and flags', () => {
    const config = loadConfig({ JWT_EXPIRES_IN: '20m', REFRESH_TTL: '2d', AUTH_REQUIRED: 'true', SHIFT_HOURS: '4' });
    assert.equal(config.jwtExpiresInSeconds, 1200);
    assert.equal(config.refreshTtlSeconds, 172800);
    assert.equal(config.authRequired, true);
    assert.equal(config.shiftHours, 4);
    assert.equal(loadConfig({}).authRequired, false);
  });
});

describe('graceful shutdown', () => {
  it('ends open streams, stops listening and closes the database', async () => {
    const t = await startApp();
    const { logger } = capture();
    const stream = await openStream(t.base);
    await waitFor(() => t.container.hub.size() === 1);

    const shutdown = createShutdown({ server: t.server, container: t.container, logger, timeoutMs: 2000 });
    await shutdown('test');
    await stream.waitForEnd();

    assert.ok(stream.frames.some((f) => f.event === 'shutdown'));
    assert.equal(t.container.db.open, false);
    assert.equal(t.server.listening, false);
    await assert.rejects(() => fetch(`${t.base}/api/health`));
    await shutdown('again');
  });
});
