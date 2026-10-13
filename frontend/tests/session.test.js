// Run with: node --test frontend/tests
import assert from 'node:assert/strict';
import { beforeEach, describe, it } from 'node:test';

import { routeAuthError } from '../src/services/authErrors.js';
import {
  bootSession,
  clearSession,
  configureSession,
  getAccessToken,
  getSession,
  guardDecision,
  refreshSession,
  resetSession,
  setSession,
} from '../src/services/session.js';

const httpError = (status, message = 'x') => Object.assign(new Error(message), { status });
const body = (token) => ({ token, expiresIn: 900, user: { id: 'SSB/104382', role: 'Operator' } });

beforeEach(() => resetSession());

describe('route guard decision for /console', () => {
  it('waits while unknown, allows a session or offline demo, redirects when anonymous', () => {
    assert.equal(guardDecision('unknown'), 'wait');
    assert.equal(guardDecision('authenticated'), 'allow');
    assert.equal(guardDecision('offline'), 'allow');
    assert.equal(guardDecision('anonymous'), 'redirect');
  });
});

describe('session boot (page load)', () => {
  it('logged out (refresh answers 401) -> anonymous -> redirect to /login', async () => {
    configureSession({ refresh: async () => Promise.reject(httpError(401, 'Session expired')) });
    assert.equal(guardDecision(getSession().status), 'wait');
    await bootSession();
    assert.equal(getSession().status, 'anonymous');
    assert.equal(guardDecision(getSession().status), 'redirect');
    assert.equal(getAccessToken(), null);
  });

  it('valid refresh cookie -> authenticated with the token held in memory', async () => {
    configureSession({ refresh: async () => body('t1') });
    await bootSession();
    assert.equal(getSession().status, 'authenticated');
    assert.equal(getAccessToken(), 't1');
    assert.equal(guardDecision(getSession().status), 'allow');
  });

  it('API unreachable (network error or 5xx) -> offline demo, console still opens', async () => {
    configureSession({ refresh: async () => Promise.reject(new TypeError('fetch failed')) });
    await bootSession();
    assert.equal(getSession().status, 'offline');
    assert.equal(guardDecision(getSession().status), 'allow');

    resetSession();
    configureSession({ refresh: async () => Promise.reject(httpError(502)) });
    await bootSession();
    assert.equal(getSession().status, 'offline');
  });
});

describe('refresh', () => {
  it('shares one in-flight request between concurrent callers (the cookie rotates)', async () => {
    let calls = 0;
    configureSession({
      refresh: async () => {
        calls += 1;
        await new Promise((r) => setTimeout(r, 20));
        return body(`t${calls}`);
      },
    });
    const [a, b, c] = await Promise.all([refreshSession(), refreshSession(), refreshSession()]);
    assert.equal(calls, 1);
    assert.deepEqual([a, b, c], ['t1', 't1', 't1']);
  });

  it('a 401 clears the session; an unreachable API keeps the token it has', async () => {
    setSession(body('old'));
    configureSession({ refresh: async () => Promise.reject(httpError(503)) });
    assert.equal(await refreshSession(), null);
    assert.equal(getAccessToken(), 'old');

    configureSession({ refresh: async () => Promise.reject(httpError(401)) });
    assert.equal(await refreshSession(), null);
    assert.equal(getSession().status, 'anonymous');
    assert.equal(getAccessToken(), null);
  });

  it('logout clears the session', () => {
    setSession(body('t'));
    clearSession();
    assert.equal(getSession().status, 'anonymous');
    assert.equal(getAccessToken(), null);
  });

  it('never touches web storage', () => {
    const touched = [];
    for (const name of ['localStorage', 'sessionStorage']) {
      Object.defineProperty(globalThis, name, {
        configurable: true,
        get() {
          touched.push(name);
          return { getItem() {}, setItem() {}, removeItem() {} };
        },
      });
    }
    setSession(body('secret-token'));
    clearSession();
    for (const name of ['localStorage', 'sessionStorage']) delete globalThis[name];
    assert.deepEqual(touched, []);
  });
});

describe('server auth errors reach the existing field error lines', () => {
  it('login', () => {
    assert.deepEqual(routeAuthError('login', 'Invalid credentials'), { field: 'pw', step: 0 });
    assert.deepEqual(routeAuthError('login', 'Official ID required'), { field: 'id', step: 0 });
    assert.deepEqual(routeAuthError('login', 'Too many attempts, try again later'), { field: 'pw', step: 0 });
  });

  it('signup jumps to the step that owns the field', () => {
    assert.deepEqual(routeAuthError('signup', 'Official ID already registered', 2), { field: 'id', step: 0 });
    assert.deepEqual(routeAuthError('signup', 'Invalid email', 2), { field: 'email', step: 1 });
    assert.deepEqual(routeAuthError('signup', 'Password must be 8 to 128 characters', 2), { field: 'np', step: 1 });
    assert.deepEqual(routeAuthError('signup', 'Unknown role', 2), { field: 'role', step: 2 });
    assert.deepEqual(routeAuthError('signup', 'Something odd', 1), { field: 'np', step: 1 });
  });
});
