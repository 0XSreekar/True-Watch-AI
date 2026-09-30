/**
 * In-memory session store. The 15-minute access token lives here and nowhere else: never in
 * localStorage, sessionStorage or a cookie the page can read. The refresh token is an httpOnly
 * cookie the browser holds and this code never sees.
 *
 * Status values:
 *   unknown        nothing decided yet (page just loaded, refresh not tried)
 *   authenticated  a valid access token is held
 *   anonymous      the server said there is no session
 *   offline        the API could not be reached, so nothing could be verified; the console opens
 *                  on its local sample data (demo insurance) and no token is held
 *
 * The transport is injected with `configureSession`, which keeps this file free of browser
 * globals and easy to test.
 */
import { isUnreachable } from './network.js';

const REFRESH_LEAD_MS = 60_000;

let refresher = null;
let state = { status: 'unknown', token: null, user: null };
let refreshTimer = null;
let inFlight = null;
let boot = null;
const listeners = new Set();

const publish = (next) => {
  state = next;
  listeners.forEach((listener) => listener());
};

/** Injects `refresh: () => Promise<{ token, expiresIn, user }>` (rejects with an error carrying `.status`). */
export const configureSession = ({ refresh }) => {
  refresher = refresh;
};

export const getSession = () => state;
export const getAccessToken = () => state.token;

/** For useSyncExternalStore. */
export const subscribeSession = (listener) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

const scheduleRefresh = (expiresInSeconds) => {
  clearTimeout(refreshTimer);
  refreshTimer = null;
  const ms = Number(expiresInSeconds) * 1000;
  if (!Number.isFinite(ms) || ms <= 0) return;
  const delay = Math.max(ms - REFRESH_LEAD_MS, ms / 2);
  refreshTimer = setTimeout(() => {
    // A failed proactive refresh is not fatal: the next 401 retries the same call.
    refreshSession().catch(() => {});
  }, delay);
  // Never keep a Node process (tests) alive for this.
  refreshTimer.unref?.();
};

/** Stores a login/register/refresh body: `{ token, expiresIn, user }`. */
export const setSession = (body) => {
  if (!body?.token) return state;
  publish({ status: 'authenticated', token: body.token, user: body.user ?? null });
  scheduleRefresh(body.expiresIn);
  return state;
};

export const clearSession = () => {
  clearTimeout(refreshTimer);
  refreshTimer = null;
  publish({ status: 'anonymous', token: null, user: null });
};

/** The API is unreachable and nobody is signed in: let the console open on its sample data. */
export const markOffline = () => {
  if (state.status === 'authenticated') return state;
  clearTimeout(refreshTimer);
  publish({ status: 'offline', token: null, user: null });
  return state;
};

/**
 * Trades the refresh cookie for a new access token. Concurrent callers share one request, which
 * matters because the server rotates the cookie: two parallel refreshes would trip its
 * reuse detection and revoke the whole session.
 * Resolves with the new token, or null when there is no session (401) or the API is unreachable.
 */
export const refreshSession = () => {
  if (!refresher) return Promise.resolve(null);
  if (inFlight) return inFlight;

  inFlight = (async () => {
    try {
      const body = await refresher();
      setSession(body);
      return state.token;
    } catch (error) {
      if (isUnreachable(error)) return null;
      clearSession();
      return null;
    } finally {
      inFlight = null;
    }
  })();
  return inFlight;
};

/**
 * Runs once on page load: asks the server whether the refresh cookie still stands. A network
 * failure is not a "no": it settles as `offline` so the demo console stays watchable.
 */
export const bootSession = () => {
  if (state.status !== 'unknown') return Promise.resolve(state);
  if (boot) return boot;
  boot = (async () => {
    const token = await refreshSession();
    if (!token && state.status === 'unknown') markOffline();
    return state;
  })().finally(() => {
    boot = null;
  });
  return boot;
};

/**
 * Route guard decision for /console, kept pure so it can be tested without React.
 * 'wait' while the first check is running, 'redirect' means send the user to /login.
 */
export const guardDecision = (status) => {
  if (status === 'authenticated' || status === 'offline') return 'allow';
  if (status === 'anonymous') return 'redirect';
  return 'wait';
};

/** Test hook: puts the module back to its initial state. */
export const resetSession = () => {
  clearTimeout(refreshTimer);
  refreshTimer = null;
  inFlight = null;
  boot = null;
  refresher = null;
  state = { status: 'unknown', token: null, user: null };
  listeners.clear();
};
