import { clearSession, configureSession, getAccessToken, refreshSession } from './session';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/+$/, '');

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/** Only the auth routes read or write the refresh cookie, so only they send credentials. */
const isAuthPath = (path) => path.startsWith('/auth/');

const send = async (path, { method = 'GET', body, signal, headers, token } = {}) => {
  const response = await fetch(`${BASE_URL}${path}`, {
    method,
    signal,
    ...(isAuthPath(path) ? { credentials: 'include' } : {}),
    headers: {
      ...(body ? { 'Content-Type': 'application/json' } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...headers,
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(payload?.error || `Request failed (${response.status})`, response.status);
  }
  // Every endpoint answers { data: … }; unwrap it here so callers stay clean.
  return payload?.data ?? payload;
};

/**
 * Sends the bearer token when one is held. A 401 on a console call triggers exactly one refresh
 * (shared between concurrent calls) and one retry; if the refresh fails the session is cleared,
 * which is what sends the operator back to /login. Credential routes are never retried: a 401
 * from /auth/login means "wrong password", not "expired token".
 */
const request = async (path, options = {}) => {
  try {
    return await send(path, { ...options, token: getAccessToken() });
  } catch (error) {
    if (error.status !== 401 || isAuthPath(path)) throw error;

    const token = await refreshSession();
    if (!token) {
      clearSession();
      throw error;
    }
    return send(path, { ...options, token });
  }
};

export const api = {
  get: (path, options) => request(path, { ...options, method: 'GET' }),
  post: (path, body, options) => request(path, { ...options, method: 'POST', body }),
  put: (path, body, options) => request(path, { ...options, method: 'PUT', body }),
  del: (path, options) => request(path, { ...options, method: 'DELETE' }),
};

// The session store refreshes through the same transport, without the retry wrapper.
configureSession({ refresh: () => send('/auth/refresh', { method: 'POST' }) });

// Labels that are currently failing, so a dead API logs each one once rather than on every tick.
const failing = new Set();

/**
 * This is a demo: the UI must stay watchable even with the API down.
 * Wraps a call so a failure logs once (until it succeeds again) and yields the supplied fallback.
 * The fallback is also used for an error the API answered with, never only for network failures,
 * so a 500 or an expired session cannot blank the console.
 */
export const withFallback = async (promise, fallback, label = 'request') => {
  try {
    const value = await promise;
    failing.delete(label);
    return value;
  } catch (error) {
    if (error.name !== 'AbortError' && !failing.has(label)) {
      failing.add(label);
      console.warn(`[api] ${label} failed, using local sample data:`, error.message);
    }
    return fallback;
  }
};

export { ApiError, BASE_URL };
export default api;
