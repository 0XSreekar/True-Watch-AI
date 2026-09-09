const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

const request = async (path, { method = 'GET', body, signal, headers } = {}) => {
  const response = await fetch(`${BASE_URL}${path}`, {
    method,
    signal,
    headers: { ...(body ? { 'Content-Type': 'application/json' } : {}), ...headers },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(payload?.error || `Request failed (${response.status})`, response.status);
  }
  // Every endpoint answers { data: … }; unwrap it here so callers stay clean.
  return payload?.data ?? payload;
};

export const api = {
  get: (path, options) => request(path, { ...options, method: 'GET' }),
  post: (path, body, options) => request(path, { ...options, method: 'POST', body }),
  put: (path, body, options) => request(path, { ...options, method: 'PUT', body }),
  del: (path, options) => request(path, { ...options, method: 'DELETE' }),
};

/**
 * This is a demo: the UI must stay watchable even with the API down.
 * Wraps a call so a failure logs once and yields the supplied fallback.
 */
export const withFallback = async (promise, fallback, label = 'request') => {
  try {
    return await promise;
  } catch (error) {
    if (error.name !== 'AbortError') {
      console.warn(`[api] ${label} failed, using local sample data:`, error.message);
    }
    return fallback;
  }
};

export { ApiError, BASE_URL };
export default api;
