import api from './apiClient';
import { clearSession, refreshSession, setSession } from './session';

/**
 * Both credential calls keep their original request and response shapes:
 *   login    -> { ok, token, expiresIn, user }
 *   register -> { ok, pendingApproval, token, expiresIn, user }
 * A successful answer also starts the in-memory session; the refresh token stays in the httpOnly
 * cookie the server sets. A failure rejects with an ApiError whose message is the server's text.
 */
export const login = async (credentials) => {
  const session = await api.post('/auth/login', credentials);
  setSession(session);
  return session;
};

export const register = async (payload) => {
  const session = await api.post('/auth/register', payload);
  setSession(session);
  return session;
};

/** Ends the session on the server (revokes the refresh cookie) and always clears it locally. */
export const logout = async () => {
  try {
    await api.post('/auth/logout');
  } catch {
    // The local session must end even if the server cannot be told.
  } finally {
    clearSession();
  }
};

/** Exchanges the refresh cookie for a new access token; resolves null when there is no session. */
export const refresh = () => refreshSession();

export const fetchMe = () => api.get('/auth/me');

export const fetchUnits = () => api.get('/auth/units');

export const fetchRoles = () => api.get('/auth/roles');
