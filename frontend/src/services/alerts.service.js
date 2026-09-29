import api, { BASE_URL } from './apiClient';

/**
 * Request/response shapes below are frozen and unchanged: `fetchAlerts` resolves to the alert
 * array, `simulateAlert` to one alert, `actOnAlert` / `dismissAlert` to the updated alert.
 */
export const fetchAlerts = () => api.get('/alerts');

/** Asks the API for one freshly detected event (mock pipeline output). */
export const simulateAlert = () => api.post('/alerts/simulate');

export const actOnAlert = (id) => api.post(`/alerts/${id}/act`);

export const dismissAlert = (id) => api.post(`/alerts/${id}/dismiss`);

/** Server-sent event stream of alert.created / alert.updated / alert.stage. */
export const streamUrl = () => `${BASE_URL}/stream`;

/**
 * Demo mode keeps the scheduled "ask the API for a detection" driver running, which is how the
 * console shows alerts while the edge service is asleep. Set VITE_DEMO_MODE=false for a deployment
 * fed only by real detections.
 */
export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE !== 'false';
