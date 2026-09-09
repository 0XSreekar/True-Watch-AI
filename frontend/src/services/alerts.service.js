import api from './apiClient';

export const fetchAlerts = () => api.get('/alerts');

/** Asks the API for one freshly detected event (mock pipeline output). */
export const simulateAlert = () => api.post('/alerts/simulate');

export const actOnAlert = (id) => api.post(`/alerts/${id}/act`);

export const dismissAlert = (id) => api.post(`/alerts/${id}/dismiss`);
