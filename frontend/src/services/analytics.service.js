import api from './apiClient';

export const fetchAnalytics = () => api.get('/analytics');

export const fetchTrafficSample = () => api.get('/analytics/traffic');
