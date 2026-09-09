import api from './apiClient';

export const fetchPlaceholders = () => api.get('/search/placeholders');

export const runSearch = (query) => api.post('/search', { query });
