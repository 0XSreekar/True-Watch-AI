import api from './apiClient';

export const fetchChain = () => api.get('/evidence/chain');

export const verifyChain = () => api.post('/evidence/verify');
