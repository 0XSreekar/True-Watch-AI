import api from './apiClient';

export const login = (credentials) => api.post('/auth/login', credentials);

export const register = (payload) => api.post('/auth/register', payload);

export const fetchUnits = () => api.get('/auth/units');

export const fetchRoles = () => api.get('/auth/roles');
