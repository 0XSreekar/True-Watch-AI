import api from './apiClient';

export const fetchCameras = (sector) =>
  api.get(`/cameras${sector && sector !== 'all' ? `?sector=${encodeURIComponent(sector)}` : ''}`);

export const fetchPosts = () => api.get('/cameras/posts');

export const fetchSectors = () => api.get('/cameras/sectors');
