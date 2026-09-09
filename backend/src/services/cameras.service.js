import { CAMERAS, POSTS } from '../data/mockData.js';

/**
 * Camera / post registry.
 * Real system: read from the device inventory table and the BOP master list.
 */
export const listCameras = ({ sector } = {}) => {
  if (!sector || sector === 'all') return CAMERAS;
  return CAMERAS.filter((c) => c.sector === sector);
};

export const getCamera = (id) => CAMERAS.find((c) => c.id === id) || null;

export const listPosts = () => POSTS;

export const listSectors = () => [...new Set(CAMERAS.map((c) => c.sector))];
