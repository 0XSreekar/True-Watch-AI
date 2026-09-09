import { ANALYTICS } from '../data/mockData.js';

/** Real system: aggregate from the event log. */
export const getAnalytics = () => ANALYTICS;

/** Rolling tracks-per-minute window, refreshed as the console polls. */
export const nextTrafficSample = () => 3 + Math.round(Math.random() * 11);
