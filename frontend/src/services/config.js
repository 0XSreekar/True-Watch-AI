/**
 * Build-time switches (Vite exposes only VITE_* variables to the browser).
 *
 * VITE_DEMO_MODE (default on) keeps this a watchable demo: the console asks the API for scheduled
 * detections and, when the API cannot be reached at all, opens on local sample data. Set it to
 * `false` for a deployment that must show only real detections and must not open without a server.
 */
export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE !== 'false';
