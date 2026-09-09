import { CAMERAS, PLATES, REASONS, SHIFT_ALARM_BUDGET } from '../data/mockData.js';
import { chance, hex, pick, stamp } from '../utils/random.js';

/**
 * Alert lifecycle: provisional (~30 ms, both channels agree) →
 * confirmed (~3 s, VLM judge writes the reason) → sealed (hash-chained evidence).
 *
 * Real system: fed by the edge pipeline over a queue/socket and persisted.
 * Here it is an in-memory store seeded on demand.
 */
const store = {
  alerts: [],
  budgetUsed: 7,
};

const MAX_ALERTS = 6;

/** Produce one plausible alert. The stage machine is driven by the client for the demo. */
export const simulateAlert = () => {
  const camera = pick(CAMERAS);
  const alert = {
    id: `A${Date.now()}${Math.floor(Math.random() * 99)}`,
    camera,
    stage: 'prov',
    time: stamp(),
    reason: pick(REASONS),
    confidence: 88 + Math.floor(Math.random() * 10),
    plate: chance(0.45) ? pick(PLATES) : null,
    hash: `${hex(8)}…${hex(6)}`,
    seed: Math.random() * 100,
    channels: { appearance: true, motion: true },
  };
  store.alerts = [alert, ...store.alerts].slice(0, MAX_ALERTS);
  store.budgetUsed = Math.min(SHIFT_ALARM_BUDGET, store.budgetUsed + 1);
  return alert;
};

export const listAlerts = () => store.alerts;

export const getBudget = () => ({
  used: store.budgetUsed,
  total: SHIFT_ALARM_BUDGET,
  near: store.budgetUsed >= SHIFT_ALARM_BUDGET - 3,
});

const setStage = (id, stage) => {
  const alert = store.alerts.find((a) => a.id === id);
  if (!alert) return null;
  alert.stage = stage;
  return alert;
};

/** Operator acted: the clip is sealed into the evidence chain. */
export const actOnAlert = (id) => setStage(id, 'sealed');

export const dismissAlert = (id) => {
  const alert = setStage(id, 'gone');
  if (alert) store.alerts = store.alerts.filter((a) => a.id !== id);
  return alert;
};
