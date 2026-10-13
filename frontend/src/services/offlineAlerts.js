/**
 * Sample detections for the demo console when the API cannot be reached at all, so the alert queue
 * keeps moving instead of freezing. They use the same shape as the API's alerts and are clearly
 * unsigned: the hash reads "offline" so nothing here can be mistaken for real evidence.
 * Used only in demo mode (see config.js); a real deployment never fabricates alerts.
 */
const REASONS = [
  'Two people moving along the fence line, one carrying a long object.',
  'A group of five crossing the dry channel in single file.',
  'Vehicle stopped at the culvert approach with its lights off.',
  'One person loitering near the pillar for several minutes.',
];

let counter = 0;

const pick = (list, random) => list[Math.floor(random() * list.length)];

export const createOfflineAlert = (cameras, random = Math.random) => {
  counter += 1;
  const camera = pick(cameras, random);
  return {
    id: `OFF${Date.now()}${counter}`,
    camera: { id: camera.id, name: camera.name, sector: camera.sector, ir: camera.ir, road: camera.road },
    stage: 'prov',
    reason: pick(REASONS, random),
    confidence: 88 + Math.floor(random() * 10),
    plate: null,
    hash: 'offline · unsigned',
    seed: random() * 100,
    channels: { appearance: true, motion: true },
    offline: true,
  };
};

export default createOfflineAlert;
