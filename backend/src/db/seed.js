import bcrypt from 'bcryptjs';

import {
  ANALYTICS,
  CAMERAS,
  OPERATOR,
  PLATES,
  POSTS,
  REASONS,
} from '../data/mockData.js';
import { ZERO_HASH, seedFrom, sha256, shortHash } from '../utils/crypto.js';
import { pad, stamp } from '../utils/random.js';
import { blockDigest } from '../repositories/evidence.repository.js';

/**
 * First-boot seeding from data/mockData.js, so a fresh clone demos with no edge service running.
 * Each block is guarded by a seed_state flag: it runs once per database file and never overwrites
 * anything on a later boot, so a restart keeps whatever the live system has written since.
 */

const DEMO_USERS = [
  { id: 'SSB/104382', name: OPERATOR.name, unit: OPERATOR.unit, role: 'Operator', initials: OPERATOR.initials },
  { id: 'SSB/200117', name: 'Insp. A. K. Mehta', unit: 'Sector HQ — Patna Frontier', role: 'Supervisor', initials: 'AM' },
  { id: 'SSB/300245', name: 'Dy. Cmdt. S. Kumar', unit: 'Sector HQ — Patna Frontier', role: 'HQ Analyst', initials: 'SK' },
];

const seedCatalog = (repos) => {
  for (const post of POSTS) repos.posts.create(post);
  for (const camera of CAMERAS) {
    repos.cameras.create({ ...camera, post_id: camera.id.split('-')[0] });
  }
};

/** Five genuinely linked blocks, so /api/evidence/verify walks a real chain even on first boot. */
const seedChain = (repos) => {
  let prev = ZERO_HASH;
  for (let i = 0; i < 5; i += 1) {
    const n = 1041 + i;
    const camera = CAMERAS[i % CAMERAS.length];
    const sealedAt = `2026-08-14T${pad(1 + i)}:${pad(12 + i * 7)}:00.000Z`;
    const clip = sha256('seed-clip', n);
    const row = {
      n,
      evidence_id: `ev_seed_${n}`,
      camera_id: camera.id,
      time_label: `14 AUG · ${pad(1 + i)}:${pad(12 + i * 7)}`,
      display_hash: shortHash(clip, 10, 6),
      clip_sha256: clip,
      prev_sha256: prev,
      clip_path: `seed/ev_seed_${n}.mp4`,
      sealed_at: sealedAt,
    };
    row.block_sha256 = blockDigest(row);
    repos.evidence.insert(row);
    prev = row.block_sha256;
  }
};

/**
 * Historical alerts (origin 'seed') that feed analytics. They match the shape of ANALYTICS in
 * mockData (per-camera counts, a 38/62 day/night split, a rising agreement tail) but are real
 * rows, so the numbers move as live alerts arrive. They never appear in GET /api/alerts.
 */
const seedHistory = (repos, now = new Date()) => {
  const total = ANALYTICS.camBars.reduce((sum, c) => sum + c.v, 0);
  const dayShare = ANALYTICS.dayNight[0].v / 100;
  const dayCount = Math.round(total * dayShare);

  const cameraSequence = ANALYTICS.camBars.flatMap((c) => Array.from({ length: c.v }, () => c.id));
  // Interleave cameras deterministically instead of grouping them. The stride must be coprime
  // with the length for the walk to visit every element exactly once.
  const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
  const stride = [5, 3, 1].find((s) => gcd(s, cameraSequence.length) === 1);
  const ordered = cameraSequence.map((_, i) => cameraSequence[(i * stride) % cameraSequence.length]);

  const entries = ordered.map((cameraId, i) => {
    const isDay = i < dayCount;
    const hour = isDay ? 6 + ((i * 5) % 12) : (18 + ((i * 7) % 12)) % 24;
    const when = new Date(now);
    when.setDate(when.getDate() - (1 + (i % 5)));
    when.setHours(hour, (i * 11) % 60, (i * 17) % 60, 0);
    return { cameraId, when, i };
  });
  entries.sort((a, b) => a.when - b.when);

  const tail = ANALYTICS.agreement;
  entries.forEach(({ cameraId, when, i }, position) => {
    const fromEnd = entries.length - 1 - position;
    const confidence = fromEnd < tail.length ? tail[tail.length - 1 - fromEnd] : 88 + (i % 5);
    const eventId = `seed-${pad(i)}`;
    repos.alerts.insert({
      id: `Aseed${pad(i)}`,
      event_id: eventId,
      camera_id: cameraId,
      stage: 'sealed',
      captured_at: when.toISOString(),
      time_label: stamp(when),
      reason: REASONS[i % REASONS.length],
      confidence,
      plate: i % 3 === 0 ? PLATES[i % PLATES.length] : null,
      hash: shortHash(sha256(eventId)),
      seed: seedFrom(eventId),
      ch_appearance: 1,
      ch_motion: 1,
      origin: 'seed',
      created_at: when.toISOString(),
    });
  });
};

const seedUsers = (repos, config) => {
  const hash = bcrypt.hashSync(config.seedUserPassword, config.bcryptRounds);
  for (const user of DEMO_USERS) repos.users.create({ ...user, password_hash: hash, approved: 1 });
};

export const seedDatabase = (db, repos, config) => {
  const flag = db.prepare('SELECT 1 FROM seed_state WHERE name = ?').pluck();
  const mark = db.prepare('INSERT INTO seed_state (name, seeded_at) VALUES (?, ?)');
  const once = (name, fn) => {
    if (flag.get(name)) return;
    db.transaction(() => {
      fn();
      mark.run(name, new Date().toISOString());
    })();
  };

  once('catalog', () => seedCatalog(repos));
  once('chain', () => seedChain(repos));
  once('history', () => seedHistory(repos));
  if (config.seedDemoUsers && config.seedUserPassword) once('users', () => seedUsers(repos, config));
};

export default seedDatabase;
