import { createAlertsRepository } from './alerts.repository.js';
import { createAuditRepository } from './audit.repository.js';
import {
  createCameraSettingsRepository,
  createCamerasRepository,
  createPostsRepository,
} from './cameras.repository.js';
import { createDailyRootsRepository, createEvidenceRepository } from './evidence.repository.js';
import { createEventLedgerRepository, createNonceRepository } from './ingest.repository.js';
import { createTracksRepository } from './tracks.repository.js';
import { createRefreshTokensRepository, createUsersRepository } from './users.repository.js';

/** One place that binds every repository to a database handle. */
export const createRepositories = (db) => ({
  cameras: createCamerasRepository(db),
  cameraSettings: createCameraSettingsRepository(db),
  posts: createPostsRepository(db),
  alerts: createAlertsRepository(db),
  tracks: createTracksRepository(db),
  evidence: createEvidenceRepository(db),
  dailyRoots: createDailyRootsRepository(db),
  users: createUsersRepository(db),
  refreshTokens: createRefreshTokensRepository(db),
  audit: createAuditRepository(db),
  nonces: createNonceRepository(db),
  eventLedger: createEventLedgerRepository(db),
});

export default createRepositories;
