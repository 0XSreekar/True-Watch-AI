import { createEdgeClient } from './adapters/edge.js';
import { createEvidenceVerifier } from './adapters/evidenceVerifier.js';
import { openDatabase } from './db/index.js';
import { seedDatabase } from './db/seed.js';
import { createGuards } from './middleware/auth.js';
import { createBus } from './realtime/bus.js';
import { createSseHub } from './realtime/sse.js';
import { createRepositories } from './repositories/index.js';
import { createAlertsService } from './services/alerts.service.js';
import { createAnalyticsService } from './services/analytics.service.js';
import { createAuditService } from './services/audit.service.js';
import { createAuthService } from './services/auth.service.js';
import { createCamerasService } from './services/cameras.service.js';
import { createEvidenceService } from './services/evidence.service.js';
import { createFacesService } from './services/faces.service.js';
import { createIngestService } from './services/ingest.service.js';
import { createSearchService } from './services/search.service.js';

/**
 * Composition root. Builds the database, repositories, realtime bus, edge adapter and services
 * once and hands them to the routes. Controllers only ever see services; services only ever see
 * repositories and adapters.
 */
export const createContainer = (config, { log = () => {} } = {}) => {
  const db = openDatabase(config.databasePath);
  const repos = createRepositories(db);
  seedDatabase(db, repos, config);

  const bus = createBus({ onError: (error) => log('error', 'bus subscriber failed', error) });
  const hub = createSseHub({
    bus,
    heartbeatSeconds: config.streamHeartbeatSeconds,
    maxClients: config.streamMaxClients,
    log,
  });
  const transaction = (fn) => db.transaction(fn)();

  const edge = createEdgeClient({
    baseUrl: config.edgeBaseUrl,
    timeoutMs: config.edgeTimeoutMs,
    searchTimeoutMs: config.edgeSearchTimeoutMs,
    log,
  });
  const verifier = createEvidenceVerifier({ repos, edge, config });
  const audit = createAuditService({ repos });

  const services = {
    audit,
    ingest: createIngestService({ repos, bus, transaction, config }),
    alerts: createAlertsService({ repos, bus, edge, audit, transaction, config }),
    cameras: createCamerasService({ repos, audit }),
    evidence: createEvidenceService({ repos, verifier, audit, edge }),
    auth: createAuthService({ repos, config, audit }),
    faces: createFacesService({ repos, audit }),
    search: createSearchService({ repos, edge }),
    analytics: createAnalyticsService({ repos }),
  };

  return {
    config,
    db,
    repos,
    bus,
    hub,
    edge,
    services,
    guards: createGuards({ config, auth: services.auth }),
    close() {
      hub.closeAll();
      if (db.open) db.close();
    },
  };
};

export default createContainer;
