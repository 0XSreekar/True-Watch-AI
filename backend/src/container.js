import { openDatabase } from './db/index.js';
import { seedDatabase } from './db/seed.js';
import { createBus } from './realtime/bus.js';
import { createSseHub } from './realtime/sse.js';
import { createRepositories } from './repositories/index.js';
import { createIngestService } from './services/ingest.service.js';

/**
 * Composition root. Builds the database, repositories, realtime bus and services once and hands
 * them to the routes. Controllers only ever see services; services only ever see repositories.
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

  const services = {
    ingest: createIngestService({ repos, bus, transaction, config }),
  };

  return {
    config,
    db,
    repos,
    bus,
    hub,
    services,
    close() {
      hub.closeAll();
      if (db.open) db.close();
    },
  };
};

export default createContainer;
