import cors from 'cors';
import express from 'express';
import morgan from 'morgan';

import { createContainer } from './container.js';
import defaultConfig from './config/index.js';
import { createErrorHandler } from './middleware/errorHandler.js';
import notFound from './middleware/notFound.js';
import { createIngestRoutes } from './routes/ingest.routes.js';
import { createStreamRoutes } from './routes/stream.routes.js';
import routes from './routes/index.js';

/**
 * Builds one API instance. `config` and `container` are injectable so tests can run several
 * isolated instances (each with its own database) in one process.
 */
export const createApp = ({ config = defaultConfig, container = createContainer(config) } = {}) => {
  const app = express();

  app.use(cors({ origin: config.corsOrigins, credentials: true }));
  if (config.env !== 'test') app.use(morgan('dev'));

  // The ingest router authenticates before it parses, so it is mounted ahead of the JSON parser.
  app.use('/api/ingest', createIngestRoutes(container));

  app.use('/api/stream', createStreamRoutes(container));

  app.use(express.json());
  app.use('/api', routes);

  app.use(notFound);
  app.use(createErrorHandler({ config, log: (level, message, meta) => console.error(`[${level}] ${message}`, meta?.error ?? '') }));

  app.locals.container = container;
  app.locals.close = () => container.close();
  return app;
};

export default createApp;
