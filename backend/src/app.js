import cookieParser from 'cookie-parser';
import cors from 'cors';
import express from 'express';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';

import defaultConfig from './config/index.js';
import { createContainer } from './container.js';
import { accessLog } from './middleware/accessLog.js';
import { createErrorHandler } from './middleware/errorHandler.js';
import notFound from './middleware/notFound.js';
import { requestId } from './middleware/requestId.js';
import { createRoutes } from './routes/index.js';
import { createIngestRoutes } from './routes/ingest.routes.js';
import { createStreamRoutes } from './routes/stream.routes.js';
import { createLogger } from './utils/logger.js';

/**
 * Builds one API instance. `config`, `logger` and `container` are injectable so tests can run
 * several isolated instances (each with its own database) in one process.
 */
export const createApp = ({
  config = defaultConfig,
  logger = createLogger({ silent: config.env === 'test' }),
  container = createContainer(config, { log: logger.log }),
} = {}) => {
  const app = express();

  app.set('trust proxy', config.trustProxy);
  app.disable('etag');

  app.use(requestId);
  if (config.logRequests) app.use(accessLog(logger));

  app.use(
    helmet({
      // The API is called cross-origin by the console; CORS below is the gate, not CORP.
      crossOriginResourcePolicy: { policy: 'cross-origin' },
      // Only meaningful (and only safe to pin) once the API is served over HTTPS.
      strictTransportSecurity: config.production,
    }),
  );

  // Explicit allow-list. A request with no Origin (curl, the edge service) is not a browser
  // cross-origin call and is unaffected; an unlisted origin simply receives no CORS headers.
  app.use(
    cors({
      origin: (origin, callback) => callback(null, !origin || config.corsOrigins.includes(origin)),
      credentials: true,
      maxAge: 600,
    }),
  );

  // API answers carry per-operator data and must never be cached by a shared proxy.
  app.use('/api', (_req, res, next) => {
    res.set('Cache-Control', 'no-store');
    next();
  });

  // Ingest authenticates and throttles itself before parsing, so it is mounted ahead of the
  // global limiter and the JSON parser.
  app.use('/api/ingest', createIngestRoutes(container));

  app.use(
    '/api',
    rateLimit({
      windowMs: 60_000,
      limit: config.apiRateLimitPerMinute,
      standardHeaders: 'draft-7',
      legacyHeaders: false,
      message: { error: 'Too many requests' },
    }),
  );

  app.use('/api/stream', createStreamRoutes(container));

  app.use(express.json({ limit: config.bodyLimit }));
  app.use(cookieParser());
  app.use('/api', createRoutes(container));

  app.use(notFound);
  app.use(createErrorHandler({ config, log: logger.log }));

  app.locals.container = container;
  app.locals.logger = logger;
  app.locals.close = () => container.close();
  return app;
};

export default createApp;
