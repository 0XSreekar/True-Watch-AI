import createApp from './app.js';
import config, { ConfigError, validateConfig } from './config/index.js';
import { createShutdown } from './lifecycle.js';
import { createLogger } from './utils/logger.js';

const logger = createLogger();

let warnings;
try {
  warnings = validateConfig(config);
} catch (error) {
  if (error instanceof ConfigError) {
    process.stderr.write(`\nTRUE WATCH API cannot start.\n${error.message}\n\nSee backend/.env.example.\n`);
    process.exit(1);
  }
  throw error;
}
warnings.forEach((warning) => logger.warn(warning));

const app = createApp({ config, logger });
const { container } = app.locals;

const server = app.listen(config.port, () => {
  logger.info('TRUE WATCH API listening', {
    url: `http://localhost:${config.port}/api`,
    env: config.env,
    corsOrigins: config.corsOrigins,
    authRequired: config.authRequired,
    edge: config.edgeBaseUrl ?? 'not configured',
  });
});

const shutdown = createShutdown({ server, container, logger });
const stop = (signal) => shutdown(signal).then(() => process.exit(0));
process.on('SIGINT', () => stop('SIGINT'));
process.on('SIGTERM', () => stop('SIGTERM'));

process.on('unhandledRejection', (error) => logger.error('unhandled rejection', { error }));
process.on('uncaughtException', (error) => {
  logger.error('uncaught exception', { error });
  shutdown('uncaughtException').finally(() => process.exit(1));
});
