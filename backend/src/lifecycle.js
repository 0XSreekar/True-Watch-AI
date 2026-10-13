/**
 * Graceful shutdown: stop taking connections, end every server-sent stream (they never finish on
 * their own), let in-flight requests drain, then close the database. A hard deadline guarantees
 * the process exits even if a client refuses to hang up.
 */
export const createShutdown = ({ server, container, logger, timeoutMs = 10_000 }) => {
  let inFlight = null;

  const run = (reason) => {
    if (inFlight) return inFlight;
    logger.info('shutting down', { reason });

    inFlight = new Promise((resolve) => {
      const deadline = setTimeout(() => {
        logger.warn('shutdown deadline reached, closing remaining connections');
        server.closeAllConnections?.();
      }, timeoutMs);
      deadline.unref();

      server.close(() => {
        clearTimeout(deadline);
        try {
          container.close();
        } catch (error) {
          logger.error('error while closing the database', { error });
        }
        logger.info('shutdown complete');
        resolve();
      });

      container.hub.closeAll();
      server.closeIdleConnections?.();
    });
    return inFlight;
  };

  return run;
};

export default createShutdown;
