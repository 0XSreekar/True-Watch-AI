/** One JSON line per finished request, tagged with the request id. Query strings are not logged. */
export const accessLog = (logger) => (req, res, next) => {
  const started = process.hrtime.bigint();
  res.on('finish', () => {
    logger.info('request', {
      requestId: req.id,
      method: req.method,
      path: req.originalUrl.split('?')[0],
      status: res.statusCode,
      ms: Number((process.hrtime.bigint() - started) / 1000n) / 1000,
      user: req.user?.id,
    });
  });
  next();
};

export default accessLog;
