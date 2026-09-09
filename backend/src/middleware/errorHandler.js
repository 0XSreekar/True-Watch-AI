import config from '../config/index.js';

// eslint-disable-next-line no-unused-vars -- Express identifies error handlers by arity.
export const errorHandler = (err, _req, res, _next) => {
  const status = err.status || 500;
  if (config.env !== 'test') console.error('[error]', err);
  res.status(status).json({
    error: err.message || 'Internal server error',
    ...(config.env === 'development' ? { stack: err.stack } : {}),
  });
};

export default errorHandler;
