/**
 * Global error handler. The client only ever receives `{ error: "<safe message>" }` (plus the
 * documented `details` of a validation failure). Stack traces, driver messages and file paths stay
 * in the server log, keyed by the request id, in every environment.
 */
const BODY_PARSER_ERRORS = {
  'entity.parse.failed': [400, 'Malformed JSON body'],
  'entity.too.large': [413, 'Request body too large'],
  'encoding.unsupported': [415, 'Unsupported content encoding'],
  'charset.unsupported': [415, 'Unsupported charset'],
};

export const createErrorHandler =
  ({ config, log = () => {} } = {}) =>
  // eslint-disable-next-line no-unused-vars -- Express identifies error handlers by arity.
  (err, req, res, next) => {
    if (res.headersSent) return res.end();

    let status = 500;
    let message = 'Internal server error';
    let extra = {};

    if (err?.type && BODY_PARSER_ERRORS[err.type]) {
      [status, message] = BODY_PARSER_ERRORS[err.type];
    } else if (err?.expose === true && Number.isInteger(err.status)) {
      status = err.status;
      message = err.message;
      extra = err.extra || {};
    }

    if (status >= 500 && config?.env !== 'test') {
      log('error', 'unhandled error', { requestId: req.id, path: req.path, error: err });
    }
    return res.status(status).json({ error: message, ...extra });
  };

export const errorHandler = createErrorHandler();

export default errorHandler;
