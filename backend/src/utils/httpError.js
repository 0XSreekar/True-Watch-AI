/** An error the API is willing to show to a client: a status, a safe message and optional extras. */
export class HttpError extends Error {
  constructor(status, message, extra = {}) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.expose = true;
    this.extra = extra;
  }
}

export const badRequest = (message, extra) => new HttpError(400, message, extra);
export const unauthorized = (message = 'Authentication required', extra) => new HttpError(401, message, extra);
export const forbidden = (message = 'Forbidden', extra) => new HttpError(403, message, extra);
export const notFound = (message = 'Not found', extra) => new HttpError(404, message, extra);
export const conflict = (message, extra) => new HttpError(409, message, extra);
