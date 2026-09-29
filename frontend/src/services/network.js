/**
 * True when a failure means "could not reach a working API" (network error, gateway or server
 * error) rather than "the API answered and said no" (a 4xx). The distinction decides between
 * quietly using local sample data and showing the server's message to the operator.
 */
export const isUnreachable = (error) => {
  if (!error || error.name === 'AbortError') return false;
  const status = Number(error.status);
  return !status || status >= 500;
};

export default isUnreachable;
