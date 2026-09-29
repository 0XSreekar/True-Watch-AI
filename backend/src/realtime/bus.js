/**
 * In-process publish/subscribe for alert lifecycle events. Producers (ingest, alert actions)
 * publish; consumers (the SSE hub) subscribe. A misbehaving subscriber can never break a
 * producer: listener errors are caught and reported, not thrown.
 *
 * Event types: 'alert.created', 'alert.updated', 'alert.stage'.
 */
export const createBus = ({ onError = () => {} } = {}) => {
  const listeners = new Set();

  return {
    publish(type, payload) {
      for (const listener of [...listeners]) {
        try {
          listener({ type, payload });
        } catch (error) {
          onError(error);
        }
      }
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    size: () => listeners.size,
  };
};

export default createBus;
