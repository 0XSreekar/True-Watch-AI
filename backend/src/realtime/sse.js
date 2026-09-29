/**
 * Server-sent events hub for GET /api/stream.
 *
 * Wire format, one frame per bus event:
 *   id: <monotonic integer>
 *   event: alert.created | alert.updated | alert.stage
 *   data: <one line of JSON>
 * plus a `: heartbeat` comment every STREAM_HEARTBEAT_SECONDS so idle-timeout proxies keep the
 * connection open. A reader that stops draining is dropped instead of buffering without bound.
 */
const MAX_BUFFERED_BYTES = 1024 * 1024;

export const createSseHub = ({ bus, heartbeatSeconds = 15, maxClients = 100, log = () => {} }) => {
  const clients = new Set();
  let nextId = 1;
  let closed = false;

  const drop = (client) => {
    if (!clients.delete(client)) return;
    try {
      client.res.end();
    } catch {
      // The socket is already gone; nothing more to release.
    }
  };

  const send = (client, frame) => {
    if (client.res.writableEnded || client.res.destroyed) return drop(client);
    client.res.write(frame);
    if (client.res.writableLength > MAX_BUFFERED_BYTES) {
      log('warn', 'dropping slow stream client');
      client.res.destroy();
      return drop(client);
    }
    return undefined;
  };

  const unsubscribe = bus.subscribe(({ type, payload }) => {
    if (!clients.size) return;
    const frame = `id: ${nextId}\nevent: ${type}\ndata: ${JSON.stringify(payload)}\n\n`;
    nextId += 1;
    for (const client of [...clients]) send(client, frame);
  });

  const heartbeat = setInterval(() => {
    for (const client of [...clients]) send(client, ': heartbeat\n\n');
  }, heartbeatSeconds * 1000);
  heartbeat.unref();

  return {
    /** Express handler. Registers the client and returns; the response stays open. */
    handler(req, res) {
      if (closed) return res.status(503).json({ error: 'Server is shutting down' });
      if (clients.size >= maxClients) return res.status(503).json({ error: 'Too many stream clients' });

      res.status(200).set({
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
        'X-Accel-Buffering': 'no',
      });
      req.socket.setKeepAlive(true);
      req.socket.setNoDelay(true);
      req.socket.setTimeout(0);
      res.flushHeaders();
      res.write('retry: 3000\n\n: connected\n\n');

      const client = { res };
      clients.add(client);
      const cleanup = () => {
        clients.delete(client);
      };
      req.on('close', cleanup);
      res.on('close', cleanup);
      res.on('error', cleanup);
      return undefined;
    },

    size: () => clients.size,

    /** Ends every open stream and stops publishing. Called on shutdown. */
    closeAll() {
      closed = true;
      unsubscribe();
      clearInterval(heartbeat);
      for (const client of [...clients]) {
        try {
          client.res.write('event: shutdown\ndata: {}\n\n');
        } catch {
          // Client already disconnected.
        }
        drop(client);
      }
    },
  };
};

export default createSseHub;
