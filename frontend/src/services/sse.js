/**
 * Server-sent events over fetch().
 *
 * EventSource cannot send an Authorization header, and GET /api/stream needs one whenever the
 * backend runs with AUTH_REQUIRED=true, so the stream is read with fetch + ReadableStream and
 * parsed here. It behaves the same with or without a token.
 */

/**
 * Incremental parser for the text/event-stream format. Feed it decoded text in arbitrary chunks;
 * it calls `onFrame({ id, event, data, retry })` per dispatched event and `onComment(text)` for
 * `: comment` lines (the server's heartbeat).
 */
export const createSseParser = ({ onFrame, onComment } = {}) => {
  let buffer = '';
  let data = [];
  let event = '';
  let id;
  let retry;

  const dispatch = () => {
    if (data.length || event) {
      onFrame?.({ id, event: event || 'message', data: data.join('\n'), retry });
    }
    data = [];
    event = '';
    retry = undefined;
  };

  const handleLine = (line) => {
    if (line === '') return dispatch();
    if (line.startsWith(':')) return onComment?.(line.slice(1).trim());
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'data') data.push(value);
    else if (field === 'event') event = value;
    else if (field === 'id') id = value;
    else if (field === 'retry' && /^\d+$/.test(value)) retry = Number(value);
    return undefined;
  };

  return {
    feed(text) {
      buffer += text;
      let start = 0;
      for (let i = 0; i < buffer.length; i += 1) {
        const c = buffer[i];
        if (c !== '\n' && c !== '\r') continue;
        // A trailing CR might be the first half of CRLF: wait for the next chunk.
        if (c === '\r' && i === buffer.length - 1) break;
        handleLine(buffer.slice(start, i));
        if (c === '\r' && buffer[i + 1] === '\n') i += 1;
        start = i + 1;
      }
      buffer = buffer.slice(start);
    },
  };
};

export class StreamError extends Error {
  constructor(message, { status, cause } = {}) {
    super(message);
    this.name = 'StreamError';
    this.status = status;
    if (cause) this.cause = cause;
  }
}

/**
 * Opens one stream connection and resolves when it ends. Rejects with a StreamError (carrying the
 * HTTP `status` when there was one) if the connection cannot be made, drops, or goes silent for
 * longer than `idleMs` (the server sends a heartbeat every 15 s, so 45 s means a dead link).
 * Aborting `signal` ends it quietly with reason 'aborted'.
 *
 * @returns {Promise<{ reason: 'closed' | 'shutdown' | 'aborted' }>}
 */
export const streamEvents = async ({
  url,
  token,
  signal,
  idleMs = 45_000,
  fetchImpl = globalThis.fetch,
  onOpen,
  onFrame,
  onComment,
}) => {
  if (signal?.aborted) return { reason: 'aborted' };

  const controller = new AbortController();
  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort, { once: true });

  let idle = false;
  let idleTimer = null;
  const armIdle = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      idle = true;
      controller.abort();
    }, idleMs);
  };

  let shutdown = false;
  try {
    armIdle();
    const response = await fetchImpl(url, {
      method: 'GET',
      signal: controller.signal,
      headers: { Accept: 'text/event-stream', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    });
    if (!response.ok) throw new StreamError(`Stream refused (${response.status})`, { status: response.status });
    if (!response.body) throw new StreamError('Stream has no body');

    onOpen?.();
    const parser = createSseParser({
      onComment: (text) => onComment?.(text),
      onFrame: (frame) => {
        if (frame.event === 'shutdown') shutdown = true;
        onFrame?.(frame);
      },
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      armIdle();
      parser.feed(decoder.decode(value, { stream: true }));
    }
    return { reason: shutdown ? 'shutdown' : 'closed' };
  } catch (error) {
    if (signal?.aborted) return { reason: 'aborted' };
    if (idle) throw new StreamError('Stream went silent', { cause: error });
    if (error instanceof StreamError) throw error;
    throw new StreamError(error?.message || 'Stream failed', { cause: error });
  } finally {
    clearTimeout(idleTimer);
    signal?.removeEventListener('abort', onAbort);
    controller.abort();
  }
};
