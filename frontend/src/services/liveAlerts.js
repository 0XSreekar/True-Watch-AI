// Explicit .js extensions keep these pure modules importable from Node's test runner as well as Vite.
import { isUnreachable } from './network.js';
import { streamEvents } from './sse.js';

export const BACKOFF = { base: 1000, factor: 2, max: 30_000 };

/** Reconnect delay for the nth consecutive failure (0-based): exponential, capped, lightly jittered. */
export const backoffDelay = (attempt, { base, factor, max } = BACKOFF, random = Math.random) => {
  const raw = Math.min(max, base * factor ** attempt);
  // +/-20% so a fleet of consoles does not reconnect in lockstep after a server restart.
  return Math.round(raw * (0.8 + random() * 0.4));
};

const abortableSleep = (ms, signal) =>
  new Promise((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const done = () => {
      clearTimeout(timer);
      signal.removeEventListener('abort', done);
      resolve();
    };
    const timer = setTimeout(done, ms);
    signal.addEventListener('abort', done, { once: true });
  });

/**
 * Keeps the alert queue live. Two delivery paths, never both silent:
 *
 *   stream   GET /api/stream is open; alerts arrive as events the moment the API publishes them.
 *   polling  the stream is down (refused, dropped, silent); the queue is refreshed on a timer
 *            while the stream reconnects with exponential backoff, capped at BACKOFF.max.
 *
 * Callbacks:
 *   onCreated(alert)                     a new alert
 *   onUpdated({ id, patch, alert })      a reason / plate / hash patch for an alert
 *   onStage({ id, stage, alert })        sealed / gone / ...
 *   onSnapshot(list, { skip, initial })  a full GET /alerts reconcile; `skip` is the set of ids a
 *                                        stream event touched while it was in flight
 *   onMode('connecting' | 'stream' | 'polling')
 *
 * Demo driver (`demo` + `simulate`): keeps the existing demo behaviour of asking the API for a
 * fresh detection on a schedule. While the stream is up the answer is ignored, because the same
 * alert arrives over the stream; otherwise the response is delivered directly. If the API cannot be
 * reached at all, `offlineAlert()` (when given) supplies a local sample so the wall never goes quiet.
 *
 * Returns { start, stop, mode }. `stop()` aborts the connection, the timers and every pending wait;
 * after it returns nothing further fires. `start()` is idempotent, so a re-render or a StrictMode
 * double-mount cannot open a second stream.
 */
export const createLiveAlerts = ({
  streamUrl,
  getToken = () => null,
  refreshToken = async () => null,
  fetchSnapshot,
  simulate,
  offlineAlert,
  demo = false,
  demoDelays = [500, 3400, 6600],
  pollMs = 9500,
  idleMs,
  backoff = BACKOFF,
  fetchImpl,
  random = Math.random,
  onCreated = () => {},
  onUpdated = () => {},
  onStage = () => {},
  onSnapshot = () => {},
  onMode = () => {},
}) => {
  let controller = null;
  let mode = 'connecting';
  let timers = [];
  let touched = new Set();
  let snapshotCount = 0;

  const setMode = (next) => {
    if (next === mode) return false;
    mode = next;
    onMode(next);
    return true;
  };

  const alive = () => Boolean(controller) && !controller.signal.aborted;

  const snapshot = async () => {
    if (!fetchSnapshot) return;
    const started = new Set();
    touched = started;
    const initial = snapshotCount === 0;
    try {
      const list = await fetchSnapshot();
      if (!alive() || !Array.isArray(list)) return;
      snapshotCount += 1;
      onSnapshot(list, { skip: started, initial });
    } catch {
      // The next poll or reconnect tries again; the queue already shows what it has.
    }
  };

  const handleFrame = (frame) => {
    let payload;
    try {
      payload = JSON.parse(frame.data);
    } catch {
      return;
    }
    if (!payload || payload.id === undefined) return;
    if (frame.event === 'alert.created') {
      touched.add(payload.id);
      onCreated(payload);
    } else if (frame.event === 'alert.updated') {
      touched.add(payload.id);
      onUpdated(payload);
    } else if (frame.event === 'alert.stage') {
      touched.add(payload.id);
      onStage(payload);
    }
  };

  const tick = async () => {
    if (demo && simulate) {
      try {
        const alert = await simulate();
        if (alive() && alert && mode !== 'stream') onCreated(alert);
      } catch (error) {
        if (alive() && isUnreachable(error) && offlineAlert) onCreated(offlineAlert());
      }
    }
    if (alive() && mode === 'polling') await snapshot();
  };

  const connectLoop = async () => {
    const { signal } = controller;
    let attempt = 0;
    let refreshed = false;

    while (!signal.aborted) {
      try {
        const result = await streamEvents({
          url: streamUrl,
          token: getToken(),
          signal,
          idleMs,
          fetchImpl,
          onOpen: () => {
            attempt = 0;
            refreshed = false;
            setMode('stream');
            snapshot();
          },
          onFrame: handleFrame,
        });
        if (result.reason === 'aborted') return;
        // A clean end (server restart, shutdown event) is still a drop: poll and reconnect.
      } catch (error) {
        if (signal.aborted) return;
        if (error.status === 401 && !refreshed) {
          // Expired access token: refresh once and reconnect straight away.
          refreshed = true;
          const token = await refreshToken();
          if (token) continue;
        }
      }

      // Enter polling on the first drop and refresh the queue right away, not a full interval later.
      if (setMode('polling')) snapshot();
      await abortableSleep(backoffDelay(attempt, backoff, random), signal);
      attempt += 1;
    }
  };

  return {
    start() {
      if (controller) return;
      controller = new AbortController();
      mode = 'connecting';
      onMode(mode);

      if (demo && simulate) demoDelays.forEach((d) => timers.push(setTimeout(tick, d)));
      timers.push(setInterval(tick, pollMs));

      connectLoop();
    },

    stop() {
      if (!controller) return;
      controller.abort();
      timers.forEach((id) => {
        clearTimeout(id);
        clearInterval(id);
      });
      timers = [];
      controller = null;
    },

    mode: () => mode,
  };
};

export default createLiveAlerts;
