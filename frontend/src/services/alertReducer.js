/**
 * Pure list transitions for the on-screen alert queue. The queue holds server alerts (the frozen
 * shape) plus three presentation fields the console owns: `typed` (characters of the reason
 * revealed), `conf` (confidence bar) and a `stage` the reveal animation moves through.
 *
 * Every function returns the SAME array when nothing changed, and never reorders or re-creates
 * cards it did not touch. That is what keeps scroll position, selection and queue position stable
 * when a patch arrives for an alert that is already on screen.
 */

export const MAX_ALERTS = 6;

const reasonOf = (alert) => (typeof alert?.reason === 'string' ? alert.reason : '');

/** Wraps a server alert for display. `stamp` supplies the clock label when the alert has none. */
export const toDisplayAlert = (alert, stamp) => {
  const reason = reasonOf(alert);
  const revealed = alert.stage && alert.stage !== 'prov';
  return {
    ...alert,
    reason,
    time: alert.time || stamp(),
    stage: alert.stage || 'prov',
    typed: revealed ? reason.length : 0,
    conf: revealed ? alert.confidence : 0,
  };
};

export const hasAlert = (list, id) => list.some((a) => a.id === id);

/** New alerts go to the top (as the queue always did); a known id is ignored. */
export const addAlert = (list, alert) =>
  hasAlert(list, alert.id) ? list : [alert, ...list].slice(0, MAX_ALERTS);

/**
 * Applies an `alert.updated` patch in place. The card keeps its position and its local stage; only
 * the patched fields change. If the reason text changed after it had finished typing, it types out
 * again; if it was mid-type, typing simply carries on.
 */
export const patchAlert = (list, id, patch) => {
  const at = list.findIndex((a) => a.id === id);
  if (at === -1 || !patch) return list;
  const current = list[at];

  const fields = { ...patch };
  // The reveal state and the local stage belong to the console, not the server.
  delete fields.stage;
  delete fields.time;
  delete fields.typed;
  delete fields.conf;
  const next = { ...current, ...fields };

  if (typeof fields.confidence === 'number' && current.stage !== 'prov') next.conf = fields.confidence;

  const reason = reasonOf(next);
  next.reason = reason;
  if (reason !== current.reason) {
    const finished = current.typed >= current.reason.length;
    next.typed = current.stage === 'prov' ? 0 : finished ? 0 : Math.min(current.typed, reason.length);
  }

  const same = Object.keys(next).every((k) => Object.is(next[k], current[k]));
  if (same) return list;
  const copy = list.slice();
  copy[at] = next;
  return copy;
};

/** Local reveal step: provisional -> confirmed. Only moves an alert that is still provisional. */
export const confirmAlert = (list, id) => {
  const at = list.findIndex((a) => a.id === id && a.stage === 'prov');
  if (at === -1) return list;
  const copy = list.slice();
  copy[at] = { ...list[at], stage: 'conf', conf: list[at].confidence };
  return copy;
};

/**
 * Applies an `alert.stage` event. 'sealed' and 'gone' are terminal and always shown; 'conf' only
 * advances a provisional card; 'prov' never moves a card backwards.
 */
export const stageAlert = (list, id, stage) => {
  const at = list.findIndex((a) => a.id === id);
  if (at === -1) return list;
  const current = list[at];
  if (current.stage === stage || current.stage === 'gone') return list;
  if (stage === 'prov') return list;
  if (stage === 'conf' && current.stage !== 'prov') return list;

  const copy = list.slice();
  copy[at] = {
    ...current,
    stage,
    conf: current.confidence,
    typed: stage === 'sealed' ? Math.max(current.typed, 0) : current.typed,
  };
  return copy;
};

export const removeAlert = (list, id) => (hasAlert(list, id) ? list.filter((a) => a.id !== id) : list);

/**
 * Plans how a GET /alerts snapshot reconciles with what the console has already seen.
 *   fresh    server alerts never seen before, oldest first (so admitting them in order leaves the
 *            newest on top, matching the server's newest-first listing)
 *   patches  { id, patch } for alerts already seen, carrying any reason / plate / hash they missed
 * Ids in `skip` were touched by a stream event while the snapshot was in flight, so the older
 * snapshot must not overwrite them.
 */
export const planSnapshot = (snapshot, { seen, skip = new Set() }) => {
  const fresh = [];
  const patches = [];
  for (const serverAlert of snapshot.slice(0, MAX_ALERTS).reverse()) {
    if (!serverAlert?.id || serverAlert.stage === 'gone') continue;
    if (!seen.has(serverAlert.id)) {
      fresh.push(serverAlert);
    } else if (!skip.has(serverAlert.id)) {
      const { reason, plate, hash } = serverAlert;
      patches.push({ id: serverAlert.id, patch: { reason, plate, hash } });
    }
  }
  return { fresh, patches };
};
