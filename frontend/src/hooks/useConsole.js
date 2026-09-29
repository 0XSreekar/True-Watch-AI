import { useCallback, useEffect, useRef, useState } from 'react';

import {
  FALLBACK_ANALYTICS,
  FALLBACK_CAMERAS,
  FALLBACK_CHAIN,
  FALLBACK_PLACEHOLDERS,
  FALLBACK_POSTS,
} from '../data/fallbacks';
import { alertsApi, analyticsApi, camerasApi, evidenceApi, searchApi, withFallback } from '../services';
import { addAlert, confirmAlert, patchAlert, planSnapshot, removeAlert, stageAlert, toDisplayAlert } from '../services/alertReducer';
import { createLiveAlerts } from '../services/liveAlerts';
import { stamp } from '../utils/format';

const SPAWN_DELAYS = [500, 3400, 6600];
const SPAWN_INTERVAL = 9500;
const CONFIRM_AFTER = 2400;
const TYPE_TICK = 22;

/**
 * All operator-console state in one place: the live camera set, the alert
 * lifecycle, footage search, the evidence chain and the analytics strip.
 *
 * Alerts arrive over the API's server-sent event stream (`GET /api/stream`);
 * if that stream is unavailable the console keeps itself current by polling
 * while it reconnects with backoff (see services/liveAlerts.js). The two-stage
 * reveal — provisional badge, then the judge's reason typing out — is animated
 * here so it stays smooth regardless of network timing. A later reason patch
 * (`alert.updated`) changes the card in place without moving it.
 */
export const useConsole = () => {
  const [cameras, setCameras] = useState(FALLBACK_CAMERAS);
  const [posts, setPosts] = useState(FALLBACK_POSTS);
  const [analytics, setAnalytics] = useState(FALLBACK_ANALYTICS);
  const [chain, setChain] = useState(FALLBACK_CHAIN);
  const [placeholders, setPlaceholders] = useState(FALLBACK_PLACEHOLDERS);

  const [sector, setSector] = useState('all');
  const [expanded, setExpanded] = useState(-1);
  const [railOpen, setRailOpen] = useState(false);

  const [alerts, setAlerts] = useState([]);
  const [budget, setBudget] = useState({ used: 7, total: 12, near: false });

  const [query, setQuery] = useState('');
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [searching, setSearching] = useState(0);
  const [results, setResults] = useState(null);

  const [verifying, setVerifying] = useState(-1);
  const [verified, setVerified] = useState(0);

  const [traffic, setTraffic] = useState(FALLBACK_ANALYTICS.traffic);
  const [clock, setClock] = useState(stamp());

  // Every timer this hook owns, cleared together on unmount.
  const timers = useRef([]);
  const later = useCallback((fn, ms) => {
    const id = setTimeout(fn, ms);
    timers.current.push(id);
    return id;
  }, []);

  // ── reference data ────────────────────────────────────────────────
  useEffect(() => {
    let alive = true;
    Promise.all([
      withFallback(camerasApi.fetchCameras(), FALLBACK_CAMERAS, 'cameras'),
      withFallback(camerasApi.fetchPosts(), FALLBACK_POSTS, 'posts'),
      withFallback(analyticsApi.fetchAnalytics(), FALLBACK_ANALYTICS, 'analytics'),
      withFallback(evidenceApi.fetchChain(), FALLBACK_CHAIN, 'evidence chain'),
      withFallback(searchApi.fetchPlaceholders(), FALLBACK_PLACEHOLDERS, 'search placeholders'),
    ]).then(([cams, pts, an, ch, ph]) => {
      if (!alive) return;
      setCameras(cams);
      setPosts(pts);
      setAnalytics(an);
      setTraffic(an.traffic);
      setChain(ch);
      setPlaceholders(ph);
    });
    return () => {
      alive = false;
    };
  }, []);

  // ── alert lifecycle ───────────────────────────────────────────────
  // Ids already on (or through) the queue, so a stream event, a poll and a POST reply for the same
  // alert never add it twice.
  const seen = useRef(new Set());
  const [link, setLink] = useState('connecting');

  const admit = useCallback(
    (incoming, { countBudget = true } = {}) => {
      if (!incoming?.id || incoming.stage === 'gone' || seen.current.has(incoming.id)) return;
      seen.current.add(incoming.id);

      const alert = toDisplayAlert(incoming, stamp);
      setAlerts((list) => addAlert(list, alert));
      if (countBudget) {
        setBudget((b) => {
          const used = Math.min(b.total, b.used + 1);
          return { ...b, used, near: used >= b.total - 3 };
        });
      }
      // The judge finishes a couple of seconds after the provisional box appears.
      if (alert.stage === 'prov') later(() => setAlerts((list) => confirmAlert(list, alert.id)), CONFIRM_AFTER);
    },
    [later],
  );

  useEffect(() => {
    const live = createLiveAlerts({
      streamUrl: alertsApi.streamUrl(),
      fetchSnapshot: alertsApi.fetchAlerts,
      simulate: alertsApi.simulateAlert,
      demo: alertsApi.DEMO_MODE,
      demoDelays: SPAWN_DELAYS,
      pollMs: SPAWN_INTERVAL,
      onMode: setLink,
      onCreated: (alert) => admit(alert),
      onUpdated: ({ id, patch, alert }) => {
        // An update for a card we never saw (it was created before we connected) adds it whole.
        if (!seen.current.has(id)) admit(alert, { countBudget: false });
        else setAlerts((list) => patchAlert(list, id, patch));
      },
      onStage: ({ id, stage }) => {
        if (!seen.current.has(id)) return;
        setAlerts((list) => stageAlert(list, id, stage));
        if (stage === 'gone') later(() => setAlerts((list) => removeAlert(list, id)), 400);
      },
      onSnapshot: (snapshot, { skip, initial }) => {
        const { fresh, patches } = planSnapshot(snapshot, { seen: seen.current, skip });
        fresh.forEach((alert) => admit(alert, { countBudget: !initial }));
        if (patches.length) {
          setAlerts((list) => patches.reduce((next, p) => patchAlert(next, p.id, p.patch), list));
        }
      },
    });
    live.start();
    // stop() aborts the stream request and every timer the controller owns.
    return () => live.stop();
  }, [admit, later]);

  // Every timer this hook owns is cleared together on unmount.
  useEffect(
    () => () => {
      timers.current.forEach(clearTimeout);
      timers.current.forEach(clearInterval);
      timers.current = [];
    },
    [],
  );

  // One typer for every card that still has reason text to reveal; it exists only while needed.
  const typing = alerts.some((a) => a.stage !== 'prov' && a.typed < a.reason.length);
  useEffect(() => {
    if (!typing) return undefined;
    const id = setInterval(() => {
      setAlerts((list) =>
        list.some((a) => a.stage !== 'prov' && a.typed < a.reason.length)
          ? list.map((a) =>
              a.stage !== 'prov' && a.typed < a.reason.length ? { ...a, typed: a.typed + 2 } : a,
            )
          : list,
      );
    }, TYPE_TICK);
    return () => clearInterval(id);
  }, [typing]);

  const actOnAlert = useCallback((id) => {
    setAlerts((list) => stageAlert(list, id, 'sealed'));
    setVerified(0);
    setVerifying(-1);
    withFallback(alertsApi.actOnAlert(id), null, 'seal evidence');
  }, []);

  const dismissAlert = useCallback(
    (id) => {
      setAlerts((list) => stageAlert(list, id, 'gone'));
      later(() => setAlerts((list) => removeAlert(list, id)), 400);
      withFallback(alertsApi.dismissAlert(id), null, 'dismiss alert');
    },
    [later],
  );

  // ── footage search ────────────────────────────────────────────────
  useEffect(() => {
    const id = setInterval(
      () => setPlaceholderIdx((i) => (i + 1) % Math.max(1, placeholders.length)),
      3400,
    );
    return () => clearInterval(id);
  }, [placeholders.length]);

  const runSearch = useCallback(async () => {
    if (searching) return;
    setSearching(1);
    setResults(null);

    // Progress bar runs while the request is in flight.
    let p = 0;
    const request = withFallback(searchApi.runSearch(query), { results: [] }, 'footage search');
    const ticker = setInterval(() => {
      p += 7 + Math.random() * 9;
      setSearching(Math.min(100, p));
      if (p >= 100) clearInterval(ticker);
    }, 120);
    timers.current.push(ticker);

    const [payload] = await Promise.all([
      request,
      new Promise((resolve) => later(resolve, 1500)),
    ]);
    clearInterval(ticker);
    setSearching(100);
    later(() => {
      setSearching(0);
      setResults(payload.results || []);
    }, 240);
  }, [later, query, searching]);

  // ── evidence chain ────────────────────────────────────────────────
  const verifyChain = useCallback(() => {
    withFallback(evidenceApi.verifyChain(), null, 'verify chain');
    let i = 0;
    setVerified(0);
    const id = setInterval(() => {
      setVerifying(i);
      setVerified(i + 1);
      i += 1;
      if (i >= chain.length) {
        clearInterval(id);
        setVerifying(-1);
      }
    }, 340);
    timers.current.push(id);
  }, [chain.length]);

  // ── live tickers ──────────────────────────────────────────────────
  useEffect(() => {
    const trafficId = setInterval(async () => {
      const payload = await withFallback(analyticsApi.fetchTrafficSample(), null, 'traffic sample');
      const sample = payload?.sample ?? 3 + Math.round(Math.random() * 11);
      setTraffic((t) => t.slice(1).concat([sample]));
    }, 2200);
    const clockId = setInterval(() => setClock(stamp()), 1000);
    return () => {
      clearInterval(trafficId);
      clearInterval(clockId);
    };
  }, []);

  const visibleCameras = sector === 'all' ? cameras : cameras.filter((c) => c.sector === sector);

  return {
    // live wall
    cameras: visibleCameras,
    sector,
    setSector,
    expanded,
    setExpanded,
    // chrome
    railOpen,
    toggleRail: () => setRailOpen((v) => !v),
    clock,
    budget,
    // alerts
    alerts,
    // 'connecting' | 'stream' | 'polling' — no UI slot exists for it yet (ConsoleTopBar shows a static "LINK OK")
    connection: link,
    actOnAlert,
    dismissAlert,
    // map
    posts,
    // search
    query,
    setQuery,
    placeholder: placeholders[placeholderIdx] || '',
    searching,
    results,
    runSearch,
    // evidence
    chain,
    verifying,
    verified,
    verifyChain,
    // analytics
    analytics,
    traffic,
  };
};

export default useConsole;
