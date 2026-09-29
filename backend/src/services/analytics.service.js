import { ANALYTICS } from '../data/mockData.js';

const AGREEMENT_POINTS = 12;
const TRAFFIC_POINTS = 12;

/**
 * Aggregates over real rows, returning exactly the frozen shapes { camBars, dayNight, agreement,
 * traffic }. When the database holds nothing to aggregate (an empty deployment), each series
 * falls back to the fixture so the panel is never blank.
 *
 *  - camBars:   alerts per camera, all time.
 *  - dayNight:  share of alerts raised 06:00-17:59 vs the rest, by the operator-facing time label.
 *  - agreement: fused agreement score (confidence) of the latest 12 alerts, oldest first.
 *  - traffic:   tracks the edge reported per minute over the last 12 minutes.
 */
export const createAnalyticsService = ({ repos }) => {
  const camBars = () => {
    const rows = repos.alerts.countsByCamera();
    return rows.some((r) => r.v > 0) ? rows.map(({ id, v }) => ({ id, v })) : ANALYTICS.camBars;
  };

  const dayNight = () => {
    const { day, night } = repos.alerts.dayNightCounts();
    const total = day + night;
    if (!total) return ANALYTICS.dayNight;
    const dayPct = Math.round((day / total) * 100);
    const [dayMeta, nightMeta] = ANALYTICS.dayNight;
    return [
      { ...dayMeta, v: dayPct },
      { ...nightMeta, v: 100 - dayPct },
    ];
  };

  const agreement = () => {
    const latest = repos.alerts.latestConfidences(AGREEMENT_POINTS);
    if (!latest.length) return ANALYTICS.agreement;
    return [...Array(AGREEMENT_POINTS - latest.length).fill(latest[0]), ...latest];
  };

  const traffic = () => {
    const series = repos.tracks.perWindow({ buckets: TRAFFIC_POINTS });
    return series.some((v) => v > 0) ? series : ANALYTICS.traffic;
  };

  return {
    getAnalytics: () => ({
      camBars: camBars(),
      dayNight: dayNight(),
      agreement: agreement(),
      traffic: traffic(),
    }),

    /** Tracks in the last minute; a plausible random sample only when the edge has reported none. */
    nextTrafficSample() {
      const recent = repos.tracks.perWindow({ buckets: TRAFFIC_POINTS });
      if (!recent.some((v) => v > 0)) return 3 + Math.round(Math.random() * 11);
      return recent[recent.length - 1];
    },
  };
};

export default createAnalyticsService;
