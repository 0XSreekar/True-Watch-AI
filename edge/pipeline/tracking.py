"""ByteTrack, implemented here, plus the loitering signals of docs/MEASUREMENTS.md section 2.

ByteTrack (Zhang et al., ECCV 2022, "ByteTrack: Multi-Object Tracking by Associating Every Detection
Box") is re-implemented from the paper rather than vendored, so edge/ carries no tracker dependency; the
only library used is scipy's Hungarian solver (BSD). The algorithm, per frame with detections:

  1. Split detections by score: high (>= track_high) and low (track_low .. track_high).
  2. Predict every tracked and lost track forward with a constant-velocity Kalman filter on
     (cx, cy, aspect, height).
  3. First association: tracked + lost tracks against HIGH detections, cost 1 - IoU, Hungarian,
     accept cost <= match_thresh.
  4. Second association: the still-unmatched TRACKED tracks against LOW detections, accept cost <= 0.5.
     This is ByteTrack's contribution: an occluded or blurred person whose score dropped keeps its id
     instead of spawning a new one; the low boxes are never allowed to start tracks.
  5. Unconfirmed (one-frame) tracks against the remaining high detections, accept cost <= 0.7.
  6. Unmatched tracked tracks become lost; lost tracks older than track_buffer_s are removed.
  7. Remaining high detections with score >= new_track start new (unconfirmed) tracks.
Association is class-gated: a box never matches a track of an incompatible class ('car' and 'truck'
are compatible, because the detector flickers between them on vans and buses).

Frame skipping. When the appearance channel skips a frame (pipeline.runner), `predict_only` advances the
Kalman filters so every track still has a current box for the motion channel to score. Track loss is
timed in SECONDS (track_buffer_s), not frames, so the same buffer holds at any detector stride.

Loitering signals (MEASUREMENTS.md section 2)
---------------------------------------------
Computed from OBSERVED centres only (frames where a detection was associated), never predictions:
  frames   number of observed frames
  dwell_s  (last_frame - first_frame) x frame period, i.e. t_last - t_first: time elapsed between the
           first and the last observation. Section 2's table pins this down: track 3, 33 frames at
           20 fps, is printed 1.6 s (32 intervals); counting both ends would give 1.65 -> 1.7 s. The
           other rows agree (70 contiguous frames -> 3.45 s, printed 3.5 s; track 12's 11 frames over
           24 intervals -> 1.2 s).
  net_px   |centre_last - centre_first|
  path_px  sum of |centre_i - centre_{i-1}|
The loitering rule of section 2 is dwell >= 3 s and net < 90 px; track 6 (5.0 s, net 70, path 163)
moved back and forth in place. The rule itself belongs to the rule engine (Phase 5); this module only
provides the signals, and tools/reproduce_rule_sanity.py re-runs the section 2 table on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline.types import BBox, Detection, Track

COMPATIBLE = {frozenset({"car", "truck"})}


def classes_compatible(a: str, b: str) -> bool:
    return a == b or frozenset({a, b}) in COMPATIBLE


# ------------------------------------------------------------------------------------ signals


def dwell_seconds(first_frame: int, last_frame: int, frame_period: float) -> float:
    return (last_frame - first_frame) * frame_period


def net_displacement(centres: Sequence[tuple[float, float]]) -> float:
    if len(centres) < 2:
        return 0.0
    (x0, y0), (x1, y1) = centres[0], centres[-1]
    return math.hypot(x1 - x0, y1 - y0)


def path_length(centres: Sequence[tuple[float, float]]) -> float:
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(centres, centres[1:])))


def frame_period_from(history: Sequence[tuple[int, float, float, float]], default: float) -> float:
    """Seconds per frame, from the history's own (frame_index, timestamp) pairs when it spans > 1 frame."""
    if len(history) >= 2:
        f0, t0 = history[0][0], history[0][1]
        f1, t1 = history[-1][0], history[-1][1]
        if f1 > f0 and t1 > t0:
            return (t1 - t0) / (f1 - f0)
    return default


# ------------------------------------------------------------------------------------ Kalman


class KalmanXYAH:
    """Constant-velocity Kalman filter on (cx, cy, a, h), the parameterisation ByteTrack uses."""

    std_pos = 1.0 / 20
    std_vel = 1.0 / 160

    def __init__(self) -> None:
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.std_pos * h, 2 * self.std_pos * h, 1e-2, 2 * self.std_pos * h,
               10 * self.std_vel * h, 10 * self.std_vel * h, 1e-5, 10 * self.std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h,
               self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h]
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(np.square(std))
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        R = np.diag(np.square([self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]))
        S = self.H @ cov @ self.H.T + R
        K = np.linalg.solve(S, (cov @ self.H.T).T).T
        mean = mean + K @ (z - self.H @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


def xyxy_to_xyah(b: BBox) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(m: np.ndarray) -> BBox:
    cx, cy, a, h = m[:4]
    h = max(h, 1e-3)
    w = a * h
    return (float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2))


def iou_matrix(a: Sequence[BBox], b: Sequence[BBox]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    A, B = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ix1 = np.maximum(A[:, None, 0], B[None, :, 0]); iy1 = np.maximum(A[:, None, 1], B[None, :, 1])
    ix2 = np.minimum(A[:, None, 2], B[None, :, 2]); iy2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _assign(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    c = np.where(cost > thresh, thresh + 1e3, cost)
    rows, cols = linear_sum_assignment(c)
    matches = [(r, k) for r, k in zip(rows, cols) if cost[r, k] <= thresh]
    mr = {r for r, _ in matches}; mc = {k for _, k in matches}
    return matches, [r for r in range(cost.shape[0]) if r not in mr], [k for k in range(cost.shape[1]) if k not in mc]


# ------------------------------------------------------------------------------------ tracks


@dataclass
class _STrack:
    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    score: float
    first_frame: int
    last_frame: int
    last_ts: float
    state: str = "tentative"                 # tentative | confirmed | lost | removed
    votes: dict = field(default_factory=dict)
    history: list = field(default_factory=list)   # (frame, ts, cx, cy, bottom_y) observed
    observed_now: bool = True
    lost_since: float | None = None

    @property
    def bbox(self) -> BBox:
        return xyah_to_xyxy(self.mean)

    @property
    def class_name(self) -> str:
        return max(self.votes.items(), key=lambda kv: kv[1])[0]

    def vote(self, det: Detection) -> None:
        self.votes[det.class_name] = self.votes.get(det.class_name, 0.0) + det.score


@dataclass(frozen=True)
class TrackerConfig:
    # ByteTrack's published defaults are 0.5 / 0.1 / 0.6. They are lowered here because in this pipeline the
    # tracker is not the filter, fusion is: a track is only a CANDIDATE, and a weak box that never becomes a
    # track can never be confirmed by motion. On LWIR the pretrained detector's person scores sit mostly
    # between 0.1 and 0.4 (MEASUREMENTS.md section 1 mean 0.54 on its set; lower on set00/V001), so a 0.6
    # birth threshold would remove exactly the night candidates the motion channel is weighted up to rescue.
    track_high: float = 0.3
    track_low: float = 0.1
    new_track: float = 0.3
    match_thresh: float = 0.8
    second_match: float = 0.5
    unconfirmed_match: float = 0.7
    track_buffer_s: float = 1.5
    default_frame_period: float = 1.0 / 20.0     # used until a track's own timestamps give one
    history_max: int = 3000
    finished_max: int = 1000                     # snapshots of ended tracks kept for rules and reports


class ByteTracker:
    def __init__(self, config: TrackerConfig | None = None, camera_id: str = "CAM-01", modality: str = "visible") -> None:
        self.config = config or TrackerConfig()
        self.camera_id = camera_id
        self.modality = modality
        self.kf = KalmanXYAH()
        self._tracks: list[_STrack] = []
        self._next_id = 1
        self._frame = -1
        self.finished: list[Track] = []          # snapshots of confirmed tracks that have ended, oldest first

    # -------------------------------------------------------------- public

    def predict_only(self, frame_index: int, timestamp: float) -> list[Track]:
        """A frame without appearance: advance every track, associate nothing."""
        self._frame = frame_index
        for t in self._tracks:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
            t.observed_now = False
        self._expire(timestamp)
        return self.active_tracks()

    def update(self, detections: Sequence[Detection], frame_index: int, timestamp: float) -> list[Track]:
        cfg = self.config
        self._frame = frame_index
        for t in self._tracks:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
            t.observed_now = False

        high = [d for d in detections if d.score >= cfg.track_high]
        low = [d for d in detections if cfg.track_low <= d.score < cfg.track_high]
        confirmed = [t for t in self._tracks if t.state in ("confirmed", "lost")]
        unconfirmed = [t for t in self._tracks if t.state == "tentative"]

        # 1. confirmed + lost vs high
        m1, u_trk, u_high = _assign(self._cost(confirmed, high), cfg.match_thresh)
        for ti, di in m1:
            self._apply(confirmed[ti], high[di], timestamp)
        # 2. remaining TRACKED (not lost) vs low
        remaining = [confirmed[i] for i in u_trk if confirmed[i].state == "confirmed"]
        m2, u_rem, _ = _assign(self._cost(remaining, low), cfg.second_match)
        for ti, di in m2:
            self._apply(remaining[ti], low[di], timestamp)
        for i in u_rem:
            t = remaining[i]
            t.state = "lost"
            t.lost_since = t.lost_since if t.lost_since is not None else timestamp
        # 3. unconfirmed vs remaining high
        rest_high = [high[i] for i in u_high]
        m3, u_unc, u_rest = _assign(self._cost(unconfirmed, rest_high), cfg.unconfirmed_match)
        for ti, di in m3:
            self._apply(unconfirmed[ti], rest_high[di], timestamp)
        for i in u_unc:
            unconfirmed[i].state = "removed"
        # 4. new tracks
        for i in u_rest:
            d = rest_high[i]
            if d.score >= cfg.new_track:
                self._start(d, frame_index, timestamp)
        self._expire(timestamp)
        return self.active_tracks()

    def active_tracks(self, include_tentative: bool = False) -> list[Track]:
        states = ("confirmed", "tentative") if include_tentative else ("confirmed",)
        return [self._snapshot(t) for t in self._tracks if t.state in states]

    def all_tracks(self) -> list[Track]:
        return [self._snapshot(t) for t in self._tracks if t.state != "removed"]

    def last_score(self, track_id: int) -> tuple[float, int] | None:
        """(score of the last associated detection, its frame index) for fusion's appearance score."""
        for t in self._tracks:
            if t.track_id == track_id:
                return t.score, t.last_frame
        return None

    # -------------------------------------------------------------- internals

    def _cost(self, tracks: Sequence[_STrack], dets: Sequence[Detection]) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        cost = 1.0 - iou_matrix([t.bbox for t in tracks], [d.bbox for d in dets])
        for i, t in enumerate(tracks):
            cls = t.class_name
            for j, d in enumerate(dets):
                if not classes_compatible(cls, d.class_name):
                    cost[i, j] = 1.0 + 1e-3
        return cost

    def _start(self, d: Detection, frame_index: int, timestamp: float) -> None:
        mean, cov = self.kf.initiate(xyxy_to_xyah(d.bbox))
        t = _STrack(self._next_id, mean, cov, d.score, frame_index, frame_index, timestamp)
        self._next_id += 1
        t.vote(d)
        cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
        t.history.append((frame_index, timestamp, cx, cy, float(d.bbox[3])))
        # ByteTrack activates tracks on the very first frame of a sequence; otherwise one more hit is needed
        t.state = "confirmed" if frame_index == 0 else "tentative"
        self._tracks.append(t)

    def _apply(self, t: _STrack, d: Detection, timestamp: float) -> None:
        t.mean, t.cov = self.kf.update(t.mean, t.cov, xyxy_to_xyah(d.bbox))
        t.score = d.score
        t.last_frame = self._frame
        t.last_ts = timestamp
        t.state = "confirmed"
        t.lost_since = None
        t.observed_now = True
        t.vote(d)
        cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
        t.history.append((self._frame, timestamp, cx, cy, float(d.bbox[3])))
        if len(t.history) > self.config.history_max:
            del t.history[: len(t.history) - self.config.history_max]

    def _expire(self, timestamp: float) -> None:
        for t in self._tracks:
            if t.state == "lost" and t.lost_since is not None and timestamp - t.last_ts > self.config.track_buffer_s:
                t.state = "removed"
            elif t.state == "confirmed" and not t.observed_now and timestamp - t.last_ts > self.config.track_buffer_s / 2:
                # no detection for half the buffer on skipped frames: treat as lost so ids are re-used carefully
                t.state = "lost"
                t.lost_since = t.last_ts
        for t in self._tracks:
            if t.state == "removed" and len(t.history) >= 2:
                self.finished.append(self._snapshot(t))
        self._tracks = [t for t in self._tracks if t.state != "removed"]
        if len(self.finished) > self.config.finished_max:
            del self.finished[: len(self.finished) - self.config.finished_max]

    def _snapshot(self, t: _STrack) -> Track:
        raw = t.history
        hist = tuple((h[0], h[1], h[2], h[3]) for h in raw)
        ground = tuple((h[1], h[2], h[4]) for h in raw)
        period = frame_period_from(hist, self.config.default_frame_period)
        centres = [(h[2], h[3]) for h in hist]
        first_frame, last_frame = hist[0][0], hist[-1][0]
        vx = vy = 0.0
        if len(hist) >= 2:
            k = hist[max(0, len(hist) - 6)]
            dt = hist[-1][1] - k[1]
            if dt > 0:
                vx, vy = (hist[-1][2] - k[2]) / dt, (hist[-1][3] - k[3]) / dt
        return Track(
            track_id=t.track_id, class_name=t.class_name, bbox=t.bbox, score=round(t.score, 4),
            camera_id=self.camera_id, modality=self.modality,
            state="lost" if t.state == "lost" else ("confirmed" if t.state == "confirmed" else "tentative"),
            first_frame=first_frame, last_frame=last_frame, first_seen=hist[0][1], last_seen=hist[-1][1],
            frames=len(hist), dwell_s=round(dwell_seconds(first_frame, last_frame, period), 4),
            net_px=round(net_displacement(centres), 3), path_px=round(path_length(centres), 3),
            velocity=(round(vx, 3), round(vy, 3)), history=ground, observations=hist,
            observed_now=t.observed_now,
        )
