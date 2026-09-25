"""Dual-channel fusion: an alert is raised only when appearance and motion AGREE (slides 2 and 3).

"Agreement", precisely
----------------------
Fusion judges candidates, and a candidate is a ByteTrack track (tracking.py) built from appearance
boxes. For each candidate on each frame:

  a   appearance score in [0, 1]: the confidence of the detection last associated to the track, used
      only while it is fresh (no older than `appearance_max_age` frames, i.e. the detector stride).
  m   motion score in [0, 1] from motion.MotionChannel.score_box: flow energy inside the box against
      the local background ring, normalised by THIS camera's learnt thresholds (calibration.py).
  iou spatial gate: IoU between the box and the flow-peak region (the connected motion-mask component
      with the most flow energy near the box). The motion must be where the object is, not merely
      somewhere in the frame.

  The candidate AGREES on this frame when all four hold:
      a >= a_min                       the detector vouches for an object here           (channel floor)
      m >= m_min                       the flow vouches for movement here                (channel floor)
      iou >= iou_min                   and they are the same place                       (spatial gate)
      s = w_a * a + w_m * m >= theta   and the weighted evidence clears the per-camera threshold

  An ALERT (a FusedEvent) is raised when a candidate agrees on `n_consecutive` consecutive frames
  (temporal gate). One alert per track; it re-arms only after agreement has been broken for `rearm_s`.

Why this is not a boolean AND. The floors are deliberately low (a_min 0.15, m_min 0.25): each channel
only has to say "something is here", not "I am sure". The decision is made by the WEIGHTED sum against
the camera's threshold, so a strong channel can carry a weak one, which is exactly what failure
independence needs; the floors stop one channel from carrying a candidate the other did not see at all;
the spatial gate stops two unrelated observations (a parked car, a tree moving beside it) from being
counted as one; and the temporal gate stops single-frame coincidences (a flicker, a compression burst).

Day and night weights, derived from docs/MEASUREMENTS.md section 1
------------------------------------------------------------------
Section 1 measured Farneback flow inside moving-object boxes against the background on paired KAIST
frames: visible 5.17 px vs 2.74 px (contrast 1.9x), LWIR 2.73 px vs 0.81 px (contrast 3.4x). The motion
weight is the share of in-box flow that belongs to the object rather than the background, 1 - bg / obj
(motion.motion_reliability):

      visible (day)    w_m = 1 - 2.74 / 5.17 = 0.47    w_a = 0.53
      LWIR   (night)   w_m = 1 - 0.81 / 2.73 = 0.70    w_a = 0.30

So motion carries more weight after dark BECAUSE its object-to-background contrast rises from 1.9x to
3.4x on LWIR, even though absolute flow on LWIR is lower (whole-frame flow is about 32% of visible). The
weights come from the measured numbers, not from a belief that flow is indifferent to the sensor.
NIGHT_MOTION_WEIGHT / DAY_MOTION_WEIGHT in the environment override them per deployment.

Failure independence, and how the rule uses it
----------------------------------------------
The channels are built on different principles (what a thing looks like, how its pixels move), so they
fail on different frames:

  appearance fails, motion holds
    - LWIR at night: the detector's confidence drops (MEASUREMENTS 1: mean 0.59 -> 0.54, 89% recall of
      visible) and on false-colour thermal it collapses (AP@50 0.133), but a warm body moving across a
      thermally flat background is the channel's best case (contrast 3.4x).
    - low texture: a person in camouflage against scrub, a figure far away at 15-20 px (MEASUREMENTS 3),
      where appearance cues are thin but displacement is not.
    - an unusual class: a load carrier, a person crawling or carrying a bundle, a hand cart the model was
      never trained on (class 4 may be withdrawn): a weak or low-confidence box, but coherent movement.
    Exploited by: the detector runs with a LOW candidate floor (appearance.AppearanceConfig.conf 0.10)
    and ByteTrack keeps low boxes attached to tracks, so a weak box survives to fusion, where a strong m
    under the night weights lifts s over theta. Example, a = 0.20, m = 0.90: night s = 0.30*0.20 +
    0.70*0.90 = 0.69 >= 0.55 (alert); the same evidence under day weights gives 0.53*0.20 + 0.47*0.90 =
    0.53 (no alert).

  motion fails, appearance holds
    - a stationary person (standing watch, lying in wait): no flow at all, m = 0.
    - camera shake or wind on the mast: everything moves, so in-box flow equals the background ring and
      the contrast term of m falls to 0; the flow peak is the whole frame, so the spatial gate fails too.
    - foliage, water, flags: strong local flow, but the camera's learnt per-cell threshold is high there
      (calibration.py), and no person/vehicle box sits on it.
    Exploited by: m_min. A confident box with no motion never alerts on its own (a parked truck, a
    mannequin, a poster of a person), and a moving tree never alerts without a box. The tracker keeps the
    stationary person's id, so the moment they move, the streak starts from an existing track.

  Both fail together only when a target is at once invisible to the detector AND not moving against
  its background, which is the case two channels cannot cover and no single channel covers better.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

from pipeline.motion import MotionChannel, MotionField, motion_reliability
from pipeline.types import ChannelScore, FrameRef, FusedEvent, Track


def _env_weight(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if 0.0 <= v <= 1.0 else None


def modality_weights(modality: str) -> tuple[float, float]:
    """(w_appearance, w_motion) for a modality, from MEASUREMENTS.md section 1 (see module docstring)."""
    override = _env_weight("NIGHT_MOTION_WEIGHT" if modality == "lwir" else "DAY_MOTION_WEIGHT")
    w_m = override if override is not None else motion_reliability(modality)
    return round(1.0 - w_m, 4), round(w_m, 4)


@dataclass(frozen=True)
class FusionConfig:
    threshold: float = 0.55             # FUSION_THRESHOLD_DEFAULT; per camera via calibration
    a_min: float = 0.15
    m_min: float = 0.25
    iou_min: float = 0.2
    n_consecutive: int = 3
    appearance_max_age: int = 2         # frames; set to the detector stride
    rearm_s: float = 10.0

    @classmethod
    def from_env(cls, **overrides) -> "FusionConfig":
        raw = os.environ.get("FUSION_THRESHOLD_DEFAULT", "").strip()
        kw = {}
        try:
            if raw:
                kw["threshold"] = float(raw)
        except ValueError:
            pass
        kw.update(overrides)
        return cls(**kw)


@dataclass(frozen=True)
class Decision:
    """Fusion's verdict on one candidate on one frame (logged; only `event` leaves the process)."""

    track_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    appearance: ChannelScore
    motion: ChannelScore
    spatial_iou: float
    fused: float
    fused_day: float                    # s under the visible weights (for the night/day comparison)
    fused_night: float                  # s under the LWIR weights
    a_ok: bool
    m_ok: bool
    iou_ok: bool
    threshold_ok: bool
    agree: bool
    streak: int
    event: FusedEvent | None = None

    @property
    def gates(self) -> str:
        return "".join(ch if ok else "-" for ch, ok in
                       (("A", self.a_ok), ("M", self.m_ok), ("S", self.iou_ok), ("T", self.threshold_ok)))


@dataclass
class _TrackState:
    streak: int = 0
    streak_start: float = 0.0
    best: FrameRef | None = None
    alerted: bool = False
    broken_since: float | None = None
    last_seen: float = 0.0


@dataclass
class FusionFrame:
    decisions: list[Decision]
    events: list[FusedEvent]
    status: str                         # 'ok' | 'warming_up' | 'no_motion_yet' | 'camera_state'


@dataclass
class Fusion:
    camera_id: str
    modality: str
    config: FusionConfig = field(default_factory=FusionConfig)
    source: str = "file"
    _states: dict = field(default_factory=dict)

    @property
    def weights(self) -> tuple[float, float]:
        return modality_weights(self.modality)

    @property
    def night(self) -> bool:
        return self.modality == "lwir"

    def reset_streaks(self) -> None:
        for s in self._states.values():
            s.streak = 0
            s.best = None

    def evaluate(
        self,
        tracks: Sequence[Track],
        field_: MotionField | None,
        *,
        appearance_of: dict[int, tuple[float, int]],
        frame_index: int,
        timestamp: float,
        captured_at: float,
        frame_size: tuple[int, int],
        bg_floor: float = 0.05,
        threshold: float | None = None,
        status: str = "ok",
    ) -> FusionFrame:
        """Judge every candidate on this frame. `appearance_of[track_id] = (score, frame_index)`.

        A `status` other than 'ok' (warming up, camera-state anomaly) scores candidates for the log but
        raises nothing and resets every streak: agreement must be re-earned on a trustworthy picture.
        """
        cfg = self.config
        theta = cfg.threshold if threshold is None else threshold
        w_a, w_m = self.weights
        wd_a, wd_m = modality_weights("visible")
        wn_a, wn_m = modality_weights("lwir")
        seen = set()
        decisions: list[Decision] = []
        events: list[FusedEvent] = []
        if field_ is None:
            status = "no_motion_yet" if status == "ok" else status

        boxes = [t.bbox for t in tracks]
        for i, t in enumerate(tracks):
            seen.add(t.track_id)
            st = self._states.setdefault(t.track_id, _TrackState())
            st.last_seen = timestamp
            a_raw, a_frame = appearance_of.get(t.track_id, (0.0, -10**9))
            fresh = frame_index - a_frame <= cfg.appearance_max_age
            a_val = a_raw if fresh else 0.0
            a = ChannelScore("appearance", round(a_val, 4), round(a_raw, 4),
                             f"detector {t.class_name} {a_raw:.2f}" + ("" if fresh else f", stale ({frame_index - a_frame} frames)"))
            if field_ is not None:
                others = boxes[:i] + boxes[i + 1:]
                m, iou = MotionChannel.score_box(field_, t.bbox, others, bg_floor=bg_floor)
            else:
                m, iou = ChannelScore("motion", 0.0, 0.0, "no flow yet (first frame)"), 0.0
            s = w_a * a.score + w_m * m.score
            a_ok, m_ok, iou_ok, thr_ok = a.score >= cfg.a_min, m.score >= cfg.m_min, iou >= cfg.iou_min, s >= theta
            agree = a_ok and m_ok and iou_ok and thr_ok and status == "ok"

            event = None
            if agree:
                if st.streak == 0:
                    st.streak_start = timestamp
                st.streak += 1
                st.broken_since = None
                if st.best is None or s > st.best.fused_score:
                    st.best = FrameRef(self.camera_id, frame_index, timestamp, captured_at, t.bbox, round(s, 4))
                if st.streak >= cfg.n_consecutive and not st.alerted:
                    st.alerted = True
                    event = FusedEvent(
                        camera_id=self.camera_id, track_id=t.track_id, class_name=t.class_name, bbox=t.bbox,
                        frame_size=frame_size, appearance=a, motion=m, fused_score=round(s, 4), threshold=theta,
                        weights=(w_a, w_m), modality=self.modality, night=self.night, spatial_iou=iou,
                        streak=st.streak, first_agreed_at=st.streak_start, timestamp=timestamp,
                        captured_at=captured_at, frame_index=frame_index, best_frame=st.best, source=self.source,
                        track=t,
                    )
                    events.append(event)
            else:
                st.streak = 0
                st.best = None
                if st.alerted:
                    st.broken_since = st.broken_since if st.broken_since is not None else timestamp
                    if timestamp - st.broken_since >= cfg.rearm_s:
                        st.alerted = False
                        st.broken_since = None
            decisions.append(Decision(
                track_id=t.track_id, class_name=t.class_name, bbox=t.bbox, appearance=a, motion=m, spatial_iou=iou,
                fused=round(s, 4), fused_day=round(wd_a * a.score + wd_m * m.score, 4),
                fused_night=round(wn_a * a.score + wn_m * m.score, 4), a_ok=a_ok, m_ok=m_ok, iou_ok=iou_ok,
                threshold_ok=thr_ok, agree=agree, streak=st.streak, event=event,
            ))
        for tid, st in list(self._states.items()):
            if tid not in seen:
                st.streak, st.best = 0, None           # a track that left the frame breaks its streak
                if timestamp - st.last_seen > max(self.config.rearm_s, 30.0):
                    del self._states[tid]
        return FusionFrame(decisions, events, status)
