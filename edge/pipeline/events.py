"""Build `truewatch.event.v1` payloads (docs/ARCHITECTURE_V2.md section 4.2) from pipeline output.

This is the only place the edge turns a fusion-agreed detection into the wire format. It fills what the
edge knows at the provisional stage and leaves the later-stage blocks null exactly as the contract says:
`explanation` is null while stage is 'provisional', `evidence` is null until 'sealed'.

Delivery to the backend (`POST /api/ingest/event`, HMAC-signed) is Phase 8. Until then `EventSink` keeps
the events in a bounded in-memory ring for `GET /pipeline/events` and, optionally, appends them to a
JSON Lines file, so nothing the pipeline decides is lost or reshaped on the way.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from pipeline.types import FusedEvent

SCHEMA = "truewatch.event.v1"


def rfc3339(epoch_s: float) -> str:
    """UTC RFC3339 with milliseconds and a Z suffix, e.g. 2026-08-14T02:41:07.312Z."""
    dt = datetime.fromtimestamp(float(epoch_s), tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def normalised_xywh(bbox_xyxy: Sequence[float], frame_size: tuple[int, int]) -> list[float]:
    """Pixel (x1, y1, x2, y2) -> normalised [x, y, w, h], origin top-left, clipped to 0..1."""
    w, h = frame_size
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    return [round(x1 / w, 4), round(y1 / h, 4), round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)]


def build_event(
    fused: FusedEvent,
    *,
    post_id: str,
    bbox: Sequence[float] | None = None,
    captured_at: float | None = None,
    rules_fired: Iterable[str] = (),
    rule_baseline: dict | None = None,
    rule_metrics: dict | None = None,
    anpr: dict | None = None,
) -> dict:
    """One provisional `truewatch.event.v1` for a fusion-agreed track.

    `fused` is the event fusion emitted for this track (it carries the channel scores, threshold and
    modality the contract asks for). `bbox` / `captured_at` override the box and time when the payload
    is raised later than the fusion event itself, e.g. when a rule fires on the same track frames later.
    """
    fired = list(dict.fromkeys(rules_fired))
    rule = None
    if fired:
        rule = {"fired": fired, "baseline": rule_baseline, "metrics": rule_metrics or {}}
    return {
        "schema": SCHEMA,
        "event_id": str(uuid.uuid4()),
        "post_id": post_id,
        "camera_id": fused.camera_id,
        "captured_at": rfc3339(fused.captured_at if captured_at is None else captured_at),
        "stage": "provisional",
        "detection": {
            "track_id": int(fused.track_id) if fused.track_id is not None else None,
            "class": fused.class_name,
            "bbox": normalised_xywh(fused.bbox if bbox is None else bbox, fused.frame_size),
            "channel_scores": {
                "appearance": round(float(fused.appearance.score), 4),
                "motion": round(float(fused.motion.score), 4),
            },
            "agreed": True,
            "threshold": round(float(fused.threshold), 4),
            "ir": fused.modality == "lwir",
            "dori_band": None,
        },
        "rule": rule,
        "explanation": None,
        "anpr": anpr,
        "evidence": None,
        "confidence": max(0, min(100, int(round(float(fused.fused_score) * 100)))),
        "source": fused.source if fused.source in ("rtsp", "file") else "rtsp",
    }


class EventSink:
    """Bounded in-memory ring of emitted events, plus an optional JSON Lines log. Thread-safe."""

    def __init__(self, maxlen: int = 500, log_path: Path | None = None) -> None:
        self._events: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.log_path = Path(log_path) if log_path else None
        self.total = 0
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: dict) -> None:
        with self._lock:
            self._events.append(event)
            self.total += 1
            if self.log_path is not None:
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._events)[-max(0, int(limit)):]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self.total = 0
