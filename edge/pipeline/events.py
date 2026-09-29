"""Build `truewatch.event.v1` payloads (docs/ARCHITECTURE_V2.md section 4.2) from pipeline output.

This is the only place the edge turns a fusion-agreed detection into the wire format. It fills what the
edge knows at the provisional stage and leaves the later-stage blocks null exactly as the contract says:
`evidence` is null until 'sealed'; `explanation` is the rule-string block the caller passes (or null).

`EventSink` keeps the events in a bounded in-memory ring for `GET /pipeline/events`, optionally appends
them to a JSON Lines file, and, when a `BackendDelivery` is attached, hands each one to a background
sender that POSTs it to the backend (section 4.1 of the architecture document): the pipeline thread
only ever appends to a bounded outbox and never waits on the network.

A created event carries the rule-string `explanation` (source 'template') from the first instant, so the
alert's reason is never empty. When the vision-language model later produces a sentence it arrives as a
REASON PATCH for the same `event_id` (`build_reason_patch`): additive to the v1 contract, it changes
nothing but the `explanation` block.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from pipeline.types import FusedEvent

SCHEMA = "truewatch.event.v1"
PATCH_TYPE = "reason_patch"

log = logging.getLogger("truewatch.edge.events")


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
    explanation: dict | None = None,
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
        "explanation": explanation,
        "anpr": anpr,
        "evidence": None,
        "confidence": max(0, min(100, int(round(float(fused.fused_score) * 100)))),
        "source": fused.source if fused.source in ("rtsp", "file") else "rtsp",
    }




def build_reason_patch(event: dict, explanation: dict) -> dict:
    """The additive follow-up message that replaces a created event's reason.

    Same `event_id` as the created event, the identifying fields the backend needs to route it, and
    nothing but the new `explanation` block; `"type": "reason_patch"` is what distinguishes it from a
    created event. It is idempotent: applying it twice leaves the same reason.
    """
    return {
        "schema": SCHEMA,
        "type": PATCH_TYPE,
        "event_id": event["event_id"],
        "post_id": event["post_id"],
        "camera_id": event["camera_id"],
        "captured_at": event["captured_at"],
        "stage": event["stage"],
        "explanation": dict(explanation),
    }


# ------------------------------------------------------------------------------------------ delivery


INGEST_PATH = "/api/ingest/event"
KEY_HEADER = "X-Truewatch-Ingest-Key"
TIMESTAMP_HEADER = "X-Truewatch-Timestamp"
NONCE_HEADER = "X-Truewatch-Nonce"

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass
class DeliveryStats:
    enqueued: int = 0
    delivered: int = 0
    duplicates: int = 0           # the backend already had this event_id (idempotent redelivery)
    retries: int = 0
    dropped_overflow: int = 0     # oldest message discarded because the outbox was full
    dropped_exhausted: int = 0    # every attempt failed (backend down / 5xx)
    rejected: int = 0             # the backend refused the message for good (4xx)
    patches_skipped: int = 0      # a patch whose created event never reached the backend
    last_error: str | None = None

    def to_dict(self) -> dict:
        return dict(vars(self))


def _http_post(url: str, headers: dict, body: bytes, timeout_s: float) -> int:
    """POST `body`; return the HTTP status, or raise OSError for a transport failure."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - scheme checked at construction
            resp.read(4096)
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise OSError(str(getattr(exc, "reason", exc))) from exc


class BackendDelivery:
    """Background sender for `POST {base_url}/api/ingest/event` (ARCHITECTURE_V2 section 4.1).

    * The pipeline thread calls `enqueue()`, which only appends to a bounded outbox: it never touches
      the network and never blocks. When the outbox is full the OLDEST message is dropped (counted),
      so a long backend outage costs the oldest undelivered events, never memory and never frame rate.
    * One sender thread delivers in order (a created event is always sent before its reason patch).
    * Each message gets `max_attempts` tries with exponential backoff (`backoff_s`, doubling up to
      `backoff_cap_s`) on a transport failure or a 408/425/429/5xx. A 2xx or a 409 (the backend already
      holds this `event_id`) counts as delivered; any other 4xx is a permanent refusal and is dropped
      with an error log, as retrying the same body cannot succeed.
    * Every attempt carries a fresh timestamp and a fresh nonce, because the backend refuses a repeated
      nonce as a replay; the `event_id` is the idempotency key that makes the retry safe.
    * The shared secret is only ever sent in the header, never logged.
    """

    def __init__(
        self,
        base_url: str,
        ingest_key: str,
        *,
        timeout_s: float = 3.0,
        max_attempts: int = 5,
        backoff_s: float = 0.5,
        backoff_cap_s: float = 8.0,
        outbox_max: int = 1000,
        post=_http_post,
        start: bool = True,
    ) -> None:
        if not base_url.lower().startswith(("http://", "https://")):
            raise ValueError(f"backend base URL must be http(s), got {base_url!r}")
        if not ingest_key:
            raise ValueError("an ingest key is required to deliver events")
        self.url = base_url.rstrip("/") + INGEST_PATH
        self._key = ingest_key
        self.timeout_s = float(timeout_s)
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_s = max(0.0, float(backoff_s))
        self.backoff_cap_s = max(self.backoff_s, float(backoff_cap_s))
        self.outbox_max = max(1, int(outbox_max))
        self._post = post
        self._outbox: deque[tuple[str, str, bytes]] = deque()   # (kind, event_id, body)
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._busy = False
        self._lost: deque[str] = deque(maxlen=2048)
        self._lost_set: set[str] = set()
        self.stats = DeliveryStats()
        self._thread: threading.Thread | None = None
        if start:
            self._thread = threading.Thread(target=self._run, name="event-delivery", daemon=True)
            self._thread.start()

    # ---------------------------------------------------------------- producer side

    def enqueue(self, message: dict) -> None:
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        kind = "patch" if message.get("type") == PATCH_TYPE else "event"
        with self._cv:
            if len(self._outbox) >= self.outbox_max:
                old_kind, old_id, _ = self._outbox.popleft()
                self.stats.dropped_overflow += 1
                if old_kind == "event":
                    self._mark_lost(old_id)
                log.warning("delivery outbox full (%d): dropped the oldest undelivered %s %s",
                            self.outbox_max, old_kind, old_id)
            self._outbox.append((kind, str(message.get("event_id")), body))
            self.stats.enqueued += 1
            self._cv.notify()

    def pending(self) -> int:
        with self._cv:
            return len(self._outbox) + (1 if self._busy else 0)

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Wait until the outbox is empty and nothing is in flight (shutdown and tests)."""
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self._outbox or self._busy:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(min(left, 0.05))
        return True

    def close(self, timeout_s: float = 2.0) -> None:
        self.flush(timeout_s)
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    # ---------------------------------------------------------------- sender side

    def _mark_lost(self, event_id: str) -> None:
        if len(self._lost) == self._lost.maxlen:
            self._lost_set.discard(self._lost[0])
        self._lost.append(event_id)
        self._lost_set.add(event_id)

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            KEY_HEADER: self._key,
            TIMESTAMP_HEADER: rfc3339(time.time()),
            NONCE_HEADER: str(uuid.uuid4()),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._outbox and not self._stop.is_set():
                    self._cv.wait(0.25)
                if self._stop.is_set() and not self._outbox:
                    return
                item = self._outbox.popleft()
                self._busy = True
            try:
                self._deliver(*item)
            except Exception:  # noqa: BLE001 - the sender must outlive any one bad message
                log.exception("event delivery crashed on one message; it is dropped")
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def _deliver(self, kind: str, event_id: str, body: bytes) -> None:
        if kind == "patch" and event_id in self._lost_set:
            self.stats.patches_skipped += 1
            return
        delay = self.backoff_s
        for attempt in range(1, self.max_attempts + 1):
            status = None
            try:
                status = self._post(self.url, self._headers(), body, self.timeout_s)
            except Exception as exc:  # noqa: BLE001 - transport errors of any kind are retryable
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
            if status is not None:
                if 200 <= status < 300:
                    self.stats.delivered += 1
                    return
                if status == 409:
                    self.stats.delivered += 1
                    self.stats.duplicates += 1
                    return
                self.stats.last_error = f"HTTP {status}"
                if status not in RETRYABLE_STATUS:
                    self.stats.rejected += 1
                    if kind == "event":
                        self._mark_lost(event_id)
                    log.error("backend refused %s %s with HTTP %s; not retrying", kind, event_id, status)
                    return
            if attempt == self.max_attempts:
                break
            self.stats.retries += 1
            if self._stop.wait(delay):
                break
            delay = min(delay * 2, self.backoff_cap_s)
        self.stats.dropped_exhausted += 1
        if kind == "event":
            self._mark_lost(event_id)
        log.warning("backend unreachable: dropped %s %s after %d attempts (%s)",
                    kind, event_id, self.max_attempts, self.stats.last_error)


class EventSink:
    """Bounded in-memory ring of emitted events, an optional JSON Lines log, and optional delivery.

    Thread-safe. `emit()` is called on the pipeline thread and only ever does memory and local-file
    work plus an outbox append; `emit_patch()` is called from the explanation worker.
    """

    def __init__(self, maxlen: int = 500, log_path: Path | None = None,
                 delivery: BackendDelivery | None = None) -> None:
        self._events: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.log_path = Path(log_path) if log_path else None
        self.delivery = delivery
        self.total = 0
        self.patches = 0
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_log(self, message: dict) -> None:
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(message, ensure_ascii=False) + "\n")

    def emit(self, event: dict) -> None:
        with self._lock:
            self._events.append(event)
            self.total += 1
            self._append_log(event)
        if self.delivery is not None:
            self.delivery.enqueue(event)

    def emit_patch(self, event: dict, explanation: dict) -> dict:
        """Replace a created event's reason: update the ring copy (if it is still there), log it, deliver it.

        `event` is the created event the patch is for. It is passed whole, not looked up, so a patch for an
        event that has already aged out of the ring is still delivered to the backend.
        """
        patch = build_reason_patch(event, explanation)
        with self._lock:
            for i in range(len(self._events) - 1, -1, -1):
                if self._events[i].get("event_id") == event["event_id"]:
                    # a new dict: readers holding the old one never see it change under them
                    self._events[i] = {**self._events[i], "explanation": dict(explanation)}
                    break
            self.patches += 1
            self._append_log(patch)
        if self.delivery is not None:
            self.delivery.enqueue(patch)
        return patch

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._events)[-max(0, int(limit)):]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self.total = 0
            self.patches = 0

    def close(self) -> None:
        if self.delivery is not None:
            self.delivery.close()
