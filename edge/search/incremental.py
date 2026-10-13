"""Keep the search index current while the service runs (no backfill needed for new footage).

Two feeds, both writing into the SAME in-memory `VectorIndex` that `POST /api/search` reads
(`search.query.get_index`), so a keyframe is searchable the moment it is embedded, and both saving the index
to `SEARCH_INDEX_PATH` so a restart keeps it:

`KeyframeIndexer`   The live path. The service hands it the frame of every event fusion has agreed on (both
                    channels confirm: the alert the operator will be asked to seal). A worker thread embeds
                    it with SigLIP and adds it under the track's clip id; later events on the same track add
                    further keyframes to that same clip (up to `MAX_KEYFRAMES_PER_CLIP`), so one person or
                    vehicle is one search result, not one per event. `submit()` never blocks the pipeline:
                    the queue is bounded and, when full, the OLDEST waiting keyframe is dropped and counted.

`ClipSync`          The sealed path. Once the evidence stage (Phase 9) writes sealed clips under
                    `SEARCH_CLIPS_DIR`, this scans for ones not yet indexed and adds three keyframes each via
                    `search.backfill.index_clips`, the same code the batch backfill uses. A clip whose
                    sidecar carries the event's id as `clip_id` joins the keyframes already indexed for that
                    event; nothing is indexed twice (`index_clips` skips a known clip id).

Both are off unless `SEARCH_ENABLED` is set, and both are wrapped so an embedding failure (model missing,
out of memory) is logged and counted, never raised into the pipeline.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Callable

import numpy as np

log = logging.getLogger("truewatch.edge.search.incremental")

MAX_KEYFRAMES_PER_CLIP = 3
EMBED_MAX_SIDE = 512          # SigLIP resizes to 224 itself; this only bounds queued memory
BATCH = 8


def _rgb_small(image_bgr: np.ndarray) -> np.ndarray:
    import cv2

    h, w = image_bgr.shape[:2]
    scale = min(1.0, EMBED_MAX_SIDE / float(max(h, w)))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image_bgr[:, :, ::-1])


def _time_label(dt: datetime) -> str:
    return dt.strftime("%d %b").upper().lstrip("0") + f" · {dt.strftime('%H:%M')}"


def _chips(event: dict) -> list[str]:
    """Short human tags for a result card, from what the event already knows (no model involved)."""
    det = event.get("detection") or {}
    chips = [str(det.get("class", "")).replace("_", " ")] if det.get("class") else []
    if det.get("ir"):
        chips.append("IR")
    chips.extend((event.get("rule") or {}).get("fired") or [])
    plate = (event.get("anpr") or {}).get("text")
    if plate:
        chips.append(f"plate {plate}")
    return [c for c in chips if c]


class KeyframeIndexer:
    def __init__(self, embedder, index, index_path, *, queue_max: int = 32, save_every_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.embedder = embedder
        self.index = index
        self.index_path = index_path
        self.save_every_s = float(save_every_s)
        self._clock = clock
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(queue_max)))
        self._clips: OrderedDict[tuple[str, int | None], tuple[str, int]] = OrderedDict()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._dirty = False
        self._last_save = clock()
        self.submitted = 0
        self.indexed = 0
        self.dropped = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, name="search-keyframes", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- producer

    def submit(self, event: dict, image_bgr: np.ndarray) -> bool:
        """Queue one event's frame. Never blocks; when the queue is full the oldest waiting frame is dropped."""
        try:
            det = event.get("detection") or {}
            key = (str(event.get("camera_id", "")), det.get("track_id"))
            with self._lock:
                clip_id, n = self._clips.get(key, (str(event["event_id"]), 0))
                if n >= MAX_KEYFRAMES_PER_CLIP:
                    return False
                self._clips[key] = (clip_id, n + 1)
                self._clips.move_to_end(key)
                while len(self._clips) > 2048:
                    self._clips.popitem(last=False)
            captured = datetime.fromisoformat(str(event["captured_at"]).replace("Z", "+00:00"))
            item = (f"{clip_id}#{n}", {
                "clip_id": clip_id,
                "camera_id": event.get("camera_id", ""),
                "captured_at": captured.astimezone(timezone.utc).isoformat(),
                "time_label": _time_label(captured),
                "chips": _chips(event),
                "keyframe_index": n,
            }, _rgb_small(image_bgr))
            while True:
                try:
                    self._q.put_nowait(item)
                    break
                except queue.Full:
                    try:
                        self._q.get_nowait()
                        self.dropped += 1
                    except queue.Empty:
                        continue
            self.submitted += 1
            return True
        except Exception:  # noqa: BLE001 - indexing is best effort and must never reach the pipeline
            self.errors += 1
            log.exception("could not queue a keyframe for indexing")
            return False

    # ---------------------------------------------------------------- worker

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = []
            try:
                batch.append(self._q.get(timeout=0.5))
                while len(batch) < BATCH:
                    batch.append(self._q.get_nowait())
            except queue.Empty:
                pass
            if batch:
                self._index(batch)
            self._maybe_save()

    def _index(self, batch: list) -> None:
        try:
            vectors = self.embedder.embed_images([b[2] for b in batch])
            self.index.add([b[0] for b in batch], vectors, [b[1] for b in batch])
            self.indexed += len(batch)
            self._dirty = True
        except Exception:  # noqa: BLE001
            self.errors += 1
            log.exception("keyframe embedding failed; %d keyframes not indexed", len(batch))

    def _maybe_save(self, force: bool = False) -> None:
        if not self._dirty or (not force and self._clock() - self._last_save < self.save_every_s):
            return
        try:
            self.index.save(self.index_path)
            self._dirty = False
            self._last_save = self._clock()
        except Exception:  # noqa: BLE001
            self.errors += 1
            log.exception("could not save the search index to %s", self.index_path)

    def flush(self, timeout_s: float = 30.0) -> bool:
        """Wait for queued keyframes to be embedded, then save (shutdown and tests)."""
        deadline = time.monotonic() + timeout_s
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.05)
        self._maybe_save(force=True)
        return self._q.empty()

    def close(self) -> None:
        self.flush(5.0)
        self._stop.set()

    def stats(self) -> dict:
        return {"submitted": self.submitted, "indexed": self.indexed, "dropped": self.dropped,
                "errors": self.errors, "pending": self._q.qsize(), "keyframes_total": len(self.index)}


class ClipSync:
    """Periodically index newly sealed clips under `clips_dir` into the shared index."""

    def __init__(self, cfg, embedder, index, *, every_s: float = 60.0) -> None:
        self.cfg, self.embedder, self.index, self.every_s = cfg, embedder, index, every_s
        self._stop = threading.Event()
        self.clips_indexed = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, name="search-clip-sync", daemon=True)
        self._thread.start()

    def sync_once(self) -> int:
        from search.backfill import discover_clips, index_clips

        clips = discover_clips(self.cfg.search_clips_dir, default_camera_id=self.cfg.camera_id)
        fresh = [c for c in clips if not self.index.contains_clip(c.clip_id)]
        if not fresh:
            return 0
        n = index_clips(fresh, self.index, self.embedder, keyframes_per_clip=self.cfg.search_keyframes_per_clip)
        if n:
            self.index.save(self.cfg.search_index_path)
            self.clips_indexed += n
        return n

    def _run(self) -> None:
        while not self._stop.wait(self.every_s):
            try:
                self.sync_once()
            except Exception:  # noqa: BLE001
                self.errors += 1
                log.exception("sealed-clip index sync failed")

    def close(self) -> None:
        self._stop.set()


def build_search_feeds(cfg) -> tuple[KeyframeIndexer, ClipSync] | None:
    """The live index feeds for a deployment with SEARCH_ENABLED, else None. Loads nothing heavy here:
    SigLIP loads on the first embedding, in the worker thread."""
    if not cfg.search_enabled:
        return None
    from search.embed import get_embedder
    from search.query import get_index

    embedder = get_embedder(cfg.embed_model_id)
    # `dim` is known without loading the model only after first use; SigLIP base is 768. A mismatch with an
    # existing index file is caught by VectorIndex.add and counted as an error, not a crash.
    index = get_index(cfg.search_index_path, dim=768)
    return KeyframeIndexer(embedder, index, cfg.search_index_path), ClipSync(cfg, embedder, index)
