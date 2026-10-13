"""Batch- and incrementally-index sealed clips' keyframes.

Phase 9 has not landed yet on this branch, so there is no `evidence_blocks`
table to read clip paths from. This module works directly off the filesystem
instead, which is also exactly the shape Phase 9 will hand it: one clip file
per sealed event, optionally paired with a small JSON sidecar of the fields a
result needs (`camera_id`, `captured_at`, `chips`). When Phase 9 lands, its
sealing step can write that sidecar next to the clip it already writes, and
this module needs no change — or a caller can pass `ClipMeta` records
straight from `evidence_blocks`, since `backfill_dir` and `index_clips` are
split apart for exactly that reason.

    my-clip.mp4
    my-clip.json   # optional: {"camera_id": "RXL-01", "captured_at": "...", "chips": [...]}

A clip with no sidecar still indexes: `camera_id` falls back to the configured
default camera, `captured_at` to the file's mtime, and `chips` to [].
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from search.embed import SiglipEmbedder, get_embedder
from search.index import VectorIndex

log = logging.getLogger("truewatch.edge.search.backfill")


@dataclass
class ClipMeta:
    clip_id: str
    clip_path: Path
    camera_id: str
    captured_at: str  # RFC3339 UTC
    time_label: str
    chips: list[str] = field(default_factory=list)


def _time_label(dt: datetime) -> str:
    return dt.strftime("%d %b").upper().lstrip("0") + f" · {dt.strftime('%H:%M')}"


def discover_clips(clips_dir: Path, default_camera_id: str = "RXL-01") -> list[ClipMeta]:
    """Find every clip under `clips_dir` and pair it with its metadata."""
    clips_dir = Path(clips_dir)
    if not clips_dir.exists():
        return []
    out: list[ClipMeta] = []
    for clip_path in sorted(clips_dir.rglob("*.mp4")):
        sidecar = clip_path.with_suffix(".json")
        meta: dict = {}
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("could not read sidecar %s: %s", sidecar, exc)

        clip_id = str(meta.get("clip_id", clip_path.stem))
        camera_id = str(meta.get("camera_id", default_camera_id))
        captured_at = meta.get("captured_at")
        if captured_at is None:
            mtime = datetime.fromtimestamp(clip_path.stat().st_mtime, tz=timezone.utc)
            captured_at = mtime.isoformat()
        else:
            mtime = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
        time_label = str(meta.get("time_label") or _time_label(mtime))
        chips = list(meta.get("chips", []))
        out.append(
            ClipMeta(
                clip_id=clip_id,
                clip_path=clip_path,
                camera_id=camera_id,
                captured_at=captured_at,
                time_label=time_label,
                chips=chips,
            )
        )
    return out


def extract_keyframes(clip_path: Path, count: int = 3) -> list[np.ndarray]:
    """Evenly-spaced RGB keyframes from a clip, `count` of them (fewer if the clip is short)."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"could not open clip: {clip_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # Some containers under-report frame count; fall back to a full read.
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not frames:
                return []
            idxs = np.linspace(0, len(frames) - 1, num=min(count, len(frames)), dtype=int)
            return [frames[i] for i in idxs]

        n = max(1, min(count, total))
        idxs = np.linspace(0, total - 1, num=n, dtype=int)
        out = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return out
    finally:
        cap.release()


def index_clips(
    clips: list[ClipMeta],
    index: VectorIndex,
    embedder: SiglipEmbedder,
    keyframes_per_clip: int = 3,
    force: bool = False,
) -> int:
    """Embed and add each clip's keyframes to `index`. Returns clips actually indexed.

    Incremental by construction: a clip already present (by clip_id) is
    skipped unless `force`, and `VectorIndex.add` is itself idempotent per
    keyframe id, so re-running a backfill is always safe.
    """
    indexed = 0
    for clip in clips:
        if not force and index.contains_clip(clip.clip_id):
            continue
        try:
            frames = extract_keyframes(clip.clip_path, count=keyframes_per_clip)
        except ValueError as exc:
            log.warning("skipping clip %s: %s", clip.clip_id, exc)
            continue
        if not frames:
            log.warning("skipping clip %s: no readable frames", clip.clip_id)
            continue

        vectors = embedder.embed_images(frames)
        keyframe_ids = [f"{clip.clip_id}#{i}" for i in range(len(frames))]
        metadata = [
            {
                "clip_id": clip.clip_id,
                "camera_id": clip.camera_id,
                "captured_at": clip.captured_at,
                "time_label": clip.time_label,
                "chips": clip.chips,
                "keyframe_index": i,
            }
            for i in range(len(frames))
        ]
        index.add(keyframe_ids, vectors, metadata)
        indexed += 1
    return indexed


def backfill(
    clips_dir: Path,
    index_path: Path,
    model_id: str | None = None,
    keyframes_per_clip: int = 3,
    default_camera_id: str = "RXL-01",
    force: bool = False,
) -> dict:
    """Discover clips under `clips_dir`, index new ones, persist, return a summary."""
    t0 = time.time()
    embedder = get_embedder(model_id)
    index = VectorIndex.load_or_create(index_path, dim=embedder.dim)

    clips = discover_clips(clips_dir, default_camera_id=default_camera_id)
    indexed = index_clips(clips, index, embedder, keyframes_per_clip=keyframes_per_clip, force=force)
    index.save(index_path)

    summary = {
        "clips_found": len(clips),
        "clips_indexed": indexed,
        "keyframes_total": len(index),
        "took_ms": round((time.time() - t0) * 1000, 1),
    }
    log.info("backfill complete: %s", summary)
    return summary
