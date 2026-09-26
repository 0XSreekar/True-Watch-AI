"""TRUEWATCH edge inference service.

One instance runs per border out post, against that post's cameras. It reaches
the rest of the system through exactly one call — POST /api/ingest/event on the
Express backend — and never talks to the browser.

Phase 1 ships the service shell and the ingest path. The appearance and motion
channels, fusion, tracking, the rule engine and evidence hashing arrive in
Phases 2 through 9; see docs/ARCHITECTURE_V2.md sections 1.3 and 5.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import ConfigError, load
from models.detector_weights import DetectorWeightsError, load_at_boot
from pipeline.ingest import IngestError, frames, probe

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("truewatch.edge")

STARTED_AT = time.time()


class PipelineState:
    """The one running ingest loop, and the counters the console will ask for.

    Phase 1 decodes and counts. Phases 2 onward hang the channels off the same
    loop, so the loop's shape does not change again.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.frames_seen = 0
        self.started_at: float | None = None
        self.last_frame_at: float | None = None
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg) -> None:
        with self._lock:
            if self.running:
                raise HTTPException(status_code=409, detail="pipeline already running")
            self._stop.clear()
            self.frames_seen = 0
            self.last_error = None
            self.started_at = time.time()
            self._thread = threading.Thread(
                target=self._run, args=(cfg,), name="ingest", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
        self._thread = None

    def _run(self, cfg) -> None:
        source = cfg.rtsp_url if cfg.ingest_mode == "rtsp" else cfg.replay_file_path
        is_rtsp = cfg.ingest_mode == "rtsp"
        log.info("ingest starting: source=%s mode=%s", source, cfg.source_label)
        try:
            for frame in frames(
                source,
                is_rtsp=is_rtsp,
                camera_id=cfg.camera_id,
                target_fps=cfg.target_fps,
                loop=not is_rtsp,
                pace=not is_rtsp,  # a replay must run at wall-clock speed, not disk speed
                stop=self._stop,
            ):
                self.frames_seen = frame.index + 1
                self.last_frame_at = frame.captured_at
                # Phases 2-9 attach here: appearance, motion, fusion, tracking,
                # rules, explanation, evidence. Phase 1 only counts.
        except IngestError as exc:
            self.last_error = str(exc)
            log.error("ingest failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 - the thread must not die silently
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.exception("ingest crashed")

    def snapshot(self) -> dict:
        return {
            "running": self.running,
            "frames_seen": self.frames_seen,
            "started_at": self.started_at,
            "last_frame_at": self.last_frame_at,
            "last_error": self.last_error,
        }


state = PipelineState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        cfg = load()
    except ConfigError as exc:
        log.error("configuration rejected: %s", exc)
        raise
    logging.getLogger().setLevel(cfg.log_level.upper())
    app.state.config = cfg
    # A configured detector that cannot be downloaded or verified stops the service: running without
    # the appearance channel while believing it is on would be worse than not starting.
    try:
        app.state.detector = load_at_boot(cfg)
    except DetectorWeightsError as exc:
        log.error("detector rejected: %s", exc)
        raise
    log.info(
        "edge service ready: mode=%s source=%s target_fps=%s",
        cfg.ingest_mode,
        cfg.source_label,
        cfg.target_fps,
    )
    yield
    state.stop()


app = FastAPI(
    title="TRUEWATCH edge",
    version="0.1.0",
    summary="Per-post inference service. Phase 1: service shell and ingest path.",
    lifespan=lifespan,
)


class StartRequest(BaseModel):
    """Empty for now. Per-camera overrides arrive with the registry in Phase 8."""


@app.get("/healthz")
def healthz() -> dict:
    """Liveness plus what this instance is actually pointed at.

    `source` is the honesty flag: 'file' means a replay, never a live camera.
    """
    cfg = app.state.config
    return {
        "status": "ok",
        "service": "truewatch-edge",
        "version": app.version,
        "uptime_s": round(time.time() - STARTED_AT, 3),
        "phase": 1,
        "ingest": {
            "mode": cfg.ingest_mode,
            "source": cfg.source_label,
            "camera_id": cfg.camera_id,
            "post_id": cfg.post_id,
            "target_fps": cfg.target_fps,
        },
        "pipeline": state.snapshot(),
        "detector": app.state.detector.describe() if getattr(app.state, "detector", None) else None,
    }


@app.get("/pipeline/probe")
def pipeline_probe() -> dict:
    """Open the configured source, read one frame, report what it is."""
    cfg = app.state.config
    source = cfg.rtsp_url if cfg.ingest_mode == "rtsp" else cfg.replay_file_path
    try:
        return probe(source, is_rtsp=cfg.ingest_mode == "rtsp")
    except IngestError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/pipeline/start", status_code=202)
def pipeline_start(_: StartRequest | None = None) -> dict:
    state.start(app.state.config)
    return {"accepted": True, "pipeline": state.snapshot()}


@app.post("/pipeline/stop")
def pipeline_stop() -> dict:
    state.stop()
    return {"stopped": True, "pipeline": state.snapshot()}


@app.get("/pipeline/status")
def pipeline_status() -> dict:
    return state.snapshot()
