"""TRUEWATCH edge inference service.

One instance runs per border out post, against that post's cameras. It reaches
the rest of the system through exactly one call — POST /api/ingest/event on the
Express backend — and never talks to the browser.

The ingest loop runs every frame through the detection pipeline (pipeline/service.py):
camera state, motion, the appearance detector, tracking and fusion. Fusion-agreed
tracks become truewatch.event.v1 payloads, readable at GET /pipeline/events until the
backend ingest route lands in Phase 8. Evidence hashing is Phase 9; see
docs/ARCHITECTURE_V2.md sections 1.3 and 5.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

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

    Every decoded frame goes through the detection pipeline (camera state, motion,
    appearance, tracking, fusion); fusion-agreed tracks become events.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.frames_seen = 0
        self.started_at: float | None = None
        self.last_frame_at: float | None = None
        self.last_error: str | None = None
        self.detector = None                  # models.detector_weights.Detector, set at boot
        self.service = None                   # pipeline.service.CameraService while running
        self.sink = None                      # pipeline.events.EventSink, kept across restarts
        self.explainer = None                 # pipeline.explain.EventExplainer, kept across restarts
        self.keyframes = None                 # search.incremental.KeyframeIndexer (SEARCH_ENABLED)
        self.evidence = None                  # evidence.EvidenceStore (EVIDENCE_ENABLED), kept across restarts

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
        service = None
        try:
            service = self.service = build_service(cfg, self.detector, self.sink, self.explainer, self.keyframes,
                                                         evidence=self.evidence)
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
                service.process(frame)
        except IngestError as exc:
            self.last_error = str(exc)
            log.error("ingest failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 - the thread must not die silently
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.exception("ingest crashed")
        finally:
            if service is not None:
                service.close()

    def snapshot(self) -> dict:
        return {
            "running": self.running,
            "frames_seen": self.frames_seen,
            "started_at": self.started_at,
            "last_frame_at": self.last_frame_at,
            "last_error": self.last_error,
            "detection": self.service.stats.to_dict() if self.service is not None else None,
            "events_emitted": self.sink.total if self.sink is not None else 0,
            "observations": self.sink.observations if self.sink is not None else 0,
            "reason_patches": self.sink.patches if self.sink is not None else 0,
            "delivery": self.sink.delivery.stats.to_dict() if self.sink is not None and self.sink.delivery else None,
            "explain": self.explainer.stats() if self.explainer is not None else None,
            "search_index": self.keyframes.stats() if self.keyframes is not None else None,
        }


def build_service(cfg, detector, sink, explainer=None, keyframes=None, evidence=None):
    """The per-camera detection loop: the same Pipeline pipeline/demo.py drives, fed by ingest.

    Imported here, not at module import time, so `import app` stays light (no scipy/onnxruntime
    until a pipeline is actually started).
    """
    from pipeline.appearance import AppearanceChannel
    from pipeline.events import EventSink
    from pipeline.service import CameraService
    from anpr.stage import PlateStage
    from rules.config import ConfigStore

    from pipeline.appearance import AppearanceConfig
    from pipeline.tiling import preset as tiling_preset

    appearance = (AppearanceChannel.from_detector(detector, AppearanceConfig(tiling=tiling_preset(cfg.detector_tiling)))
                  if detector is not None else None)
    if appearance is None:
        log.warning("no detector loaded: the pipeline runs motion and camera state only, and fusion "
                    "cannot agree on anything, so no detection events will be emitted")
    face = None
    if cfg.face_enabled:
        from face.stage import FaceStage    # needs cryptography (sealing); only imported when enabled

        face = FaceStage(cfg.face_store_dir, min_height_frac=cfg.face_min_person_height_frac)
        if not face.warm_up():
            face = None
    return CameraService(
        appearance,
        camera_id=cfg.camera_id,
        post_id=cfg.post_id,
        sink=sink if sink is not None else EventSink(),
        stride=cfg.detector_stride,
        detector_max_interval_s=cfg.detector_max_interval_s,
        calibration_dir=cfg.calibration_dir,
        warmup_seconds=cfg.calibration_warmup_seconds,
        warmup_min_frames=cfg.calibration_warmup_min_frames,
        source=cfg.source_label,
        rules=ConfigStore(directory=cfg.rules_config_dir),
        anpr=PlateStage() if cfg.anpr_enabled else None,
        face=face,
        keyframes=keyframes,
        explainer=explainer,
        camera_name=cfg.camera_name,
        camera_sector=cfg.camera_sector,
        sealer=_sealer(cfg, evidence),
        alert_policy=cfg.alert_policy,
    )


def _sealer(cfg, evidence):
    if evidence is None:
        return None
    from pipeline.sealing import ClipSealer

    return ClipSealer(evidence, cfg.camera_id, clip_seconds=cfg.evidence_clip_seconds)


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
    from pipeline.events import BackendDelivery, EventSink
    from pipeline.explain import build_explainer

    state.detector = app.state.detector
    delivery = None
    if cfg.delivery_enabled and cfg.ingest_key:
        delivery = BackendDelivery(cfg.backend_base_url, cfg.ingest_key, timeout_s=cfg.delivery_timeout_s,
                                   max_attempts=cfg.delivery_max_attempts, outbox_max=cfg.delivery_outbox_max)
        log.info("delivering events to %s/api/ingest/event", cfg.backend_base_url)
    else:
        log.warning("event delivery to the backend is OFF (%s); events stay in memory and the log file",
                    "EDGE_DELIVERY_ENABLED is false" if not cfg.delivery_enabled else "TRUEWATCH_INGEST_KEY is not set")
    state.sink = EventSink(log_path=Path(cfg.events_log_path) if cfg.events_log_path else None, delivery=delivery)
    state.explainer = build_explainer(cfg)
    if cfg.evidence_enabled:
        from evidence import EvidenceStore

        # opening the store resumes the chain after an unclean stop and logs the gap (slide 4)
        state.evidence = EvidenceStore(cfg.evidence_dir, hq_dir=cfg.evidence_hq_dir)
    feeds = None
    if cfg.search_enabled:
        from search.incremental import build_search_feeds

        feeds = build_search_feeds(cfg)
        state.keyframes = feeds[0] if feeds else None
    log.info(
        "edge service ready: mode=%s source=%s target_fps=%s",
        cfg.ingest_mode,
        cfg.source_label,
        cfg.target_fps,
    )
    yield
    state.stop()
    if state.explainer is not None:
        state.explainer.close()
    if feeds is not None:
        feeds[0].close()
        feeds[1].close()
    if state.sink is not None:
        state.sink.close()
    if state.evidence is not None:
        state.evidence.close()


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


@app.get("/pipeline/events")
def pipeline_events(limit: int = 50) -> dict:
    """The most recent truewatch.event.v1 payloads this instance emitted, oldest first.

    Under ALERT_POLICY=rules the list mixes operator alerts (delivered to the backend) with local track
    observations, which carry "kind": "observation" and are never delivered. `total` counts alerts only.
    """
    events = state.sink.recent(limit) if state.sink is not None else []
    return {"count": len(events), "total": state.sink.total if state.sink is not None else 0,
            "observations": state.sink.observations if state.sink is not None else 0, "events": events}


class FeedbackRequest(BaseModel):
    """What the backend posts when an operator acts on or dismisses an alert."""

    action: str
    alert_id: str | None = None
    event_id: str | None = None
    camera_id: str | None = None
    track_id: int | str | None = None
    origin: str | None = None


@app.post("/pipeline/feedback")
def pipeline_feedback(body: FeedbackRequest) -> dict:
    """A dismissal updates this camera's baseline; any other action is acknowledged and ignored."""
    service = state.service
    if body.action != "dismiss" or service is None or body.track_id is None:
        return {"applied": False, "rules": []}
    if body.camera_id and body.camera_id != service.camera_id:
        return {"applied": False, "rules": []}
    try:
        track_id = int(body.track_id)
    except (TypeError, ValueError):
        return {"applied": False, "rules": []}
    rules = service.on_dismissal(track_id)
    return {"applied": bool(rules), "rules": rules}


@app.get("/evidence/chain")
def evidence_chain(limit: int = 5) -> list[dict]:
    """The tail of this post's hash chain in the FROZEN `GET /api/evidence/chain` shape (+ full_hash)."""
    from evidence import chain_view

    return chain_view(app.state.config.evidence_dir, limit=max(1, min(int(limit), 500)))


@app.post("/evidence/verify")
def evidence_verify() -> dict:
    """Recompute clips, blocks, links, daily roots and signatures: FROZEN `{intact, verified, total, blocks}`."""
    from evidence import verify_view

    cfg = app.state.config
    return verify_view(cfg.evidence_dir, hq_dir=cfg.evidence_hq_dir)


class SearchRequest(BaseModel):
    query: str


@app.post("/api/search")
def api_search(body: SearchRequest) -> dict:
    """Semantic footage search (Phase 6), mounted behind SEARCH_ENABLED.

    Off by default so a deployment that hasn't provisioned SigLIP (or backfilled
    an index yet) never pays that cost, and so this route cannot regress any
    earlier phase's tests. Phase 8 will decide whether the backend proxies to
    this route or the edge posts results upstream; either shape is compatible
    with returning the same body query.run_search already produces.

    The heavy imports (torch, transformers) happen here, inside the handler,
    not at module import time, so importing edge.app never requires them.
    """
    cfg = app.state.config
    if not cfg.search_enabled:
        raise HTTPException(status_code=404, detail="search is not enabled on this instance")
    from search.query import search_with_config

    return search_with_config(cfg, body.query)
