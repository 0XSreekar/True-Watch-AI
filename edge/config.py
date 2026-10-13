"""Configuration for the edge service.

Every value comes from the environment. Nothing is hardcoded, and no secret has
a usable default: a process that needs a secret and does not find one refuses to
start rather than falling back to something insecure.

See docs/ARCHITECTURE_V2.md section 8 for the full variable list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EDGE_ROOT = Path(__file__).resolve().parent


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default


def _path(name: str, default: str) -> Path:
    raw = _str(name, default) or default
    p = Path(raw)
    return p if p.is_absolute() else (EDGE_ROOT / p).resolve()


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class ConfigError(RuntimeError):
    """Raised when a required value is missing or contradictory."""


@dataclass(frozen=True)
class Config:
    # --- service ---
    host: str = field(default_factory=lambda: _str("EDGE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("EDGE_PORT", 8000))
    log_level: str = field(default_factory=lambda: _str("EDGE_LOG_LEVEL", "info").lower())
    env: str = field(default_factory=lambda: _str("EDGE_ENV", "development").lower())

    # --- where events go (Phase 8) ---
    backend_base_url: str = field(
        default_factory=lambda: (_str("BACKEND_BASE_URL") or _str("BACKEND_URL") or "http://localhost:4000").rstrip("/")
    )
    ingest_key: str = field(default_factory=lambda: _str("TRUEWATCH_INGEST_KEY"))
    # Deliver events to the backend (pipeline/events.py BackendDelivery). Needs TRUEWATCH_INGEST_KEY; without
    # one the edge logs that delivery is off and keeps events in memory / the JSON Lines log only.
    delivery_enabled: bool = field(default_factory=lambda: _bool("EDGE_DELIVERY_ENABLED", True))
    delivery_timeout_s: float = field(default_factory=lambda: _float("EDGE_DELIVERY_TIMEOUT_S", 3.0))
    delivery_max_attempts: int = field(default_factory=lambda: _int("EDGE_DELIVERY_MAX_ATTEMPTS", 5))
    delivery_outbox_max: int = field(default_factory=lambda: _int("EDGE_DELIVERY_OUTBOX_MAX", 1000))

    # --- ingest source (Phase 1) ---
    ingest_mode: str = field(default_factory=lambda: _str("INGEST_MODE", "file").lower())
    rtsp_url: str = field(default_factory=lambda: _str("RTSP_URL"))
    replay_file_path: Path = field(
        default_factory=lambda: _path("REPLAY_FILE_PATH", "./var/samples/sample.mp4")
    )
    # 25: keep every frame of a 20-25 fps camera. The motion score is calibrated on frame-to-frame flow and the
    # warm-up is counted in frames, so decimating a 20 fps clip to 10 fps cut motion-gate passes from 8% to 2%
    # and left the sample clips with no alerts (docs/VERIFICATION_REPORT.md section 5).
    target_fps: float = field(default_factory=lambda: _float("TARGET_FPS", 25.0))
    camera_id: str = field(default_factory=lambda: _str("CAMERA_ID", "RXL-01"))
    post_id: str = field(default_factory=lambda: _str("POST_ID", "RXL"))

    # --- appearance-channel detector (Phase 2); see models/detector_weights.py ---
    # URL + SHA-256 of the exported ONNX. Both set, or both empty to use the committed manifest.
    yolo_model_url: str = field(default_factory=lambda: _str("YOLO_MODEL_URL"))
    yolo_model_sha256: str = field(default_factory=lambda: _str("YOLO_MODEL_SHA256").lower())
    # training/results/hf_model.json in a repository checkout; a container sets the two values above.
    yolo_model_manifest: Path = field(
        default_factory=lambda: _path("YOLO_MODEL_MANIFEST", "../training/results/hf_model.json")
    )
    model_cache_dir: Path = field(default_factory=lambda: _path("MODEL_CACHE_DIR", "./models/cache"))
    yolo_img_size: int = field(default_factory=lambda: _int("YOLO_IMG_SIZE", 640))
    onnx_provider: str = field(default_factory=lambda: _str("ONNX_PROVIDER", "CPUExecutionProvider"))

    # --- detection pipeline in the service loop (Phases 3-5) ---
    # Run the detector every Nth ingested frame; motion, tracking and fusion run on every frame.
    detector_stride: int = field(default_factory=lambda: _int("DETECTOR_STRIDE", 2))
    # Per-camera learnt thresholds (pipeline/calibration.py) are written here.
    calibration_dir: Path = field(default_factory=lambda: _path("CALIBRATION_DIR", "./var/calibration"))
    # Unlabelled frames a camera with no stored calibration learns its thresholds from before fusing.
    calibration_warmup_frames: int = field(default_factory=lambda: _int("CALIBRATION_WARMUP_FRAMES", 100))
    # Per-camera rule files <camera_id>.json (rules/config.py format; rules/examples/ has a sample).
    # No file for a camera = its rules are off (logged), detection events still flow.
    rules_config_dir: Path = field(default_factory=lambda: _path("RULES_CONFIG_DIR", "./var/rules"))
    # Read plates on fusion-agreed vehicle tracks (anpr/stage.py). Needs paddleocr for recognition;
    # without it plates are still located and counted.
    anpr_enabled: bool = field(default_factory=lambda: _bool("ANPR_ENABLED", True))
    # Optional JSON Lines log of every emitted truewatch.event.v1 (empty = in-memory only).
    events_log_path: str = field(default_factory=lambda: _str("EVENTS_LOG_PATH"))

    # --- camera context handed to the explanation model (Phase 6) ---
    camera_name: str = field(default_factory=lambda: _str("CAMERA_NAME"))
    camera_sector: str = field(default_factory=lambda: _str("CAMERA_SECTOR"))

    # --- plain-language reason (Phase 6, vlm/) ---
    # "auto" (default): run the local Moondream 2 only if its pinned weights are already in the local
    # Hugging Face cache, so a boot never starts a surprise multi-gigabyte download and an air-gapped box
    # never reaches out. "1": load it (downloading if needed). "0": never; every reason stays the rule string.
    vlm_enabled: str = field(default_factory=lambda: _str("VLM_ENABLED", "auto").lower())
    # Explanation jobs waiting for the model; when full the OLDEST waiting job is dropped (its alert keeps
    # the rule-string reason). See vlm/queue.py.
    vlm_queue_max: int = field(default_factory=lambda: _int("VLM_QUEUE_MAX", 8))
    vlm_timeout_s: float = field(default_factory=lambda: _float("VLM_TIMEOUT_S", 60.0))
    # Seconds the model rests between explanations (0 = back to back). A busy model shares compute with the
    # frame loop on some machines; raising this caps how much of it the model may take.
    vlm_cooldown_s: float = field(default_factory=lambda: _float("VLM_COOLDOWN_S", 0.0))
    # Frames are downscaled to this longest side before they wait in the queue (memory bound).
    vlm_frame_max_side: int = field(default_factory=lambda: _int("VLM_FRAME_MAX_SIDE", 768))

    # --- face detection on close-range person tracks (Phase 5 module, wired here) ---
    # Off by default: it costs frame rate (measured in edge/README.md) and stores personal data.
    face_enabled: bool = field(default_factory=lambda: _bool("FACE_ENABLED", False))
    # A person track is "close range" when its box is at least this fraction of the frame height.
    face_min_person_height_frac: float = field(default_factory=lambda: _float("FACE_MIN_PERSON_HEIGHT_FRAC", 0.25))
    face_store_dir: Path = field(default_factory=lambda: _path("FACE_STORE_DIR", "./var/face_crops"))

    # --- evidence (Phase 9) ---
    evidence_dir: Path = field(default_factory=lambda: _path("EVIDENCE_DIR", "./var/evidence"))
    # Sector-HQ mirror of the signed daily roots (a second local directory; never a network path).
    evidence_hq_dir: Path = field(default_factory=lambda: _path("EVIDENCE_HQ_DIR", "./var/evidence-hq"))
    # Seal a clip for every emitted event: the last EVIDENCE_CLIP_SECONDS of frames, hashed as written.
    evidence_enabled: bool = field(default_factory=lambda: _bool("EVIDENCE_ENABLED", True))
    evidence_clip_seconds: float = field(default_factory=lambda: _float("EVIDENCE_CLIP_SECONDS", 4.0))

    # --- semantic footage search (Phase 6) ---
    # Off by default: mounting the route, and loading SigLIP, only happens when
    # a deployment opts in. Existing endpoints and tests are unaffected either way.
    search_enabled: bool = field(default_factory=lambda: _bool("SEARCH_ENABLED", False))
    embed_model_id: str = field(
        default_factory=lambda: _str("EMBED_MODEL_ID", "google/siglip-base-patch16-224")
    )
    search_clips_dir: Path = field(
        default_factory=lambda: _path("SEARCH_CLIPS_DIR", "./var/evidence")
    )
    search_index_path: Path = field(
        default_factory=lambda: _path("SEARCH_INDEX_PATH", "./var/search/index.npz")
    )
    search_keyframes_per_clip: int = field(
        default_factory=lambda: _int("SEARCH_KEYFRAMES_PER_CLIP", 3)
    )
    search_default_limit: int = field(default_factory=lambda: _int("SEARCH_DEFAULT_LIMIT", 6))

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def source_label(self) -> str:
        """The honesty flag carried on every event: 'rtsp' or 'file'.

        docs/PHASE_MINUS1_SCOPE.md section 6 requires that a replayed file is
        never presented as a live camera.
        """
        return "rtsp" if self.ingest_mode == "rtsp" else "file"

    def validate(self) -> None:
        if self.ingest_mode not in {"rtsp", "file"}:
            raise ConfigError(f"INGEST_MODE must be 'rtsp' or 'file', got {self.ingest_mode!r}")
        if self.ingest_mode == "rtsp" and not self.rtsp_url:
            raise ConfigError("INGEST_MODE=rtsp requires RTSP_URL to be set")
        if self.target_fps <= 0:
            raise ConfigError(f"TARGET_FPS must be positive, got {self.target_fps}")
        if bool(self.yolo_model_url) != bool(self.yolo_model_sha256):
            raise ConfigError(
                "YOLO_MODEL_URL and YOLO_MODEL_SHA256 must be set together (a URL is only trusted with the "
                "hash of the file it serves), or both left empty to use YOLO_MODEL_MANIFEST"
            )
        if self.detector_stride < 1:
            raise ConfigError(f"DETECTOR_STRIDE must be at least 1, got {self.detector_stride}")
        if self.yolo_img_size < 32:
            raise ConfigError(f"YOLO_IMG_SIZE must be at least 32, got {self.yolo_img_size}")
        if self.is_production and not self.ingest_key:
            raise ConfigError(
                "TRUEWATCH_INGEST_KEY is required in production. "
                "Generate one with: openssl rand -hex 32"
            )


def load() -> Config:
    cfg = Config()
    cfg.validate()
    return cfg
