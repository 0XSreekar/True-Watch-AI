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
        default_factory=lambda: _str("BACKEND_BASE_URL", "http://localhost:4000").rstrip("/")
    )
    ingest_key: str = field(default_factory=lambda: _str("TRUEWATCH_INGEST_KEY"))

    # --- ingest source (Phase 1) ---
    ingest_mode: str = field(default_factory=lambda: _str("INGEST_MODE", "file").lower())
    rtsp_url: str = field(default_factory=lambda: _str("RTSP_URL"))
    replay_file_path: Path = field(
        default_factory=lambda: _path("REPLAY_FILE_PATH", "./var/samples/sample.mp4")
    )
    target_fps: float = field(default_factory=lambda: _float("TARGET_FPS", 8.0))
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

    # --- evidence (Phase 9) ---
    evidence_dir: Path = field(default_factory=lambda: _path("EVIDENCE_DIR", "./var/evidence"))

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
