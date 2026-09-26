"""Face privacy: blur-by-default storage, access-controlled unsealing, retention.

Implements slide 4's privacy commitment word for word: "face data stays on the
post, 30-day retention, unsealing is access-controlled."

- Every stored clip/crop carries the BLURRED face by default
  (`blur_faces_in_frame`). The unblurred crop is never written to disk in the
  clear: it is encrypted immediately (`seal_face_crop`) with a key that only
  the Supervisor role can use to decrypt.
- `unseal()` is the only way to get the unblurred crop back. It refuses
  anything but the Supervisor role, and EVERY call — granted or refused —
  appends one row to the audit log (`AuditLog`): who, when, which clip, why.
  Nothing about a refusal is silent.
- `purge_expired()` deletes any sealed crop older than `RETENTION_DAYS` (30)
  and audits the purge as its own row, so the retention policy is provable
  from the log, not just asserted by code comments.

The encryption key comes from `FACE_UNSEAL_KEY` in the environment (a
Fernet key, `cryptography.fernet.Fernet.generate_key()`), matching how
`edge/config.py` treats every other secret: it comes from the environment and
nothing falls back to a hardcoded value. Unlike `edge/config.py`'s `Config`,
this module manages its own environment lookups rather than adding fields
there, because Phase 5 does not own that file.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger("truewatch.edge.face.privacy")

SUPERVISOR_ROLE = "supervisor"
RETENTION_DAYS = 30
RETENTION_SECONDS = RETENTION_DAYS * 24 * 3600

DEFAULT_STORE_DIR = Path(__file__).resolve().parent.parent / "var" / "face_crops"
DEFAULT_AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "var" / "face_audit" / "audit_log.jsonl"


class PrivacyError(RuntimeError):
    """A privacy invariant could not be honoured (missing key, bad role, corrupt seal)."""


class UnsealRefused(PrivacyError):
    """Raised when a non-Supervisor caller asks to unseal a face crop."""


# --------------------------------------------------------------------------------------------
# Blur-by-default
# --------------------------------------------------------------------------------------------


def blur_region(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Return `frame_bgr` with a strong Gaussian blur over `bbox`. Does not mutate the input."""
    x1, y1, x2, y2 = bbox
    out = frame_bgr.copy()
    region = out[y1:y2, x1:x2]
    if region.size == 0:
        return out
    # Kernel scales with the face box so a close-range (large) face is not left
    # partially legible by a fixed small kernel.
    k = max(15, (min(region.shape[0], region.shape[1]) // 2) | 1)  # odd, >= 15
    out[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
    return out


def blur_faces_in_frame(frame_bgr: np.ndarray, faces) -> np.ndarray:
    """Blur every detected face. This is what gets written to a stored clip/thumbnail."""
    out = frame_bgr
    for face in faces:
        out = blur_region(out, face.bbox)
    return out


# --------------------------------------------------------------------------------------------
# Encryption at rest for the unblurred crop
# --------------------------------------------------------------------------------------------


def _load_key() -> bytes:
    raw = os.environ.get("FACE_UNSEAL_KEY", "").strip()
    if not raw:
        raise PrivacyError(
            "FACE_UNSEAL_KEY is not set. Generate one with "
            "`python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"` "
            "and hold it with the Supervisor role only; it must never be committed."
        )
    try:
        return raw.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PrivacyError("FACE_UNSEAL_KEY must be ASCII (a Fernet key is base64url bytes)") from exc


def seal_face_crop(crop_bgr: np.ndarray) -> bytes:
    """Encrypt an unblurred face crop for storage. Only `unseal()` with a valid key reverses this."""
    ok, encoded = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise PrivacyError("could not encode the face crop before sealing")
    fernet = Fernet(_load_key())
    return fernet.encrypt(encoded.tobytes())


def _decrypt(token: bytes) -> np.ndarray:
    fernet = Fernet(_load_key())
    try:
        plaintext = fernet.decrypt(token)
    except InvalidToken as exc:
        raise PrivacyError("the sealed crop does not decrypt with FACE_UNSEAL_KEY (wrong key or corrupt data)") from exc
    array = np.frombuffer(plaintext, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise PrivacyError("decrypted bytes did not decode as an image")
    return image


# --------------------------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditRow:
    audit_id: str
    action: str  # 'unseal.granted' | 'unseal.refused' | 'retention.purge'
    role: str
    clip_id: str
    reason: str
    at: float  # time.time()

    def to_json(self) -> dict:
        return {
            "audit_id": self.audit_id,
            "action": self.action,
            "role": self.role,
            "clip_id": self.clip_id,
            "reason": self.reason,
            "at": self.at,
        }


class AuditLog:
    """Append-only JSONL log. Every unseal attempt, granted or refused, writes exactly one row."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or DEFAULT_AUDIT_LOG_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, action: str, role: str, clip_id: str, reason: str) -> AuditRow:
        row = AuditRow(audit_id=str(uuid.uuid4()), action=action, role=role, clip_id=clip_id, reason=reason, at=time.time())
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
        return row

    def read_all(self) -> list[dict]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


# --------------------------------------------------------------------------------------------
# Unseal
# --------------------------------------------------------------------------------------------


def unseal(sealed_token: bytes, role: str, clip_id: str, reason: str, audit_log: AuditLog | None = None) -> np.ndarray:
    """Return the unblurred face crop, IF AND ONLY IF `role` is Supervisor.

    Every call writes an audit row, before returning or raising, so a refusal
    is exactly as visible in the log as a grant.
    """
    audit_log = audit_log or AuditLog()
    role_normalised = (role or "").strip().lower()
    if not reason.strip():
        audit_log.record("unseal.refused", role_normalised or "unknown", clip_id, "refused: no reason given")
        raise UnsealRefused("a reason is required to unseal a face crop")
    if role_normalised != SUPERVISOR_ROLE:
        audit_log.record("unseal.refused", role_normalised or "unknown", clip_id, reason)
        raise UnsealRefused(f"role {role!r} is not authorised to unseal face crops; only {SUPERVISOR_ROLE!r} may")

    image = _decrypt(sealed_token)
    audit_log.record("unseal.granted", role_normalised, clip_id, reason)
    return image


# --------------------------------------------------------------------------------------------
# Storage + retention
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredCrop:
    clip_id: str
    path: Path
    sealed_at: float


def store_sealed_crop(crop_bgr: np.ndarray, clip_id: str, store_dir: Path | None = None) -> StoredCrop:
    """Seal a face crop and write it to disk. The clear crop is never written."""
    store_dir = Path(store_dir or DEFAULT_STORE_DIR)
    store_dir.mkdir(parents=True, exist_ok=True)
    token = seal_face_crop(crop_bgr)
    path = store_dir / f"{clip_id}.sealed"
    path.write_bytes(token)
    sealed_at = time.time()
    os.utime(path, (sealed_at, sealed_at))
    return StoredCrop(clip_id=clip_id, path=path, sealed_at=sealed_at)


def purge_expired(
    store_dir: Path | None = None,
    now: float | None = None,
    retention_seconds: int = RETENTION_SECONDS,
    audit_log: AuditLog | None = None,
) -> list[Path]:
    """Delete every sealed crop whose mtime is older than `retention_seconds`. Returns what was purged."""
    store_dir = Path(store_dir or DEFAULT_STORE_DIR)
    audit_log = audit_log or AuditLog()
    now = time.time() if now is None else now
    purged: list[Path] = []
    if not store_dir.is_dir():
        return purged
    for path in sorted(store_dir.glob("*.sealed")):
        age = now - path.stat().st_mtime
        if age > retention_seconds:
            clip_id = path.stem
            path.unlink()
            purged.append(path)
            audit_log.record(
                "retention.purge",
                "system",
                clip_id,
                f"retention window exceeded: {age / 86400:.1f} days old (limit {retention_seconds / 86400:.0f})",
            )
    return purged
