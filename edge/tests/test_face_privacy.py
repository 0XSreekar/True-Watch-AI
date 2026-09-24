"""Face privacy: blur-by-default, Supervisor-only unsealing with an audit trail, retention purge."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from cryptography.fernet import Fernet

EDGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDGE_ROOT))

from face import privacy  # noqa: E402


@pytest.fixture(autouse=True)
def face_unseal_key(monkeypatch):
    """A real Fernet key for every test in this module; never a hardcoded/shared one."""
    monkeypatch.setenv("FACE_UNSEAL_KEY", Fernet.generate_key().decode("ascii"))


@pytest.fixture
def crop() -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.random((48, 48, 3)) * 255).astype("uint8")


def test_blur_region_obscures_the_face_box():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # A checkerboard inside the box, standing in for recognisable facial
    # detail: a Gaussian blur smooths its internal hard edges (a flat colour
    # block would blur to itself and prove nothing).
    frame[20:60, 20:60] = 255
    frame[30:50, 30:50] = 0
    original_edge_pixel = frame[30, 40].copy()

    blurred = privacy.blur_region(frame, (20, 20, 60, 60))

    # The internal edge is no longer a hard 0/255 transition.
    assert blurred[30, 40].tolist() != original_edge_pixel.tolist()
    assert 0 < int(blurred[30, 40][0]) < 255
    # Outside the box is untouched.
    assert frame[0, 0].tolist() == blurred[0, 0].tolist()
    assert frame[70, 70].tolist() == blurred[70, 70].tolist()


def test_seal_then_unseal_round_trips_the_crop(crop):
    token = privacy.seal_face_crop(crop)
    assert isinstance(token, bytes)
    assert token != crop.tobytes()  # not stored in the clear

    recovered = privacy._decrypt(token)  # internal, used by unseal() under a granted role
    assert recovered.shape == crop.shape


def test_unseal_without_supervisor_role_is_refused_and_audited(tmp_path, crop):
    audit = privacy.AuditLog(tmp_path / "audit.jsonl")
    token = privacy.seal_face_crop(crop)

    for role in ("operator", "watch-commander", "", "supervisorx"):
        with pytest.raises(privacy.UnsealRefused):
            privacy.unseal(token, role=role, clip_id="clip-1", reason="checking", audit_log=audit)

    rows = audit.read_all()
    assert len(rows) == 4
    assert all(row["action"] == "unseal.refused" for row in rows)
    assert all(row["clip_id"] == "clip-1" for row in rows)


def test_unseal_without_a_reason_is_refused_even_for_supervisor(tmp_path, crop):
    audit = privacy.AuditLog(tmp_path / "audit.jsonl")
    token = privacy.seal_face_crop(crop)
    with pytest.raises(privacy.UnsealRefused):
        privacy.unseal(token, role="supervisor", clip_id="clip-1", reason="   ", audit_log=audit)
    rows = audit.read_all()
    assert len(rows) == 1
    assert rows[0]["action"] == "unseal.refused"


def test_unseal_with_supervisor_role_is_granted_and_audited(tmp_path, crop):
    audit = privacy.AuditLog(tmp_path / "audit.jsonl")
    token = privacy.seal_face_crop(crop)

    recovered = privacy.unseal(token, role="Supervisor", clip_id="clip-2", reason="court order 44/2026", audit_log=audit)
    assert recovered.shape == crop.shape

    rows = audit.read_all()
    assert len(rows) == 1
    row = rows[0]
    assert row["action"] == "unseal.granted"
    assert row["role"] == "supervisor"  # normalised
    assert row["clip_id"] == "clip-2"
    assert row["reason"] == "court order 44/2026"
    assert isinstance(row["at"], float)
    assert row["audit_id"]


def test_every_unseal_attempt_writes_exactly_one_row_granted_or_refused(tmp_path, crop):
    audit = privacy.AuditLog(tmp_path / "audit.jsonl")
    token = privacy.seal_face_crop(crop)

    privacy.unseal(token, role="supervisor", clip_id="a", reason="r1", audit_log=audit)
    with pytest.raises(privacy.UnsealRefused):
        privacy.unseal(token, role="operator", clip_id="b", reason="r2", audit_log=audit)
    privacy.unseal(token, role="supervisor", clip_id="c", reason="r3", audit_log=audit)

    rows = audit.read_all()
    assert len(rows) == 3
    assert [r["action"] for r in rows] == ["unseal.granted", "unseal.refused", "unseal.granted"]


def test_wrong_key_cannot_decrypt_a_sealed_crop(tmp_path, crop, monkeypatch):
    token = privacy.seal_face_crop(crop)
    monkeypatch.setenv("FACE_UNSEAL_KEY", Fernet.generate_key().decode("ascii"))
    with pytest.raises(privacy.PrivacyError):
        privacy._decrypt(token)


def test_missing_key_refuses_to_seal(monkeypatch, crop):
    monkeypatch.delenv("FACE_UNSEAL_KEY", raising=False)
    with pytest.raises(privacy.PrivacyError, match="FACE_UNSEAL_KEY"):
        privacy.seal_face_crop(crop)


# --- Retention -------------------------------------------------------------------------------


def test_store_sealed_crop_never_writes_the_clear_image(tmp_path, crop):
    store = tmp_path / "crops"
    stored = privacy.store_sealed_crop(crop, "clip-3", store_dir=store)
    assert stored.path.is_file()
    on_disk = stored.path.read_bytes()
    assert on_disk != crop.tobytes()
    assert b"\x00\x00\x00" not in on_disk[:16]  # not a bare image header either


def test_purge_expired_removes_only_crops_past_30_days(tmp_path, crop):
    store = tmp_path / "crops"
    audit = privacy.AuditLog(tmp_path / "audit.jsonl")

    fresh = privacy.store_sealed_crop(crop, "fresh", store_dir=store)
    stale = privacy.store_sealed_crop(crop, "stale", store_dir=store)

    now = time.time()
    os.utime(stale.path, (now - 31 * 86400, now - 31 * 86400))
    os.utime(fresh.path, (now - 5 * 86400, now - 5 * 86400))

    purged = privacy.purge_expired(store_dir=store, now=now, audit_log=audit)

    assert [p.name for p in purged] == [stale.path.name]
    assert not stale.path.exists()
    assert fresh.path.exists()

    rows = audit.read_all()
    assert len(rows) == 1
    assert rows[0]["action"] == "retention.purge"
    assert rows[0]["clip_id"] == "stale"


def test_purge_expired_is_a_no_op_on_an_empty_store(tmp_path):
    audit = privacy.AuditLog(tmp_path / "audit.jsonl")
    purged = privacy.purge_expired(store_dir=tmp_path / "does-not-exist", audit_log=audit)
    assert purged == []
    assert audit.read_all() == []


def test_retention_window_is_thirty_days():
    assert privacy.RETENTION_DAYS == 30
    assert privacy.RETENTION_SECONDS == 30 * 24 * 3600
