"""Face detection on close-range person tracks (face/stage.py) and the privacy it must not weaken."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("cryptography")
from cryptography.fernet import Fernet  # noqa: E402

from face import privacy  # noqa: E402
from face.detect import Face  # noqa: E402
from face.stage import FaceStage  # noqa: E402
from tests.synth import make_track  # noqa: E402

H, W = 400, 600


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("FACE_UNSEAL_KEY", Fernet.generate_key().decode("ascii"))


def textured_frame() -> np.ndarray:
    rng = np.random.default_rng(3)
    return (rng.random((H, W, 3)) * 255).astype("uint8")     # noise: any blur is measurable


class FakeDetector:
    """Finds one 'face' in the head crop it is given, or nothing."""

    def __init__(self, finds: bool = True) -> None:
        self.finds, self.calls = finds, 0

    def __call__(self, crop):
        self.calls += 1
        if not self.finds:
            return []
        h, w = crop.shape[:2]
        return [Face(bbox=(w // 4, h // 4, 3 * w // 4, 3 * h // 4), score=0.93,
                     landmarks=((0, 0),) * 5)]


def stage(tmp_path, detector=None, **kw) -> FaceStage:
    return FaceStage(tmp_path / "faces", detect=detector or FakeDetector(), retry_every=1,
                     audit_log=privacy.AuditLog(tmp_path / "audit.jsonl"), **kw)


CLOSE = (200, 20, 320, 380)         # 360 px tall in a 400 px frame
FAR = (200, 20, 230, 80)            # 60 px tall


def test_close_person_gets_a_blurred_crop_and_a_sealed_face(tmp_path):
    st = stage(tmp_path)
    img = textured_frame()
    out = st.run(img, [make_track(1, CLOSE)], frame_index=7, camera_id="CAM")
    assert len(out) == 1
    rec = out[0]
    assert rec["track_id"] == 1 and rec["faces"] == 1
    stored = cv2.imread(rec["blurred_crop"])
    assert stored is not None
    # the face region of the stored crop is blurred: far smoother than the noisy source
    x1, y1, x2, y2 = rec["face_boxes_in_crop"][0]
    face_region = stored[y1 + 2:y2 - 2, x1 + 2:x2 - 2]
    assert cv2.Laplacian(face_region, cv2.CV_64F).var() < 0.05 * cv2.Laplacian(img, cv2.CV_64F).var()
    # the unblurred face exists only sealed: no clear jpg of it on disk, and the token decrypts only via unseal
    files = sorted(p.name for p in (tmp_path / "faces").iterdir())
    assert files == ["CAM-t1-f7.blurred.jpg", "CAM-t1-f7.sealed"]
    token = (tmp_path / "faces" / "CAM-t1-f7.sealed").read_bytes()
    assert token[:5] != b"\xff\xd8\xff"[:5]
    with pytest.raises(privacy.UnsealRefused):
        privacy.unseal(token, "operator", "CAM-t1-f7", "curious", audit_log=st.audit_log)
    assert privacy.unseal(token, "supervisor", "CAM-t1-f7", "case 12", audit_log=st.audit_log).size
    rows = st.audit_log.read_all()
    assert [r["action"] for r in rows] == ["unseal.refused", "unseal.granted"]


def test_without_an_unseal_key_only_the_blurred_crop_is_stored_never_a_clear_one(tmp_path, monkeypatch):
    monkeypatch.delenv("FACE_UNSEAL_KEY")
    st = stage(tmp_path)
    out = st.run(textured_frame(), [make_track(1, CLOSE)], frame_index=1, camera_id="CAM")
    assert out[0]["sealed_crop"] is None
    assert [p.suffix for p in (tmp_path / "faces").iterdir()] == [".jpg"]
    assert st.stats()["sealed_skipped_no_key"] == 1 and st.stats()["blurred_stored"] == 1


def test_only_close_persons_are_examined(tmp_path):
    det = FakeDetector()
    st = stage(tmp_path, det)
    tracks = [make_track(1, FAR), make_track(2, CLOSE, class_name="truck")]
    assert st.run(textured_frame(), tracks, frame_index=1, camera_id="CAM") == []
    assert det.calls == 0 and st.stats()["considered"] == 0


def test_one_stored_record_per_track_and_bounded_retries_when_no_face(tmp_path):
    det = FakeDetector()
    st = stage(tmp_path, det)
    img = textured_frame()
    assert len(st.run(img, [make_track(1, CLOSE)], frame_index=1, camera_id="CAM")) == 1
    for i in range(10):
        assert st.run(img, [make_track(1, CLOSE)], frame_index=2 + i, camera_id="CAM") == []
    assert det.calls == 1                       # found once: never looked at again

    none = FakeDetector(finds=False)
    st2 = stage(tmp_path / "b", none, max_attempts=4)
    for i in range(50):
        st2.run(img, [make_track(9, CLOSE)], frame_index=i, camera_id="CAM")
    assert none.calls == 4                      # a back-turned person costs a handful of calls, not one per frame


def test_retention_purge_runs_and_is_audited(tmp_path):
    import time

    clock = [time.time()]
    st = stage(tmp_path, clock=lambda: clock[0])
    st.run(textured_frame(), [make_track(1, CLOSE)], frame_index=1, camera_id="CAM")
    sealed = tmp_path / "faces" / "CAM-t1-f1.sealed"
    assert sealed.exists()
    import os

    old = sealed.stat().st_mtime - 31 * 86400
    os.utime(sealed, (old, old))
    clock[0] += 2 * 3600.0                      # the hourly purge is due; the crop is 31 days old
    st.run(textured_frame(), [], frame_index=2, camera_id="CAM")
    assert not sealed.exists() and st.stats()["purged"] == 1
    assert any(r["action"] == "retention.purge" for r in st.audit_log.read_all())


def test_a_detector_failure_never_reaches_the_pipeline(tmp_path):
    def broken(crop):
        raise RuntimeError("onnx exploded")

    st = stage(tmp_path, broken)
    assert st.run(textured_frame(), [make_track(1, CLOSE)], frame_index=1, camera_id="CAM") == []
    assert st.stats()["errors"] == 1


def test_service_runs_the_face_stage_and_reports_it(tmp_path):
    pytest.importorskip("scipy")      # the tracker; the core environment has it, the OCR one may not
    from pipeline.events import EventSink
    from pipeline.service import CameraService
    from tests.synth import Scene
    from tests.test_service import FakeAppearance, ingest_frames

    scene = Scene(speed=1.5, start=(20.0, 110.0))
    st = stage(tmp_path, min_height_frac=0.05)   # the synthetic person is small; treat it as close range
    svc = CameraService(FakeAppearance(scene), camera_id="CAM-T", post_id="RXL", sink=EventSink(), stride=1,
                        calibration_dir=tmp_path / "cal", warmup_frames=30, face=st)
    for f in ingest_frames(scene, 90):
        svc.process(f)
    svc.close()
    assert svc.stats.face is not None and svc.stats.face["detector_calls"] >= 1
    assert svc.faces and svc.stats.to_dict()["face"]["blurred_stored"] >= 1
