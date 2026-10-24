"""The service loop runs the detection pipeline on ingested frames and emits truewatch.event.v1."""

from __future__ import annotations



from pipeline.events import SCHEMA, EventSink, normalised_xywh
from pipeline.ingest import Frame as IngestFrame
from pipeline.service import CameraService
from pipeline.types import Detection
from tests.synth import Scene

REQUIRED = {"schema", "event_id", "post_id", "camera_id", "captured_at", "stage", "detection", "rule",
            "explanation", "anpr", "evidence", "confidence", "source"}


class FakeAppearance:
    """Stands in for AppearanceChannel: one confident box on the scene's moving object."""

    def __init__(self, scene: Scene, class_name: str = "person") -> None:
        self.scene, self.class_name, self.last_ms, self.calls = scene, class_name, 0.0, 0

    def detect(self, image, *, camera_id, frame_index, timestamp, modality):
        self.calls += 1
        return [Detection(bbox=self.scene.obj_box(frame_index), class_name=self.class_name, score=0.8,
                          camera_id=camera_id, frame_index=frame_index, timestamp=timestamp, modality=modality,
                          class_id=0)]


def ingest_frames(scene: Scene, n: int, camera_id: str = "CAM-T"):
    t0 = 1_760_000_000.0
    for k in range(n):
        yield IngestFrame(index=k, image=scene.frame(k), captured_at=t0 + k / 20, monotonic=100.0 + k / 20,
                          source="file", camera_id=camera_id)


def check_event(ev: dict) -> None:
    assert REQUIRED <= ev.keys()
    assert ev["schema"] == SCHEMA and ev["stage"] == "provisional"
    # the reason is in place from the first instant: the rule sentence, marked as such (never the model's)
    assert ev["evidence"] is None
    assert ev["explanation"]["source"] == "template" and ev["explanation"]["text"].strip()
    assert ev["explanation"]["model"] is None
    d = ev["detection"]
    assert d["agreed"] is True and d["class"] in {"person", "two_wheeler", "car", "truck", "cart"}
    assert len(d["bbox"]) == 4 and all(0.0 <= v <= 1.0 for v in d["bbox"])
    assert 0.0 <= d["channel_scores"]["appearance"] <= 1.0 and 0.0 <= d["channel_scores"]["motion"] <= 1.0
    assert isinstance(ev["confidence"], int) and 0 <= ev["confidence"] <= 100
    assert ev["source"] == "file" and ev["captured_at"].endswith("Z")


def check_observation(ev: dict) -> None:
    """A fusion-only event under ALERT_POLICY=rules: contract-shaped, marked local, no reason, no evidence."""
    assert REQUIRED <= ev.keys() and ev["kind"] == "observation"
    assert ev["schema"] == SCHEMA and ev["stage"] == "provisional"
    assert ev["rule"] is None and ev["explanation"] is None and ev["evidence"] is None
    assert ev["detection"]["agreed"] is True


def test_camera_service_turns_fusion_agreement_into_contract_events_legacy_all(tmp_path):
    scene = Scene(speed=1.5, start=(20.0, 110.0))
    sink = EventSink(log_path=tmp_path / "events.jsonl")
    svc = CameraService(FakeAppearance(scene), camera_id="CAM-T", post_id="RXL", sink=sink, stride=1,
                        calibration_dir=tmp_path / "cal", warmup_frames=30, alert_policy="all")
    for f in ingest_frames(scene, 90):
        svc.process(f)
    svc.close()
    events = sink.recent(100)
    assert svc.stats.frames == 90 and svc.stats.detector_passes == 90
    assert events, "a moving, detected object must produce a fusion-agreed event"
    for ev in events:
        check_event(ev)
        assert "kind" not in ev
        assert ev["post_id"] == "RXL" and ev["camera_id"] == "CAM-T"
    assert sink.total == len(events) and sink.observations == 0
    assert (tmp_path / "events.jsonl").read_text().count("\n") == len(events)


def test_fusion_only_agreement_is_a_local_observation_under_the_rules_policy(tmp_path):
    scene = Scene(speed=1.5, start=(20.0, 110.0))
    sink = EventSink(log_path=tmp_path / "events.jsonl")
    svc = CameraService(FakeAppearance(scene), camera_id="CAM-T", post_id="RXL", sink=sink, stride=1,
                        calibration_dir=tmp_path / "cal", warmup_frames=30)
    assert svc.alert_policy == "rules"
    for f in ingest_frames(scene, 90):
        svc.process(f)
    svc.close()
    events = sink.recent(100)
    assert events and sink.total == 0 and sink.observations == len(events) == svc.stats.observations
    assert svc.stats.events_emitted == 0
    for ev in events:
        check_observation(ev)
    assert (tmp_path / "events.jsonl").read_text().count("\n") == len(events)


def test_normalised_bbox_is_xywh_and_clipped():
    assert normalised_xywh((10, 20, 110, 70), (200, 100)) == [0.05, 0.2, 0.5, 0.5]
    assert normalised_xywh((-5, -5, 250, 50), (200, 100)) == [0.0, 0.0, 1.0, 0.5]


def test_service_loop_runs_the_pipeline_on_every_ingested_frame(tmp_path, monkeypatch):
    import app as app_mod
    from config import Config
    from pipeline import appearance as appearance_mod

    scene = Scene(speed=1.5, start=(20.0, 110.0))
    fake = FakeAppearance(scene)
    monkeypatch.setattr(appearance_mod.AppearanceChannel, "from_detector", classmethod(lambda cls, det, cfg=None: fake))
    monkeypatch.setattr(app_mod, "frames", lambda *a, **k: ingest_frames(scene, 90))
    monkeypatch.setenv("CALIBRATION_DIR", str(tmp_path / "cal"))
    monkeypatch.setenv("CALIBRATION_WARMUP_SECONDS", "1.5")
    monkeypatch.setenv("DETECTOR_STRIDE", "1")
    monkeypatch.setenv("CAMERA_ID", "CAM-T")
    cfg = Config()

    state = app_mod.PipelineState()
    state.detector = object()          # anything non-None: from_detector is patched
    state.sink = EventSink()
    state._run(cfg)

    snap = state.snapshot()
    assert snap["last_error"] is None
    assert snap["frames_seen"] == 90 and fake.calls == 90
    assert snap["detection"]["frames"] == 90
    # no rule file for CAM-T: under the default ALERT_POLICY=rules nothing alerts, the agreement is observed
    assert snap["events_emitted"] == 0 and snap["observations"] >= 1
    for ev in state.sink.recent(100):
        check_observation(ev)
