"""ALERT_POLICY: operator alerts fire only when a rule fires; dual-channel agreement is necessary, not sufficient."""

from __future__ import annotations

import logging

import pytest

from pipeline.events import UPDATE_SCHEMA, EventSink
from pipeline.explain import EventExplainer
from pipeline.service import CameraService
from rules.config import CameraRuleConfig, ConfigStore
from tests.synth import Scene
from tests.test_service import FakeAppearance, check_event, check_observation, ingest_frames
from tests.test_service_rules import fence_config
from vlm.judge import CallableRuntime
from vlm.queue import ExplanationQueue


class RecordingDelivery:
    """Stands in for BackendDelivery: records what would be POSTed."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def enqueue(self, message: dict) -> None:
        self.messages.append(message)

    def created(self) -> list[dict]:
        return [m for m in self.messages if m.get("schema") != UPDATE_SCHEMA]

    def close(self) -> None:
        pass


class CountingSealer:
    def __init__(self) -> None:
        self.added = 0
        self.sealed = 0

    def add(self, image, captured_at) -> None:
        self.added += 1

    def seal(self) -> dict:
        self.sealed += 1
        return {"clip_sha256": f"{self.sealed:064x}", "block": self.sealed}

    def stats(self) -> dict:
        return {"sealed": self.sealed, "failed": 0, "buffered_frames": self.added}


def run(tmp_path, *, rules=None, policy="rules", n=130):
    delivery, sealer, prompts = RecordingDelivery(), CountingSealer(), []

    def model(images, prompt, max_tokens, timeout_s):
        prompts.append(prompt)
        return "unused"

    explainer = EventExplainer(ExplanationQueue(CallableRuntime(model, "stub-vlm"), maxsize=64))
    sink = EventSink(delivery=delivery)
    scene = Scene(speed=1.5, start=(20.0, 110.0))
    svc = CameraService(FakeAppearance(scene), camera_id="CAM-T", post_id="RXL", sink=sink, stride=1,
                        calibration_dir=tmp_path / "cal", warmup_frames=30, rules=rules, explainer=explainer,
                        sealer=sealer, alert_policy=policy)
    for f in ingest_frames(scene, n):
        svc.process(f)
    svc.close()
    assert explainer.queue.join(10)
    explainer.close()
    return svc, sink, delivery, sealer, prompts


def test_fusion_only_is_never_delivered_sealed_or_explained_under_the_rules_policy(tmp_path):
    svc, sink, delivery, sealer, prompts = run(tmp_path, rules=None)
    assert svc.stats.fused_events >= 1 and sink.observations == svc.stats.fused_events
    assert delivery.messages == [] and sealer.sealed == 0 and prompts == []
    assert sealer.added == 130, "the clip ring still records every frame, ready for a later alert"
    for ev in sink.recent(1000):
        check_observation(ev)


def test_a_rule_firing_delivers_exactly_one_alert_per_track_and_rule(tmp_path):
    svc, sink, delivery, sealer, prompts = run(tmp_path, rules=CameraRuleConfig.from_dict(fence_config()), n=140)
    created = delivery.created()
    assert len(created) == 1, "one (track, rule) firing = one alert, even though the track stays past the fence"
    assert created[0]["rule"]["fired"] == ["fence crossed"] and created[0]["explanation"] is None
    assert svc.stats.rule_events == 1 and svc.stats.events_emitted == 1 and sink.total == 1
    assert sealer.sealed == 1, "sealed at the moment the alert fired"
    updates = [m for m in delivery.messages if m.get("schema") == UPDATE_SCHEMA]
    assert {m["event_id"] for m in updates} == {created[0]["event_id"]}
    assert any("evidence" in m for m in updates) and any("explanation" in m for m in updates)
    assert len(prompts) == 1
    assert sink.observations >= 1, "the fusion event before the crossing stayed a local observation"


def test_the_throttle_still_refuses_a_rule_firing_past_the_budget(tmp_path):
    from rules.throttle import AlertThrottle

    delivery = RecordingDelivery()
    sink = EventSink(delivery=delivery)
    scene = Scene(speed=1.5, start=(20.0, 110.0))
    throttle = AlertThrottle(per_camera_limit=0)
    svc = CameraService(FakeAppearance(scene), camera_id="CAM-T", post_id="RXL", sink=sink, stride=1,
                        calibration_dir=tmp_path / "cal", warmup_frames=30,
                        rules=CameraRuleConfig.from_dict(fence_config()), throttle=throttle)
    for f in ingest_frames(scene, 130):
        svc.process(f)
    svc.close()
    assert svc.stats.rule_fires == {"fence crossed": 1} and svc.stats.rule_suppressed == 1
    assert delivery.created() == []


def test_legacy_all_policy_alerts_on_every_fusion_event(tmp_path):
    svc, sink, delivery, sealer, prompts = run(tmp_path, rules=None, policy="all")
    created = delivery.created()
    assert len(created) == svc.stats.fused_events >= 1 and sink.observations == 0
    assert sealer.sealed >= 1 and len(prompts) == len(created)
    for ev in created:
        check_event(ev | {"explanation": {"source": "template", "text": "x", "model": None}})
        assert "kind" not in ev


def test_a_camera_without_rules_is_warned_about_at_start(tmp_path, caplog):
    (tmp_path / "rules").mkdir()
    with caplog.at_level(logging.WARNING, logger="truewatch.edge.service"):
        CameraService(None, camera_id="CAM-X", post_id="RXL", sink=EventSink(), calibration_dir=tmp_path / "cal",
                      rules=ConfigStore(directory=tmp_path / "rules")).close()
    assert any("NO RULE CONFIGURATION" in r.message and "can never raise" in r.message for r in caplog.records)


def test_unknown_policy_is_refused(monkeypatch):
    from config import Config, ConfigError

    with pytest.raises(ValueError):
        CameraService(None, camera_id="C", post_id="P", sink=EventSink(), alert_policy="some")
    monkeypatch.setenv("ALERT_POLICY", "everything")
    with pytest.raises(ConfigError):
        Config().validate()
    monkeypatch.setenv("ALERT_POLICY", "rules")
    Config().validate()
