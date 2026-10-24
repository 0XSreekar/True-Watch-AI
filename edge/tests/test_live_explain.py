"""The explanation model wired into the live service path (pipeline/service.py + pipeline/explain.py).

Everything here drives the REAL `CameraService` (motion, fusion, rules, events) with a stand-in for the model,
never the real Moondream 2: the assertions are about ordering, bounds and the never-empty reason, not about
sentence quality.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from pipeline.events import EventSink
from pipeline.explain import EventExplainer, build_explainer, downscale
from pipeline.service import CameraService
from rules.config import CameraRuleConfig
from tests.synth import Scene
from tests.test_service import FakeAppearance, check_event, ingest_frames
from tests.test_service_rules import fence_config
from vlm.judge import CallableRuntime, VlmUnavailable
from vlm.queue import ExplanationQueue


class GatedRuntime:
    """A model that answers only when the test lets it, and answers with the event's own rule sentence
    (which passes the guardrails), so a patch is a legitimate 'vlm' result."""

    model_name = "stub-vlm"

    def __init__(self, sink: EventSink) -> None:
        self.sink = sink
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.calls = 0

    def describe(self, images, prompt, *, max_tokens, timeout_s):
        self.entered.set()
        assert self.gate.wait(30), "the pipeline waited on the model instead of the other way round"
        events = self.sink.recent(1000)
        text = events[self.calls]["explanation"]["text"]      # jobs run in submit order (one worker)
        self.calls += 1
        return text


def run_service(tmp_path, explainer, sink, *, rules=None, n=130, alert_policy="all"):
    # The reason mechanics are tested on every event the legacy 'all' policy alerts on (fusion AND rule
    # events); test_only_rule_alerts_are_explained_under_the_rules_policy covers the default policy.
    scene = Scene(speed=1.5, start=(20.0, 110.0))
    svc = CameraService(FakeAppearance(scene), camera_id="CAM-T", post_id="RXL", sink=sink, stride=1,
                        calibration_dir=tmp_path / "cal", warmup_frames=30, rules=rules, explainer=explainer,
                        camera_name="BOP Raxaul", camera_sector="Raxaul", alert_policy=alert_policy)
    t0 = time.perf_counter()
    for f in ingest_frames(scene, n):
        svc.process(f)
    elapsed = time.perf_counter() - t0
    svc.close()
    return svc, elapsed


def fence_rules():
    return CameraRuleConfig.from_dict(fence_config())


# ------------------------------------------------------------------ (a) emitted before the model finishes


def test_events_are_emitted_with_a_reason_before_the_model_finishes(tmp_path):
    sink = EventSink(log_path=tmp_path / "events.jsonl")
    runtime = GatedRuntime(sink)
    explainer = EventExplainer(ExplanationQueue(runtime, maxsize=64))
    svc, _ = run_service(tmp_path, explainer, sink, rules=fence_rules())

    # The whole clip has been processed and every event emitted while the model is still blocked on its gate.
    events = sink.recent(1000)
    assert len(events) >= 2, "expected a fusion event and a rule event"
    assert runtime.entered.wait(5), "the model was never asked"
    assert not runtime.gate.is_set() and sink.patches == 0
    for ev in events:
        check_event(ev)                                   # stage 'provisional', reason = the rule sentence
        assert ev["explanation"]["source"] == "template" and ev["explanation"]["text"].strip()
    assert any(e["rule"] and "fence crossed" in e["rule"]["fired"] for e in events)
    rule_ev = next(e for e in events if e["rule"])
    assert "fence" in rule_ev["explanation"]["text"].lower()   # grounded in the rule that fired

    runtime.gate.set()                                     # now let the model answer
    assert explainer.queue.join(10)
    assert sink.patches == len(events)
    for ev in sink.recent(1000):
        assert ev["explanation"]["source"] == "vlm" and ev["explanation"]["model"] == "stub-vlm"
        assert ev["explanation"]["text"].strip()

    # on the wire (the JSON Lines log carries what delivery carries): created first, its patch after
    lines = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    seen = set()
    updates = 0
    for msg in lines:
        if msg.get("schema") == "truewatch.update.v1":
            updates += 1
            assert msg["event_id"] in seen, "a patch was sent before its created event"
            assert set(msg) == {"schema", "event_id", "explanation"}
            assert msg["explanation"]["source"] == "vlm"
        else:
            seen.add(msg["event_id"])
    assert updates == len(events)
    assert svc.stats.explain["submitted"] == len(events)


def test_only_rule_alerts_are_explained_under_the_rules_policy(tmp_path):
    sink = EventSink()
    calls = []

    def fn(images, prompt, max_tokens, timeout_s):
        calls.append(prompt)
        return "unused"

    explainer = EventExplainer(ExplanationQueue(CallableRuntime(fn, "stub-vlm"), maxsize=64))
    svc, _ = run_service(tmp_path, explainer, sink, rules=fence_rules(), alert_policy="rules")
    assert explainer.queue.join(10)
    alerts = [e for e in sink.recent(1000) if e.get("kind") != "observation"]
    observations = [e for e in sink.recent(1000) if e.get("kind") == "observation"]
    assert observations, "the fusion event before the fence is a local observation"
    assert all(e["explanation"] is None for e in observations)
    assert len(alerts) == 1 and alerts[0]["rule"]["fired"] == ["fence crossed"]
    assert svc.stats.explain["submitted"] == 1 and len(calls) == 1, "the model is asked for the alert only"


def test_the_pipeline_never_blocks_on_a_model_that_never_answers(tmp_path):
    sink = EventSink()
    hang = threading.Event()

    def fn(images, prompt, max_tokens, timeout_s):
        hang.wait(60)
        return "unreachable"

    explainer = EventExplainer(ExplanationQueue(CallableRuntime(fn, "stub-vlm"), maxsize=4, timeout_s=0.2))
    _, elapsed = run_service(tmp_path, explainer, sink, rules=fence_rules())
    hang.set()
    assert sink.total >= 2 and elapsed < 30
    assert all(e["explanation"]["text"].strip() for e in sink.recent(1000))


# ------------------------------------------------------------------ (b) the reason is never empty


def _raises(images, prompt, max_tokens, timeout_s):
    raise RuntimeError("model crashed")


def _unavailable(images, prompt, max_tokens, timeout_s):
    raise VlmUnavailable("weights missing")


def _times_out(images, prompt, max_tokens, timeout_s):
    raise TimeoutError("too slow")


def _late(images, prompt, max_tokens, timeout_s):
    time.sleep(0.6)
    return "Person moving near Pillar 42."


CASES = {
    "raises": _raises,
    "unavailable": _unavailable,
    "timeout": _times_out,
    "answers_after_the_deadline": _late,
    "two_sentences": lambda *a: "The person walked to the fence. Then they stopped.",
    "speculation": lambda *a: "The person appears to be planning to cross the fence.",
    "names_someone": lambda *a: "Rakesh Kumar crossed the fence at 14:23.",
    "invented_fact": lambda *a: "A red bus with 40 passengers crossed the fence at 14:23.",
    "empty": lambda *a: "",
    "whitespace": lambda *a: "   \n ",
    "not_text": lambda *a: None,
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_reason_is_never_empty_when_the_model_fails_or_is_rejected(tmp_path, name):
    sink = EventSink()
    queue = ExplanationQueue(CallableRuntime(CASES[name], "stub-vlm"), maxsize=64, timeout_s=0.3)
    explainer = EventExplainer(queue)
    run_service(tmp_path, explainer, sink, rules=fence_rules())
    assert queue.join(15)
    events = sink.recent(1000)
    assert events
    for ev in events:
        assert isinstance(ev["explanation"]["text"], str) and ev["explanation"]["text"].strip()
        assert ev["explanation"]["source"] == "template"    # the rule sentence stayed
    assert sink.patches == 0                                 # nothing was patched over it


def test_reason_is_present_with_no_model_at_all(tmp_path):
    sink = EventSink()
    run_service(tmp_path, EventExplainer(None), sink, rules=fence_rules())
    events = sink.recent(1000)
    assert events and all(e["explanation"]["source"] == "template" and e["explanation"]["text"].strip()
                          for e in events)


def test_a_queue_that_raises_cannot_stop_events(tmp_path):
    explainer = EventExplainer(ExplanationQueue(CallableRuntime(_raises, "stub-vlm")))
    explainer.queue.submit = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("queue exploded"))
    sink = EventSink()
    run_service(tmp_path, explainer, sink)
    assert sink.total >= 1 and all(e["explanation"]["text"].strip() for e in sink.recent(1000))


# ------------------------------------------------------------------ (d) bounds


def test_queue_is_bounded_and_drops_the_oldest(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def fn(images, prompt, max_tokens, timeout_s):
        started.set()
        release.wait(30)
        return ""

    queue = ExplanationQueue(CallableRuntime(fn, "stub-vlm"), maxsize=3)
    sink = EventSink()
    explainer = EventExplainer(queue, frame_max_side=64)
    svc, _ = run_service(tmp_path, explainer, sink, rules=fence_rules())
    assert started.wait(5)
    events = sink.recent(1000)
    assert len(events) >= 2
    assert queue.qsize() <= 3
    # many more submissions than room: the surplus is dropped, counted, and nothing grows
    scene = Scene()
    extra = 20
    for i in range(extra):
        ev = dict(events[0], event_id=f"extra-{i}")
        pend = explainer.prepare(ev, camera=svc.camera_context(), fired=(), captured_at=1_760_000_000.0)
        explainer.submit(pend, sink, scene.frame(0))
        assert queue.qsize() <= 3
    stats = queue.stats()
    assert stats.dropped >= extra - 3
    assert queue.qsize() == 3
    release.set()
    assert queue.join(10), "join() must terminate even after drops"


def test_queued_frames_are_downscaled_copies():
    import numpy as np

    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    small = downscale(big, 768)
    assert max(small.shape[:2]) == 768 and small.shape[2] == 3 and small is not big
    same = downscale(small, 768)
    assert same.shape == small.shape and same is not small       # a copy either way


# ------------------------------------------------------------------ enabling


class Cfg:
    vlm_enabled = "auto"
    vlm_queue_max = 4
    vlm_timeout_s = 5.0
    vlm_frame_max_side = 256


def test_vlm_is_off_by_default_under_tests_and_when_disabled(monkeypatch):
    cfg = Cfg()
    assert build_explainer(cfg).enabled is False           # auto + pytest: off
    cfg.vlm_enabled = "0"
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert build_explainer(cfg).enabled is False           # explicit off never loads anything


def test_vlm_auto_stays_off_without_cached_weights(monkeypatch):
    from vlm import runtime as vlm_runtime

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv(vlm_runtime.HOSTED_ENV_VAR, raising=False)
    monkeypatch.setattr("pipeline.explain._weights_cached", lambda *a: False)
    assert build_explainer(Cfg()).enabled is False


def test_hosted_fallback_is_only_built_when_asked_for_and_warns(monkeypatch, caplog):
    from vlm import runtime as vlm_runtime

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv(vlm_runtime.HOSTED_ENV_VAR, raising=False)
    assert isinstance(vlm_runtime.build_default_runtime(), vlm_runtime.MoondreamRuntime)
    monkeypatch.setenv(vlm_runtime.HOSTED_ENV_VAR, "1")
    monkeypatch.delenv("TRUEWATCH_VLM_HOSTED_URL", raising=False)
    cfg = Cfg()
    cfg.vlm_enabled = "1"
    assert build_explainer(cfg).enabled is False           # opted in but no endpoint: refused, rule sentence kept
    monkeypatch.setenv("TRUEWATCH_VLM_HOSTED_URL", "http://127.0.0.1:9/never-called")
    monkeypatch.setenv("TRUEWATCH_VLM_HOSTED_KEY", "k")
    with caplog.at_level("WARNING"):
        explainer = build_explainer(cfg)
    try:
        assert explainer.enabled and "BREAKS the air-gapped" in caplog.text
    finally:
        explainer.close()


def test_an_injected_runtime_enables_the_explainer():
    class R:
        model_name = "x"

        def describe(self, *a, **k):
            return ""

    explainer = build_explainer(Cfg(), runtime=R())
    try:
        assert explainer.enabled and explainer.stats()["enabled"] is True
    finally:
        explainer.close()


def test_cooldown_spaces_out_model_calls_and_the_rest_is_not_on_the_pipeline_thread():
    from tests.test_vlm_async import _event

    starts = []

    def fn(images, prompt, max_tokens, timeout_s):
        starts.append(time.monotonic())
        return ""

    q = ExplanationQueue(CallableRuntime(fn, "stub-vlm"), cooldown_s=0.3)
    t0 = time.perf_counter()
    for i in range(3):
        q.submit(f"j{i}", _event(), lambda *a: None)
    assert time.perf_counter() - t0 < 0.1                    # submitting never waits on the cooldown
    assert q.join(5)
    assert len(starts) == 3 and starts[1] - starts[0] >= 0.29 and starts[2] - starts[1] >= 0.29
    q.shutdown()
