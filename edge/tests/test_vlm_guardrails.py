"""Guardrail and fallback tests for the explanation layer (edge/vlm).

The model is never loaded here: `judge.explain` takes the runtime as an
injected `VlmRuntime`, so each test hands it a stub that returns one fixed
string and asserts what reaches the alert.

Groups:
1. `rule_sentence` renders the backend/src/data/mockData.js REASONS register
   and passes its own guardrail.
2. A valid model sentence is accepted (and safe repairs are applied).
3. Multi-sentence, over-length, speculative, person-identifying,
   vendor-mentioning and invented-fact outputs are each rejected and the
   alert falls back to the rule string.
4. Runtime failures (exception, timeout, late answer, non-text) fall back.
5. The fallback chain below the rule sentence, and a seeded fuzz asserting
   the reason is never empty on any input.
"""

from __future__ import annotations

import codecs
import math
import random
import string
from datetime import datetime
from types import SimpleNamespace

import pytest

from rules.engine import (
    RULE_FENCE_CROSSED,
    RULE_GROUP_FORMED,
    RULE_LOITERING,
    RULE_OUT_OF_HOURS,
    RULE_WRONG_DIRECTION,
    FiredRule,
)
from vlm import judge as judge_mod
from vlm.guardrails import check
from vlm.judge import (
    LAST_RESORT_REASON,
    CallableRuntime,
    JudgeInput,
    VlmUnavailable,
    explain,
    fallback_reason,
    initial_reason,
)
from vlm.prompt import (
    MAX_REASON_CHARS,
    POST_TIMEZONE,
    CameraContext,
    EventContext,
    build_facts,
    build_prompt,
    few_shot_examples,
    rule_sentence,
)

# --------------------------------------------------------------------------
# Fixtures: the three demo events used throughout
# --------------------------------------------------------------------------

RAXAUL = {"id": "RXL-01", "name": "BOP Raxaul — Pillar 42", "sector": "Raxaul", "ir": False, "road": True}
PANITANKI = {"id": "PNT-07", "name": "Panitanki — River Bank", "sector": "Panitanki", "ir": True, "road": True}
GALGALIA = {"id": "GLG-05", "name": "Galgalia — Culvert Approach", "sector": "Galgalia", "ir": True, "road": False}


def _ts(h: int, m: int, s: int = 7) -> float:
    return datetime(2026, 8, 14, h, m, s, tzinfo=POST_TIMEZONE).timestamp()


def _fired(name: str, evidence: dict, reason: str) -> FiredRule:
    return FiredRule(rule_name=name, track_id=6, evidence=evidence, reason=reason)


def truck_night_northbound() -> JudgeInput:
    """Truck at 02:41 IST, out of hours, against the learnt flow; camera calibrated so image-up is north."""
    camera = CameraContext.from_camera_row(RAXAUL, sanctioned=[("06:00", "20:00")], image_right_bearing_deg=90.0)
    fired = (
        _fired(RULE_OUT_OF_HOURS, {"local_time": "02:41:07", "timezone": "Asia/Kolkata"},
               "track 6 moved at 02:41:07 IST, outside sanctioned hours"),
        _fired(RULE_WRONG_DIRECTION,
               {"heading_deg": 268.0, "baseline_heading_deg": 92.0, "angle_from_baseline_deg": 176.0,
                "net_displacement_px": 210.0},
               "track 6 moved at 268°, 176° off the learnt majority flow of 92°"),
    )
    return JudgeInput(EventContext(camera, _ts(2, 41), "truck"), fired, frames=("frame0", "frame1"))


def group_crossing_ir() -> JudgeInput:
    camera = CameraContext.from_camera_row(PANITANKI, sanctioned=[("05:00", "19:00")])
    fired = (
        _fired(RULE_FENCE_CROSSED,
               {"fence_name": "Channel", "crossed_at_s": 1.2, "point_px": (320.0, 211.0), "direction": "in"},
               "track 6 crossed fence 'Channel' (inbound) at (320, 211)"),
        _fired(RULE_GROUP_FORMED,
               {"track_ids": [3, 4, 5, 6, 7, 8], "size": 6, "duration_s": 48.2, "radius": 3.0, "unit": "m"},
               "group of 6 (3, 4, 5, 6, 7, 8) held within 3.0m for 48.2s"),
        _fired(RULE_OUT_OF_HOURS, {"local_time": "01:12:07", "timezone": "Asia/Kolkata"},
               "track 6 moved at 01:12:07 IST, outside sanctioned hours"),
    )
    return JudgeInput(EventContext(camera, _ts(1, 12), "person"), fired)


def person_loitering_ir() -> JudgeInput:
    camera = CameraContext.from_camera_row(GALGALIA)
    fired = (
        _fired(RULE_LOITERING,
               {"dwell_s": 42.3, "net_displacement_px": 20.0, "path_length_px": 160.0,
                "net_displacement_m": 1.2, "threshold": 2.0, "unit": "m"},
               "track 6 loitered 42.3s, net 1.2m (path 160px) under the 2.0m threshold"),
    )
    return JudgeInput(EventContext(camera, _ts(23, 7), "person"), fired)


ALL_EVENTS = [truck_night_northbound, group_crossing_ir, person_loitering_ir]

VALID_TRUCK = (
    "White truck moving north near Pillar 42 at 02:41, outside sanctioned hours, "
    "against the learnt flow on this route."
)


def _runtime(text):
    return CallableRuntime(lambda images, prompt, max_tokens, timeout_s: text, model_name="moondream2")


def _assert_fell_back(inp: JudgeInput, raw: str, expect_code: str | None = None):
    result = explain(inp, _runtime(raw))
    assert result.source == "template"
    assert result.text == fallback_reason(inp) == rule_sentence(inp.evidence, inp.context)
    assert result.model is None
    if expect_code is not None:
        assert any(v.startswith(expect_code) for v in result.violations), result.violations
    return result


# --------------------------------------------------------------------------
# 1. The deterministic rule sentence
# --------------------------------------------------------------------------


class TestRuleSentence:
    def test_truck_sample_matches_register(self):
        inp = truck_night_northbound()
        assert rule_sentence(inp.evidence, inp.context) == (
            "Truck moving north near Pillar 42 at 02:41, outside sanctioned hours (06:00–20:00), "
            "against the learnt flow on this route."
        )

    def test_group_sample_matches_register(self):
        inp = group_crossing_ir()
        assert rule_sentence(inp.evidence, inp.context) == (
            "Group of six crossing the Channel fence line inbound near River Bank at 01:12, "
            "outside sanctioned hours (05:00–19:00), held together for 48 s, seen on IR."
        )

    def test_loitering_sample_matches_register(self):
        inp = person_loitering_ir()
        assert rule_sentence(inp.evidence, inp.context) == (
            "Person halted 42 s within 1.2 m near Culvert Approach at 23:07, seen on IR."
        )

    @pytest.mark.parametrize("make", ALL_EVENTS)
    def test_one_sentence_under_limit_with_time_and_passes_own_guardrail(self, make):
        inp = make()
        s = rule_sentence(inp.evidence, inp.context)
        assert 0 < len(s) <= MAX_REASON_CHARS
        assert s.endswith(".") and s.count(". ") == 0
        assert inp.context.local_dt.strftime("%H:%M") in s
        assert check(s, build_facts(inp.evidence, inp.context)).accepted

    def test_no_compass_without_calibration(self):
        inp = truck_night_northbound()
        cam = CameraContext.from_camera_row(RAXAUL, sanctioned=[("06:00", "20:00")])
        s = rule_sentence(inp.evidence, EventContext(cam, inp.context.captured_at, "truck"))
        assert "north" not in s and "moving up the frame" in s
        assert check(s, build_facts(inp.evidence, EventContext(cam, inp.context.captured_at, "truck"))).accepted
        # and the model may not add a compass word the camera cannot support
        facts = build_facts(inp.evidence, EventContext(cam, inp.context.captured_at, "truck"))
        assert "invented_direction" in check("Truck moving north near Pillar 42 at 02:41.", facts).codes
        assert "invented_direction" in check("Truck moving left to right near Pillar 42 at 02:41.", facts).codes

    def test_hours_accepts_rule_engine_windows(self):
        from datetime import time as dt_time
        from rules.hours import SanctionedWindow

        cam = CameraContext.from_camera_row(RAXAUL, sanctioned=[SanctionedWindow(dt_time(6), dt_time(20))])
        assert cam.sanctioned == (("06:00", "20:00"),)

    def test_empty_evidence_still_renders(self):
        cam = CameraContext.from_camera_row(RAXAUL)
        s = rule_sentence({}, EventContext(cam, _ts(10, 5), "car"))
        assert s == "Car moving near Pillar 42 at 10:05, flagged by both detection channels."


class TestPrompt:
    def test_prompt_carries_evidence_and_instructions(self):
        inp = truck_night_northbound()
        p = build_prompt(inp.evidence, inp.context)
        for needle in ("02:41 IST", "north", "06:00-20:00", "Pillar 42", "truck", "one sentence",
                       "220 characters", "Do not guess intent", "identity", "Never mention AI"):
            assert needle in p, needle
        assert p.rstrip().endswith("Reason:")
        assert len(p) < 2000  # short enough for a ~2B model

    def test_examples_are_register_sentences_not_mock_fixtures(self):
        mock_reasons = (
            "Loaded vehicle moving north at 02:41",
            "Group of six crossing the dry channel",
            "Pickup halted 40 s at the fence line",
        )
        examples = few_shot_examples()
        assert len(examples) == 3
        for facts, reason in examples:
            assert len(reason) <= MAX_REASON_CHARS and reason.endswith(".")
            assert not any(m in reason for m in mock_reasons)
            assert "time:" in facts


# --------------------------------------------------------------------------
# 2. Valid output accepted
# --------------------------------------------------------------------------


class TestAccepted:
    def test_valid_output_is_used(self):
        inp = truck_night_northbound()
        result = explain(inp, _runtime(VALID_TRUCK))
        assert result.source == "vlm"
        assert result.text == VALID_TRUCK
        assert result.model == "moondream2"
        assert result.to_event() == {
            "text": VALID_TRUCK, "source": "vlm", "model": "moondream2", "latency_ms": result.latency_ms,
        }

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("  Reason:  " + VALID_TRUCK + "\n", VALID_TRUCK),
            ('"' + VALID_TRUCK + '"', VALID_TRUCK),
            ("The image shows a white truck moving north near Pillar 42 at 02:41, outside sanctioned hours.",
             "A white truck moving north near Pillar 42 at 02:41, outside sanctioned hours."),
            (VALID_TRUCK + " The tru", VALID_TRUCK),  # token-limit cut
            ("Loaded truck heading northbound at 2:41 am, outside sanctioned hours",
             "Loaded truck heading northbound at 2:41 am, outside sanctioned hours."),
        ],
    )
    def test_safe_repairs(self, raw, expected):
        result = explain(truck_night_northbound(), _runtime(raw))
        assert result.source == "vlm", result.violations
        assert result.text == expected

    def test_group_output_with_visible_detail_accepted(self):
        raw = "Group of six crossing the Channel fence line inbound at 01:12 in single file, no lamps, seen on IR."
        assert explain(group_crossing_ir(), _runtime(raw)).source == "vlm"


# --------------------------------------------------------------------------
# 3. Rejections -> rule string fallback
# --------------------------------------------------------------------------


class TestRejected:
    def test_multi_sentence(self):
        raw = "Truck moving north at 02:41. It is outside sanctioned hours."
        _assert_fell_back(truck_night_northbound(), raw, "multi_sentence")

    def test_over_length(self):
        raw = (
            "Truck moving north near Pillar 42 at 02:41, outside sanctioned hours, against the learnt flow "
            "on this route, with its load visible in the open bed and no lights on, travelling steadily along "
            "the track beside the embankment and the open field."
        )
        assert len(raw) > MAX_REASON_CHARS
        _assert_fell_back(truck_night_northbound(), raw, "too_long")

    @pytest.mark.parametrize(
        "raw",
        [
            "Truck moving north at 02:41, appears to be planning a crossing outside sanctioned hours.",
            "Truck moving north at 02:41, likely carrying goods outside sanctioned hours.",
            "Truck with suspicious intent moving north at 02:41, outside sanctioned hours.",
            "Truck moving north at 02:41 might be smuggling goods outside sanctioned hours.",
            "Truck moving north at 02:41, probably trying to avoid the post, outside sanctioned hours.",
            "Truck moving north at 02:41 carrying contraband, outside sanctioned hours.",
            "Truck moving north at 02:41, which suggests an illegal crossing outside sanctioned hours.",
        ],
    )
    def test_speculation(self, raw):
        _assert_fell_back(truck_night_northbound(), raw, "speculation")

    @pytest.mark.parametrize(
        "raw",
        [
            "Person halted 42 s near Culvert Approach at 23:07, his face visible on IR.",
            "Nepali woman halted 42 s near Culvert Approach at 23:07, seen on IR.",
            "Person identified as a local farmer halted 42 s near Culvert Approach at 23:07.",
            "Elderly person with a beard halted 42 s near Culvert Approach at 23:07.",
            "Person wearing a red jacket halted 42 s near Culvert Approach at 23:07.",
            "Muslim person halted 42 s near Culvert Approach at 23:07, seen on IR.",
        ],
    )
    def test_person_identifying(self, raw):
        _assert_fell_back(person_loitering_ir(), raw, "person_identity")

    # Vendor names are ROT13-encoded so the words never appear in the repo.
    @pytest.mark.parametrize("encoded", ["pynhqr", "bcranv", "pungtcg", "trzvav", "naguebcvp",
                                         "pbcvybg", "zbbaqernz", "uhttvat snpr", "bcra nv"])
    def test_vendor_mention(self, encoded):
        vendor = codecs.decode(encoded, "rot13")
        raw = f"Truck moving north at 02:41 outside sanctioned hours, per {vendor}."
        _assert_fell_back(truck_night_northbound(), raw, "vendor")

    def test_vendor_mention_case_and_suffix(self):
        vendor = codecs.decode("PungTCG", "rot13")
        raw = f"Truck moving north at 02:41 outside sanctioned hours ({vendor}-4 summary)."
        _assert_fell_back(truck_night_northbound(), raw, "vendor")

    def test_self_reference(self):
        raw = "As an AI, I see a truck moving north at 02:41 outside sanctioned hours."
        _assert_fell_back(truck_night_northbound(), raw, "self_reference")

    @pytest.mark.parametrize(
        "raw, code",
        [
            # wrong time
            ("Truck moving north near Pillar 42 at 03:15, outside sanctioned hours.", "invented_time"),
            # wrong direction for this calibration
            ("Truck moving south near Pillar 42 at 02:41, outside sanctioned hours.", "invented_direction"),
            # wrong object class
            ("Motorcycle moving north near Pillar 42 at 02:41, outside sanctioned hours.", "invented_object"),
            # a distance that is not in the evidence
            ("Truck moving north 140 m east of Pillar 42 at 02:41, outside sanctioned hours.", "invented_number"),
            # a count that is not in the evidence
            ("Truck with two figures moving north at 02:41, outside sanctioned hours.", "invented_number"),
            # a rule that did not fire
            ("Truck halted near Pillar 42 at 02:41, outside sanctioned hours.", "invented_claim"),
            ("Truck crossed the fence near Pillar 42 at 02:41, outside sanctioned hours.", "invented_claim"),
            # daylight claim at 02:41
            ("Truck moving north in broad daylight at 02:41, outside sanctioned hours.", "invented_claim"),
            # IR on a day camera
            ("Truck moving north at 02:41 on thermal, outside sanctioned hours.", "invented_claim"),
            # a name that is nowhere in the context
            ("Truck moving north near Birgunj Chowk at 02:41, outside sanctioned hours.", "invented_name"),
            # a plate that is not in the evidence
            ("Truck BR 06 GA 4821 moving north at 02:41, outside sanctioned hours.", "invented_name"),
            # a pixel figure leaking through
            ("Truck moving north 210 px at 02:41, outside sanctioned hours.", "invented_number"),
        ],
    )
    def test_invented_fact(self, raw, code):
        _assert_fell_back(truck_night_northbound(), raw, code)

    def test_plural_when_single_track(self):
        raw = "People halted 42 s near Culvert Approach at 23:07, seen on IR."
        _assert_fell_back(person_loitering_ir(), raw, "invented_object")

    def test_wrong_group_size(self):
        raw = "Group of four crossing the Channel fence line inbound at 01:12, seen on IR."
        _assert_fell_back(group_crossing_ir(), raw, "invented_number")

    @pytest.mark.parametrize(
        "raw, code",
        [
            ("A truck is driving on a dirt road at night.", "missing_time"),
            ("Truck on the road near Pillar 42 at 02:41 with its load covered.", "missing_reason"),
        ],
    )
    def test_incomplete(self, raw, code):
        _assert_fell_back(truck_night_northbound(), raw, code)

    @pytest.mark.parametrize("raw", ["", "   ", "Truck?", "Is the truck moving north at 02:41?",
                                     "Truck moving north at 02:41, outside"])
    def test_empty_question_or_truncated(self, raw):
        _assert_fell_back(truck_night_northbound(), raw)


# --------------------------------------------------------------------------
# 4. Runtime failures
# --------------------------------------------------------------------------


class TestRuntimeFailures:
    def _raising(self, exc):
        def fn(images, prompt, max_tokens, timeout_s):
            raise exc

        return CallableRuntime(fn)

    @pytest.mark.parametrize("exc", [TimeoutError("slow"), VlmUnavailable("not loaded"),
                                     RuntimeError("segfault-ish"), MemoryError()])
    def test_runtime_exception_falls_back(self, exc):
        inp = truck_night_northbound()
        result = explain(inp, self._raising(exc))
        assert result.source == "template" and result.text == fallback_reason(inp)

    def test_no_runtime(self):
        inp = truck_night_northbound()
        assert explain(inp, None).text == fallback_reason(inp)

    def test_late_answer_is_dropped(self):
        ticks = iter([0.0, 30.0])
        inp = truck_night_northbound()
        result = explain(inp, _runtime(VALID_TRUCK), timeout_s=8.0, clock=lambda: next(ticks))
        assert result.source == "template" and "timeout" in result.violations
        assert result.latency_ms == 30000

    @pytest.mark.parametrize("raw", [None, 42, b"bytes", ["list"]])
    def test_non_text_output(self, raw):
        inp = truck_night_northbound()
        assert explain(inp, _runtime(raw)).text == fallback_reason(inp)

    def test_runtime_receives_prompt_frames_and_budget(self):
        seen = {}

        def fn(images, prompt, max_tokens, timeout_s):
            seen.update(images=images, prompt=prompt, max_tokens=max_tokens, timeout_s=timeout_s)
            return VALID_TRUCK

        inp = truck_night_northbound()
        explain(inp, CallableRuntime(fn), max_tokens=64, timeout_s=5.0)
        assert seen["images"] == ("frame0", "frame1")
        assert seen["prompt"] == build_prompt(inp.evidence, inp.context)
        assert (seen["max_tokens"], seen["timeout_s"]) == (64, 5.0)

    def test_initial_reason_is_the_rule_string(self):
        inp = truck_night_northbound()
        first = initial_reason(inp)
        assert first.source == "template" and first.text == rule_sentence(inp.evidence, inp.context)


# --------------------------------------------------------------------------
# 5. Fallback chain and never-empty
# --------------------------------------------------------------------------


class TestFallbackChain:
    def test_phase4_string_when_rule_sentence_breaks(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("template bug")

        monkeypatch.setattr(judge_mod, "rule_sentence", boom)
        inp = truck_night_northbound()
        text = fallback_reason(inp)
        assert text.startswith("Track 6 moved at 02:41:07 IST, outside sanctioned hours; track 6 moved at 268°")
        assert len(text) <= MAX_REASON_CHARS

    def test_built_from_rule_name_and_evidence_when_phase4_string_missing(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "rule_sentence", lambda *a, **k: "")
        fired = (SimpleNamespace(rule_name="loitering", evidence={"dwell_s": 42.3, "unit": "m"}, reason=""),)
        inp = JudgeInput(person_loitering_ir().context, fired)
        assert fallback_reason(inp) == "Loitering (dwell s 42.3, unit m)."

    def test_last_resort(self, monkeypatch):
        monkeypatch.setattr(judge_mod, "rule_sentence", lambda *a, **k: "")
        inp = JudgeInput(person_loitering_ir().context, (SimpleNamespace(),))
        assert fallback_reason(inp) == LAST_RESORT_REASON

    def test_broken_context(self):
        inp = JudgeInput(EventContext(None, float("nan"), "ufo"), ())  # type: ignore[arg-type]
        result = explain(inp, _runtime(VALID_TRUCK))
        assert result.text.strip()


def _rand_text(rng: random.Random) -> str:
    vocab = [
        "Truck", "moving", "north", "south", "at", "02:41", "03:15", "outside", "sanctioned", "hours",
        "likely", "man", "his", "face", "Pillar", "42", "140", "m", "s", "group", "of", "six", ".", ",",
        "?", "!", "\n", "  ", "fence", "halted", "IR", "the", "learnt", "flow", "against", "é", "—", "",
        codecs.decode("bcranv", "rot13"),
    ]
    mode = rng.random()
    if mode < 0.1:
        return ""
    if mode < 0.2:
        return "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 400)))
    return " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 60)))


def _rand_value(rng: random.Random):
    return rng.choice([None, "", "x", -1, 0, 3.5, 1e308, float("nan"), float("inf"), [], {}, (1, 2),
                       True, "in", "out", "m", "px", 12, 400.0])


def _rand_input(rng: random.Random) -> JudgeInput:
    names = [RULE_FENCE_CROSSED, RULE_GROUP_FORMED, RULE_LOITERING, RULE_OUT_OF_HOURS, RULE_WRONG_DIRECTION,
             "tamper", "", None]
    keys = ["fence_name", "direction", "dwell_s", "net_displacement_m", "unit", "heading_deg",
            "angle_from_baseline_deg", "size", "duration_s", "track_ids", "local_time"]
    fired = []
    for _ in range(rng.randint(0, 4)):
        ev = {k: _rand_value(rng) for k in rng.sample(keys, rng.randint(0, len(keys)))}
        fired.append(SimpleNamespace(
            rule_name=rng.choice(names),
            evidence=rng.choice([ev, None, "junk"]),
            reason=rng.choice(["", "   ", None, "track 1 did something", 7]),
        ))
    row = rng.choice([RAXAUL, PANITANKI, GALGALIA, {"id": "", "name": "", "sector": ""}])
    windows = rng.choice([(), (("06:00", "20:00"),), (("22:00", "05:00"), ("12:00", "13:00"))])
    bearing = rng.choice([None, 0.0, 90.0, 271.3, float("nan")])
    cam = CameraContext.from_camera_row(row, sanctioned=windows, image_right_bearing_deg=bearing)
    captured = rng.choice([_ts(rng.randint(0, 23), rng.randint(0, 59)), 0.0, float("nan"), 1e20, -5.0])
    cls = rng.choice(["person", "truck", "car", "two_wheeler", "cart", "unknown", ""])
    return JudgeInput(EventContext(cam, captured, cls), tuple(fired))


@pytest.mark.parametrize("seed", range(400))
def test_reason_never_empty_fuzz(seed):
    rng = random.Random(seed)
    inp = _rand_input(rng)
    raw = _rand_text(rng)
    behaviour = rng.random()

    def fn(images, prompt, max_tokens, timeout_s):
        if behaviour < 0.1:
            raise RuntimeError("crash")
        if behaviour < 0.15:
            raise TimeoutError()
        return raw

    first = initial_reason(inp)
    final = explain(inp, CallableRuntime(fn))
    for r in (first, final):
        assert isinstance(r.text, str) and r.text.strip()
        assert r.source in ("vlm", "template")
        if r.source == "vlm":
            assert len(r.text) <= MAX_REASON_CHARS
    assert not math.isnan(len(final.text))
