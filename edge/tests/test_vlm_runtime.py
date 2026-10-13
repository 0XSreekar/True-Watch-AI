"""The local Moondream 2 runtime's device selection and load path, with the checkpoint itself stubbed out.

The real model is exercised by vlm/latency.py (results in vlm/results/); these tests cover the parts of
runtime.py that are ours: which device is chosen, that the pinned revision is what is requested, that the
MPS probe is put back after a CPU load, and that a failed load degrades instead of raising into the caller.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from vlm.judge import VlmUnavailable  # noqa: E402
from vlm.runtime import PINNED_REVISION, MoondreamRuntime  # noqa: E402


def _tf():
    """The transformers module the runtime will import from, looked up now: another test module may have
    replaced the one imported when this file was collected."""
    import importlib

    return importlib.import_module("transformers")


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def query(self, image, prompt, settings=None):
        return {"answer": f"seen {image.size[0]}x{image.size[1]} with {settings['max_tokens']} tokens"}


def fake_auto(seen: dict):
    class Auto:
        @staticmethod
        def from_pretrained(model_id, **kw):
            seen["model_id"], seen["kw"] = model_id, kw
            seen["mps_probe"] = torch.backends.mps.is_available()
            seen["probe_has_wrapped"] = hasattr(torch.backends.mps.is_available, "__wrapped__")
            return FakeModel()

    return Auto


def test_cpu_load_requests_the_pinned_revision_hides_mps_only_during_the_load_and_restores_it(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(_tf(), "AutoModelForCausalLM", fake_auto(seen))
    before = torch.backends.mps.is_available
    rt = MoondreamRuntime(device="cpu", attempt_quantization=False)
    rt._ensure_loaded()
    assert rt.device == "cpu" and rt.is_loaded and rt.quantized is False
    assert seen["kw"]["revision"] == PINNED_REVISION and seen["kw"]["trust_remote_code"] is True
    assert seen["mps_probe"] is False, "the checkpoint must take its CPU path"
    assert seen["probe_has_wrapped"], "torch._dynamo reads __wrapped__ from this probe"
    assert torch.backends.mps.is_available is before


def test_auto_picks_mps_when_available_else_cpu(monkeypatch):
    monkeypatch.setattr(_tf(), "AutoModelForCausalLM", fake_auto({}))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    rt = MoondreamRuntime(device="auto", attempt_quantization=False)
    monkeypatch.setattr(FakeModel, "to", lambda self, *a, **k: self)     # no real MPS tensors in a unit test
    rt._ensure_loaded()
    assert rt.device == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    rt2 = MoondreamRuntime(device="auto", attempt_quantization=False)
    rt2._ensure_loaded()
    assert rt2.device == "cpu"


def test_asking_for_mps_without_it_is_a_clean_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setattr(_tf(), "AutoModelForCausalLM", fake_auto({}))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    rt = MoondreamRuntime(device="mps")
    with pytest.raises(VlmUnavailable):
        rt.describe([np.zeros((8, 8, 3), np.uint8)], "p", max_tokens=8, timeout_s=5)
    assert rt.warm_up() is False                 # boot warm-up never raises


def test_env_selects_the_device_and_junk_is_refused(monkeypatch):
    monkeypatch.setenv("VLM_DEVICE", "cpu")
    assert MoondreamRuntime()._requested_device == "cpu"
    with pytest.raises(ValueError):
        MoondreamRuntime(device="gpu")


def test_describe_passes_the_frame_and_token_bound_to_the_model(monkeypatch):
    monkeypatch.setattr(_tf(), "AutoModelForCausalLM", fake_auto({}))
    rt = MoondreamRuntime(device="cpu", attempt_quantization=False)
    out = rt.describe([np.zeros((30, 40, 3), np.uint8)], "prompt", max_tokens=17, timeout_s=5)
    assert out == "seen 40x30 with 17 tokens"


def test_a_load_failure_becomes_unavailable_and_is_remembered(monkeypatch):
    class Boom:
        @staticmethod
        def from_pretrained(*a, **k):
            raise OSError("weights not on disk and the network is off")

    monkeypatch.setattr(_tf(), "AutoModelForCausalLM", Boom)
    rt = MoondreamRuntime(device="cpu")
    for _ in range(2):
        with pytest.raises(VlmUnavailable):
            rt.describe([np.zeros((8, 8, 3), np.uint8)], "p", max_tokens=8, timeout_s=5)
