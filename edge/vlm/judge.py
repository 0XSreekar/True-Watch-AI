"""The explanation judge: prompt -> model call -> guardrails -> fallback.

This module owns the decision of *what sentence an alert carries*; it owns no
model and no threads. The model is injected as a `VlmRuntime`, so every path
here is testable with a stub, and the scheduling (the alert goes out at
t=0 with the rule sentence, the model result patches it ~3 s later) belongs
to `vlm.queue`.

Interfaces the rest of Phase 6 builds on
----------------------------------------
``VlmRuntime`` (implemented by ``vlm/runtime.py``)::

    class VlmRuntime(Protocol):
        model_name: str                       # e.g. "moondream2"; goes into explanation.model
        def describe(self, images: Sequence[Any], prompt: str, *,
                     max_tokens: int, timeout_s: float) -> str: ...

    * `images`: 1..`MAX_FRAMES` frames of the fused event, best first — BGR
      ``uint8`` numpy arrays as the pipeline produces them, or PIL images; the
      runtime converts to whatever the model wants.
    * `prompt`: the complete text from `vlm.prompt.build_prompt`; the runtime
      must pass it through unchanged (no extra system text).
    * returns the raw generated text; guardrails handle every clean-up.
    * raises `VlmUnavailable` when the model is not loaded / failed to load,
      `TimeoutError` when `timeout_s` elapsed. Any other exception is also
      caught here and turned into the fallback — a runtime can never blank
      a reason or crash the caller.
    * must be safe to call from a worker thread; `explain` is synchronous and
      blocking by design so `queue.py` decides where it runs.

``vlm/queue.py`` uses exactly two functions from here::

    initial = initial_reason(inp)          # t=0: Explanation(source="template"), instant, no model
    final   = explain(inp, runtime)        # t~3 s, in a worker; never raises, never empty

    and patches ``alert.reason = final.text`` / ``event.explanation =
    final.to_event()`` only when ``final.source == "vlm"`` (a template
    result is already what the alert carries).

``vlm/latency.py`` reads ``Explanation.latency_ms`` (wall time of the
``describe`` call alone, measured here with a monotonic clock) and
``Explanation.violations`` (to report rejection rate alongside p50/p95).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, runtime_checkable

from . import guardrails
from .prompt import MAX_REASON_CHARS, EventContext, FiredRuleLike, build_facts, build_prompt, rule_sentence

log = logging.getLogger("truewatch.vlm")

DEFAULT_MAX_TOKENS = 80  # one 220-char sentence is ~50 tokens; headroom, not licence
DEFAULT_TIMEOUT_S = 8.0  # CPU budget; the ~3 s in the PDF is a Jetson target
MAX_FRAMES = 3
LAST_RESORT_REASON = "Movement flagged by both detection channels."


class VlmUnavailable(RuntimeError):
    """The runtime has no usable model (not loaded, failed to load, or shut down)."""


@runtime_checkable
class VlmRuntime(Protocol):
    model_name: str

    def describe(self, images: Sequence[Any], prompt: str, *, max_tokens: int, timeout_s: float) -> str: ...


class CallableRuntime:
    """Adapt a plain ``fn(images, prompt, max_tokens, timeout_s) -> str`` to `VlmRuntime`."""

    def __init__(self, fn: Callable[[Sequence[Any], str, int, float], str], model_name: str = "stub") -> None:
        self._fn = fn
        self.model_name = model_name

    def describe(self, images: Sequence[Any], prompt: str, *, max_tokens: int, timeout_s: float) -> str:
        return self._fn(images, prompt, max_tokens, timeout_s)


@dataclass(frozen=True)
class JudgeInput:
    """One fused event, as the judge sees it."""

    context: EventContext
    fired: tuple[Any, ...] = ()  # rules.engine.FiredRule (or anything FiredRuleLike)
    frames: tuple[Any, ...] = ()

    @classmethod
    def from_verdict(cls, verdict: Any, context: EventContext, frames: Sequence[Any] = ()) -> "JudgeInput":
        """Build from a Phase 4 `TrackVerdict`."""
        return cls(context=context, fired=tuple(getattr(verdict, "fired", ()) or ()), frames=tuple(frames))

    @property
    def evidence(self) -> dict[str, Any]:
        """``{rule_name: evidence}`` — the same shape as `TrackVerdict.evidence`."""
        out: dict[str, Any] = {}
        for f in self.fired:
            name = getattr(f, "rule_name", None)
            if isinstance(name, str) and name:
                ev = getattr(f, "evidence", None)
                out[name] = ev if isinstance(ev, Mapping) else {}
        return out


@dataclass(frozen=True)
class Explanation:
    text: str
    source: Literal["vlm", "template"]
    model: str | None = None
    latency_ms: int | None = None
    violations: tuple[str, ...] = ()
    repairs: tuple[str, ...] = ()
    raw: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("an Explanation must never carry an empty reason")

    def to_event(self) -> dict[str, Any]:
        """The ``explanation`` block of truewatch.event.v1 (docs/ARCHITECTURE_V2.md section 4.2)."""
        return {"text": self.text, "source": self.source, "model": self.model, "latency_ms": self.latency_ms}


# --------------------------------------------------------------------------
# Fallback chain — the reason is never empty
# --------------------------------------------------------------------------


def _clip(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > MAX_REASON_CHARS - 1:
        text = text[: MAX_REASON_CHARS - 1].rsplit(" ", 1)[0].rstrip(",;:")
    text = text[:1].upper() + text[1:]
    return text if text.endswith(".") else text + "."


def _fmt_value(v: Any) -> str | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return f"{v:.1f}".rstrip("0").rstrip(".") if math.isfinite(v) else None
    if isinstance(v, str) and v.strip():
        return " ".join(v.split())[:32]
    return None


def _phase4_strings(fired: Sequence[Any]) -> str | None:
    parts = [r.strip() for r in (getattr(f, "reason", None) for f in fired) if isinstance(r, str) and r.strip()]
    return _clip("; ".join(parts)) if parts else None


def _from_rule_evidence(fired: Sequence[Any]) -> str | None:
    parts = []
    for f in fired:
        name = getattr(f, "rule_name", None)
        if not isinstance(name, str) or not name.strip():
            continue
        ev = getattr(f, "evidence", None)
        items = []
        if isinstance(ev, Mapping):
            for k, v in ev.items():
                s = _fmt_value(v)
                if s is not None:
                    items.append(f"{str(k).replace('_', ' ')} {s}")
        parts.append(name.strip() + (f" ({', '.join(items[:4])})" if items else ""))
    return _clip("; ".join(parts)) if parts else None


def fallback_reason(inp: JudgeInput) -> str:
    """The rule-string reason. Tried in order, first non-empty wins:

    1. `rule_sentence` — the REASONS-register sentence rendered from evidence,
       kept only if it passes the same guardrails as model output;
    2. the Phase 4 `FiredRule.reason` strings, joined;
    3. a sentence built from each fired rule's name and its evidence values;
    4. `LAST_RESORT_REASON` — only fused (both-channel) events reach here.
    """
    try:
        sentence = rule_sentence(inp.evidence, inp.context)
        if guardrails.check(sentence, build_facts(inp.evidence, inp.context)).accepted:
            return sentence
        log.warning("rule sentence failed its own guardrail, using phase 4 strings: %r", sentence)
    except Exception:  # noqa: BLE001 - the reason must survive any bad input
        log.exception("rule sentence could not be rendered")
    for build in (_phase4_strings, _from_rule_evidence):
        try:
            text = build(inp.fired)
        except Exception:  # noqa: BLE001
            log.exception("fallback %s failed", build.__name__)
            continue
        if text and text.strip(". "):
            return text
    return LAST_RESORT_REASON


def initial_reason(inp: JudgeInput) -> Explanation:
    """The t=0 reason a provisional alert carries. Instant; never touches the model."""
    return Explanation(text=fallback_reason(inp), source="template")


# --------------------------------------------------------------------------
# The judge
# --------------------------------------------------------------------------


def explain(
    inp: JudgeInput,
    runtime: VlmRuntime | None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    clock: Callable[[], float] = time.monotonic,
) -> Explanation:
    """Ask the model for one grounded sentence; return it if it passes, else the rule string.

    Never raises and never returns an empty reason. A result that arrives
    after `timeout_s` (a runtime that ignored its own deadline) is dropped the
    same way a rejected one is, so the latency the operator sees is bounded
    by what this function promises, not by what the runtime honours.
    """
    model_name = getattr(runtime, "model_name", None) if runtime is not None else None

    def fallback(violations: Sequence[str], raw: str | None = None, latency_ms: int | None = None,
                 repairs: Sequence[str] = ()) -> Explanation:
        return Explanation(
            text=fallback_reason(inp),
            source="template",
            model=None,
            latency_ms=latency_ms,
            violations=tuple(violations),
            repairs=tuple(repairs),
            raw=raw,
        )

    if runtime is None:
        return fallback(("no_runtime",))

    try:
        facts = build_facts(inp.evidence, inp.context)
        prompt = build_prompt(inp.evidence, inp.context)
    except Exception as exc:  # noqa: BLE001
        log.exception("could not build the prompt")
        return fallback((f"prompt_error: {type(exc).__name__}",))

    frames = tuple(inp.frames[:MAX_FRAMES])
    started = clock()
    try:
        raw = runtime.describe(frames, prompt, max_tokens=max_tokens, timeout_s=timeout_s)
    except TimeoutError:
        return fallback(("timeout",), latency_ms=_ms(clock() - started))
    except VlmUnavailable as exc:
        return fallback((f"unavailable: {exc}",))
    except Exception as exc:  # noqa: BLE001 - a crashing runtime must not blank the reason
        log.exception("vision-language runtime raised")
        return fallback((f"runtime_error: {type(exc).__name__}",), latency_ms=_ms(clock() - started))
    elapsed = clock() - started
    latency_ms = _ms(elapsed)

    if elapsed > timeout_s:
        return fallback(("timeout",), raw=_raw(raw), latency_ms=latency_ms)
    if not isinstance(raw, str):
        return fallback(("empty: non-text output",), latency_ms=latency_ms)

    try:
        result = guardrails.check(raw, facts)
    except Exception as exc:  # noqa: BLE001
        log.exception("guardrail check raised")
        return fallback((f"guardrail_error: {type(exc).__name__}",), raw=raw, latency_ms=latency_ms)

    if not result.accepted:
        reasons = tuple(str(v) for v in result.violations)
        log.info("explanation rejected (%s): %r", "; ".join(reasons), raw)
        return fallback(reasons, raw=raw, latency_ms=latency_ms, repairs=result.repairs)

    return Explanation(
        text=result.text,
        source="vlm",
        model=model_name,
        latency_ms=latency_ms,
        repairs=result.repairs,
        raw=raw,
    )


def _ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


def _raw(v: Any) -> str | None:
    return v if isinstance(v, str) else None


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT_S",
    "LAST_RESORT_REASON",
    "MAX_FRAMES",
    "CallableRuntime",
    "Explanation",
    "FiredRuleLike",
    "JudgeInput",
    "VlmRuntime",
    "VlmUnavailable",
    "explain",
    "fallback_reason",
    "initial_reason",
]
