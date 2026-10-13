"""Plain-language explanation for fused alerts (slide 2 claim (c)).

prompt.py      the fact sheet built from Phase 4 rule evidence + camera context,
               the grounded prompt, and `rule_sentence` — the deterministic
               REASONS-register sentence used at t=0 and as the fallback.
guardrails.py  reject/repair model output; every claim must be in the evidence.
judge.py       prompt -> `VlmRuntime.describe` -> guardrails -> fallback.
               Defines the `VlmRuntime` protocol that runtime.py implements
               and the `initial_reason` / `explain` pair queue.py schedules.
runtime.py     local Moondream 2 (CPU), implementing `VlmRuntime`. The only
               runtime loaded by default; a hosted dev-only fallback exists,
               off by default, env-gated, and loudly labelled.
queue.py       the asynchronous scheduler: the alert is emitted at t=0 with
               the template reason, a worker thread later patches it with
               the VLM sentence. Never blocks the pipeline.

Invariant: every path returns a non-empty `reason`.
"""

from .guardrails import GuardrailResult, Violation, check
from .judge import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_S,
    CallableRuntime,
    Explanation,
    JudgeInput,
    VlmRuntime,
    VlmUnavailable,
    explain,
    fallback_reason,
    initial_reason,
)
from .prompt import (
    MAX_REASON_CHARS,
    CameraContext,
    EventContext,
    FactSheet,
    build_facts,
    build_prompt,
    rule_sentence,
)
from .queue import ExplanationQueue, QueueStats, get_queue

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT_S",
    "MAX_REASON_CHARS",
    "CallableRuntime",
    "CameraContext",
    "EventContext",
    "Explanation",
    "ExplanationQueue",
    "FactSheet",
    "GuardrailResult",
    "JudgeInput",
    "QueueStats",
    "Violation",
    "VlmRuntime",
    "VlmUnavailable",
    "build_facts",
    "build_prompt",
    "check",
    "explain",
    "fallback_reason",
    "get_queue",
    "initial_reason",
    "rule_sentence",
]
