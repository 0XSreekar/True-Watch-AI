"""The live pipeline's link to the explanation model (vlm/).

Every event the service raises carries a reason from its first instant: `EventExplainer.prepare()` renders
the rule sentence (vlm.judge.initial_reason: pure string work, no model) and the service puts it in the
event's `explanation` block with source 'template' BEFORE the event is emitted. After the emit,
`EventExplainer.submit()` hands the best frame, the rule evidence and the camera context to the bounded
queue in vlm/queue.py and returns at once; a worker thread runs the model and, only if its sentence
passes the guardrails, `EventSink.emit_patch` replaces the reason for the same `event_id`.

What that guarantees
    * the pipeline thread never waits on the model: `submit()` is a queue append (plus one frame resize);
    * the reason is never empty: a disabled model, a load failure, a timeout, an exception, a guardrail
      rejection and a full queue all leave the rule sentence in place (vlm/judge.py holds the chain);
    * memory is bounded: at most `queue_max` jobs wait, each holding one frame downscaled to
      `frame_max_side`, and when the queue is full the OLDEST waiting job is dropped (its event keeps the
      rule sentence) so the newest event, the one an operator is most likely looking at, is explained.

The model is loaded once, in a background thread, so a boot never waits for it.
`build_explainer(cfg)` decides whether there is a model at all (see `vlm_enabled` in config.py).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from pipeline.events import EventSink
from vlm.judge import LAST_RESORT_REASON, Explanation, JudgeInput, initial_reason
from vlm.prompt import CameraContext, EventContext
from vlm.queue import ExplanationQueue

log = logging.getLogger("truewatch.edge.explain")

DEFAULT_FRAME_MAX_SIDE = 768


def downscale(image: np.ndarray, max_side: int) -> np.ndarray:
    """A copy of `image` whose longest side is at most `max_side` (a copy either way: the pipeline reuses
    its frame buffers, and a queued job must not see them change)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if max_side <= 0 or longest <= max_side:
        return image.copy()
    import cv2

    scale = max_side / float(longest)
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class PendingExplanation:
    """One event's judge input, built before the event is emitted and submitted after."""

    event: dict
    inp: JudgeInput
    initial: Explanation


class EventExplainer:
    """Reason for every event now, model sentence later. `queue=None` means no model: template only."""

    def __init__(
        self,
        queue: ExplanationQueue | None = None,
        *,
        frame_max_side: int = DEFAULT_FRAME_MAX_SIDE,
    ) -> None:
        self.queue = queue
        self.frame_max_side = frame_max_side
        self.submitted = 0

    @property
    def enabled(self) -> bool:
        return self.queue is not None

    def prepare(self, event: dict, *, camera: CameraContext, fired: Sequence[Any], captured_at: float) -> PendingExplanation:
        """Build the judge input and the t=0 reason for `event`; sets `event["explanation"]` to it."""
        det = event.get("detection") or {}
        ir = bool(det.get("ir"))
        context = EventContext(
            camera=replace(camera, ir=ir) if camera.ir != ir else camera,
            captured_at=captured_at,
            object_class=det.get("class") or "person",
        )
        inp = JudgeInput(context=context, fired=tuple(fired))
        try:
            initial = initial_reason(inp)
        except Exception:  # noqa: BLE001 - even a broken template must not leave an event without a reason
            log.exception("could not render the rule sentence")
            initial = Explanation(text=LAST_RESORT_REASON, source="template")
        event["explanation"] = initial.to_event()
        return PendingExplanation(event=event, inp=inp, initial=initial)

    def submit(self, pending: PendingExplanation, sink: EventSink, image: np.ndarray | None) -> bool:
        """Queue the model call for an event that is already emitted. Never blocks, never raises."""
        if self.queue is None or image is None:
            return False
        try:
            inp = replace(pending.inp, frames=(downscale(image, self.frame_max_side),))
            event = pending.event

            def store(_job_id: str, explanation: Explanation) -> None:
                sink.emit_patch(event, explanation.to_event())

            self.queue.submit(event["event_id"], inp, store)
            self.submitted += 1
            return True
        except Exception:  # noqa: BLE001 - a reason upgrade must never take the pipeline down
            log.exception("could not queue the explanation for event %s; it keeps the rule sentence",
                          pending.event.get("event_id"))
            return False

    def stats(self) -> dict:
        if self.queue is None:
            return {"enabled": False}
        q = self.queue.stats()
        return {"enabled": True, "submitted": self.submitted, "pending": self.queue.qsize(), **vars(q)}

    def close(self) -> None:
        if self.queue is not None:
            self.queue.shutdown()


# ------------------------------------------------------------------------------------------ boot


def _weights_cached(model_id: str, revision: str) -> bool:
    """Whether the pinned checkpoint's weights are already in the local Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(model_id, "model.safetensors", revision=revision)
        return isinstance(found, str)
    except Exception:  # noqa: BLE001 - "cannot tell" means "not there"
        return False


def build_explainer(cfg, runtime=None) -> EventExplainer:
    """The explainer the service boots with, decided by `VLM_ENABLED`:

    * "0" (or false/off): no model. Every reason stays the rule sentence.
    * "1" (or true/on): use the model; its weights are downloaded if the cache lacks them.
    * "auto" (default): use the model only if its weights are already cached locally, and never under a test
      run; otherwise log why not and run without it.

    `runtime` is an injection point for tests; production leaves it None and gets `vlm.runtime`'s local
    Moondream 2 (or, only when TRUEWATCH_VLM_HOSTED=1, the loudly-labelled development fallback).
    The model loads in a background thread; until it is ready the queue's jobs simply wait, and any that
    fail leave the rule sentence in place.
    """
    mode = (cfg.vlm_enabled or "auto").lower()
    max_side = getattr(cfg, "vlm_frame_max_side", DEFAULT_FRAME_MAX_SIDE)
    if runtime is None:
        if mode in {"0", "false", "no", "off"}:
            log.info("VLM_ENABLED=%s: explanations stay the rule sentence", mode)
            return EventExplainer(None, frame_max_side=max_side)
        from vlm import runtime as vlm_runtime

        hosted = os.environ.get(vlm_runtime.HOSTED_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}
        if mode == "auto":
            if "PYTEST_CURRENT_TEST" in os.environ:
                return EventExplainer(None, frame_max_side=max_side)
            if not hosted and not _weights_cached(vlm_runtime.MODEL_ID, vlm_runtime.PINNED_REVISION):
                log.warning(
                    "VLM_ENABLED=auto: Moondream 2 (%s@%s) is not in the local cache, so explanations stay "
                    "the rule sentence. Pre-download it once (see edge/README.md) or set VLM_ENABLED=1 to "
                    "fetch it at boot.", vlm_runtime.MODEL_ID, vlm_runtime.PINNED_REVISION[:8])
                return EventExplainer(None, frame_max_side=max_side)
        try:
            runtime = vlm_runtime.build_default_runtime()
        except Exception as exc:  # noqa: BLE001 - e.g. hosted fallback opted into without its URL/key
            log.error("VLM runtime could not be built (%s); explanations stay the rule sentence", exc)
            return EventExplainer(None, frame_max_side=max_side)
    warm = getattr(runtime, "warm_up", None)
    if callable(warm):
        threading.Thread(target=warm, name="vlm-warm-up", daemon=True).start()
    queue = ExplanationQueue(
        runtime,
        maxsize=getattr(cfg, "vlm_queue_max", 8),
        timeout_s=getattr(cfg, "vlm_timeout_s", 60.0),
        cooldown_s=getattr(cfg, "vlm_cooldown_s", 0.0),
    )
    return EventExplainer(queue, frame_max_side=max_side)
