"""The asynchronous explanation queue — the mechanism that guarantees the
alert is emitted before the VLM runs.

ASYNC CONTRACT (matches the PPT timeline, docs/ARCHITECTURE_V2.md section 4.2)
    t=0ms   a fused event becomes an alert: `judge.initial_reason(inp)` runs
            synchronously on the caller's thread (it never touches a model —
            it is pure string rendering over already-computed rule evidence),
            and the caller emits the alert immediately with that reason.
    t~3s    a worker thread runs `judge.explain(inp, runtime)`; when it
            finishes, the result *patches* the alert already emitted at t=0.
    operator acts -> stage 'sealed' (outside this module's concern).

Why the pipeline can never block on the VLM
--------------------------------------------
`ExplanationQueue.submit()` does three things and returns, all synchronous
and cheap: render the template reason, hand the caller that reason to emit
right now, and hand the slow work to a background worker thread. It never
calls `judge.explain` on the caller's thread. The worker thread is a
`threading.Thread`, not asyncio — this queue is used from `pipeline/`'s
synchronous ingest loop (Phase 1-5), not from an event loop, so a plain
thread with a bounded `queue.Queue` in front of it is the mechanism that
actually exists on the caller's stack, not a hypothetical future one. A
FastAPI/asyncio caller (Phase 8's HTTP layer) can await `submit_async`
instead, which wraps the same submission in `run_in_executor` — the
worker pool doing the actual VLM call is identical either way.

Bounded, drop-oldest
---------------------
The queue is bounded (`maxsize`); a producer that outruns the worker (a
model stuck on a slow request, or a burst of fused events) drops the
*oldest* still-pending job rather than blocking the producer or growing
without limit. A dropped job's alert already carries its t=0 template
reason (submitted before the drop decision), so dropping it costs the
operator a possible upgrade to a VLM sentence, never a reason. Drops are
counted (`ExplanationQueue.dropped_count`) so an operator dashboard or test
can see it happening.

Both polling and SSE, from one patch
---------------------------------------
`submit()` takes a `store` callback (`Callable[[str, Explanation], None]`,
job_id -> patch the already-emitted alert/event in place — the polling
side, e.g. writing into the same dict `GET /alerts` reads) and an optional
`on_result` hook (any number of subscribers, e.g. an SSE broadcaster Phase 8
adds) that both fire from the same worker call, in that order, so neither
consumer can see a different final reason than the other. Nothing here
imports FastAPI, so Phase 8 wires SSE by registering a subscriber, not by
this module knowing what SSE is.

Timeout handling
-----------------
`judge.explain` already enforces its own `timeout_s` and — critically — drops
a result that arrives *after* that deadline (see its docstring: "A result
that arrives after timeout_s... is dropped the same way a rejected one is").
This module's own worker-side timeout is a second, coarser safety net for a
runtime that hangs completely past even that: `submit()`'s `hard_timeout_s`
bounds how long the worker will wait for `judge.explain` to return at all
before giving up on that job and moving to the next one, so one wedged model
call cannot starve every alert behind it in the queue.
"""

from __future__ import annotations

import logging
import queue as _queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .judge import DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT_S, Explanation, JudgeInput, VlmRuntime, explain, initial_reason

log = logging.getLogger("truewatch.vlm.queue")

DEFAULT_MAXSIZE = 64
DEFAULT_HARD_TIMEOUT_S = 20.0  # judge.explain's own DEFAULT_TIMEOUT_S (8s) plus headroom

StoreFn = Callable[[str, Explanation], None]
SubscriberFn = Callable[[str, Explanation], None]


@dataclass
class _Job:
    job_id: str
    inp: JudgeInput
    submitted_at: float
    store: StoreFn
    on_result: Optional[SubscriberFn]
    hard_timeout_s: float


@dataclass
class QueueStats:
    submitted: int = 0
    completed: int = 0
    dropped: int = 0
    timed_out: int = 0
    errored: int = 0


class ExplanationQueue:
    """Bounded, single-worker background queue that turns `JudgeInput`s into
    `Explanation` patches without ever blocking the caller.

    One instance per process is the normal deployment (`get_queue()` below);
    tests construct their own so runs never share state.
    """

    def __init__(
        self,
        runtime: VlmRuntime | None,
        *,
        maxsize: int = DEFAULT_MAXSIZE,
        workers: int = 1,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cooldown_s: float = 0.0,
    ) -> None:
        if workers < 1:
            raise ValueError("ExplanationQueue needs at least one worker")
        if maxsize < 1:
            raise ValueError("ExplanationQueue needs room for at least one job")
        self._runtime = runtime
        # Handed to `judge.explain` for every job. A CPU deployment measures its own latency
        # (vlm/latency.py) and sets a timeout above it; the judge default is sized for the edge target.
        self._timeout_s = float(timeout_s)
        self._max_tokens = int(max_tokens)
        # Rest between one job's end and the next job's start. On a machine where the model shares compute
        # with the frame loop (measured on Apple silicon: edge/README.md) a busy model costs frame rate; a
        # cooldown caps its duty cycle. 0 = run back to back.
        self._cooldown_s = max(0.0, float(cooldown_s))
        self._q: "_queue.Queue[_Job]" = _queue.Queue(maxsize=maxsize)
        self._stats = QueueStats()
        self._stats_lock = threading.Lock()
        self._global_subscribers: list[SubscriberFn] = []
        self._subscribers_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        for i in range(workers):
            t = threading.Thread(
                target=self._worker_loop, name=f"vlm-explain-worker-{i}", daemon=True
            )
            t.start()
            self._threads.append(t)

    # -- submission ----------------------------------------------------------

    def submit(
        self,
        job_id: str,
        inp: JudgeInput,
        store: StoreFn,
        *,
        on_result: SubscriberFn | None = None,
        hard_timeout_s: float | None = None,
    ) -> Explanation:
        """t=0: compute and return the template reason immediately, and enqueue
        the slow VLM work. The caller emits the alert with the returned
        `Explanation` on its own thread, right after this call returns —
        `submit` never runs the model and never blocks on it.

        `store(job_id, explanation)` is called once, later, from a worker
        thread, ONLY if the VLM path produces a result (`explanation.source
        == "vlm"`) or hard-fails in a way that still needs to overwrite a
        stale template reason with a fresher one (never with an empty one —
        see `judge.explain`). If the queue is full, the oldest pending job is
        dropped (counted in `stats().dropped`) to make room; the job just
        submitted is never the one dropped, so a caller's own request is
        always accepted (bounded by `maxsize` total in flight, not per
        caller).
        """
        initial = initial_reason(inp)
        if hard_timeout_s is None:
            hard_timeout_s = max(DEFAULT_HARD_TIMEOUT_S, self._timeout_s + 12.0)
        job = _Job(
            job_id=job_id,
            inp=inp,
            submitted_at=time.monotonic(),
            store=store,
            on_result=on_result,
            hard_timeout_s=hard_timeout_s,
        )
        while True:
            try:
                self._q.put_nowait(job)
                break
            except _queue.Full:
                try:
                    dropped = self._q.get_nowait()
                    # A job taken off here never reaches a worker's own task_done();
                    # without this `unfinished_tasks` never returns to zero after a
                    # drop and `join()` waits forever.
                    self._q.task_done()
                    with self._stats_lock:
                        self._stats.dropped += 1
                    log.warning(
                        "explanation queue full (maxsize=%d); dropped oldest pending job %r "
                        "to make room for %r",
                        self._q.maxsize,
                        dropped.job_id,
                        job_id,
                    )
                except _queue.Empty:  # pragma: no cover - racing consumer, retry
                    continue
        with self._stats_lock:
            self._stats.submitted += 1
        return initial

    async def submit_async(
        self,
        job_id: str,
        inp: JudgeInput,
        store: StoreFn,
        *,
        on_result: SubscriberFn | None = None,
        hard_timeout_s: float | None = None,
    ) -> Explanation:
        """`asyncio` callers (Phase 8's HTTP layer): identical contract to
        `submit`, off the event loop thread so it never blocks it either."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.submit(job_id, inp, store, on_result=on_result, hard_timeout_s=hard_timeout_s),
        )

    # -- subscribers (SSE, Phase 8) ------------------------------------------

    def subscribe(self, fn: SubscriberFn) -> Callable[[], None]:
        """Register a process-wide subscriber (e.g. an SSE broadcaster) that
        fires for every completed job, in addition to any per-job `on_result`
        passed to `submit`. Returns an unsubscribe callable."""
        with self._subscribers_lock:
            self._global_subscribers.append(fn)

        def _unsubscribe() -> None:
            with self._subscribers_lock:
                if fn in self._global_subscribers:
                    self._global_subscribers.remove(fn)

        return _unsubscribe

    # -- worker ----------------------------------------------------------------

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except _queue.Empty:
                continue
            try:
                self._run_job(job)
            except Exception:  # noqa: BLE001 - a worker must never die on a bad job
                log.exception("explanation worker failed on job %r", job.job_id)
                with self._stats_lock:
                    self._stats.errored += 1
            finally:
                self._q.task_done()
            if self._cooldown_s:
                self._stop.wait(self._cooldown_s)

    def _run_job(self, job: _Job) -> None:
        result_box: dict[str, Explanation] = {}

        def _call() -> None:
            result_box["result"] = explain(
                job.inp, self._runtime, max_tokens=self._max_tokens, timeout_s=self._timeout_s
            )

        worker = threading.Thread(target=_call, daemon=True, name=f"vlm-explain-{job.job_id}")
        worker.start()
        worker.join(job.hard_timeout_s)

        if worker.is_alive():
            # judge.explain itself hung past its own timeout_s *and* our
            # coarser hard_timeout_s: give up on this job so it cannot starve
            # the ones behind it. The alert already carries its t=0 template
            # reason from submit(); we simply do not patch it. The daemon
            # thread is abandoned — its eventual answer is dropped, exactly
            # like a late answer inside judge.explain itself.
            with self._stats_lock:
                self._stats.timed_out += 1
            log.error(
                "explanation worker hard-timeout (%.1fs) on job %r; leaving its template reason in place",
                job.hard_timeout_s,
                job.job_id,
            )
            return

        result = result_box.get("result")
        if result is None:  # pragma: no cover - _call always sets it or raises into the outer handler
            return

        with self._stats_lock:
            self._stats.completed += 1

        # Only patch when there's something worth patching to: a template
        # result here just re-confirms what submit() already emitted, so
        # skipping the store call avoids a redundant write while keeping the
        # invariant "reason is never empty" trivially true (nothing changes).
        if result.source != "vlm":
            log.info("job %r stayed on the template reason (%s)", job.job_id, "; ".join(result.violations) or "no violations recorded")
            return

        try:
            job.store(job.job_id, result)
        except Exception:  # noqa: BLE001
            log.exception("store callback failed for job %r; alert keeps its previous reason", job.job_id)
            return

        subscribers: list[SubscriberFn] = []
        with self._subscribers_lock:
            subscribers.extend(self._global_subscribers)
        if job.on_result is not None:
            subscribers.append(job.on_result)
        for fn in subscribers:
            try:
                fn(job.job_id, result)
            except Exception:  # noqa: BLE001 - one bad subscriber must not break the others
                log.exception("subscriber failed for job %r", job.job_id)

    # -- introspection ---------------------------------------------------------

    def stats(self) -> QueueStats:
        with self._stats_lock:
            return QueueStats(**vars(self._stats))

    @property
    def dropped_count(self) -> int:
        return self.stats().dropped

    def qsize(self) -> int:
        return self._q.qsize()

    def join(self, timeout: float | None = None) -> bool:
        """Block until every currently-enqueued job has been processed (or
        `timeout` elapses). Test-only convenience; production code never
        waits on the queue it just fed."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._q.unfinished_tasks == 0:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def shutdown(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------
# Process-wide singleton, the shape edge/app.py (Phase 8) will reach for.
# --------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton: ExplanationQueue | None = None


def get_queue(runtime_factory: Callable[[], VlmRuntime | None] | None = None) -> ExplanationQueue:
    """The process-wide explanation queue. Built lazily on first use so
    importing this module never loads a model. `runtime_factory` is called
    at most once, only if no queue exists yet; production passes
    `vlm.runtime.build_default_runtime`, tests pass a stub factory."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            runtime = runtime_factory() if runtime_factory is not None else None
            _singleton = ExplanationQueue(runtime)
        return _singleton


def reset_queue_for_tests() -> None:
    """Test-only: drop the process-wide singleton so the next `get_queue()`
    builds a fresh one. Never called from production code."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.shutdown()
        _singleton = None


__all__ = [
    "DEFAULT_HARD_TIMEOUT_S",
    "DEFAULT_MAXSIZE",
    "ExplanationQueue",
    "QueueStats",
    "get_queue",
    "reset_queue_for_tests",
]
