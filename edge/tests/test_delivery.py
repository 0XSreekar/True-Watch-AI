"""Edge -> backend delivery (pipeline/events.py BackendDelivery / EventSink), ARCHITECTURE_V2 section 4.1.

A real local HTTP server plays the backend, so the bytes on the wire (headers, JSON body) are what is
asserted, not a mock of the sender.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pipeline.events import (INGEST_PATH, KEY_HEADER, NONCE_HEADER, TIMESTAMP_HEADER, BackendDelivery, EventSink,
                             build_reason_patch)

KEY = "0f" * 32


class Backend:
    """Records every request; answers from a script of statuses, then 200."""

    def __init__(self, statuses=()) -> None:
        self.requests: list[dict] = []
        self.statuses = list(statuses)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": json.loads(body)})
                status = outer.statuses.pop(0) if outer.statuses else 200
                payload = json.dumps({"accepted": status < 300, "duplicate": status == 409}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def backend():
    b = Backend()
    yield b
    b.close()


def event(n: int = 1) -> dict:
    return {"schema": "truewatch.event.v1", "event_id": f"00000000-0000-4000-8000-{n:012d}", "post_id": "RXL",
            "camera_id": "RXL-01", "captured_at": "2026-08-14T02:41:07.312Z", "stage": "provisional",
            "detection": {"track_id": n}, "rule": None,
            "explanation": {"text": "Movement flagged by both detection channels.", "source": "template",
                            "model": None, "latency_ms": None},
            "anpr": None, "evidence": None, "confidence": 90, "source": "file"}


def fast(url, **kw) -> BackendDelivery:
    kw.setdefault("backoff_s", 0.01)
    kw.setdefault("backoff_cap_s", 0.05)
    return BackendDelivery(url, KEY, **kw)


def free_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"      # nothing listens here once s is closed


# ------------------------------------------------------------------ the wire format


def test_created_event_and_reason_patch_reach_the_backend_in_order_with_the_documented_headers(backend):
    sink = EventSink(delivery=fast(backend.url))
    ev = event()
    sink.emit(ev)
    patch = sink.emit_patch(ev, {"text": "Person crossed the fence at 02:41.", "source": "vlm",
                                 "model": "moondream2", "latency_ms": 2840})
    assert sink.delivery.flush(5)
    created, patched = backend.requests
    for req in (created, patched):
        assert req["path"] == INGEST_PATH
        h = req["headers"]
        assert h["content-type"] == "application/json"
        assert h[KEY_HEADER.lower()] == KEY
        assert h[TIMESTAMP_HEADER.lower()].endswith("Z") and h[NONCE_HEADER.lower()]
    assert created["headers"][NONCE_HEADER.lower()] != patched["headers"][NONCE_HEADER.lower()]
    assert created["body"] == ev
    assert patched["body"] == patch == build_reason_patch(ev, patch["explanation"])
    assert patched["body"]["event_id"] == ev["event_id"] and patched["body"]["type"] == "reason_patch"
    assert patched["body"]["explanation"]["source"] == "vlm"
    assert sink.delivery.stats.delivered == 2 and sink.delivery.stats.retries == 0


def test_the_ring_copy_is_patched_but_a_reader_holding_the_old_dict_is_not_disturbed():
    sink = EventSink()
    ev = event()
    sink.emit(ev)
    before = sink.recent(1)[0]
    sink.emit_patch(ev, {"text": "New reason.", "source": "vlm", "model": "m", "latency_ms": 1})
    assert sink.recent(1)[0]["explanation"]["text"] == "New reason."
    assert before["explanation"]["source"] == "template"


def test_secret_is_never_logged(backend, caplog):
    with caplog.at_level("DEBUG"):
        d = fast(backend.url)
        d.enqueue(event())
        d.flush(5)
    assert KEY not in caplog.text


# ------------------------------------------------------------------ retry


def test_retries_a_5xx_with_a_fresh_nonce_and_the_same_event_id_then_succeeds():
    b = Backend(statuses=[503, 500])
    try:
        d = fast(b.url)
        d.enqueue(event())
        assert d.flush(5)
        assert len(b.requests) == 3
        assert d.stats.retries == 2 and d.stats.delivered == 1 and d.stats.dropped_exhausted == 0
        assert len({r["headers"][NONCE_HEADER.lower()] for r in b.requests}) == 3     # never a replayed nonce
        assert {r["body"]["event_id"] for r in b.requests} == {event()["event_id"]}
    finally:
        b.close()


def test_a_409_means_the_backend_already_has_the_event_and_is_not_retried():
    b = Backend(statuses=[409])
    try:
        d = fast(b.url)
        d.enqueue(event())
        assert d.flush(5)
        assert len(b.requests) == 1 and d.stats.duplicates == 1 and d.stats.delivered == 1
    finally:
        b.close()


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_a_permanent_refusal_is_dropped_without_retry_and_its_patch_is_skipped(status):
    b = Backend(statuses=[status])
    try:
        sink = EventSink(delivery=fast(b.url))
        ev = event()
        sink.emit(ev)
        sink.emit_patch(ev, {"text": "x.", "source": "vlm", "model": "m", "latency_ms": 1})
        assert sink.delivery.flush(5)
        assert len(b.requests) == 1
        assert sink.delivery.stats.rejected == 1 and sink.delivery.stats.patches_skipped == 1
    finally:
        b.close()


# ------------------------------------------------------------------ backend down


def test_backend_down_drops_cleanly_after_bounded_retries_and_never_blocks_the_pipeline():
    sink = EventSink(delivery=fast(free_port_url(), max_attempts=3, timeout_s=0.5))
    t0 = time.perf_counter()
    ev = event()
    for i in range(50):
        sink.emit(event(i))                       # the pipeline thread's only cost is an outbox append
    sink.emit_patch(ev, {"text": "x.", "source": "vlm", "model": "m", "latency_ms": 1})
    assert time.perf_counter() - t0 < 1.0, "emit must not wait on the network"
    assert sink.total == 50 and len(sink.recent(100)) == 50          # local ring and log unaffected
    assert sink.delivery.flush(30)
    st = sink.delivery.stats
    assert st.delivered == 0 and st.dropped_exhausted == 50
    assert st.retries == 50 * 2                                       # max_attempts=3 -> 2 retries per event
    assert st.patches_skipped == 1                                    # its created event was never delivered
    assert st.last_error
    sink.close()                                                       # and shutdown does not hang or raise


def test_recovers_when_the_backend_comes_back(backend):
    d = fast(backend.url, max_attempts=2)
    backend.statuses = [503, 503]              # first event exhausts its attempts
    d.enqueue(event(1))
    assert d.flush(5)
    assert d.stats.dropped_exhausted == 1
    d.enqueue(event(2))                        # backend healthy again
    assert d.flush(5)
    assert d.stats.delivered == 1 and backend.requests[-1]["body"]["detection"]["track_id"] == 2


def test_outbox_is_bounded_and_drops_the_oldest_when_full():
    gate = threading.Event()
    sent: list[str] = []

    def slow_post(url, headers, body, timeout):
        gate.wait(10)
        sent.append(json.loads(body)["event_id"])
        return 200

    d = BackendDelivery("http://127.0.0.1:9", KEY, outbox_max=5, post=slow_post, backoff_s=0.01)
    d.enqueue(event(0))                        # picked up by the sender, which then blocks in slow_post
    deadline = time.time() + 5
    while d.pending() == 0 or len(d._outbox) > 0:
        assert time.time() < deadline
        time.sleep(0.005)
    for i in range(1, 21):
        d.enqueue(event(i))
        assert len(d._outbox) <= 5
    assert d.stats.dropped_overflow == 15
    gate.set()
    assert d.flush(5)
    assert sent[0].endswith("000000000000")                     # the in-flight one
    assert sent[1:] == [event(i)["event_id"] for i in range(16, 21)]     # the 5 newest survived, in order


def test_a_transport_exception_of_any_kind_is_retried_not_raised():
    calls = []

    def flaky(url, headers, body, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("weird")
        return 202

    d = BackendDelivery("http://127.0.0.1:9", KEY, post=flaky, backoff_s=0.001, backoff_cap_s=0.002)
    d.enqueue(event())
    assert d.flush(5) and d.stats.delivered == 1 and len(calls) == 3


def test_construction_refuses_a_missing_key_and_a_non_http_url():
    with pytest.raises(ValueError):
        BackendDelivery("http://127.0.0.1:1", "")
    with pytest.raises(ValueError):
        BackendDelivery("file:///etc/passwd", KEY)
