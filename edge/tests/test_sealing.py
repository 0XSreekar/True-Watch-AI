"""Clips are sealed into the hash chain at capture, and the block follows its event to the backend."""

from __future__ import annotations

import numpy as np

from evidence import EvidenceStore, chain_view, verify_view
from pipeline.events import EventSink
from pipeline.sealing import ClipSealer

T0 = 1_790_000_000.0


def _frame(i: int) -> np.ndarray:
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, i % 64] = 255
    return img


def _store(tmp_path):
    return EvidenceStore(tmp_path / "box", hq_dir=tmp_path / "hq")


def test_seal_chains_the_recent_window_and_verifies(tmp_path):
    store = _store(tmp_path)
    sealer = ClipSealer(store, "RXL-01", clip_seconds=2.0)
    for i in range(50):                                   # 5 s at 10 fps: only the last 2 s are kept
        sealer.add(_frame(i), T0 + i * 0.1)
    assert sealer.stats()["buffered_frames"] == 21
    first = sealer.seal()
    for i in range(50, 60):
        sealer.add(_frame(i), T0 + i * 0.1)
    second = sealer.seal()
    store.close()

    assert first["block_n"] + 1 == second["block_n"]
    assert len(first["clip_sha256"]) == 64 and first["clip_sha256"] != second["clip_sha256"]
    chain = chain_view(tmp_path / "box", include_full_hash=False)
    assert [set(b) for b in chain] == [{"n", "hash", "camera", "time"}] * 2 and chain[0]["camera"] == "RXL-01"
    result = verify_view(tmp_path / "box", hq_dir=tmp_path / "hq")
    assert result["intact"] is True and result["verified"] == result["total"] == 2


def test_nothing_buffered_seals_nothing(tmp_path):
    assert ClipSealer(_store(tmp_path), "RXL-01").seal() is None


def test_a_store_failure_is_counted_not_raised(tmp_path):
    class Broken:
        def heartbeat(self):
            pass

        def open_clip(self, *a, **k):
            raise OSError("disk full")

    sealer = ClipSealer(Broken(), "RXL-01")
    sealer.add(_frame(0), T0)
    assert sealer.seal() is None and sealer.stats()["failed"] == 1


def test_the_evidence_update_follows_the_created_event(tmp_path):
    sent = []

    class Outbox:
        def enqueue(self, message):
            sent.append(message)

    sink = EventSink(delivery=Outbox())
    event = {"schema": "truewatch.event.v1", "event_id": "e-1", "stage": "provisional", "explanation": None}
    sink.emit(event)
    evidence = {"evidence_id": "x", "block_n": 1, "clip_sha256": "a" * 64, "prev_sha256": "0" * 64,
                "sealed_at": "2026-09-30T01:00:00.000Z"}
    update = sink.emit_evidence(event, evidence)
    assert sent[0]["event_id"] == "e-1" and sent[0].get("schema") == "truewatch.event.v1"
    assert sent[-1] == update == {"schema": "truewatch.update.v1", "event_id": "e-1", "evidence": evidence}
    assert sink.recent(1)[0]["evidence"] == evidence
