"""Tests for edge/evidence: tamper, reorder, deletion, Merkle, signatures, gaps and the verify CLI.

Stores are built through the same EvidenceStore code path the pipeline uses, with a simulated
clock so a day of blocks takes milliseconds. Every failure case is checked twice: through the
report (which block, which check) and through `verify.main` (non-zero exit status).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import re
import signal
import stat
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evidence import certificate, tamper_demo, verify
from evidence.chain import GENESIS_PREV, Ledger, chain_view, ledger_path, make_block
from evidence.gaps import KIND_POWER_LOSS, KIND_RESTART, GapLog
from evidence.hasher import HashingWriter, hash_file
from evidence.merkle import (
    inclusion_proof,
    leaf_hash,
    merkle_root,
    node_hash,
    verify_inclusion,
)
from evidence.signer import PRIVATE_KEY_NAME, InsecureKeyFile, Signer
from evidence.store import ChainCorrupt, EvidenceStore, proof_for_block

EDGE_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = EDGE_DIR / "evidence"
CAMS = ("RXL-01", "JGB-03", "PNT-07")
DAY = "2026-09-10"  # IST day of every block built by `build` with the default start


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def build(tmp_path: Path, n: int, step: timedelta = timedelta(minutes=5), seal: bool = True):
    """n blocks starting 2026-09-10 08:30 IST; seal=True rolls the clock to the next day and signs."""
    clock = Clock(datetime(2026, 9, 10, 3, 0, tzinfo=timezone.utc))
    root, hq = tmp_path / "ev", tmp_path / "hq"
    store = EvidenceStore(root, hq_dir=hq, device_id="post-test", clock=clock)
    rng = random.Random(7)
    for i in range(n):
        w = store.open_clip(CAMS[i % len(CAMS)], clock.t)
        for _ in range(3):
            w.write_frame(bytes(rng.getrandbits(8) for _ in range(64)))
        clock.t += step
        w.seal(clock.t)
    if seal:
        clock.t += timedelta(days=2)
        store.seal_past_days()
    store.close()
    return root, hq


def run_verify(root: Path, hq: Path, *extra: str) -> int:
    return verify.main(["--data-dir", str(root), "--hq-dir", str(hq), *extra])


def rewrite_lines(path: Path, fn) -> None:
    lines = path.read_text("ascii").splitlines(keepends=True)
    path.write_text("".join(fn(lines)), "ascii")


def writable(path: Path) -> Path:
    os.chmod(path, 0o644)
    return path


# --------------------------------------------------------------------------- hashing at capture


def test_hash_during_write_matches_rehash_from_disk(tmp_path):
    p = tmp_path / "clip.mjpeg"
    with HashingWriter(p) as w:
        w.write(b"HDR")
        for i in range(10):
            w.write_frame(bytes([i]) * 100)
    assert w.digest.frame_count == 10 and w.digest.byte_length == 1003
    assert hash_file(p, frame_count=10) == w.digest
    # frame count and byte length are part of the digest input
    assert hash_file(p, frame_count=9).sha256 != w.digest.sha256
    expected = hashlib.sha256(p.read_bytes() + b"TWCLIP1" + (10).to_bytes(8, "big") + (1003).to_bytes(8, "big"))
    assert w.digest.sha256 == expected.hexdigest()


def test_canonical_serialisation_is_fixed():
    b = make_block(
        n=1, clip_id="c1", camera_id="RXL-01", started_at=1789000000, ended_at=1789000010.5,
        sealed_at=1789000011, clip_sha256="a" * 64, prev_block_hash=GENESIS_PREV,
    )
    assert b.canonical() == (
        b"truewatch.block.v1\n"
        b"n=1\n"
        b"clip_id=c1\n"
        b"camera_id=RXL-01\n"
        b"started_at=2026-09-10T00:26:40.000Z\n"
        b"ended_at=2026-09-10T00:26:50.500Z\n"
        b"sealed_at=2026-09-10T00:26:51.000Z\n"
        b"clip_sha256=" + b"a" * 64 + b"\n"
        b"prev_block_hash=" + b"0" * 64 + b"\n"
    )
    assert b.block_hash == "a41e85b3ad4187f3dc0d2e11633cdffc00a55a4d9032bf04a99de47ea5a34879"
    with pytest.raises(ValueError):
        make_block(n=1, clip_id="bad\nid", camera_id="RXL-01", started_at=0, ended_at=0, sealed_at=0,
                   clip_sha256="a" * 64, prev_block_hash=GENESIS_PREV)


# --------------------------------------------------------------------------- intact chain + frozen shapes


def test_append_100_blocks_verifies_intact(tmp_path):
    root, hq = build(tmp_path, 100)
    report = verify.verify_store(root, hq_dir=hq)
    assert report.ok, verify.render(report)
    frozen = report.frozen()
    assert set(frozen) == {"intact", "verified", "total", "blocks"}
    assert frozen["intact"] is True and frozen["verified"] == frozen["total"] == 100
    assert frozen["blocks"][0] == {"n": 1, "ok": True} and all(set(b) == {"n", "ok"} for b in frozen["blocks"])
    assert verify.verify_view(root, hq_dir=hq) == frozen
    assert run_verify(root, hq, "--from", DAY, "--to", DAY) == 0


def test_frozen_chain_shape(tmp_path):
    root, _ = build(tmp_path, 7)
    items = chain_view(root, limit=5)
    assert [i["n"] for i in items] == [3, 4, 5, 6, 7]
    for item in items:
        assert set(item) == {"n", "hash", "camera", "time", "full_hash"}
        assert re.fullmatch(r"[0-9a-f]{10}…[0-9a-f]{6}", item["hash"])
        assert item["hash"][:10] == item["full_hash"][:10] and item["hash"][-6:] == item["full_hash"][-6:]
        assert re.fullmatch(r"\d{2} [A-Z]{3} · \d{2}:\d{2}", item["time"])
        assert item["camera"] in CAMS
    assert set(chain_view(root, include_full_hash=False)[0]) == {"n", "hash", "camera", "time"}
    assert chain_view(root)[0]["time"] == "10 SEP · 08:30"  # 03:00 UTC is 08:30 IST


# --------------------------------------------------------------------------- tampering


def test_one_clip_byte_breaks_that_block_and_every_later_block(tmp_path):
    root, hq = build(tmp_path, 20)
    target = next(b for b in Ledger(ledger_path(root)).blocks() if b.n == 7)
    tamper_demo.flip_byte(root / target.clip_path, target.byte_length // 2)

    report = verify.verify_store(root, hq_dir=hq)
    assert not report.ok
    status = {b.n: b for b in report.blocks}
    assert all(status[n].ok for n in range(1, 7))
    assert not status[7].ok and status[7].root_cause
    assert "clip bytes" in " ".join(status[7].reasons)
    assert all(not status[n].ok and not status[n].root_cause for n in range(8, 21))
    assert report.first_broken.n == 7
    merkle = [c for c in report.checks if c.name == "daily Merkle root" and c.scope == DAY]
    assert merkle and merkle[0].status == "FAIL"
    assert report.frozen()["verified"] == 6
    assert run_verify(root, hq) == 1


def test_tamper_demo_reports_detection_and_restores(tmp_path, capsys):
    root, hq = build(tmp_path, 9)
    args = ["--data-dir", str(root), "--hq-dir", str(hq), "--block", "4", "--offset", "10"]
    assert tamper_demo.main(args) == 0
    out = capsys.readouterr().out
    assert "broken here" in out and "every block after it (blocks 5..9)" in out
    assert run_verify(root, hq) == 1
    assert tamper_demo.main([*args, "--restore"]) == 0
    assert run_verify(root, hq) == 0


def test_reordering_two_blocks_fails(tmp_path):
    root, hq = build(tmp_path, 12)

    def swap(lines):
        lines[4], lines[5] = lines[5], lines[4]
        return lines

    rewrite_lines(writable(ledger_path(root)), swap)
    report = verify.verify_store(root, hq_dir=hq)
    assert not report.ok
    assert report.first_broken.n == 6  # block 6 now sits where block 5 should be
    assert all(not b.ok for b in report.blocks[4:])
    assert run_verify(root, hq) == 1


@pytest.mark.parametrize("index", [0, 9, -1], ids=["genesis", "middle", "last"])
def test_deleting_a_block_fails(tmp_path, index):
    root, hq = build(tmp_path, 12)
    rewrite_lines(writable(ledger_path(root)), lambda lines: lines[:index] + lines[index + 1 :] if index != -1 else lines[:-1])
    report = verify.verify_store(root, hq_dir=hq)
    assert not report.ok
    if index == -1:  # the tail is still a valid chain; the signed root is what catches it
        root_check = next(c for c in report.checks if c.name == "daily Merkle root")
        assert root_check.status == "FAIL" and "11" in root_check.detail
    else:
        assert not report.blocks[index].ok
    assert run_verify(root, hq) == 1


def test_editing_a_block_record_fails(tmp_path):
    root, hq = build(tmp_path, 6)

    def edit(lines):
        row = json.loads(lines[2])
        row["camera_id"] = "SNL-01"
        lines[2] = json.dumps(row, separators=(",", ":")) + "\n"
        return lines

    rewrite_lines(writable(ledger_path(root)), edit)
    report = verify.verify_store(root, hq_dir=hq)
    assert report.first_broken.n == 3 and "block record altered" in " ".join(report.first_broken.reasons)
    assert run_verify(root, hq) == 1


def test_store_refuses_to_append_on_a_broken_chain(tmp_path):
    root, hq = build(tmp_path, 5, seal=False)
    rewrite_lines(writable(ledger_path(root)), lambda lines: [lines[0], lines[2], lines[1], *lines[3:]])
    with pytest.raises(ChainCorrupt):
        EvidenceStore(root, hq_dir=hq)


# --------------------------------------------------------------------------- Merkle


def _hashes(n: int) -> list[str]:
    return [hashlib.sha256(f"block-{i}".encode()).hexdigest() for i in range(n)]


@pytest.mark.parametrize("n", [1, 3, 5, 7, 11, 13])
def test_odd_leaf_count_root_is_deterministic(n):
    hs = _hashes(n)
    assert merkle_root(hs) == merkle_root(list(hs)) == merkle_root(_hashes(n))
    # promote-not-duplicate: appending a copy of the last leaf must change the root
    assert merkle_root(hs + [hs[-1]]) != merkle_root(hs)
    assert merkle_root(hs[::-1]) != merkle_root(hs) or n == 1


def test_odd_leaf_rule_is_promote():
    a, b, c = _hashes(3)
    expected = node_hash(node_hash(leaf_hash(a), leaf_hash(b)), leaf_hash(c))
    assert merkle_root([a, b, c]) == expected.hex()
    assert merkle_root([a]) == leaf_hash(a).hex()


def test_inclusion_proof_valid_and_non_member_rejected():
    hs = _hashes(13)
    root = merkle_root(hs)
    rng = random.Random(1)
    for i in [rng.randrange(13) for _ in range(5)] + [0, 12]:
        proof = inclusion_proof(hs, i)
        assert verify_inclusion(hs[i], proof, root)
        assert not verify_inclusion(hashlib.sha256(b"not a block").hexdigest(), proof, root)
        assert not verify_inclusion(hs[(i + 1) % 13], proof, root)
        bad = type(proof)(proof.index, proof.leaf_count, tuple(reversed(proof.siblings)) if len(proof.siblings) > 1
                          else (hashlib.sha256(b"x").hexdigest(),))
        assert not verify_inclusion(hs[i], bad, root)
        wrong_size = type(proof)(proof.index, proof.leaf_count + 1, proof.siblings)
        assert not verify_inclusion(hs[i], wrong_size, root, expected_leaf_count=13)
        assert verify_inclusion(hs[i], proof, root, expected_leaf_count=13)


def test_inclusion_proof_against_the_signed_root(tmp_path):
    root, hq = build(tmp_path, 9)
    info = proof_for_block(root, 5)
    signed = info["signed_root"]
    assert verify_inclusion(info["block"].block_hash, info["proof"], signed.manifest.merkle_root, signed.manifest.leaf_count)
    assert not verify_inclusion(hashlib.sha256(b"forged").hexdigest(), info["proof"], signed.manifest.merkle_root)


# --------------------------------------------------------------------------- signatures, keys, HQ


def test_private_key_file_is_0600_and_never_serialised(tmp_path):
    root, _ = build(tmp_path, 2, seal=False)
    key = root / "keys" / PRIVATE_KEY_NAME
    assert stat.S_IMODE(os.stat(key).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(root / "keys").st_mode) == 0o700
    signer = Signer.load_or_create(root / "keys")
    assert "PRIVATE" not in repr(signer) and signer.key_id in repr(signer)
    with pytest.raises(TypeError):
        pickle.dumps(signer)
    os.chmod(key, 0o644)
    with pytest.raises(InsecureKeyFile):
        Signer.load_or_create(root / "keys")


def test_tampered_signature_fails(tmp_path):
    root, hq = build(tmp_path, 8)
    path = writable(root / "roots" / f"{DAY}.json")
    doc = json.loads(path.read_text())
    sig = bytearray(bytes.fromhex(doc["signature"]))
    sig[0] ^= 0x01
    doc["signature"] = sig.hex()
    path.write_text(json.dumps(doc))
    report = verify.verify_store(root, hq_dir=hq)
    statuses = {c.name: c.status for c in report.checks if c.scope == DAY}
    assert statuses["Ed25519 signature"] == "FAIL"
    assert statuses["HQ mirror"] == "FAIL"
    assert run_verify(root, hq) == 1


def test_resigning_with_a_new_box_key_is_caught_by_hq(tmp_path):
    """An insider rewrites a clip, rebuilds the chain and re-signs with a fresh key on the box."""
    root, hq = build(tmp_path, 4)
    for f in (root / "keys").iterdir():
        f.unlink()
    rogue = Signer.load_or_create(root / "keys")
    path = writable(root / "roots" / f"{DAY}.json")
    doc = json.loads(path.read_text())
    doc["manifest"]["key_id"] = rogue.key_id
    from evidence.signer import DayManifest

    m = DayManifest.from_dict(doc["manifest"])
    doc["signature"] = rogue.sign(m.canonical())
    doc["public_key"] = rogue.public_raw.hex()
    path.write_text(json.dumps(doc))
    report = verify.verify_store(root, hq_dir=hq)
    failed = {c.name for c in report.checks if c.status == "FAIL"}
    assert {"Ed25519 signature", "HQ public key", "HQ mirror"} <= failed
    assert run_verify(root, hq) == 1


def test_missing_hq_mirror_copy_fails(tmp_path):
    root, hq = build(tmp_path, 4)
    (hq / "post-test" / "roots" / f"{DAY}.json").unlink()
    assert run_verify(root, hq) == 1


# --------------------------------------------------------------------------- gaps


POWER_LOSS_SCRIPT = textwrap.dedent(
    """
    import os, signal, sys
    from evidence.store import EvidenceStore
    from evidence.chain import ledger_path
    store = EvidenceStore(sys.argv[1], hq_dir=sys.argv[2], device_id="post-test")
    for i in range(5):
        w = store.open_clip("RXL-01", "2026-01-01T00:00:00.000Z")
        w.write_frame(os.urandom(256))
        w.seal("2026-01-01T00:00:10.000Z")
    w = store.open_clip("JGB-03", "2026-01-01T00:01:00.000Z")   # being written when power drops
    for _ in range(4):
        w.write_frame(os.urandom(256))
    with open(ledger_path(sys.argv[1]), "ab") as fh:          # a block line half-way to disk
        fh.write(b'{"n":6,"clip_id":"clip-0000')
        fh.flush(); os.fsync(fh.fileno())
    os.kill(os.getpid(), signal.SIGKILL)
    """
)


def test_power_loss_mid_write_resumes_and_logs_exactly_one_gap(tmp_path):
    root, hq = tmp_path / "ev", tmp_path / "hq"
    proc = subprocess.run([sys.executable, "-c", POWER_LOSS_SCRIPT, str(root), str(hq)], cwd=EDGE_DIR)
    assert proc.returncode == -signal.SIGKILL
    assert GapLog(root / "gaps.jsonl").gaps() == []

    store = EvidenceStore(root, hq_dir=hq)
    gap = store.last_gap
    assert gap is not None and gap.kind == KIND_POWER_LOSS and gap.last_block_n == 5
    assert gap.start <= gap.end and "torn ledger tail" in gap.reason and "1 unsealed partial clip" in gap.reason
    quarantined = sorted(p.name for p in (root / "quarantine").iterdir())
    assert any(n.endswith("torn-ledger-tail.bin") for n in quarantined) and any(n.endswith(".mjpeg") for n in quarantined)
    assert not any((root / "clips" / "pending").iterdir())

    for _ in range(3):  # the chain resumes from block 5
        w = store.open_clip("PNT-07", "2026-01-01T01:00:00.000Z")
        w.write_frame(os.urandom(128))
        block = w.seal("2026-01-01T01:00:05.000Z")
    assert block.n == 8
    store.close()

    gaps = GapLog(root / "gaps.jsonl").gaps()
    assert len(gaps) == 1
    report = verify.verify_store(root, hq_dir=hq)
    assert report.ok, verify.render(report)
    assert [b.n for b in report.blocks] == list(range(1, 9))
    assert len(report.gaps) == 1
    assert run_verify(root, hq) == 0


def test_clean_restart_logs_one_restart_gap(tmp_path):
    root, hq = build(tmp_path, 3, seal=False)
    store = EvidenceStore(root, hq_dir=hq)
    assert store.last_gap.kind == KIND_RESTART
    store.close()
    assert len(GapLog(root / "gaps.jsonl").gaps()) == 1


def test_deleting_a_logged_gap_fails(tmp_path):
    root, hq = build(tmp_path, 3, seal=False)
    clock = Clock(datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc))
    store = EvidenceStore(root, hq_dir=hq, clock=clock)  # clean restart -> one gap
    clock.t += timedelta(days=2)
    store.seal_past_days()  # the signed manifest now carries the gap head
    store.close()
    gaps_file = root / "gaps.jsonl"
    rewrite_lines(writable(gaps_file), lambda lines: lines[:-1])
    report = verify.verify_store(root, hq_dir=hq)
    root_check = next(c for c in report.checks if c.name == "daily Merkle root" and c.scope == DAY)
    assert root_check.status == "FAIL" and "gap log" in root_check.detail
    assert run_verify(root, hq) == 1


# --------------------------------------------------------------------------- CLI + certificate


def test_verify_cli_process_exit_codes(tmp_path):
    root, hq = build(tmp_path, 5)
    cmd = [sys.executable, "-m", "evidence.verify", "--data-dir", str(root), "--hq-dir", str(hq)]
    ok = subprocess.run([*cmd, "--from", "2026-09-01", "--to", "2026-09-30"], cwd=EDGE_DIR, capture_output=True, text=True)
    assert ok.returncode == 0 and "RESULT: PASS" in ok.stdout
    target = Ledger(ledger_path(root)).blocks()[2]
    tamper_demo.flip_byte(root / target.clip_path, 0)
    bad = subprocess.run(cmd, cwd=EDGE_DIR, capture_output=True, text=True)
    assert bad.returncode == 1 and "first broken block 3" in bad.stdout
    js = subprocess.run([*cmd, "--json"], cwd=EDGE_DIR, capture_output=True, text=True)
    assert js.returncode == 1 and json.loads(js.stdout)["intact"] is False
    missing = subprocess.run([*cmd[:3], "--data-dir", str(tmp_path / "nope")], cwd=EDGE_DIR, capture_output=True)
    assert missing.returncode == 2


def test_section63_certificate(tmp_path):
    root, hq = build(tmp_path, 5)
    text, ok = certificate.build_certificate(
        root, block_ns=[3], hq_dir=hq,
        signatories={"partA_name": "Officer A", "partB_name": "Examiner B", "place": "Raxaul"},
    )
    assert ok
    block = Ledger(ledger_path(root)).blocks()[2]
    info = proof_for_block(root, 3)
    assert "must be reviewed by a qualified person before any real use" in text
    assert "SECTION 63" in text
    assert block.clip_sha256 in text and block.block_hash in text
    assert info["signed_root"].signature in text
    assert all(s in text for s in info["proof"].siblings)
    assert "Officer A" in text and "Examiner B" in text
    assert not re.search(r"\$(record_table|hash_table|proof_section|partA_name|device_id|gap_table)\b", text)


# --------------------------------------------------------------------------- air gap


def test_no_network_imports_under_evidence():
    pattern = re.compile(r"requests|urllib|httpx|socket|http\.client")
    offenders = [
        f"{p.relative_to(EDGE_DIR)}:{i}"
        for p in sorted(EVIDENCE_DIR.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".md"}
        for i, line in enumerate(p.read_text("utf-8").splitlines(), start=1)
        if pattern.search(line)
    ]
    assert offenders == []
