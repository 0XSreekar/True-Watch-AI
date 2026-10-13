"""Chain gaps: a restart or power loss becomes one first-class, hash-chained record — never a silent hole.

When the board loses power the camera keeps recording on its own (slide 4), but nothing is sealed
until the board is back. On every start the store runs `recover_after_restart`, which:

1. reads the run marker the store keeps (`state/run.json`: "running" + last-alive time while the
   store is open, "stopped" after a clean close);
2. removes a torn final ledger line (a block write that never completed and was never acknowledged)
   and moves it to `quarantine/`, never deleting it;
3. moves clips that were still being written (`clips/pending/`) to `quarantine/`;
4. appends EXACTLY ONE gap record covering [last moment the store was known alive, now], with a
   kind (`power_loss` or `restart`) and a reason that says what was recovered.

Gap records are chained like blocks (framing as in chain.py):

    gap_hash = SHA-256( "truewatch.gap.v1" LF  g= LF  kind= LF  start= LF  end= LF
                        last_block_n= LF  reason= LF  prev_gap_hash= LF )

`g` counts from 1, `prev_gap_hash` of the first gap is 64 '0's, and every signed daily manifest
carries the gap count and the latest gap_hash, so a logged gap cannot later be deleted or edited
without breaking a signature. A gap is not a verification failure — it is disclosed. A missing or
altered gap record is.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from evidence.chain import (
    GENESIS_PREV,
    Ledger,
    canonical_bytes,
    parse_ts,
    sha256_hex,
    to_rfc3339,
)
from evidence.hasher import fsync_dir

GAP_DOMAIN = "truewatch.gap.v1"
GAP_FIELDS = ("g", "kind", "start", "end", "last_block_n", "reason", "prev_gap_hash")
GAPS_NAME = "gaps.jsonl"
KIND_POWER_LOSS = "power_loss"
KIND_RESTART = "restart"
_RE_REASON = re.compile(r"^[\x20-\x7e]{1,512}$")


@dataclass(frozen=True)
class Gap:
    g: int
    kind: str
    start: str
    end: str
    last_block_n: int
    reason: str
    prev_gap_hash: str
    gap_hash: str = ""

    def canonical(self) -> bytes:
        if self.kind not in (KIND_POWER_LOSS, KIND_RESTART):
            raise ValueError(f"unknown gap kind {self.kind!r}")
        if not _RE_REASON.match(self.reason):
            raise ValueError("gap reason must be 1..512 printable ASCII characters")
        return canonical_bytes(GAP_DOMAIN, ((f, getattr(self, f)) for f in GAP_FIELDS))

    def compute_hash(self) -> str:
        return sha256_hex(self.canonical())

    @property
    def duration_s(self) -> float:
        return (parse_ts(self.end) - parse_ts(self.start)).total_seconds()

    def to_row(self) -> dict:
        row = {f: getattr(self, f) for f in GAP_FIELDS}
        row["gap_hash"] = self.gap_hash
        return row

    @classmethod
    def from_row(cls, row: dict) -> "Gap":
        return cls(
            g=int(row["g"]),
            kind=str(row["kind"]),
            start=str(row["start"]),
            end=str(row["end"]),
            last_block_n=int(row["last_block_n"]),
            reason=str(row["reason"]),
            prev_gap_hash=str(row["prev_gap_hash"]),
            gap_hash=str(row.get("gap_hash", "")),
        )


class GapLog:
    """Append-only JSON Lines file of Gap records (same durability rules as the block ledger)."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self._ledger = Ledger(self.path)  # reuse the torn-tail handling; rows are parsed here

    def read(self) -> tuple[list[Gap], list[str]]:
        if not self.path.exists():
            return [], []
        raw = self.path.read_bytes()
        body = raw[: raw.rfind(b"\n") + 1]
        gaps: list[Gap] = []
        problems: list[str] = []
        for lineno, line in enumerate(body.split(b"\n")[:-1], start=1):
            try:
                gaps.append(Gap.from_row(json.loads(line.decode("utf-8"))))
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
                problems.append(f"gap log line {lineno} is not a valid gap record ({type(exc).__name__})")
        if raw[len(body):]:
            problems.append("gap log ends in a partial line")
        return gaps, problems

    def gaps(self) -> list[Gap]:
        return self.read()[0]

    def truncate_torn_tail(self) -> bytes:
        return self._ledger.truncate_torn_tail()

    def append(self, kind: str, start, end, last_block_n: int, reason: str) -> Gap:
        existing = self.gaps()
        prev = existing[-1].gap_hash if existing else GENESIS_PREV
        gap = Gap(
            g=(existing[-1].g + 1) if existing else 1,
            kind=kind,
            start=to_rfc3339(start),
            end=to_rfc3339(end),
            last_block_n=int(last_block_n),
            reason=reason,
            prev_gap_hash=prev,
        )
        gap = replace(gap, gap_hash=gap.compute_hash())
        line = (json.dumps(gap.to_row(), separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
        try:
            written = 0
            while written < len(line):
                written += os.write(fd, line[written:])
            os.fsync(fd)
        finally:
            os.close(fd)
        return gap


def check_gap_chain(gaps: list[Gap]) -> list[str]:
    """Problems with the gap log's own integrity (sequence, hashes, links). Empty list = intact."""
    problems: list[str] = []
    prev = GENESIS_PREV
    for i, gap in enumerate(gaps, start=1):
        if gap.g != i:
            problems.append(f"gap #{i}: sequence number is {gap.g}")
        try:
            recomputed = gap.compute_hash()
        except ValueError as exc:
            problems.append(f"gap #{gap.g}: {exc}")
            recomputed = ""
        if recomputed != gap.gap_hash:
            problems.append(f"gap #{gap.g}: record hash does not recompute (record altered)")
        if gap.prev_gap_hash != prev:
            problems.append(f"gap #{gap.g}: prev link broken (a gap record was removed or reordered)")
        prev = gap.gap_hash
    return problems


def gaps_head(gaps: list[Gap], count: int | None = None) -> str:
    """Hash of the `count`-th gap (default: the last one), or 64 '0's when there are none."""
    count = len(gaps) if count is None else count
    return gaps[count - 1].gap_hash if count > 0 else GENESIS_PREV


# --------------------------------------------------------------------------- run marker


class RunMarker:
    """`state/run.json`: tells the next start whether the previous run ended cleanly."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)

    def read(self) -> dict | None:
        try:
            return json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            return None
        except (ValueError, UnicodeDecodeError):
            return {"state": "unreadable"}

    def _write(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        fsync_dir(self.path.parent)

    def mark_running(self, now: str) -> None:
        self._write({"state": "running", "since": now, "last_alive": now})

    def touch(self, now: str) -> None:
        state = self.read() or {}
        self._write({"state": "running", "since": state.get("since", now), "last_alive": now})

    def mark_stopped(self, now: str) -> None:
        self._write({"state": "stopped", "stopped_at": now})


# --------------------------------------------------------------------------- recovery


def _quarantine(src: Path, quarantine_dir: Path, stamp: str) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dst = quarantine_dir / f"{stamp}-{src.name}"
    shutil.move(str(src), dst)
    return dst


def recover_after_restart(
    *,
    ledger: Ledger,
    gap_log: GapLog,
    marker: RunMarker,
    pending_dir: Path,
    quarantine_dir: Path,
    clock: Callable[[], datetime],
) -> Gap | None:
    """Run once when the store opens. Returns the single gap it logged, or None on first boot."""
    now = to_rfc3339(clock())
    stamp = now.replace(":", "").replace("-", "").replace(".", "")
    state = marker.read()

    notes: list[str] = []
    torn = ledger.truncate_torn_tail()
    if torn:
        (quarantine_dir).mkdir(parents=True, exist_ok=True)
        (quarantine_dir / f"{stamp}-torn-ledger-tail.bin").write_bytes(torn)
        notes.append(f"torn ledger tail of {len(torn)} bytes quarantined")
    torn_gap = gap_log.truncate_torn_tail()
    if torn_gap:
        (quarantine_dir / f"{stamp}-torn-gap-tail.bin").write_bytes(torn_gap)
        notes.append(f"torn gap-log tail of {len(torn_gap)} bytes quarantined")
    partial = sorted(p for p in pending_dir.glob("*") if p.is_file()) if pending_dir.is_dir() else []
    for p in partial:
        _quarantine(p, quarantine_dir, stamp)
    if partial:
        notes.append(f"{len(partial)} unsealed partial clip(s) quarantined")

    blocks = ledger.blocks()
    last_n = blocks[-1].n if blocks else 0
    last_sealed = blocks[-1].sealed_at if blocks else None

    if state is None and not blocks and not notes:
        return None  # genuine first boot: nothing existed before, so nothing is missing

    if state is not None and state.get("state") == "stopped" and not notes:
        kind = KIND_RESTART
        start = state.get("stopped_at") or last_sealed or now
        reason = "service restarted after a clean shutdown; no clips were sealed while it was stopped"
    else:
        kind = KIND_POWER_LOSS
        candidates = [t for t in ((state or {}).get("last_alive"), last_sealed) if t]
        start = max(candidates, key=lambda t: parse_ts(to_rfc3339(t))) if candidates else now
        if state is None:
            lead = "run marker missing; previous run ended without a clean shutdown"
        else:
            lead = "unclean shutdown (power loss or crash)"
        reason = "; ".join([lead, *notes])
    start = to_rfc3339(start)
    if parse_ts(start) > parse_ts(now):
        start = now  # a clock that moved backwards must not produce a negative gap
    return gap_log.append(kind, start, now, last_n, reason[:512])
