"""Append-only hash chain of sealed clips, stored as a JSON Lines ledger.

CANONICAL BLOCK SERIALISATION (version 1 — fixed forever; a change is a new domain tag, never an edit)

    block_hash = SHA-256( canonical_bytes ), written as 64 lowercase hex characters

    canonical_bytes = ASCII bytes of
        "truewatch.block.v1" LF
        "n="               <n>               LF
        "clip_id="         <clip_id>         LF
        "camera_id="       <camera_id>       LF
        "started_at="      <started_at>      LF
        "ended_at="        <ended_at>        LF
        "sealed_at="       <sealed_at>       LF
        "clip_sha256="     <clip_sha256>     LF
        "prev_block_hash=" <prev_block_hash> LF

- Field order is exactly the order above. LF is the single byte 0x0A; every line, including the
  last, ends with it. There is no BOM, no whitespace around '=', and no other separator.
- Encoding is ASCII (so identical in UTF-8). Every value is validated before hashing and must
  match its grammar, which excludes LF and every non-printable byte, so the framing cannot be
  forged by a value:
    n                decimal integer >= 1, no sign, no leading zeros
    clip_id          [A-Za-z0-9._:-]{1,128}
    camera_id        [A-Za-z0-9._:-]{1,64}
    *_at             YYYY-MM-DDTHH:MM:SS.mmmZ  (UTC, exactly millisecond precision)
    clip_sha256      64 lowercase hex (see hasher.py for how it is computed)
    prev_block_hash  64 lowercase hex; the genesis block (n = 1) uses 64 '0' characters
- `sealed_at` is set by the store's clock when the block is appended and never decreases along
  the chain; it decides which day (IST, UTC+05:30, a fixed offset with no DST) a block's Merkle
  root belongs to.

The same framing (domain line, then name=value lines in a fixed order, every value printable
ASCII 0x20-0x7E) is reused for gap records and daily manifests via `canonical_bytes`.

STORAGE: JSON Lines, one block per line, appended with O_APPEND and fsync'd before the seal call
returns. Chosen over SQLite because the ledger is append-only by nature and a flat file makes
that property literal: there is no UPDATE path to guard, a torn write after power loss can only
ever be the final partial line (detected and quarantined on restart), and an examiner can read
and re-hash it with nothing but a text editor and `sha256sum`. One post has one writer and a
bounded number of clips per day, so a linear scan is cheap. The JSON is only a container; the
hash is never computed over JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

BLOCK_DOMAIN = "truewatch.block.v1"
BLOCK_FIELDS = (
    "n",
    "clip_id",
    "camera_id",
    "started_at",
    "ended_at",
    "sealed_at",
    "clip_sha256",
    "prev_block_hash",
)
GENESIS_PREV = "0" * 64
LEDGER_NAME = "chain.jsonl"

# Days are IST calendar days. IST has had a fixed +05:30 offset with no DST since 1945, so a fixed
# offset is exact and needs no timezone database.
DAY_TZ = timezone(timedelta(hours=5, minutes=30), "IST")
DAY_TZ_LABEL = "+05:30"

_RE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_RE_CAMERA = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_RE_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
_RE_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RE_N = re.compile(r"^[1-9][0-9]*$")
_RE_PRINTABLE = re.compile(r"^[\x20-\x7e]*$")
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


class CanonicalError(ValueError):
    """A value does not fit the canonical grammar, so it cannot be hashed unambiguously."""


# --------------------------------------------------------------------------- canonical form


def canonical_bytes(domain: str, pairs: Iterable[tuple[str, object]]) -> bytes:
    """domain LF, then name=value LF for each pair in the given order. ASCII, printable values only."""
    lines = [domain]
    for name, value in pairs:
        text = str(value)
        if not _RE_PRINTABLE.match(text):
            raise CanonicalError(f"{name}: value must be printable ASCII without line breaks")
        lines.append(f"{name}={text}")
    return ("\n".join(lines) + "\n").encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check(name: str, value: str, pattern: re.Pattern) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise CanonicalError(f"{name}={value!r} does not match the canonical grammar")
    return value


# --------------------------------------------------------------------------- time helpers


def to_rfc3339(value: datetime | float | int | str) -> str:
    """Normalise a timestamp to the canonical 'YYYY-MM-DDTHH:MM:SS.mmmZ' UTC form.

    Accepts an aware datetime, a naive datetime (taken as UTC), POSIX seconds, or an ISO-8601 string.
    """
    if isinstance(value, str):
        text = value.strip()
        if _RE_TS.match(text):
            return text
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        raise CanonicalError(f"unsupported timestamp type {type(value).__name__}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def day_of(ts: str) -> str:
    """The IST calendar day ('YYYY-MM-DD') a canonical UTC timestamp falls on."""
    return parse_ts(ts).astimezone(DAY_TZ).strftime("%Y-%m-%d")


def time_label(ts: str) -> str:
    """The frozen display form '14 AUG · 01:12' (IST), independent of the process locale."""
    local = parse_ts(ts).astimezone(DAY_TZ)
    return f"{local.day:02d} {_MONTHS[local.month - 1]} · {local.hour:02d}:{local.minute:02d}"


def short_hash(h: str) -> str:
    """The frozen display form '<10hex>…<6hex>' the console already renders."""
    return f"{h[:10]}…{h[-6:]}"


# --------------------------------------------------------------------------- block


@dataclass(frozen=True)
class Block:
    n: int
    clip_id: str
    camera_id: str
    started_at: str
    ended_at: str
    sealed_at: str
    clip_sha256: str
    prev_block_hash: str
    block_hash: str = ""
    # Stored next to the block, not inside the canonical form. frame_count and byte_length are
    # bound by clip_sha256 (hasher.py); clip_path only says where to find the bytes.
    clip_path: str = ""
    frame_count: int = 0
    byte_length: int = 0

    def canonical(self) -> bytes:
        return canonical_block_bytes(self)

    def compute_hash(self) -> str:
        return sha256_hex(self.canonical())

    @property
    def day(self) -> str:
        return day_of(self.sealed_at)

    def to_row(self) -> dict:
        row = {name: getattr(self, name) for name in BLOCK_FIELDS}
        row["block_hash"] = self.block_hash
        row["clip_path"] = self.clip_path
        row["frame_count"] = self.frame_count
        row["byte_length"] = self.byte_length
        return row

    @classmethod
    def from_row(cls, row: dict) -> "Block":
        return cls(
            n=int(row["n"]),
            clip_id=str(row["clip_id"]),
            camera_id=str(row["camera_id"]),
            started_at=str(row["started_at"]),
            ended_at=str(row["ended_at"]),
            sealed_at=str(row["sealed_at"]),
            clip_sha256=str(row["clip_sha256"]),
            prev_block_hash=str(row["prev_block_hash"]),
            block_hash=str(row.get("block_hash", "")),
            clip_path=str(row.get("clip_path", "")),
            frame_count=int(row.get("frame_count", 0)),
            byte_length=int(row.get("byte_length", 0)),
        )

    @property
    def evidence_id(self) -> str:
        return f"ev_{self.block_hash[:20]}"

    def evidence_dict(self) -> dict:
        """The `evidence` block of a truewatch.event.v1 (ARCHITECTURE_V2 §4.2), minus the backend's URLs."""
        return {
            "evidence_id": self.evidence_id,
            "block_n": self.n,
            "clip_sha256": self.clip_sha256,
            "prev_sha256": self.prev_block_hash,
            "sealed_at": self.sealed_at,
        }

    def chain_entry(self, include_full_hash: bool = True) -> dict:
        """The FROZEN `GET /api/evidence/chain` item `{ n, hash, camera, time }` (+ optional `full_hash`)."""
        entry = {
            "n": self.n,
            "hash": short_hash(self.block_hash),
            "camera": self.camera_id,
            "time": time_label(self.started_at),
        }
        if include_full_hash:
            entry["full_hash"] = self.block_hash
        return entry


def canonical_block_bytes(b: Block) -> bytes:
    if isinstance(b.n, bool) or not isinstance(b.n, int):
        raise CanonicalError("n must be an integer")
    _check("n", str(b.n), _RE_N)
    _check("clip_id", b.clip_id, _RE_ID)
    _check("camera_id", b.camera_id, _RE_CAMERA)
    for name in ("started_at", "ended_at", "sealed_at"):
        _check(name, getattr(b, name), _RE_TS)
    _check("clip_sha256", b.clip_sha256, _RE_HEX64)
    _check("prev_block_hash", b.prev_block_hash, _RE_HEX64)
    return canonical_bytes(BLOCK_DOMAIN, ((name, getattr(b, name)) for name in BLOCK_FIELDS))


def make_block(
    *,
    n: int,
    clip_id: str,
    camera_id: str,
    started_at,
    ended_at,
    sealed_at,
    clip_sha256: str,
    prev_block_hash: str,
    clip_path: str = "",
    frame_count: int = 0,
    byte_length: int = 0,
) -> Block:
    """Build a block and fill in its block_hash from the canonical form."""
    b = Block(
        n=n,
        clip_id=clip_id,
        camera_id=camera_id,
        started_at=to_rfc3339(started_at),
        ended_at=to_rfc3339(ended_at),
        sealed_at=to_rfc3339(sealed_at),
        clip_sha256=clip_sha256,
        prev_block_hash=prev_block_hash,
        clip_path=clip_path,
        frame_count=frame_count,
        byte_length=byte_length,
    )
    return replace(b, block_hash=b.compute_hash())


# --------------------------------------------------------------------------- ledger


@dataclass
class LedgerRead:
    blocks: list[Block]
    problems: list[str]  # malformed lines (line number + reason); any entry means the ledger is not clean
    torn_tail: bytes  # bytes after the last LF (a write interrupted by power loss), b"" when clean


class Ledger:
    """The JSON Lines file that holds the chain. Appends only; there is no update or delete method."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def read(self) -> LedgerRead:
        if not self.path.exists():
            return LedgerRead([], [], b"")
        raw = self.path.read_bytes()
        cut = raw.rfind(b"\n") + 1
        body, tail = raw[:cut], raw[cut:]
        blocks: list[Block] = []
        problems: list[str] = []
        for lineno, line in enumerate(body.split(b"\n")[:-1], start=1):
            try:
                blocks.append(Block.from_row(json.loads(line.decode("utf-8"))))
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
                problems.append(f"ledger line {lineno} is not a valid block record ({type(exc).__name__})")
        return LedgerRead(blocks, problems, tail)

    def blocks(self) -> list[Block]:
        return self.read().blocks

    def append(self, block: Block) -> None:
        line = json.dumps(block.to_row(), separators=(",", ":"), ensure_ascii=True) + "\n"
        data = line.encode("ascii")
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
        try:
            written = 0
            while written < len(data):
                written += os.write(fd, data[written:])
            os.fsync(fd)
        finally:
            os.close(fd)

    def truncate_torn_tail(self) -> bytes:
        """Remove a partial final line (never acknowledged, so never part of the chain). Returns it."""
        raw = self.path.read_bytes() if self.path.exists() else b""
        cut = raw.rfind(b"\n") + 1
        tail = raw[cut:]
        if tail:
            with open(self.path, "r+b") as fh:
                fh.truncate(cut)
                fh.flush()
                os.fsync(fh.fileno())
        return tail


def ledger_path(root: str | os.PathLike) -> Path:
    return Path(root) / LEDGER_NAME


def chain_view(root: str | os.PathLike, limit: int | None = None, include_full_hash: bool = True) -> list[dict]:
    """FROZEN `GET /api/evidence/chain` payload: `[{ n, hash, camera, time }]`, oldest first.

    `limit` keeps only the newest `limit` blocks (the console shows the tail of the chain).
    `full_hash` is the one additive optional field; pass include_full_hash=False for the bare shape.
    """
    blocks = Ledger(ledger_path(root)).blocks()
    if limit is not None:
        blocks = blocks[-limit:] if limit > 0 else []
    return [b.chain_entry(include_full_hash=include_full_hash) for b in blocks]
