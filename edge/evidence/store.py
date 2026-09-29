"""EvidenceStore — the one object the pipeline talks to when it seals a clip.

    store = EvidenceStore("./var/evidence")                 # recovers, logs a gap if needed
    w = store.open_clip("RXL-01", started_at)                # hash-at-capture writer
    w.write_frame(jpeg_bytes) ...                            # every byte hashed as it is written
    block = w.seal(ended_at)                                 # fsync, move into place, append block

    block = store.seal_clip(path, "RXL-01", started_at, ended_at)   # a file another writer produced

    block.evidence_dict()    # the `evidence` object for truewatch.event.v1
    block.chain_entry()      # the FROZEN { n, hash, camera, time } item (+ full_hash)
    store.close()            # clean shutdown (the next start logs a 'restart' gap, not 'power_loss')

Directory layout under the store root:

    store.json            format tag and device_id (written once)
    chain.jsonl           the block ledger (chain.py)
    gaps.jsonl            the gap log (gaps.py)
    roots/<day>.json      signed daily manifests (signer.py), written once per day
    keys/                 signing_key.pem (0600) and signing_key.pub.pem, directory 0700
    clips/<day>/          sealed clips, made read-only (0444) at seal time
    clips/pending/        clips being written right now
    quarantine/           torn ledger tails and partial clips recovered after a power loss
    state/run.json        the run marker (gaps.py)
    .lock                 exclusive flock: one writer per store

Days roll over on their own: the first block of a new day seals every earlier unsealed day, signs
it and mirrors it to HQ. The store also seals past days when it opens. A day is never re-sealed.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from evidence.chain import (
    DAY_TZ,
    DAY_TZ_LABEL,
    GENESIS_PREV,
    Block,
    Ledger,
    day_of,
    ledger_path,
    make_block,
    parse_ts,
    to_rfc3339,
)
from evidence.gaps import GAPS_NAME, Gap, GapLog, RunMarker, gaps_head, recover_after_restart
from evidence.hasher import ClipDigest, ClipHasher, HashingWriter, fsync_dir, hash_file
from evidence.merkle import InclusionProof, inclusion_proof, merkle_root
from evidence.mirror import HQMirror
from evidence.signer import DayManifest, SignedRoot, Signer, sign_manifest

FORMAT = "truewatch.evidence.v1"


class StoreLocked(RuntimeError):
    """Another process already holds this store open for writing."""


class ChainCorrupt(RuntimeError):
    """The ledger on disk does not verify; the store refuses to append on top of it."""


class DaySealed(RuntimeError):
    """A block would land in a day whose root is already signed."""


def default_hq_dir(root: str | os.PathLike) -> Path:
    root = Path(root).resolve()
    return root.parent / f"{root.name}-hq"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def today(clock: Callable[[], datetime] = utc_now) -> str:
    return clock().astimezone(DAY_TZ).strftime("%Y-%m-%d")


def read_meta(root: str | os.PathLike) -> dict:
    return json.loads((Path(root) / "store.json").read_text("utf-8"))


def roots_dir(root: str | os.PathLike) -> Path:
    return Path(root) / "roots"


def load_signed_root(root: str | os.PathLike, day: str) -> SignedRoot | None:
    path = roots_dir(root) / f"{day}.json"
    if not path.exists():
        return None
    return SignedRoot.from_dict(json.loads(path.read_text("utf-8")))


def resolve_clip(root: str | os.PathLike, block: Block) -> Path:
    p = Path(block.clip_path)
    return p if p.is_absolute() else Path(root) / p


def proof_for_block(root: str | os.PathLike, n: int) -> dict:
    """Everything needed to show block `n` is in its day's signed root (used by certificate.py)."""
    blocks = Ledger(ledger_path(root)).blocks()
    by_n = {b.n: b for b in blocks}
    if n not in by_n:
        raise KeyError(f"block {n} is not in the ledger")
    block = by_n[n]
    day_blocks = [b for b in blocks if b.day == block.day]
    hashes = [b.block_hash for b in day_blocks]
    proof = inclusion_proof(hashes, day_blocks.index(block))
    return {
        "block": block,
        "day": block.day,
        "proof": proof,
        "day_root": merkle_root(hashes),
        "signed_root": load_signed_root(root, block.day),
    }


class ClipWriter:
    """Returned by `EvidenceStore.open_clip`. Hashes while writing; `seal()` appends the block."""

    def __init__(self, store: "EvidenceStore", camera_id: str, started_at: str, suffix: str) -> None:
        self._store = store
        self.camera_id = camera_id
        self.started_at = started_at
        self.suffix = suffix
        self.path = store.pending_dir / f"{uuid.uuid4().hex}{suffix}"
        self._writer = HashingWriter(self.path)
        self.block: Block | None = None

    def write(self, data) -> int:
        return self._writer.write(data)

    def write_frame(self, data) -> int:
        return self._writer.write_frame(data)

    @property
    def frame_count(self) -> int:
        return self._writer.frame_count

    def seal(self, ended_at, clip_id: str | None = None) -> Block:
        if self.block is not None:
            return self.block
        digest = self._writer.close()
        self.block = self._store._append_clip(
            src=self.path,
            digest=digest,
            camera_id=self.camera_id,
            started_at=self.started_at,
            ended_at=to_rfc3339(ended_at),
            clip_id=clip_id,
            suffix=self.suffix,
            move=True,
        )
        return self.block

    def abort(self) -> None:
        """Stop without sealing. The partial file goes to quarantine, it is not deleted."""
        self._writer.abort()
        if self.path.exists():
            self._store.quarantine_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.path), self._store.quarantine_dir / f"aborted-{self.path.name}")


class EvidenceStore:
    def __init__(
        self,
        root: str | os.PathLike,
        hq_dir: str | os.PathLike | None = None,
        device_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock = clock or utc_now
        self._mutex = threading.RLock()
        self._lock_fd = os.open(self.root / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._lock_fd)
            raise StoreLocked(f"{self.root} is already open for writing by another process") from None
        try:
            self._open(hq_dir, device_id)
        except BaseException:
            os.close(self._lock_fd)
            self._lock_fd = -1
            raise

    # -- open / close ----------------------------------------------------------------------------

    def _open(self, hq_dir, device_id) -> None:
        self.pending_dir = self.root / "clips" / "pending"
        self.quarantine_dir = self.root / "quarantine"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        roots_dir(self.root).mkdir(exist_ok=True)

        self.signer = Signer.load_or_create(self.root / "keys")
        meta_path = self.root / "store.json"
        if meta_path.exists():
            self.meta = read_meta(self.root)
            if device_id and device_id != self.meta["device_id"]:
                raise ValueError(f"store belongs to device {self.meta['device_id']!r}, not {device_id!r}")
        else:
            self.meta = {
                "format": FORMAT,
                "device_id": device_id or os.environ.get("TRUEWATCH_DEVICE_ID") or f"post-{self.signer.key_id[:8]}",
                "created_at": self._now(),
                "day_tz": DAY_TZ_LABEL,
            }
            meta_path.write_text(json.dumps(self.meta, indent=2) + "\n", "utf-8")
        self.device_id = self.meta["device_id"]

        self.mirror = HQMirror(hq_dir or default_hq_dir(self.root), self.device_id)
        self.mirror.register_key(self.signer.public_pem())

        self.ledger = Ledger(ledger_path(self.root))
        self.gap_log = GapLog(self.root / GAPS_NAME)
        self.marker = RunMarker(self.root / "state" / "run.json")

        self.last_gap: Gap | None = recover_after_restart(
            ledger=self.ledger,
            gap_log=self.gap_log,
            marker=self.marker,
            pending_dir=self.pending_dir,
            quarantine_dir=self.quarantine_dir,
            clock=self.clock,
        )
        self._blocks = self._load_checked()
        self.marker.mark_running(self._now())
        self.seal_past_days()

    def _load_checked(self) -> list[Block]:
        read = self.ledger.read()
        if read.problems:
            raise ChainCorrupt("; ".join(read.problems) + " — run `python -m evidence.verify`")
        prev = GENESIS_PREV
        for i, b in enumerate(read.blocks, start=1):
            if b.n != i or b.prev_block_hash != prev or b.compute_hash() != b.block_hash:
                raise ChainCorrupt(f"block {b.n} does not verify — run `python -m evidence.verify`")
            prev = b.block_hash
        return read.blocks

    def close(self) -> None:
        with self._mutex:
            if self._lock_fd < 0:
                return
            self.marker.mark_stopped(self._now())
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = -1

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def heartbeat(self) -> None:
        """Call periodically from the service loop so a later gap starts as late as truthfully possible."""
        with self._mutex:
            self.marker.touch(self._now())

    # -- sealing ---------------------------------------------------------------------------------

    def open_clip(self, camera_id: str, started_at, suffix: str = ".mjpeg") -> ClipWriter:
        """Start a clip whose bytes are hashed as they are written (hash at capture)."""
        return ClipWriter(self, camera_id, to_rfc3339(started_at), suffix)

    def seal_clip(
        self,
        path: str | os.PathLike,
        camera_id: str,
        started_at,
        ended_at,
        frame_count: int | None = None,
        clip_id: str | None = None,
        move: bool = True,
    ) -> Block:
        """Seal a finished clip file that another writer produced (e.g. a container muxer).

        The file is hashed now, at the moment its writer closed it. With move=True (default) it is
        moved into the store; across filesystems it is copied through the hasher and the source is
        removed only after the copy is fsync'd. With move=False the block records its current path.
        `frame_count` defaults to the container's own count when OpenCV can read it, else 0.
        """
        src = Path(path)
        if frame_count is None:
            frame_count = _probe_frame_count(src)
        digest = hash_file(src, frame_count=frame_count)
        return self._append_clip(
            src=src,
            digest=digest,
            camera_id=camera_id,
            started_at=to_rfc3339(started_at),
            ended_at=to_rfc3339(ended_at),
            clip_id=clip_id,
            suffix=src.suffix or ".bin",
            move=move,
        )

    def _append_clip(self, *, src: Path, digest: ClipDigest, camera_id, started_at, ended_at, clip_id, suffix, move) -> Block:
        with self._mutex:
            if self._lock_fd < 0:
                raise RuntimeError("store is closed")
            last = self._blocks[-1] if self._blocks else None
            n = last.n + 1 if last else 1
            prev = last.block_hash if last else GENESIS_PREV
            sealed_at = self._now()
            if last and parse_ts(sealed_at) < parse_ts(last.sealed_at):
                sealed_at = last.sealed_at  # sealed_at never decreases along the chain
            day = day_of(sealed_at)
            if (roots_dir(self.root) / f"{day}.json").exists():
                raise DaySealed(f"day {day} is already sealed; cannot append block {n}")
            self._seal_days_before(day)

            clip_id = clip_id or f"clip-{n:08d}"
            if move:
                dest_dir = self.root / "clips" / day
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / f"{clip_id}{suffix}"
                _move_verified(src, dest, digest)
                os.chmod(dest, 0o444)
                fsync_dir(dest_dir)
                clip_path = str(dest.relative_to(self.root))
            else:
                clip_path = str(src.resolve())

            block = make_block(
                n=n,
                clip_id=clip_id,
                camera_id=camera_id,
                started_at=started_at,
                ended_at=ended_at,
                sealed_at=sealed_at,
                clip_sha256=digest.sha256,
                prev_block_hash=prev,
                clip_path=clip_path,
                frame_count=digest.frame_count,
                byte_length=digest.byte_length,
            )
            self.ledger.append(block)
            self._blocks.append(block)
            self.marker.touch(sealed_at)
            return block

    # -- daily roots -----------------------------------------------------------------------------

    def _unsealed_days(self) -> list[str]:
        days = sorted({b.day for b in self._blocks})
        return [d for d in days if not (roots_dir(self.root) / f"{d}.json").exists()]

    def _seal_days_before(self, day: str) -> list[SignedRoot]:
        return [self.seal_day(d, allow_open=True) for d in self._unsealed_days() if d < day]

    def seal_past_days(self) -> list[SignedRoot]:
        """Seal every day before today (IST) that has blocks and no root yet."""
        with self._mutex:
            return self._seal_days_before(today(self.clock))

    def seal_day(self, day: str, allow_open: bool = False) -> SignedRoot:
        """Compute, sign, store and mirror the root for `day`. Days are sealed once.

        Sealing today closes it for good (later blocks today raise DaySealed), so it needs
        allow_open=True; normal operation lets the day roll over instead.
        """
        with self._mutex:
            existing = load_signed_root(self.root, day)
            if existing is not None:
                self.mirror.publish(existing)
                return existing
            if not allow_open and day >= today(self.clock):
                raise DaySealed(f"day {day} is still open; pass allow_open=True to close it early")
            blocks = [b for b in self._blocks if b.day == day]
            gaps = self.gap_log.gaps()
            last = blocks[-1] if blocks else None
            manifest = DayManifest(
                device_id=self.device_id,
                day=day,
                day_tz=DAY_TZ_LABEL,
                leaf_count=len(blocks),
                first_n=blocks[0].n if blocks else 0,
                last_n=last.n if last else 0,
                last_block_hash=last.block_hash if last else GENESIS_PREV,
                merkle_root=merkle_root([b.block_hash for b in blocks]),
                gaps_count=len(gaps),
                gaps_head=gaps_head(gaps),
                key_id=self.signer.key_id,
                signed_at=self._now(),
            )
            signed = sign_manifest(self.signer, manifest)
            path = roots_dir(self.root) / f"{day}.json"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            try:
                os.write(fd, signed.to_json().encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            fsync_dir(path.parent)
            self.mirror.publish(signed)
            return signed

    # -- reads -----------------------------------------------------------------------------------

    def blocks(self) -> list[Block]:
        with self._mutex:
            return list(self._blocks)

    def gaps(self) -> list[Gap]:
        return self.gap_log.gaps()

    def proof(self, n: int) -> InclusionProof:
        return proof_for_block(self.root, n)["proof"]

    def _now(self) -> str:
        return to_rfc3339(self.clock())


def _move_verified(src: Path, dest: Path, digest: ClipDigest) -> None:
    """Move `src` to `dest`. Across filesystems, copy through a hasher and check it before unlinking."""
    if dest.exists():
        raise FileExistsError(dest)
    try:
        os.link(src, dest)  # same filesystem: atomic, no copy
        os.unlink(src)
        return
    except OSError:
        pass
    hasher = ClipHasher()
    with open(src, "rb") as fin, open(dest, "xb") as fout:
        while True:
            chunk = fin.read(1 << 20)
            if not chunk:
                break
            fout.write(chunk)
            hasher.update(chunk)
        fout.flush()
        os.fsync(fout.fileno())
    if hasher.finalize(frame_count=digest.frame_count).sha256 != digest.sha256:
        os.unlink(dest)
        raise IOError(f"{src} changed while it was being moved into the evidence store")
    os.unlink(src)


def _probe_frame_count(path: Path) -> int:
    try:
        import cv2  # optional: only used to read a container's own frame count
    except ImportError:
        return 0
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    finally:
        cap.release()
    return max(count, 0)
