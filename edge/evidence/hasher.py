"""SHA-256 of a clip, taken while the clip is being written.

Hashing happens at capture, not afterwards: `HashingWriter` sits between the recorder and the file,
so every byte goes into the digest in the same call that writes it to disk. When the writer is closed
the digest is already final; nothing re-reads the file to "seal" it. A clip that some other process
wrote (for example a muxer that owns its own file handle) is sealed with `hash_file`, the same
function the verifier uses, at the moment the recorder closes it.

Digest definition (version 1, never changed once clips exist):

    clip_sha256 = SHA-256( content || b"TWCLIP1" || uint64_be(frame_count) || uint64_be(byte_length) )

- `content` is the exact bytes of the file, in order.
- `frame_count` is the number of frames the recorder wrote (0 when the caller cannot count frames).
- `byte_length` is len(content). It is repeated in the fixed-width trailer so a truncated or
  extended file can never produce the digest of the original, and the content/trailer boundary is
  unambiguous (the trailer is always the last 23 bytes of the digest input).

The frame count cannot be recomputed from arbitrary bytes, so it is stored next to the block; it is
still bound by the digest, so editing the stored count breaks verification exactly like editing
the clip does.
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

TRAILER_TAG = b"TWCLIP1"
CHUNK = 1 << 20  # 1 MiB reads when re-hashing a file from disk


@dataclass(frozen=True)
class ClipDigest:
    sha256: str  # 64 lowercase hex
    frame_count: int
    byte_length: int


class ClipHasher:
    """Incremental clip digest. Feed it bytes in write order, then `finalize()` once."""

    def __init__(self) -> None:
        self._h = hashlib.sha256()
        self._frames = 0
        self._bytes = 0
        self._final: ClipDigest | None = None

    def update(self, data: bytes | bytearray | memoryview, frames: int = 0) -> None:
        if self._final is not None:
            raise ValueError("ClipHasher already finalized")
        if frames < 0:
            raise ValueError("frames must be >= 0")
        mv = memoryview(data)
        self._h.update(mv)
        self._bytes += mv.nbytes
        self._frames += frames

    @property
    def frame_count(self) -> int:
        return self._frames

    @property
    def byte_length(self) -> int:
        return self._bytes

    def finalize(self, frame_count: int | None = None) -> ClipDigest:
        """Close the digest. `frame_count` overrides the counted frames when the caller knows better."""
        if self._final is None:
            frames = self._frames if frame_count is None else int(frame_count)
            if frames < 0:
                raise ValueError("frame_count must be >= 0")
            h = self._h.copy()
            h.update(TRAILER_TAG + struct.pack(">QQ", frames, self._bytes))
            self._final = ClipDigest(h.hexdigest(), frames, self._bytes)
        return self._final


class HashingWriter:
    """A binary file writer that hashes every byte in the same call that writes it.

    Use `write_frame(data)` for one encoded frame (counts a frame) or `write(data)` for container
    bytes that are not a frame (headers, padding). `close()` flushes, fsyncs and returns the
    `ClipDigest`; the file on disk and the digest describe exactly the same bytes.
    """

    def __init__(self, path: str | os.PathLike, mode: int = 0o640) -> None:
        self.path = Path(path)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        self._fh: BinaryIO | None = os.fdopen(fd, "wb")
        self._hasher = ClipHasher()
        self.digest: ClipDigest | None = None

    def write(self, data: bytes | bytearray | memoryview) -> int:
        return self._write(data, frames=0)

    def write_frame(self, data: bytes | bytearray | memoryview) -> int:
        return self._write(data, frames=1)

    def _write(self, data, frames: int) -> int:
        if self._fh is None:
            raise ValueError("HashingWriter is closed")
        mv = memoryview(data)
        written = 0
        while written < mv.nbytes:  # a short write must not leave the digest ahead of the file
            n = self._fh.write(mv[written:])
            written += n if n is not None else mv.nbytes - written
        self._hasher.update(mv, frames=frames)
        return written

    @property
    def frame_count(self) -> int:
        return self._hasher.frame_count

    @property
    def byte_length(self) -> int:
        return self._hasher.byte_length

    @property
    def closed(self) -> bool:
        return self._fh is None

    def close(self) -> ClipDigest:
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None
            self.digest = self._hasher.finalize()
        assert self.digest is not None
        return self.digest

    def abort(self) -> None:
        """Close without producing a digest (the caller will discard or quarantine the file)."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "HashingWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def hash_file(path: str | os.PathLike, frame_count: int = 0) -> ClipDigest:
    """Recompute a clip digest from the file on disk (used at seal time for foreign files and by verify)."""
    hasher = ClipHasher()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.finalize(frame_count=frame_count)


def fsync_dir(path: str | os.PathLike) -> None:
    """Make a rename or file creation inside `path` durable (no-op where directories cannot be opened)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
