"""Sector-HQ mirror of the signed daily roots — a second directory, never a network endpoint.

The deck commits to a fully air-gapped post, so this module only ever touches the filesystem. In
the field `hq_dir` is removable media or a separately mounted volume that is carried to sector HQ;
in the demo it is a sibling directory. Nothing here opens a connection of any kind.

Layout under `hq_dir`:

    <device_id>/signing_key.pub.pem     the post's public key, registered once, never replaced
    <device_id>/roots/<YYYY-MM-DD>.json one SignedRoot per sealed day, written once

Both are WRITE-ONCE. Publishing a different root for a day HQ already holds raises MirrorConflict
instead of overwriting, so rewriting history on the box cannot silently rewrite HQ's copy, and
replacing the box's key cannot re-register a new one. Files are written to a temporary name,
fsync'd and renamed, so a power cut leaves either the old state or the complete new file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from evidence.hasher import fsync_dir
from evidence.signer import SignedRoot, public_raw_from_pem

_RE_DEVICE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_RE_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class MirrorConflict(RuntimeError):
    """HQ already holds a different value for something that is write-once."""


def _atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    fsync_dir(path.parent)


class HQMirror:
    def __init__(self, hq_dir: str | os.PathLike, device_id: str) -> None:
        if not _RE_DEVICE.match(device_id):
            raise ValueError(f"device_id {device_id!r} must match [A-Za-z0-9._:-]{{1,64}}")
        self.hq_dir = Path(hq_dir)
        self.device_id = device_id
        self.base = self.hq_dir / device_id

    @property
    def key_path(self) -> Path:
        return self.base / "signing_key.pub.pem"

    def root_path(self, day: str) -> Path:
        if not _RE_DAY.match(day):
            raise ValueError(f"day {day!r} must be YYYY-MM-DD")
        return self.base / "roots" / f"{day}.json"

    # -- public key ------------------------------------------------------------------------------

    def register_key(self, public_pem: bytes) -> None:
        raw = public_raw_from_pem(public_pem)
        if self.key_path.exists():
            if public_raw_from_pem(self.key_path.read_bytes()) != raw:
                raise MirrorConflict(f"HQ already holds a different public key for {self.device_id}")
            return
        _atomic_write(self.key_path, public_pem)

    def registered_key_raw(self) -> bytes | None:
        if not self.key_path.exists():
            return None
        return public_raw_from_pem(self.key_path.read_bytes())

    # -- daily roots -----------------------------------------------------------------------------

    def publish(self, signed: SignedRoot) -> None:
        path = self.root_path(signed.manifest.day)
        if path.exists():
            held = SignedRoot.from_dict(json.loads(path.read_text("utf-8")))
            if held != signed:
                raise MirrorConflict(f"HQ already holds a different root for {signed.manifest.day}")
            return
        _atomic_write(path, signed.to_json().encode("utf-8"))

    def read_root(self, day: str) -> SignedRoot | None:
        path = self.root_path(day)
        if not path.exists():
            return None
        return SignedRoot.from_dict(json.loads(path.read_text("utf-8")))

    def days(self) -> list[str]:
        d = self.base / "roots"
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.json") if _RE_DAY.match(p.stem))
