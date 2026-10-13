"""Ed25519 signing of the daily Merkle root, with a key that is generated on the box and never leaves it.

Key handling
- The key pair is generated on first use with `cryptography`'s Ed25519 implementation (no crypto is
  implemented here). The private key is written once, as unencrypted PKCS#8 PEM, to
  `<store>/keys/signing_key.pem`, created with O_EXCL and mode 0600 inside a 0700 directory, so there
  is no window in which another user can read it. An unattended post has no operator to type a
  passphrase at boot; file permissions and the box's physical custody are the protection.
- Loading refuses a key file readable or writable by group/other.
- `Signer` has no method that returns private key bytes, its repr shows only the public key id, and
  it refuses to be pickled, so the private key cannot be logged or serialised by accident.
- The public key is exported as `<store>/keys/signing_key.pub.pem` and registered with sector HQ
  (mirror.py) the first time the store opens. Verification trusts HQ's registered copy.

What is signed: the canonical bytes of a DayManifest (framing as in chain.py):

    "truewatch.dayroot.v1" LF
    device_id= LF  day= LF  day_tz= LF  leaf_count= LF  first_n= LF  last_n= LF
    last_block_hash= LF  merkle_root= LF  gaps_count= LF  gaps_head= LF  key_id= LF  signed_at= LF

in exactly that order, so the signature covers the root, the block range it spans, the tip of the
chain at the end of the day, and the gap log as it stood when the day was sealed.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from evidence.chain import canonical_bytes

PRIVATE_KEY_NAME = "signing_key.pem"
PUBLIC_KEY_NAME = "signing_key.pub.pem"
MANIFEST_DOMAIN = "truewatch.dayroot.v1"
MANIFEST_FIELDS = (
    "device_id",
    "day",
    "day_tz",
    "leaf_count",
    "first_n",
    "last_n",
    "last_block_hash",
    "merkle_root",
    "gaps_count",
    "gaps_head",
    "key_id",
    "signed_at",
)


class InsecureKeyFile(PermissionError):
    """The private key file is accessible to group or other; it is not used."""


def key_id_of(public_raw: bytes) -> str:
    return hashlib.sha256(public_raw).hexdigest()[:16]


def public_raw_from_pem(pem: bytes) -> bytes:
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("not an Ed25519 public key")
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def verify_signature(public_raw: bytes, data: bytes, signature_hex: str) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(public_raw).verify(bytes.fromhex(signature_hex), data)
        return True
    except (InvalidSignature, ValueError):
        return False


class Signer:
    """Holds the box's Ed25519 private key in memory and signs with it."""

    __slots__ = ("_key", "public_raw", "key_id")

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._key = private_key
        self.public_raw = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        self.key_id = key_id_of(self.public_raw)

    # -- lifecycle -------------------------------------------------------------------------------

    @classmethod
    def load_or_create(cls, key_dir: str | os.PathLike) -> "Signer":
        key_dir = Path(key_dir)
        key_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(key_dir, 0o700)
        priv_path = key_dir / PRIVATE_KEY_NAME
        if priv_path.exists():
            mode = stat.S_IMODE(os.stat(priv_path).st_mode)
            if mode & 0o077:
                raise InsecureKeyFile(f"{priv_path} has mode {mode:04o}; it must be 0600 (chmod 600 it)")
            key = serialization.load_pem_private_key(priv_path.read_bytes(), password=None)
            if not isinstance(key, Ed25519PrivateKey):
                raise ValueError(f"{priv_path} is not an Ed25519 private key")
            signer = cls(key)
        else:
            key = Ed25519PrivateKey.generate()
            pem = key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
            fd = os.open(priv_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.fchmod(fd, 0o600)  # independent of the process umask
                os.write(fd, pem)
                os.fsync(fd)
            finally:
                os.close(fd)
            signer = cls(key)
        pub_path = key_dir / PUBLIC_KEY_NAME
        if not pub_path.exists() or public_raw_from_pem(pub_path.read_bytes()) != signer.public_raw:
            pub_path.write_bytes(signer.public_pem())
            os.chmod(pub_path, 0o644)
        return signer

    # -- public material -------------------------------------------------------------------------

    def public_pem(self) -> bytes:
        return Ed25519PublicKey.from_public_bytes(self.public_raw).public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def sign(self, data: bytes) -> str:
        return self._key.sign(data).hex()

    # -- never leak the private key --------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Signer(ed25519, key_id={self.key_id})"

    __str__ = __repr__

    def __getstate__(self):
        raise TypeError("Signer holds a private key and cannot be serialised")

    def __reduce__(self):
        raise TypeError("Signer holds a private key and cannot be serialised")


@dataclass(frozen=True)
class DayManifest:
    device_id: str
    day: str
    day_tz: str
    leaf_count: int
    first_n: int
    last_n: int
    last_block_hash: str
    merkle_root: str
    gaps_count: int
    gaps_head: str
    key_id: str
    signed_at: str

    def canonical(self) -> bytes:
        return canonical_bytes(MANIFEST_DOMAIN, ((f, getattr(self, f)) for f in MANIFEST_FIELDS))

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in MANIFEST_FIELDS}

    @classmethod
    def from_dict(cls, d: dict) -> "DayManifest":
        kw = {f: d[f] for f in MANIFEST_FIELDS}
        for f in ("leaf_count", "first_n", "last_n", "gaps_count"):
            kw[f] = int(kw[f])
        return cls(**kw)


@dataclass(frozen=True)
class SignedRoot:
    manifest: DayManifest
    signature: str  # Ed25519 signature over manifest.canonical(), hex
    public_key: str  # raw Ed25519 public key, hex (informational; verify uses the HQ-registered key)

    def to_dict(self) -> dict:
        return {"manifest": self.manifest.to_dict(), "signature": self.signature, "public_key": self.public_key}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"

    @classmethod
    def from_dict(cls, d: dict) -> "SignedRoot":
        return cls(DayManifest.from_dict(d["manifest"]), str(d["signature"]), str(d["public_key"]))

    def verify_with(self, public_raw: bytes) -> bool:
        return verify_signature(public_raw, self.manifest.canonical(), self.signature)


def sign_manifest(signer: Signer, manifest: DayManifest) -> SignedRoot:
    if manifest.key_id != signer.key_id:
        raise ValueError("manifest key_id does not match the signer")
    return SignedRoot(manifest, signer.sign(manifest.canonical()), signer.public_raw.hex())


__all__ = [
    "DayManifest",
    "InsecureKeyFile",
    "SignedRoot",
    "Signer",
    "key_id_of",
    "public_raw_from_pem",
    "sign_manifest",
    "verify_signature",
]
