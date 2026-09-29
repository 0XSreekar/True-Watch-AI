"""SHA-256 hashing at capture, the append-only chain, and the daily Merkle root.

Public API for the rest of the edge service (and, in Phase 8, the backend bridge):

    from evidence import EvidenceStore, chain_view, verify_view

    store = EvidenceStore(config.evidence_dir)             # one per post; recovers + logs gaps on open
    w = store.open_clip(camera_id, started_at)             # hash while writing
    w.write_frame(jpeg_bytes); block = w.seal(ended_at)
    block = store.seal_clip(path, camera_id, started_at, ended_at)   # a finished file
    event["evidence"] = block.evidence_dict()              # truewatch.event.v1 `evidence`

    chain_view(root, limit=5)   # FROZEN GET /api/evidence/chain  -> [{ n, hash, camera, time }]
    verify_view(root)           # FROZEN POST /api/evidence/verify -> { intact, verified, total, blocks }

CLIs: python -m evidence.verify | evidence.tamper_demo | evidence.seed_demo | evidence.certificate
"""

# Imported lazily so `python -m evidence.verify` (and the other CLIs) do not load their own module
# twice through the package import.
_EXPORTS = {
    "Block": "evidence.chain",
    "chain_view": "evidence.chain",
    "ClipDigest": "evidence.hasher",
    "ClipHasher": "evidence.hasher",
    "HashingWriter": "evidence.hasher",
    "hash_file": "evidence.hasher",
    "ChainCorrupt": "evidence.store",
    "ClipWriter": "evidence.store",
    "DaySealed": "evidence.store",
    "EvidenceStore": "evidence.store",
    "StoreLocked": "evidence.store",
    "VerifyReport": "evidence.verify",
    "verify_store": "evidence.verify",
    "verify_view": "evidence.verify",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'evidence' has no attribute {name!r}")


__all__ = [
    "Block",
    "ChainCorrupt",
    "ClipDigest",
    "ClipHasher",
    "ClipWriter",
    "DaySealed",
    "EvidenceStore",
    "HashingWriter",
    "StoreLocked",
    "VerifyReport",
    "chain_view",
    "hash_file",
    "verify_store",
    "verify_view",
]
