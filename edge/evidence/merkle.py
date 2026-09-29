"""Daily Merkle root over the day's block hashes, with inclusion proofs.

Hashing (domain-separated, as in RFC 6962 so a leaf can never be passed off as an inner node):

    leaf_hash(block_hash) = SHA-256( 0x00 || bytes.fromhex(block_hash) )
    node_hash(left, right) = SHA-256( 0x01 || left || right )
    root of an empty day    = SHA-256( b"" )

Tree shape — ODD-LEAF RULE: PROMOTE. Leaves stay in chain order (ascending n). Each level pairs
nodes left to right: (0,1), (2,3), ... When a level has an odd count, its last node is carried up
to the next level UNCHANGED (it is not duplicated and not hashed with itself). Duplicate-last is
deliberately not used: it makes [a, b, c] and [a, b, c, c] share one root (the ambiguity behind
CVE-2012-2459), so an attacker could add a duplicated final block without changing the root.

An inclusion proof is the list of sibling hashes from the leaf up, plus the leaf index and the
leaf count. The verifier derives on which side each sibling sits, and which levels promote the node
with no sibling, from (index, leaf_count) alone — it never trusts direction flags from the proof —
so a proof cannot be bent to fit a different position or a different tree size.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"
EMPTY_ROOT = hashlib.sha256(b"").hexdigest()


def leaf_hash(block_hash_hex: str) -> bytes:
    return hashlib.sha256(LEAF_PREFIX + bytes.fromhex(block_hash_hex)).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def _levels(block_hashes: list[str]) -> list[list[bytes]]:
    level = [leaf_hash(h) for h in block_hashes]
    levels = [level]
    while len(level) > 1:
        nxt = [node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(level[-1])  # promote the odd node unchanged
        level = nxt
        levels.append(level)
    return levels


def merkle_root(block_hashes: list[str]) -> str:
    """Root (64 lowercase hex) over block hashes in chain order."""
    if not block_hashes:
        return EMPTY_ROOT
    return _levels(list(block_hashes))[-1][0].hex()


@dataclass(frozen=True)
class InclusionProof:
    index: int  # 0-based position of the leaf among the day's blocks
    leaf_count: int
    siblings: tuple[str, ...] = field(default_factory=tuple)  # hex, leaf level upward

    def to_dict(self) -> dict:
        return {"index": self.index, "leaf_count": self.leaf_count, "siblings": list(self.siblings)}

    @classmethod
    def from_dict(cls, d: dict) -> "InclusionProof":
        return cls(int(d["index"]), int(d["leaf_count"]), tuple(str(s) for s in d["siblings"]))


def inclusion_proof(block_hashes: list[str], index: int) -> InclusionProof:
    n = len(block_hashes)
    if not 0 <= index < n:
        raise IndexError(f"leaf index {index} outside 0..{n - 1}")
    siblings: list[str] = []
    idx = index
    for level in _levels(list(block_hashes))[:-1]:
        if idx % 2 == 1:
            siblings.append(level[idx - 1].hex())
        elif idx + 1 < len(level):
            siblings.append(level[idx + 1].hex())
        # else: promoted, no sibling at this level
        idx //= 2
    return InclusionProof(index, n, tuple(siblings))


def verify_inclusion(
    block_hash_hex: str, proof: InclusionProof, root_hex: str, expected_leaf_count: int | None = None
) -> bool:
    """True iff `block_hash_hex` sits at `proof.index` in a tree of `proof.leaf_count` leaves with `root_hex`.

    The root does not commit to the leaf count by itself (two sizes can share a path shape), so
    pass `expected_leaf_count` from the signed manifest whenever one exists; certificate.py does.
    """
    try:
        if expected_leaf_count is not None and proof.leaf_count != expected_leaf_count:
            return False
        if not 0 <= proof.index < proof.leaf_count:
            return False
        acc = leaf_hash(block_hash_hex)
        idx, size = proof.index, proof.leaf_count
        it = iter(proof.siblings)
        while size > 1:
            if idx % 2 == 1:
                acc = node_hash(bytes.fromhex(next(it)), acc)
            elif idx + 1 < size:
                acc = node_hash(acc, bytes.fromhex(next(it)))
            idx //= 2
            size = (size + 1) // 2
        if next(it, None) is not None:  # unused siblings: the proof is for a different shape
            return False
        return acc.hex() == root_hex
    except (StopIteration, ValueError):
        return False
