"""Tamper demo: flip ONE byte in ONE sealed clip and watch verification name the block that broke.

    python -m evidence.tamper_demo                 # middle block of the chain, middle byte of its clip
    python -m evidence.tamper_demo --block 5       # a specific block
    python -m evidence.tamper_demo --restore       # flip the same byte back (XOR is its own inverse)

Sealed clips are read-only (0444). The demo does what an insider with file access would do — makes
the file writable, changes one byte, puts the permissions back — then runs the same verification
as `python -m evidence.verify`. The block whose clip changed fails because its SHA-256 no longer
matches, and every later block fails because its recomputed predecessor hash no longer links to
it; the day's Merkle root stops matching the signed root too.

Exit status: 0 when the demo behaved as it should (tamper detected at exactly the chosen block, or,
with --restore, the chain passes again); 1 otherwise; 2 if the store cannot be read.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

from evidence.chain import Ledger, ledger_path
from evidence.store import resolve_clip
from evidence.verify import StoreUnreadable, default_data_dir, render, verify_store

FLIP_MASK = 0x01


def flip_byte(path: Path, offset: int, mask: int = FLIP_MASK) -> tuple[int, int]:
    """XOR one byte in place, keeping the file's permission bits. Returns (before, after)."""
    mode = stat.S_IMODE(os.stat(path).st_mode)
    os.chmod(path, mode | stat.S_IWUSR)
    try:
        with open(path, "r+b") as fh:
            fh.seek(offset)
            old = fh.read(1)
            if len(old) != 1:
                raise ValueError(f"offset {offset} is beyond the end of {path}")
            new = bytes([old[0] ^ mask])
            fh.seek(offset)
            fh.write(new)
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        os.chmod(path, mode)
    return old[0], new[0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evidence.tamper_demo", description="Flip one byte in one sealed clip.")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--hq-dir", type=Path, default=None)
    p.add_argument("--block", type=int, default=None, help="block number (default: middle of the chain)")
    p.add_argument("--offset", type=int, default=None, help="byte offset in the clip (default: middle)")
    p.add_argument("--restore", action="store_true", help="flip the same byte back and re-verify")
    args = p.parse_args(argv)

    root = args.data_dir or default_data_dir()
    hq = args.hq_dir or (Path(os.environ["EVIDENCE_HQ_DIR"]) if os.environ.get("EVIDENCE_HQ_DIR") else None)
    blocks = Ledger(ledger_path(root)).blocks()
    if not blocks:
        print(f"no sealed blocks in {root}; run `python -m evidence.seed_demo` first", file=sys.stderr)
        return 2
    by_n = {b.n: b for b in blocks}
    n = args.block if args.block is not None else blocks[(len(blocks) - 1) // 2].n
    if n not in by_n:
        print(f"block {n} is not in the chain (1..{blocks[-1].n})", file=sys.stderr)
        return 2
    block = by_n[n]
    clip = resolve_clip(root, block)
    offset = args.offset if args.offset is not None else max(block.byte_length // 2, 0)

    before, after = flip_byte(clip, offset)
    verb = "Restored" if args.restore else "Tampered"
    print(f"{verb}: block {n} ({block.camera_id}, sealed {block.sealed_at})")
    print(f"  clip   {clip}")
    print(f"  byte   offset {offset} of {block.byte_length}: 0x{before:02x} -> 0x{after:02x}")
    print(f"  sealed SHA-256 {block.clip_sha256}")
    print()

    try:
        report = verify_store(root, hq_dir=hq)
    except StoreUnreadable as exc:
        print(f"cannot verify: {exc}", file=sys.stderr)
        return 2
    print(render(report, all_blocks=True))
    print()

    if args.restore:
        print("Chain verifies again." if report.ok else "Chain still fails: something else was changed.")
        return 0 if report.ok else 1

    fb = report.first_broken
    later = [b.n for b in report.blocks if b.n > n]
    downstream_bad = [b.n for b in report.blocks if b.n > n and not b.ok]
    detected = (
        not report.ok
        and fb is not None
        and fb.n == n
        and fb.root_cause
        and downstream_bad == later
    )
    if detected:
        span = f"blocks {later[0]}..{later[-1]}" if later else "no later blocks"
        print(f"DETECTED: one flipped byte broke block {n}, and every block after it ({span}) is now invalid.")
        print(f"Undo with: python -m evidence.tamper_demo --block {n} --offset {offset} --restore")
        return 0
    print("NOT DETECTED as expected — investigate.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
