"""THE ONE COMMAND: verify the evidence store from the bytes on disk up to the signed daily roots.

    python -m evidence.verify --from 2026-09-01 --to 2026-09-30
    python -m evidence.verify                      # the whole chain
    python -m evidence.verify --json               # the FROZEN verify shape, for the backend

Steps, in order, each reported in the PASS/FAIL table:
  1. recompute every clip's SHA-256 from the file on disk (hasher.py digest definition)
  2. recompute every block hash from its canonical serialisation (chain.py)
  3. check every prev-link and that block numbers run 1, 2, 3, ... with no hole or swap
  4. recompute each day's Merkle root from the RECOMPUTED chain and compare with the signed manifest
  5. verify each manifest's Ed25519 signature with the public key registered at sector HQ
  6. cross-check the local manifest and public key against the HQ mirror
  7. check the gap log's own chain and list every logged gap

The chain is walked from genesis even when --from starts later, because a block is only as good as
every block before it. A block is valid only when the hash recomputed from ITS clip bytes and ITS
recomputed predecessor equals the stored block hash — so one changed byte in one clip breaks that
block AND every block after it, which is what the chain promises.

Exit status: 0 all checks pass (an open day — today, not yet sealed — is reported as OPEN, not a
failure), 1 any check fails, 2 the store cannot be read or the arguments are wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from evidence.chain import GENESIS_PREV, CanonicalError, Ledger, day_of, ledger_path, sha256_hex
from evidence.gaps import GAPS_NAME, Gap, GapLog, check_gap_chain, gaps_head
from evidence.hasher import hash_file
from evidence.merkle import merkle_root
from evidence.mirror import HQMirror
from evidence.signer import PUBLIC_KEY_NAME, SignedRoot, public_raw_from_pem
from evidence.store import default_hq_dir, read_meta, resolve_clip, roots_dir, today

PASS, FAIL, OPEN, INFO = "PASS", "FAIL", "OPEN", "INFO"


class StoreUnreadable(RuntimeError):
    pass


@dataclass
class Check:
    step: int
    name: str
    scope: str
    status: str
    detail: str = ""


@dataclass
class BlockResult:
    n: int
    camera_id: str
    day: str
    ok: bool
    reasons: list[str] = field(default_factory=list)
    root_cause: bool = False  # True: this block's own data is wrong; False with ok=False: broken upstream


@dataclass
class VerifyReport:
    root: str
    device_id: str
    date_from: str | None
    date_to: str | None
    checks: list[Check]
    blocks: list[BlockResult]
    gaps: list[Gap]

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks) and all(b.ok for b in self.blocks)

    @property
    def first_broken(self) -> BlockResult | None:
        return next((b for b in self.blocks if not b.ok), None)

    def frozen(self) -> dict:
        """FROZEN `POST /api/evidence/verify` shape: { intact, verified, total, blocks: [{ n, ok }] }."""
        return {
            "intact": self.ok,
            "verified": sum(1 for b in self.blocks if b.ok),
            "total": len(self.blocks),
            "blocks": [{"n": b.n, "ok": b.ok} for b in self.blocks],
        }


def _in_range(day: str, date_from: str | None, date_to: str | None) -> bool:
    return (date_from is None or day >= date_from) and (date_to is None or day <= date_to)


def _load_root_file(path: Path) -> tuple[SignedRoot | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return SignedRoot.from_dict(json.loads(path.read_text("utf-8"))), None
    except (ValueError, KeyError, TypeError) as exc:
        return None, f"{path.name} is not a valid signed root ({type(exc).__name__})"


def verify_store(
    root: str | os.PathLike,
    date_from: str | None = None,
    date_to: str | None = None,
    hq_dir: str | os.PathLike | None = None,
    today_str: str | None = None,
) -> VerifyReport:
    root = Path(root)
    if not (root / "store.json").exists():
        raise StoreUnreadable(f"{root} is not an evidence store (no store.json)")
    meta = read_meta(root)
    device_id = meta["device_id"]
    today_str = today_str or today()
    checks: list[Check] = []

    # ---- ledger --------------------------------------------------------------------------------
    read = Ledger(ledger_path(root)).read()
    for p in read.problems:
        checks.append(Check(2, "ledger records", "chain", FAIL, p))
    if read.torn_tail:
        checks.append(
            Check(2, "ledger records", "chain", FAIL, f"{len(read.torn_tail)} bytes of an unfinished write at the end")
        )

    # ---- steps 1-3: walk the whole chain from genesis ------------------------------------------
    results: list[BlockResult] = []
    recomputed: dict[int, str] = {}  # position -> block hash recomputed from disk + recomputed predecessor
    prev_stored, prev_recomputed = GENESIS_PREV, GENESIS_PREV
    last_sealed = None
    first_bad: int | None = None
    for pos, b in enumerate(read.blocks):
        reasons: list[str] = []
        expected_n = pos + 1
        if b.n != expected_n:
            reasons.append(f"sequence: expected block {expected_n} here, found {b.n} (block removed or reordered)")
        clip = resolve_clip(root, b)
        clip_sha = None
        if not clip.is_file():
            reasons.append(f"clip file missing: {b.clip_path}")
        else:
            clip_sha = hash_file(clip, frame_count=b.frame_count).sha256
            if clip_sha != b.clip_sha256:
                reasons.append("clip bytes on disk do not match the sealed SHA-256")
        try:
            own = b.compute_hash()
            if own != b.block_hash:
                reasons.append("block record altered: canonical hash does not recompute")
        except CanonicalError as exc:
            reasons.append(f"block record not canonical: {exc}")
        if b.prev_block_hash != prev_stored:
            reasons.append("prev link broken: prev_block_hash is not the previous block's hash")
        if last_sealed and b.sealed_at < last_sealed:
            reasons.append("sealed_at goes backwards")
        last_sealed = b.sealed_at

        try:
            again = replace(b, clip_sha256=clip_sha or ("0" * 64), prev_block_hash=prev_recomputed)
            rh = sha256_hex(again.canonical())
        except CanonicalError:
            rh = ""
        ok = not reasons and rh == b.block_hash
        root_cause = bool(reasons)
        if not ok and not reasons:
            reasons.append(f"invalid downstream of block {first_bad}: its recomputed hash no longer links here")
        if not ok and first_bad is None:
            first_bad = b.n
        try:
            day = b.day
        except ValueError:
            day = "????-??-??"
        results.append(BlockResult(b.n, b.camera_id, day, ok, reasons, root_cause))
        recomputed[pos] = rh
        prev_stored, prev_recomputed = b.block_hash, rh

    in_range = [r for r in results if _in_range(r.day, date_from, date_to)]
    before = [r for r in results if date_from is not None and r.day < date_from]
    if before:
        bad_before = [r for r in before if not r.ok]
        checks.append(
            Check(
                3,
                "chain before --from",
                f"blocks {before[0].n}..{before[-1].n}",
                FAIL if bad_before else PASS,
                f"first broken block {bad_before[0].n}" if bad_before else f"{len(before)} earlier blocks link cleanly",
            )
        )
    scope = f"{len(in_range)} blocks" if in_range else "no blocks"
    for step, name, needle in (
        (1, "clip SHA-256 from disk", ("clip",)),
        (2, "block hash recompute", ("block record",)),
        (3, "prev links + sequence", ("prev link", "sequence", "sealed_at", "downstream")),
    ):
        bad = [r for r in in_range if any(any(s in reason for s in needle) for reason in r.reasons)]
        detail = (
            "blocks " + ", ".join(str(r.n) for r in bad[:12]) + (" ..." if len(bad) > 12 else "")
            if bad
            else f"{len(in_range)}/{len(in_range)} ok"
        )
        checks.append(Check(step, name, scope, FAIL if bad else PASS, detail))

    # ---- steps 4-6: daily roots, signatures, HQ mirror -----------------------------------------
    mirror = HQMirror(hq_dir or default_hq_dir(root), device_id)
    local_pub_path = root / "keys" / PUBLIC_KEY_NAME
    local_key = public_raw_from_pem(local_pub_path.read_bytes()) if local_pub_path.exists() else None
    key_checks: list[Check] = []
    try:
        hq_key = mirror.registered_key_raw()
    except ValueError:
        hq_key = None
        key_checks.append(Check(6, "HQ public key", "mirror", FAIL, "HQ key file is not a valid Ed25519 public key"))
    if hq_key is None:
        key_checks.append(Check(6, "HQ public key", "mirror", FAIL, f"no key registered at {mirror.key_path}"))
    elif local_key != hq_key:
        key_checks.append(Check(6, "HQ public key", "mirror", FAIL, "box public key differs from the key registered at HQ"))
    else:
        key_checks.append(Check(6, "HQ public key", "mirror", PASS, "box key matches HQ registration"))
    trusted_key = hq_key  # never the box's own copy: an attacker on the box could re-key it

    gaps, gap_problems = GapLog(root / GAPS_NAME).read()
    gap_problems += check_gap_chain(gaps)

    positions_by_day: dict[str, list[int]] = {}
    for pos, b in enumerate(read.blocks):
        positions_by_day.setdefault(results[pos].day, []).append(pos)
    local_days = {p.stem for p in roots_dir(root).glob("*.json")} if roots_dir(root).is_dir() else set()
    all_days = sorted(set(positions_by_day) | local_days | set(mirror.days()))
    last_day = results[-1].day if results else None

    for day in (d for d in all_days if _in_range(d, date_from, date_to)):
        positions = positions_by_day.get(day, [])
        local, local_err = _load_root_file(roots_dir(root) / f"{day}.json")
        remote, remote_err = _load_root_file(mirror.root_path(day))
        is_open = local is None and remote is None and day == last_day and day >= today_str
        if is_open:
            checks.append(Check(4, "daily Merkle root", day, OPEN, "today's chain; the root is signed when the day ends"))
            checks.append(Check(5, "Ed25519 signature", day, OPEN, "not signed yet"))
            checks.append(Check(6, "HQ mirror", day, OPEN, "nothing to mirror yet"))
            continue
        if local is None:
            why = local_err or ("signed root deleted from the box (HQ still holds it)" if remote else "day has no signed root")
            checks.append(Check(4, "daily Merkle root", day, FAIL, why))
            checks.append(Check(5, "Ed25519 signature", day, FAIL, "no local signed root to check"))
        else:
            m = local.manifest
            hashes = [recomputed[p] for p in positions]
            root_now = merkle_root(hashes)
            problems = []
            if root_now != m.merkle_root:
                problems.append(f"recomputed root {root_now[:12]}... != signed {m.merkle_root[:12]}...")
            if m.leaf_count != len(positions):
                problems.append(f"signed for {m.leaf_count} blocks, ledger has {len(positions)}")
            ns = [read.blocks[p].n for p in positions]
            if positions and (m.first_n, m.last_n) != (ns[0], ns[-1]):
                problems.append(f"signed range {m.first_n}..{m.last_n}, ledger has {ns[0]}..{ns[-1]}")
            if positions and m.last_block_hash != recomputed[positions[-1]]:
                problems.append("last block hash differs from the signed tip")
            if m.gaps_count > len(gaps) or gaps_head(gaps, m.gaps_count) != m.gaps_head:
                problems.append("gap log differs from the one signed with this day")
            if m.device_id != device_id or m.day != day:
                problems.append("manifest names a different device or day")
            checks.append(
                Check(4, "daily Merkle root", day, FAIL if problems else PASS,
                      "; ".join(problems) or f"{m.leaf_count} leaves, root {m.merkle_root[:12]}...")
            )
            if trusted_key is None:
                checks.append(Check(5, "Ed25519 signature", day, FAIL, "no HQ-registered key to verify against"))
            elif local.verify_with(trusted_key):
                checks.append(Check(5, "Ed25519 signature", day, PASS, f"valid, key {m.key_id}"))
            else:
                checks.append(Check(5, "Ed25519 signature", day, FAIL, "signature does not verify under the HQ-registered key"))
        if remote is None:
            checks.append(Check(6, "HQ mirror", day, FAIL, remote_err or "HQ holds no root for this day"))
        elif local is not None and remote != local:
            checks.append(Check(6, "HQ mirror", day, FAIL, "HQ copy differs from the box's copy"))
        elif local is not None:
            checks.append(Check(6, "HQ mirror", day, PASS, "HQ copy identical"))

    checks.extend(key_checks)

    # ---- step 7: gaps --------------------------------------------------------------------------
    listed = [
        g for g in gaps
        if (date_from is None or day_of(g.end) >= date_from) and (date_to is None or day_of(g.start) <= date_to)
    ]
    if gap_problems:
        checks.append(Check(7, "gap log", "gaps", FAIL, "; ".join(gap_problems)))
    else:
        checks.append(Check(7, "gap log", "gaps", INFO if listed else PASS,
                            f"{len(gaps)} gap(s) logged, chain intact; {len(listed)} in range"))

    return VerifyReport(str(root), device_id, date_from, date_to, checks, in_range, listed)


def verify_view(
    root: str | os.PathLike,
    hq_dir: str | os.PathLike | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """FROZEN `POST /api/evidence/verify` payload: { intact, verified, total, blocks: [{ n, ok }] }."""
    return verify_store(root, date_from, date_to, hq_dir).frozen()


# --------------------------------------------------------------------------- printing


def _table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths)).rstrip()  # noqa: E731
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def render(report: VerifyReport, all_blocks: bool | None = None) -> str:
    out = []
    rng = f"{report.date_from or 'start'} .. {report.date_to or 'end'}"
    out.append(f"TRUEWATCH evidence verification  store={report.root}  device={report.device_id}  range={rng}")
    out.append("")
    out.append(_table([[str(c.step), c.name, c.scope, c.status, c.detail] for c in report.checks],
                      ["#", "CHECK", "SCOPE", "RESULT", "DETAIL"]))
    out.append("")
    show_all = all_blocks if all_blocks is not None else len(report.blocks) <= 60
    rows = []
    for b in report.blocks:
        if show_all or not b.ok:
            tag = "PASS" if b.ok else ("FAIL  <-- broken here" if b.root_cause else "FAIL  (downstream)")
            rows.append([str(b.n), b.camera_id, b.day, tag, "; ".join(b.reasons)])
    if rows:
        out.append(_table(rows, ["BLOCK", "CAMERA", "DAY", "RESULT", "WHY"]))
    if not show_all:
        out.append(f"({sum(b.ok for b in report.blocks)} passing blocks not listed; --all-blocks lists them)")
    out.append("")
    if report.gaps:
        out.append(_table([[str(g.g), g.kind, g.start, g.end, f"{g.duration_s:.0f}s", str(g.last_block_n), g.reason]
                           for g in report.gaps],
                          ["GAP", "KIND", "START", "END", "LENGTH", "AFTER BLOCK", "REASON"]))
    else:
        out.append("No chain gaps logged in range.")
    out.append("")
    if report.ok:
        out.append(f"RESULT: PASS  ({len(report.blocks)} blocks verified)")
    else:
        fb = report.first_broken
        if fb is not None:
            downstream = [b.n for b in report.blocks if not b.ok and b.n != fb.n]
            tail = f"; {len(downstream)} downstream block(s) invalid" if downstream else ""
            out.append(f"RESULT: FAIL  first broken block {fb.n}{tail}")
        else:
            out.append("RESULT: FAIL  (see the table above)")
    return "\n".join(out)


def _date(text: str) -> str:
    return date.fromisoformat(text).isoformat()


def default_data_dir() -> Path:
    return Path(os.environ.get("EVIDENCE_DIR", "./var/evidence"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evidence.verify", description=__doc__.split("\n")[0])
    p.add_argument("--from", dest="date_from", type=_date, help="first IST day to report, YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", type=_date, help="last IST day to report, YYYY-MM-DD")
    p.add_argument("--data-dir", type=Path, default=None, help="evidence store (default $EVIDENCE_DIR or ./var/evidence)")
    p.add_argument("--hq-dir", type=Path, default=None, help="HQ mirror (default $EVIDENCE_HQ_DIR or <data-dir>-hq)")
    p.add_argument("--json", action="store_true", help="print only the frozen verify JSON")
    p.add_argument("--all-blocks", action="store_true", help="list every block, not only failures")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir or default_data_dir()
    hq_dir = args.hq_dir or (Path(os.environ["EVIDENCE_HQ_DIR"]) if os.environ.get("EVIDENCE_HQ_DIR") else None)
    if args.date_from and args.date_to and args.date_from > args.date_to:
        print("--from is after --to", file=sys.stderr)
        return 2
    try:
        report = verify_store(data_dir, args.date_from, args.date_to, hq_dir)
    except (StoreUnreadable, OSError, ValueError, KeyError) as exc:
        print(f"cannot verify: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.frozen()))
    else:
        print(render(report, all_blocks=True if args.all_blocks else None))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
